"""Follow-up to extra3.py: T4 (no exchange, K/(1+beta), D/(1+beta)) recovered ~93-99% of ln(1+beta),
so the dispersion term does NOT explain the ~70-80% shortfall at equilibrium (T1).
Remaining difference between T1 and T4: the injected tracer mass is fixed. At equilibrium a fraction
beta/(1+beta) of it sits in the immobile zone, so the mobile (measured) concentration is lower by 1/(1+beta),
while discharge Q is unchanged. In T4 Q is lower by 1/(1+beta) and the concentration is not reduced.
Because the integral of the outlet concentration over time equals M/Q, the BTC area carries discharge information.

A  clean BTC check: compare (1+beta)*BTC_T1 with BTC_T4 (should agree if this is the only difference).
T5 equilibrium exchange (Da=1000) with BTC multiplied by (1+beta) before noise  -> expect ~100%
T6 no exchange, K/(1+beta), D/(1+beta) with BTC divided by (1+beta) before noise -> expect ~ T1 (70-80%)
T7 memory-blind model retrained on area-normalized BTCs (each BTC divided by its own time integral),
   tested on T1 inputs                                                            -> expect closer to 100%
"""
from __future__ import annotations
import os, sys, warnings
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from fhinv import learn as L, sim  # noqa: E402

DATA = os.path.join(ROOT, "data", "main")                                   # main datasets (run_all.py)
OUT = os.path.join(ROOT, "results", "review_checks", "baseline"); os.makedirs(OUT, exist_ok=True)
sim.configure(32, 16)
sets = {k: dict(np.load(os.path.join(DATA, f"{k}.npz"))) for k in ("train_mixed", "train_blind", "test_controlled")}
basis = L.PCABasis(sets["train_mixed"]["lnk"])
tc = sets["test_controlled"]; i0 = np.where(tc["cond"] == 0)[0]
lnk_f = tc["lnk"][i0].astype(float); nf = len(i0)
rng = np.random.default_rng(5151)
tests, check = {}, []
for b in (0.25, 0.5, 1.0):
    c1 = sim.simulate(lnk_f, np.full(nf, b), np.full(nf, 1000.0 / sim.T_REF))
    c4 = sim.simulate(lnk_f - np.log1p(b), np.zeros(nf), np.full(nf, 1.0 / sim.T_REF), disp=np.full(nf, sim.DISP / (1 + b)))
    a1, a4 = (1 + b) * c1["btc"], c4["btc"]
    check.append(dict(beta=b, rel_diff_scaled=float(np.abs(a1 - a4).sum() / np.abs(a4).sum()),
                      rel_diff_unscaled=float(np.abs(c1["btc"] - a4).sum() / np.abs(a4).sum()),
                      area_ratio_T1_T4=float((np.trapezoid(c1["btc"], sim.BTC_TIMES, axis=1) / np.trapezoid(a4, sim.BTC_TIMES, axis=1)).mean()),
                      Q_ratio_T1_T4=float((c1["Q"] / c4["Q"]).mean())))
    for name, c in ((f"T1 equilibrium exchange, beta={b:g}", c1),
                    (f"T5 equilibrium exchange, BTC x (1+beta), beta={b:g}", {**c1, "btc": a1}),
                    (f"T4 no exchange, K/(1+beta), D/(1+beta), beta={b:g}", c4),
                    (f"T6 no exchange, K/(1+beta), D/(1+beta), BTC / (1+beta), beta={b:g}", {**c4, "btc": c4["btc"] / (1 + b)})):
        tests[name] = {f"obs_{k}": v for k, v in sim.add_noise(c, rng).items()}
    print("simulated beta", b, flush=True)
print(pd.DataFrame(check).round(4).to_string(index=False))
base = {k: tc[k][i0] for k in ("obs_heads", "obs_btc", "obs_snap", "obs_Q")}


def norm_btc(d):
    area = np.trapezoid(d["obs_btc"], sim.BTC_TIMES, axis=1)
    return {**d, "obs_btc": d["obs_btc"] / np.maximum(area, 1e-12)[:, None] * sim.T_REF}


rows = []
for variant in ("raw", "area_normalized"):
    tr = sets["train_blind"] if variant == "raw" else norm_btc(sets["train_blind"])
    Xtr, Ytr = L.features(tr, "BH"), L.targets(tr, basis, False)
    for seed in range(5):
        sx, sy = L.StandardScaler().fit(Xtr), L.StandardScaler().fit(Ytr)
        net = L.MLPRegressor(hidden_layer_sizes=(128, 128), activation="relu", alpha=1e-3, learning_rate_init=1e-3, max_iter=400,
                             early_stopping=True, validation_fraction=0.15, n_iter_no_change=25, random_state=seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore"); net.fit(sx.transform(Xtr), sy.transform(Ytr))
        prep = (lambda d: d) if variant == "raw" else norm_btc
        mu0 = sy.inverse_transform(net.predict(sx.transform(L.features(prep(base), "BH"))))[:, 0]
        rows += [dict(variant=variant, case="no memory (accuracy)", seed=seed, field=i, dmu=mu0[i] - lnk_f[i].mean()) for i in range(nf)]
        for name, t in tests.items():
            if variant == "area_normalized" and not (name.startswith("T1") or name.startswith("T4")):
                continue
            mu = sy.inverse_transform(net.predict(sx.transform(L.features(prep(t), "BH"))))[:, 0]
            rows += [dict(variant=variant, case=name, seed=seed, field=i, dmu=mu[i] - mu0[i]) for i in range(nf)]
        print(variant, "seed", seed, "done", flush=True)
df = pd.DataFrame(rows); df.to_csv(os.path.join(OUT, "extra4_per_case.csv"), index=False)
f = df.groupby(["variant", "case", "field"]).dmu.mean().reset_index()
bs = lambda x: np.percentile(np.random.default_rng(0).choice(np.asarray(x), (4000, len(x))).mean(1), [2.5, 97.5])
out = []
for (variant, case), g in f.groupby(["variant", "case"]):
    if case.startswith("no memory"):
        out.append(dict(variant=variant, case=case, dmu=g.dmu.mean(), rmse=float(np.sqrt((df[(df.variant == variant) & (df.case == case)].groupby("seed").dmu.apply(lambda x: np.mean(x ** 2))).mean())),
                        fraction=np.nan, frac_lo=np.nan, frac_hi=np.nan)); continue
    b = float(case.split("=")[-1]); lo, hi = bs(g.dmu / -np.log1p(b))
    out.append(dict(variant=variant, case=case, dmu=g.dmu.mean(), rmse=np.nan, fraction=g.dmu.mean() / -np.log1p(b), frac_lo=lo, frac_hi=hi))
out = pd.DataFrame(out).sort_values(["variant", "case"]); out.to_csv(os.path.join(OUT, "extra4_summary.csv"), index=False)
pd.DataFrame(check).to_csv(os.path.join(OUT, "extra4_btc_check.csv"), index=False)
pd.set_option("display.width", 220); print(out.round(3).to_string(index=False))

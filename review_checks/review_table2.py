"""Table A1 (Appendix A) of the revised Chapter 3: memory-blind shift of mean ln K in the review tests (one run, one noise sequence).

Rows (beta = 0.25, 0.5, 1.0; 60 test fields; five training seeds; memory-blind model, BTC + heads):
  Da = 30 main test                       (from results/review_checks/baseline/table_shift.csv; run run_check.py --name baseline first)
  exchange, Da = 100 and Da = 1000        (uniform beta, single rate)
  no exchange, K/R, dispersion D          (R = 1 + beta)
  no exchange, K/R, dispersion D/R        (the equilibrium-equivalent aquifer, Eq. 7)
  exchange Da = 1000, BTC multiplied by R (concentration level of the equivalent aquifer restored)
  no exchange K/R, D/R, BTC divided by R  (concentration level of the exchange case imposed)
  exchange Da = 1000, model retrained on area-normalized BTCs
The shift is the change of the estimated mean ln K relative to the same field without memory (same seed).
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
rng = np.random.default_rng(20261010)
BETAS = (0.25, 0.5, 1.0)
tests = {}
for b in BETAS:
    R = 1 + b
    c100 = sim.simulate(lnk_f, np.full(nf, b), np.full(nf, 100.0 / sim.T_REF))
    c1k = sim.simulate(lnk_f, np.full(nf, b), np.full(nf, 1000.0 / sim.T_REF))
    cKD = sim.simulate(lnk_f - np.log(R), np.zeros(nf), np.full(nf, 1.0 / sim.T_REF))
    cKDR = sim.simulate(lnk_f - np.log(R), np.zeros(nf), np.full(nf, 1.0 / sim.T_REF), disp=np.full(nf, sim.DISP / R))
    for key, c in (("exchange, Da = 100", c100), ("exchange, Da = 1000", c1k), ("no exchange, K/R, D", cKD),
                   ("no exchange, K/R, D/R", cKDR), ("exchange Da = 1000, BTC x R", {**c1k, "btc": c1k["btc"] * R}),
                   ("no exchange K/R, D/R, BTC / R", {**cKDR, "btc": cKDR["btc"] / R})):
        tests[(key, b)] = {f"obs_{k}": v for k, v in sim.add_noise(c, rng).items()}
    print("simulated beta", b, flush=True)
base = {k: tc[k][i0] for k in ("obs_heads", "obs_btc", "obs_snap", "obs_Q")}


def norm_btc(d):
    area = np.trapezoid(d["obs_btc"], sim.BTC_TIMES, axis=1)
    return {**d, "obs_btc": d["obs_btc"] / np.maximum(area, 1e-12)[:, None] * sim.T_REF}


rows = []
for variant in ("raw", "area-normalized"):
    prep = (lambda d: d) if variant == "raw" else norm_btc
    tr = prep(sets["train_blind"])
    Xtr, Ytr = L.features(tr, "BH"), L.targets(tr, basis, False)
    for seed in range(5):
        sx, sy = L.StandardScaler().fit(Xtr), L.StandardScaler().fit(Ytr)
        net = L.MLPRegressor(hidden_layer_sizes=(128, 128), activation="relu", alpha=1e-3, learning_rate_init=1e-3, max_iter=400,
                             early_stopping=True, validation_fraction=0.15, n_iter_no_change=25, random_state=seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore"); net.fit(sx.transform(Xtr), sy.transform(Ytr))
        mu0 = sy.inverse_transform(net.predict(sx.transform(L.features(prep(base), "BH"))))[:, 0]
        rows += [dict(row="no memory, error of mean ln K", variant=variant, beta=0.0, seed=seed, field=i, d=mu0[i] - lnk_f[i].mean()) for i in range(nf)]
        for (key, b), t in tests.items():
            if variant != "raw" and key != "exchange, Da = 1000":
                continue
            mu = sy.inverse_transform(net.predict(sx.transform(L.features(prep(t), "BH"))))[:, 0]
            name = key if variant == "raw" else "exchange Da = 1000, model trained on area-normalized BTCs"
            rows += [dict(row=name, variant=variant, beta=b, seed=seed, field=i, d=mu[i] - mu0[i]) for i in range(nf)]
        print(variant, "seed", seed, flush=True)
df = pd.DataFrame(rows); df.to_csv(os.path.join(OUT, "table2_per_case.csv"), index=False)
ts = pd.read_csv(os.path.join(OUT, "table_shift.csv")); ts = ts[(ts.method == "blind_btc_h") & (ts.Da == 30)]
out = [dict(row="equilibrium-equivalence value -ln(1+beta)", beta=b, shift=-np.log1p(b), lo=np.nan, hi=np.nan) for b in BETAS]
out += [dict(row="exchange, Da = 30 (main test)", beta=r.beta, shift=r.dmu, lo=r.lo, hi=r.hi) for r in ts.itertuples()]
bs = lambda x: np.percentile(np.random.default_rng(0).choice(np.asarray(x), (4000, len(x))).mean(1), [2.5, 97.5])
for (row, b), g in df[df.beta > 0].groupby(["row", "beta"]):
    f = g.groupby("field").d.mean(); lo, hi = bs(f)
    out.append(dict(row=row, beta=b, shift=f.mean(), lo=lo, hi=hi))
out = pd.DataFrame(out); out["fraction"] = out["shift"] / -np.log1p(out.beta)
out.to_csv(os.path.join(OUT, "table2_summary.csv"), index=False)
acc = df[df.beta == 0].groupby(["variant", "seed"]).d.apply(lambda x: np.sqrt(np.mean(x ** 2))).groupby("variant").agg(["mean", "std"])
acc.to_csv(os.path.join(OUT, "table2_nomemory_accuracy.csv"))
pd.set_option("display.width", 200)
print(out.pivot(index="row", columns="beta", values="shift").round(3).to_string())
print(out.pivot(index="row", columns="beta", values="fraction").round(3).to_string())
print(acc.round(3))

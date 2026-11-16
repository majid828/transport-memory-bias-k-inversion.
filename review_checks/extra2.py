"""Why is the memory-blind shift only ~70-80% of ln(1+beta), even at equilibrium?

T1  equilibrium exchange (Da = 1000), uniform beta            -> real input
T2  NO exchange, but K divided by (1 + beta)                    -> in-class 'equivalent' input
    (advection identical to T1; dispersion D instead of D/R)
If the network recovers ~ln(1+beta) on T2 but not on T1, the shortfall comes from the shape of
the breakthrough curve (dispersion is not rescaled by R). If it also falls short on T2, it comes
from the network (shrinkage toward the training range of mean ln K).
T3  spatially variable beta with the model that also sees the plume snapshot and Q (blind_all):
    can spatially resolved transport data turn variable retention into spurious structure?
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from fhinv import learn as L, sim  # noqa: E402

DATA = os.path.join(ROOT, "data", "main")                                   # main datasets (run_all.py)
OUT = os.path.join(ROOT, "results", "review_checks", "baseline"); os.makedirs(OUT, exist_ok=True)
sim.configure(32, 16)
sets = {k: dict(np.load(os.path.join(DATA, f"{k}.npz"))) for k in ("train_mixed", "train_blind", "test_controlled")}
basis = L.PCABasis(sets["train_mixed"]["lnk"])
tc = sets["test_controlled"]
i0 = np.where(tc["cond"] == 0)[0]
lnk_f = tc["lnk"][i0].astype(float)
nf = len(i0)
rng = np.random.default_rng(4242)
pat = lnk_f - lnk_f.mean(axis=(1, 2), keepdims=True)
zK = pat / pat.std(axis=(1, 2), keepdims=True)


def field_beta(mean_beta, z, gamma=0.7):
    w = np.exp(gamma * z)
    return mean_beta * w / w.mean(axis=(1, 2), keepdims=True)


cases = []
for b in (0.25, 0.5, 1.0):
    cases.append((f"T1 equilibrium exchange beta={b:g}", lnk_f, np.full(nf, b), 1000.0))
    cases.append((f"T2 no exchange, K/(1+beta), beta={b:g}", lnk_f - np.log1p(b), np.zeros(nf), 1.0))
for b in (0.5, 1.0):
    cases.append((f"T3 beta(x) correlated with low K, mean {b:g}", lnk_f, field_beta(b, -zK), 30.0))
    cases.append((f"T3 uniform beta={b:g}", lnk_f, np.full(nf, b), 30.0))
tests = {}
for name, lk, beta, da in cases:
    clean = sim.simulate(lk, beta, np.full(nf, da / sim.T_REF))
    tests[name] = {f"obs_{k}": v for k, v in sim.add_noise(clean, rng).items()}
    print("simulated:", name, flush=True)
base = {k: tc[k][i0] for k in ("obs_heads", "obs_btc", "obs_snap", "obs_Q")}

rows = []
for method, inputs in (("blind_btc_h", "BH"), ("blind_all", "BHSQ")):
    for seed in range(5):
        tr = sets["train_blind"]
        Xtr, Ytr = L.features(tr, inputs), L.targets(tr, basis, False)
        sx, sy = L.StandardScaler().fit(Xtr), L.StandardScaler().fit(Ytr)
        net = L.MLPRegressor(hidden_layer_sizes=(128, 128), activation="relu", alpha=1e-3, learning_rate_init=1e-3,
                             max_iter=400, early_stopping=True, validation_fraction=0.15, n_iter_no_change=25,
                             random_state=seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            net.fit(sx.transform(Xtr), sy.transform(Ytr))
        pr = {}
        for name, t in [("cond0", base)] + list(tests.items()):
            Y = sy.inverse_transform(net.predict(sx.transform(L.features(t, inputs))))
            f = basis.field(Y[:, 0], Y[:, 1:1 + basis.V.shape[0]])
            pr[name] = (Y[:, 0], f - f.mean(axis=(1, 2), keepdims=True))
        mu0, p0 = pr["cond0"]
        for name in tests:
            mu, p = pr[name]
            for i in range(nf):
                rows.append(dict(method=method, case=name, seed=seed, field=i, mu_true=lnk_f[i].mean(), dmu=mu[i] - mu0[i],
                                 pattern_change=float(np.sqrt(((p[i] - p0[i]) ** 2).mean())),
                                 var_ratio_change=float((p[i].var() - p0[i].var()) / pat[i].var()),
                                 corr_with_true=float(np.corrcoef(p[i].ravel(), pat[i].ravel())[0, 1]),
                                 corr0=float(np.corrcoef(p0[i].ravel(), pat[i].ravel())[0, 1])))
        print(method, "seed", seed, "done", flush=True)
df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "extra2_per_case.csv"), index=False)
g = df.groupby(["method", "case", "field"]).mean(numeric_only=True).groupby(["method", "case"])
out = g.agg(dmu=("dmu", "mean"), pattern_change=("pattern_change", "mean"), var_ratio_change=("var_ratio_change", "mean"),
            corr_with_true=("corr_with_true", "mean"), corr0=("corr0", "mean")).reset_index()
out.to_csv(os.path.join(OUT, "extra2_summary.csv"), index=False)
pd.set_option("display.width", 250)
print(out.round(3).to_string(index=False))
# does the shortfall depend on how far the shifted mean lies outside the training range?
t2 = df[df.case.str.startswith("T2") & (df.method == "blind_btc_h")].groupby(["case", "field"]).mean(numeric_only=True).reset_index()
for case, gg in t2.groupby("case"):
    b = float(case.split("=")[-1])
    lo = gg[gg.mu_true > np.median(gg.mu_true)]; hi = gg[gg.mu_true <= np.median(gg.mu_true)]
    print(f"{case}: recovered fraction of ln(1+beta) = {gg.dmu.mean() / -np.log1p(b):.2f} "
          f"(fields with higher true mean {lo.dmu.mean() / -np.log1p(b):.2f}, lower true mean {hi.dmu.mean() / -np.log1p(b):.2f})")

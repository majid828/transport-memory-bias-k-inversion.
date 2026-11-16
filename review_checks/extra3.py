"""Direct test of the dispersion explanation.

T1  equilibrium exchange (Da = 1000), uniform beta                       [as in extra2.py]
T2  no exchange, K/(1+beta), dispersion D                                [as in extra2.py]
T4  no exchange, K/(1+beta), dispersion D/(1+beta)  (= the equilibrium retarded equation, no exchange term)
If T4 reproduces T1 and differs from T2, the shortfall of the memory-blind shift relative to
ln(1+beta) is caused by the dispersion term. Same trained models (same seeds) and the same noise
generator sequence for T1 and T2 as extra2.py.
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
rng = np.random.default_rng(9090)
tests = {}
for b in (0.25, 0.5, 1.0):
    for name, lk, beta, da, disp in ((f"T1 equilibrium exchange (Da=1000), beta={b:g}", lnk_f, np.full(nf, b), 1000.0, None),
                                     (f"T2 no exchange, K/(1+beta), D, beta={b:g}", lnk_f - np.log1p(b), np.zeros(nf), 1.0, None),
                                     (f"T4 no exchange, K/(1+beta), D/(1+beta), beta={b:g}", lnk_f - np.log1p(b), np.zeros(nf), 1.0, np.full(nf, sim.DISP / (1 + b)))):
        clean = sim.simulate(lk, beta, np.full(nf, da / sim.T_REF), disp=disp)
        tests[name] = {f"obs_{k}": v for k, v in sim.add_noise(clean, rng).items()}
        print("simulated:", name, flush=True)
base = {k: tc[k][i0] for k in ("obs_heads", "obs_btc", "obs_snap", "obs_Q")}
rows = []
for seed in range(5):
    tr = sets["train_blind"]
    Xtr, Ytr = L.features(tr, "BH"), L.targets(tr, basis, False)
    sx, sy = L.StandardScaler().fit(Xtr), L.StandardScaler().fit(Ytr)
    net = L.MLPRegressor(hidden_layer_sizes=(128, 128), activation="relu", alpha=1e-3, learning_rate_init=1e-3, max_iter=400,
                         early_stopping=True, validation_fraction=0.15, n_iter_no_change=25, random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore"); net.fit(sx.transform(Xtr), sy.transform(Ytr))
    mu0 = sy.inverse_transform(net.predict(sx.transform(L.features(base, "BH"))))[:, 0]
    for name, t in tests.items():
        mu = sy.inverse_transform(net.predict(sx.transform(L.features(t, "BH"))))[:, 0]
        rows += [dict(case=name, seed=seed, field=i, dmu=mu[i] - mu0[i]) for i in range(nf)]
    print("seed", seed, "done", flush=True)
df = pd.DataFrame(rows); df.to_csv(os.path.join(OUT, "extra3_per_case.csv"), index=False)
f = df.groupby(["case", "field"]).dmu.mean().reset_index()
bs = lambda x: np.percentile(np.random.default_rng(0).choice(np.asarray(x), (4000, len(x))).mean(1), [2.5, 97.5])
out = []
for case, g in f.groupby("case"):
    b = float(case.split("=")[-1]); lo, hi = bs(g.dmu / -np.log1p(b))
    out.append(dict(case=case, dmu=g.dmu.mean(), fraction=g.dmu.mean() / -np.log1p(b), frac_lo=lo, frac_hi=hi))
out = pd.DataFrame(out).sort_values("case"); out.to_csv(os.path.join(OUT, "extra3_summary.csv"), index=False)
pd.set_option("display.width", 200); print(out.round(3).to_string(index=False))

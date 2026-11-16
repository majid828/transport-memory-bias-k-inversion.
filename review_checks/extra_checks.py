"""Two short mechanism tests (equilibrium limit; spatially variable beta), run on the main data.

E1  Equilibrium limit: uniform beta in {0.25, 0.5, 1.0} at Da = 100 and 1000 (Da = 30 and 1
    are in the main test). If exchange acts as a constant retardation R = 1 + beta, the
    memory-blind shift of mean ln K should approach -ln(1 + beta) as Da grows.
E2  Spatially variable beta (Da = 30, mean beta 0.5 or 1.0): R(x) = 1 + beta(x) is no longer
    a constant that can be taken out of the spatial derivative. beta(x) is either correlated
    with low K (more immobile capacity in low-K zones) or an independent random field.
E0  No memory, second noise draw: noise floor of the pattern-change metric.
The memory-blind model (BTC + heads) is trained with five seeds on the baseline training set.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from fhinv import data as D, learn as L, sim  # noqa: E402

DATA = os.path.join(ROOT, "data", "main")                                   # main datasets (run_all.py)
OUT = os.path.join(ROOT, "results", "review_checks", "baseline"); os.makedirs(OUT, exist_ok=True)
sim.configure(32, 16)
sets = {k: dict(np.load(os.path.join(DATA, f"{k}.npz"))) for k in ("train_mixed", "train_blind", "test_controlled")}
basis = L.PCABasis(sets["train_mixed"]["lnk"])
tc = sets["test_controlled"]
i0 = np.where(tc["cond"] == 0)[0]
lnk_f = tc["lnk"][i0].astype(float)
nf = len(i0)
rng = np.random.default_rng(777)

pat = lnk_f - lnk_f.mean(axis=(1, 2), keepdims=True)
zK = pat / pat.std(axis=(1, 2), keepdims=True)
zR = np.stack([sim.gaussian_field(rng, 5.0, sim.NX, sim.NY) for _ in range(nf)])


def field_beta(mean_beta, z, gamma=0.7):
    w = np.exp(gamma * z)
    return mean_beta * w / w.mean(axis=(1, 2), keepdims=True)


cases = [("E0 no memory (new noise)", np.zeros(nf), 1.0)]
for da in (100.0, 1000.0):
    for b in (0.25, 0.5, 1.0):
        cases.append((f"E1 uniform beta={b:g}, Da={da:g}", np.full(nf, b), da))
for b in (0.5, 1.0):
    cases.append((f"E2 beta(x) correlated with low K, mean {b:g}, Da=30", field_beta(b, -zK), 30.0))
    cases.append((f"E2 beta(x) independent of K, mean {b:g}, Da=30", field_beta(b, zR), 30.0))
    cases.append((f"E2 reference uniform beta={b:g}, Da=30", np.full(nf, b), 30.0))

tests = {}
for name, beta, da in cases:
    alpha = np.full(nf, da / sim.T_REF)
    clean = sim.simulate(lnk_f, beta, alpha)
    noisy = sim.add_noise(clean, rng)
    tests[name] = dict(lnk=lnk_f, mu=lnk_f.mean(axis=(1, 2)), **{f"obs_{k}": v for k, v in noisy.items()})
    print("simulated:", name, flush=True)

base_test = {k: (v[i0] if hasattr(v, "__len__") and len(v) == len(tc["cond"]) else v) for k, v in tc.items()}
rows = []
for seed in range(5):
    preds = {}
    tr = sets["train_blind"]
    Xtr, Ytr = L.features(tr, "BH"), L.targets(tr, basis, False)
    sx, sy = L.StandardScaler().fit(Xtr), L.StandardScaler().fit(Ytr)
    net = L.MLPRegressor(hidden_layer_sizes=(128, 128), activation="relu", alpha=1e-3, learning_rate_init=1e-3,
                         max_iter=400, early_stopping=True, validation_fraction=0.15, n_iter_no_change=25,
                         random_state=seed)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        net.fit(sx.transform(Xtr), sy.transform(Ytr))
    for name, t in [("cond0", base_test)] + list(tests.items()):
        Y = sy.inverse_transform(net.predict(sx.transform(L.features(t, "BH"))))
        f = basis.field(Y[:, 0], Y[:, 1:1 + basis.V.shape[0]])
        preds[name] = (Y[:, 0], f - f.mean(axis=(1, 2), keepdims=True))
    mu0, pat0 = preds["cond0"]
    for name in tests:
        mu, pt = preds[name]
        for i in range(nf):
            rows.append(dict(case=name, seed=seed, field=i, dmu=mu[i] - mu0[i],
                             pattern_change=float(np.sqrt(((pt[i] - pat0[i]) ** 2).mean())),
                             var_ratio_change=float(pt[i].var() / pat[i].var() - pat0[i].var() / pat[i].var()),
                             pattern_rmse=float(np.sqrt(((pt[i] - pat[i]) ** 2).mean()))))
    print("seed", seed, "done", flush=True)
df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "extra_per_case.csv"), index=False)
g = df.groupby(["case", "field"]).mean(numeric_only=True).groupby("case")
bs = lambda x: np.percentile(np.random.default_rng(0).choice(np.asarray(x), (2000, len(x))).mean(1), [2.5, 97.5])
summ = []
for case, gg in g:
    lo, hi = bs(gg.dmu); plo, phi = bs(gg.pattern_change)
    summ.append(dict(case=case, dmu=gg.dmu.mean(), dmu_lo=lo, dmu_hi=hi, pattern_change=gg.pattern_change.mean(),
                     pc_lo=plo, pc_hi=phi, var_ratio_change=gg.var_ratio_change.mean(), pattern_rmse=gg.pattern_rmse.mean()))
summ = pd.DataFrame(summ).sort_values("case")
summ.to_csv(os.path.join(OUT, "extra_summary.csv"), index=False)
pd.set_option("display.width", 250)
print(summ.round(3).to_string(index=False))

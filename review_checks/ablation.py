"""Observation ablation (Experiment D): which observations fix the magnitude and the pattern of K?
Same networks, seeds and metrics as run_check.py, on the main data; adds input sets that were not in the first run
(heads + BTC + Q for the memory-blind, memory-agnostic and joint models)."""
import os, sys, json
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from fhinv import data as D, learn as L, sim
L.METHODS.update({
    "blind_btc": dict(inputs="B", train="blind", joint=False),
    "joint_btc_h_s": dict(inputs="BHS", train="mixed", joint=True),
    "joint_btc_h_q": dict(inputs="BHQ", train="mixed", joint=True),
    "agnostic_btc_h_q": dict(inputs="BHQ", train="mixed", joint=False),
    "blind_btc_h_q": dict(inputs="BHQ", train="blind", joint=False),
})
DATA = os.path.join(ROOT, "data", "main")
out = os.path.join(ROOT, "results", "review_checks", "ablation"); os.makedirs(out, exist_ok=True)
sets = {k: dict(np.load(os.path.join(DATA, f"{k}.npz"))) for k in ("train_mixed", "train_blind", "test_controlled")}
basis = L.PCABasis(sets["train_mixed"]["lnk"]); tc = sets["test_controlled"]
methods = ["heads_only", "blind_btc", "blind_btc_h", "blind_btc_h_q", "blind_all", "agnostic_btc_h", "agnostic_btc_h_q",
           "agnostic_all", "joint_btc_h", "joint_btc_h_s", "joint_btc_h_q", "joint_all"]
rows = []
pat_true = tc["lnk"] - tc["lnk"].mean(axis=(1, 2), keepdims=True)
for m in methods:
    for seed in range(5):
        p = L.fit_predict(m, sets, basis, seed)["test_controlled"]
        f = basis.field(p["mu"], p["coefs"]); pat = f - f.mean(axis=(1, 2), keepdims=True)
        prmse = np.sqrt(((pat - pat_true) ** 2).mean(axis=(1, 2)))
        for c in range(7):
            i = tc["cond"] == c
            e = p["mu"][i] - tc["mu"][i]
            r = dict(method=m, seed=seed, cond=c, mean_rmse=float(np.sqrt(np.mean(e ** 2))), mean_bias=float(e.mean()),
                     pattern_rmse=float(prmse[i].mean()))
            if "beta" in p:
                r["beta_rmse"] = float(np.sqrt(np.mean((p["beta"][i] - tc["beta"][i]) ** 2)))
            rows.append(r)
    print("done", m, flush=True)
df = pd.DataFrame(rows)
tab = df.groupby(["method", "cond"]).agg(["mean", "std"]).drop(columns="seed")
tab.columns = [f"{a}_{b}" for a, b in tab.columns]
tab = tab.reset_index(); tab.to_csv(os.path.join(out, "ablation_table.csv"), index=False)
show = tab[tab.cond.isin([0, 3, 6])]
print(show[["method", "cond", "mean_rmse_mean", "mean_rmse_std", "beta_rmse_mean", "pattern_rmse_mean"]].round(3).to_string(index=False))

"""Does the BTC area (= M/Q for a known injected mass) let the memory-aware models fix the magnitude of K without Q?
Retrain the memory-agnostic and joint models on area-normalized BTCs (each BTC divided by its own time integral)
and compare with the raw-BTC models of the main test (same seeds, same data). joint_all also sees Q.
"""
from __future__ import annotations
import os, sys
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from fhinv import learn as L, sim  # noqa: E402

DATA = os.path.join(ROOT, "data", "main")                                   # main datasets (run_all.py)
OUT = os.path.join(ROOT, "results", "review_checks", "baseline"); os.makedirs(OUT, exist_ok=True)
sim.configure(32, 16)
sets = {k: dict(np.load(os.path.join(DATA, f"{k}.npz"))) for k in ("train_mixed", "train_blind", "test_controlled")}
basis = L.PCABasis(sets["train_mixed"]["lnk"])


def norm_btc(d):
    area = np.trapezoid(d["obs_btc"], sim.BTC_TIMES, axis=1)
    return {**d, "obs_btc": d["obs_btc"] / np.maximum(area, 1e-12)[:, None] * sim.T_REF}


rows = []
tc = sets["test_controlled"]
for variant in ("raw", "area_normalized"):
    s = sets if variant == "raw" else {k: norm_btc(v) for k, v in sets.items()}
    for name in ("blind_btc_h", "agnostic_btc_h", "joint_btc_h", "joint_all"):
        for seed in range(5):
            p = L.fit_predict(name, s, basis, seed)["test_controlled"]
            for i in range(len(tc["cond"])):
                rows.append(dict(variant=variant, method=name, seed=seed, cond=int(tc["cond"][i]), field=int(tc["field"][i]),
                                 mean_err=float(p["mu"][i] - tc["mu"][i]),
                                 beta_err=float(p["beta"][i] - tc["beta"][i]) if "beta" in p else np.nan))
        print(variant, name, "done", flush=True)
df = pd.DataFrame(rows); df.to_csv(os.path.join(OUT, "extra5_per_case.csv"), index=False)
g = df.groupby(["variant", "method", "cond", "seed"]).agg(
    mean_rmse=("mean_err", lambda x: float(np.sqrt(np.mean(np.square(x))))), mean_bias=("mean_err", "mean"),
    beta_rmse=("beta_err", lambda x: float(np.sqrt(np.nanmean(np.square(x)))) if x.notna().any() else np.nan)).reset_index()
t = g.groupby(["variant", "method", "cond"]).agg(["mean", "std"]).drop(columns="seed")
t.columns = [f"{a}_{b}" for a, b in t.columns]; t = t.reset_index()
t.to_csv(os.path.join(OUT, "extra5_summary.csv"), index=False)
pd.set_option("display.width", 220)
print(t[t.cond.isin([0, 3, 6])].round(3).to_string(index=False))

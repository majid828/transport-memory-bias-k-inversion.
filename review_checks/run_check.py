"""Review checks for Chapter 3: the six main models re-run on one design.

  python review_checks/run_check.py --name baseline                    # main design (reproduction)
  python review_checks/run_check.py --name porosity --theta 0.2 0.4    # mobile porosity unknown, 0.2-0.4
  python review_checks/run_check.py --name domain64 --nx 64 --ny 32    # domain twice as long and wide

The main design (32 x 16, porosity 0.30) uses the datasets in data/main (made by run_all.py, or generated
here if missing); other designs generate their own datasets in data/<name>.
Results: results/review_checks/<name>/ (table_methods.csv, table_shift.csv, per_case.csv, info.json).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from fhinv import data as D, learn as L, sim  # noqa: E402

METHODS = ["blind_btc_h", "blind_all", "agnostic_btc_h", "joint_btc_h", "agnostic_all", "joint_all"]


def ci95(x):
    x = np.asarray(x, float)
    bs = np.random.default_rng(0).choice(x, (2000, len(x))).mean(1)
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--nx", type=int, default=32)
    ap.add_argument("--ny", type=int, default=16)
    ap.add_argument("--theta", type=float, nargs=2, default=None)
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()
    main_design = (a.nx, a.ny) == (32, 16) and a.theta is None
    data = os.path.join(ROOT, "data", "main" if main_design else a.name)
    out = os.path.join(ROOT, "results", "review_checks", a.name); os.makedirs(out, exist_ok=True)
    sim.configure(a.nx, a.ny)
    t0 = time.time()
    if not os.path.exists(os.path.join(data, "test_controlled.npz")):
        D.generate(data, theta_range=tuple(a.theta) if a.theta else None, n_random=400 if main_design else 0)
    sets = {k: dict(np.load(os.path.join(data, f"{k}.npz"))) for k in ("train_mixed", "train_blind", "test_controlled")}
    basis = L.PCABasis(sets["train_mixed"]["lnk"])
    tc = sets["test_controlled"]
    rows = []
    for name in METHODS:
        for seed in range(a.seeds):
            p = L.fit_predict(name, sets, basis, seed)["test_controlled"]
            lnk_hat = basis.field(p["mu"], p["coefs"])
            pat_hat = lnk_hat - lnk_hat.mean(axis=(1, 2), keepdims=True)
            pat_true = tc["lnk"] - tc["lnk"].mean(axis=(1, 2), keepdims=True)
            prmse = np.sqrt(((pat_hat - pat_true) ** 2).mean(axis=(1, 2)))
            for i in range(len(tc["cond"])):
                rows.append(dict(method=name, seed=seed, cond=int(tc["cond"][i]), field=int(tc["field"][i]),
                                 mu_true=float(tc["mu"][i]), mu_hat=float(p["mu"][i]), pattern_rmse=float(prmse[i]),
                                 beta_true=float(tc["beta"][i]),
                                 beta_hat=float(p["beta"][i]) if "beta" in p else np.nan))
        print(f"[learn] {name} done ({time.time() - t0:.0f} s)", flush=True)
    df = pd.DataFrame(rows)
    df["mean_err"] = df.mu_hat - df.mu_true
    df["beta_err"] = df.beta_hat - df.beta_true
    df.to_csv(os.path.join(out, "per_case.csv"), index=False)
    g = df.groupby(["method", "cond", "seed"]).agg(
        mean_rmse=("mean_err", lambda x: float(np.sqrt(np.mean(np.square(x))))), mean_bias=("mean_err", "mean"),
        pattern_rmse=("pattern_rmse", "mean"),
        beta_rmse=("beta_err", lambda x: float(np.sqrt(np.nanmean(np.square(x)))) if x.notna().any() else np.nan)).reset_index()
    tab = g.groupby(["method", "cond"]).agg(["mean", "std"]).drop(columns="seed")
    tab.columns = [f"{c}_{s}" for c, s in tab.columns]
    tab = tab.reset_index()
    tab.to_csv(os.path.join(out, "table_methods.csv"), index=False)
    fh = []
    for name in METHODS:
        sub = df[df.method == name]
        base = sub[sub.cond == 0].set_index(["seed", "field"])
        for ci, (b, da) in enumerate(D.CONDITIONS):
            if ci == 0:
                continue
            s = sub[sub.cond == ci].set_index(["seed", "field"])
            dmu = (s["mu_hat"] - base["mu_hat"]).groupby("field").mean()
            lo, hi = ci95(dmu)
            fh.append(dict(method=name, beta=b, Da=da, dmu=float(dmu.mean()), lo=lo, hi=hi, eq_limit=-np.log1p(b)))
    fhd = pd.DataFrame(fh)
    fhd.to_csv(os.path.join(out, "table_shift.csv"), index=False)
    ell = (3.0, 7.0); I = tuple(np.sqrt(np.pi) / 2 * e for e in ell)
    info = dict(name=a.name, grid=[a.nx, a.ny], theta=a.theta or [sim.THETA_REF] * 2, seeds=a.seeds,
                pca_explained=basis.explained, ell_cells=ell, integral_scale_cells=I,
                length_in_ell=[a.nx / ell[1], a.nx / ell[0]], width_in_ell=[a.ny / ell[1], a.ny / ell[0]],
                length_in_I=[a.nx / I[1], a.nx / I[0]], width_in_I=[a.ny / I[1], a.ny / I[0]],
                runtime_s=time.time() - t0)
    json.dump(info, open(os.path.join(out, "info.json"), "w"), indent=1)
    show = tab[tab.cond.isin([0, 3, 6])][["method", "cond", "mean_rmse_mean", "mean_rmse_std", "mean_bias_mean",
                                           "beta_rmse_mean", "pattern_rmse_mean"]]
    print(show.round(3).to_string(index=False))
    print(fhd[fhd.method == "blind_btc_h"].round(3).to_string(index=False))
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()

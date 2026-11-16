"""Run the complete Chapter 3 synthetic experiment (Experiments A-C and the controls).

    python run_all.py              # full run (about 5-10 min on one CPU core)
    python run_all.py --quick      # small smoke test (about 1-2 min; data/quick, results/quick)

Datasets are generated once into --data (default data/main) and reused if present.
Outputs (in --out, default results/main): tables/*.csv, figures/*.png|pdf, run_info.json.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from fhinv import data as D, learn as L, sim  # noqa: E402


def ci95(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(0)
    bs = rng.choice(x, (2000, len(x))).mean(1)
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "main"))
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "main"))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()
    if a.quick:
        a.out = os.path.join(ROOT, "results", "quick"); a.data = os.path.join(ROOT, "data", "quick")
        sizes = dict(n_train=300, n_fields=10, n_random=60)
        a.seeds, max_iter = 2, 150
    else:
        sizes = dict(n_train=2400, n_fields=60, n_random=400)
        max_iter = 400
    t0 = time.time()
    for sub in ("tables", "figures"):
        os.makedirs(os.path.join(a.out, sub), exist_ok=True)
    if not os.path.exists(os.path.join(a.data, "test_random.npz")):
        D.generate(a.data, seed=1, **sizes)
    sets = D.load(a.data)
    basis = L.PCABasis(sets["train_mixed"]["lnk"])
    tc, trd = sets["test_controlled"], sets["test_random"]
    floor = L.pca_floor(tc, basis)
    print(f"[learn] PCA: {L.N_MODES} modes explain {basis.explained:.3f} of pattern variance; "
          f"representation floor (total ln K RMSE) = {floor.mean():.3f}", flush=True)

    rows, rows_r, preds_keep = [], [], {}
    for name in L.METHODS:
        for seed in range(a.seeds):
            t1 = time.time()
            p = L.fit_predict(name, sets, basis, seed, max_iter=max_iter)
            m, lnk_hat = L.field_metrics(p["test_controlled"], tc, basis)
            for i in range(len(tc["cond"])):
                r = dict(method=name, seed=seed, cond=int(tc["cond"][i]), field=int(tc["field"][i]),
                         beta=float(tc["beta"][i]), logda=float(tc["logda"][i]), mu_true=float(tc["mu"][i]),
                         mu_hat=float(p["test_controlled"]["mu"][i]))
                r.update({k: float(v[i]) for k, v in m.items()})
                rows.append(r)
            mr, _ = L.field_metrics(p["test_random"], trd, basis)
            for i in range(len(trd["mu"])):
                r = dict(method=name, seed=seed, beta=float(trd["beta"][i]), logda=float(trd["logda"][i]))
                r.update({k: float(v[i]) for k, v in mr.items()})
                rows_r.append(r)
            if seed == 0 and name in ("blind_btc_h", "joint_all", "agnostic_all"):
                preds_keep[name] = (p["test_controlled"], lnk_hat)
            print(f"[learn] {name:15s} seed {seed}: {p['n_iter']} epochs, {time.time() - t1:.1f} s", flush=True)
    df = pd.DataFrame(rows)
    dfr = pd.DataFrame(rows_r)
    df.to_csv(os.path.join(a.out, "tables", "per_case_controlled.csv"), index=False)
    dfr.to_csv(os.path.join(a.out, "tables", "per_case_random.csv"), index=False)

    # ---------------- method comparison on the controlled test (seed-level means, then mean +- sd across seeds)
    cond_label = {i: ("no memory" if b == 0 else f"beta={b:g}, Da={da:g}") for i, (b, da) in enumerate(D.CONDITIONS)}
    g = df.groupby(["method", "cond", "seed"]).agg(
        mean_rmse=("mean_err", lambda x: float(np.sqrt(np.mean(np.square(x))))),
        mean_bias=("mean_err", "mean"), pattern_rmse=("pattern_rmse", "mean"),
        total_rmse=("total_rmse", "mean"), var_ratio=("var_ratio", "mean"),
        beta_rmse=("beta_err", lambda x: float(np.sqrt(np.nanmean(np.square(x)))) if x.notna().any() else np.nan)
    ).reset_index()
    tab = g.groupby(["method", "cond"]).agg(["mean", "std"]).drop(columns="seed")
    tab.columns = [f"{c}_{s}" for c, s in tab.columns]
    tab = tab.reset_index()
    tab.insert(2, "condition", tab["cond"].map(cond_label))
    tab.to_csv(os.path.join(a.out, "tables", "table_methods_controlled.csv"), index=False)

    # ---------------- in-distribution test
    gr = dfr.groupby(["method", "seed"]).agg(
        mean_rmse=("mean_err", lambda x: float(np.sqrt(np.mean(np.square(x))))), pattern_rmse=("pattern_rmse", "mean"),
        total_rmse=("total_rmse", "mean"),
        beta_rmse=("beta_err", lambda x: float(np.sqrt(np.nanmean(np.square(x)))) if x.notna().any() else np.nan)
    ).reset_index().groupby("method").agg(["mean", "std"]).drop(columns="seed")
    gr.columns = [f"{c}_{s}" for c, s in gr.columns]
    gr.reset_index().to_csv(os.path.join(a.out, "tables", "table_methods_random.csv"), index=False)

    # ---------------- false heterogeneity: same field, memory vs no memory (paired, per method)
    fh = []
    for name in L.METHODS:
        sub = df[df.method == name]
        base = sub[sub.cond == 0].set_index(["seed", "field"])
        for ci, (b, da) in enumerate(D.CONDITIONS):
            if ci == 0:
                continue
            s = sub[sub.cond == ci].set_index(["seed", "field"])
            dmu = (s["mu_hat"] - base["mu_hat"]).groupby("field").mean()          # average over seeds
            dvar = (s["var_ratio"] - base["var_ratio"]).groupby("field").mean()
            dpat = (s["pattern_rmse"] - base["pattern_rmse"]).groupby("field").mean()
            fh.append(dict(method=name, beta=b, Da=da, predicted_equilibrium_shift=-np.log1p(b),
                           dmu_mean=float(dmu.mean()), dmu_ci_lo=ci95(dmu)[0], dmu_ci_hi=ci95(dmu)[1],
                           dpattern_rmse_mean=float(dpat.mean()), dpattern_ci_lo=ci95(dpat)[0], dpattern_ci_hi=ci95(dpat)[1],
                           dvar_ratio_mean=float(dvar.mean())))
    fhd = pd.DataFrame(fh)
    fhd.to_csv(os.path.join(a.out, "tables", "table_false_heterogeneity.csv"), index=False)

    # pattern change caused by memory alone: RMSE between the inferred patterns (memory vs no memory), seed 0
    pc = []
    for name, (pred, lnk_hat) in preds_keep.items():
        pat = lnk_hat - lnk_hat.mean(axis=(1, 2), keepdims=True)
        for ci in range(1, len(D.CONDITIONS)):
            i0 = np.where(tc["cond"] == 0)[0]; i1 = np.where(tc["cond"] == ci)[0]
            d = np.sqrt(((pat[i1] - pat[i0]) ** 2).mean(axis=(1, 2)))
            b, da = D.CONDITIONS[ci]
            pc.append(dict(method=name, beta=b, Da=da, pattern_change_rmse=float(d.mean()),
                           ci_lo=ci95(d)[0], ci_hi=ci95(d)[1]))
    pd.DataFrame(pc).to_csv(os.path.join(a.out, "tables", "table_pattern_change_seed0.csv"), index=False)

    # ---------------- forward consistency (seed-0 models)
    fw = []
    fw_curves = {}
    for name in ("blind_btc_h", "joint_all"):
        pred, lnk_hat = preds_keep[name]
        if name.startswith("blind"):
            bh, lh = np.zeros(len(lnk_hat)), np.zeros(len(lnk_hat))
        else:
            bh, lh = np.clip(pred["beta"], 0, None), np.clip(pred["logda"], *D.LOG_DA_RANGE)
        rel, tpk = L.forward_check(lnk_hat, bh, lh, tc)
        for ci, (b, da) in enumerate(D.CONDITIONS):
            idx = tc["cond"] == ci
            fw.append(dict(method=name, beta=b, Da=da if da else np.nan, btc_rel_l2_mean=float(rel[idx].mean()),
                           btc_rel_l2_ci_lo=ci95(rel[idx])[0], btc_rel_l2_ci_hi=ci95(rel[idx])[1],
                           peak_time_ratio_median=float(np.median(tpk[idx]))))
        fw_curves[name] = rel
    pd.DataFrame(fw).to_csv(os.path.join(a.out, "tables", "table_forward_check.csv"), index=False)

    info = dict(sizes=sizes, seeds=a.seeds, max_iter=max_iter, n_modes=L.N_MODES,
                pca_explained=basis.explained, pca_floor_total_rmse=float(floor.mean()),
                prior_mu_sd=float(np.std(sets["train_mixed"]["mu"])), grid=[sim.NX, sim.NY],
                T_ref=sim.T_REF, T_end=sim.T_END, runtime_s=time.time() - t0,
                python=platform.python_version(), versions=_versions())
    json.dump(info, open(os.path.join(a.out, "run_info.json"), "w"), indent=1)
    from fhinv import controls
    print("[controls] noise floor and shrinkage", flush=True)
    controls.run(a.out, a.data, seeds=a.seeds)
    from fhinv import plots
    plots.make_all(a.out, sets, basis, df, fhd, tab, preds_keep, pd.DataFrame(fw))
    print(f"[done] {time.time() - t0:.0f} s; results in {a.out}", flush=True)


def _versions():
    import matplotlib, scipy, sklearn
    return dict(numpy=np.__version__, scipy=scipy.__version__, scikit_learn=sklearn.__version__,
                pandas=pd.__version__, matplotlib=matplotlib.__version__)


if __name__ == "__main__":
    main()

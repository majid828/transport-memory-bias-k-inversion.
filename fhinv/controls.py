"""Two controls needed to interpret the main results (run by run_all.py after the main loop):
(1) noise floor of the pattern-change metric: the same no-memory data with a second,
    independent noise realization, passed through the same trained model;
(2) attenuation (shrinkage) of each estimator: slope of estimated vs true mean ln K on the
    no-memory controlled cases.  A slope below 1 means that a true shift of the mean ln K
    produces a smaller shift of the estimate, which scales the expected memory-induced shift."""
import os

import numpy as np
import pandas as pd

from . import learn as L, sim
from .data import load


def run(out, data_dir, methods=("blind_btc_h", "blind_all", "agnostic_btc_h", "joint_btc_h", "agnostic_all", "joint_all"),
        seeds=5):
    sets = load(data_dir)
    basis = L.PCABasis(sets["train_mixed"]["lnk"])
    tc = sets["test_controlled"]
    i0 = np.where(tc["cond"] == 0)[0]
    # second noise realization of the no-memory cases
    rng = np.random.default_rng(12345)
    clean = {k: tc["clean_" + k][i0] for k in ("heads", "btc", "snap", "Q")}
    noisy2 = sim.add_noise(clean, rng)
    tc2 = {k: (v[i0] if hasattr(v, "__len__") and len(v) == len(tc["cond"]) else v) for k, v in tc.items()}
    for k, v in noisy2.items():
        tc2["obs_" + k] = v
    rows = []
    for name in methods:
        for seed in range(seeds):
            sets2 = dict(sets); sets2["test_controlled"] = tc2
            p1 = L.fit_predict(name, sets, basis, seed)["test_controlled"]
            p2 = L.fit_predict(name, sets2, basis, seed)["test_controlled"]
            f1 = basis.field(p1["mu"][i0], p1["coefs"][i0]); f2 = basis.field(p2["mu"], p2["coefs"])
            pat1 = f1 - f1.mean(axis=(1, 2), keepdims=True); pat2 = f2 - f2.mean(axis=(1, 2), keepdims=True)
            noise_pc = np.sqrt(((pat1 - pat2) ** 2).mean(axis=(1, 2))).mean()
            mu_t = tc["mu"][i0]; slope = np.polyfit(mu_t, p1["mu"][i0], 1)[0]
            rows.append(dict(method=name, seed=seed, noise_floor_pattern_change=noise_pc, mu_slope=slope,
                             noise_floor_dmu_sd=float(np.std(p1["mu"][i0] - p2["mu"]))))
    df = pd.DataFrame(rows)
    g = df.groupby("method").agg(["mean", "std"]).drop(columns="seed")
    g.columns = [f"{a}_{b}" for a, b in g.columns]
    g.reset_index().to_csv(os.path.join(out, "tables", "table_controls.csv"), index=False)
    return g

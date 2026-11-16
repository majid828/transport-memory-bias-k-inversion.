"""Learned inverse models and evaluation.

All models are identical multilayer perceptrons (scikit-learn MLPRegressor, two hidden
layers of 128 ReLU units, early stopping, same hyper-parameters); they differ only in
(i) which observations they receive, (ii) which training set they see and (iii) which
outputs they predict.  ln K is represented by its spatial mean plus the leading principal
components of the training fields (a reduced-order stand-in for the CNN decoder planned
for the full study).

Methods
  heads_only        heads                    mixed training   K only
  blind_btc         outlet BTC               beta = 0 only    K only   (conventional)
  blind_btc_h       BTC + heads              beta = 0 only    K only   (conventional)
  blind_all         BTC + heads + snap + Q   beta = 0 only    K only   (conventional)
  agnostic_btc_h    BTC + heads              mixed            K only   (fair control)
  joint_btc_h       BTC + heads              mixed            K + (beta, log10 Da)
  joint_btc_h_s     BTC + heads + snapshot   mixed            K + memory
  agnostic_all      BTC + heads + snap + Q   mixed            K only   (fair control)
  joint_all         BTC + heads + snap + Q   mixed            K + memory
"""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from . import sim
from .data import LOG_DA_NONE

N_MODES = 40
METHODS = {
    "heads_only": dict(inputs="H", train="mixed", joint=False),
    "blind_btc": dict(inputs="B", train="blind", joint=False),
    "blind_btc_h": dict(inputs="BH", train="blind", joint=False),
    "blind_all": dict(inputs="BHSQ", train="blind", joint=False),
    "agnostic_btc_h": dict(inputs="BH", train="mixed", joint=False),
    "joint_btc_h": dict(inputs="BH", train="mixed", joint=True),
    "joint_btc_h_s": dict(inputs="BHS", train="mixed", joint=True),
    "agnostic_all": dict(inputs="BHSQ", train="mixed", joint=False),
    "joint_all": dict(inputs="BHSQ", train="mixed", joint=True),
}


class PCABasis:
    """Principal components of ln K deviations from the field mean (training fields)."""

    def __init__(self, lnk_train, n_modes=N_MODES):
        X = (lnk_train - lnk_train.mean(axis=(1, 2), keepdims=True)).reshape(len(lnk_train), -1)
        _, s, Vt = np.linalg.svd(X, full_matrices=False)
        self.V = Vt[:n_modes]
        self.explained = float((s[:n_modes] ** 2).sum() / (s ** 2).sum())

    def coefs(self, lnk):
        return (lnk - lnk.mean(axis=(1, 2), keepdims=True)).reshape(len(lnk), -1) @ self.V.T

    def field(self, mu, coefs):
        return mu[:, None, None] + (coefs @ self.V).reshape(-1, sim.NY, sim.NX)


def features(d, inputs):
    parts = []
    if "H" in inputs:
        parts.append(d["obs_heads"])
    if "B" in inputs:
        parts.append(np.log10(d["obs_btc"] + 1e-5))
    if "S" in inputs:
        parts.append(np.log10(d["obs_snap"] + 1e-5))
    if "Q" in inputs:
        parts.append(np.log(np.maximum(d["obs_Q"], 1e-12))[:, None])
    return np.concatenate(parts, axis=1)


def targets(d, basis, joint):
    y = [d["mu"][:, None], basis.coefs(d["lnk"])]
    if joint:
        logda = np.where(d["beta"] > 0, d["logda"], LOG_DA_NONE)
        y += [d["beta"][:, None], logda[:, None]]
    return np.concatenate(y, axis=1)


def fit_predict(name, sets, basis, seed, max_iter=400):
    """Train one network (method `name`, random seed `seed`) and predict every test set present in `sets`.
    Returns {test set name: dict(mu, coefs[, beta, logda])} and the number of training epochs ("n_iter")."""
    cfg = METHODS[name]
    tr = sets["train_" + cfg["train"]]
    Xtr, Ytr = features(tr, cfg["inputs"]), targets(tr, basis, cfg["joint"])
    sx, sy = StandardScaler().fit(Xtr), StandardScaler().fit(Ytr)
    net = MLPRegressor(hidden_layer_sizes=(128, 128), activation="relu", alpha=1e-3, learning_rate_init=1e-3,
                       max_iter=max_iter, early_stopping=True, validation_fraction=0.15, n_iter_no_change=25,
                       random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        net.fit(sx.transform(Xtr), sy.transform(Ytr))
    out = {}
    for tname in ("test_controlled", "test_random"):
        if tname not in sets:                     # the review checks load only the controlled test
            continue
        Y = sy.inverse_transform(net.predict(sx.transform(features(sets[tname], cfg["inputs"]))))
        pred = dict(mu=Y[:, 0], coefs=Y[:, 1:1 + basis.V.shape[0]])
        if cfg["joint"]:
            pred["beta"], pred["logda"] = Y[:, -2], Y[:, -1]
        out[tname] = pred
    out["n_iter"] = net.n_iter_
    return out


def field_metrics(pred, d, basis):
    """Mean-ln K error, pattern error (deviations from the field mean), total error,
    and variance ratio of the inferred pattern."""
    lnk_hat = basis.field(pred["mu"], pred["coefs"])
    true = d["lnk"].astype(float)
    pat_hat = lnk_hat - lnk_hat.mean(axis=(1, 2), keepdims=True)
    pat_true = true - true.mean(axis=(1, 2), keepdims=True)
    m = dict(mean_err=pred["mu"] - d["mu"],
             pattern_rmse=np.sqrt(((pat_hat - pat_true) ** 2).mean(axis=(1, 2))),
             total_rmse=np.sqrt(((lnk_hat - true) ** 2).mean(axis=(1, 2))),
             var_ratio=pat_hat.var(axis=(1, 2)) / pat_true.var(axis=(1, 2)),
             pattern_corr=np.array([np.corrcoef(a.ravel(), b.ravel())[0, 1] for a, b in zip(pat_hat, pat_true)]))
    if "beta" in pred:
        m["beta_err"] = pred["beta"] - d["beta"]
    return m, lnk_hat


def pca_floor(d, basis):
    """Best achievable pattern error with the reduced-order representation."""
    c = basis.coefs(d["lnk"])
    rec = basis.field(d["mu"], c)
    return np.sqrt(((rec - d["lnk"]) ** 2).mean(axis=(1, 2)))


def forward_check(lnk_hat, beta_hat, logda_hat, d):
    """Re-simulate the tracer test with the inferred parameters and compare with the
    noise-free observed BTC: relative L2 misfit and peak-time ratio."""
    alpha = 10.0 ** np.asarray(logda_hat) / sim.T_REF
    sim_ = sim.simulate(lnk_hat, np.maximum(np.asarray(beta_hat), 0.0), alpha)
    obs = d["clean_btc"]
    rel = np.linalg.norm(sim_["btc"] - obs, axis=1) / np.linalg.norm(obs, axis=1)
    tpk = sim.BTC_TIMES[np.argmax(sim_["btc"], axis=1)] / sim.BTC_TIMES[np.argmax(obs, axis=1)]
    return rel, tpk

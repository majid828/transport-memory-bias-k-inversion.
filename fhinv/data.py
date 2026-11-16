"""Dataset generation.

train_mixed.npz  : N_train random aquifers, random memory (25% no memory; otherwise
                   beta ~ U(0.1, 1.5), log10 Da ~ U(log10 0.3, log10 100)).
train_blind.npz  : the SAME aquifers simulated without memory (beta = 0): the training set
                   of a conventional, memory-blind inversion.
test_controlled  : N_fields new aquifers, each simulated under 7 conditions with the SAME
                   true K field: no memory, and beta in {0.25, 0.5, 1.0} x Da in {30, 1}.
test_random      : new aquifers drawn from the training distribution (in-distribution test);
                   generated last, so it does not change the other three sets (n_random=0 skips it).
Noise is added independently to every dataset (5% relative on concentrations and Q,
0.005 absolute on heads).  Truth (ln K, beta, Da, porosity) is stored alongside.

theta_range=None keeps mobile porosity at 0.30 for every aquifer (main design).
theta_range=(lo, hi) draws porosity per aquifer from a SEPARATE random generator, so the
K fields, memory parameters and noise draws are identical to the main design and only
porosity differs.  Porosity is never given to the inverse models.
"""
from __future__ import annotations

import os
import time

import numpy as np

from . import sim

LOG_DA_RANGE = (np.log10(0.3), np.log10(100.0))
LOG_DA_NONE = float(np.mean(LOG_DA_RANGE))      # placeholder target when beta = 0
CONDITIONS = [(0.0, None)] + [(b, da) for da in (30.0, 1.0) for b in (0.25, 0.5, 1.0)]


def sample_memory(rng, n, p_zero=0.25, beta_range=(0.1, 1.5)):
    beta = rng.uniform(*beta_range, n)
    beta[rng.uniform(size=n) < p_zero] = 0.0
    logda = rng.uniform(*LOG_DA_RANGE, n)
    logda[beta == 0] = LOG_DA_NONE
    return beta, logda


def _simulate(lnk, beta, logda, theta, rng, verbose):
    alpha = 10.0 ** logda / sim.T_REF
    clean = sim.simulate(lnk, beta, alpha, theta=theta, verbose=verbose)
    return clean, sim.add_noise(clean, rng)


def _save(path, lnk, beta, logda, theta, clean, noisy, **extra):
    np.savez_compressed(path, lnk=lnk.astype(np.float32), mu=lnk.mean(axis=(1, 2)), beta=beta, logda=logda, theta=theta,
                        **{f"obs_{k}": v for k, v in noisy.items()},
                        **{f"clean_{k}": v for k, v in clean.items()}, **extra)


def generate(out, n_train=2400, n_fields=60, n_random=400, seed=1, theta_range=None, verbose=True):
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(seed)
    trng = np.random.default_rng(seed + 1000)
    draw_theta = (lambda n: np.full(n, sim.THETA_REF)) if theta_range is None else (lambda n: trng.uniform(*theta_range, n))
    t0 = time.time()
    lnk, _ = sim.sample_lnk(rng, n_train)
    beta, logda = sample_memory(rng, n_train)
    theta = draw_theta(n_train)
    if verbose:
        print(f"[data] {n_train} training aquifers with memory", flush=True)
    c, o = _simulate(lnk, beta, logda, theta, rng, verbose)
    _save(os.path.join(out, "train_mixed.npz"), lnk, beta, logda, theta, c, o)
    if verbose:
        print(f"[data] the same {n_train} aquifers without memory", flush=True)
    z = np.zeros(n_train); ld = np.full(n_train, LOG_DA_NONE)
    c, o = _simulate(lnk, z, ld, theta, rng, verbose)
    _save(os.path.join(out, "train_blind.npz"), lnk, z, ld, theta, c, o)

    lnk_f, _ = sim.sample_lnk(rng, n_fields)
    theta_f = draw_theta(n_fields)
    L, B, D, T, cond, fid = [], [], [], [], [], []
    for ci, (b, da) in enumerate(CONDITIONS):
        L.append(lnk_f); B.append(np.full(n_fields, b)); T.append(theta_f)
        D.append(np.full(n_fields, LOG_DA_NONE if da is None else np.log10(da)))
        cond.append(np.full(n_fields, ci)); fid.append(np.arange(n_fields))
    L, B, D, T = map(np.concatenate, (L, B, D, T))
    if verbose:
        print(f"[data] controlled test: {n_fields} fields x {len(CONDITIONS)} memory conditions", flush=True)
    c, o = _simulate(L, B, D, T, rng, verbose)
    _save(os.path.join(out, "test_controlled.npz"), L, B, D, T, c, o, cond=np.concatenate(cond), field=np.concatenate(fid))

    if n_random:
        lnk_r, _ = sim.sample_lnk(rng, n_random)
        beta_r, logda_r = sample_memory(rng, n_random)
        theta_r = draw_theta(n_random)
        if verbose:
            print(f"[data] random test: {n_random} aquifers", flush=True)
        c, o = _simulate(lnk_r, beta_r, logda_r, theta_r, rng, verbose)
        _save(os.path.join(out, "test_random.npz"), lnk_r, beta_r, logda_r, theta_r, c, o)
    if verbose:
        print(f"[data] done in {time.time() - t0:.0f} s", flush=True)


def load(folder, names=("train_mixed", "train_blind", "test_controlled", "test_random")):
    """Load datasets as dicts of arrays (missing optional sets are skipped)."""
    return {k: dict(np.load(os.path.join(folder, f"{k}.npz"))) for k in names
            if os.path.exists(os.path.join(folder, f"{k}.npz"))}

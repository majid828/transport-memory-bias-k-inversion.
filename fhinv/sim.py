"""Synthetic aquifer simulator for Chapter 3.

Model (dimensionless units; cell size dx = dy = 1):
  * ln K(x, y) = mu + sigma * z(x, y), z a standardized Gaussian random field with a
    squared-exponential covariance exp(-r^2 / ell^2) (spectral method).
  * Steady confined flow  div(K grad h) = 0  on an nx-by-ny cell-centred grid, fixed heads
    h = 1 (left face) and h = 0 (right face), no-flow top and bottom (finite volumes,
    harmonic-mean transmissibilities).  With fixed-head boundaries the head field does not
    depend on the overall magnitude of K (scaling K by a constant leaves h unchanged).
  * Conservative tracer, mobile-immobile exchange (single rate):
        dCm/dt = [-div(q Cm) + theta_m D lap(Cm)] / theta_m - beta*alpha*(Cm - Cim)
        dCim/dt = alpha*(Cm - Cim)
    beta = theta_im/theta_m (capacity ratio), alpha (exchange rate).  beta = 0 is the
    advection-dispersion model.  Explicit upwind advection + explicit dispersion
    (operator split) and an exact update of the linear exchange step.
  * Observations: heads at 12 wells, the flux-weighted outlet breakthrough curve (BTC) at
    48 log-spaced times, one sparse plume snapshot (24 points) and the total discharge Q.

Default configuration (used for all main results): 32 x 16 grid, mobile porosity 0.30 for
every aquifer, dispersion 0.05.  Options used only by the review checks:
  * configure(nx, ny): change the grid size (wells and snapshot points are scaled with it);
  * theta: per-aquifer mobile porosity (array), never given to the inverse models;
  * disp: per-aquifer dispersion coefficient;
  * beta may be a field (N, ny, nx) instead of one value per aquifer.
With the defaults these options reproduce the original numerics exactly.
"""
from __future__ import annotations

import os

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

THETA_REF = 0.30                         # mobile porosity
DISP = 0.05                              # local dispersion coefficient (cell^2 / time)


def configure(nx=32, ny=16):
    """Set grid size and everything that scales with it (call before generating data)."""
    global NX, NY, T_REF, T_END, BTC_TIMES, SNAP_TIME, WELLS, SNAP_PTS
    NX, NY = nx, ny
    T_REF = THETA_REF * NX ** 2 / 1.0      # advective time for K = 1, unit head drop, theta = 0.3 (307.2 on 32 x 16)
    T_END = 6.0 * T_REF
    BTC_TIMES = np.geomspace(0.05 * T_REF, T_END, 48)     # dense early, sparse late
    SNAP_TIME = 0.75 * T_REF
    fx, fy = nx / 32, ny / 16
    WELLS = [(int(round(j * fy)), int(round(i * fx))) for i in (6, 13, 19, 26) for j in (3, 8, 12)]          # (row, col) heads
    SNAP_PTS = [(int(round(j * fy)), int(round(i * fx))) for i in (8, 16, 24) for j in (1, 3, 5, 7, 8, 10, 12, 14)]


configure()


# ------------------------------------------------------------------ random fields
def gaussian_field(rng, ell, nx, ny, pad=2):
    """Standardized (spatial mean 0, sd 1) Gaussian random field, SE covariance."""
    Nx, Ny = nx * pad, ny * pad
    kx = np.fft.fftfreq(Nx) * 2 * np.pi
    ky = np.fft.fftfreq(Ny) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky)
    spec = np.exp(-0.25 * ell ** 2 * (KX ** 2 + KY ** 2))       # SE covariance spectrum (unnormalized)
    noise = rng.standard_normal((Ny, Nx)) + 1j * rng.standard_normal((Ny, Nx))
    f = np.real(np.fft.ifft2(noise * np.sqrt(spec)))[:ny, :nx]
    return (f - f.mean()) / f.std()


def sample_lnk(rng, n, mu_sd=0.5, sigma_range=(0.5, 1.0), ell_range=(3.0, 7.0)):
    """n random ln K fields: mean mu ~ N(0, mu_sd^2), sd sigma ~ U(sigma_range), correlation length ~ U(ell_range)."""
    mu = rng.normal(0.0, mu_sd, n)
    sig = rng.uniform(*sigma_range, n)
    ell = rng.uniform(*ell_range, n)
    fields = np.stack([mu[k] + sig[k] * gaussian_field(rng, ell[k], NX, NY) for k in range(n)])
    return fields, dict(mu=mu, sigma=sig, ell=ell)


# ------------------------------------------------------------------ steady flow
def solve_flow(lnk):
    """Return heads (ny, nx), face fluxes Fx (ny, nx+1) and Fy (ny+1, nx), and discharge Q."""
    K = np.exp(lnk)
    ny, nx = K.shape
    idx = np.arange(nx * ny).reshape(ny, nx)
    Tx = 2 * K[:, :-1] * K[:, 1:] / (K[:, :-1] + K[:, 1:])        # between (j,i) and (j,i+1)
    Ty = 2 * K[:-1, :] * K[1:, :] / (K[:-1, :] + K[1:, :])        # between (j,i) and (j+1,i)
    Tl, Tr = 2 * K[:, 0], 2 * K[:, -1]                              # half-cell boundary transmissibility
    rows, cols, vals = [], [], []
    diag = np.zeros((ny, nx))
    for T, a, b in ((Tx, idx[:, :-1], idx[:, 1:]), (Ty, idx[:-1, :], idx[1:, :])):
        rows += [a.ravel(), b.ravel()]; cols += [b.ravel(), a.ravel()]; vals += [-T.ravel(), -T.ravel()]
    diag[:, :-1] += Tx; diag[:, 1:] += Tx; diag[:-1, :] += Ty; diag[1:, :] += Ty
    diag[:, 0] += Tl; diag[:, -1] += Tr
    rows.append(idx.ravel()); cols.append(idx.ravel()); vals.append(diag.ravel())
    A = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(nx * ny,) * 2)
    rhs = np.zeros((ny, nx)); rhs[:, 0] = Tl * 1.0
    h = spla.spsolve(A, rhs.ravel()).reshape(ny, nx)
    Fx = np.zeros((ny, nx + 1)); Fy = np.zeros((ny + 1, nx))
    Fx[:, 0] = Tl * (1.0 - h[:, 0]); Fx[:, 1:-1] = Tx * (h[:, :-1] - h[:, 1:]); Fx[:, -1] = Tr * h[:, -1]
    Fy[1:-1, :] = Ty * (h[:-1, :] - h[1:, :])
    return h, Fx, Fy, float(Fx[:, -1].sum())


# ------------------------------------------------------------------ batched transport
def _cfl_dt(Fx, Fy, theta=THETA_REF, disp=None):
    """Positivity (hence stability) of the explicit upwind + dispersion update:
    dt * (outflow/theta_m + 4 D) <= 0.9 in every cell."""
    disp = DISP if disp is None else disp
    out = (np.maximum(Fx[:, 1:], 0) + np.maximum(-Fx[:, :-1], 0) + np.maximum(Fy[1:, :], 0) + np.maximum(-Fy[:-1, :], 0))
    return 0.9 / (out / theta + 4 * disp).max()


def transport_batch(Fx, Fy, beta, alpha, theta, t_end=None, disp=None):
    """Simulate a batch of N aquifers at once (common time step = smallest CFL step).
    Fx: (N, ny, nx+1), Fy: (N, ny+1, nx), beta: (N,) or (N, ny, nx), alpha, theta: (N,).
    Returns BTC (N, len(BTC_TIMES)), snapshot concentrations (N, len(SNAP_PTS)), mass left, number of steps."""
    t_end = T_END if t_end is None else t_end
    N, ny, nxp = Fx.shape
    nx = nxp - 1
    dsp = np.full(N, DISP) if disp is None else np.asarray(disp, float)   # local dispersion per aquifer
    dt = min(_cfl_dt(Fx[k], Fy[k], theta[k], dsp[k]) for k in range(N))
    nsteps = int(np.ceil(t_end / dt)); dt = t_end / nsteps
    th = theta[:, None, None]
    Cm = np.zeros((N, ny, nx)); Cim = np.zeros((N, ny, nx))
    inflow = np.maximum(Fx[:, :, 0], 0)
    Cm[:, :, 0] = inflow / inflow.sum(axis=1, keepdims=True) / th[:, :, 0]       # unit mass, flux weighted
    Fxp, Fxn = np.maximum(Fx, 0), np.minimum(Fx, 0)
    Fyp, Fyn = np.maximum(Fy, 0), np.minimum(Fy, 0)
    b = beta[:, None, None] if beta.ndim == 1 else beta      # beta may be uniform (N,) or a field (N, ny, nx)
    a = alpha[:, None, None]
    decay = np.exp(-a * (1 + b) * dt)
    qout = Fx[:, :, -1]; qsum = qout.sum(axis=1)
    btc = np.zeros((N, len(BTC_TIMES))); snap = np.zeros((N, len(SNAP_PTS)))
    ti, snap_done = 0, False
    sj = np.array([p[0] for p in SNAP_PTS]); si = np.array([p[1] for p in SNAP_PTS])
    t = 0.0
    for step in range(nsteps):
        # upwind advective face fluxes of mass
        Cxl = np.concatenate([np.zeros((N, ny, 1)), Cm], axis=2)     # upstream candidate for +x flow
        Cxr = np.concatenate([Cm, np.zeros((N, ny, 1))], axis=2)     # upstream candidate for -x flow
        Gx = Fxp * Cxl + Fxn * Cxr
        Cyl = np.concatenate([np.zeros((N, 1, nx)), Cm], axis=1)
        Cyr = np.concatenate([Cm, np.zeros((N, 1, nx))], axis=1)
        Gy = Fyp * Cyl + Fyn * Cyr
        adv = -(Gx[:, :, 1:] - Gx[:, :, :-1]) - (Gy[:, 1:, :] - Gy[:, :-1, :])
        lap = np.zeros_like(Cm)
        dxp = np.diff(Cm, axis=2); lap[:, :, :-1] += dxp; lap[:, :, 1:] -= dxp
        dyp = np.diff(Cm, axis=1); lap[:, :-1, :] += dyp; lap[:, 1:, :] -= dyp
        Cm = Cm + dt * (adv / th + (DISP * lap if disp is None else dsp[:, None, None] * lap))
        # exact mobile-immobile exchange step
        s = Cm + b * Cim; d = (Cm - Cim) * decay
        Cm = (s + b * d) / (1 + b); Cim = (s - d) / (1 + b)
        t += dt
        if not snap_done and t >= SNAP_TIME:
            snap = Cm[:, sj, si].copy(); snap_done = True
        while ti < len(BTC_TIMES) and t >= BTC_TIMES[ti] - 1e-9:
            btc[:, ti] = (qout * Cm[:, :, -1]).sum(axis=1) / qsum
            ti += 1
    if beta.ndim == 1:
        mass = theta * (Cm.sum(axis=(1, 2)) + b[:, 0, 0] * Cim.sum(axis=(1, 2)))
    else:
        mass = theta * (Cm.sum(axis=(1, 2)) + (b * Cim).sum(axis=(1, 2)))
    return btc, snap, mass, nsteps


def simulate(lnk, beta, alpha, theta=None, batch=256, verbose=False, disp=None):
    """Full pipeline for many aquifers: flow per field, transport batched by similar time step.
    Set the environment variable FHINV_WORKERS > 1 to run the batches in parallel (identical results)."""
    n = len(lnk)
    theta = np.full(n, THETA_REF) if theta is None else np.asarray(theta, float)
    heads = np.zeros((n, len(WELLS))); Q = np.zeros(n)
    Fx = np.zeros((n, NY, NX + 1)); Fy = np.zeros((n, NY + 1, NX)); dts = np.zeros(n)
    wj = np.array([w[0] for w in WELLS]); wi = np.array([w[1] for w in WELLS])
    for k in range(n):
        h, fx, fy, q = solve_flow(lnk[k])
        heads[k] = h[wj, wi]; Q[k] = q; Fx[k] = fx; Fy[k] = fy
        dts[k] = _cfl_dt(fx, fy, theta[k], None if disp is None else disp[k])
    order = np.argsort(dts)                       # group aquifers with similar stable time steps
    btc = np.zeros((n, len(BTC_TIMES))); snap = np.zeros((n, len(SNAP_PTS))); mass = np.zeros(n)
    groups = [order[s:s + batch] for s in range(0, n, batch)]
    workers = int(os.environ.get("FHINV_WORKERS", "1"))
    jobs = [(Fx[ids], Fy[ids], beta[ids], alpha[ids], theta[ids], None, None if disp is None else np.asarray(disp)[ids]) for ids in groups]
    if workers > 1 and len(groups) > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(workers) as pool:
            results = pool.starmap(transport_batch, jobs)     # each batch is independent: identical results
    else:
        results = [transport_batch(*j) for j in jobs]
    for gi, (ids, (b_, s_, m_, ns)) in enumerate(zip(groups, results)):
        btc[ids], snap[ids], mass[ids] = b_, s_, m_
        if verbose:
            print(f"  transport batch {gi + 1}/{len(groups)}: {len(ids)} aquifers, {ns} steps", flush=True)
    return dict(heads=heads, Q=Q, btc=btc, snap=snap, mass_left=mass)


# ------------------------------------------------------------------ observation noise
def add_noise(obs, rng, rel=0.05, head_sd=0.005, conc_floor=1e-5):
    """Heads: additive N(0, 0.005); BTC / snapshot: 5% multiplicative + small floor; Q: 5%."""
    out = {}
    out["heads"] = obs["heads"] + rng.normal(0, head_sd, obs["heads"].shape)
    for k in ("btc", "snap"):
        x = obs[k] * (1 + rel * rng.standard_normal(obs[k].shape)) + conc_floor * rng.standard_normal(obs[k].shape)
        out[k] = np.maximum(x, 0.0)
    out["Q"] = obs["Q"] * (1 + rel * rng.standard_normal(obs["Q"].shape))
    return out

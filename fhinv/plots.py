"""Figures for the preliminary results (drawn from saved results)."""
from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import sim  # noqa: E402
from .data import CONDITIONS  # noqa: E402

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "font.family": "serif", "mathtext.fontset": "cm", "axes.linewidth": 0.6})
OK = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]
LABEL = {"heads_only": "heads only", "blind_btc": "blind: BTC", "blind_btc_h": "blind: BTC+heads",
         "blind_all": "blind: all data", "agnostic_btc_h": "agnostic: BTC+heads", "joint_btc_h": "joint: BTC+heads",
         "joint_btc_h_s": "joint: +snapshot", "agnostic_all": "agnostic: all data", "joint_all": "joint: all data"}


def _save(fig, out, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, "figures", f"{name}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def _idx(tc, cond, field):
    return int(np.where((tc["cond"] == cond) & (tc["field"] == field))[0][0])


def make_all(out, sets, basis, df, fhd, tab, preds_keep, fw):
    tc = sets["test_controlled"]
    info = json.load(open(os.path.join(out, "run_info.json")))

    # ---- Figure 1: one fixed true field, inferred under different memory conditions
    f = int(np.argsort(tc["mu"][tc["cond"] == 0])[len(np.unique(tc["field"])) // 2])
    blind = preds_keep["blind_btc_h"][1]; joint = preds_keep["joint_all"][1]
    panels = [("true ln K", tc["lnk"][_idx(tc, 0, f)]),
              ("blind, no memory", blind[_idx(tc, 0, f)]),
              (r"blind, $\beta$=1, Da=30", blind[_idx(tc, 3, f)]),
              (r"blind, $\beta$=1, Da=1", blind[_idx(tc, 6, f)]),
              (r"joint (all data), $\beta$=1, Da=30", joint[_idx(tc, 3, f)])]
    vmin = min(p[1].min() for p in panels); vmax = max(p[1].max() for p in panels)
    fig = plt.figure(figsize=(7.0, 3.6))
    gs = fig.add_gridspec(2, 5, height_ratios=[1, 1.1], hspace=0.55, wspace=0.08)
    for k, (t, a) in enumerate(panels):
        ax = fig.add_subplot(gs[0, k])
        im = ax.imshow(a, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(t + f"\nmean ln K = {a.mean():.2f}", fontsize=6.5); ax.set_xticks([]); ax.set_yticks([])
    cb = fig.colorbar(im, ax=fig.axes, fraction=0.012, pad=0.01); cb.set_label("ln K", fontsize=7)
    ax = fig.add_subplot(gs[1, :3])
    for c, col, lab in ((0, OK[0], "no memory"), (3, OK[5], r"$\beta$=1, Da=30 (equilibrium)"),
                        (6, OK[6], r"$\beta$=1, Da=1 (non-equilibrium)")):
        ax.plot(sim.BTC_TIMES / sim.T_REF, tc["clean_btc"][_idx(tc, c, f)], color=col, lw=1.2, label=lab)
    ax.set_xlabel(r"time / $T_{ref}$"); ax.set_ylabel("outlet concentration"); ax.legend(frameon=False)
    ax.set_title("same true K field, three memory conditions", fontsize=7)
    _save(fig, out, "fig1_example_fields")

    # ---- Figure 2: false heterogeneity -- change in inferred mean and pattern caused by memory alone
    pc = pd.read_csv(os.path.join(out, "tables", "table_pattern_change_seed0.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
    ax = axes[0]
    bb = np.linspace(0, 1.05, 50)
    ax.plot(bb, -np.log1p(bb), color="0.5", lw=1, ls=":", label=r"prediction $-\ln(1+\beta)$")
    for name, col in (("blind_btc", OK[1]), ("blind_btc_h", OK[5]), ("blind_all", OK[3])):
        for da, ls, mk in ((30.0, "-", "o"), (1.0, "--", "s")):
            s = fhd[(fhd.method == name) & (fhd.Da == da)].sort_values("beta")
            x = np.concatenate([[0], s.beta]); y = np.concatenate([[0], s.dmu_mean])
            lo = np.concatenate([[0], s.dmu_mean - s.dmu_ci_lo]); hi = np.concatenate([[0], s.dmu_ci_hi - s.dmu_mean])
            ax.errorbar(x, y, yerr=[lo, hi], color=col, ls=ls, marker=mk, ms=3, lw=1, capsize=1.5,
                        label=f"{LABEL[name]}, Da={da:g}")
    ax.set_xlabel(r"capacity ratio $\beta$"); ax.set_ylabel(r"change in estimated mean ln K")
    ax.set_title("(a) memory-induced shift of the inferred mean", loc="left")
    ax.legend(frameon=False, fontsize=5.6, ncol=1)
    ax = axes[1]
    for name, col in (("blind_btc_h", OK[5]), ("agnostic_all", OK[7]), ("joint_all", OK[6])):
        for da, ls, mk in ((30.0, "-", "o"), (1.0, "--", "s")):
            s = pc[(pc.method == name) & (pc.Da == da)].sort_values("beta")
            ax.errorbar(s.beta, s.pattern_change_rmse, yerr=[s.pattern_change_rmse - s.ci_lo, s.ci_hi - s.pattern_change_rmse],
                        color=col, ls=ls, marker=mk, ms=3, lw=1, capsize=1.5, label=f"{LABEL[name]}, Da={da:g}")
    ctrl = os.path.join(out, "tables", "table_controls.csv")
    if os.path.exists(ctrl):
        nf = pd.read_csv(ctrl)["noise_floor_pattern_change_mean"].mean()
        ax.axhspan(0, nf, color="0.88", lw=0, zorder=0)
        ax.text(0.26, nf * 1.005, "noise floor (same data, new noise)", fontsize=5.8, color="0.3", va="bottom")
        ax.set_ylim(nf * 0.9, None)
    ax.set_xlabel(r"capacity ratio $\beta$"); ax.set_ylabel("RMSE between inferred patterns\n(memory vs no memory)")
    ax.set_title("(b) memory-induced change of the inferred pattern", loc="left")
    ax.legend(frameon=False, fontsize=5.6)
    fig.tight_layout()
    _save(fig, out, "fig2_false_heterogeneity")

    # ---- Figure 3: method comparison on three conditions
    conds = [(0, "no memory"), (3, r"$\beta$=1, Da=30"), (6, r"$\beta$=1, Da=1")]
    methods = list(LABEL)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    w = 0.26
    for j, (metric, ttl) in enumerate((("mean_rmse", "(a) error of estimated mean ln K (RMS)"),
                                       ("pattern_rmse", "(b) error of estimated ln K pattern (RMSE)"))):
        ax = axes[j]
        for k, (c, lab) in enumerate(conds):
            s = tab[tab.cond == c].set_index("method").loc[methods]
            ax.bar(np.arange(len(methods)) + (k - 1) * w, s[f"{metric}_mean"], w, yerr=s[f"{metric}_std"],
                   color=[OK[0], OK[5], OK[6]][k], alpha=0.85, capsize=1.5, label=lab, error_kw=dict(lw=0.6))
        if metric == "mean_rmse":
            ax.axhline(info["prior_mu_sd"], color="0.4", ls=":", lw=0.8)
            ax.text(len(methods) - 0.5, info["prior_mu_sd"] * 1.03, "prior spread", ha="right", fontsize=6, color="0.3")
        else:
            ax.axhline(info["pca_floor_total_rmse"], color="0.4", ls=":", lw=0.8)
            ax.text(len(methods) - 0.5, info["pca_floor_total_rmse"] * 1.03, "representation floor", ha="right",
                    fontsize=6, color="0.3")
        ax.set_xticks(np.arange(len(methods))); ax.set_xticklabels([LABEL[m] for m in methods], rotation=60, ha="right")
        ax.set_title(ttl, loc="left")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    _save(fig, out, "fig3_method_comparison")

    # ---- Figure 4: forward consistency
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    for name, col in (("blind_btc_h", OK[5]), ("joint_all", OK[6])):
        for da, ls, mk in ((30.0, "-", "o"), (1.0, "--", "s")):
            s = fw[(fw.method == name) & ((fw.Da == da) | (fw.beta == 0))].sort_values("beta")
            ax.errorbar(s.beta, s.btc_rel_l2_mean, yerr=[s.btc_rel_l2_mean - s.btc_rel_l2_ci_lo, s.btc_rel_l2_ci_hi - s.btc_rel_l2_mean],
                        color=col, ls=ls, marker=mk, ms=3, lw=1, capsize=1.5, label=f"{LABEL[name]}, Da={da:g}")
    ax.set_xlabel(r"capacity ratio $\beta$"); ax.set_ylabel("relative BTC misfit of re-simulation")
    ax.legend(frameon=False, fontsize=5.8)
    fig.tight_layout()
    _save(fig, out, "fig4_forward_consistency")

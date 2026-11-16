"""Trace every number on the Chapter 3 proposal-defense slides to the stored results in this repository.

    python slide_numbers.py           # prints slide, quantity, value on the slide, value recomputed, source
                                      # and exits with an error if any value does not match

Sources (see README.md):
  results/first_run/      tables of the first full run (slides 29, 30, 32 and the written proposal)
  results/review_checks/  review checks run with this code (slide 31, notes of slides 29 and 31)
  results/main/           re-run of run_all.py with this code (design numbers on slides 26, 27, 87, 88)
  slide_figures/output/   statistics written by the slide-figure scripts (slides 25, 28 and backup slide 81)
  fhinv/                  design constants of the simulator and the datasets
Run run_all.py, the review checks and the slide-figure scripts first (or use the stored results).
"""
import inspect
import json
import math
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from fhinv import data as D, learn as L, sim  # noqa: E402

FR = os.path.join(ROOT, "results", "first_run")
RC = os.path.join(ROOT, "results", "review_checks")
MAIN = os.path.join(ROOT, "results", "main")
SF = os.path.join(ROOT, "slide_figures", "output")
csv = lambda *p: pd.read_csv(os.path.join(*p))
js = lambda *p: json.load(open(os.path.join(*p)))
f2 = lambda x: f"{x:.2f}".replace("-", "−")
f0 = lambda x: f"{x:.0f}"
rng2 = lambda a, b: f"{f2(a)}–{f2(b)}"
pct_low = lambda d: 100 * (1 - math.exp(d))               # shift of mean ln K -> % lower typical K
ROWS = []


def check(slide, what, shown, value, source):
    ROWS.append(dict(slide=slide, quantity=what, on_slide=shown, recomputed=value, source=source, ok=shown == value))


# ---------------------------------------------------------------- first run (slides 23, 26, 29, 30, 32)
fh = csv(FR, "tables", "table_false_heterogeneity.csv")
mt = csv(FR, "tables", "table_methods_controlled.csv").set_index(["method", "cond"])
fw = csv(FR, "tables", "table_forward_check.csv")
ct = csv(FR, "tables", "table_controls.csv").set_index("method")
info_fr = js(FR, "run_info.json")
v = lambda m, c, col="mean_rmse_mean", t=mt: float(t.loc[(m, c), col])
blind = fh[fh.method == "blind_btc_h"]
fast, slow = blind[blind.Da == 30].sort_values("beta"), blind[blind.Da == 1].sort_values("beta")
SRC_FH = "first_run/tables/table_false_heterogeneity.csv (blind_btc_h, dmu_mean)"
for s in (23, 32):
    check(s, "memory-blind shift of mean ln K, all 6 conditions", "−0.17 to −0.54",
          f"{f2(blind.dmu_mean.max())} to {f2(blind.dmu_mean.min())}", SRC_FH)
check(26, "mean ln K dropped by (all 6 conditions)", "0.17–0.54", f"{-blind.dmu_mean.max():.2f}–{-blind.dmu_mean.min():.2f}", SRC_FH)

# slide 29 (Experiment B)
st = js(SF, "s12_simple_stats.json")
check(29, "K too low, fast exchange (%)", "16, 27 and 38", ", ".join(f0(pct_low(x)) for x in fast.dmu_mean[:2]) + " and " + f0(pct_low(fast.dmu_mean.iloc[2])), SRC_FH)
check(29, "timing rule K/(1+β) (%)", "20, 33 and 50", ", ".join(f0(100 * (1 - 1 / (1 + b))) for b in (0.25, 0.5)) + " and " + f0(50.0), "formula 1 − 1/(1 + β)")
check(29, "K too low, slow exchange (%)", "25–42", f"{f0(min(map(pct_low, slow.dmu_mean)))}–{f0(max(map(pct_low, slow.dmu_mean)))}", SRC_FH)
pc = csv(FR, "tables", "table_pattern_change_seed0.csv"); pc = pc[pc.method == "blind_btc_h"]
nf = float(ct.noise_floor_pattern_change_mean.mean())
check(29, "pattern change above noise", "0.02–0.09", rng2(pc.pattern_change_rmse.min() - nf, pc.pattern_change_rmse.max() - nf),
      "first_run/tables/table_pattern_change_seed0.csv − noise floor")
check(29, "noise floor of the pattern change", "0.20", f2(nf), "first_run/tables/table_controls.csv")
check(29, "error vs true map rose at most", "≤ 0.03", f"≤ {f2(blind.dpattern_rmse_mean.max())}", SRC_FH.replace("dmu_mean", "dpattern_rmse_mean"))
check(29, "heads only: largest shift", "≤ 0.002", f"≤ {fh[fh.method == 'heads_only'].dmu_mean.abs().max():.3f}", SRC_FH.replace("blind_btc_h", "heads_only"))
check(29, "notes: shift in ln K, fast", "0.17, 0.31 and 0.48", ", ".join(f"{-x:.2f}" for x in fast.dmu_mean[:2]) + f" and {-fast.dmu_mean.iloc[2]:.2f}", SRC_FH)
check(29, "notes: shift in ln K, slow", "0.29, 0.41 and 0.54", ", ".join(f"{-x:.2f}" for x in slow.dmu_mean[:2]) + f" and {-slow.dmu_mean.iloc[2]:.2f}", SRC_FH)
r = [x / -math.log1p(b) * 100 for x, b in zip(fast.dmu_mean, fast.beta)]
check(29, "notes: fast shift as % of ln(1+β)", "69 to 78", f"{f0(min(r))} to {f0(max(r))}", SRC_FH)
check(29, "notes: slope est. vs true mean ln K (this network)", "1.01", f2(ct.loc["blind_btc_h", "mu_slope_mean"]), "first_run/tables/table_controls.csv")
check(29, "notes: slope, all networks", "0.95 to 1.05", f"{f2(ct.mu_slope_mean.min())} to {f2(ct.mu_slope_mean.max())}", "first_run/tables/table_controls.csv")
t2 = csv(RC, "baseline", "table2_summary.csv")
an = t2[t2.row.str.contains("area-normalized")].fraction * 100
check(29, "notes: Da = 1000, area-normalized BTC, % of rule", "92 to 98", f"{f0(an.min())} to {f0(an.max())}", "review_checks/baseline/table2_summary.csv")
for k, x in zip(("pct_Da30", "pct_Da1"), (fast, slow)):
    assert np.allclose(st[k], [round(pct_low(d), 1) for d in x.dmu_mean]), "slide figure stats differ from the table"

# slide 30 (Experiment C)
B, A, J = "blind_btc_h", "agnostic_btc_h", "joint_btc_h"
SRC_MT = "first_run/tables/table_methods_controlled.csv (mean_rmse_mean)"
check(30, "chart, test without memory (blind, agnostic, joint)", "0.081, 0.189, 0.159", ", ".join(f"{v(m, 0):.3f}" for m in (B, A, J)), SRC_MT)
check(30, "chart, β = 1 fast exchange", "0.488, 0.210, 0.173", ", ".join(f"{v(m, 3):.3f}" for m in (B, A, J)), SRC_MT)
check(30, "chart, β = 1 slow exchange", "0.554, 0.153, 0.147", ", ".join(f"{v(m, 6):.3f}" for m in (B, A, J)), SRC_MT)
check(30, "β = 1: error fell from ... to ... (fast)", "0.49 to 0.21", f"{f2(v(B, 3))} to {f2(v(A, 3))}", SRC_MT)
check(30, "β = 1: error fell from ... to ... (slow)", "0.55 to 0.15", f"{f2(v(B, 6))} to {f2(v(A, 6))}", SRC_MT)
d = [v(J, c) - v(A, c) for c in range(7)]
check(30, "joint − agnostic, chart conditions: at most ... lower", "0.04", f2(-min(v(J, c) - v(A, c) for c in (0, 3, 6))), SRC_MT)
check(30, "notes: joint − agnostic, all 7 conditions", "0.04 better to 0.01 worse", f"{-min(d):.2f} better to {max(d):.2f} worse", SRC_MT)
hi = sorted(round(100 * (math.exp(v(m, 0, "mean_bias_mean")) - 1)) for m in (A, J))
check(30, "no memory: K too high (%)", "10–15", f"{hi[0]}–{hi[1]}", "first_run/tables/table_methods_controlled.csv (mean_bias_mean)")
check(30, "no memory: error with / without memory training", "0.16–0.19 instead of 0.08",
      f"{f2(min(v(A, 0), v(J, 0)))}–{f2(max(v(A, 0), v(J, 0)))} instead of {f2(v(B, 0))}", SRC_MT)
check(30, "notes: β error of the joint model at β = 1 (fast, slow)", "0.29, 0.16",
      f"{f2(v(J, 3, 'beta_rmse_mean'))}, {f2(v(J, 6, 'beta_rmse_mean'))}", "first_run/tables/table_methods_controlled.csv (beta_rmse_mean)")

# slide 32 (conclusions)
check(32, "predicting β and Da changed the error by at most", "0.04", f2(max(abs(x) for x in d)), SRC_MT)
fb = fw[fw.method == "blind_btc_h"]; fj = fw[fw.method == "joint_all"]
SRC_FW = "first_run/tables/table_forward_check.csv (btc_rel_l2_mean)"
check(32, "forward check, memory-blind: no memory → β = 1", "0.43 → 1.36–1.47",
      f"{f2(fb[fb.beta == 0].btc_rel_l2_mean.iloc[0])} → {rng2(fb[fb.beta == 1].btc_rel_l2_mean.min(), fb[fb.beta == 1].btc_rel_l2_mean.max())}", SRC_FW)
check(32, "forward check, joint (all data) at β = 1", "0.26–0.38", rng2(fj[fj.beta == 1].btc_rel_l2_mean.min(), fj[fj.beta == 1].btc_rel_l2_mean.max()), SRC_FW)

# ---------------------------------------------------------------- review checks (slide 31, Experiment D)
ab = csv(RC, "ablation", "ablation_table.csv").set_index(["method", "cond"])
por = csv(RC, "porosity", "table_methods.csv").set_index(["method", "cond"])
SRC_AB = "review_checks/ablation/ablation_table.csv"
M = ["heads_only", "joint_btc_h", "joint_btc_h_s", "joint_btc_h_q", "joint_all"]
for c, lab, shown in ((0, "no memory", "0.481, 0.159, 0.136, 0.111, 0.105"), (3, "β = 1 fast", "0.484, 0.180, 0.163, 0.115, 0.121"),
                      (6, "β = 1 slow", "0.482, 0.142, 0.104, 0.118, 0.102")):
    check(31, f"chart, test {lab} (H, H+BTC, +snapshot, +Q, all four)", shown, ", ".join(f"{v(m, c, t=ab):.3f}" for m in M), SRC_AB + " (mean_rmse_mean)")
check(31, "heads only, all three tests", "0.48", "0.48" if all(f2(v("heads_only", c, t=ab)) == "0.48" for c in (0, 3, 6)) else "differs", SRC_AB)
check(31, "+ BTC: error range", "0.14 to 0.18", f"{f2(min(v('joint_btc_h', c, t=ab) for c in (0, 3, 6)))} to {f2(max(v('joint_btc_h', c, t=ab) for c in (0, 3, 6)))}", SRC_AB)
check(31, "+ Q, fast exchange", "0.18 → 0.11", f"{f2(v('joint_btc_h', 3, t=ab))} → {f2(v('joint_btc_h_q', 3, t=ab))}", SRC_AB)
check(31, "+ snapshot, slow exchange", "0.14 → 0.10", f"{f2(v('joint_btc_h', 6, t=ab))} → {f2(v('joint_btc_h_s', 6, t=ab))}", SRC_AB)
check(31, "notes: prior spread of mean ln K", "0.5", f"{js(MAIN, 'run_info.json')['prior_mu_sd']:.1f}", "main/run_info.json (prior_mu_sd)")
pat = lambda m: float(ab.loc[m]["pattern_rmse_mean"].mean())
check(31, "notes: pattern error, + snapshot", "0.54 to 0.48", f"{f2(pat('joint_btc_h'))} to {f2(pat('joint_btc_h_s'))}", SRC_AB + " (pattern_rmse_mean, mean of 7 tests)")
check(31, "notes: pattern error, heads only / all four", "0.49 / 0.47", f"{f2(pat('heads_only'))} / {f2(pat('joint_all'))}", SRC_AB)
check(31, "notes: β error, + Q (β = 1 fast)", "0.30 to 0.18", f"{f2(v('joint_btc_h', 3, 'beta_rmse_mean', ab))} to {f2(v('joint_btc_h_q', 3, 'beta_rmse_mean', ab))}", SRC_AB)
check(31, "notes: memory-blind with Q (β = 1 fast)", "0.43", f2(v("blind_btc_h_q", 3, t=ab)), SRC_AB)
check(31, "notes: β error, porosity unknown (joint, all data)", "0.18 to 0.31", f"{f2(v('joint_all', 3, 'beta_rmse_mean', ab))} to {f2(v('joint_all', 3, 'beta_rmse_mean', por))}",
      "review_checks/porosity/table_methods.csv")
dd = max(abs(v(m, c, t=ab) - v(m, c)) for m in ("heads_only", "joint_btc_h", "joint_btc_h_s", "joint_all") for c in (0, 3, 6))
check(31, "notes: chart bars vs first run, largest difference", "within 0.007", "within 0.007" if dd < 0.007 else f"{dd:.4f}",
      "ablation vs first_run/tables/table_methods_controlled.csv")

# ---------------------------------------------------------------- slide-figure statistics (slides 25, 28, 81)
s4 = js(SF, "s4_btc_simple_stats.json")
check(25, "storage curve: arrival vs no storage", "2 times later", f"{s4['storage']['mean_time'] / s4['true_K']['mean_time']:.0f} times later", "slide_figures/output/s4_btc_simple_stats.json")
check(25, "storage curve height vs half-K curve", "half as high", "half as high" if abs(s4["storage"]["peak"] / s4["half_K"]["peak"] - 0.5) < 1e-3 else "differs", "slide_figures/output/s4_btc_simple_stats.json")
fs = js(SF, "stats.json")
s11 = fs["s11"]
check(28, "test field shown", "38", str(s11["field"]), "slide_figures/output/stats.json (s11)")
check(28, "mean ln K estimate: no memory → Da = 30, Da = 1", "0.00 → −0.44, −0.40",
      f"{f2(abs(s11['est_mean']['no memory']) if abs(s11['est_mean']['no memory']) < 0.005 else s11['est_mean']['no memory'])} → "
      f"{f2(s11['est_mean']['β = 1, Da = 30'])}, {f2(s11['est_mean']['β = 1, Da = 1'])}", "slide_figures/output/stats.json (s11)")
check(28, "notes: pattern error of this field", "about 0.7 to 0.8",
      f"about {min(s11['pattern_rmse'].values()):.1f} to {max(s11['pattern_rmse'].values()):.1f}", "slide_figures/output/stats.json (s11)")
s3 = fs["s3"]
check(81, "largest head change when K is doubled", "9 × 10−15", f"{s3['max_head_change'] * 1e15:.0f} × 10−15", "slide_figures/output/stats.json (s3)")
check(81, "discharge Q: A → B (ratio)", "0.574 → 1.148 (ratio 2.000)", f"{s3['Q_A']:.3f} → {s3['Q_B']:.3f} (ratio {s3['ratio']:.3f})", "slide_figures/output/stats.json (s3)")

# ---------------------------------------------------------------- design (slides 24, 26, 27, 82, 85, 87, 88)
info = js(MAIN, "run_info.json")
check(26, "test β values", "0.25, 0.5, 1", ", ".join(f"{b:g}" for b in sorted({b for b, _ in D.CONDITIONS if b})), "fhinv/data.py CONDITIONS")
check(26, "test Da values", "30 and 1", " and ".join(f"{x:g}" for x in (30.0, 1.0) if any(da == x for _, da in D.CONDITIONS)), "fhinv/data.py CONDITIONS")
check(26, "mobile porosity θm", "0.30", f2(sim.THETA_REF), "fhinv/sim.py THETA_REF")
check(27, "training aquifers / test fields / conditions", "2,400 / 60 / 7", f"{info['sizes']['n_train']:,} / {info['sizes']['n_fields']} / {len(D.CONDITIONS)}", "main/run_info.json, fhinv/data.py")
check(27, "grid; head wells; snapshot points", "32 × 16; 12; 24", f"{sim.NX} × {sim.NY}; {len(sim.WELLS)}; {len(sim.SNAP_PTS)}", "fhinv/sim.py")
check(85, "N test fields, M cells", "60, 512", f"{info['sizes']['n_fields']}, {sim.NX * sim.NY}", "main/run_info.json, fhinv/sim.py")
check(87, "variance captured by 40 principal components", "97 %", f"{100 * info_fr['pca_explained']:.0f} %", "first_run/run_info.json (pca_explained)")
check(87, "principal components", "40", str(L.N_MODES), "fhinv/learn.py N_MODES")
check(88, "BTC times; snapshot time", "48 times, 0.05–6 Tref; 0.75 Tref",
      f"{len(sim.BTC_TIMES)} times, {sim.BTC_TIMES[0] / sim.T_REF:.2f}–{sim.BTC_TIMES[-1] / sim.T_REF:.0f} Tref; {sim.SNAP_TIME / sim.T_REF:.2f} Tref", "fhinv/sim.py")
check(88, "dispersion D", "0.05", f2(sim.DISP), "fhinv/sim.py DISP")
dflt = lambda f: {k: p.default for k, p in inspect.signature(f).parameters.items() if p.default is not inspect.Parameter.empty}
lk, mem, nz = dflt(sim.sample_lnk), dflt(D.sample_memory), dflt(sim.add_noise)
check(88, "ln K: mean sd; σ range; correlation length (cells)", "N(0, 0.5²); U(0.5, 1); U(3, 7)",
      f"N(0, {lk['mu_sd']:g}²); U({lk['sigma_range'][0]:g}, {lk['sigma_range'][1]:g}); U({lk['ell_range'][0]:g}, {lk['ell_range'][1]:g})", "fhinv/sim.py sample_lnk")
check(88, "training memory: share without memory; β range", "25 %; U(0.1, 1.5)",
      f"{100 * mem['p_zero']:.0f} %; U({mem['beta_range'][0]:g}, {mem['beta_range'][1]:g})", "fhinv/data.py sample_memory")
check(88, "noise: heads; concentrations and Q", "± 0.005; 5 %", f"± {nz['head_sd']:g}; {100 * nz['rel']:.0f} %", "fhinv/sim.py add_noise")
check(88, "training Da range", "0.3–100", f"{10 ** D.LOG_DA_RANGE[0]:.1f}–{10 ** D.LOG_DA_RANGE[1]:.0f}", "fhinv/data.py LOG_DA_RANGE")
check(24, "gravel: 1 m at K = 1e-2 m/s, i = 0.01, θ = 0.3", "≈ 50 minutes", f"≈ {1 * 0.3 / (1e-2 * 0.01) / 60:.0f} minutes", "t = L θ / (K i)")
check(24, "clay: 1 m at K = 1e-10 m/s", "≈ 9,500 years", f"≈ {round(1 * 0.3 / (1e-10 * 0.01) / (365.25 * 86400), -2):,.0f} years", "t = L θ / (K i)")
check(82, "Q = K i A (K = 1e-4 m/s, i = 0.05, A = 1 m²)", "0.432 m³/day", f"{1e-4 * 0.05 * 1 * 86400:.3f} m³/day", "Darcy's law")

# ---------------------------------------------------------------- report
df = pd.DataFrame(ROWS)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 70); pd.set_option("display.max_rows", 500)
print(df[["slide", "quantity", "on_slide", "recomputed", "ok"]].to_string(index=False))
print(f"\n{df.ok.sum()} of {len(df)} slide numbers match the stored results.")
df.to_csv(os.path.join(ROOT, "results", "slide_numbers.csv"), index=False)
if not df.ok.all():
    print("MISMATCH:\n" + df[~df.ok].to_string(index=False))
    sys.exit(1)

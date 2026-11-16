"""Slide 29 (Experiment B): memory-blind network (heads + BTC) on the controlled test.
(a) the magnitude shift shown as 'how much too low the typical K is' (1 - exp(shift in mean ln K), in %),
    with the timing rule K / (1 + beta);
(b) the pattern change, with the band for measurement noise alone.

    python slide_figures/fig_experiment_B.py              # tables of the first run (as on the slide)
    python slide_figures/fig_experiment_B.py --run main   # tables of results/main (reproducible re-run)
Output: output/s12_simple.png (first run) or output/s12_simple_main.png, plus the matching _stats.json."""
import os, sys, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
FONT = next((f for f in ('Liberation Sans', 'Arial') if any(e.name == f for e in font_manager.fontManager.ttflist)),
            'DejaVu Sans')       # Liberation Sans (used for the slides) and Arial have the same metrics

HERE = os.path.dirname(os.path.abspath(__file__))
SCR = os.path.abspath(os.path.join(HERE, '..'))
RUN = sys.argv[sys.argv.index('--run') + 1] if '--run' in sys.argv else 'first_run'
ORIG = os.path.join(SCR, 'results', RUN, 'tables')
OUT = os.path.join(HERE, 'output'); os.makedirs(OUT, exist_ok=True)
TAG = '' if RUN == 'first_run' else '_' + RUN
plt.rcParams.update({'font.family': FONT, 'font.size': 13, 'text.color': 'black', 'axes.labelcolor': 'black',
                     'xtick.color': 'black', 'ytick.color': 'black', 'axes.edgecolor': 'black', 'axes.linewidth': 0.8,
                     'axes.spines.top': False, 'axes.spines.right': False, 'savefig.dpi': 250, 'legend.frameon': False})
BLUE, ORANGE, GRAY, INK = '#0072B2', '#D55E00', '#7F7F7F', '#1A1A1A'

fh = pd.read_csv(os.path.join(ORIG, 'table_false_heterogeneity.csv'))
pc = pd.read_csv(os.path.join(ORIG, 'table_pattern_change_seed0.csv'))
ctrl = pd.read_csv(os.path.join(ORIG, 'table_controls.csv'))
nf = float(ctrl['noise_floor_pattern_change_mean'].mean())
pct = lambda d: 100 * (1 - np.exp(d))          # shift in mean ln K -> % lower typical (geometric-mean) K
S = {'noise_floor': nf}

fig, axs = plt.subplots(1, 2, figsize=(12.0, 3.75))
fig.subplots_adjust(left=0.065, right=0.99, bottom=0.17, top=0.88, wspace=0.22)
ax = axs[0]
bb = np.linspace(0, 1.05, 80)
ax.plot(bb, 100 * (1 - 1 / (1 + bb)), color=GRAY, lw=1.8, ls=(0, (1.5, 2)), label='simple timing rule: K ÷ (1 + β)')
for da, col, ls, mk, lab in [(30.0, BLUE, '-', 'o', 'fast exchange (Da = 30)'), (1.0, ORANGE, '--', 's', 'slow exchange (Da = 1)')]:
    s = fh[(fh.method == 'blind_btc_h') & (fh.Da == da)].sort_values('beta')
    x = np.r_[0, s.beta]; y = np.r_[0, pct(s.dmu_mean)]
    lo = np.r_[0, pct(s.dmu_mean) - pct(s.dmu_ci_hi)]; hi = np.r_[0, pct(s.dmu_ci_lo) - pct(s.dmu_mean)]
    ax.errorbar(x, y, yerr=[lo, hi], color=col, ls=ls, marker=mk, ms=7, lw=2.2, capsize=3, label=lab)
    S[f'pct_Da{int(da)}'] = [round(float(v), 1) for v in pct(s.dmu_mean)]
    S[f'shift_Da{int(da)}'] = [round(float(v), 3) for v in s.dmu_mean]
    for xv, yv in zip(s.beta, pct(s.dmu_mean)):
        if da == 30:
            ax.text(xv + 0.035, yv - 3.2, f'{yv:.0f} %', fontsize=11.5, color=col, ha='left', va='center')
        elif xv < 0.9:
            ax.text(xv - 0.035, yv + 3.6, f'{yv:.0f} %', fontsize=11.5, color=col, ha='right', va='center')
        else:
            ax.text(xv + 0.035, yv + 1.8, f'{yv:.0f} %', fontsize=11.5, color=col, ha='left', va='center')
S['rule_pct'] = [round(100 * (1 - 1 / (1 + b)), 1) for b in (0.25, 0.5, 1.0)]
ax.text(1.0, 52.5, '50 %', fontsize=11.5, color=GRAY, ha='center', va='bottom')
ax.set_xlim(-0.03, 1.2); ax.set_ylim(0, 58)
ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_xlabel('storage size β (stagnant ÷ flowing water)', fontsize=12.5)
ax.set_ylabel('K estimated too low by (%)', fontsize=12.5)
ax.legend(fontsize=11.5, loc='upper left', handlelength=2.4)
ax.set_title('(a) Size of K: how much too low?', fontsize=13.5, loc='left', fontweight='bold')

ax = axs[1]
ax.axhspan(0, nf, color='#E6E6E6', lw=0)
ax.text(0.2, nf - 0.004, 'new measurement noise alone changes the map this much', fontsize=11.5, va='top', color='#404040')
for da, col, ls, mk, lab in [(30.0, BLUE, '-', 'o', 'fast exchange'), (1.0, ORANGE, '--', 's', 'slow exchange')]:
    s = pc[(pc.method == 'blind_btc_h') & (pc.Da == da)].sort_values('beta')
    ax.errorbar(s.beta, s.pattern_change_rmse, yerr=[s.pattern_change_rmse - s.ci_lo, s.ci_hi - s.pattern_change_rmse],
                color=col, ls=ls, marker=mk, ms=7, lw=2.2, capsize=3, label=lab)
    S[f'pattern_Da{int(da)}'] = [round(float(v), 3) for v in s.pattern_change_rmse]
ax.set_xlim(0.15, 1.1); ax.set_ylim(0.17, 0.33)
ax.set_xticks([0.25, 0.5, 0.75, 1.0])
ax.set_xlabel('storage size β (stagnant ÷ flowing water)', fontsize=12.5)
ax.set_ylabel('pattern change (RMSE of ln K)', fontsize=12.5)
ax.legend(fontsize=11.5, loc='upper left', handlelength=2.4)
ax.set_title('(b) Pattern of K: did the map change?', fontsize=13.5, loc='left', fontweight='bold')
for a in axs:
    a.tick_params(labelsize=11.5)
fig.savefig(os.path.join(OUT, f's12_simple{TAG}.png'), facecolor='white')
above = [v - nf for k in ('pattern_Da30', 'pattern_Da1') for v in S[k]]
S['above_noise'] = [round(min(above), 3), round(max(above), 3)]
heads = fh[fh.method == 'heads_only'].dmu_mean.abs().max()
bb_ = fh[fh.method == 'blind_btc_h']
S['truth_err_rise_max'] = float(bb_.dpattern_rmse_mean.max())
S['ratio_ln_fast'] = [round(float(v), 3) for v in (bb_[bb_.Da == 30].dmu_mean / bb_[bb_.Da == 30].predicted_equilibrium_shift)]
ct_ = ctrl.set_index('method')
S['slope_this'] = float(ct_.loc['blind_btc_h', 'mu_slope_mean'])
S['slope_all'] = [float(ctrl.mu_slope_mean.min()), float(ctrl.mu_slope_mean.max())]
S['heads_only_max_abs'] = float(heads)
json.dump(S, open(os.path.join(OUT, f's12_simple{TAG}_stats.json'), 'w'), indent=1)
print(json.dumps(S))

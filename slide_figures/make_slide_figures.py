"""Chapter 3 slide figures computed from the experiment data (run run_all.py first, which makes data/main).

  s1_k_range.png   K ranges of four materials (redrawn after Freeze & Cherry 1979, Table 2.2)    slide 24
  s1_kmaps.png     one test field; same pattern with K halved; same mean with zones moved           slide 24
  s3_heads.png     heads of a field and of the same field with 2 x K (heads unchanged, Q doubled)   backup slide 81
  s4_causes.png    schematic: slow flow paths vs temporary storage                                  slide 25
  s9_design.png    one synthetic aquifer with the 12 head wells and 24 snapshot points             slides 27, 88
  s11_maps.png     true field and memory-blind estimates under three memory conditions (Exp. A)   slide 28
  s11_btc.png      breakthrough curves of the same field for three memory conditions (Exp. A)     slide 28
Numbers quoted on the slides are written to output/stats.json."""
import os, sys, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
FONT = next((f for f in ('Liberation Sans', 'Arial') if any(e.name == f for e in font_manager.fontManager.ttflist)),
            'DejaVu Sans')       # Liberation Sans (used for the slides) and Arial have the same metrics
from matplotlib.patches import Ellipse, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
SCR = os.path.abspath(os.path.join(HERE, '..'))
DATA = os.path.join(SCR, 'data', 'main')
OUT = os.path.join(HERE, 'output'); os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, SCR)
from fhinv import sim, learn as L  # noqa: E402

plt.rcParams.update({'font.family': FONT, 'font.size': 13, 'text.color': 'black', 'axes.labelcolor': 'black',
                     'xtick.color': 'black', 'ytick.color': 'black', 'axes.edgecolor': 'black', 'axes.linewidth': 0.8,
                     'axes.spines.top': False, 'axes.spines.right': False, 'savefig.dpi': 250, 'legend.frameon': False,
                     'mathtext.fontset': 'stix'})
C = dict(blue='#0072B2', orange='#E69F00', verm='#D55E00', green='#009E73', sky='#56B4E9', purple='#CC79A7',
         gray='#7F7F7F', light='#BFBFBF')
CMAP = 'viridis'
STATS = {}


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), facecolor='white')
    plt.close(fig)


def fmt(v):
    v = round(float(v), 2)
    v = 0.0 if v == 0 else v
    return f'{v:.2f}'.replace('-', '−')


def frame(a):
    a.set_xticks([]); a.set_yticks([])
    for s in a.spines.values():
        s.set_visible(True); s.set_linewidth(0.8); s.set_color('black')


TC = dict(np.load(os.path.join(DATA, 'test_controlled.npz')))
FIELD = int(np.argsort(TC['mu'][TC['cond'] == 0])[len(np.unique(TC['field'])) // 2])   # same choice as original Fig. 3.1


def idx(cond, f=FIELD):
    return int(np.where((TC['cond'] == cond) & (TC['field'] == f))[0][0])


# ------------------------------------------------------------------ slide 24
def fig_k_range():
    bands = [('Unweathered\nmarine clay', -12, -9), ('Silt, loess', -9, -5), ('Clean sand', -6, -2), ('Gravel', -3, 0)]
    cols = ['#C6DBEF', '#6BAED6', '#2171B5', '#08306B']
    fig, ax = plt.subplots(figsize=(5.9, 2.55))
    fig.subplots_adjust(left=0.2, right=0.97, bottom=0.25, top=0.97)
    for k, ((lab, a, b), col) in enumerate(zip(bands, cols)):
        ax.barh(3 - k, b - a, left=a, height=0.62, color=col)
        ax.text(a - 0.2, 3 - k, lab, ha='right', va='center', fontsize=12)
    ax.set_xlim(-12.4, 0.3); ax.set_ylim(-0.6, 3.6); ax.set_yticks([]); ax.spines['left'].set_visible(False)
    ax.set_xticks([-12, -10, -8, -6, -4, -2, 0])
    ax.set_xticklabels([r'$10^{-12}$', r'$10^{-10}$', r'$10^{-8}$', r'$10^{-6}$', r'$10^{-4}$', r'$10^{-2}$', '1'],
                       fontsize=12)
    ax.set_xlabel('hydraulic conductivity K (m/s)', fontsize=13)
    save(fig, 's1_k_range.png')


def fig_kmaps():
    true = TC['lnk'][idx(0)].astype(float)
    maps = [('Reference K map', true), ('Magnitude changed:\nevery K halved', true - np.log(2)),
            ('Pattern changed:\nsame mean, zones moved', true[::-1, ::-1])]
    STATS['s1_maps_mean'] = [round(float(m.mean()), 3) for _, m in maps]
    lo, hi = min(m.min() for _, m in maps), max(m.max() for _, m in maps)
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.05))
    fig.subplots_adjust(left=0.005, right=0.905, bottom=0.27, top=0.98, wspace=0.06)
    for a, (t, z) in zip(axs, maps):
        im = a.imshow(z, cmap=CMAP, vmin=lo, vmax=hi, origin='lower', interpolation='bilinear')
        frame(a); a.set_xlabel(t, fontsize=12, labelpad=5)
    cax = fig.add_axes([0.92, 0.3, 0.015, 0.66])
    cb = fig.colorbar(im, cax=cax); cb.set_label('ln K', fontsize=12); cb.ax.tick_params(labelsize=10)
    save(fig, 's1_kmaps.png')


# ------------------------------------------------------------------ backup slide 81
def fig_heads():
    lnk = TC['lnk'][idx(0)].astype(float)
    h1, fx1, fy1, Q1 = sim.solve_flow(lnk)
    h2, fx2, fy2, Q2 = sim.solve_flow(lnk + np.log(2))
    STATS['s3'] = dict(max_head_change=float(np.abs(h1 - h2).max()), Q_A=Q1, Q_B=Q2, ratio=Q2 / Q1)
    vmin, vmax = lnk.min(), lnk.max() + np.log(2)
    fig, axs = plt.subplots(1, 2, figsize=(8.2, 2.75))
    fig.subplots_adjust(left=0.01, right=0.9, bottom=0.03, top=0.83, wspace=0.05)
    for a, z, h, t in [(axs[0], lnk, h1, 'Aquifer A: K(x)'), (axs[1], lnk + np.log(2), h2, 'Aquifer B: 2 × K(x)')]:
        im = a.imshow(z, cmap=CMAP, origin='lower', vmin=vmin, vmax=vmax, interpolation='bilinear')
        cs = a.contour(h, levels=np.linspace(0.1, 0.9, 9), colors='white', linewidths=1.1, origin='lower')
        for txt in a.clabel(cs, levels=[0.2, 0.5, 0.8], fmt='%.1f', fontsize=10.5):
            txt.set_rotation(0)
        a.set_title(t, fontsize=13, loc='left', fontweight='bold'); frame(a)
    cax = fig.add_axes([0.915, 0.06, 0.016, 0.74])
    cb = fig.colorbar(im, cax=cax); cb.set_label('ln K', fontsize=12); cb.ax.tick_params(labelsize=10)
    save(fig, 's3_heads.png')


# ------------------------------------------------------------------ slide 25: schematic
def fig_causes():
    fig, axs = plt.subplots(1, 2, figsize=(6.2, 2.2))
    fig.subplots_adjust(left=0.005, right=0.995, bottom=0.01, top=0.99, wspace=0.06)
    ax = axs[0]; ax.set_xlim(0, 10); ax.set_ylim(0, 5.6); ax.axis('off')
    ax.add_patch(Rectangle((0.2, 0.3), 9.6, 4.2, fill=False, lw=0.9))
    ax.add_patch(Ellipse((5.2, 2.4), 3.4, 1.9, fc='#BDBDBD', ec='#7F7F7F', lw=0.8))
    ax.text(5.2, 2.4, 'low-K zone', ha='center', va='center', fontsize=12)
    xx = np.linspace(0.4, 9.6, 80)
    for yy, amp in [(0.75, 0.3), (1.3, 0.5), (3.5, 0.5), (4.05, 0.3)]:
        bend = amp * np.exp(-((xx - 5.2) / 2.0) ** 2) * (1 if yy > 2.4 else -1)
        ax.plot(xx, yy + bend, color=C['blue'], lw=1.4)
    for seg in (xx[xx < 3.45], xx[xx > 6.95]):
        ax.plot(seg, 2.4 + 0 * seg, color=C['blue'], lw=1.4, ls=(0, (2, 2)))
    ax.text(0.25, 5.15, 'A  Slow flow paths', fontsize=13, fontweight='bold', va='center')
    ax.annotate('', xy=(9.75, 5.15), xytext=(8.3, 5.15), arrowprops=dict(arrowstyle='->', lw=1.0))
    ax.text(8.2, 5.15, 'flow', fontsize=11.5, ha='right', va='center')
    ax = axs[1]; ax.set_xlim(0, 10); ax.set_ylim(0, 5.6); ax.axis('off')
    ax.add_patch(Rectangle((0.2, 0.3), 9.6, 4.2, fill=False, lw=0.9))
    ax.add_patch(Rectangle((0.2, 1.8), 9.6, 1.2, fc='#DCEBF5', lw=0))
    for x0 in [1.0, 3.6, 6.2]:
        ax.add_patch(Rectangle((x0, 3.0), 1.6, 1.1, fc='#BDBDBD', ec='#7F7F7F', lw=0.8))
        ax.add_patch(Rectangle((x0 + 1.2, 0.7), 1.6, 1.1, fc='#BDBDBD', ec='#7F7F7F', lw=0.8))
        ax.annotate('', xy=(x0 + 0.8, 3.3), xytext=(x0 + 0.8, 2.6), arrowprops=dict(arrowstyle='<->', lw=1.1,
                                                                                    color=C['verm']))
    ax.annotate('', xy=(9.4, 2.4), xytext=(7.2, 2.4), arrowprops=dict(arrowstyle='->', lw=1.3, color=C['blue']))
    ax.text(7.1, 2.4, 'flowing water', ha='right', va='center', fontsize=12)
    ax.text(0.25, 5.15, 'B  Temporary storage', fontsize=13, fontweight='bold', va='center')
    save(fig, 's4_causes.png')


# ------------------------------------------------------------------ slides 27 and 88: design of one synthetic aquifer
def fig_design():
    lnk = TC['lnk'][idx(0)].astype(float)
    fig = plt.figure(figsize=(6.6, 3.7))
    ax = fig.add_axes([0.13, 0.2, 0.64, 0.68])
    im = ax.imshow(lnk, cmap=CMAP, origin='lower', interpolation='bilinear', extent=(0, sim.NX, 0, sim.NY))
    wj = np.array([w[0] for w in sim.WELLS]); wi = np.array([w[1] for w in sim.WELLS])
    sj = np.array([p[0] for p in sim.SNAP_PTS]); si = np.array([p[1] for p in sim.SNAP_PTS])
    ax.plot(wi + 0.5, wj + 0.5, 'o', ms=8.5, mfc='white', mec='black', mew=1.2, label='12 head wells')
    ax.plot(si + 0.5, sj + 0.5, 's', ms=5.5, mfc=C['verm'], mec='black', mew=0.6, label='24 snapshot points')
    ax.set_xlim(0, sim.NX); ax.set_ylim(0, sim.NY); frame(ax)
    ax.text(-0.8, sim.NY / 2, 'inflow\nh = 1\n+ tracer', ha='right', va='center', fontsize=12)
    ax.text(sim.NX + 0.8, sim.NY / 2, 'outflow\nh = 0\nBTC, Q', ha='left', va='center', fontsize=12)
    ax.text(sim.NX / 2, sim.NY + 0.5, 'no-flow boundary', ha='center', va='bottom', fontsize=12)
    ax.text(sim.NX / 2, -0.5, 'no-flow boundary (32 × 16 cells)', ha='center', va='top', fontsize=12)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=12, handletextpad=0.3)
    cax = fig.add_axes([0.9, 0.2, 0.02, 0.68])
    cb = fig.colorbar(im, cax=cax, ticks=[-1, 0, 1, 2]); cb.set_label('ln K', fontsize=12); cb.ax.tick_params(labelsize=10.5)
    save(fig, 's9_design.png')


# ------------------------------------------------------------------ slide 28 (Experiment A): one field, three memory conditions
def fig_example():
    sets = {k: dict(np.load(os.path.join(DATA, f'{k}.npz'))) for k in ('train_mixed', 'train_blind', 'test_controlled')}
    basis = L.PCABasis(sets['train_mixed']['lnk'])
    p = L.fit_predict('blind_btc_h', sets, basis, 0)['test_controlled']        # memory-blind, heads + BTC, seed 0
    est = basis.field(p['mu'], p['coefs'])
    true = TC['lnk'][idx(0)].astype(float)
    conds = [(0, 'no memory', C['gray']), (3, 'β = 1, Da = 30', C['blue']), (6, 'β = 1, Da = 1', C['verm'])]
    panels = [('True field', true)] + [(f'Estimate: {lab}', est[idx(c)]) for c, lab, _ in conds]
    STATS['s11'] = dict(field=FIELD, true_mean=round(float(true.mean()), 3),
                        est_mean={lab: round(float(est[idx(c)].mean()), 3) for c, lab, _ in conds},
                        pattern_rmse={lab: round(float(np.sqrt((((est[idx(c)] - est[idx(c)].mean()) -
                                                                 (true - true.mean())) ** 2).mean())), 3)
                                      for c, lab, _ in conds})
    vmin = min(z.min() for _, z in panels); vmax = max(z.max() for _, z in panels)
    fig = plt.figure(figsize=(12.0, 2.15))
    for k, (t, z) in enumerate(panels):
        ax = fig.add_axes([0.005 + k * 0.232, 0.05, 0.218, 0.62])
        im = ax.imshow(z, cmap=CMAP, origin='lower', vmin=vmin, vmax=vmax, interpolation='bilinear')
        frame(ax)
        ax.set_title(f'{t}\nmean ln K = {fmt(z.mean())}', fontsize=12.5, pad=4)
    cax = fig.add_axes([0.94, 0.08, 0.009, 0.56])
    cb = fig.colorbar(im, cax=cax); cb.set_label('ln K', fontsize=11); cb.ax.tick_params(labelsize=9.5)
    save(fig, 's11_maps.png')
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    fig.subplots_adjust(left=0.13, right=0.97, bottom=0.2, top=0.96)
    tt = sim.BTC_TIMES / sim.T_REF
    for c, lab, col in conds:
        ax.plot(tt, TC['clean_btc'][idx(c)], color=col, lw=2.3, label=lab)
    ax.set_xscale('log'); ax.set_xlim(0.05, 6)
    ax.set_xlabel(r'time ÷ nominal travel time $T_\mathrm{ref}$', fontsize=12.5); ax.set_ylabel('outlet concentration', fontsize=12.5)
    ax.tick_params(labelsize=11); ax.legend(fontsize=11.5, loc='upper left')
    ax.set_yticks([])
    save(fig, 's11_btc.png')


if __name__ == '__main__':
    fig_k_range(); fig_kmaps(); fig_heads(); fig_causes(); fig_design(); fig_example()
    json.dump(STATS, open(os.path.join(OUT, 'stats.json'), 'w'), indent=1, default=float)
    print(json.dumps(STATS, indent=1, default=float))

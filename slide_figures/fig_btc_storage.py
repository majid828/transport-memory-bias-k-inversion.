"""Slide 25 ('slow flow or temporary storage?'): computed 1-D illustration, not experiment data.
1-D advection-dispersion, Peclet number 20, pulse input; exact Laplace-domain solution inverted with Talbot's method.
Curves: true K without storage; true K with fast storage (beta = 1, R = 2); half K (and half D) without storage.
Output: output/s4_btc_simple.png and output/s4_btc_simple_stats.json."""
import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
FONT = next((f for f in ('Liberation Sans', 'Arial') if any(e.name == f for e in font_manager.fontManager.ttflist)),
            'DejaVu Sans')       # Liberation Sans (used for the slides) and Arial have the same metrics

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'output'); os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'font.family': FONT, 'font.size': 13, 'text.color': 'black', 'axes.labelcolor': 'black',
                     'xtick.color': 'black', 'ytick.color': 'black', 'axes.edgecolor': 'black', 'axes.linewidth': 0.8,
                     'axes.spines.top': False, 'axes.spines.right': False, 'savefig.dpi': 250, 'legend.frameon': False})
GRAY, ORANGE, BLUE, INK = '#7F7F7F', '#E69F00', '#0072B2', '#1A1A1A'


def talbot(F, t, M=32):
    t = np.atleast_1d(t).astype(float); out = np.zeros_like(t)
    for i, ti in enumerate(t):
        r = 2 * M / (5 * ti); k = np.arange(1, M); th = k * np.pi / M; cot = 1 / np.tan(th)
        s = r * th * (cot + 1j); sig = th + (th * cot - 1) * cot
        val = 0.5 * np.exp(r * ti) * F(np.array([r + 0j]))[0] + np.sum((np.exp(ti * s) * F(s) * (1 + 1j * sig)).real)
        out[i] = (r / M * val).real
    return out


Lx, v, Dd = 1.0, 1.0, 0.05                       # Peclet number v L / D = 20
Fb = lambda g: (lambda s: np.exp(v * Lx / (2 * Dd) * (1 - np.sqrt(1 + 4 * Dd * g(s) / v ** 2))))
t = np.linspace(0.02, 6, 600)
c0 = talbot(Fb(lambda s: s), t)                   # true K, no storage
cR = talbot(Fb(lambda s: 2.0 * s), t)             # true K + fast storage, beta = 1 (R = 2)
cK = 2.0 * cR                                     # half K (and half D), no storage: same timing, half the discharge
tr = lambda c: float(np.trapezoid(c, t))
mean = lambda c: float(np.trapezoid(t * c, t) / np.trapezoid(c, t))
S = {k: dict(area=round(tr(c), 3), mean_time=round(mean(c), 3), peak=round(float(c.max()), 3), t_peak=round(float(t[np.argmax(c)]), 2))
     for k, c in (('true_K', c0), ('storage', cR), ('half_K', cK))}
print(json.dumps(S))
assert np.allclose(2 * cR, cK) and abs(S['half_K']['peak'] - S['true_K']['peak']) < 2e-3 and abs(S['half_K']['area'] - 2 * S['storage']['area']) < 1e-3

fig, ax = plt.subplots(figsize=(5.75, 3.55))
fig.subplots_adjust(left=0.09, right=0.985, bottom=0.165, top=0.975)
ax.plot(t, c0, color=GRAY, lw=2.2)
ax.plot(t, cK, color=ORANGE, lw=2.4, ls=(0, (5, 2)))
ax.plot(t, cR, color=BLUE, lw=2.8)
ax.set_xlim(0, 5); ax.set_ylim(0, 1.85); ax.set_yticks([])
ax.set_xticks([0, 1, 2, 3, 4, 5])
ax.set_xlabel('time  (1 = travel time without storage)', fontsize=12.5)
ax.set_ylabel('tracer at the outlet', fontsize=12.5)
ax.legend(['true K, no storage', 'half K, no storage', 'true K + storage'], fontsize=12, loc='upper right',
          bbox_to_anchor=(1.0, 0.93), handlelength=2.0, labelspacing=0.35)
ax.annotate('', xy=(1.73, S['half_K']['peak'] - 0.02), xytext=(1.73, S['storage']['peak'] + 0.02),
            arrowprops=dict(arrowstyle='<->', lw=1.1, color=INK, shrinkA=0, shrinkB=0))
ax.text(1.79, 0.89, 'half\nheight', fontsize=11.5, ha='left', va='center', linespacing=1.05)
# same arrival time
tp = S['half_K']['t_peak']
ax.annotate('', xy=(tp, 1.66), xytext=(S['true_K']['t_peak'], 1.66), arrowprops=dict(arrowstyle='->', lw=1.1, color=INK))
ax.text((tp + S['true_K']['t_peak']) / 2, 1.70, '2 × later', fontsize=11.5, ha='center', va='bottom')
ax.axvline(tp, ymin=0, ymax=0.68 / 1.85, color=INK, lw=0.7, ls=(0, (1.5, 2)))
ax.text(tp + 0.06, 1.62, 'same arrival time', fontsize=11.5, ha='left', va='center')
fig.savefig(os.path.join(OUT, 's4_btc_simple.png'), facecolor='white')
json.dump(S, open(os.path.join(OUT, 's4_btc_simple_stats.json'), 'w'), indent=1)
print('saved')

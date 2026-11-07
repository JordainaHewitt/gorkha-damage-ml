"""
Figure 1: the neural network used in this study (scripts/04_mlp.py), drawn in the style of the course's
MLP diagrams: standardized inputs and an optional learned VDC embedding, three hidden ReLU layers, and the
three output heads that define the three training losses. Widths are read from the script's SETTINGS.

Output: figures/network.pdf
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from common import colors, column_width, save_figure, script_settings

# --- Settings read from the training script's SETTINGS block ---
cfg = script_settings('04_mlp.py')
hidden, dropout, emb = cfg['HIDDEN'], cfg['DROPOUT'], cfg['EMBED_DIM']

# --- Figure ---
fig, ax = plt.subplots(figsize=(column_width, 2.0))
ax.set_xlim(0, 11); ax.set_ylim(1.0, 7.6); ax.axis('off')
node = dict(s=20, zorder=3, edgecolors='black', linewidths=0.5)


def column(x, ys, color, **kw):
    ax.scatter([x] * len(ys), ys, color=color, **{**node, **kw})
    return list(ys)


def connect(x1, ys1, x2, ys2, **kw):
    for a in ys1:
        for b in ys2:
            ax.plot([x1, x2], [a, b], color='0.78', lw=0.3, zorder=1, **kw)


def spread(n, y0, gap=0.55):
    return [y0 + gap * (i - (n - 1) / 2) for i in range(n)]


xs = [2.3, 4.0, 5.4, 6.8]
x_in = column(xs[0], spread(5, 4.6), 'white')
e_in = column(xs[0], spread(2, 1.75), colors[3])                 # optional VDC embedding e_v
h = [column(xs[1], spread(7, 3.9), colors[0]), column(xs[2], spread(7, 3.9), colors[0]),
     column(xs[3], spread(5, 3.9), colors[0])]
connect(xs[0], x_in, xs[1], h[0])
connect(xs[0], e_in, xs[1], h[0], ls='--')
connect(xs[1], h[0], xs[2], h[1]); connect(xs[2], h[1], xs[3], h[2])
for x, ys in zip(xs[1:], h):
    ax.text(x, min(ys) - 0.4, r'$\vdots$', ha='center', va='center', fontsize=7)
ax.text(xs[0], min(x_in) - 0.4, r'$\vdots$', ha='center', va='center', fontsize=7)

ax.text(xs[0], 6.15, '$h_0$\n', ha='center', va='bottom', fontsize=6.5)
for i, (x, w) in enumerate(zip(xs[1:], hidden)):
    ax.text(x, 6.15, f'$h_{i + 1}$\n{w}', ha='center', va='bottom', fontsize=6.5)
ax.text(xs[2], 7.05, f'ReLU, dropout {dropout}', ha='center', va='bottom', fontsize=6, color='0.3')
ax.text(xs[0] - 0.4, 4.6, '26 inputs $\\tilde x$\n(standardized):\nstories, age,\narea, height,\nMSI, material,\nslope, roof,\nposition',
        ha='right', va='center', fontsize=5.3, color='0.25')
ax.text(xs[0] - 0.4, 1.75, f'VDC embedding\n$e_v$ ({emb}, learned;\n"+ VDC emb." only)',
        ha='right', va='center', fontsize=5.3, color='0.25')

heads = [(5.5, 'softmax $\\to \\hat p_{1..5}$', 'cross-entropy'),
         (3.9, 'linear $\\to f$', 'squared error'),
         (2.3, '$P(y{>}k)=\\sigma(g{+}b_k)$', 'CORAL, $k=1..4$')]
for y, top, bottom in heads:
    ax.add_patch(FancyBboxPatch((7.55, y - 0.6), 3.35, 1.2, boxstyle='round,pad=0.03', fc='white',
                                ec=colors[1], lw=0.8))
    ax.text(9.22, y + 0.2, top, ha='center', va='center', fontsize=5.8)
    ax.text(9.22, y - 0.28, bottom, ha='center', va='center', fontsize=5.3, color='0.35')
    for b in h[2]:
        ax.plot([xs[3], 7.55], [b, y], color='0.78', lw=0.3, zorder=1)
ax.text(9.22, 6.35, 'one head per model', ha='center', va='bottom', fontsize=6)

save_figure(fig, 'network')

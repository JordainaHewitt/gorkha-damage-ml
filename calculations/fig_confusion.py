"""
Figure 2: the same model family trained three ways, against the paper's model.
(a)-(d) Test confusion matrices, five grades, full features, in the format of the paper's Fig 8:
        the paper's random forest regression (reproduced) and XGBoost trained as a classifier,
        a regressor and an ordinal model. Rows are true grades and sum to 1 (leading zeros omitted).
(e)     How far off the predicted grades are: share exactly right and share off by 3 or more grades.

Input:  results/{rf,xgb}/confusions.json
Output: figures/confusion.pdf
"""

import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from common import colors, family_color, full_width, result, save_figure

# --- Calculations ---
models = [('rf', 'RFR', 'RF regression\n(paper model)', 'RF'),
          ('xgb', 'XGBC', 'XGBoost\nclassifier', 'class'),
          ('xgb', 'XGBR', 'XGBoost\nregressor', 'regr'),
          ('xgb', 'XGBO', 'XGBoost\nordinal', 'ord')]
cms, exact, far = [], [], []
dist = np.abs(np.subtract.outer(np.arange(5), np.arange(5)))      # |true - predicted| in grades
for folder, model, _, _ in models:
    cm = np.array(json.load(open(result(folder, 'confusions.json')))[f'{model} full 5dg'], dtype=float)
    cms.append(cm / cm.sum(1, keepdims=True))
    exact.append(100 * cm[dist == 0].sum() / cm.sum())
    far.append(100 * cm[dist >= 3].sum() / cm.sum())

# --- Figure ---
fig = plt.figure(figsize=(full_width, 1.7))
gs = GridSpec(2, 5, figure=fig, width_ratios=[5, 5, 5, 5, 5.8], wspace=0.35, hspace=0.9)
ticks = [str(i) for i in range(1, 6)]
for i, (m, (_, _, title, _)) in enumerate(zip(cms, models)):
    ax = fig.add_subplot(gs[:, i])
    ax.imshow(m, cmap='Blues', vmin=0, vmax=1)
    for r in range(5):
        for c in range(5):
            ax.text(c, r, f'{m[r, c]:.2f}'.lstrip('0'), ha='center', va='center', fontsize=6,
                    color='white' if m[r, c] > 0.5 else 'black')
    ax.set_xticks(range(5), ticks, fontsize=6); ax.set_yticks(range(5), ticks if i == 0 else [], fontsize=6)
    ax.set_title(f'({"abcd"[i]}) {title}', fontsize=6.5)
    ax.set_xlabel('Predicted grade', fontsize=6.5)
    ax.grid(False); ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    if i == 0:
        ax.set_ylabel('True grade', fontsize=6.5)

bar_colors = [family_color['RF']] + [family_color['XGB']] * 3
labels = [m[3] for m in models]
for row, (vals, name, fmt) in enumerate(((exact, '(e) exactly right (%)', '{:.0f}'),
                                         (far, 'off by 3+ grades (%)', '{:.1f}'))):
    ax = fig.add_subplot(gs[row, 4])
    ax.bar(range(4), vals, color=bar_colors, width=0.65)
    for x, v in enumerate(vals):
        ax.text(x, v, fmt.format(v), ha='center', va='bottom', fontsize=5.8)
    ax.set_ylim(0, max(vals) * 1.3); ax.set_yticks([])
    ax.spines['left'].set_visible(False); ax.grid(False)
    ax.set_title(name, fontsize=6.5, loc='left')
    ax.set_xticks(range(4), labels if row == 1 else [], fontsize=5.6)
    if row == 1:
        ax.text(2, -0.42, 'XGBoost', transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=6,
                color=family_color['XGB'])
    ax.tick_params(length=0)

save_figure(fig, 'confusion')

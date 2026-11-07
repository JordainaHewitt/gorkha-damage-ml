"""
Figure 3: SHAP explanations of XGBoost, full features, 5 grades.
(a) Grouped importance: share of RF impurity importance (the paper's Fig 7 method) vs share of
    XGBoost-regression mean |SHAP|.
(b) Ordinal XGBoost: share of mean |SHAP| per damage threshold P(DG > k), top four parameters.

Inputs: results/shap/grouped_importance_vs_rf.csv, grouped_importance_ordinal_thresholds.csv
Output: figures/shap.pdf
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import colors, family_color, full_width, result, save_figure

# --- Calculations ---
g = pd.read_csv(result('shap', 'grouped_importance_vs_rf.csv'), index_col=0)
g = g.sort_values('xgbr_shap_share')
t = pd.read_csv(result('shap', 'grouped_importance_ordinal_thresholds.csv'), index_col=0)
share = t / t.sum(0)

# --- Figure ---
fig, (a, b) = plt.subplots(1, 2, figsize=(full_width, 1.95), gridspec_kw={'width_ratios': [1.15, 1]})
y = np.arange(len(g)); h = 0.38
a.barh(y + h / 2, 100 * g.rf_impurity_share, height=h, color=family_color['RF'], label='RF impurity (paper)')
a.barh(y - h / 2, 100 * g.xgbr_shap_share, height=h, color=family_color['XGB'], label='XGBoost mean |SHAP|')
a.set_yticks(y, g.index, fontsize=7); a.set_xlabel('Share of importance (%)')
a.legend(frameon=False, fontsize=7, loc='lower right')
a.grid(axis='y', visible=False)
a.set_title('(a) Grouped importance', fontsize=8, loc='left')

top = share.index[:4]
for i, name in enumerate(top):
    b.plot(range(4), 100 * share.loc[name], marker='o', color=colors[i], label=name)
b.set_xticks(range(4), [c.replace('DG>', 'DG > ') for c in share.columns], fontsize=7)
b.set_xlabel('Ordinal threshold'); b.set_ylabel('Share of mean |SHAP| (%)')
b.legend(frameon=False, fontsize=7, loc='center right')
b.set_title('(b) Ordinal XGBoost, per threshold', fontsize=8, loc='left')

save_figure(fig, 'shap')

"""
Figure 2: MSI acts as a location identifier, and what SHAP says drives damage.
(a) MLP training curves (cross-entropy vs epoch), plain vs with a learned old-VDC embedding.
(b) XGBoost regression: mean SHAP value of MSI in each old VDC, against the VDC's MSI.
(c) Grouped importance: RF impurity share (the paper's Fig 7 method) vs XGBoost mean |SHAP| share.

Inputs: results/mlp*/training_history.json (scripts/04_mlp.py),
        results/shap/msi_effect_per_vdc.csv, grouped_importance_vs_rf.csv (scripts/05_shap.py)
Output: figures/location.pdf
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import family_color, full_width, result, save_figure

# --- Calculations ---
hist = {lab: json.load(open(result(folder, 'training_history.json')))['full_5dg_ce']
        for lab, folder in (('MLP plain', 'mlp'), ('MLP', 'mlp_embed'))}

per_vdc = pd.read_csv(result('shap', 'msi_effect_per_vdc.csv')).set_index('msi')
per_vdc = per_vdc[per_vdc.n_buildings >= 5]                 # VDCs with at least 5 explained buildings
trend = per_vdc.mean_shap_msi.rolling(25, center=True, min_periods=10).median()

g = pd.read_csv(result('shap', 'grouped_importance_vs_rf.csv'), index_col=0).sort_values('xgbr_shap_share')

# --- Figure ---
fig, (a, b, c) = plt.subplots(1, 3, figsize=(full_width, 1.5), gridspec_kw={'width_ratios': [1, 1, 1.15]})
for lab, name in (('MLP plain', 'MSI only'), ('MLP', '+ VDC embedding')):
    h = hist[lab][1:]                                        # skip epoch 0 (untrained network)
    e = [r['epoch'] for r in h]
    a.plot(e, [r['train'] for r in h], ls='--', lw=1, color=family_color[lab])
    a.plot(e, [r['val'] for r in h], lw=1.5, color=family_color[lab], label=name)
a.plot([], [], ls='--', lw=1, color='0.4', label='training')
a.set_xlabel('Epoch'); a.set_ylabel('Cross-entropy loss')
a.legend(frameon=False, fontsize=6.5, loc='center right')
a.set_title('(a) MLP training curves', fontsize=8, loc='left')

b.scatter(per_vdc.index, per_vdc.mean_shap_msi, s=5, color=family_color['XGB'], alpha=0.7, lw=0,
          label='one VDC')
b.plot(trend.index, trend.values, color='black', lw=1.2, label='running median')
b.axhline(0, color='0.5', lw=0.6)
b.set_xlabel('MSI of the VDC (MMI)'); b.set_ylabel('Mean SHAP of MSI')
b.legend(fontsize=6.5, loc='lower right', frameon=True, facecolor='white', edgecolor='none', framealpha=0.9)
b.set_title('(b) XGBoost: MSI effect per VDC', fontsize=8, loc='left')

y = np.arange(len(g)); h = 0.38
c.barh(y + h / 2, 100 * g.rf_impurity_share, height=h, color=family_color['RF'], label='RF impurity (paper)')
c.barh(y - h / 2, 100 * g.xgbr_shap_share, height=h, color=family_color['XGB'], label='XGBoost |SHAP|')
c.set_yticks(y, g.index, fontsize=5.2); c.set_xlabel('Share of importance (%)')
c.legend(frameon=False, fontsize=6.5, loc='lower right')
c.grid(axis='y', visible=False)
c.set_title('(c) Grouped importance', fontsize=8, loc='left')

save_figure(fig, 'location')

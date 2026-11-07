"""
Numbers quoted in the text of the paper, written as LaTeX macros so the prose updates with the results.
Use them in paper.tex as, e.g., \\shapMSIxgb{} or \\nTrain{}. (LaTeX macro names: letters only.)

Inputs:  results/ (scripts/01..06), data/processed/split.json
Output:  tables/numbers.tex
"""

import json
import os

import pandas as pd

from common import repo_dir, result, save_table

# --- Calculations ---
macros = {}

split = json.load(open(os.path.join(repo_dir, 'data', 'processed', 'split.json')))
n = {k: len(v) for k, v in split.items()}
macros['nBuildings'] = f"{sum(n.values()):,}"
macros['nTrain'], macros['nVal'], macros['nTest'] = (f"{n[k]:,}" for k in ('train', 'val', 'test'))

a = pd.read_csv(result('comparison', 'all_models_test.csv'))


def m(fam, model, metric, setting='full', scale='5dg'):
    r = a[(a.family == fam) & (a.model == model) & (a.setting == setting) & (a.scale == scale)]
    return float(r[metric].iloc[0])


def recall(fam, model, k, scale='5dg'):
    r = a[(a.family == fam) & (a.model == model) & (a.setting == 'full') & (a.scale == scale)]
    return json.loads(r.recall.iloc[0])[k]


for tag, fam, model in (('RFR', 'RF', 'RFR'), ('RFC', 'RF', 'RFC'), ('XGBC', 'XGB', 'XGBC'),
                        ('XGBR', 'XGB', 'XGBR'), ('XGBO', 'XGB', 'XGBO'),
                        ('MLPC', 'MLP + VDC emb.', 'MLPC'), ('MLPO', 'MLP + VDC emb.', 'MLPO'),
                        ('MLPOmean', 'MLP + VDC emb.', 'MLPO_mean'), ('MLPplainC', 'MLP (plain)', 'MLPC'),
                        ('MLPplainO', 'MLP (plain)', 'MLPO')):
    for met, short in (('accuracy', 'Acc'), ('mae', 'MAE'), ('mse', 'MSE'), ('qwk', 'QWK')):
        macros[f'{tag}{short}'] = f"{m(fam, model, met):.3f}"
        macros[f'{tag}{short}TL'] = f"{m(fam, model, met, scale='tl'):.3f}"

# recall change at the extreme grades, ordinal models vs RFR
for tag, fam, model in (('XGBO', 'XGB', 'XGBO'), ('MLPO', 'MLP + VDC emb.', 'MLPO')):
    macros[f'{tag}dRecallDGone'] = f"{recall(fam, model, 0) - recall('RF', 'RFR', 0):+.2f}"
    macros[f'{tag}dRecallDGfive'] = f"{recall(fam, model, 4) - recall('RF', 'RFR', 4):+.2f}"
    macros[f'{tag}dRecallDGfour'] = f"{recall(fam, model, 3) - recall('RF', 'RFR', 3):+.2f}"

# traffic light: recall of S, M, H for the paper's model and the ordinal models
for tag, fam, model in (('RFR', 'RF', 'RFR'), ('XGBO', 'XGB', 'XGBO'), ('MLPO', 'MLP + VDC emb.', 'MLPO')):
    for k, name in enumerate(('S', 'M', 'H')):
        macros[f'{tag}RecallTL{name}'] = f"{recall(fam, model, k, scale='tl'):.2f}"

# how far off the predicted grade is (five grades, full features): % exact and % off by 3 or more grades
import numpy as np
dist = np.abs(np.subtract.outer(np.arange(5), np.arange(5)))
for tag, folder, key in (('RFR', 'rf', 'RFR full 5dg'), ('XGBC', 'xgb', 'XGBC full 5dg'),
                         ('XGBR', 'xgb', 'XGBR full 5dg'), ('XGBO', 'xgb', 'XGBO full 5dg'),
                         ('MLPC', 'mlp_embed', 'MLPC full 5dg'), ('MLPO', 'mlp_embed', 'MLPO full 5dg')):
    cm = np.array(json.load(open(result(folder, 'confusions.json')))[key], dtype=float)
    macros[f'{tag}pctExact'] = f"{100 * cm[dist == 0].sum() / cm.sum():.0f}"
    macros[f'{tag}pctFar'] = f"{100 * cm[dist >= 3].sum() / cm.sum():.1f}"
    macros[f'{tag}recallDGone'] = f"{cm[0, 0] / cm[0].sum():.2f}"
    macros[f'{tag}recallDGfive'] = f"{cm[4, 4] / cm[4].sum():.2f}"

# XGBoost ordinal: threshold crossing before the monotonicity fix
d = json.load(open(result('xgb', 'diagnostics.json')))
macros['crossFiveDG'] = f"{100 * d['full_5dg']['ordinal_crossing_rate_test']:.1f}"
macros['crossTL'] = f"{100 * d['full_tl']['ordinal_crossing_rate_test']:.1f}"

# CORAL thresholds (MLP with VDC embedding)
c = json.load(open(result('mlp_embed', 'diagnostics.json')))
macros['coralBiases'] = ', '.join(f"{b:.2f}" for b in c['full_5dg_coral']['coral_biases'])
macros['coralOrderedAll'] = 'all' if all(v['biases_ordered'] for v in c.values()) else 'not all'

# plain vs embedding MLP: final validation loss (cross-entropy)
h_plain = json.load(open(result('mlp', 'training_history.json')))['full_5dg_ce']
h_emb = json.load(open(result('mlp_embed', 'training_history.json')))['full_5dg_ce']
macros['valCEplain'] = f"{min(r['val'] for r in h_plain):.2f}"
macros['valCEemb'] = f"{min(r['val'] for r in h_emb):.2f}"
macros['trainCEplain'] = f"{h_plain[-1]['train']:.2f}"

# SHAP: grouped importance shares and per-threshold shares
g = pd.read_csv(result('shap', 'grouped_importance_vs_rf.csv'), index_col=0)
pct = lambda x: f"{100 * x:.0f}"
macros['shapMSIxgb'] = pct(g.loc['MSI', 'xgbr_shap_share'])
macros['shapMSIrf'] = pct(g.loc['MSI', 'rf_impurity_share'])
macros['shapMatxgb'] = pct(g.loc['Construction material', 'xgbr_shap_share'])
macros['shapMatrf'] = pct(g.loc['Construction material', 'rf_impurity_share'])
macros['shapAgexgb'] = pct(g.loc['Age', 'xgbr_shap_share'])
t = pd.read_csv(result('shap', 'grouped_importance_ordinal_thresholds.csv'), index_col=0)
s = t / t.sum(0)
macros['shapMSIfirst'] = pct(s.loc['MSI', 'DG>1'])
macros['shapMSIlast'] = pct(s.loc['MSI', 'DG>4'])
macros['shapMatfirst'] = pct(s.loc['Construction material', 'DG>1'])
macros['shapMatlast'] = pct(s.loc['Construction material', 'DG>4'])
macros['nShap'] = f"{int(pd.read_csv(result('shap', 'msi_effect_per_vdc.csv')).n_buildings.sum()):,}"

# --- Output ---
text = "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in macros.items()) + "\n"
save_table(text, 'numbers')

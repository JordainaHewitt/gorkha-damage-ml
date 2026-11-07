"""
Table 2: every model family x training loss, full features, test set.
Five-grade metrics, plus accuracy and QWK on the traffic-light scale. Best value per column in bold.

The first row is the paper's own random forest result (Ghimire et al. 2022, Table 3), so the table also
shows the reproduction.

Input:  results/comparison/all_models_test.csv (scripts/06_compare.py), results/rf/vs_paper.csv
Output: tables/models.tex
"""

import pandas as pd

from common import result, save_table

# --- Calculations ---
a = pd.read_csv(result('comparison', 'all_models_test.csv'))
a = a[a.setting == 'full']
order = [  # (family in the csv, model, label in the table)
    ('RF', 'RFC', 'RF, classification'),
    ('RF', 'RFR', 'RF, regression (paper)'),
    ('XGB', 'XGBC', 'XGBoost, classification'),
    ('XGB', 'XGBR', 'XGBoost, regression'),
    ('XGB', 'XGBO', 'XGBoost, ordinal (median)'),
    ('XGB', 'XGBO_mean', 'XGBoost, ordinal (mean)'),
    ('MLP + VDC emb.', 'MLPC', 'MLP, classification'),
    ('MLP + VDC emb.', 'MLPR', 'MLP, regression'),
    ('MLP + VDC emb.', 'MLPO', 'MLP, CORAL (median)'),
    ('MLP + VDC emb.', 'MLPO_mean', 'MLP, CORAL (mean)'),
    ('MLP (plain)', 'MLPC', 'MLP without VDC emb., classif.'),
]
cols = [('5dg', 'accuracy', 'max'), ('5dg', 'mae', 'min'), ('5dg', 'mse', 'min'), ('5dg', 'within_1', 'max'),
        ('5dg', 'qwk', 'max'), ('tl', 'accuracy', 'max'), ('tl', 'qwk', 'max')]


def value(fam, model, scale, metric):
    r = a[(a.family == fam) & (a.model == model) & (a.scale == scale)]
    return float(r[metric].iloc[0])


table = {lab: [value(f, m, s, c) for s, c, _ in cols] for f, m, lab in order}
best = []
for j, (_, _, sense) in enumerate(cols):
    vals = [round(v[j], 3) for v in table.values()]
    best.append(max(vals) if sense == 'max' else min(vals))

paper = pd.read_csv(result('rf', 'vs_paper.csv')).query("setting == 'full'").set_index('scale')

# --- Table ---
p5, pt = paper.loc['5dg'], paper.loc['tl']
rows = [r'\textit{RF, regression: paper, reported} & \textit{%.2f} & \textit{%.2f} & \textit{%.2f} & -- & -- & '
        r'\textit{%.2f} & -- \\' % (p5.accuracy_paper, p5.mae_paper, p5.mse_paper, pt.accuracy_paper)]
for lab, vals in table.items():
    cells = [(r'\textbf{%.3f}' if round(v, 3) == b else '%.3f') % v for v, b in zip(vals, best)]
    rows.append(lab + ' & ' + ' & '.join(cells) + r' \\')
    if lab.startswith('RF, regression') or lab.startswith('XGBoost, ordinal (mean)') \
            or lab.startswith('MLP, CORAL (mean)'):
        rows.append(r'\addlinespace')
text = r"""\begin{tabular}{lccccccc}
\toprule
 & \multicolumn{5}{c}{5 damage grades} & \multicolumn{2}{c}{Traffic light} \\
\cmidrule(lr){2-6}\cmidrule(lr){7-8}
Model, training loss & Acc. & MAE & MSE & Within-1 & QWK & Acc. & QWK \\ \midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
"""
save_table(text, 'models')

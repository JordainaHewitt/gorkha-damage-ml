"""
Table 1: model setup and hyperparameters. Random forest values are the paper's (Ghimire et al. 2022,
Table 2), which scripts/01_baseline_rf.py uses unchanged; XGBoost and MLP values are read from the SETTINGS
blocks of scripts/03_xgboost.py and scripts/04_mlp.py, so the table always matches the code that ran.

Output: tables/setup.tex
"""

from common import save_table, script_settings

# --- Calculations ---
x = script_settings('03_xgboost.py')
p, rounds, early = x['PARAMS'], x['ROUNDS'], x['EARLY_STOP']
m = script_settings('04_mlp.py')
rf = script_settings('01_baseline_rf.py')
widths = '--'.join(str(w) for w in m['HIDDEN'])

rows = [
    ('Structure', f"{rf['TREES']} trees; depth 20 (regr.), 30 (class.)",
     f"$\\le${rounds} trees, depth {p['max_depth']}, $\\eta$ = {p['learning_rate']}",
     f"{len(m['HIDDEN'])} ReLU layers, {widths}"),
    ('Regulariz.', 'bootstrap rows, random features',
     f"row/col.\\ subsample {p['subsample']}/{p['colsample_bytree']}, $\\lambda$ = {p['reg_lambda']:g}",
     f"dropout {m['DROPOUT']}, weight decay {m['WEIGHT_DECAY']:.0e}".replace('e-0', 'e-')),
    ('Training', 'greedy splits', f"boosting on errors, {p['max_bin']} bins",
     f"Adam, lr {m['LR']:.0e}, batch {m['BATCH']}".replace('e-0', 'e-')),
    ('Stopping', 'fixed size', f"{early} rounds w/o val.\\ gain",
     f"{m['PATIENCE']} epochs w/o val.\\ gain ($\\le${m['EPOCHS']})"),
]

# --- Table ---
body = "\n".join(f"{a} & {b} & {c} & {d} \\\\" for a, b, c, d in rows)
text = r"""\begin{tabularx}{\columnwidth}{@{}l>{\raggedright\arraybackslash}X>{\raggedright\arraybackslash}X>{\raggedright\arraybackslash}X@{}}
\toprule
 & RF (paper) & XGBoost & MLP \\ \midrule
""" + body + r"""
\bottomrule
\end{tabularx}
"""
save_table(text, 'setup')

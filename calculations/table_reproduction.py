"""
Table 1: random forest regression (RFR) reproduction vs the paper's Table 3 (test set).

Input:  results/rf/vs_paper.csv (scripts/01_baseline_rf.py)
Output: tables/reproduction.tex
"""

import pandas as pd

from common import result, save_table

# --- Calculations ---
v = pd.read_csv(result('rf', 'vs_paper.csv'))
names = {('full', '5dg'): 'Full, 5 grades', ('full', 'tl'): 'Full, traffic light',
         ('basic', '5dg'): 'Basic, 5 grades', ('basic', 'tl'): 'Basic, traffic light'}

# --- Table ---
rows = []
for _, r in v.iterrows():
    rows.append(f"{names[(r.setting, r.scale)]} & {r.accuracy_paper:.2f} & {r.accuracy_ours:.3f} & "
                f"{r.mae_paper:.2f} & {r.mae_ours:.3f} & {r.mse_paper:.2f} & {r.mse_ours:.3f} \\\\")
text = r"""\begin{tabular}{lcccccc}
\toprule
 & \multicolumn{2}{c}{Accuracy} & \multicolumn{2}{c}{MAE} & \multicolumn{2}{c}{MSE} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}
Features, scale & Paper & Ours & Paper & Ours & Paper & Ours \\ \midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
"""
save_table(text, 'reproduction')

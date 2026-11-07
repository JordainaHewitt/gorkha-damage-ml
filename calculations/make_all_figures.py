"""
Regenerate every figure and table in the report.

Run this after changing any result or the style file:

    python3 calculations/make_all_figures.py      (or press Run in VS Code)

then recompile paper.tex (Overleaf does this on its own once Dropbox sync is on).
The model results must exist first: run scripts/01_baseline_rf.py ... scripts/06_compare.py.
Months later, this file is also the fastest way to find the code behind a figure.
"""

import os
import runpy

calc_dir = os.path.dirname(os.path.abspath(__file__))

# Listed in the order they appear in the paper
scripts = [
    'make_numbers.py',          # tables/numbers.tex: values quoted in the text
    'fig_network.py',           # Figure 1: the neural network
    'table_setup.py',           # Table 1: model setup and hyperparameters
    'table_models.py',          # Table 2: all models + the paper's reported values
    'fig_confusion.py',         # Figure 2: confusion matrices, paper Fig 8/9 format
    'fig_location.py',          # Figure 3: training curves, MSI effect per VDC, grouped SHAP importance
    'table_reproduction.py',    # all four settings vs the paper; not in the 3-page text (appendix option)
    'fig_shap.py',              # SHAP per ordinal threshold; not in the 3-page text (appendix option)
]

for script in scripts:
    print(f'Running {script}')
    runpy.run_path(os.path.join(calc_dir, script), run_name='__main__')

print('Done. Recompile paper.tex to see the updated figures and tables.')

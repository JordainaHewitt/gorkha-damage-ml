# Predicting earthquake building damage: Gorkha 2015

A reproduction and extension of:

> Ghimire, S., Guéguen, P., Giffard-Roisin, S., & Schorlemmer, D. (2022). Testing machine learning
> models for seismic damage prediction at a regional scale using building-damage dataset compiled
> after the 2015 Gorkha Nepal earthquake. *Earthquake Spectra*, 38(4), 2970–2993.
> https://doi.org/10.1177/87552930221106495

Course project, CEE 292 Scientific Machine Learning (Stanford, fall 2026).

## What was done

The goal is to predict the damage grade (DG1 = slight to DG5 = collapse) of 762,106 buildings
surveyed after the 2015 Gorkha earthquake, from building features (material, age, height, area, ...)
and the shaking intensity at the building's location (MSI, from the USGS ShakeMap).

1. **Reproduce the paper.** Random forest regression with the paper's settings (1000 trees).
2. **Try stronger models.** XGBoost and a neural network (MLP, PyTorch).
3. **Respect the order of the grades.** Each model is trained three ways:
   - *classification*: pick one grade; grades are treated as unrelated labels
   - *regression*: predict a number on the 1–5 scale
   - *ordinal*: predict the chance that damage is above each grade
     (Frank & Hall for XGBoost, CORAL for the MLP)
4. **Give the network location.** MSI is one value per village area (VDC), so it mostly acts as a
   location label. Trees can use it that way; the plain MLP cannot. Adding a learned 16-number code
   for each VDC (an embedding) fixes this.
5. **Explain the models.** SHAP values show how much each input pushes each prediction up or down.

## Main result (test set, all features, 5 grades)

| Model | Accuracy | MAE | QWK |
|---|---|---|---|
| Random forest regression (paper's method) | 0.495 | 0.655 | 0.715 |
| XGBoost classification | **0.571** | 0.603 | 0.715 |
| XGBoost regression | 0.502 | 0.649 | 0.727 |
| XGBoost ordinal | 0.553 | **0.569** | 0.741 |
| MLP ordinal, no location code | 0.447 | 0.709 | 0.656 |
| MLP ordinal + VDC embedding | 0.528 | 0.580 | **0.744** |

Classification gets the most grades exactly right; ordinal models make the smallest and fewest
large errors (MAE, QWK). All models and settings: `results/comparison/all_models_test.csv`.

## Phases and scripts

| Phase | Script | Output |
|---|---|---|
| 0. Check the survey file | `scripts/check_survey.py` | printed report |
| 1. Shaking intensity (MSI) per area | `scripts/00_msi.py` | `data/processed/msi_*.csv` |
| 2. Attach MSI to each building | `scripts/00b_join_msi.py` | `data/processed/msi_buildings.csv` |
| 3. Random forest (paper reproduction) | `scripts/01_baseline_rf.py` | `results/rf/` |
| 4. Paper figures redone with our data | `scripts/02_paper_figures.py` | `results/paper_figures/` |
| 5. XGBoost: classification / regression / ordinal | `scripts/03_xgboost.py` | `results/xgb/` |
| 6. MLP: same three losses (`EMBED = True` adds the VDC code) | `scripts/04_mlp.py` | `results/mlp/`, `results/mlp_embed/` |
| 7. SHAP explanations | `scripts/05_shap.py` | `results/shap/` |
| 8. Comparison and training-curve figures | `scripts/06_compare.py` | `results/comparison/` |
| Report figures and tables | `calculations/make_all_figures.py` | `paper/figures/`, `paper/tables/` |

Shared code (features, train/val/test split, metrics, plots) is in `src/`.

## How to run

```bash
pip install -r requirements.txt
python3 scripts/00_msi.py
python3 scripts/00b_join_msi.py
python3 scripts/01_baseline_rf.py
python3 scripts/02_paper_figures.py
python3 scripts/03_xgboost.py
python3 scripts/04_mlp.py               # plain MLP
python3 scripts/04_mlp.py --embed       # MLP with the VDC embedding
python3 scripts/05_shap.py
python3 scripts/06_compare.py
python3 calculations/make_all_figures.py
```

Each script has a SETTINGS block at the top with its defaults (you can also press Run in VS Code).
Command-line flags override them, e.g. `python3 scripts/01_baseline_rf.py --trees 50` for a quick test.
The train/val/test split is saved once in `data/processed/split.json` and reused by every model.
Rough run times on CPU: XGBoost about 6 min, each MLP run 4–7 min.

The `results/` folder already holds the outputs, so `calculations/make_all_figures.py` works
without retraining. Trained models and the SHAP cache are not stored (too large); rerun the
scripts to rebuild them.

To test the pipeline without the real data:
`python3 scripts/make_synthetic.py --n 20000 --out data/synthetic` (fake numbers, for checking only).

## Data (not included; download into `data/raw/`)

| File | Source | Save to |
|---|---|---|
| Building survey table | 2015 Nepal Earthquake Open Data Portal (National Planning Commission / Kathmandu Living Labs), http://eq2015.npc.gov.np/; used here through the Kaggle mirror https://www.kaggle.com/datasets/arashnic/earthquake-magnitude-damage-and-impact (Open Database License) | `data/raw/npc/eq2015.csv` |
| ShakeMap grids (MMI) | USGS ShakeMap, event us20002926 (Atlas version) | `data/raw/shakemap/` |
| Pre-2017 VDC boundaries | HDX, MoFALD/HRRP 2016 shapefiles + code crosswalk | `data/raw/boundaries/mofald/` |
| Post-2017 municipality boundaries | HDX, COD-AB Nepal admin boundaries | `data/raw/boundaries/cod_ab/` |

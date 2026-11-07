"""SHAP explanations for the XGBoost models (professor's suggestion: "XGBoost and explanations via Shapley").

For one building x and one model f, the Shapley value of input j is its average marginal contribution
over all orders in which the inputs could be revealed:

Press Run, or: python3 scripts/05_shap.py        Output: results/shap/
"""
import argparse
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import data as D       # noqa: E402

# SETTINGS 
DATA = ROOT / "data/raw/npc/eq2015.csv"
MSI = ROOT / "data/processed/msi_buildings.csv"
MSI_COL = "msi_vdc"
XGB_DIR = ROOT / "results/xgb"                               # models from scripts/03_xgboost.py
RF_IMPORTANCE = ROOT / "results/paper_figures/fig07_importance.csv"   # from scripts/02_paper_figures.py
TAG = "full_5dg"
N_REG = 20000          # test buildings explained for XGBR
N_ORD = 20000          # test buildings explained for XGBO (same buildings as XGBR)
SEED = 0
OUT = ROOT / "results/shap"

BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1f1f1e", "#6b6a65", "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})


def nice_names(cols):
    s = pd.Index(cols)
    return (s.str.replace("has_superstructure_", "material: ").str.replace("land_surface_condition_", "slope: ")
            .str.replace("roof_type_", "roof: ").str.replace("position_", "position: ")
            .str.replace("count_floors_pre_eq", "Stories").str.replace("age_building", "Age")
            .str.replace("plinth_area_sq_ft", "Plinth area").str.replace("height_ft_pre_eq", "Height")
            .str.replace("msi", "MSI").str.replace("_", " "))


def groups(cols):
    """Parameter group of each column, same grouping as the paper's Fig 7(b)."""
    s = pd.Index(cols)
    g = np.select([s.str.startswith("has_superstructure"), s.str.startswith("land_surface"),
                   s.str.startswith("roof"), s.str.startswith("position")],
                  ["Construction material", "Ground slope", "Roof type", "Position"], default="")
    return np.where(g == "", nice_names(cols), g)


def load_model(path):
    """Booster plus the number of trees early stopping kept (best_iteration + 1)."""
    b = xgb.Booster(); b.load_model(path)
    best = b.attr("best_iteration")
    return b, (int(best) + 1 if best is not None else b.num_boosted_rounds())


def contribs(path, X):
    b, n = load_model(path)
    t0 = time.time()
    c = b.predict(xgb.DMatrix(X), pred_contribs=True, iteration_range=(0, n))
    print(f"  {Path(path).name}: {n} trees, {len(X)} buildings, {time.time() - t0:.0f}s")
    return c[:, :-1], c[:, -1], b, n                    # phi (n, p), phi_0 (n,)


def grouped_abs(phi, cols):
    g = groups(cols)
    G = pd.DataFrame(phi, columns=g).T.groupby(level=0).sum().T      # sum within group per building
    return G.abs().mean().sort_values()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-reg", type=int, default=N_REG)
    ap.add_argument("--n-ord", type=int, default=N_ORD)
    ap.add_argument("--redo", action="store_true", help="recompute SHAP values even if cached")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT / f"shap_values_{TAG}.npz"

    df = D.load_raw(DATA, MSI, MSI_COL)
    y = D.make_target(df, "5dg")
    split = D.load_or_make_split(y, ROOT / "data/processed/split.json")
    tr, te = np.array(split["train"]), np.array(split["test"])
    X = D.build_features(df, "full", use_msi=True)
    _, Xte = D.impute(X.iloc[tr], X.iloc[te])
    cols = list(Xte.columns)
    rng = np.random.default_rng(SEED)
    ir = np.sort(rng.choice(len(te), args.n_reg, replace=False))
    io = ir[:args.n_ord] if args.n_ord <= args.n_reg else np.sort(rng.choice(len(te), args.n_ord, replace=False))

    if cache.exists() and not args.redo:
        z = np.load(cache)
        phi_r, base_r, phi_o = z["phi_r"], z["base_r"], z["phi_o"]
        ir, io = z["ir"], z["io"]
        print(f"[shap] loaded cached values from {cache.name}")
    else:
        print("[shap] TreeSHAP ...")
        phi_r, base_r, b, n = contribs(XGB_DIR / f"models/XGBR_{TAG}.json", Xte.iloc[ir])
        # additivity check: phi_0 + sum phi = model output, and = the prediction saved by 03_xgboost.py
        out = b.predict(xgb.DMatrix(Xte.iloc[ir]), iteration_range=(0, n))
        saved = np.load(XGB_DIR / "test_predictions.npz")
        assert np.array_equal(saved["test_idx"], te), "test split differs from 03_xgboost.py"
        print(f"  additivity max |phi0+sum(phi) - f(x)| = {np.abs(base_r + phi_r.sum(1) - out).max():.2e}; "
              f"max |f(x) - saved XGBR pred| = {np.abs(out - saved[f'XGBR_{TAG}'][ir]).max():.2e}")
        phi_o = np.stack([contribs(XGB_DIR / f"models/XGBO_{TAG}_gt{k}.json", Xte.iloc[io])[0] for k in range(1, 5)])
        np.savez_compressed(cache, phi_r=phi_r.astype(np.float32), base_r=base_r.astype(np.float32),
                            phi_o=phi_o.astype(np.float32), ir=ir, io=io, cols=np.array(cols))
    Xr, Xo = Xte.iloc[ir], Xte.iloc[io]
    names = nice_names(cols)

    # tables
    per_input = pd.Series(np.abs(phi_r).mean(0), index=cols).sort_values()
    g_shap = grouped_abs(phi_r, cols)
    rf = pd.read_csv(RF_IMPORTANCE, index_col=0)["importance"]
    g_rf = rf.groupby(groups(rf.index)).sum()
    tab = pd.DataFrame({"xgbr_shap_mean_abs": g_shap, "xgbr_shap_share": g_shap / g_shap.sum(),
                        "rf_impurity_share": g_rf / g_rf.sum()}).sort_values("xgbr_shap_share", ascending=False)
    tab.to_csv(OUT / "grouped_importance_vs_rf.csv")
    per_input.rename("mean_abs_shap").to_csv(OUT / "per_input_importance_xgbr.csv")
    print("\nGrouped importance (share of total):\n", tab.round(3).to_string())

    # Fig S1: Fig 7 recreated with SHAP, next to the paper-replica RF
    fig, (a, b2) = plt.subplots(1, 2, figsize=(11.5, 6.5), gridspec_kw={"width_ratios": [1.3, 1]},
                                constrained_layout=True)
    a.barh(nice_names(per_input.index), per_input.values, color=ORANGE, height=0.65)
    a.set_xlabel("mean |SHAP| (damage grades)"); a.grid(axis="x", color=GRID, lw=0.6); a.set_axisbelow(True)
    a.set_title("(a) XGBR, per input (26 one-hot features)", loc="left", fontsize=9)
    t = tab.sort_values("xgbr_shap_share")
    yy = np.arange(len(t)); h = 0.38
    b2.barh(yy + h / 2, t.rf_impurity_share, height=h, color=BLUE, label="RF impurity (paper method)")
    b2.barh(yy - h / 2, t.xgbr_shap_share, height=h, color=ORANGE, label="XGBR mean |SHAP|")
    for i, (r_, s_) in enumerate(zip(t.rf_impurity_share, t.xgbr_shap_share)):
        b2.text(r_ + 0.005, i + h / 2, f"{r_:.0%}", va="center", fontsize=7, color=MUTED)
        b2.text(s_ + 0.005, i - h / 2, f"{s_:.0%}", va="center", fontsize=7, color=MUTED)
    b2.set_yticks(yy, t.index); b2.set_xlabel("share of total importance")
    b2.set_xlim(0, t[["rf_impurity_share", "xgbr_shap_share"]].values.max() * 1.25)
    b2.grid(axis="x", color=GRID, lw=0.6); b2.set_axisbelow(True); b2.legend(frameon=False, loc="lower right")
    b2.set_title("(b) Grouped by parameter: RF vs XGBoost", loc="left", fontsize=9)
    fig.suptitle(f"Fig S1. Feature importance, full features, 5 grades (SHAP on {len(ir):,} test buildings)",
                 x=0.01, ha="left", fontsize=10)
    fig.savefig(OUT / "figS1_importance_shap_vs_rf.png", dpi=200); plt.close(fig)

    # ---------- Fig S2: beeswarm (direction of each effect)
    import shap
    ex = shap.Explanation(values=phi_r, base_values=base_r, data=Xr.values, feature_names=list(names))
    plt.figure()
    shap.plots.beeswarm(ex, max_display=15, show=False, plot_size=(8, 6.5))
    plt.title("Fig S2. XGBR SHAP values per building (colour = input value)", loc="left", fontsize=10)
    plt.xlabel("SHAP value (damage grades; + = more damage)")
    plt.savefig(OUT / "figS2_beeswarm_xgbr.png", dpi=200, bbox_inches="tight"); plt.close("all")

    # ---------- Fig S3: dependence plots for the numeric inputs
    mud = Xr["has_superstructure_mud_mortar_stone"].values.astype(bool)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    for ax, c in zip(axes.flat, ["msi", "age_building", "plinth_area_sq_ft", "height_ft_pre_eq",
                                 "count_floors_pre_eq"]):
        j = cols.index(c); xv = Xr[c].values
        jit = rng.uniform(-0.25, 0.25, len(xv)) if c == "count_floors_pre_eq" else 0
        for m_, col, lab in ((mud, ORANGE, "mud-mortar stone"), (~mud, BLUE, "other material")):   # minority on top
            ax.scatter(xv[m_] + (jit[m_] if np.ndim(jit) else 0), phi_r[m_, j], s=3, alpha=0.25, color=col,
                       lw=0, label=lab, rasterized=True)
        ax.axhline(0, color=MUTED, lw=0.6)
        if c in ("age_building", "plinth_area_sq_ft"):
            ax.set_xlim(*np.percentile(xv, [0, 99]))
        ax.set_xlabel(nice_names([c])[0]); ax.set_ylabel(f"SHAP for {nice_names([c])[0]}")
        ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
    axes.flat[0].legend(frameon=False, markerscale=4, loc="upper left")
    # last panel: MSI effect shown as a function of location identity (mean SHAP per MSI value)
    ax = axes.flat[5]; j = cols.index("msi")
    per_vdc = pd.DataFrame({"msi": Xr["msi"].values, "phi": phi_r[:, j]}).groupby("msi").phi.mean()
    (pd.DataFrame({"msi": Xr["msi"].values, "phi": phi_r[:, j]}).groupby("msi").phi.agg(["mean", "size"])
       .rename(columns={"mean": "mean_shap_msi", "size": "n_buildings"})
       .to_csv(OUT / "msi_effect_per_vdc.csv"))      # one row per old VDC (MSI value); used by the report
    ax.plot(per_vdc.index, per_vdc.values, color=BLUE, lw=1)
    ax.axhline(0, color=MUTED, lw=0.6); ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.set_xlabel("MSI"); ax.set_ylabel("mean SHAP for MSI at each MSI value")
    ax.set_title("MSI effect is jagged, not monotone:\nMSI also acts as a location ID", loc="left", fontsize=8)
    fig.suptitle("Fig S3. XGBR dependence plots: SHAP of an input vs its value (orange = mud-mortar stone)",
                 x=0.01, ha="left", fontsize=10)
    fig.savefig(OUT / "figS3_dependence_xgbr.png", dpi=200); plt.close(fig)

    # ---------- Fig S4: ordinal model, which inputs drive which damage transition
    g_ord = pd.DataFrame({f"DG>{k + 1}": grouped_abs(phi_o[k], cols) for k in range(4)})
    g_ord = g_ord.loc[g_ord.sum(1).sort_values(ascending=False).index]
    g_ord.to_csv(OUT / "grouped_importance_ordinal_thresholds.csv")
    share = g_ord / g_ord.sum(0)
    fig, (a, b2) = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    im = a.imshow(g_ord.values, cmap="Blues", aspect="auto")
    for i in range(g_ord.shape[0]):
        for k in range(4):
            v = g_ord.values[i, k]
            a.text(k, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                   color="white" if v > 0.6 * g_ord.values.max() else INK)
    a.set_xticks(range(4), g_ord.columns); a.set_yticks(range(len(g_ord)), g_ord.index)
    a.set_title("(a) mean |SHAP| (log-odds of exceeding grade k)", loc="left", fontsize=9)
    fig.colorbar(im, ax=a, shrink=0.8)
    for s in a.spines.values():
        s.set_visible(False)
    for i, gname in enumerate(share.index[:6]):
        b2.plot(range(4), share.loc[gname].values, marker="o", ms=4, lw=1.5,
                color=["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"][i], label=gname)
    b2.set_xticks(range(4), share.columns); b2.set_ylabel("share of importance at this threshold")
    b2.grid(color=GRID, lw=0.6); b2.set_axisbelow(True); b2.legend(frameon=False, fontsize=7)
    b2.set_title("(b) Share per threshold (top 6 parameters)", loc="left", fontsize=9)
    fig.suptitle(f"Fig S4. XGBO (Frank & Hall) SHAP per damage threshold ({len(io):,} test buildings)",
                 x=0.01, ha="left", fontsize=10)
    fig.savefig(OUT / "figS4_ordinal_thresholds.png", dpi=200); plt.close(fig)
    print("\nOrdinal, share per threshold:\n", share.round(3).to_string())
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()

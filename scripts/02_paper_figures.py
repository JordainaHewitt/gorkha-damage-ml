"""Recreate the paper's descriptive and RFR figures with our data.

  Fig 1  share of DG5 buildings per district (map)
  Fig 2  distribution of building parameters (+ table vs. paper Table 1)
  Fig 3  (a) ShakeMap MMI field, (b) distribution of building MSI
  Fig 4  RFR on the validation set: (a) error distribution, (b) confusion matrix
  Fig 7  RFR feature importance (impurity), per input and grouped
  Fig 8  RFR test confusion matrices, 5 grades, full vs basic   (from results/rf, 1000 trees)
  Fig 9  RFR test confusion matrices, traffic light, full vs basic

Press Run, or: python3 scripts/02_paper_figures.py      Output: results/paper_figures/
"""
import argparse
import json
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from sklearn.ensemble import RandomForestRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import data as D                 # noqa: E402
from src.metrics import row_normalize     # noqa: E402

# SETTINGS
DATA = ROOT / "data/raw/npc/eq2015.csv"
MSI = ROOT / "data/processed/msi_buildings.csv"
RF_RESULTS = ROOT / "results/rf"           # 1000-tree run, for Figs 8-9
TREES_FIG = 200                            # model refit for Figs 4 and 7
OUT = ROOT / "results/paper_figures"
FIGS = (1, 2, 3, 4, 7, 8, 9)               # which figures to make; (1,) = only Fig 1 (no RF refit)
# Fig 1: districts too small for their name get an outside label, offset (lon, lat) in degrees
LABEL_OFFSETS = {"Rasuwa": (0.05, 0.62), "Sindhupalchok": (0.85, 0.72), "Dolakha": (0.95, 0.15),
                 "Kavrepalanchok": (-0.25, -0.62)}


BLUE, INK, MUTED, GRID = "#2a78d6", "#1f1f1e", "#6b6a65", "#e4e3df"
CBS = {12: "Okhaldhunga", 20: "Sindhuli", 21: "Ramechhap", 22: "Dolakha", 23: "Sindhupalchok",
       24: "Kavrepalanchok", 28: "Nuwakot", 29: "Rasuwa", 30: "Dhading", 31: "Makwanpur", 36: "Gorkha"}
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
OUT.mkdir(parents=True, exist_ok=True)


def bars(ax, labels, values, title):
    ax.bar(labels, values, color=BLUE, width=0.6)
    ax.set_title(title, fontsize=9, color=INK, loc="left")
    ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    for i, v in enumerate(values):
        ax.text(i, v + 1, f"{v:.1f}", ha="center", fontsize=7, color=MUTED)
    ax.set_ylim(0, max(values) * 1.18 + 2)


def confusion(ax, cm, title, ticks):
    m = row_normalize(cm)
    ax.imshow(m, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(m)):
        for j in range(len(m)):
            ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if m[i, j] > 0.5 else INK)
    ax.set_xticks(range(len(m)), ticks); ax.set_yticks(range(len(m)), ticks)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title, fontsize=9, color=INK)
    for s in ax.spines.values():
        s.set_visible(False)


def fig1(df):
    """Fig 1: share of DG5 buildings per district (choropleth).

    Labels sit at each district's representative point, with a white halo so they read on any fill.
    Districts too small for their name (LABEL_OFFSETS) get the label outside, joined by a leader line.
    """
    import matplotlib.patheffects as pe
    halo = [pe.withStroke(linewidth=2.2, foreground="white")]
    dist = gpd.read_file(ROOT / "data/raw/boundaries/cod_ab/npl_admin2.geojson", engine="pyogrio",
                         columns=["adm2_name"])
    dg5 = (df.assign(dg5=df.damage_grade == 5).groupby("district").dg5.mean() * 100).rename("dg5")
    dist = dist.merge(dg5, left_on="adm2_name", right_index=True, how="left")
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    dist.plot(color="#f1f0ec", edgecolor="white", lw=0.4, ax=ax)
    sub = dist.dropna(subset=["dg5"])
    sub.plot(column="dg5", cmap="Blues", vmin=0, vmax=100, edgecolor="white", lw=0.8, ax=ax, legend=True,
             legend_kwds={"label": "Buildings tagged DG5 (%)", "shrink": 0.6})
    for _, r in sub.iterrows():
        p = r.geometry.representative_point()
        dx, dy = LABEL_OFFSETS.get(r.adm2_name, (0, 0))
        kw = dict(ha="center", va="center", fontsize=7, color=INK, path_effects=halo)
        if dx or dy:   # outside the district, with a leader line to its interior point
            ax.annotate(f"{r.adm2_name} {r.dg5:.0f}%", xy=(p.x, p.y), xytext=(p.x + dx, p.y + dy),
                        arrowprops=dict(arrowstyle="-", color=INK, lw=0.6, shrinkA=0, shrinkB=0), **kw)
            ax.plot(p.x, p.y, "o", ms=2.5, color=INK)
        else:
            ax.annotate(f"{r.adm2_name}\n{r.dg5:.0f}%", (p.x, p.y), **kw)
    b = sub.total_bounds; ax.set_xlim(b[0] - 0.4, b[2] + 1.0); ax.set_ylim(b[1] - 0.3, b[3] + 0.55)
    ax.set_axis_off()
    ax.set_title("Fig 1. Share of buildings tagged DG5 (collapse), 11 surveyed districts", loc="left", fontsize=10)
    fig.savefig(OUT / "fig01_dg5_by_district.png", dpi=200, bbox_inches="tight"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--figs", type=int, nargs="*", default=list(FIGS), help="e.g. --figs 1  (only Fig 1)")
    figs = set(ap.parse_args().figs)
    df = D.load_raw(DATA, MSI, "msi_vdc")
    df["district"] = df.district_id.map(CBS)
    pct = lambda s: (s.value_counts(normalize=True) * 100)

    if 1 in figs:
        fig1(df)

    if 2 in figs:
        # ---- Fig 2: parameter distributions (+ comparison table with paper Table 1)
        cut = lambda s, b, l: pd.cut(s, b, labels=l, right=True, include_lowest=True)
        mats = [c for c in df.columns if c.startswith("has_superstructure_")]
        panels = {
            "(a) Damage grade": pct(df.damage_grade).sort_index().rename(lambda g: f"DG{g}"),
            "(b) Stories": pct(cut(df.count_floors_pre_eq, [0, 3, 5, 99], ["NS1", "NS2", "NS3"])).sort_index(),
            "(c) Age (yr)": pct(cut(df.age_building, [0, 10, 50, 200], ["Ag1", "Ag2", "Ag3"])).sort_index(),
            "(d) Plinth area (sq ft)": pct(cut(df.plinth_area_sq_ft, [0, 500, 1000, 1e6], ["Ar1", "Ar2", "Ar3"])).sort_index(),
            "(e) Height (ft)": pct(cut(df.height_ft_pre_eq, [0, 10, 20, 1e3], ["Ht1", "Ht2", "Ht3"])).sort_index(),
            "(f) Ground slope": pct(df.land_surface_condition).reindex(["Flat", "Moderate slope", "Steep slope"]).set_axis(["FS", "MS", "SS"]),
            "(g) Roof": pct(df.roof_type).reindex(["Bamboo/Timber-Heavy roof", "Bamboo/Timber-Light roof", "RCC/RB/RBC"]).set_axis(["RT1", "RT2", "RT3"]),
            "(h) Position": pct(df.position).reindex(["Not attached", "Attached-1 side", "Attached-2 side", "Attached-3 side"]).set_axis(["NA", "A1", "A2", "A3"]),
            "(i) Superstructure material": pd.Series(df[mats].mean().values * 100, index=[f"CM{i}" for i in range(1, 12)]),
        }
        fig, axes = plt.subplots(3, 3, figsize=(11, 8), constrained_layout=True)
        for ax, (t, s) in zip(axes.flat, panels.items()):
            bars(ax, list(s.index.astype(str)), s.values, t)
            ax.set_ylabel("%")
        fig.suptitle("Fig 2. Distribution of building parameters (n = 762,094)", x=0.01, ha="left", fontsize=10)
        fig.savefig(OUT / "fig02_parameter_distributions.png", dpi=200); plt.close(fig)
        paper_t1 = [10.34, 11.45, 17.90, 24.12, 36.19, 98.9, 1.04, 0.06, 32.03, 62.92, 5.05, 79.39, 18.54, 2.07,
                    13.19, 70.93, 15.88, 82.89, 13.87, 3.24, 28.05, 66.10, 5.85, 79.32, 16.98, 3.53, 0.17,
                    4.24, 80.02, 3.51, 1.58, 2.29, 7.15, 25.87, 8.04, 3.97, 1.63, 1.20]
        t1 = pd.concat({k: v for k, v in panels.items()}).rename("ours_pct").round(2).to_frame()
        t1["paper_pct"] = paper_t1
        t1["diff"] = (t1.ours_pct - t1.paper_pct).round(2)
        t1.to_csv(OUT / "table1_vs_paper.csv")

    if 3 in figs:
        # ---- Fig 3: ShakeMap field and MSI distribution
        dist = gpd.read_file(ROOT / "data/raw/boundaries/cod_ab/npl_admin2.geojson", engine="pyogrio",
                             columns=["adm2_name"])
        dist["dg5"] = dist.adm2_name.map(df.assign(dg5=df.damage_grade == 5).groupby("district").dg5.mean())
        sub = dist.dropna(subset=["dg5"])
        with rasterio.open(ROOT / "data/raw/shakemap/mmi_mean.flt") as r:
            mmi = r.read(1).astype(float); mmi[mmi == r.nodata] = np.nan
            ext = [r.bounds.left, r.bounds.right, r.bounds.bottom, r.bounds.top]
        fig, (a, b2) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1.5, 1]}, constrained_layout=True)
        im = a.imshow(mmi, extent=ext, cmap="YlOrRd", vmin=4, vmax=9)
        dist.boundary.plot(ax=a, color="0.35", lw=0.3)
        sub.boundary.plot(ax=a, color="k", lw=0.9)
        a.plot(84.731, 28.231, marker="*", color="#2a78d6", ms=13, mec="white")
        a.set_xlim(82.5, 88.5); a.set_ylim(26.3, 29.6); a.set_xlabel("Longitude"); a.set_ylabel("Latitude")
        fig.colorbar(im, ax=a, shrink=0.8, label="MMI (ShakeMap mean)")
        a.set_title("(a) ShakeMap intensity; bold = surveyed districts, star = epicentre", loc="left", fontsize=9)
        b2.hist(df.msi, bins=np.arange(6.0, 8.6, 0.1), color=BLUE, edgecolor="white", lw=0.5)
        b2.set_xlabel("MSI assigned to building (old-VDC mean MMI)"); b2.set_ylabel("Buildings")
        b2.grid(axis="y", color=GRID, lw=0.6); b2.set_axisbelow(True)
        b2.set_title(f"(b) MSI in the database: {df.msi.min():.2f}-{df.msi.max():.2f} (paper: 5.30-8.30)", loc="left", fontsize=9)
        fig.suptitle("Fig 3. Ground-motion input", x=0.01, ha="left", fontsize=10)
        fig.savefig(OUT / "fig03_msi.png", dpi=200); plt.close(fig)

    if 4 in figs or 7 in figs:
        # ---- Figs 4 and 7: refit RFR (full features, 5 grades)
        y = D.make_target(df, "5dg")
        split = D.load_or_make_split(y, ROOT / "data/processed/split.json")
        tr, va = np.array(split["train"]), np.array(split["val"])
        X = D.build_features(df, "full", use_msi=True)
        Xtr, Xva = D.impute(X.iloc[tr], X.iloc[va])
        print(f"[fig] fitting RFR with {TREES_FIG} trees for Figs 4 and 7 ...")
        m = RandomForestRegressor(n_estimators=TREES_FIG, max_depth=20, n_jobs=-1, random_state=0).fit(Xtr, y[tr])
        pred = m.predict(Xva); err = pred - y[va]
        lab = np.clip(np.rint(pred), 1, 5).astype(int)
        cm = pd.crosstab(pd.Categorical(y[va], range(1, 6)), pd.Categorical(lab, range(1, 6)), dropna=False).values
        fig, (a, b2) = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        a.hist(err, bins=np.arange(-4, 4.01, 0.2), color=BLUE, edgecolor="white", lw=0.5)
        a.axvline(0, color=INK, lw=0.8)
        a.set_xlabel("Predicted - true damage grade (unrounded)"); a.set_ylabel("Buildings")
        a.grid(axis="y", color=GRID, lw=0.6); a.set_axisbelow(True)
        a.set_title(f"(a) Error distribution, MAE = {np.abs(err).mean():.2f} (paper: 0.66)", loc="left", fontsize=9)
        confusion(b2, cm, "(b) Normalised confusion matrix", [f"DG{i}" for i in range(1, 6)])
        fig.suptitle("Fig 4. RFR on the validation set", x=0.01, ha="left", fontsize=10)
        fig.savefig(OUT / "fig04_rfr_validation.png", dpi=200); plt.close(fig)

        imp = pd.Series(m.feature_importances_, index=X.columns).sort_values()
        nice = (imp.index.str.replace("has_superstructure_", "material: ").str.replace("land_surface_condition_", "slope: ")
                .str.replace("roof_type_", "roof: ").str.replace("position_", "position: ").str.replace("_", " "))
        group = np.select([imp.index.str.startswith("has_superstructure"), imp.index.str.startswith("land_surface"),
                           imp.index.str.startswith("roof"), imp.index.str.startswith("position")],
                          ["Construction material", "Ground slope", "Roof type", "Position"], default="")
        group = np.where(group == "", nice.str.replace("count floors pre eq", "Stories").str.replace("age building", "Age")
                         .str.replace("plinth area sq ft", "Plinth area").str.replace("height ft pre eq", "Height")
                         .str.replace("msi", "MSI"), group)
        grouped = imp.groupby(group).sum().sort_values()
        fig, (a, b2) = plt.subplots(1, 2, figsize=(11, 6.5), gridspec_kw={"width_ratios": [1.4, 1]}, constrained_layout=True)
        a.barh(nice, imp.values, color=BLUE, height=0.65)
        a.set_xlabel("Importance (mean decrease in impurity)"); a.grid(axis="x", color=GRID, lw=0.6); a.set_axisbelow(True)
        a.set_title("(a) Per input (26 one-hot features)", loc="left", fontsize=9)
        b2.barh(grouped.index, grouped.values, color=BLUE, height=0.6)
        for i, v in enumerate(grouped.values):
            b2.text(v + 0.005, i, f"{v:.0%}", va="center", fontsize=8, color=MUTED)
        b2.set_xlim(0, grouped.max() * 1.2); b2.grid(axis="x", color=GRID, lw=0.6); b2.set_axisbelow(True)
        b2.set_title("(b) Grouped by parameter", loc="left", fontsize=9)
        fig.suptitle("Fig 7. RFR feature importance (paper: mud-mortar stone 32%, MSI 31%, area 8%, age 7%, height 5%)",
                     x=0.01, ha="left", fontsize=10)
        fig.savefig(OUT / "fig07_feature_importance.png", dpi=200); plt.close(fig)
        imp.rename("importance").to_csv(OUT / "fig07_importance.csv")

    if 8 in figs or 9 in figs:
        # ---- Figs 8-9: test confusion matrices from the final RF run
        cms = json.loads((RF_RESULTS / "confusions.json").read_text())
        for scale, fname, ticks, title in (("5dg", "fig08_test_5dg.png", [f"DG{i}" for i in range(1, 6)], "Fig 8. RFR test set, 5 damage grades"),
                                           ("tl", "fig09_test_tl.png", ["S", "M", "H"], "Fig 9. RFR test set, traffic light")):
            fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
            for ax, setting in zip(axes, ("full", "basic")):
                confusion(ax, cms[f"RFR {setting} {scale}"], f"({'a' if setting == 'full' else 'b'}) {setting}-feature setting", ticks)
            fig.suptitle(title, x=0.01, ha="left", fontsize=10)
            fig.savefig(OUT / fname, dpi=200); plt.close(fig)
    print(f"Saved figures to {OUT}")


if __name__ == "__main__":
    main()

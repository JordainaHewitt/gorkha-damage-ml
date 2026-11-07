"""Stage 2 (prep): mean ShakeMap intensity (MMI) per administrative unit.

We have polygons for two candidate unit systems, so we compute both:
  A. new municipalities (COD-AB admin level 3, 2017+)  -> primary
  B. old VDCs (MoFALD/HRRP 2016, pre-2017)              -> fallback
The survey CSV decides which one we join (see scripts/check_survey.py).

Outputs
  data/processed/msi_new_municipality.csv
  data/processed/msi_old_vdc.csv
  results/msi/msi_maps.png
Run from the repo root:  python scripts/00_msi.py
"""
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import openpyxl
import pandas as pd
import rasterio
from rasterstats import zonal_stats

ROOT = Path(__file__).resolve().parents[1]
RAW, PROC, RES = ROOT / "data/raw", ROOT / "data/processed", ROOT / "results/msi"
PROC.mkdir(parents=True, exist_ok=True); RES.mkdir(parents=True, exist_ok=True)

# The 11 districts of the NPC building survey (paper Fig. 1)
SURVEY_DISTRICTS = ["Gorkha", "Dhading", "Rasuwa", "Nuwakot", "Sindhupalchok", "Dolakha",
                    "Ramechhap", "Okhaldhunga", "Kavrepalanchok", "Makwanpur", "Sindhuli"]


def read_grid(name):
    """ShakeMap ESRI .flt grid -> (array with NaN for nodata, affine). The .hdr has no CRS; it is WGS84 lon/lat."""
    with rasterio.open(RAW / "shakemap" / f"{name}.flt") as r:
        a = r.read(1).astype(float)
        a[a == r.nodata] = np.nan
        return a, r.transform


def unit_stats(gdf, mmi, mmi_sd, transform):
    """Mean (and min/max) MMI per polygon. Units smaller than a grid cell (~1.8 km)
    can contain no cell centre; for those, fall back to all_touched=True."""
    kw = dict(affine=transform, nodata=np.nan)
    s = zonal_stats(gdf.geometry, mmi, stats=["mean", "min", "max", "count"], **kw)
    s_touch = zonal_stats(gdf.geometry, mmi, stats=["mean", "count"], all_touched=True, **kw)
    sd = zonal_stats(gdf.geometry, mmi_sd, stats=["mean"], all_touched=True, **kw)
    out = pd.DataFrame(s)
    small = out["count"].fillna(0) == 0
    out.loc[small, "mean"] = [t["mean"] for t, m in zip(s_touch, small) if m]
    out["all_touched_fallback"] = small
    out["mmi_sd"] = [d["mean"] for d in sd]
    return out.rename(columns={"mean": "msi", "min": "msi_min", "max": "msi_max", "count": "n_cells"})


def read_crosswalk():
    """HRRP sheet 'Conversion_FINAL': old HLCIT VDC code -> HRRP_VCODE (the shapefile key)."""
    ws = openpyxl.load_workbook(RAW / "boundaries/mofald/160922_31dist_spatialcode_conversion_v2.xlsx",
                                read_only=True)["Conversion_FINAL"]
    rows = []
    for i, r in enumerate(ws.iter_rows(values_only=True)):
        r = r[:8]
        if i == 0:
            header = r; continue
        if all(v is None for v in r):
            break  # sheet reports ~1M rows of empty formatting; stop at the first blank row
        rows.append(r)
    return pd.DataFrame(rows, columns=header)


def main():
    mmi, tf = read_grid("mmi_mean")
    mmi_sd, _ = read_grid("mmi_std")
    print(f"[grid] MMI range over grid: {np.nanmin(mmi):.2f}-{np.nanmax(mmi):.2f}")

    # A. new municipalities (COD-AB adm3)
    new = gpd.read_file(RAW / "boundaries/cod_ab/npl_admin3.geojson", engine="pyogrio",
                        columns=["adm3_name", "adm3_pcode", "adm2_name", "adm2_pcode", "area_sqkm"])
    new = new[new.adm2_name.isin(SURVEY_DISTRICTS)].reset_index(drop=True)
    new = pd.concat([new, unit_stats(new, mmi, mmi_sd, tf)], axis=1)
    new.drop(columns="geometry").to_csv(PROC / "msi_new_municipality.csv", index=False)

    # B. old VDCs (MoFALD / HRRP)
    old = gpd.read_file(RAW / "boundaries/mofald/31adist_polbnda_adm4_mofald_hrrp_wgs84/"
                        "31Adist_polbnda_adm4_MoFALD_HRRP_wgs84.shp", engine="pyogrio",
                        columns=["HRRP_DNAME", "HRRP_VNAME", "HRRP_VCODE", "Sq_km"])
    old = old[old.HRRP_DNAME.isin(SURVEY_DISTRICTS)].reset_index(drop=True)
    old = pd.concat([old, unit_stats(old, mmi, mmi_sd, tf)], axis=1)
    old.drop(columns="geometry").to_csv(PROC / "msi_old_vdc.csv", index=False)  # one row per polygon
    # Some 2016 polygons are municipalities formed by merging several older VDCs, so one
    # HRRP_VCODE maps to several HLCIT codes. Keep that one-to-many table separately:
    # it is what we need if the survey turns out to use the older HLCIT VDC codes.
    xw = read_crosswalk()[["HRRP_VCODE", "HLCIT_CODE", "VDC_NAME", "OCHA_VNAME"]]
    xw = xw.merge(old[["HRRP_VCODE", "HRRP_DNAME", "HRRP_VNAME", "msi"]], on="HRRP_VCODE", how="inner")
    xw.to_csv(PROC / "msi_old_vdc_crosswalk.csv", index=False)

    for label, g, dcol in (("new municipalities", new, "adm2_name"), ("old VDCs", old, "HRRP_DNAME")):
        missing = sorted(set(SURVEY_DISTRICTS) - set(g[dcol]))
        print(f"\n[{label}] {len(g)} units in {g[dcol].nunique()} districts"
              f"{' | MISSING: ' + str(missing) if missing else ''}")
        print(f"  MSI {g.msi.min():.2f}-{g.msi.max():.2f}  (paper, per ward: 5.30-8.30)")
        print(f"  units needing all_touched fallback: {int(g.all_touched_fallback.sum())}; "
              f"no MSI at all: {int(g.msi.isna().sum())}")
    print(f"  crosswalk: {len(xw)} older HLCIT VDC codes -> {xw.HRRP_VCODE.nunique()} polygons")

    # Sanity-check figure: both unit systems on the same colour scale
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    for ax, (g, title) in zip(axes, ((new, f"New municipalities (n={len(new)})"),
                                     (old, f"Old VDCs (n={len(old)})"))):
        g.plot(column="msi", ax=ax, cmap="YlOrRd", vmin=5, vmax=9, edgecolor="0.4", linewidth=0.2,
               missing_kwds={"color": "lightgrey"})
        g.dissolve(by=g.columns[2] if "adm2_name" in g else "HRRP_DNAME").boundary.plot(ax=ax, color="k", lw=0.8)
        ax.plot(84.731, 28.231, marker="*", color="blue", ms=14)  # USGS epicentre
        ax.set_title(title); ax.set_axis_off()
    sm = plt.cm.ScalarMappable(cmap="YlOrRd", norm=plt.Normalize(5, 9))
    fig.colorbar(sm, ax=axes, shrink=0.7, label="Mean MMI (ShakeMap)")
    fig.suptitle("Mean shaking intensity per unit, 11 survey districts (star = epicentre)")
    fig.savefig(RES / "msi_maps.png", dpi=150)
    print(f"\nSaved {PROC/'msi_new_municipality.csv'}, {PROC/'msi_old_vdc.csv'}, {RES/'msi_maps.png'}")


if __name__ == "__main__":
    main()

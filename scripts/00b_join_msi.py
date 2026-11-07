"""Attach shaking intensity (MSI) to every surveyed building.

How a building is located

building_id has the form  DD VV WW SSSSSS  (e.g. 12 01 03 000011):
  DD = district (CBS code, = district_id), VV = pre-2017 VDC index within the district,
  WW = old ward, SSSSSS = serial number.

Output: data/processed/msi_buildings.csv with one row per building:
  msi_vdc   mean MMI over the building's old VDC polygon (577 polygons; finest available,
            comparable to the paper's 949 wards)
  msi_muni  building-weighted mean of msi_vdc within the building's vdcmun_id (110 units),
            used to test how much spatial resolution matters
Run from repo root after scripts/00_msi.py:  python scripts/00b_join_msi.py
"""
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW, PROC = ROOT / "data/raw", ROOT / "data/processed"
CBS = {12: "Okhaldhunga", 20: "Sindhuli", 21: "Ramechhap", 22: "Dolakha", 23: "Sindhupalchok",
       24: "Kavrepalanchok", 28: "Nuwakot", 29: "Rasuwa", 30: "Dhading", 31: "Makwanpur", 36: "Gorkha"}


def read_vdc_table():
    """HRRP sheet 'Conversion_v1': one row per pre-2017 VDC with OCHA P-code and HRRP polygon code."""
    ws = openpyxl.load_workbook(RAW / "boundaries/mofald/160922_31dist_spatialcode_conversion_v2.xlsx",
                                read_only=True)["Conversion_v1"]
    rows = list(ws.iter_rows(values_only=True, max_row=2000))
    t = pd.DataFrame(rows[1:], columns=rows[0]).dropna(subset=["OCHA_PCODE"])
    t = t[t.DIST_NAME.isin(CBS.values())].copy()
    t["vv"] = t.groupby("DIST_NAME")["OCHA_PCODE"].rank(method="first").astype(int)
    return t[["DIST_NAME", "vv", "VDC_NAME", "OCHA_VNAME", "OCHA_PCODE", "HRRP_VCODE"]]


def main():
    s = pd.read_csv(next((RAW / "npc").glob("*.csv")), usecols=["building_id", "district_id", "vdcmun_id"],
                    dtype={"building_id": str})
    s["DIST_NAME"] = s.district_id.map(CBS)
    s["vv"] = s.building_id.str[2:4].astype(int)

    vdc = read_vdc_table()
    per_d = s.groupby("DIST_NAME").vv.nunique().to_dict()
    tab_d = vdc.groupby("DIST_NAME").size().to_dict()
    bad = {d: (per_d[d], tab_d.get(d)) for d in per_d if per_d[d] != tab_d.get(d)}
    print(f"[check] VDC count per district, survey vs table: {'all 11 equal' if not bad else bad}")

    msi = pd.read_csv(PROC / "msi_old_vdc.csv")[["HRRP_VCODE", "msi"]]
    vdc = vdc.merge(msi, on="HRRP_VCODE", how="left")
    s = s.merge(vdc[["DIST_NAME", "vv", "OCHA_PCODE", "OCHA_VNAME", "msi"]], on=["DIST_NAME", "vv"], how="left")
    s = s.rename(columns={"msi": "msi_vdc"})
    s["msi_muni"] = s.groupby("vdcmun_id").msi_vdc.transform("mean")

    print(f"[join] {len(s):,} buildings; without msi_vdc: {s.msi_vdc.isna().sum():,}")
    print(f"[join] msi_vdc range {s.msi_vdc.min():.2f}-{s.msi_vdc.max():.2f} over {s.OCHA_PCODE.nunique()} VDCs; "
          f"msi_muni range {s.msi_muni.min():.2f}-{s.msi_muni.max():.2f} over {s.vdcmun_id.nunique()} municipalities")
    s[["building_id", "OCHA_PCODE", "OCHA_VNAME", "msi_vdc", "msi_muni"]].to_csv(PROC / "msi_buildings.csv", index=False)
    print(f"Saved {PROC / 'msi_buildings.csv'}")


if __name__ == "__main__":
    main()

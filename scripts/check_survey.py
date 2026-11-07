"""Run this FIRST when the NPC survey CSVs arrive. It changes nothing; it only reports.

  python scripts/check_survey.py                      # looks in data/raw/npc/
  python scripts/check_survey.py --dir some/folder

Answers five questions before any model is trained:
  1. Are all columns that src/config.py expects present?
  2. Does the damage-grade distribution look like the paper's?
  3. Do the two files agree on damage grade for the same building?
  4. Are the location IDs NEW federal units or OLD VDCs? (decides which MSI table to join)
  5. Can the IDs be matched to MSI polygons by name?
"""
import argparse
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import config as C  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default=str(ROOT / "data/raw/npc"))
a = ap.parse_args()
d = Path(a.dir)


def find(pattern):
    hits = sorted(d.rglob(pattern))
    return hits[0] if hits else None


def norm(s):
    """Lower-case, strip accents, drop suffixes like 'N.P.'/'Rural Municipality', remove non-letters."""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"\b(n\.?\s?p\.?|nagarpalika|gaunpalika|rural municipality|municipality|"
               r"sub-?metropolitan( city)?|metropolitan( city)?|vdc)\b", "", s)
    return re.sub(r"[^a-z]", "", s)


print("Files found:"); [print("  ", p.relative_to(d), f"{p.stat().st_size/1e6:.1f} MB") for p in sorted(d.rglob("*.csv"))]
fs, fd = find("*building_structure*.csv") or find("eq2015*.csv"), find("*damage_assessment*.csv")
if fs is None:
    sys.exit("\nNo *building_structure*.csv found. Put it in data/raw/npc/ and rerun.")

s = pd.read_csv(fs, low_memory=False)
from src.data import _expand_superstructure  # noqa: E402
s = _expand_superstructure(s)
print(f"\n== 1. Columns: {fs.name}  {s.shape[0]:,} rows x {s.shape[1]} cols ==")
print(s.columns.tolist())
expected = C.NUMERIC + C.CATEGORICAL + C.MATERIALS + [C.TARGET] + C.LOCATION_KEYS + ["building_id"]
missing = [c for c in expected if c not in s.columns]
print("All expected columns present." if not missing else f"MISSING (fix names in src/config.py): {missing}")
for c in C.CATEGORICAL:
    if c in s:
        print(f"  {c}: {s[c].value_counts(dropna=False).to_dict()}")
leaky = [c for c in s.columns if re.search(r"post_eq|technical_solution|condition_post", c)]
print(f"  Columns that must NOT be model inputs (measured after the quake): {leaky}")

print("\n== 2. Damage grade ==")
g = s[C.TARGET].astype(str).str.extract(r"(\d)", expand=False)
print((g.value_counts(normalize=True, dropna=False).sort_index() * 100).round(1).to_string())
print(f"  rows: {len(s):,} (paper: 762,106 buildings, 11 districts)")

if fd is not None:
    print(f"\n== 3. {fd.name} vs structure file ==")
    da = pd.read_csv(fd, low_memory=False)
    print(f"  {da.shape[0]:,} rows x {da.shape[1]} cols")
    if {"building_id", C.TARGET} <= set(da.columns):
        m = s[["building_id", C.TARGET]].merge(da[["building_id", C.TARGET]], on="building_id", suffixes=("_s", "_d"))
        agree = (m[C.TARGET + "_s"].astype(str) == m[C.TARGET + "_d"].astype(str)).mean()
        print(f"  {len(m):,} buildings matched on building_id; damage_grade agrees for {agree:.2%}")
    for c in ("area_assesed", "area_assessed"):
        if c in da:
            print(f"  {c}: {da[c].value_counts(dropna=False).to_dict()}")
    geo = [c for c in da.columns if c.startswith("has_geotechnical_risk")]
    print(f"  geotechnical-risk flags (optional ablation): {geo}")
else:
    print("\n== 3. damage assessment file not found; skipped ==")

print("\n== 4. Which administrative system do the IDs use? ==")
n = {c: s[c].nunique() for c in C.LOCATION_KEYS if c in s}
print(f"  unique values: {n}")
print("  reference counts in the 11 districts:  new federal -> 114 municipalities / 949 wards (paper)"
      "\n                                         old system  -> 577 VDC polygons (628 pre-merger VDCs)")
v = n.get("vdcmun_id", 0)
system = "new" if 90 <= v <= 140 else "old" if 450 <= v <= 700 else "unclear"
print(f"  -> vdcmun_id looks like the {system.upper()} system")
print(f"  sample IDs: {sorted(s['vdcmun_id'].dropna().unique())[:10] if 'vdcmun_id' in s else 'n/a'}")

print("\n== 5. Name matching to MSI polygons ==")
name_cols = [c for c in s.columns if "name" in c.lower()]
lookup = find("*mapping*.csv") or find("*name*.csv")
if lookup is not None:
    lk = pd.read_csv(lookup); print(f"  lookup file {lookup.name}: {lk.columns.tolist()}"); print(lk.head().to_string())
    name_cols = [c for c in lk.columns if "name" in c.lower()]; src_names = lk
elif name_cols:
    print(f"  name columns inside structure file: {name_cols}"); src_names = s
else:
    src_names = None
    print("  No names found. We will need a separate ID->name lookup (portal docs or HRRP).")
if src_names is not None:
    msi_new = pd.read_csv(ROOT / "data/processed/msi_new_municipality.csv")
    msi_old = pd.read_csv(ROOT / "data/processed/msi_old_vdc_crosswalk.csv")
    targets = {"new": set(msi_new.adm3_name.map(norm)),
               "old": set(msi_old.VDC_NAME.map(norm)) | set(msi_old.OCHA_VNAME.map(norm)) | set(msi_old.HRRP_VNAME.map(norm))}
    for c in name_cols:
        vals = src_names[c].dropna().unique()
        for k, t in targets.items():
            rate = pd.Series([norm(x) in t for x in vals]).mean()
            print(f"  {c!r} vs {k} names: {rate:.0%} of {len(vals)} unique names match exactly after normalising")
print("\nDone. Paste this whole output into the chat.")

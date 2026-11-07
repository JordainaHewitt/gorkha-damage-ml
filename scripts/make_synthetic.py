"""Generate a small FAKE dataset with the same columns as the NPC survey.

Only for testing that the pipeline runs end to end.The numbers it produces
mean nothing.

  python scripts/make_synthetic.py --n 20000 --out data/synthetic
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=20000)
ap.add_argument("--out", default="data/synthetic")
a = ap.parse_args()
rng = np.random.default_rng(0)
n, out = a.n, Path(a.out)
out.mkdir(parents=True, exist_ok=True)

n_vdc = 120
vdc = rng.integers(0, n_vdc, n)
vdc_msi = rng.uniform(5.3, 8.3, n_vdc)
df = pd.DataFrame({
    "building_id": np.arange(n),
    "district_id": vdc // 12,
    "vdcmun_id": vdc + 1000,
    "ward_id": (vdc + 1000) * 100 + rng.integers(1, 10, n),
    "count_floors_pre_eq": rng.choice([1, 2, 3, 4], n, p=[.25, .5, .2, .05]),
    "age_building": np.where(rng.random(n) < 0.01, 999, rng.gamma(2, 12, n).astype(int)),
    "plinth_area_sq_ft": rng.lognormal(5.9, 0.4, n).astype(int),
    "land_surface_condition": rng.choice(["Flat", "Moderate slope", "Steep slope"], n, p=[.83, .14, .03]),
    "roof_type": rng.choice(["Bamboo/Timber-Light roof", "Bamboo/Timber-Heavy roof", "RCC/RB/RBC"], n, p=[.66, .28, .06]),
    "position": rng.choice(["Not attached", "Attached-1 side", "Attached-2 side", "Attached-3 side"], n, p=[.79, .17, .035, .005]),
})
df["height_ft_pre_eq"] = df["count_floors_pre_eq"] * rng.normal(7.5, 1.0, n)
mats = ["adobe_mud", "mud_mortar_stone", "stone_flag", "cement_mortar_stone", "mud_mortar_brick",
        "cement_mortar_brick", "timber", "bamboo", "rc_non_engineered", "rc_engineered", "other"]
probs = [.04, .80, .035, .016, .023, .07, .26, .08, .04, .016, .012]
for m, p in zip(mats, probs):
    df[f"has_superstructure_{m}"] = (rng.random(n) < p).astype(int)

# A made-up damage process: shaking x vulnerability + noise.
msi = vdc_msi[vdc]
vuln = (1.2 * df.has_superstructure_mud_mortar_stone + 0.8 * df.has_superstructure_adobe_mud
        - 1.0 * df.has_superstructure_rc_engineered - 0.6 * df.has_superstructure_cement_mortar_brick
        + 0.01 * df.age_building.clip(0, 200) + 0.2 * (df.roof_type == "Bamboo/Timber-Heavy roof"))
score = 1.1 * (msi - 6.8) + vuln + rng.logistic(0, 0.9, n)
df["damage_grade"] = ["Grade %d" % g for g in np.digitize(score, [-1.2, -0.4, 0.4, 1.1]) + 1]
df.loc[rng.choice(n, 3, replace=False), "damage_grade"] = np.nan

df.to_csv(out / "csv_building_structure.csv", index=False)
pd.DataFrame({"vdcmun_id": np.arange(n_vdc) + 1000, "msi": vdc_msi}).to_csv(out / "msi_vdc.csv", index=False)
print(f"wrote {n:,} fake buildings to {out}")

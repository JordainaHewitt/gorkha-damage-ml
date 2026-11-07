"""Pipeline check while the MSI join is pending.

Replaces MSI with a per-municipality location feature: the mean damage grade of
each vdcmun_id, computed on TRAINING rows only (target encoding; no test leakage).

  Press Run, or: python3 scripts/01b_location_proxy.py [--trees 30]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import data as D          # noqa: E402
from src.metrics import evaluate  # noqa: E402

PAPER = {("full", "5dg"): (0.49, 0.69, 0.81), ("full", "tl"): (0.68, 0.41, 0.34),
         ("basic", "5dg"): (0.46, 0.73, 0.93), ("basic", "tl"): (0.64, 0.45, 0.39)}

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=str(ROOT / "data/raw/npc/eq2015.csv"))
ap.add_argument("--trees", type=int, default=30)
a = ap.parse_args()

df = D.load_raw(a.data)
y5 = D.make_target(df, "5dg")
split = D.load_or_make_split(y5, ROOT / "data/processed/split.json")
tr, te = np.array(split["train"]), np.array(split["test"])
loc = df["vdcmun_id"].to_numpy()
rows = []
for setting in ("full", "basic"):
    X = D.build_features(df, setting, use_msi=False)
    for scale, k in (("5dg", 5), ("tl", 3)):
        y = D.make_target(df, scale)
        enc = pd.Series(y[tr]).groupby(loc[tr]).mean()
        X2 = X.assign(loc_mean_grade=pd.Series(loc).map(enc).to_numpy())
        Xtr, Xte = D.impute(X2.iloc[tr], X2.iloc[te])
        m = RandomForestRegressor(n_estimators=a.trees, max_depth=20, n_jobs=-1, random_state=0).fit(Xtr, y[tr])
        r = evaluate(y[te], m.predict(Xte), k)
        p = PAPER[(setting, scale)]
        rows.append({"setting": setting, "scale": scale, "acc_paper": p[0], "acc_ours": round(r["accuracy"], 3),
                     "mae_paper": p[1], "mae_ours": round(r["mae"], 3), "mse_paper": p[2], "mse_ours": round(r["mse"], 3)})
out = pd.DataFrame(rows)
(ROOT / "results/location_proxy").mkdir(parents=True, exist_ok=True)
out.to_csv(ROOT / "results/location_proxy/vs_paper.csv", index=False)
print(out.to_string(index=False))

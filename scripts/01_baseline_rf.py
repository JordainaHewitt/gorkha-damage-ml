"""Stage 1: reproduce the random forest baselines of Ghimire et al. (2022).

Paper Table 2 hyperparameters (validation set):
  RFR  n_estimators=1000, max_depth=20   -> acc 0.49, MAE 0.66, MSE 0.76
  RFC  n_estimators=1000, max_depth=30   -> acc 0.53, MAE 0.67, MSE 1.25
Paper Table 3 (test set, RFR):
  full  5DG acc 0.49 MAE 0.69 MSE 0.81 | TL acc 0.68 MAE 0.41 MSE 0.34
  basic 5DG acc 0.46 MAE 0.73 MSE 0.93 | TL acc 0.64 MAE 0.45 MSE 0.39

Run: press Run in VS Code, or `python3 scripts/01_baseline_rf.py`.
Defaults are set in the SETTINGS block below; command-line flags override them, e.g.
  python3 scripts/01_baseline_rf.py --trees 50          # quick test
  python3 scripts/01_baseline_rf.py --msi-col msi_muni  # municipality-level MSI
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import data as D          # noqa: E402
from src.metrics import evaluate  # noqa: E402
from src.plots import plot_confusions  # noqa: E402

#  SETTINGS
DATA = ROOT / "data/raw/npc/eq2015.csv"            # survey table (Kaggle mirror)
MSI = ROOT / "data/processed/msi_buildings.csv"    # from scripts/00b_join_msi.py
MSI_COL = "msi_vdc"                                # "msi_vdc" (old VDC) or "msi_muni" (municipality)
TREES = 1000                                      # paper: 1000
OUT = ROOT / "results/rf"

PAPER = {  # test-set RFR numbers, Table 3
    ("full", "5dg"): {"accuracy": 0.49, "mae": 0.69, "mse": 0.81},
    ("full", "tl"): {"accuracy": 0.68, "mae": 0.41, "mse": 0.34},
    ("basic", "5dg"): {"accuracy": 0.46, "mae": 0.73, "mse": 0.93},
    ("basic", "tl"): {"accuracy": 0.64, "mae": 0.45, "mse": 0.39},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--msi", default=str(MSI))
    ap.add_argument("--msi-col", default=MSI_COL, help="msi_vdc (old-VDC level) or msi_muni (municipality level)")
    ap.add_argument("--no-msi", action="store_true")
    ap.add_argument("--trees", type=int, default=TREES)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    use_msi = not args.no_msi
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    df = D.load_raw(args.data, args.msi if use_msi else None, args.msi_col)
    y5 = D.make_target(df, "5dg")
    split = D.load_or_make_split(y5, ROOT / "data" / "processed" / "split.json")
    tr, va, te = (np.array(split[k]) for k in ("train", "val", "test"))

    rows, confusions = [], {}
    for setting in ("full", "basic"):
        X = D.build_features(df, setting, use_msi)
        Xtr, Xva, Xte = D.impute(X.iloc[tr], X.iloc[va], X.iloc[te])
        print(f"\n=== {setting} features: {X.shape[1]} columns ===")

        for scale in ("5dg", "tl"):
            y = D.make_target(df, scale)
            k = 5 if scale == "5dg" else 3
            models = {
                "RFR": RandomForestRegressor(n_estimators=args.trees, max_depth=20, n_jobs=-1, random_state=0),
                "RFC": RandomForestClassifier(n_estimators=args.trees, max_depth=30, n_jobs=-1, random_state=0),
                "uniform": DummyClassifier(strategy="uniform", random_state=0),
                "stratified": DummyClassifier(strategy="stratified", random_state=0),
            }
            for name, m in models.items():
                t0 = time.time()
                m.fit(Xtr, y[tr])
                for part, Xp, idx in (("val", Xva, va), ("test", Xte, te)):
                    r = evaluate(y[idx], m.predict(Xp), k)
                    rows.append({"model": name, "setting": setting, "scale": scale, "split": part,
                                 **{k_: r[k_] for k_ in ("accuracy", "mae", "mse", "within_1", "qwk")},
                                 "recall": r["recall"], "fit_s": round(time.time() - t0, 1)})
                    if part == "test" and name in ("RFR", "RFC"):
                        confusions[f"{name} {setting} {scale}"] = r["confusion"]
                print(f"  {name:10s} {scale}: test acc={rows[-1]['accuracy']:.3f} "
                      f"mae={rows[-1]['mae']:.3f} qwk={rows[-1]['qwk']:.3f}  ({rows[-1]['fit_s']}s)")

    res = pd.DataFrame(rows)
    res.to_csv(out / "results.csv", index=False)
    (out / "confusions.json").write_text(json.dumps(confusions))
    plot_confusions(confusions, out / "confusion_matrices.png")

    # Side-by-side with the paper (test set, RFR)
    cmp = []
    for (setting, scale), p in PAPER.items():
        r = res.query("model=='RFR' and setting==@setting and scale==@scale and split=='test'").iloc[0]
        cmp.append({"setting": setting, "scale": scale,
                    **{f"{m}_paper": p[m] for m in p}, **{f"{m}_ours": round(r[m], 3) for m in p}})
    cmp = pd.DataFrame(cmp)
    cmp.to_csv(out / "vs_paper.csv", index=False)
    print("\nRFR test set vs. paper Table 3:\n", cmp.to_string(index=False))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()

"""Stage 2a: XGBoost with three treatments of the ordinal target (professor's suggestion).

Same features, split (data/processed/split.json), imputation and metrics as 01_baseline_rf.py.

Models (K = 5 grades, or 3 for the traffic light):
  XGBC   classification, objective multi:softprob. Label = argmax_k P(y=k).
         Loss treats grades as unordered: predicting 5 for a true 1 costs the same as predicting 2.
  XGBR   regression, objective reg:squarederror on y in {1..K}. Label = round(f(x)).
         Paper convention: accuracy/confusion on the rounded value, MAE/MSE on the unrounded one.
  XGBO   ordinal, Frank & Hall (2001): K-1 binary models h_k(x) = P(y > k | x), k = 1..K-1.
         Class probabilities  P(y=k) = P(y>k-1) - P(y>k), with P(y>0)=1, P(y>K)=0.
         The h_k are fit independently, so they can cross (P(y>2) > P(y>1)); we enforce
         monotonicity with a running minimum over k and report how often it was needed.
         Two decodings:
           XGBO       label = 1 + sum_k 1[P(y>k) > 0.5]   (median of the predicted distribution;
                      minimises expected absolute error)
           XGBO_mean  score = 1 + sum_k P(y>k)            (expected grade E[y]; continuous, so it is
                      scored like XGBR: rounded for accuracy, raw for MAE/MSE)

Tuning: fixed learning rate, depth etc. in SETTINGS; the number of trees is chosen by early stopping
on the validation set. Validation metrics are therefore slightly optimistic; use the test rows.

Run: press Run in VS Code, or `python3 scripts/03_xgboost.py`.
  python3 scripts/03_xgboost.py --rounds 100 --out results/xgb_quick   # quick test (~1 min)
  python3 scripts/03_xgboost.py --only full_5dg   # one combination
Each setting x scale combination is saved to OUT/parts/ as it finishes; finished parts are skipped on
a rerun (use --redo to refit). The summary tables are rebuilt from all parts at the end.
"""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import data as D          # noqa: E402
from src.metrics import evaluate  # noqa: E402
from src.plots import plot_confusions  # noqa: E402

# SETTINGS
DATA = ROOT / "data/raw/npc/eq2015.csv"
MSI = ROOT / "data/processed/msi_buildings.csv"
MSI_COL = "msi_vdc"
ROUNDS = 2000            # max boosting rounds (early stopping picks the actual number); 100 = quick test
EARLY_STOP = 50          # stop if the validation loss has not improved for this many rounds
PARAMS = dict(           # shared by all XGBoost models
    learning_rate=0.1,
    max_depth=8,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    tree_method="hist",
    max_bin=1024,        # hist bins per feature. MSI has 570 distinct values (one per old VDC); the default
                         # 256 merges neighbouring VDCs and costs ~0.03 MAE (XGBR full 5DG: 0.676 -> 0.649).
    n_jobs=-1,
    random_state=0,
)
SETTINGS_RUN = ("full", "basic")
SCALES_RUN = ("5dg", "tl")
SAVE_MODELS = True       # saves models (for SHAP, scripts/05_shap.py) and test predictions
OUT = ROOT / "results/xgb"
RF_RESULTS = ROOT / "results/rf/results.csv"   # for the side-by-side table



def fit(model, Xtr, ytr, Xva, yva):
    model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    return model


def make(kind, rounds, early):
    common = dict(n_estimators=rounds, early_stopping_rounds=early, **PARAMS)
    if kind == "clf":
        return xgb.XGBClassifier(objective="multi:softprob", eval_metric="mlogloss", **common)
    if kind == "reg":
        return xgb.XGBRegressor(objective="reg:squarederror", eval_metric="rmse", **common)
    if kind == "bin":
        return xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss", **common)
    raise ValueError(kind)


def ordinal_decode(P_gt):
    """P_gt: (n, K-1) raw P(y>k). Returns monotone P, median label, expected grade, crossing rate."""
    crossed = np.any(np.diff(P_gt, axis=1) > 0, axis=1).mean()
    P = np.minimum.accumulate(P_gt, axis=1)          # enforce P(y>1) >= P(y>2) >= ...
    label = 1 + (P > 0.5).sum(axis=1)
    mean = 1 + P.sum(axis=1)
    return P, label, mean, float(crossed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--msi", default=str(MSI))
    ap.add_argument("--msi-col", default=MSI_COL)
    ap.add_argument("--no-msi", action="store_true")
    ap.add_argument("--rounds", type=int, default=ROUNDS)
    ap.add_argument("--early", type=int, default=EARLY_STOP)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--only", nargs="*", default=None, help="combinations to fit, e.g. full_5dg basic_tl")
    ap.add_argument("--redo", action="store_true", help="refit combinations that already have a part file")
    args = ap.parse_args()
    use_msi = not args.no_msi
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "models").mkdir(exist_ok=True)
    parts = out / "parts"; parts.mkdir(exist_ok=True)
    todo = [(s_, c_) for s_ in SETTINGS_RUN for c_ in SCALES_RUN
            if (args.only is None or f"{s_}_{c_}" in args.only)
            and (args.redo or not (parts / f"{s_}_{c_}.pkl").exists())]
    print(f"[xgb] combinations to fit: {[f'{a}_{b}' for a, b in todo] or 'none (rebuilding tables)'}")

    df = D.load_raw(args.data, args.msi if use_msi else None, args.msi_col)
    y5 = D.make_target(df, "5dg")
    split = D.load_or_make_split(y5, ROOT / "data" / "processed" / "split.json")
    tr, va, te = (np.array(split[k]) for k in ("train", "val", "test"))

    for setting in dict.fromkeys(s_ for s_, _ in todo):
        X = D.build_features(df, setting, use_msi)
        Xtr, Xva, Xte = D.impute(X.iloc[tr], X.iloc[va], X.iloc[te])
        print(f"\n=== {setting} features: {X.shape[1]} columns ===")

        for scale in [c_ for s_, c_ in todo if s_ == setting]:
            rows, confusions, preds, diag = [], {}, {}, {}
            y = D.make_target(df, scale)
            k = 5 if scale == "5dg" else 3
            ytr, yva = y[tr], y[va]
            tag = f"{setting}_{scale}"
            outputs = {}   # name -> (val pred, test pred, fit seconds, n_trees)

            #  classification
            t0 = time.time()
            m = fit(make("clf", args.rounds, args.early), Xtr, ytr - 1, Xva, yva - 1)
            outputs["XGBC"] = (m.predict(Xva) + 1, m.predict(Xte) + 1, time.time() - t0, m.best_iteration + 1)
            if SAVE_MODELS: m.save_model(out / "models" / f"XGBC_{tag}.json")
            preds[f"XGBC_{tag}_proba"] = m.predict_proba(Xte).astype(np.float32)

            # regression
            t0 = time.time()
            m = fit(make("reg", args.rounds, args.early), Xtr, ytr, Xva, yva)
            outputs["XGBR"] = (m.predict(Xva), m.predict(Xte), time.time() - t0, m.best_iteration + 1)
            if SAVE_MODELS: m.save_model(out / "models" / f"XGBR_{tag}.json")

            # ordinal (Frank & Hall): one binary model per threshold
            t0 = time.time()
            Pva, Pte, trees = [], [], []
            for thr in range(1, k):
                m = fit(make("bin", args.rounds, args.early), Xtr, (ytr > thr).astype(int),
                        Xva, (yva > thr).astype(int))
                Pva.append(m.predict_proba(Xva)[:, 1]); Pte.append(m.predict_proba(Xte)[:, 1])
                trees.append(m.best_iteration + 1)
                if SAVE_MODELS: m.save_model(out / "models" / f"XGBO_{tag}_gt{thr}.json")
            dt = time.time() - t0
            _, lab_va, mean_va, _ = ordinal_decode(np.column_stack(Pva))
            Pm, lab_te, mean_te, crossed = ordinal_decode(np.column_stack(Pte))
            outputs["XGBO"] = (lab_va, lab_te, dt, sum(trees))
            outputs["XGBO_mean"] = (mean_va, mean_te, dt, sum(trees))
            preds[f"XGBO_{tag}_Pgt"] = Pm.astype(np.float32)
            diag[tag] = {"ordinal_crossing_rate_test": crossed, "ordinal_trees_per_threshold": trees}

            for name, (pv, pt, sec, ntree) in outputs.items():
                for part, p, idx in (("val", pv, va), ("test", pt, te)):
                    r = evaluate(y[idx], np.asarray(p), k)
                    rows.append({"model": name, "setting": setting, "scale": scale, "split": part,
                                 **{k_: r[k_] for k_ in ("accuracy", "mae", "mse", "within_1", "qwk")},
                                 "recall": r["recall"], "fit_s": round(sec, 1), "n_trees": int(ntree)})
                    if part == "test":
                        confusions[f"{name} {setting} {scale}"] = r["confusion"]
                preds[f"{name}_{tag}"] = np.asarray(pt, dtype=np.float32)
                t = rows[-1]
                print(f"  {name:9s} {scale}: test acc={t['accuracy']:.3f} mae={t['mae']:.3f} "
                      f"mse={t['mse']:.3f} qwk={t['qwk']:.3f} w1={t['within_1']:.3f}  "
                      f"recall={t['recall']}  ({t['fit_s']}s, {ntree} trees)")
            print(f"  ordinal threshold crossing (before fix): {crossed:.2%} of test rows")
            with open(parts / f"{tag}.pkl", "wb") as f:
                pickle.dump({"rows": rows, "confusions": confusions, "preds": preds, "diag": diag}, f)

    # assemble every finished combination
    rows, confusions, preds, diag = [], {}, {}, {}
    for s_ in SETTINGS_RUN:
        for c_ in SCALES_RUN:
            f = parts / f"{s_}_{c_}.pkl"
            if f.exists():
                d = pickle.loads(f.read_bytes())
                rows += d["rows"]; confusions.update(d["confusions"]); preds.update(d["preds"]); diag.update(d["diag"])
    if not rows:
        print("no finished combinations"); return

    res = pd.DataFrame(rows)
    res.to_csv(out / "results.csv", index=False)
    (out / "confusions.json").write_text(json.dumps(confusions))
    (out / "diagnostics.json").write_text(json.dumps(diag, indent=2))
    plot_confusions(confusions, out / "confusion_matrices.png")
    if SAVE_MODELS:
        np.savez_compressed(out / "test_predictions.npz", test_idx=te, **preds)

    # Side-by-side with the random forests (test set)
    allres = res
    if Path(RF_RESULTS).exists():
        rf = pd.read_csv(RF_RESULTS).query("model in ['RFR','RFC']")
        allres = pd.concat([rf, res], ignore_index=True)
    cols = ["accuracy", "mae", "mse", "within_1", "qwk"]
    cmp = (allres.query("split=='test'")
                 .set_index(["setting", "scale", "model"])[cols + ["recall"]]
                 .sort_index())
    cmp[cols] = cmp[cols].round(3)
    cmp.to_csv(out / "comparison_test.csv")
    with pd.option_context("display.width", 200, "display.max_colwidth", 40):
        print("\nTest set, all tree models:\n", cmp.to_string())
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()

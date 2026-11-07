"""Comparison figures: the paper's method (RFR) vs XGBoost and MLP, plus training curves.

  Fig C1  test metrics, every model family x loss type (full and basic features, 5 grades and traffic light)
  Fig C2  Fig 8/9 redone for the best models: test confusion matrices RFR vs XGBO vs MLPO,
          plus the difference (best - RFR) so the gain per cell is visible
  Fig C3  recall per damage grade (what each model gets right, grade by grade)
  Fig C4  MLP training curves, loss vs epoch: plain vs VDC-embedding, for CE / MSE / CORAL
  Fig C5  XGBoost training curves, loss vs boosting round (the tree-model analogue of epochs),
          recomputed from the saved models on a subsample of train and validation buildings

Random forest has no training curve of this kind: its trees are fit independently (bagging), so there
is no sequence of updates to plot. The paper's Fig 10 is a learning curve over training-set size instead.

Inputs: results/{rf,xgb,mlp,mlp_embed} from scripts 01, 03, 04, and results/xgb/models for Fig C5.
Press Run, or: python3 scripts/06_compare.py        Output: results/comparison/
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.metrics import row_normalize     # noqa: E402

# ---------------- SETTINGS ----------------
RESULTS = {"RF": ROOT / "results/rf", "XGB": ROOT / "results/xgb",
           "MLP + VDC emb.": ROOT / "results/mlp_embed", "MLP (plain)": ROOT / "results/mlp"}
PAPER = ROOT / "results/rf/vs_paper.csv"            # paper's reported RFR test values
XGB_MODELS = ROOT / "results/xgb/models"
DATA = ROOT / "data/raw/npc/eq2015.csv"
MSI = ROOT / "data/processed/msi_buildings.csv"
CURVE_ROWS = 50000          # buildings sampled from train and from validation for Fig C5
CURVE_POINTS = 30           # boosting-round checkpoints per model
OUT = ROOT / "results/comparison"
# ------------------------------------------

INK, MUTED, GRID = "#1f1f1e", "#6b6a65", "#e4e3df"
FAM_COLOR = {"RF": "#2a78d6", "XGB": "#eb6834", "MLP + VDC emb.": "#1baf7a", "MLP (plain)": "#eda100"}
LOSS_OF = {"C": "Classification (CE)", "R": "Regression (MSE)", "O": "Ordinal, median", "O_mean": "Ordinal, mean"}
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})


def load_results():
    rows = []
    for fam, d in RESULTS.items():
        r = pd.read_csv(d / "results.csv")
        r = r[(r.split == "test") & r.model.str.match(r"^(RF|XGB|MLP)")].copy()
        r["family"] = fam
        r["loss"] = r.model.str.replace(r"^(RF|XGB|MLP)", "", regex=True).map(LOSS_OF)
        rows.append(r)
    return pd.concat(rows, ignore_index=True)


def confusions():
    cms = {}
    for fam, d in RESULTS.items():
        for k, v in json.loads((d / "confusions.json").read_text()).items():
            cms[(fam, k)] = np.array(v)
    return cms


def style(ax, axis="both"):
    ax.grid(axis=axis, color=GRID, lw=0.6); ax.set_axisbelow(True)


# ---------------------------------------------------------------- Fig C1
def fig_metrics(res, paper, setting):
    metrics = [("accuracy", "Accuracy (higher better)"), ("mae", "MAE (lower better)"),
               ("mse", "MSE (lower better)"), ("qwk", "Quadratic weighted kappa (higher better)")]
    losses = list(LOSS_OF.values())
    fams = list(RESULTS)
    fig, axes = plt.subplots(2, 4, figsize=(14, 6.2), constrained_layout=True, sharey=True)
    for row, (scale, sname) in enumerate((("5dg", "5 damage grades"), ("tl", "traffic light"))):
        sub = res[(res.setting == setting) & (res.scale == scale)]
        p = paper[(paper.setting == setting) & (paper.scale == scale)].iloc[0]
        for ax, (m, title) in zip(axes[row], metrics):
            for f_i, fam in enumerate(fams):
                s = sub[sub.family == fam].set_index("loss")[m].reindex(losses)
                y = np.arange(len(losses)) + (f_i - 1.5) * 0.17
                ax.plot(s.values, y, "o", ms=6, color=FAM_COLOR[fam], mec="white", mew=0.8,
                        label=fam if (row, m) == (0, "accuracy") else None)
            if f"{m}_paper" in p:
                ax.axvline(p[f"{m}_paper"], color=INK, lw=0.8, ls="--",
                           label="paper RFR" if (row, m) == (0, "accuracy") else None)
            ax.set_yticks(range(len(losses)), losses); ax.invert_yaxis(); style(ax)
            ax.set_title(f"{title}\n{sname}" if row == 0 else sname, loc="left", fontsize=8.5)
    fig.legend(loc="outside upper right", ncol=5, frameon=False)
    fig.suptitle(f"Fig C1. Test metrics by model family and training loss ({setting} features)",
                 x=0.01, ha="left", fontsize=10)
    fig.savefig(OUT / f"figC1_metrics_{setting}.png", dpi=200); plt.close(fig)


# ---------------------------------------------------------------- Fig C2
def draw_cm(ax, m, title, ticks, diff=False):
    if diff:
        im = ax.imshow(m, cmap="RdBu", vmin=-0.3, vmax=0.3)
    else:
        im = ax.imshow(m, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(m)):
        for j in range(len(m)):
            v = m[i, j]
            txt = f"{v:+.2f}" if diff else f"{v:.2f}"
            dark = abs(v) > 0.2 if diff else v > 0.5
            ax.text(j, i, txt, ha="center", va="center", fontsize=7.5, color="white" if dark else INK)
    ax.set_xticks(range(len(m)), ticks); ax.set_yticks(range(len(m)), ticks)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title, fontsize=8.5, loc="left")
    for s in ax.spines.values():
        s.set_visible(False)
    return im


def fig_confusions(cms, setting):
    fig, axes = plt.subplots(2, 5, figsize=(17, 7), constrained_layout=True)
    for row, (scale, ticks) in enumerate((("5dg", [f"DG{i}" for i in range(1, 6)]), ("tl", ["S", "M", "H"]))):
        rfr = row_normalize(cms[("RF", f"RFR {setting} {scale}")])
        xgbo = row_normalize(cms[("XGB", f"XGBO {setting} {scale}")])
        mlpo = row_normalize(cms[("MLP + VDC emb.", f"MLPO {setting} {scale}")])
        draw_cm(axes[row, 0], rfr, "RFR (paper method)", ticks)
        draw_cm(axes[row, 1], xgbo, "XGBO (ordinal XGBoost)", ticks)
        draw_cm(axes[row, 2], mlpo, "MLPO (CORAL MLP + VDC emb.)", ticks)
        draw_cm(axes[row, 3], xgbo - rfr, "XGBO - RFR", ticks, diff=True)
        im = draw_cm(axes[row, 4], mlpo - rfr, "MLPO - RFR", ticks, diff=True)
    fig.colorbar(im, ax=axes[:, 4], shrink=0.6, label="change in row share (blue = more)")
    fig.suptitle(f"Fig C2. Test confusion matrices (rows sum to 1), {setting} features. Top: 5 grades (cf. paper Fig 8); "
                 "bottom: traffic light (cf. Fig 9). Diagonal gains are blue.", x=0.01, ha="left", fontsize=10)
    fig.savefig(OUT / f"figC2_confusions_{setting}.png", dpi=200); plt.close(fig)


# ---------------------------------------------------------------- Fig C3
def fig_recall(res, setting):
    # colour = model family (same as Fig C1); hatching separates the two XGBoost losses
    pick = [("RF", "RFR"), ("XGB", "XGBC"), ("XGB", "XGBO"), ("MLP + VDC emb.", "MLPO")]
    colors = [FAM_COLOR[f] for f, _ in pick]
    hatches = ["", "///", "", ""]
    labels = ["RFR (paper method)", "XGBC (classification)", "XGBO (ordinal)", "MLPO (CORAL + VDC emb.)"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), constrained_layout=True,
                             gridspec_kw={"width_ratios": [5, 3]})
    for ax, (scale, ticks) in zip(axes, (("5dg", [f"DG{i}" for i in range(1, 6)]), ("tl", ["S", "M", "H"]))):
        w = 0.8 / len(pick)
        for i, ((fam, model), c, hat, lab) in enumerate(zip(pick, colors, hatches, labels)):
            r = res[(res.family == fam) & (res.model == model) & (res.setting == setting) & (res.scale == scale)]
            rec = json.loads(r.recall.iloc[0])
            x = np.arange(len(rec)) + (i - (len(pick) - 1) / 2) * w
            ax.bar(x, rec, width=w * 0.9, color=c, hatch=hat, edgecolor="white", lw=0, label=lab)
        ax.set_xticks(range(len(ticks)), ticks); ax.set_ylim(0, 1); ax.set_ylabel("recall (share of true class found)")
        style(ax, "y")
    axes[0].legend(frameon=False, ncol=4, loc="upper left", fontsize=8)
    fig.suptitle(f"Fig C3. Recall per damage class, test set, {setting} features", x=0.01, ha="left", fontsize=10)
    fig.savefig(OUT / f"figC3_recall_{setting}.png", dpi=200); plt.close(fig)


# ---------------------------------------------------------------- Fig C4
def fig_mlp_curves(tag="full_5dg"):
    names = {"ce": "MLPC: cross-entropy", "mse": "MLPR: squared error", "coral": "MLPO: CORAL (sum of BCE)"}
    variants = (("MLP (plain)", "#eda100"), ("MLP + VDC emb.", "#1baf7a"))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for ax, (loss, title) in zip(axes, names.items()):
        for fam, c in variants:
            h = json.loads((RESULTS[fam] / "training_history.json").read_text())[f"{tag}_{loss}"]
            e = [r["epoch"] for r in h]
            ax.plot(e, [r["train"] for r in h], color=c, lw=1.3, ls="--")
            ax.plot(e, [r["val"] for r in h], color=c, lw=1.8, label=fam)
            best = int(np.argmin([r["val"] for r in h]))
            ax.plot(e[best], h[best]["val"], "o", color=c, ms=5, mec="white")
        ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.set_title(title, loc="left", fontsize=9); style(ax)
        lo = min(min(min(r["val"], r["train"]) for r in
                     json.loads((RESULTS[f] / "training_history.json").read_text())[f"{tag}_{loss}"][1:])
                 for f, _ in variants)
        hi = max(json.loads((RESULTS[f] / "training_history.json").read_text())[f"{tag}_{loss}"][1]["val"]
                 for f, _ in variants)
        ax.set_ylim(lo - 0.08 * (hi - lo), hi + 0.15 * (hi - lo))     # skip the epoch-0 spike
    axes[0].plot([], [], color=MUTED, ls="--", label="train"); axes[0].plot([], [], color=MUTED, label="validation")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"Fig C4. MLP training curves, {tag.replace('_', ', ')} (dot = best validation epoch, "
                 "weights kept from there)", x=0.01, ha="left", fontsize=10)
    fig.savefig(OUT / f"figC4_mlp_training_{tag}.png", dpi=200); plt.close(fig)


# ---------------------------------------------------------------- Fig C5
def xgb_curves(tag="full_5dg", redo=False):
    cache = OUT / f"xgb_curves_{tag}.json"
    if cache.exists() and not redo:
        return json.loads(cache.read_text())
    import xgboost as xgb
    from src import data as D
    setting, scale = tag.split("_")
    df = D.load_raw(DATA, MSI, "msi_vdc")
    y = D.make_target(df, scale)
    split = D.load_or_make_split(D.make_target(df, "5dg"), ROOT / "data/processed/split.json")
    rng = np.random.default_rng(0)
    tr = rng.choice(np.array(split["train"]), CURVE_ROWS, replace=False)
    va = rng.choice(np.array(split["val"]), CURVE_ROWS, replace=False)
    X = D.build_features(df, setting, use_msi=True)
    Xall_tr = np.array(split["train"])
    _, Xtr, Xva = D.impute(X.iloc[Xall_tr], X.iloc[tr], X.iloc[va])   # impute with full-train statistics
    dtr, dva = xgb.DMatrix(Xtr), xgb.DMatrix(Xva)
    k = 5 if scale == "5dg" else 3
    eps = 1e-7
    models = {"XGBC": ("mlogloss", y - 1), "XGBR": ("mse", y)}
    models.update({f"XGBO_gt{t}": ("logloss", (y > t).astype(int)) for t in range(1, k)})
    curves = {}
    for name, (metric, target) in models.items():
        b = xgb.Booster(); b.load_model(XGB_MODELS / f"{name.replace('_gt', f'_{tag}_gt') if 'gt' in name else name + '_' + tag}.json")
        n = b.num_boosted_rounds(); best = int(b.attr("best_iteration")) + 1
        steps = np.unique(np.r_[np.geomspace(1, n, CURVE_POINTS).astype(int), best])
        out = {"metric": metric, "rounds": steps.tolist(), "best": best, "train": [], "val": []}
        for d, idx, key in ((dtr, tr, "train"), (dva, va, "val")):
            t = target[idx]
            for s in steps:
                p = b.predict(d, iteration_range=(0, int(s)))
                if metric == "mse":
                    v = np.mean((p - t) ** 2)
                elif metric == "mlogloss":
                    v = -np.mean(np.log(np.clip(p[np.arange(len(t)), t], eps, 1)))
                else:
                    p = np.clip(p, eps, 1 - eps); v = -np.mean(t * np.log(p) + (1 - t) * np.log(1 - p))
                out[key].append(float(v))
        curves[name] = out
        print(f"  {name}: {n} rounds stored, best {best}, val {out['val'][-1]:.4f}")
    cache.write_text(json.dumps(curves))
    return curves


def fig_xgb_curves(curves, tag="full_5dg"):
    names = list(curves)
    nc = 3; nr = int(np.ceil(len(names) / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(12, 3.4 * nr), constrained_layout=True, squeeze=False)
    for ax in axes.flat[len(names):]:
        ax.axis("off")
    axes = axes.flat
    label = {"XGBC": "XGBC: multiclass log-loss", "XGBR": "XGBR: MSE"}
    for ax, n in zip(axes, names):
        c = curves[n]
        ax.plot(c["rounds"], c["train"], color="#eb6834", lw=1.3, ls="--", label="train")
        ax.plot(c["rounds"], c["val"], color="#eb6834", lw=1.8, label="validation")
        ax.axvline(c["best"], color=MUTED, lw=0.8, ls=":")
        ax.set_xscale("log"); ax.set_xlabel("boosting round (trees)"); style(ax)
        ax.set_title(label.get(n, f"XGBO P(DG > {n[-1]}): log-loss"), loc="left", fontsize=8.5)
        v = np.array(c["val"]); ax.set_ylim(min(min(c["train"]), v.min()) * 0.97, v[len(v) // 3] * 1.05)
    for i in range(0, len(names), nc):
        axes[i].set_ylabel("loss")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"Fig C5. XGBoost training curves, {tag.replace('_', ', ')} (dotted = early-stopping choice;\n"
                 f"{CURVE_ROWS:,} buildings each from train and validation; log x-axis)", x=0.01, ha="left", fontsize=10)
    fig.savefig(OUT / f"figC5_xgb_training_{tag}.png", dpi=200); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-xgb-curves", action="store_true", help="Fig C5 needs the data and models (~2 min)")
    ap.add_argument("--redo", action="store_true", help="recompute cached XGBoost curves")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    res = load_results()
    paper = pd.read_csv(PAPER)
    cms = confusions()
    cols = ["family", "model", "loss", "setting", "scale", "accuracy", "mae", "mse", "within_1", "qwk", "recall"]
    res[cols].sort_values(["setting", "scale", "loss", "family"]).to_csv(OUT / "all_models_test.csv", index=False)
    for setting in ("full", "basic"):
        fig_metrics(res, paper, setting)
        fig_confusions(cms, setting)
        fig_recall(res, setting)
    fig_mlp_curves("full_5dg")
    if not args.skip_xgb_curves:
        print("[compare] XGBoost loss vs boosting round ...")
        fig_xgb_curves(xgb_curves("full_5dg", args.redo), "full_5dg")
    print(f"Saved to {OUT}")


if __name__ == "__main__":
    main()

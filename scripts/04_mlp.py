"""Stage 2b: MLP (PyTorch) with three output heads / losses on an identical backbone.

Same features, split, imputation and metrics as 01_baseline_rf.py and 03_xgboost.py.
Numeric inputs (stories, age, area, height, MSI) are standardised with TRAINING mean/std;
0/1 indicator columns are left as they are.

Backbone  z = phi(x): Linear -> ReLU -> Dropout, repeated for each width in HIDDEN.
Each loss gets its own network (same architecture, same seed), so differences come from the loss only.

  MLPC   cross-entropy.   logits = W z + b in R^K,  P(y=k) = softmax_k.          label = argmax
  MLPR   squared error.   f(x) = w.z + b in R.                                      label = round(f)
  MLPO   CORAL (Cao, Mirjalili & Raschka 2020).
         One shared score g(x) = w.z (no bias) and K-1 threshold biases b_1..b_{K-1}:
             P(y > k | x) = sigmoid(g(x) + b_k)
         Loss = sum_k BCE( sigmoid(g + b_k), 1[y > k] ), averaged over the batch.
         Because every threshold shares the same g(x), P(y>k) is ordered across k whenever
         b_1 >= b_2 >= ... >= b_{K-1}; Cao et al. show the optimum has this ordering (checked below).
         Compare Frank & Hall in 03_xgboost.py: K-1 independent models, which can cross.
         Decoding as in 03_xgboost.py:
           MLPO       label = 1 + sum_k 1[P(y>k) > 0.5]     (median)
           MLPO_mean  score = 1 + sum_k P(y>k)              (expected grade)

Output-layer biases start at the training-set answer for a featureless model
(log class priors / mean grade / logit of P(y>k)), so training starts from the "stratified" baseline.
Optimiser: Adam, learning rate halved when validation loss stalls; early stopping on validation loss.

Optional ablation (EMBED = True): add a learned embedding of the old VDC (OCHA P-code, ~577 units)
to the input. Tests whether the network gains from explicit location identity beyond MSI.

Code layout follows the course notebooks (mlp.ipynb and HW0):
  MLP()            hidden stack as nn.Sequential, like HW0's MLP(width, depth, act) / the course's DeepNetwork
  n_params(), to_t(), predict()   same helpers as in HW0
  criterion        loss objects in the nn.MSELoss() style: CrossEntropyCriterion, GradeMSECriterion, CoralCriterion
  train_network()  the course loop (predictions = model(x); loss = criterion(...); zero_grad; backward; step),
                   extended with the changes marked [changed]: mini-batches, a validation set,
                   learning-rate halving on plateau, early stopping.

Run: press Run in VS Code, or `python3 scripts/04_mlp.py`.
  python3 scripts/04_mlp.py --epochs 3 --out results/mlp_quick   # quick test
  python3 scripts/04_mlp.py --only full_5dg --losses coral         # one combination / loss
  python3 scripts/04_mlp.py --embed                                # VDC-embedding ablation
Finished (combination, loss) runs are saved in OUT/parts/ and skipped on a rerun (--redo to refit).
"""
import argparse
import copy
import json
import pickle
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import config as C        # noqa: E402
from src import data as D          # noqa: E402
from src.metrics import evaluate  # noqa: E402
from src.plots import plot_confusions  # noqa: E402

# SETTINGS
DATA = ROOT / "data/raw/npc/eq2015.csv"
MSI = ROOT / "data/processed/msi_buildings.csv"
MSI_COL = "msi_vdc"
HIDDEN = (256, 256, 128)   # backbone widths
DROPOUT = 0.1
LR = 1e-3
WEIGHT_DECAY = 1e-5
BATCH = 1024
EPOCHS = 40                # maximum; early stopping usually ends sooner. 3 = quick test
PATIENCE = 5               # epochs without validation-loss improvement before stopping
EMBED = False              # True: add an old-VDC embedding (ablation)
EMBED_DIM = 16
SETTINGS_RUN = ("full", "basic")
SCALES_RUN = ("5dg", "tl")
LOSSES = ("ce", "mse", "coral")
THREADS = None             # torch CPU threads; None = torch default
OUT = ROOT / "results/mlp"
OTHER_RESULTS = {"rf": ROOT / "results/rf/results.csv", "xgb": ROOT / "results/xgb/results.csv"}
SEED = 0


NUMERIC_IN = C.NUMERIC + ["msi"]
NAMES = {"ce": "MLPC", "mse": "MLPR", "coral": "MLPO"}


# MLP backbone, loss objects, training loop, and helpers
def MLP(d_in, hidden=HIDDEN, act=nn.ReLU, dropout=DROPOUT):
    """Hidden stack phi(x) as nn.Sequential, like HW0's MLP(width, depth, act).

    [changed] widths can differ per layer (HIDDEN), dropout after each activation, and no output
    layer here: the output layer ("head") depends on the loss, so DamageNet adds it.
    """
    layers, d = [], d_in
    for h in hidden:
        layers += [nn.Linear(d, h), act(), nn.Dropout(dropout)]
        d = h
    return nn.Sequential(*layers)


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def to_t(a, dtype=torch.float32):
    """numpy / pandas -> torch tensor (course helper; here rows = buildings, so no reshape to (N,1))."""
    return torch.tensor(np.asarray(a), dtype=dtype)


class DamageNet(nn.Module):
    """phi = MLP(x [, location embedding]) followed by a loss-specific head.

    SingleLayerNN/DeepNetwork end in one Linear output for regression; this class keeps
    that for MSE and swaps the last layer for the other two losses.
    """
    def __init__(self, d_in, k, loss, n_loc=0):
        super().__init__()
        self.loss, self.k = loss, k
        self.emb = nn.Embedding(n_loc, EMBED_DIM) if n_loc else None
        self.backbone = MLP(d_in + (EMBED_DIM if n_loc else 0))
        d = HIDDEN[-1]
        if loss == "ce":
            self.head = nn.Linear(d, k)
        elif loss == "mse":
            self.head = nn.Linear(d, 1)
        else:  # coral
            self.head = nn.Linear(d, 1, bias=False)
            self.thr_bias = nn.Parameter(torch.zeros(k - 1))

    def forward(self, x, loc=None):
        if self.emb is not None:
            x = torch.cat([x, self.emb(loc)], dim=1)
        z = self.head(self.backbone(x))
        if self.loss == "ce":
            return z                                   # (n, K) logits
        if self.loss == "mse":
            return z.squeeze(1)                        # (n,) grade
        return z + self.thr_bias                       # (n, K-1) logits of P(y>k)

    @torch.no_grad()
    def init_output_bias(self, y, k):
        """Start from the featureless solution (training-set marginal)."""
        if self.loss == "ce":
            p = np.bincount(y - 1, minlength=k) / len(y)
            self.head.bias.copy_(torch.tensor(np.log(p), dtype=torch.float32))
        elif self.loss == "mse":
            self.head.bias.fill_(float(y.mean()))
        else:
            p = np.array([(y > j).mean() for j in range(1, k)])
            self.thr_bias.copy_(torch.tensor(np.log(p / (1 - p)), dtype=torch.float32))


# Criteria
class CrossEntropyCriterion(nn.Module):
    """-log softmax(logits)[y]; nn.CrossEntropyLoss wants classes 0..K-1."""
    def __init__(self):
        super().__init__(); self.ce = nn.CrossEntropyLoss()

    def forward(self, logits, y):
        return self.ce(logits, y - 1)


class GradeMSECriterion(nn.Module):
    """(f(x) - y)^2: the course's nn.MSELoss on the damage grade."""
    def __init__(self):
        super().__init__(); self.mse = nn.MSELoss()

    def forward(self, f, y):
        return self.mse(f, y.float())


class CoralCriterion(nn.Module):
    """sum_k BCE(sigmoid(g + b_k), 1[y > k]), averaged over buildings (Cao et al. 2020)."""
    def __init__(self, k):
        super().__init__(); self.k = k

    def forward(self, logits, y):
        levels = (y.unsqueeze(1) > torch.arange(1, self.k, device=y.device)).float()   # 1[y > k], (n, K-1)
        return F.binary_cross_entropy_with_logits(logits, levels, reduction="none").sum(1).mean()


def make_criterion(loss, k):
    return {"ce": CrossEntropyCriterion, "mse": GradeMSECriterion}.get(loss, lambda: CoralCriterion(k))()


@torch.no_grad()
def predict(model, X, L=None, bs=65536):
    """Raw network output for every row (course helper), in eval mode (dropout off), in chunks."""
    model.eval()
    outs = [model(X[i:i + bs], None if L is None else L[i:i + bs]) for i in range(0, len(X), bs)]
    return torch.cat(outs)


@torch.no_grad()
def evaluate_loss(model, criterion, X, L, y):
    return float(criterion(predict(model, X, L), y))


def decode(loss, out):
    """Raw network output -> dict of {model name: prediction (labels or continuous scores)}."""
    out = out.cpu()
    if loss == "ce":
        return {"MLPC": (out.argmax(1) + 1).numpy()}
    if loss == "mse":
        return {"MLPR": out.numpy()}
    P = torch.sigmoid(out).numpy()
    return {"MLPO": 1 + (P > 0.5).sum(1), "MLPO_mean": 1 + P.sum(1)}


def train_network(model, criterion, x_train, l_train, y_train, x_val, l_val, y_val,
                  epochs=EPOCHS, lr=LR, batch=BATCH, patience=PATIENCE):
    """The course training loop (mlp.ipynb / HW0 train_network), returning the loss history.

    [changed] vs the course loop, because there are 457k training buildings instead of ~1000 points:
      - mini-batches of `batch` rows, reshuffled every epoch (the course uses the full batch);
      - weight decay (L2 penalty) in Adam;
      - validation loss after every epoch; learning rate halved after 2 epochs without improvement
        (HW0 used a fixed StepLR schedule instead);
      - early stopping after `patience` epochs without improvement, keeping the best weights.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)
    best, best_state, wait, history = np.inf, None, 0, []
    g = torch.Generator().manual_seed(SEED)
    sub = slice(0, 100000)   # epoch-0 training loss on a subset (cheap; it is only the start point)
    history.append({"epoch": 0,
                     "train": evaluate_loss(model, criterion, x_train[sub], None if l_train is None else l_train[sub],
                                            y_train[sub]),
                     "val": evaluate_loss(model, criterion, x_val, l_val, y_val)})
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(x_train), generator=g)
        total, n = 0.0, 0
        for i in range(0, len(perm), batch):
            b = perm[i:i + batch]
            # ---- course loop, unchanged ----
            predictions = model(x_train[b], None if l_train is None else l_train[b])
            loss = criterion(predictions, y_train[b])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # --------------------------------
            total += loss.item() * len(b); n += len(b)
        v = evaluate_loss(model, criterion, x_val, l_val, y_val)
        scheduler.step(v)
        history.append({"epoch": epoch, "train": total / n, "val": v, "lr": optimizer.param_groups[0]["lr"]})
        if v < best - 1e-5:
            best, best_state, wait = v, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
        print(f"    epoch {epoch:2d}  train {total / n:.4f}  val {v:.4f}  lr {optimizer.param_groups[0]['lr']:.1e}"
              + ("  *" if wait == 0 else ""), flush=True)
        if wait >= patience:
            break
    model.load_state_dict(best_state)
    return history, epoch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--msi", default=str(MSI))
    ap.add_argument("--msi-col", default=MSI_COL)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--embed", action="store_true", default=EMBED)
    ap.add_argument("--only", nargs="*", default=None, help="combinations, e.g. full_5dg basic_tl")
    ap.add_argument("--losses", nargs="*", default=list(LOSSES))
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = Path(args.out) if args.out else (OUT.parent / (OUT.name + "_embed") if args.embed else OUT)
    (out / "parts").mkdir(parents=True, exist_ok=True); (out / "models").mkdir(exist_ok=True)
    if THREADS:
        torch.set_num_threads(THREADS)

    todo = [(s, c, l) for s in SETTINGS_RUN for c in SCALES_RUN for l in args.losses
            if (args.only is None or f"{s}_{c}" in args.only)
            and (args.redo or not (out / "parts" / f"{s}_{c}_{l}.pkl").exists())]
    print(f"[mlp] runs to fit: {[f'{s}_{c}_{l}' for s, c, l in todo] or 'none (rebuilding tables)'}")

    if todo:
        df = D.load_raw(args.data, args.msi, args.msi_col)
        y5 = D.make_target(df, "5dg")
        split = D.load_or_make_split(y5, ROOT / "data" / "processed" / "split.json")
        tr, va, te = (np.array(split[k]) for k in ("train", "val", "test"))

        L = {"tr": None, "va": None, "te": None}; n_loc = 0
        if args.embed:
            pc = pd.read_csv(args.msi, usecols=["building_id", "OCHA_PCODE"])
            codes = df[["building_id"]].merge(pc, on="building_id", how="left")["OCHA_PCODE"].astype(str)
            cats = {c: i + 1 for i, c in enumerate(sorted(codes.iloc[tr].unique()))}   # 0 = unseen
            loc = torch.tensor(codes.map(cats).fillna(0).astype(int).to_numpy())
            L = {"tr": loc[tr], "va": loc[va], "te": loc[te]}; n_loc = len(cats) + 1
            print(f"[mlp] VDC embedding: {len(cats)} units, {int((L['te'] == 0).sum())} unseen test rows")

    for setting in dict.fromkeys(s for s, _, _ in todo):
        X = D.build_features(df, setting, use_msi=True)
        Xtr, Xva, Xte = D.impute(X.iloc[tr], X.iloc[va], X.iloc[te])
        num = [c for c in NUMERIC_IN if c in X.columns]
        mu, sd = Xtr[num].mean(), Xtr[num].std()
        T = {}
        for name, Xp in (("tr", Xtr), ("va", Xva), ("te", Xte)):
            Xp = Xp.copy(); Xp[num] = (Xp[num] - mu) / sd
            T[name] = to_t(Xp.to_numpy(dtype=np.float32))
        print(f"\n=== {setting} features: {X.shape[1]} columns (standardised: {num}) ===")

        for scale in dict.fromkeys(c for s, c, _ in todo if s == setting):
            y = D.make_target(df, scale)
            k = 5 if scale == "5dg" else 3
            Y = {n: torch.tensor(y[i], dtype=torch.long) for n, i in (("tr", tr), ("va", va), ("te", te))}
            for loss in [l for s, c, l in todo if s == setting and c == scale]:
                tag = f"{setting}_{scale}_{loss}"
                t0 = time.time()
                print(f"  -- {tag}")
                torch.manual_seed(SEED)                       # same initial weights for every loss
                model = DamageNet(T["tr"].shape[1], k, loss, n_loc)
                model.init_output_bias(Y["tr"].numpy(), k)
                print(f"     {n_params(model):,} parameters")
                criterion = make_criterion(loss, k)
                hist, n_ep = train_network(model, criterion, T["tr"], L["tr"], Y["tr"], T["va"], L["va"], Y["va"],
                                           epochs=args.epochs)
                sec = time.time() - t0
                torch.save(model.state_dict(), out / "models" / f"{tag}.pt")
                pv = decode(loss, predict(model, T["va"], L["va"]))
                pt = decode(loss, predict(model, T["te"], L["te"]))
                rows, confusions, preds = [], {}, {}
                for name in pv:
                    for part, p, idx in (("val", pv[name], va), ("test", pt[name], te)):
                        r = evaluate(y[idx], np.asarray(p), k)
                        rows.append({"model": name, "setting": setting, "scale": scale, "split": part,
                                     **{k_: r[k_] for k_ in ("accuracy", "mae", "mse", "within_1", "qwk")},
                                     "recall": r["recall"], "fit_s": round(sec, 1), "epochs": n_ep})
                        if part == "test":
                            confusions[f"{name} {setting} {scale}"] = r["confusion"]
                    preds[f"{name}_{setting}_{scale}"] = np.asarray(pt[name], dtype=np.float32)
                    t = rows[-1]
                    print(f"  {name:9s} {scale}: test acc={t['accuracy']:.3f} mae={t['mae']:.3f} "
                          f"mse={t['mse']:.3f} qwk={t['qwk']:.3f} w1={t['within_1']:.3f}  "
                          f"recall={t['recall']}  ({sec:.0f}s, {n_ep} epochs)")
                diag = {}
                if loss == "coral":
                    b = model.thr_bias.detach().numpy()
                    diag = {"coral_biases": b.round(3).tolist(), "biases_ordered": bool(np.all(np.diff(b) <= 0))}
                    print(f"  CORAL biases {diag['coral_biases']} ordered={diag['biases_ordered']}")
                with open(out / "parts" / f"{tag}.pkl", "wb") as f:
                    pickle.dump({"rows": rows, "confusions": confusions, "preds": preds,
                                 "hist": hist, "diag": diag}, f)

    #  assemble every finished run
    rows, confusions, hists, diag = [], {}, {}, {}
    for s in SETTINGS_RUN:
        for c in SCALES_RUN:
            for l in LOSSES:
                f = out / "parts" / f"{s}_{c}_{l}.pkl"
                if f.exists():
                    d = pickle.loads(f.read_bytes())
                    rows += d["rows"]; confusions.update(d["confusions"])
                    hists[f"{s}_{c}_{l}"] = d["hist"]
                    if d["diag"]: diag[f"{s}_{c}_{l}"] = d["diag"]
    if not rows:
        print("no finished runs"); return
    res = pd.DataFrame(rows)
    res.to_csv(out / "results.csv", index=False)
    (out / "confusions.json").write_text(json.dumps(confusions))
    (out / "diagnostics.json").write_text(json.dumps(diag, indent=2))
    (out / "training_history.json").write_text(json.dumps(hists))
    plot_confusions(confusions, out / "confusion_matrices.png")
    plot_history(hists, out / "training_curves.png")

    #  one table with every model family (test set)
    allres = [res.assign(family="MLP")]
    for fam, p in OTHER_RESULTS.items():
        if Path(p).exists():
            r = pd.read_csv(p)
            allres.append(r[~r["model"].isin(["uniform", "stratified"])].assign(family=fam.upper()))
    cols = ["accuracy", "mae", "mse", "within_1", "qwk"]
    cmp = (pd.concat(allres, ignore_index=True).query("split=='test'")
             .set_index(["setting", "scale", "model"])[cols + ["recall"]].sort_index())
    cmp[cols] = cmp[cols].round(3)
    cmp.to_csv(out / "comparison_all_test.csv")
    with pd.option_context("display.width", 200, "display.max_colwidth", 40, "display.max_rows", 200):
        print("\nTest set, all models:\n", cmp.to_string())
    print(f"\nSaved to {out}")


def plot_history(hists, path):
    if not hists:
        return
    keys = list(hists)
    cols = 3
    rows = int(np.ceil(len(keys) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 2.8 * rows), squeeze=False)
    for ax, key in zip(axes.flat, keys):
        h = pd.DataFrame(hists[key])
        ax.plot(h["epoch"], h["train"], label="train")
        ax.plot(h["epoch"], h["val"], label="validation")
        best = h.loc[h["val"].idxmin()]
        ax.axvline(best["epoch"], color="grey", lw=0.8, ls="--")
        ax.set_title(key, fontsize=9); ax.set_xlabel("epoch"); ax.set_ylabel("loss")
        ax.legend(fontsize=7)
    for ax in list(axes.flat)[len(keys):]:
        ax.axis("off")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()

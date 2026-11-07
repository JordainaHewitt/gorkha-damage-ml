"""Load the NPC building survey, build the paper's feature matrix, and split.

Pipeline (paper Sec. "Building-damage prediction database" and "Method"):
  1. drop rows with missing damage grade
  2. treat invalid ages as missing; impute with the TRAINING-set mean
  3. one-hot encode slope / roof / position; keep material flags as-is
  4. attach shaking intensity (MSI) per VDC if available
  5. stratified 60/20/20 train/val/test split with a fixed seed
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config as C


def _parse_grade(s: pd.Series) -> pd.Series:
    """'Grade 3' -> 3. Already-numeric values pass through."""
    if pd.api.types.is_numeric_dtype(s):
        return s.astype("Int64")
    return s.astype(str).str.extract(r"(\d)", expand=False).astype("Int64")


def _expand_superstructure(df: pd.DataFrame) -> pd.DataFrame:
    """Two input formats exist:
      - portal original: 11 binary has_superstructure_* columns (multi-hot; a building can use several)
      - Kaggle mirror (arashnic, eq2015.csv): ONE 'superstructure' column naming a single material.
        It keeps only the first true flag in the original column order, so secondary materials
        (mostly timber and bamboo) are lost: timber 4.75% vs 25.87% in the paper.
    Convert the Kaggle form to the same 11 columns so the rest of the pipeline is identical."""
    if "superstructure" in df.columns and not all(c in df.columns for c in C.MATERIALS):
        for c in C.MATERIALS:
            df[c] = (df["superstructure"] == c).astype(np.uint8)
        print("[data] Kaggle format: expanded single 'superstructure' column to one-hot "
              "(secondary materials unavailable; see README)")
    return df


def load_raw(structure_csv: str | Path, msi_csv: str | Path | None = None, msi_col: str = "msi_vdc") -> pd.DataFrame:
    df = pd.read_csv(structure_csv, low_memory=False)
    df = _expand_superstructure(df)
    df[C.TARGET] = _parse_grade(df[C.TARGET])
    n0 = len(df)
    df = df.dropna(subset=[C.TARGET]).copy()
    df[C.TARGET] = df[C.TARGET].astype(int)
    print(f"[data] {n0:,} rows, dropped {n0 - len(df)} with missing damage grade")

    age = pd.to_numeric(df["age_building"], errors="coerce")
    age = age.where((age >= 0) & (age <= C.AGE_MAX_VALID))
    df["age_building"] = age
    print(f"[data] {age.isna().sum():,} buildings with missing/invalid age (imputed later)")

    if msi_csv is not None:
        msi = pd.read_csv(msi_csv)
        if "building_id" in msi.columns:   # per-building file from scripts/00b_join_msi.py
            msi = msi[["building_id", msi_col]].rename(columns={msi_col: "msi"})
            df = df.merge(msi, on="building_id", how="left")
        else:                              # per-unit file (synthetic test data)
            df = df.merge(msi[["vdcmun_id", "msi"]], on="vdcmun_id", how="left")
            msi_col = "msi"
        print(f"[data] MSI ({msi_col}) attached; {df['msi'].isna().sum():,} buildings without a match")
    return df


def build_features(df: pd.DataFrame, setting: str = "full", use_msi: bool = True) -> pd.DataFrame:
    """Return the model input matrix X (no imputation yet)."""
    if setting == "basic":
        X = df[C.BASIC].copy()
    elif setting == "full":
        cats = pd.get_dummies(df[C.CATEGORICAL].astype("category"), prefix=C.CATEGORICAL, dtype=np.uint8)
        X = pd.concat([df[C.NUMERIC], df[C.MATERIALS].astype(np.uint8), cats], axis=1)
    else:
        raise ValueError(f"unknown setting {setting!r}")
    if use_msi:
        if "msi" not in df:
            raise KeyError("use_msi=True but no 'msi' column. Pass --msi or --no-msi.")
        X["msi"] = df["msi"]
    X.columns = [re.sub(r"[^0-9a-zA-Z_]+", "_", c) for c in X.columns]  # safe for XGBoost later
    return X


def make_target(df: pd.DataFrame, scale: str = "5dg") -> np.ndarray:
    y = df[C.TARGET].to_numpy()
    if scale == "tl":
        y = np.vectorize(C.TL_MAP.get)(y)
    return y


def split_indices(y5: np.ndarray, out: str | Path | None = None, seed: int = C.SEED) -> dict:
    """Stratified 60/20/20 split on the 5-grade label. Saved so every model uses the same rows."""
    idx = np.arange(len(y5))
    tr, tmp = train_test_split(idx, test_size=0.4, stratify=y5, random_state=seed)
    va, te = train_test_split(tmp, test_size=0.5, stratify=y5[tmp], random_state=seed)
    split = {"train": tr.tolist(), "val": va.tolist(), "test": te.tolist()}
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(split))
    return split


def load_or_make_split(y5: np.ndarray, path: str | Path) -> dict:
    p = Path(path)
    if p.exists():
        split = json.loads(p.read_text())
        if sum(len(v) for v in split.values()) == len(y5):
            return split
        print("[data] saved split does not match data size; regenerating")
    return split_indices(y5, out=p)


def impute(X_train: pd.DataFrame, *others: pd.DataFrame):
    """Fill NaNs with training-set column means (fit on train only to avoid leakage)."""
    means = X_train.mean(numeric_only=True)
    return [X_train.fillna(means)] + [X.fillna(means) for X in others]

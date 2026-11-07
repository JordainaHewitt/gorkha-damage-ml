"""Metrics used by the paper plus two ordinal-aware ones.

Paper convention (Sec. "Method"): for regression models, accuracy and the
confusion matrix use the prediction rounded to the nearest grade, but MAE and
MSE use the UNROUNDED prediction. `evaluate` reproduces that.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix


def to_labels(pred: np.ndarray, k: int) -> np.ndarray:
    return np.clip(np.rint(pred), 1, k).astype(int)


def evaluate(y_true: np.ndarray, pred: np.ndarray, k: int) -> dict:
    """y_true: integer grades 1..k. pred: integer labels or continuous scores."""
    lab = to_labels(pred, k)
    labels = list(range(1, k + 1))
    cm = confusion_matrix(y_true, lab, labels=labels)
    recall = cm.diagonal() / cm.sum(axis=1).clip(min=1)
    precision = cm.diagonal() / cm.sum(axis=0).clip(min=1)
    err = pred.astype(float) - y_true
    return {
        "accuracy": accuracy_score(y_true, lab),
        "mae": float(np.mean(np.abs(err))),
        "mse": float(np.mean(err ** 2)),
        "within_1": float(np.mean(np.abs(lab - y_true) <= 1)),
        "qwk": cohen_kappa_score(y_true, lab, weights="quadratic"),
        "recall": recall.round(3).tolist(),
        "precision": precision.round(3).tolist(),
        "confusion": cm.tolist(),
    }


def row_normalize(cm) -> np.ndarray:
    cm = np.asarray(cm, dtype=float)
    return cm / cm.sum(axis=1, keepdims=True).clip(min=1)

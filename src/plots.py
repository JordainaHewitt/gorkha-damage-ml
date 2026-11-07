"""Row-normalized confusion matrices in the style of the paper's Figures 4-9."""
from __future__ import annotations

import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .metrics import row_normalize


def plot_confusions(confusions: dict, path, cols: int = 4):
    n = len(confusions)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.0 * rows), squeeze=False)
    for ax, (title, cm) in zip(axes.flat, confusions.items()):
        m = row_normalize(cm)
        k = m.shape[0]
        ticks = ["S", "M", "H"] if k == 3 else [str(i) for i in range(1, k + 1)]
        ax.imshow(m, cmap="Blues", vmin=0, vmax=1)
        for i in range(k):
            for j in range(k):
                ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if m[i, j] > 0.5 else "black")
        ax.set_xticks(range(k), ticks); ax.set_yticks(range(k), ticks)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title, fontsize=9)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_scatter_plots(
    y_true: np.ndarray, y_pred: np.ndarray, out_dir: Path, target_names: list[str]
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, target in enumerate(target_names):
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(y_true[:, idx], y_pred[:, idx], alpha=0.7)
        min_val = float(min(np.min(y_true[:, idx]), np.min(y_pred[:, idx])))
        max_val = float(max(np.max(y_true[:, idx]), np.max(y_pred[:, idx])))
        ax.plot([min_val, max_val], [min_val, max_val], linestyle="--")
        ax.set_title(f"Predicted vs True: {target}")
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")
        fig.tight_layout()
        fig.savefig(out_dir / f"scatter_{target}.png", dpi=150)
        plt.close(fig)


def save_residual_histograms(
    y_true: np.ndarray, y_pred: np.ndarray, out_dir: Path, target_names: list[str]
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    residual = y_pred - y_true
    for idx, target in enumerate(target_names):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(residual[:, idx], bins=20)
        ax.set_title(f"Residual Histogram: {target}")
        ax.set_xlabel("Prediction - True")
        ax.set_ylabel("Count")
        fig.tight_layout()
        fig.savefig(out_dir / f"residual_hist_{target}.png", dpi=150)
        plt.close(fig)


def save_fold_metric_bar(
    fold_scores: list[dict[str, float]], out_path: Path, metric_key: str
) -> None:
    folds = list(range(len(fold_scores)))
    values = [score[metric_key] for score in fold_scores]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(folds, values)
    ax.set_title(f"Fold-wise {metric_key}")
    ax.set_xlabel("Fold")
    ax.set_ylabel(metric_key)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

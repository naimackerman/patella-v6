"""Visualization utilities for evaluation results."""

from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    save_path: str,
    title: str = "Confusion Matrix",
    normalize: bool = True,
):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype(np.float64) / cm.sum(axis=1, keepdims=True)
        fmt = ".2f"
    else:
        fmt = "d"

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curves(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    class_names: List[str],
    save_path: str,
    title: str = "ROC Curves (One-vs-Rest)",
):
    """Plot and save multi-class ROC curves."""
    fig, ax = plt.subplots(figsize=(8, 6))
    n_classes = len(class_names)

    for i in range(n_classes):
        y_binary = (y_true == i).astype(int)
        if y_probs.ndim == 2:
            y_score = y_probs[:, i]
        else:
            y_score = (y_true == i).astype(float)
        fpr, tpr, _ = roc_curve(y_binary, y_score)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{class_names[i]} (AUC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_class_distribution(
    labels: np.ndarray,
    title: str,
    save_path: str,
    class_names: Optional[List[str]] = None,
):
    """Plot and save class distribution bar chart."""
    unique, counts = np.unique(labels, return_counts=True)
    names = class_names if class_names else [str(u) for u in unique]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, counts, color=sns.color_palette("viridis", len(names)))
    for bar, count in zip(bars, counts):
        pct = 100 * count / counts.sum()
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                f"{count}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

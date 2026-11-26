"""Publication-style visualizations for classification and regression evaluation."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns

if (
    "MPLBACKEND" not in os.environ
    and "ipykernel" not in sys.modules
    and "matplotlib_inline" not in sys.modules
):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from predictive_modeling.data import TaskType  # noqa: E402

sns.set_theme(style="whitegrid", context="notebook")
PALETTE = ["#2563EB", "#059669", "#D97706", "#DC2626", "#7C3AED", "#0891B2"]


def plot_target_distribution(
    y: pd.Series,
    task_type: TaskType,
    title: str,
    output_path: Path,
    class_names: list[str] | None = None,
) -> Path:
    """Plot class balance bar chart or continuous target histogram + boxplot."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if task_type == "classification":
        fig, ax = plt.subplots(figsize=(7, 4.5))
        counts = y.value_counts().sort_index()
        labels = (
            [class_names[int(idx)] if int(idx) < len(class_names) else str(idx) for idx in counts.index]
            if class_names
            else [str(idx) for idx in counts.index]
        )
        bars = ax.bar(labels, counts.values, color=PALETTE[: len(counts)], width=0.55)
        for bar, cnt in zip(bars, counts.values):
            pct = cnt / max(len(y), 1) * 100.0
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(counts.values) * 0.02,
                f"{cnt} ({pct:.1f}%)",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )
        ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
        ax.set_ylabel("Sample Count")
        ax.set_ylim(0, max(counts.values) * 1.18)
    else:
        fig, (ax_hist, ax_box) = plt.subplots(
            2, 1, figsize=(8, 5), gridspec_kw={"height_ratios": [3.5, 1]}, sharex=True
        )
        sns.histplot(y, kde=True, bins=25, color=PALETTE[0], ax=ax_hist)
        ax_hist.axvline(y.mean(), color=PALETTE[3], linestyle="--", label=f"Mean: {y.mean():.2f}")
        ax_hist.axvline(y.median(), color=PALETTE[1], linestyle=":", label=f"Median: {y.median():.2f}")
        ax_hist.set_title(title, fontsize=13, fontweight="bold", pad=10)
        ax_hist.set_ylabel("Count")
        ax_hist.legend()
        sns.boxplot(x=y, color=PALETTE[0], ax=ax_box)
        ax_box.set_xlabel(y.name or "Target")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_model_comparison(
    comparison_df: pd.DataFrame,
    metric_col: str,
    metric_label: str,
    title: str,
    output_path: Path,
) -> Path:
    """Plot Train vs Validation vs Test scores across models to expose overfitting."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    melted = comparison_df.melt(
        id_vars=["model"],
        value_vars=[f"train_{metric_col}", f"val_{metric_col}", f"test_{metric_col}"],
        var_name="split",
        value_name="score",
    )
    split_map = {
        f"train_{metric_col}": "Train",
        f"val_{metric_col}": "Validation",
        f"test_{metric_col}": "Test",
    }
    melted["split"] = melted["split"].map(split_map)

    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    sns.barplot(
        data=melted,
        x="model",
        y="score",
        hue="split",
        palette=[PALETTE[0], PALETTE[1], PALETTE[2]],
        ax=ax,
    )
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("")
    ax.set_ylabel(metric_label)
    ax.tick_params(axis="x", rotation=18)
    ax.legend(title="Split")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_confusion_matrices(
    model_results: list[dict[str, Any]],
    class_names: list[str] | None,
    output_path: Path,
) -> Path:
    """Plot side-by-side annotated confusion matrices (counts and row percentages)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    candidates = [m for m in model_results if not m["is_baseline"] or m["key"] != "naive_baseline"]
    n_models = len(candidates)
    cols = min(2, n_models)
    rows = int(np.ceil(n_models / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(5.8 * cols, 4.8 * rows))
    axes_flat = np.atleast_1d(axes).ravel()

    for ax, res in zip(axes_flat, candidates):
        cm = np.asarray(res["test"]["confusion_matrix"])
        row_sums = cm.sum(axis=1, keepdims=True)
        pct = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0) * 100.0
        annot = np.array(
            [
                [f"{cm[i, j]}\n({pct[i, j]:.1f}%)" for j in range(cm.shape[1])]
                for i in range(cm.shape[0])
            ]
        )
        tick_labels = class_names if class_names and len(class_names) == cm.shape[0] else "auto"
        sns.heatmap(
            cm,
            annot=annot,
            fmt="",
            cmap="Blues",
            cbar=False,
            xticklabels=tick_labels,
            yticklabels=tick_labels,
            ax=ax,
        )
        acc = res["test"]["accuracy"]
        f1 = res["test"]["f1"]
        ax.set_title(f"{res['display_name']}\nAcc={acc:.3f} | F1={f1:.3f}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")

    for unused_ax in axes_flat[len(candidates) :]:
        unused_ax.axis("off")

    fig.suptitle("Test-Set Confusion Matrices", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_roc_curves(
    model_results: list[dict[str, Any]],
    best_model_key: str,
    output_path: Path,
) -> Path | None:
    """Plot ROC curves with AUC scores and Youden's J threshold marker on the best model."""
    roc_models = [m for m in model_results if m["test"].get("roc_curve") is not None]
    if not roc_models:
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.8, 6.0))
    for idx, res in enumerate(roc_models):
        curve = res["test"]["roc_curve"]
        auc_val = res["test"]["roc_auc"]
        lw = 2.6 if res["key"] == best_model_key else 1.8
        color = PALETTE[idx % len(PALETTE)]
        ax.plot(
            curve["fpr"],
            curve["tpr"],
            linewidth=lw,
            color=color,
            label=f"{res['display_name']} (AUC = {auc_val:.4f})",
        )
        if res["key"] == best_model_key:
            ax.scatter(
                [curve["optimal_fpr"]],
                [curve["optimal_tpr"]],
                color=color,
                s=80,
                zorder=5,
                edgecolor="black",
                label=f"Best Youden J threshold = {curve['optimal_threshold']:.3f}",
            )

    ax.plot([0, 1], [0, 1], linestyle="--", color="#6B7280", linewidth=1.3, label="Random Chance (AUC = 0.50)")
    ax.set_title("ROC Curves Comparison (Test Set)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Recall / Sensitivity)")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_precision_recall_curves(
    model_results: list[dict[str, Any]],
    y_test: pd.Series,
    output_path: Path,
) -> Path | None:
    """Plot Precision-Recall curves with Average Precision (PR-AUC)."""
    pr_models = [m for m in model_results if m["test"].get("pr_curve") is not None]
    if not pr_models:
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.8, 5.8))
    prevalence = float(np.mean(y_test))
    for idx, res in enumerate(pr_models):
        curve = res["test"]["pr_curve"]
        ap = res["test"]["pr_auc"]
        ax.plot(
            curve["recall"],
            curve["precision"],
            linewidth=2.0,
            color=PALETTE[idx % len(PALETTE)],
            label=f"{res['display_name']} (PR-AUC = {ap:.4f})",
        )

    ax.axhline(prevalence, linestyle="--", color="#6B7280", label=f"Positive Prevalence = {prevalence:.3f}")
    ax.set_title("Precision-Recall Curves (Test Set)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_regression_diagnostics(
    model_results: list[dict[str, Any]],
    y_test: pd.Series,
    output_path: Path,
) -> Path:
    """Plot Actual vs Predicted and Residual diagnostics for regression models."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    candidates = [m for m in model_results if m["key"] != "naive_baseline"]
    n_models = len(candidates)

    fig, axes = plt.subplots(2, n_models, figsize=(4.8 * n_models, 8.2))
    if n_models == 1:
        axes = np.asarray(axes).reshape(2, 1)

    y_arr = np.asarray(y_test, dtype=float)
    min_val, max_val = float(np.min(y_arr)), float(np.max(y_arr))

    for col_idx, res in enumerate(candidates):
        y_pred = res["test"]["y_pred"]
        residuals = res["test"]["residuals"]
        r2 = res["test"]["r2"]
        rmse = res["test"]["rmse"]

        ax_top = axes[0, col_idx]
        ax_top.scatter(y_arr, y_pred, alpha=0.72, color=PALETTE[col_idx % len(PALETTE)], edgecolor="white", s=45)
        ax_top.plot([min_val, max_val], [min_val, max_val], "--", color="#111827", linewidth=1.4)
        ax_top.set_title(f"{res['display_name']}\nR2={r2:.3f} | RMSE={rmse:.2f}", fontsize=10.5, fontweight="bold")
        ax_top.set_xlabel("Actual Target")
        ax_top.set_ylabel("Predicted Target")

        ax_bot = axes[1, col_idx]
        ax_bot.scatter(y_pred, residuals, alpha=0.72, color=PALETTE[col_idx % len(PALETTE)], edgecolor="white", s=45)
        ax_bot.axhline(0.0, linestyle="--", color="#DC2626", linewidth=1.4)
        ax_bot.set_title("Residuals vs Predicted", fontsize=10)
        ax_bot.set_xlabel("Predicted Target")
        ax_bot.set_ylabel("Residual (Actual - Predicted)")

    fig.suptitle("Regression Diagnostic Plots (Test Set)", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_feature_importances(
    importances: list[tuple[str, float]],
    title: str,
    output_path: Path,
) -> Path | None:
    """Plot horizontal bar chart of top model feature importances."""
    if not importances:
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    names = [item[0] for item in reversed(importances)]
    values = [item[1] for item in reversed(importances)]

    fig, ax = plt.subplots(figsize=(8.4, max(4.2, 0.42 * len(names))))
    ax.barh(names, values, color=PALETTE[0], height=0.65)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Importance / Absolute Coefficient Magnitude")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path

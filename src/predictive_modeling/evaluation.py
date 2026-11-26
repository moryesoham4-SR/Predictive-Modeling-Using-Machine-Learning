"""Evaluation metrics, cross-validation, bootstrap confidence intervals, and slice analysis."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from predictive_modeling.data import TaskType


def _extract_positive_scores(model: Pipeline, X: pd.DataFrame) -> np.ndarray | None:
    """Return 1D positive-class scores for binary classification when available."""
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        if probs.ndim == 2 and probs.shape[1] == 2:
            return probs[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        if np.ndim(scores) == 1:
            return np.asarray(scores, dtype=float)
    return None


def evaluate_classification(
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    labels: list[int] | None = None,
) -> dict[str, Any]:
    """Evaluate classification predictions, confusion matrix, ROC/PR curves, and AUC."""
    y_true = np.asarray(y)
    y_pred = np.asarray(model.predict(X))
    unique_classes = sorted(np.unique(y_true).tolist()) if labels is None else labels
    is_binary = len(unique_classes) == 2

    cm = confusion_matrix(y_true, y_pred, labels=unique_classes)
    avg_mode = "binary" if is_binary else "weighted"

    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=avg_mode, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=avg_mode, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average=avg_mode, zero_division=0)),
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(y_true, y_pred, zero_division=0),
        "roc_auc": None,
        "pr_auc": None,
        "specificity": None,
        "y_pred": y_pred,
        "y_score": None,
        "roc_curve": None,
        "pr_curve": None,
    }

    if is_binary and cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics["tn"] = int(tn)
        metrics["fp"] = int(fp)
        metrics["fn"] = int(fn)
        metrics["tp"] = int(tp)
        metrics["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

        y_score = _extract_positive_scores(model, X)
        metrics["y_score"] = y_score
        if y_score is not None and len(np.unique(y_true)) == 2:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
            metrics["pr_auc"] = float(average_precision_score(y_true, y_score))
            fpr, tpr, roc_thresholds = roc_curve(y_true, y_score)
            youden_idx = int(np.argmax(tpr - fpr))
            metrics["roc_curve"] = {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "thresholds": roc_thresholds.tolist(),
                "optimal_threshold": float(roc_thresholds[youden_idx]),
                "optimal_fpr": float(fpr[youden_idx]),
                "optimal_tpr": float(tpr[youden_idx]),
            }
            pr_prec, pr_rec, _ = precision_recall_curve(y_true, y_score)
            metrics["pr_curve"] = {
                "precision": pr_prec.tolist(),
                "recall": pr_rec.tolist(),
            }
    elif hasattr(model, "predict_proba"):
        try:
            probs = model.predict_proba(X)
            metrics["roc_auc"] = float(
                roc_auc_score(y_true, probs, multi_class="ovr", average="weighted")
            )
        except ValueError:
            metrics["roc_auc"] = None

    return metrics


def evaluate_regression(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
    """Evaluate regression predictions with MAE, MSE, RMSE, and R2."""
    y_true = np.asarray(y, dtype=float)
    y_pred = np.asarray(model.predict(X), dtype=float)
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(y_true, y_pred)),
        "y_pred": y_pred,
        "residuals": y_true - y_pred,
    }


def cross_validate_pipeline(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    task_type: TaskType,
    cv: int = 5,
    random_state: int = 42,
) -> dict[str, dict[str, float]]:
    """Run k-fold cross-validation on an unfitted pipeline clone."""
    if task_type == "classification":
        n_splits = max(2, min(cv, int(y.value_counts().min())))
        cv_splitter: Any = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state
        )
        is_binary = y.nunique() == 2
        scoring = {
            "accuracy": "accuracy",
            "f1": "f1" if is_binary else "f1_weighted",
        }
        if is_binary:
            scoring["roc_auc"] = "roc_auc"
    else:
        n_splits = max(2, min(cv, len(X)))
        cv_splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        scoring = {
            "r2": "r2",
            "neg_rmse": "neg_root_mean_squared_error",
            "neg_mae": "neg_mean_absolute_error",
        }

    raw = cross_validate(
        clone(pipeline),
        X,
        y,
        cv=cv_splitter,
        scoring=scoring,
        n_jobs=1,
        error_score="raise",
    )

    summary: dict[str, dict[str, float]] = {}
    for metric_key in scoring:
        vals = raw[f"test_{metric_key}"]
        clean_name = metric_key.replace("neg_", "")
        sign = -1.0 if metric_key.startswith("neg_") else 1.0
        adjusted = vals * sign
        summary[clean_name] = {
            "mean": float(np.mean(adjusted)),
            "std": float(np.std(adjusted)),
        }
    return summary


def bootstrap_metric_ci(
    y_true: pd.Series | np.ndarray,
    y_estimate: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    random_state: int = 42,
) -> dict[str, float]:
    """Compute a nonparametric bootstrap confidence interval for a metric."""
    y_arr = np.asarray(y_true)
    est_arr = np.asarray(y_estimate)
    rng = np.random.default_rng(random_state)
    n = len(y_arr)
    samples: list[float] = []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        try:
            score = float(metric_fn(y_arr[idx], est_arr[idx]))
        except ValueError:
            continue
        if np.isfinite(score):
            samples.append(score)

    point_estimate = float(metric_fn(y_arr, est_arr))
    if not samples:
        return {"estimate": point_estimate, "ci_low": point_estimate, "ci_high": point_estimate}

    alpha = (1.0 - ci) / 2.0
    return {
        "estimate": point_estimate,
        "ci_low": float(np.quantile(samples, alpha)),
        "ci_high": float(np.quantile(samples, 1.0 - alpha)),
    }


def slice_error_analysis(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    task_type: TaskType,
) -> list[dict[str, Any]]:
    """Evaluate model performance across subpopulations of the highest-variance feature."""
    numeric_cols = X_test.select_dtypes(include=[np.number]).columns.tolist()
    y_pred = np.asarray(model.predict(X_test))
    y_true = np.asarray(y_test)

    if numeric_cols:
        variances = X_test[numeric_cols].var(numeric_only=True).dropna()
        slice_col = str(variances.idxmax()) if not variances.empty else numeric_cols[0]
        series = X_test[slice_col]
        try:
            buckets = pd.qcut(series, q=3, labels=["Low", "Medium", "High"], duplicates="drop")
        except ValueError:
            buckets = pd.Series(["All"] * len(X_test), index=X_test.index)
    else:
        cat_col = str(X_test.columns[0])
        slice_col = cat_col
        top_vals = X_test[cat_col].value_counts().head(3).index
        buckets = X_test[cat_col].where(X_test[cat_col].isin(top_vals), other="Other").astype(str)

    results: list[dict[str, Any]] = []
    for label in pd.unique(buckets.dropna()):
        mask = (buckets == label).to_numpy()
        if int(mask.sum()) == 0:
            continue
        row: dict[str, Any] = {
            "slice_feature": slice_col,
            "slice_group": str(label),
            "n_samples": int(mask.sum()),
        }
        if task_type == "classification":
            row["accuracy"] = float(accuracy_score(y_true[mask], y_pred[mask]))
            row["error_rate"] = float(1.0 - row["accuracy"])
        else:
            row["mae"] = float(mean_absolute_error(y_true[mask], y_pred[mask]))
            row["rmse"] = float(np.sqrt(mean_squared_error(y_true[mask], y_pred[mask])))
        results.append(row)

    return results


def extract_feature_importances(
    fitted_pipeline: Pipeline,
    top_n: int = 12,
) -> list[tuple[str, float]]:
    """Extract readable top feature importances or absolute linear coefficients."""
    preprocessor = fitted_pipeline.named_steps["preprocessor"]
    estimator = fitted_pipeline.named_steps["model"]
    raw_names = preprocessor.get_feature_names_out()
    clean_names = [
        name.split("__", 1)[1] if "__" in name else name for name in raw_names
    ]

    if hasattr(estimator, "feature_importances_"):
        scores = np.asarray(estimator.feature_importances_, dtype=float)
    elif hasattr(estimator, "coef_"):
        coef = np.asarray(estimator.coef_, dtype=float)
        scores = np.mean(np.abs(coef), axis=0) if coef.ndim > 1 else np.abs(coef)
    else:
        return []

    pairs = sorted(zip(clean_names, scores.tolist()), key=lambda item: item[1], reverse=True)
    return pairs[:top_n]

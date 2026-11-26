"""End-to-end supervised learning experiment orchestration and inference."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.pipeline import Pipeline

from predictive_modeling.data import DatasetBundle, sanitize_feature_frame, split_dataset
from predictive_modeling.evaluation import (
    bootstrap_metric_ci,
    cross_validate_pipeline,
    evaluate_classification,
    evaluate_regression,
    extract_feature_importances,
    slice_error_analysis,
)
from predictive_modeling.models import build_model_specs, build_pipeline, tune_random_forest
from predictive_modeling.preprocessing import analyze_data_quality, build_preprocessor
from predictive_modeling.visualization import (
    plot_confusion_matrices,
    plot_feature_importances,
    plot_model_comparison,
    plot_precision_recall_curves,
    plot_regression_diagnostics,
    plot_roc_curves,
    plot_target_distribution,
)


@dataclass
class ExperimentResult:
    """Outputs and metrics from a complete predictive modeling experiment."""

    dataset_name: str
    task_type: str
    output_dir: Path
    best_model_key: str
    best_model_name: str
    best_model_path: Path
    comparison_df: pd.DataFrame
    model_results: list[dict[str, Any]]
    bootstrap_cis: dict[str, dict[str, float]]
    slice_analysis: list[dict[str, Any]]
    feature_importances: list[tuple[str, float]]
    report_path: Path
    metrics_path: Path
    plot_paths: dict[str, Path]


def _serialize_split_metrics(split_metrics: dict[str, Any]) -> dict[str, Any]:
    """Strip large arrays so JSON reports stay concise and readable."""
    skip_keys = {"y_pred", "y_score", "residuals", "roc_curve", "pr_curve", "classification_report"}
    cleaned: dict[str, Any] = {}
    for k, v in split_metrics.items():
        if k in skip_keys:
            continue
        if isinstance(v, np.ndarray):
            cleaned[k] = v.tolist()
        elif isinstance(v, (np.floating, np.integer)):
            cleaned[k] = v.item()
        else:
            cleaned[k] = v
    return cleaned


def _selection_score(res: dict[str, Any], task_type: str) -> tuple[float, float]:
    """Return validation ranking tuple (higher is better) without touching test data."""
    val = res["val"]
    if task_type == "classification":
        primary = val["roc_auc"] if val.get("roc_auc") is not None else val["f1"]
        return (float(primary), float(val["accuracy"]))
    return (float(val["r2"]), -float(val["rmse"]))


def _evaluate_model_entry(
    key: str,
    display_name: str,
    is_baseline: bool,
    notes: str,
    pipeline: Pipeline,
    split: Any,
    task_type: str,
    random_state: int,
    tuned_params: dict[str, Any] | None = None,
    tuning_cv_score: float | None = None,
    already_fitted: bool = False,
    preset_train_seconds: float | None = None,
) -> tuple[dict[str, Any], Pipeline]:
    """Fit (if needed), time, cross-validate, and evaluate one model across all splits."""
    start_train = time.perf_counter()
    if not already_fitted:
        pipeline.fit(split.X_train, split.y_train)
    train_seconds = (
        preset_train_seconds
        if preset_train_seconds is not None
        else (time.perf_counter() - start_train)
    )

    start_pred = time.perf_counter()
    _ = pipeline.predict(split.X_test)
    pred_seconds = time.perf_counter() - start_pred
    infer_ms_per_1k = (pred_seconds / max(len(split.X_test), 1)) * 1000.0 * 1000.0

    eval_fn = evaluate_classification if task_type == "classification" else evaluate_regression
    train_metrics = eval_fn(pipeline, split.X_train, split.y_train)
    val_metrics = eval_fn(pipeline, split.X_val, split.y_val)
    test_metrics = eval_fn(pipeline, split.X_test, split.y_test)
    cv_metrics = cross_validate_pipeline(
        pipeline, split.X_train, split.y_train, task_type=task_type, cv=5, random_state=random_state
    )

    entry: dict[str, Any] = {
        "key": key,
        "display_name": display_name,
        "is_baseline": is_baseline,
        "notes": notes,
        "train_time_sec": round(train_seconds, 4),
        "infer_ms_per_1k": round(infer_ms_per_1k, 4),
        "cv": cv_metrics,
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
    }
    if tuned_params is not None:
        entry["tuned_params"] = tuned_params
        entry["tuning_cv_score"] = round(float(tuning_cv_score or 0.0), 4)
    return entry, pipeline


def _build_comparison_dataframe(
    model_results: list[dict[str, Any]],
    task_type: str,
) -> pd.DataFrame:
    """Create a clean comparison table across train, validation, CV, and test splits."""
    rows: list[dict[str, Any]] = []
    for res in model_results:
        if task_type == "classification":
            rows.append(
                {
                    "key": res["key"],
                    "model": res["display_name"],
                    "train_accuracy": res["train"]["accuracy"],
                    "val_accuracy": res["val"]["accuracy"],
                    "test_accuracy": res["test"]["accuracy"],
                    "val_f1": res["val"]["f1"],
                    "test_f1": res["test"]["f1"],
                    "val_roc_auc": res["val"]["roc_auc"],
                    "test_roc_auc": res["test"]["roc_auc"],
                    "cv_accuracy_mean": res["cv"]["accuracy"]["mean"],
                    "cv_accuracy_std": res["cv"]["accuracy"]["std"],
                    "overfit_gap": res["train"]["accuracy"] - res["val"]["accuracy"],
                    "train_time_sec": res["train_time_sec"],
                }
            )
        else:
            rows.append(
                {
                    "key": res["key"],
                    "model": res["display_name"],
                    "train_r2": res["train"]["r2"],
                    "val_r2": res["val"]["r2"],
                    "test_r2": res["test"]["r2"],
                    "val_rmse": res["val"]["rmse"],
                    "test_rmse": res["test"]["rmse"],
                    "val_mae": res["val"]["mae"],
                    "test_mae": res["test"]["mae"],
                    "cv_r2_mean": res["cv"]["r2"]["mean"],
                    "cv_r2_std": res["cv"]["r2"]["std"],
                    "overfit_gap": res["train"]["r2"] - res["val"]["r2"],
                    "train_time_sec": res["train_time_sec"],
                }
            )
    return pd.DataFrame(rows)


def _compute_bootstrap_cis(
    best_entry: dict[str, Any],
    y_test: pd.Series,
    task_type: str,
    random_state: int,
) -> dict[str, dict[str, float]]:
    """Calculate 95% bootstrap confidence intervals on the held-out test set."""
    y_pred = best_entry["test"]["y_pred"]
    if task_type == "classification":
        cis = {
            "accuracy": bootstrap_metric_ci(
                y_test, y_pred, accuracy_score, random_state=random_state
            ),
            "f1": bootstrap_metric_ci(
                y_test,
                y_pred,
                lambda yt, yp: f1_score(
                    yt, yp, average="binary" if len(np.unique(yt)) == 2 else "weighted", zero_division=0
                ),
                random_state=random_state,
            ),
        }
        if best_entry["test"].get("y_score") is not None:
            cis["roc_auc"] = bootstrap_metric_ci(
                y_test,
                best_entry["test"]["y_score"],
                roc_auc_score,
                random_state=random_state,
            )
        return cis

    return {
        "r2": bootstrap_metric_ci(y_test, y_pred, r2_score, random_state=random_state),
        "rmse": bootstrap_metric_ci(
            y_test,
            y_pred,
            lambda yt, yp: float(np.sqrt(mean_squared_error(yt, yp))),
            random_state=random_state,
        ),
        "mae": bootstrap_metric_ci(y_test, y_pred, mean_absolute_error, random_state=random_state),
    }


def _write_markdown_report(
    bundle: DatasetBundle,
    quality: dict[str, Any],
    prep_meta: dict[str, Any],
    split: Any,
    comparison_df: pd.DataFrame,
    best_entry: dict[str, Any],
    bootstrap_cis: dict[str, dict[str, float]],
    slice_analysis: list[dict[str, Any]],
    feature_importances: list[tuple[str, float]],
    report_path: Path,
) -> Path:
    """Write a comprehensive human-readable evaluation report in Markdown."""
    lines: list[str] = [
        f"# Predictive Modeling Report: {bundle.name}",
        "",
        f"- **Task Type:** `{bundle.task_type}`",
        f"- **Description:** {bundle.description}",
        f"- **Target Column:** `{bundle.target_col}`",
        f"- **Dataset Shape:** {quality['n_rows']} rows x {quality['n_features']} features",
        f"- **Data Cleaning Summary:** {json.dumps(bundle.cleaning_summary)}",
        f"- **Missing Value Cells:** {quality['total_missing_cells']} | **Duplicate Rows:** {quality['duplicate_rows']}",
        f"- **Strict Split Sizes (Before Preprocessing):** Train={len(split.X_train)}, Val={len(split.X_val)}, Test={len(split.X_test)}",
        f"- **Preprocessing Summary:** {len(prep_meta['numeric_columns'])} numeric features, {len(prep_meta['categorical_columns'])} categorical features",
        "",
        "## Model Comparison",
        "",
        comparison_df.round(4).to_markdown(index=False),
        "",
        f"## Selected Best Model (by Validation Set): {best_entry['display_name']}",
        "",
    ]

    if "tuned_params" in best_entry:
        lines.append(f"- **Tuned Hyperparameters:** `{json.dumps(best_entry['tuned_params'])}`")
    lines.append(f"- **Train Time:** {best_entry['train_time_sec']}s | **Inference Latency:** {best_entry['infer_ms_per_1k']} ms / 1k rows")
    lines.append("")
    lines.append("### Held-Out Test Performance & 95% Bootstrap Confidence Intervals")
    lines.append("")
    for metric_name, ci_vals in bootstrap_cis.items():
        lines.append(
            f"- **{metric_name.upper()}:** {ci_vals['estimate']:.4f} "
            f"(95% CI: [{ci_vals['ci_low']:.4f}, {ci_vals['ci_high']:.4f}])"
        )

    if bundle.task_type == "classification" and "tn" in best_entry["test"]:
        t = best_entry["test"]
        lines.extend(
            [
                "",
                "### Confusion Matrix Breakdown (Test Set)",
                "",
                f"- True Negatives (TN): {t['tn']} | False Positives (FP): {t['fp']}",
                f"- False Negatives (FN): {t['fn']} | True Positives (TP): {t['tp']}",
                f"- Precision: {t['precision']:.4f} | Recall (Sensitivity): {t['recall']:.4f} | Specificity: {t['specificity']:.4f}",
            ]
        )

    if slice_analysis:
        lines.extend(["", "### Slice-Based Error Analysis (Test Set)", ""])
        lines.append(pd.DataFrame(slice_analysis).round(4).to_markdown(index=False))

    if feature_importances:
        lines.extend(["", "### Top Feature Importances", ""])
        imp_df = pd.DataFrame(feature_importances, columns=["feature", "importance"])
        lines.append(imp_df.round(4).to_markdown(index=False))

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def run_experiment(
    bundle: DatasetBundle,
    output_dir: str | Path,
    random_state: int = 42,
    tune_best: bool = True,
) -> ExperimentResult:
    """Execute the full supervised ML workflow on a DatasetBundle."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    quality = analyze_data_quality(bundle.df, bundle.target_col)
    X = bundle.df[bundle.feature_names]
    y = bundle.df[bundle.target_col]

    # CRITICAL ML PRACTICE: split BEFORE fitting any preprocessing transformations.
    split = split_dataset(X, y, task_type=bundle.task_type, random_state=random_state)
    preprocessor, prep_meta = build_preprocessor(split.X_train)

    model_results: list[dict[str, Any]] = []
    fitted_pipelines: dict[str, Pipeline] = {}

    for spec in build_model_specs(bundle.task_type, random_state=random_state):
        pipe = build_pipeline(preprocessor, spec.estimator)
        entry, fitted_pipe = _evaluate_model_entry(
            key=spec.key,
            display_name=spec.display_name,
            is_baseline=spec.is_baseline,
            notes=spec.notes,
            pipeline=pipe,
            split=split,
            task_type=bundle.task_type,
            random_state=random_state,
        )
        model_results.append(entry)
        fitted_pipelines[spec.key] = fitted_pipe

    if tune_best:
        tuned_pipe, best_params, best_cv_score, refit_seconds = tune_random_forest(
            preprocessor, split.X_train, split.y_train, bundle.task_type, random_state=random_state
        )
        tuned_entry, _ = _evaluate_model_entry(
            key="random_forest_tuned",
            display_name="Random Forest (GridSearchCV)",
            is_baseline=False,
            notes="Cross-validated Random Forest hyperparameter tuning on training split.",
            pipeline=tuned_pipe,
            split=split,
            task_type=bundle.task_type,
            random_state=random_state,
            tuned_params=best_params,
            tuning_cv_score=best_cv_score,
            already_fitted=True,
            preset_train_seconds=refit_seconds,
        )
        model_results.append(tuned_entry)
        fitted_pipelines["random_forest_tuned"] = tuned_pipe

    ranked = sorted(
        model_results,
        key=lambda item: _selection_score(item, bundle.task_type),
        reverse=True,
    )
    best_entry = ranked[0]
    best_key = best_entry["key"]
    best_pipeline = fitted_pipelines[best_key]

    comparison_df = _build_comparison_dataframe(model_results, bundle.task_type)
    comparison_df.to_csv(out_dir / "model_comparison.csv", index=False)

    bootstrap_cis = _compute_bootstrap_cis(
        best_entry, split.y_test, bundle.task_type, random_state=random_state
    )
    slice_analysis = slice_error_analysis(
        best_pipeline, split.X_test, split.y_test, bundle.task_type
    )
    feature_importances = extract_feature_importances(best_pipeline, top_n=12)

    plot_paths: dict[str, Path] = {}
    plot_paths["target_distribution"] = plot_target_distribution(
        y,
        bundle.task_type,
        f"Target Distribution: {bundle.name}",
        out_dir / "target_distribution.png",
        class_names=bundle.target_names,
    )

    comp_metric = "accuracy" if bundle.task_type == "classification" else "r2"
    comp_label = "Accuracy" if bundle.task_type == "classification" else "R2 Score"
    plot_paths["model_comparison"] = plot_model_comparison(
        comparison_df,
        metric_col=comp_metric,
        metric_label=comp_label,
        title=f"Train vs Validation vs Test {comp_label}: {bundle.name}",
        output_path=out_dir / "model_comparison.png",
    )

    if bundle.task_type == "classification":
        plot_paths["confusion_matrices"] = plot_confusion_matrices(
            model_results, bundle.target_names, out_dir / "confusion_matrices.png"
        )
        roc_path = plot_roc_curves(model_results, best_key, out_dir / "roc_curves.png")
        if roc_path is not None:
            plot_paths["roc_curves"] = roc_path
        pr_path = plot_precision_recall_curves(
            model_results, split.y_test, out_dir / "precision_recall_curves.png"
        )
        if pr_path is not None:
            plot_paths["precision_recall_curves"] = pr_path
    else:
        plot_paths["regression_diagnostics"] = plot_regression_diagnostics(
            model_results, split.y_test, out_dir / "regression_diagnostics.png"
        )

    imp_path = plot_feature_importances(
        feature_importances,
        f"Top Feature Importances ({best_entry['display_name']})",
        out_dir / "feature_importances.png",
    )
    if imp_path is not None:
        plot_paths["feature_importances"] = imp_path

    best_model_path = out_dir / "best_model.joblib"
    joblib.dump(
        {
            "pipeline": best_pipeline,
            "dataset_name": bundle.name,
            "task_type": bundle.task_type,
            "target_col": bundle.target_col,
            "feature_names": bundle.feature_names,
            "target_names": bundle.target_names,
            "best_model_key": best_key,
            "best_model_name": best_entry["display_name"],
        },
        best_model_path,
    )

    metrics_payload = {
        "dataset_name": bundle.name,
        "task_type": bundle.task_type,
        "target_col": bundle.target_col,
        "target_names": bundle.target_names,
        "data_quality": quality,
        "cleaning_summary": bundle.cleaning_summary,
        "preprocessing": prep_meta,
        "split_sizes": {
            "train": len(split.X_train),
            "val": len(split.X_val),
            "test": len(split.X_test),
        },
        "best_model_key": best_key,
        "best_model_name": best_entry["display_name"],
        "bootstrap_cis_test": bootstrap_cis,
        "slice_analysis_test": slice_analysis,
        "feature_importances": [
            {"feature": name, "importance": float(score)}
            for name, score in feature_importances
        ],
        "models": [
            {
                "key": m["key"],
                "display_name": m["display_name"],
                "is_baseline": m["is_baseline"],
                "train_time_sec": m["train_time_sec"],
                "infer_ms_per_1k": m["infer_ms_per_1k"],
                "cv": m["cv"],
                "train": _serialize_split_metrics(m["train"]),
                "val": _serialize_split_metrics(m["val"]),
                "test": _serialize_split_metrics(m["test"]),
                **({"tuned_params": m["tuned_params"]} if "tuned_params" in m else {}),
            }
            for m in model_results
        ],
    }
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    report_path = _write_markdown_report(
        bundle=bundle,
        quality=quality,
        prep_meta=prep_meta,
        split=split,
        comparison_df=comparison_df,
        best_entry=best_entry,
        bootstrap_cis=bootstrap_cis,
        slice_analysis=slice_analysis,
        feature_importances=feature_importances,
        report_path=out_dir / "report.md",
    )

    return ExperimentResult(
        dataset_name=bundle.name,
        task_type=bundle.task_type,
        output_dir=out_dir,
        best_model_key=best_key,
        best_model_name=best_entry["display_name"],
        best_model_path=best_model_path,
        comparison_df=comparison_df,
        model_results=model_results,
        bootstrap_cis=bootstrap_cis,
        slice_analysis=slice_analysis,
        feature_importances=feature_importances,
        report_path=report_path,
        metrics_path=metrics_path,
        plot_paths=plot_paths,
    )


def predict_from_model(
    model_path: str | Path,
    input_csv: str | Path,
    output_csv: str | Path | None = None,
) -> pd.DataFrame:
    """Run batch inference on a CSV file using a saved predictive modeling artifact."""
    artifact = joblib.load(model_path)
    df = pd.read_csv(input_csv)

    if isinstance(artifact, dict) and "pipeline" in artifact:
        pipeline: Pipeline = artifact["pipeline"]
        feature_names: list[str] = artifact["feature_names"]
        target_col: str | None = artifact.get("target_col")
        target_names: list[str] | None = artifact.get("target_names")
        task_type: str = artifact.get("task_type", "classification")
    else:
        pipeline = artifact
        feature_names = list(df.columns)
        target_col = None
        target_names = None
        task_type = "classification"

    missing_cols = [c for c in feature_names if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Input CSV is missing required feature columns: {missing_cols}")

    X_infer = sanitize_feature_frame(df[feature_names])
    preds = pipeline.predict(X_infer)

    out_df = df.copy()
    if target_col and target_col in out_df.columns:
        out_df = out_df.rename(columns={target_col: f"actual_{target_col}"})
    out_df["prediction"] = preds

    if task_type == "classification" and target_names:
        out_df["predicted_label"] = [
            target_names[int(p)] if 0 <= int(p) < len(target_names) else str(p) for p in preds
        ]
    if task_type == "classification" and hasattr(pipeline, "predict_proba"):
        probs = pipeline.predict_proba(X_infer)
        if probs.ndim == 2 and probs.shape[1] == 2:
            out_df["positive_probability"] = np.round(probs[:, 1], 4)

    if output_csv is not None:
        out_path = Path(output_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(out_path, index=False)
    return out_df

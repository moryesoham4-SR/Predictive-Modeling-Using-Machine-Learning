"""End-to-end tests for supervised experiments, evaluation CIs, and inference."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from predictive_modeling.data import (
    load_builtin_classification,
    load_builtin_regression,
    load_csv_dataset,
)
from predictive_modeling.pipeline import predict_from_model, run_experiment


def test_classification_experiment_and_prediction_roundtrip(tmp_path: Path) -> None:
    bundle = load_builtin_classification()
    out_dir = tmp_path / "cls_out"
    res = run_experiment(bundle, output_dir=out_dir, random_state=42, tune_best=False)

    assert res.best_model_path.exists()
    assert res.metrics_path.exists()
    assert res.report_path.exists()
    assert "confusion_matrices" in res.plot_paths and res.plot_paths["confusion_matrices"].exists()
    assert "roc_curves" in res.plot_paths and res.plot_paths["roc_curves"].exists()
    assert "precision_recall_curves" in res.plot_paths

    payload = json.loads(res.metrics_path.read_text(encoding="utf-8"))
    assert payload["best_model_key"] in {
        "logistic_regression",
        "decision_tree_unconstrained",
        "decision_tree_regularized",
        "random_forest",
    }
    acc_ci = res.bootstrap_cis["accuracy"]
    assert acc_ci["ci_low"] <= acc_ci["estimate"] <= acc_ci["ci_high"]
    assert acc_ci["estimate"] > 0.85

    sample_input = tmp_path / "new_samples.csv"
    bundle.df[bundle.feature_names].head(10).to_csv(sample_input, index=False)
    preds_df = predict_from_model(res.best_model_path, sample_input)
    assert len(preds_df) == 10
    assert "prediction" in preds_df.columns
    assert "predicted_label" in preds_df.columns
    assert "positive_probability" in preds_df.columns


def test_regression_experiment_generates_diagnostics(tmp_path: Path) -> None:
    bundle = load_builtin_regression()
    out_dir = tmp_path / "reg_out"
    res = run_experiment(bundle, output_dir=out_dir, random_state=42, tune_best=False)

    assert "regression_diagnostics" in res.plot_paths
    assert res.plot_paths["regression_diagnostics"].exists()
    assert {"linear_regression", "decision_tree_regularized", "random_forest"}.issubset(
        set(res.comparison_df["key"])
    )
    r2_ci = res.bootstrap_cis["r2"]
    assert r2_ci["ci_low"] <= r2_ci["estimate"] <= r2_ci["ci_high"]


def test_custom_csv_mixed_features_and_missing_values(tmp_path: Path) -> None:
    rng = np.random.default_rng(12)
    n = 120
    score = rng.normal(0, 1, n)
    region = rng.choice(["north", "south", "east", "west"], size=n)
    income = 50000 + score * 12000 + rng.normal(0, 2500, n)
    income[0:6] = np.nan
    label = np.where(score + (region == "north") * 0.6 > 0.2, "approved", "denied")

    csv_path = tmp_path / "loans.csv"
    pd.DataFrame({"score": score, "region": region, "income": income, "status": label}).to_csv(
        csv_path, index=False
    )

    bundle = load_csv_dataset(csv_path, target_col="status", positive_label="approved")
    res = run_experiment(bundle, output_dir=tmp_path / "custom_out", random_state=42, tune_best=False)
    assert res.best_model_path.exists()
    assert res.bootstrap_cis["roc_auc"]["estimate"] > 0.70

"""Tests for dataset loading, quality validation, and leakage-free preprocessing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from predictive_modeling.data import (
    load_builtin_classification,
    load_builtin_regression,
    load_csv_dataset,
    split_dataset,
)
from predictive_modeling.preprocessing import analyze_data_quality, build_preprocessor


def test_builtin_loaders_and_target_mapping() -> None:
    cls_bundle = load_builtin_classification()
    assert cls_bundle.task_type == "classification"
    assert len(cls_bundle.df) == 569
    assert set(cls_bundle.df[cls_bundle.target_col].unique()) == {0, 1}

    reg_bundle = load_builtin_regression()
    assert reg_bundle.task_type == "regression"
    assert len(reg_bundle.df) == 442


def test_custom_csv_drops_missing_target_and_handles_inf(tmp_path: Path) -> None:
    csv_path = tmp_path / "sample.csv"
    df = pd.DataFrame(
        {
            "age": [25.0, 30.0, np.inf, 45.0, 50.0, 35.0, 40.0, 28.0, 60.0, 55.0],
            "plan": ["basic", "pro", "basic", None, "pro", "basic", "pro", "basic", "pro", "basic"],
            "active": [True, False, True, True, False, True, False, True, False, True],
            "churned": ["no", "yes", "no", "yes", None, "no", "yes", "no", "yes", "no"],
        }
    )
    df.to_csv(csv_path, index=False)

    bundle = load_csv_dataset(csv_path, target_col="churned", positive_label="yes")
    assert bundle.task_type == "classification"
    assert bundle.cleaning_summary["dropped_missing_target_rows"] == 1
    assert len(bundle.df) == 9
    assert bundle.target_names == ["no", "yes"]
    assert np.isfinite(bundle.df["age"].dropna()).all()


def test_preprocessor_fits_only_on_train_without_leakage() -> None:
    train_x = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, np.nan], "cat": ["a", "b", "a", "a", "b", "a", None]})
    test_x = pd.DataFrame({"x": [1000.0, 2000.0, np.nan], "cat": ["c", "a", "b"]})

    preprocessor, meta = build_preprocessor(train_x)
    assert meta["numeric_missing_indicator_enabled"] is True

    transformed_train = preprocessor.fit_transform(train_x)
    transformed_test = preprocessor.transform(test_x)

    # Median of train_x['x'] is 2.5; if test leaked, median would jump sharply.
    num_imputer = preprocessor.named_transformers_["num"].named_steps["imputer"]
    assert num_imputer.statistics_[0] == pytest.approx(2.5)
    assert transformed_train.shape[1] == transformed_test.shape[1]
    assert np.isfinite(transformed_test).all()


def test_split_dataset_preserves_stratification_and_sizes() -> None:
    bundle = load_builtin_classification()
    X = bundle.df[bundle.feature_names]
    y = bundle.df[bundle.target_col]
    split = split_dataset(X, y, task_type="classification", test_size=0.2, val_size=0.2, random_state=42)

    assert len(split.X_train) + len(split.X_val) + len(split.X_test) == len(bundle.df)
    train_rate = split.y_train.mean()
    test_rate = split.y_test.mean()
    assert abs(train_rate - test_rate) < 0.02

    quality = analyze_data_quality(bundle.df, bundle.target_col)
    assert quality["n_rows"] == 569
    assert quality["n_features"] == 30

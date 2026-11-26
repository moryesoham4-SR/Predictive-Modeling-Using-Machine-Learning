"""Data quality inspection and leakage-free preprocessing pipelines."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def analyze_data_quality(df: pd.DataFrame, target_col: str) -> dict[str, Any]:
    """Inspect schema, missingness, duplicates, and target distribution."""
    feature_cols = [c for c in df.columns if c != target_col]
    missing_counts = df[feature_cols].isna().sum()
    missing_cols = {
        col: {
            "count": int(cnt),
            "pct": round(float(cnt) / max(len(df), 1) * 100.0, 2),
        }
        for col, cnt in missing_counts.items()
        if cnt > 0
    }

    numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]
    high_cardinality_cols = [
        c for c in categorical_cols if int(df[c].nunique(dropna=True)) > 20
    ]
    constant_cols = [c for c in feature_cols if int(df[c].nunique(dropna=False)) <= 1]

    target_series = df[target_col]
    if pd.api.types.is_numeric_dtype(target_series) and target_series.nunique() > 15:
        target_summary: dict[str, Any] = {
            "mean": round(float(target_series.mean()), 4),
            "std": round(float(target_series.std()), 4),
            "min": round(float(target_series.min()), 4),
            "median": round(float(target_series.median()), 4),
            "max": round(float(target_series.max()), 4),
        }
    else:
        counts = target_series.value_counts().sort_index()
        target_summary = {str(k): int(v) for k, v in counts.items()}

    return {
        "n_rows": int(len(df)),
        "n_features": int(len(feature_cols)),
        "numeric_features": numeric_cols,
        "categorical_features": categorical_cols,
        "high_cardinality_categorical_features": high_cardinality_cols,
        "constant_features": constant_cols,
        "missing_value_columns": missing_cols,
        "total_missing_cells": int(missing_counts.sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "target_summary": target_summary,
    }


def build_preprocessor(
    X_train: pd.DataFrame,
    max_ohe_categories: int = 20,
) -> tuple[ColumnTransformer, dict[str, Any]]:
    """Build an unfitted ColumnTransformer configured strictly from X_train schema.

    Preserves existing numeric representations, adds missingness indicators when
    numeric missingness exists in X_train, caps high-cardinality categorical
    features via ``max_categories``, and standardizes numeric inputs.
    """
    all_nan_cols = [c for c in X_train.columns if X_train[c].isna().all()]
    usable_cols = [c for c in X_train.columns if c not in all_nan_cols]

    numeric_cols = X_train[usable_cols].select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in usable_cols if c not in numeric_cols]

    has_missing_numeric = bool(numeric_cols and X_train[numeric_cols].isna().any().any())
    transformers: list[tuple[str, Any, list[str]]] = []

    if numeric_cols:
        num_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median", add_indicator=has_missing_numeric)),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("num", num_pipe, numeric_cols))

    if categorical_cols:
        cat_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        max_categories=max_ohe_categories,
                        sparse_output=False,
                    ),
                ),
            ]
        )
        transformers.append(("cat", cat_pipe, categorical_cols))

    if not transformers:
        raise ValueError("No usable feature columns remain after dropping all-NaN columns.")

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=True,
    )

    metadata = {
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "dropped_all_nan_columns": all_nan_cols,
        "numeric_missing_indicator_enabled": has_missing_numeric,
        "max_ohe_categories": max_ohe_categories,
    }
    return preprocessor, metadata

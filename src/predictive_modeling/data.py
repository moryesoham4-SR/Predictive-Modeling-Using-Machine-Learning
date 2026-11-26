"""Dataset loading, quality sanitization, and strict train/validation/test splitting."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split

TaskType = Literal["classification", "regression"]


@dataclass
class DatasetBundle:
    """Container for a supervised learning dataset and its metadata."""

    name: str
    task_type: TaskType
    df: pd.DataFrame
    target_col: str
    feature_names: list[str]
    target_names: list[str] | None = None
    description: str = ""
    cleaning_summary: dict[str, Any] = field(default_factory=dict)


@dataclass
class DataSplit:
    """Three-way supervised data split created prior to preprocessing."""

    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series


def sanitize_feature_frame(X: pd.DataFrame) -> pd.DataFrame:
    """Normalize infinite values and boolean dtypes for reliable sklearn pipelines."""
    cleaned = X.copy()
    bool_cols = cleaned.select_dtypes(include=["bool"]).columns.tolist()
    for col in bool_cols:
        cleaned[col] = cleaned[col].astype(int)

    num_cols = cleaned.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        cleaned[num_cols] = cleaned[num_cols].replace([np.inf, -np.inf], np.nan)
    return cleaned


def _clean_dataframe(
    df: pd.DataFrame,
    target_col: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Drop rows missing the target value and normalize non-finite feature values."""
    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found. Available columns: {list(df.columns)}"
        )

    cleaned = df.copy()
    initial_rows = len(cleaned)
    duplicate_rows = int(cleaned.duplicated().sum())

    # Replace +/-inf in target if numeric before checking missingness
    if pd.api.types.is_numeric_dtype(cleaned[target_col]):
        cleaned[target_col] = cleaned[target_col].replace([np.inf, -np.inf], np.nan)

    missing_target_mask = cleaned[target_col].isna()
    dropped_missing_target = int(missing_target_mask.sum())
    if dropped_missing_target > 0:
        cleaned = cleaned.loc[~missing_target_mask].reset_index(drop=True)

    feature_cols = [c for c in cleaned.columns if c != target_col]
    cleaned[feature_cols] = sanitize_feature_frame(cleaned[feature_cols])

    summary = {
        "initial_rows": initial_rows,
        "final_rows": len(cleaned),
        "dropped_missing_target_rows": dropped_missing_target,
        "duplicate_rows_detected": duplicate_rows,
    }
    return cleaned, summary


def load_builtin_classification(export_csv_path: Path | None = None) -> DatasetBundle:
    """Load the Breast Cancer Wisconsin diagnostic dataset for binary classification.

    Target mapping sets ``1 = Malignant`` and ``0 = Benign`` so clinical positive
    metrics (Recall, Precision, ROC-AUC) measure detection of malignancy.
    """
    bunch = load_breast_cancer(as_frame=True)
    df = bunch.frame.copy()
    # Original sklearn target encodes 0=malignant, 1=benign. Map malignant -> 1 for clarity.
    target_col = "diagnosis_malignant"
    df[target_col] = (df["target"] == 0).astype(int)
    df = df.drop(columns=["target"])

    cleaned, summary = _clean_dataframe(df, target_col)
    if export_csv_path is not None:
        export_csv_path.parent.mkdir(parents=True, exist_ok=True)
        cleaned.to_csv(export_csv_path, index=False)

    feature_names = [c for c in cleaned.columns if c != target_col]
    return DatasetBundle(
        name="breast_cancer_diagnostic",
        task_type="classification",
        df=cleaned,
        target_col=target_col,
        feature_names=feature_names,
        target_names=["Benign (0)", "Malignant (1)"],
        description=(
            "Breast Cancer Wisconsin Diagnostic dataset (569 FNA cell-nuclei samples, "
            "30 numeric features; target diagnosis_malignant: 1=Malignant, 0=Benign)."
        ),
        cleaning_summary=summary,
    )


def load_builtin_regression(export_csv_path: Path | None = None) -> DatasetBundle:
    """Load the scikit-learn Diabetes dataset (unscaled features) for regression."""
    bunch = load_diabetes(as_frame=True, scaled=False)
    df = bunch.frame.copy()
    target_col = "disease_progression"
    df = df.rename(columns={"target": target_col})

    cleaned, summary = _clean_dataframe(df, target_col)
    if export_csv_path is not None:
        export_csv_path.parent.mkdir(parents=True, exist_ok=True)
        cleaned.to_csv(export_csv_path, index=False)

    feature_names = [c for c in cleaned.columns if c != target_col]
    return DatasetBundle(
        name="diabetes_progression",
        task_type="regression",
        df=cleaned,
        target_col=target_col,
        feature_names=feature_names,
        target_names=None,
        description=(
            "Diabetes dataset (442 patients, 10 baseline clinical measurements; "
            "continuous target disease_progression one year after baseline)."
        ),
        cleaning_summary=summary,
    )


def infer_task_type(series: pd.Series) -> TaskType:
    """Infer whether a target series represents classification or regression."""
    if (
        pd.api.types.is_bool_dtype(series)
        or pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series)
        or isinstance(series.dtype, pd.CategoricalDtype)
    ):
        return "classification"

    unique_count = int(series.nunique(dropna=True))
    if pd.api.types.is_integer_dtype(series) and unique_count <= 15:
        return "classification"
    if unique_count <= 2:
        return "classification"
    return "regression"


def _encode_classification_target(
    y: pd.Series,
    positive_label: str | None = None,
) -> tuple[pd.Series, list[str]]:
    """Normalize classification target labels to consecutive integer codes 0..K-1."""
    if pd.api.types.is_integer_dtype(y):
        unique_vals = sorted(y.unique().tolist())
        if unique_vals == list(range(len(unique_vals))):
            return y.astype(int), [str(v) for v in unique_vals]

    unique_labels = sorted(y.astype(str).unique().tolist())
    if len(unique_labels) == 2 and positive_label is not None:
        pos_str = str(positive_label)
        if pos_str not in unique_labels:
            raise ValueError(
                f"positive_label '{positive_label}' not found in target values {unique_labels}."
            )
        neg_str = [lbl for lbl in unique_labels if lbl != pos_str][0]
        ordered = [neg_str, pos_str]
    else:
        ordered = unique_labels

    mapping = {lbl: idx for idx, lbl in enumerate(ordered)}
    encoded = y.astype(str).map(mapping).astype(int)
    return encoded, ordered


def load_csv_dataset(
    csv_path: str | Path,
    target_col: str,
    task_type: Literal["auto", "classification", "regression"] = "auto",
    positive_label: str | None = None,
) -> DatasetBundle:
    """Load and validate a custom CSV dataset for supervised learning."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    raw_df = pd.read_csv(path)
    cleaned, summary = _clean_dataframe(raw_df, target_col)
    if cleaned.empty:
        raise ValueError("No valid rows remain after removing rows with missing target values.")

    resolved_task: TaskType = (
        infer_task_type(cleaned[target_col]) if task_type == "auto" else task_type
    )

    target_names: list[str] | None = None
    if resolved_task == "classification":
        encoded_y, target_names = _encode_classification_target(
            cleaned[target_col], positive_label=positive_label
        )
        cleaned[target_col] = encoded_y
    else:
        cleaned[target_col] = pd.to_numeric(cleaned[target_col], errors="raise")

    feature_names = [c for c in cleaned.columns if c != target_col]
    return DatasetBundle(
        name=path.stem,
        task_type=resolved_task,
        df=cleaned,
        target_col=target_col,
        feature_names=feature_names,
        target_names=target_names,
        description=f"Custom CSV dataset loaded from {path.name} ({len(cleaned)} rows, {len(feature_names)} features).",
        cleaning_summary=summary,
    )


def split_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    task_type: TaskType,
    test_size: float = 0.20,
    val_size: float = 0.20,
    random_state: int = 42,
) -> DataSplit:
    """Split data into Train, Validation, and Test sets BEFORE fitting preprocessors."""
    if not 0 < test_size < 0.5 or not 0 < val_size < 0.5 or (test_size + val_size) >= 0.8:
        raise ValueError("test_size and val_size must be in (0, 0.5) and sum to < 0.8.")

    stratify_full = y if (task_type == "classification" and y.value_counts().min() >= 3) else None
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_full,
    )

    val_fraction_of_remaining = val_size / (1.0 - test_size)
    stratify_rem = (
        y_train_val
        if (task_type == "classification" and y_train_val.value_counts().min() >= 2)
        else None
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=val_fraction_of_remaining,
        random_state=random_state,
        stratify=stratify_rem,
    )

    return DataSplit(
        X_train=X_train.reset_index(drop=True),
        X_val=X_val.reset_index(drop=True),
        X_test=X_test.reset_index(drop=True),
        y_train=y_train.reset_index(drop=True),
        y_val=y_val.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
    )

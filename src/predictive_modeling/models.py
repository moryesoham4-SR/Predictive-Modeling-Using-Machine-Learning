"""Model specifications and hyperparameter tuning helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from predictive_modeling.data import TaskType


@dataclass(frozen=True)
class ModelSpec:
    """Definition of a candidate supervised model."""

    key: str
    display_name: str
    estimator: BaseEstimator
    is_baseline: bool
    notes: str


def build_model_specs(task_type: TaskType, random_state: int = 42) -> list[ModelSpec]:
    """Return baseline and candidate model specifications for the requested task."""
    if task_type == "classification":
        return [
            ModelSpec(
                key="naive_baseline",
                display_name="Naive Baseline (Majority)",
                estimator=DummyClassifier(strategy="most_frequent"),
                is_baseline=True,
                notes="Always predicts the majority class; minimum accuracy floor.",
            ),
            ModelSpec(
                key="logistic_regression",
                display_name="Logistic Regression",
                estimator=LogisticRegression(max_iter=2000, random_state=random_state),
                is_baseline=True,
                notes="Linear decision-boundary classifier baseline with calibrated log-odds.",
            ),
            ModelSpec(
                key="decision_tree_unconstrained",
                display_name="Decision Tree (Unconstrained)",
                estimator=DecisionTreeClassifier(random_state=random_state),
                is_baseline=False,
                notes="Unrestricted depth tree; demonstrates variance and overfitting risk.",
            ),
            ModelSpec(
                key="decision_tree_regularized",
                display_name="Decision Tree (Regularized)",
                estimator=DecisionTreeClassifier(
                    max_depth=4, min_samples_leaf=5, random_state=random_state
                ),
                is_baseline=False,
                notes="Depth- and leaf-constrained tree to reduce overfitting.",
            ),
            ModelSpec(
                key="random_forest",
                display_name="Random Forest",
                estimator=RandomForestClassifier(
                    n_estimators=200,
                    max_depth=8,
                    min_samples_leaf=2,
                    random_state=random_state,
                    n_jobs=1,
                ),
                is_baseline=False,
                notes="Bagged ensemble of decorrelated decision trees.",
            ),
        ]

    return [
        ModelSpec(
            key="naive_baseline",
            display_name="Naive Baseline (Mean)",
            estimator=DummyRegressor(strategy="mean"),
            is_baseline=True,
            notes="Always predicts the training target mean; R2=0 floor.",
        ),
        ModelSpec(
            key="linear_regression",
            display_name="Linear Regression (OLS)",
            estimator=LinearRegression(),
            is_baseline=True,
            notes="Ordinary Least Squares linear regression baseline.",
        ),
        ModelSpec(
            key="decision_tree_unconstrained",
            display_name="Decision Tree (Unconstrained)",
            estimator=DecisionTreeRegressor(random_state=random_state),
            is_baseline=False,
            notes="Unrestricted depth tree; typically achieves near-zero train error by overfitting.",
        ),
        ModelSpec(
            key="decision_tree_regularized",
            display_name="Decision Tree (Regularized)",
            estimator=DecisionTreeRegressor(
                max_depth=4, min_samples_leaf=5, random_state=random_state
            ),
            is_baseline=False,
            notes="Depth- and leaf-constrained tree balancing bias and variance.",
        ),
        ModelSpec(
            key="random_forest",
            display_name="Random Forest",
            estimator=RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                min_samples_leaf=2,
                random_state=random_state,
                n_jobs=1,
            ),
            is_baseline=False,
            notes="Bagged tree ensemble reducing prediction variance.",
        ),
    ]


def build_pipeline(preprocessor: ColumnTransformer, estimator: BaseEstimator) -> Pipeline:
    """Wrap an unfitted preprocessor clone and estimator into a single sklearn Pipeline."""
    return Pipeline(steps=[("preprocessor", clone(preprocessor)), ("model", clone(estimator))])


def tune_random_forest(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    task_type: TaskType,
    random_state: int = 42,
) -> tuple[Pipeline, dict[str, Any], float, float]:
    """Tune Random Forest hyperparameters via cross-validation on X_train only."""
    if task_type == "classification":
        base_rf: BaseEstimator = RandomForestClassifier(random_state=random_state, n_jobs=1)
        is_binary = y_train.nunique() == 2
        scoring = "roc_auc" if is_binary else "f1_weighted"
        n_splits = max(2, min(5, int(y_train.value_counts().min())))
        cv_splitter: Any = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state
        )
    else:
        base_rf = RandomForestRegressor(random_state=random_state, n_jobs=1)
        scoring = "r2"
        n_splits = max(2, min(5, len(X_train)))
        cv_splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    pipe = build_pipeline(preprocessor, base_rf)
    param_grid = {
        "model__n_estimators": [100, 250],
        "model__max_depth": [4, 8, None],
        "model__min_samples_leaf": [1, 3],
    }

    search = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv_splitter,
        n_jobs=1,
        refit=True,
    )
    search.fit(X_train, y_train)
    clean_params = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
    refit_seconds = float(getattr(search, "refit_time_", 0.0))
    return search.best_estimator_, clean_params, float(search.best_score_), refit_seconds

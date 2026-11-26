# Predictive Modeling Report: diabetes_progression

- **Task Type:** `regression`
- **Description:** Diabetes dataset (442 patients, 10 baseline clinical measurements; continuous target disease_progression one year after baseline).
- **Target Column:** `disease_progression`
- **Dataset Shape:** 442 rows x 10 features
- **Data Cleaning Summary:** {"initial_rows": 442, "final_rows": 442, "dropped_missing_target_rows": 0, "duplicate_rows_detected": 0}
- **Missing Value Cells:** 0 | **Duplicate Rows:** 0
- **Strict Split Sizes (Before Preprocessing):** Train=264, Val=89, Test=89
- **Preprocessing Summary:** 10 numeric features, 0 categorical features

## Model Comparison

| key                         | model                         |   train_r2 |   val_r2 |   test_r2 |   val_rmse |   test_rmse |   val_mae |   test_mae |   cv_r2_mean |   cv_r2_std |   overfit_gap |   train_time_sec |
|:----------------------------|:------------------------------|-----------:|---------:|----------:|-----------:|------------:|----------:|-----------:|-------------:|------------:|--------------:|-----------------:|
| naive_baseline              | Naive Baseline (Mean)         |     0      |  -0.0443 |   -0.0031 |    75.6392 |     72.9002 |   64.1288 |    63.5314 |      -0.0231 |      0.0173 |        0.0443 |           0.0054 |
| linear_regression           | Linear Regression (OLS)       |     0.5203 |   0.5216 |    0.4464 |    51.1937 |     54.1564 |   42.9389 |    42.9245 |       0.4629 |      0.0887 |       -0.0014 |           0.0167 |
| decision_tree_unconstrained | Decision Tree (Unconstrained) |     1      |   0.2025 |   -0.2764 |    66.1011 |     82.2349 |   50.5506 |    65.5506 |      -0.2859 |      0.3288 |        0.7975 |           0.005  |
| decision_tree_regularized   | Decision Tree (Regularized)   |     0.6137 |   0.3196 |    0.3113 |    61.0539 |     60.4061 |   48.7993 |    48.5994 |       0.1981 |      0.0824 |        0.294  |           0.0041 |
| random_forest               | Random Forest                 |     0.8816 |   0.4704 |    0.4384 |    53.8682 |     54.5466 |   43.9619 |    44.5518 |       0.3734 |      0.0812 |        0.4113 |           0.1803 |
| random_forest_tuned         | Random Forest (GridSearchCV)  |     0.6862 |   0.47   |    0.4571 |    53.8847 |     53.6337 |   43.285  |    43.2546 |       0.3968 |      0.0779 |        0.2162 |           0.1832 |

## Selected Best Model (by Validation Set): Linear Regression (OLS)

- **Train Time:** 0.0167s | **Inference Latency:** 28.7157 ms / 1k rows

### Held-Out Test Performance & 95% Bootstrap Confidence Intervals

- **R2:** 0.4464 (95% CI: [0.2582, 0.5851])
- **RMSE:** 54.1564 (95% CI: [46.8169, 61.3594])
- **MAE:** 42.9245 (95% CI: [36.2444, 50.0168])

### Slice-Based Error Analysis (Test Set)

| slice_feature   | slice_group   |   n_samples |     mae |    rmse |
|:----------------|:--------------|------------:|--------:|--------:|
| s1              | High          |          30 | 35.1821 | 46.2718 |
| s1              | Low           |          31 | 50.8526 | 63.5691 |
| s1              | Medium        |          28 | 42.4423 | 50.5418 |

### Top Feature Importances

| feature   |   importance |
|:----------|-------------:|
| s1        |      51.1014 |
| s5        |      37.2716 |
| s2        |      27.4808 |
| bmi       |      25.0766 |
| bp        |      15.1673 |
| s4        |      15.0321 |
| sex       |      11.3593 |
| s3        |      10.8403 |
| s6        |       3.8171 |
| age       |       1.1318 |

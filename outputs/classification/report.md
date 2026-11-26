# Predictive Modeling Report: breast_cancer_diagnostic

- **Task Type:** `classification`
- **Description:** Breast Cancer Wisconsin Diagnostic dataset (569 FNA cell-nuclei samples, 30 numeric features; target diagnosis_malignant: 1=Malignant, 0=Benign).
- **Target Column:** `diagnosis_malignant`
- **Dataset Shape:** 569 rows x 30 features
- **Data Cleaning Summary:** {"initial_rows": 569, "final_rows": 569, "dropped_missing_target_rows": 0, "duplicate_rows_detected": 0}
- **Missing Value Cells:** 0 | **Duplicate Rows:** 0
- **Strict Split Sizes (Before Preprocessing):** Train=341, Val=114, Test=114
- **Preprocessing Summary:** 30 numeric features, 0 categorical features

## Model Comparison

| key                         | model                         |   train_accuracy |   val_accuracy |   test_accuracy |   val_f1 |   test_f1 |   val_roc_auc |   test_roc_auc |   cv_accuracy_mean |   cv_accuracy_std |   overfit_gap |   train_time_sec |
|:----------------------------|:------------------------------|-----------------:|---------------:|----------------:|---------:|----------:|--------------:|---------------:|-------------------:|------------------:|--------------:|-----------------:|
| naive_baseline              | Naive Baseline (Majority)     |           0.6276 |         0.6228 |          0.6316 |   0      |    0      |        0.5    |         0.5    |             0.6276 |            0.0061 |        0.0048 |           0.0048 |
| logistic_regression         | Logistic Regression           |           0.9883 |         0.9737 |          0.9737 |   0.9655 |    0.9639 |        0.9957 |         0.9954 |             0.9648 |            0.0199 |        0.0146 |           0.0091 |
| decision_tree_unconstrained | Decision Tree (Unconstrained) |           1      |         0.8947 |          0.9035 |   0.8667 |    0.8642 |        0.8972 |         0.8889 |             0.9384 |            0.0142 |        0.1053 |           0.0071 |
| decision_tree_regularized   | Decision Tree (Regularized)   |           0.9736 |         0.9035 |          0.9298 |   0.8764 |    0.9024 |        0.9387 |         0.9552 |             0.959  |            0.011  |        0.0701 |           0.0061 |
| random_forest               | Random Forest                 |           0.9941 |         0.9737 |          0.9561 |   0.9647 |    0.9383 |        0.9912 |         0.997  |             0.9444 |            0.031  |        0.0205 |           0.186  |
| random_forest_tuned         | Random Forest (GridSearchCV)  |           0.9941 |         0.9737 |          0.9561 |   0.9647 |    0.9383 |        0.9898 |         0.9983 |             0.9414 |            0.0276 |        0.0205 |           0.1002 |

## Selected Best Model (by Validation Set): Logistic Regression

- **Train Time:** 0.0091s | **Inference Latency:** 20.3149 ms / 1k rows

### Held-Out Test Performance & 95% Bootstrap Confidence Intervals

- **ACCURACY:** 0.9737 (95% CI: [0.9386, 1.0000])
- **F1:** 0.9639 (95% CI: [0.9136, 1.0000])
- **ROC_AUC:** 0.9954 (95% CI: [0.9846, 1.0000])

### Confusion Matrix Breakdown (Test Set)

- True Negatives (TN): 71 | False Positives (FP): 1
- False Negatives (FN): 2 | True Positives (TP): 40
- Precision: 0.9756 | Recall (Sensitivity): 0.9524 | Specificity: 0.9861

### Slice-Based Error Analysis (Test Set)

| slice_feature   | slice_group   |   n_samples |   accuracy |   error_rate |
|:----------------|:--------------|------------:|-----------:|-------------:|
| worst area      | Low           |          38 |     1      |       0      |
| worst area      | High          |          38 |     0.9737 |       0.0263 |
| worst area      | Medium        |          38 |     0.9474 |       0.0526 |

### Top Feature Importances

| feature                 |   importance |
|:------------------------|-------------:|
| worst texture           |       1.3553 |
| mean concave points     |       1.1043 |
| radius error            |       1.0259 |
| worst symmetry          |       0.9519 |
| worst radius            |       0.8134 |
| worst area              |       0.7948 |
| area error              |       0.7822 |
| worst concavity         |       0.7689 |
| mean concavity          |       0.7621 |
| compactness error       |       0.7591 |
| worst perimeter         |       0.6617 |
| fractal dimension error |       0.6594 |

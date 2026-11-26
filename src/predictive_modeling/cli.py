"""Command-line interface for running predictive modeling experiments and predictions."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from predictive_modeling.data import (
    load_builtin_classification,
    load_builtin_regression,
    load_csv_dataset,
)
from predictive_modeling.pipeline import ExperimentResult, predict_from_model, run_experiment


def _print_result_summary(result: ExperimentResult) -> None:
    """Print concise formatted summary of an experiment run."""
    print(f"\n=== {result.dataset_name} ({result.task_type}) ===")
    print(result.comparison_df.round(4).to_string(index=False))
    print(f"\nBest Model (selected on Validation): {result.best_model_name}")
    for metric_name, ci in result.bootstrap_cis.items():
        print(
            f"  Test {metric_name}: {ci['estimate']:.4f} "
            f"(95% CI: [{ci['ci_low']:.4f}, {ci['ci_high']:.4f}])"
        )
    print(f"Saved model:  {result.best_model_path}")
    print(f"Saved report: {result.report_path}")
    print(f"Saved plots:  {', '.join(p.name for p in result.plot_paths.values())}")


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="predictive-modeling",
        description="End-to-end supervised learning with model comparison, confusion matrices, and ROC/AUC curves.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Train, evaluate, and visualize models.")
    run_parser.add_argument(
        "--task",
        choices=["classification", "regression", "both", "auto"],
        default="both",
        help="Built-in task track to run, or 'auto' for a custom CSV.",
    )
    run_parser.add_argument("--data", type=Path, default=None, help="Optional custom CSV path.")
    run_parser.add_argument("--target", type=str, default=None, help="Target column in custom CSV.")
    run_parser.add_argument(
        "--positive-label",
        type=str,
        default=None,
        help="Optional positive class name for binary custom classification CSVs.",
    )
    run_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory where plots, metrics, and models are written.",
    )
    run_parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory to export raw copies of built-in sample datasets.",
    )
    run_parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    run_parser.add_argument(
        "--no-tune",
        action="store_true",
        help="Skip GridSearchCV hyperparameter tuning for Random Forest.",
    )

    pred_parser = subparsers.add_parser("predict", help="Predict using a saved .joblib model.")
    pred_parser.add_argument("--model", type=Path, required=True, help="Path to best_model.joblib.")
    pred_parser.add_argument("--input", type=Path, required=True, help="Input CSV file.")
    pred_parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/predictions.csv"),
        help="Destination CSV path for predictions.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        if args.data is not None:
            if not args.target:
                parser.error("--target is required when --data is provided.")
            task_arg = "auto" if args.task in {"both", "auto"} else args.task
            bundle = load_csv_dataset(
                args.data,
                target_col=args.target,
                task_type=task_arg,
                positive_label=args.positive_label,
            )
            res = run_experiment(
                bundle,
                output_dir=args.output_dir,
                random_state=args.seed,
                tune_best=not args.no_tune,
            )
            _print_result_summary(res)
            return 0

        if args.task == "auto":
            parser.error("--task auto requires --data and --target.")

        if args.task in {"classification", "both"}:
            cls_bundle = load_builtin_classification(
                export_csv_path=args.data_dir / "breast_cancer.csv"
            )
            cls_out = args.output_dir / "classification" if args.task == "both" else args.output_dir
            cls_res = run_experiment(
                cls_bundle,
                output_dir=cls_out,
                random_state=args.seed,
                tune_best=not args.no_tune,
            )
            _print_result_summary(cls_res)

        if args.task in {"regression", "both"}:
            reg_bundle = load_builtin_regression(
                export_csv_path=args.data_dir / "diabetes.csv"
            )
            reg_out = args.output_dir / "regression" if args.task == "both" else args.output_dir
            reg_res = run_experiment(
                reg_bundle,
                output_dir=reg_out,
                random_state=args.seed,
                tune_best=not args.no_tune,
            )
            _print_result_summary(reg_res)
        return 0

    if args.command == "predict":
        preds = predict_from_model(args.model, args.input, args.output)
        print(f"Wrote {len(preds)} predictions to {args.output}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())

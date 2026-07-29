"""
Auto Email / Support Ticket Categorizer Main CLI Application.

Provides unified command-line entry points for training, evaluation, single prediction, batch prediction, and interactive mode.
"""

import argparse
import os
import sys
import json
import pandas as pd

from src.models.train import run_training_pipeline
from src.models.evaluator import evaluate_pipeline
from src.prediction.predictor import TicketPredictor
from src.preprocessing.text_cleaner import preprocess_dataset
from src.utils.io_utils import load_config, load_dataset, load_artifact, save_json
from src.visualization.plots import (
    plot_confusion_matrix,
    plot_class_distribution,
    plot_ticket_length_distribution,
    plot_feature_importance,
    plot_confidence_distribution,
    plot_confidence_by_department,
)
from src.utils.logger import get_logger

logger = get_logger("main_cli")


def handle_train_command(config_path: str) -> None:
    """Executes model training, grid tuning, evaluation, plot generation, and artifact saving."""
    logger.info("Executing systematic model optimization workflow...")
    config = load_config(config_path)

    # Run model training, GridSearchCV hyperparameter tuning & Repeated CV
    pipeline, df_train, df_test, report, feat_importance = run_training_pipeline(config_path)

    # Plot initial exploratory visualizations on raw dataset
    raw_df = load_dataset(config["data"]["raw_path"])
    plot_class_distribution(
        raw_df,
        target_col=config["data"]["target_column"],
        output_path=os.path.join(config["reports"]["dir"], config["reports"]["class_distribution_filename"]),
    )
    plot_ticket_length_distribution(
        raw_df,
        text_col=config["data"]["text_column"],
        target_col=config["data"]["target_column"],
        output_path=os.path.join(config["reports"]["dir"], config["reports"]["ticket_length_filename"]),
    )

    # Plot Feature Importance charts
    if feat_importance:
        plot_feature_importance(feat_importance, output_dir=config["reports"]["dir"], top_n=15)

    # Compute detailed evaluation STRICTLY on held-out test split (df_test)
    eval_results = evaluate_pipeline(
        pipeline,
        df_test,
        text_col="cleaned_ticket",
        target_col=config["data"]["target_column"],
        config_path=config_path,
    )

    # Plot confusion matrix on held-out test set predictions
    plot_confusion_matrix(
        eval_results["confusion_matrix"],
        eval_results["classes"],
        output_path=os.path.join(config["reports"]["dir"], config["reports"]["confusion_matrix_filename"]),
    )

    # Batch test prediction for confidence plots
    predictor = TicketPredictor(config_path=config_path)
    df_pred_test = predictor.predict_batch(df_test, text_col="cleaned_ticket")

    plot_confidence_distribution(
        df_pred_test["confidence_pct"].tolist(),
        output_path=os.path.join(config["reports"]["dir"], config["reports"]["confidence_distribution_filename"]),
    )
    plot_confidence_by_department(
        df_pred_test,
        dept_col="predicted_department",
        conf_col="confidence_pct",
        output_path=os.path.join(config["reports"]["dir"], config["reports"]["confidence_by_department_filename"]),
    )

    # Display Model Comparison Dashboard Table
    print("\n" + "=" * 85)
    print("                 MODEL COMPARISON DASHBOARD & BENCHMARKS             ")
    print("=" * 85)
    print(f"{'Algorithm':<22} | {'CV F1 (Mean)':<14} | {'CV F1 (Std)':<12} | {'Fit Time':<10} | {'Status'}")
    print("-" * 85)

    for name, metrics in report["model_comparisons"].items():
        status = "SELECTED OPTIMAL" if name == report["summary"]["best_algorithm"] else "Candidate"
        mean_str = f"{metrics['cv_f1_weighted_mean'] * 100:.2f}%"
        std_str = f"±{metrics['cv_f1_weighted_std'] * 100:.2f}%"
        time_str = f"{metrics['train_time_sec']:.3f}s"
        print(f"{name:<22} | {mean_str:<14} | {std_str:<12} | {time_str:<10} | {status}")

    print("=" * 85)

    print("\n" + "=" * 60)
    print("        OPTIMIZED MODEL - FINAL PERFORMANCE SUMMARY       ")
    print("=" * 60)
    print(f"Selected Optimal Algorithm:  {report['summary']['best_algorithm']}")
    print(f"Optimal Hyperparameters:     {report['summary']['best_params']}")
    print(f"Repeated 5-Fold CV F1:       {report['summary']['best_cv_f1_score'] * 100:.2f}%")
    print(f"Held-Out Test Accuracy:      {eval_results['accuracy'] * 100:.2f}%")
    print(f"Held-Out Test F1-Weighted:   {eval_results['f1_weighted'] * 100:.2f}%")
    print(f"Average Test Confidence:     {eval_results['confidence_metrics']['avg_confidence']:.2f}%")
    print(f"Held-Out Test Misclassified: {eval_results['misclassified_count']} / {len(df_test)}")
    print("=" * 60 + "\n")


def print_formatted_prediction(res: dict) -> None:
    """Prints formatted prediction output matching enterprise helpdesk CLI specification."""
    decision_badge = f"[AUTO ASSIGN]" if res['decision'] == "AUTO ASSIGN" else f"[NEEDS HUMAN REVIEW]"
    print("\n" + "=" * 55)
    print("Support Ticket")
    print(f"\"{res['ticket']}\"")
    print("-" * 55)
    print(f"Department       : {res['prediction']}")
    print(f"Confidence       : {res['confidence']}%")
    print(f"Priority         : {res.get('priority', 'NORMAL')}")
    print(f"Confidence Level : {res['confidence_label']}")
    print("Top Predictions  :")
    for idx, pred in enumerate(res["top_predictions"], 1):
        print(f"  {idx}. {pred['department']:<12} {pred['confidence']}%")
    print(f"Status           : {decision_badge}")
    print("=" * 55 + "\n")


def handle_predict_command(text: str, config_path: str) -> None:
    """Predicts department for a single ticket text."""
    predictor = TicketPredictor(config_path=config_path)
    res = predictor.predict_single(text)
    print_formatted_prediction(res)


def handle_batch_command(file_path: str, output_path: str, config_path: str) -> None:
    """Processes batch ticket predictions from CSV file."""
    if not os.path.exists(file_path):
        logger.error(f"Input batch file not found: {file_path}")
        sys.exit(1)

    df_batch = pd.read_csv(file_path)
    predictor = TicketPredictor(config_path=config_path)
    results = predictor.predict_batch(df_batch)

    if output_path:
        results.to_csv(output_path, index=False)
        print(f"Batch prediction results saved to: {output_path}")
    else:
        print(results.to_string(index=False))


def handle_interactive_command(config_path: str) -> None:
    """Runs interactive ticket routing shell."""
    predictor = TicketPredictor(config_path=config_path)
    print("\n" + "=" * 60)
    print("   INTERACTIVE SUPPORT TICKET ROUTER CLI (Type 'exit' to quit)")
    print("=" * 60 + "\n")

    while True:
        try:
            user_input = input("Enter support ticket text > ").strip()
            if user_input.lower() in ["exit", "quit", "q"]:
                print("Exiting ticket router. Goodbye!")
                break
            if not user_input:
                continue

            res = predictor.predict_single(user_input)
            print_formatted_prediction(res)
        except KeyboardInterrupt:
            print("\nSession interrupted. Exiting.")
            break


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Auto Email / Support Ticket Categorizer CLI")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config file")

    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Train subcommand
    subparsers.add_parser("train", help="Train models and save optimal pipeline artifact")

    # Evaluate subcommand
    subparsers.add_parser("evaluate", help="Evaluate saved model")

    # Predict subcommand
    predict_parser = subparsers.add_parser("predict", help="Predict department for single ticket")
    predict_parser.add_argument("--text", type=str, required=True, help="Support ticket text string")

    # Batch subcommand
    batch_parser = subparsers.add_parser("batch", help="Batch process tickets from CSV file")
    batch_parser.add_argument("--file", type=str, required=True, help="Input CSV file path")
    batch_parser.add_argument("--out", type=str, default="reports/batch_predictions.csv", help="Output CSV path")

    # Interactive subcommand
    subparsers.add_parser("interactive", help="Run interactive ticket triage shell")

    args = parser.parse_args()

    if args.command in ["train", "evaluate"]:
        handle_train_command(args.config)
    elif args.command == "predict":
        handle_predict_command(args.text, args.config)
    elif args.command == "batch":
        handle_batch_command(args.file, args.out, args.config)
    elif args.command == "interactive":
        handle_interactive_command(args.config)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

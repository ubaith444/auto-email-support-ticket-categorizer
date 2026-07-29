"""
Model Evaluation and Diagnostics Module.

Computes exact evaluation metrics (Accuracy, Precision, Recall, F1, Confusion Matrix, Classification Report)
and prediction confidence distribution statistics.
"""

import os
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline

from src.prediction.confidence import ConfidenceScorer
from src.utils.io_utils import save_json
from src.utils.logger import get_logger

logger = get_logger(__name__)


def evaluate_pipeline(
    pipeline: Pipeline,
    df_test: pd.DataFrame,
    text_col: str = "cleaned_ticket",
    target_col: str = "category",
    config_path: str = "config/config.yaml",
) -> Dict[str, Any]:
    """Evaluates pipeline on test dataframe and produces diagnostic metrics and confidence statistics.

    Args:
        pipeline (Pipeline): Trained scikit-learn pipeline.
        df_test (pd.DataFrame): Test dataframe.
        text_col (str): Ticket text column name.
        target_col (str): True label column name.
        config_path (str): Path to config file.

    Returns:
        Dict[str, Any]: Comprehensive evaluation report.
    """
    logger.info(f"Evaluating trained pipeline on test set ({len(df_test)} samples)...")

    scorer = ConfidenceScorer(config_path=config_path)

    X_test = df_test[text_col]
    y_true = df_test[target_col]

    predictions = []
    confidences = []
    conf_labels = []
    decisions = []
    correct_flags = []

    for text, true_lbl in zip(X_test, y_true):
        res = scorer.predict_with_confidence(pipeline, str(text), str(text))
        pred_lbl = res["prediction"]
        conf = res["confidence"]

        predictions.append(pred_lbl)
        confidences.append(conf)
        conf_labels.append(res["confidence_label"])
        decisions.append(res["decision"])
        correct_flags.append(pred_lbl == true_lbl)

    y_pred = np.array(predictions)
    conf_arr = np.array(confidences)
    correct_arr = np.array(correct_flags)

    acc = accuracy_score(y_true, y_pred)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)

    classes = sorted(list(set(y_true.unique()) | set(y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    cls_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    # Compute confidence metrics
    avg_conf = float(np.mean(conf_arr)) if len(conf_arr) > 0 else 0.0
    avg_correct_conf = float(np.mean(conf_arr[correct_arr])) if np.sum(correct_arr) > 0 else 0.0
    avg_incorrect_conf = float(np.mean(conf_arr[~correct_arr])) if np.sum(~correct_arr) > 0 else 0.0
    max_conf = float(np.max(conf_arr)) if len(conf_arr) > 0 else 0.0
    min_conf = float(np.min(conf_arr)) if len(conf_arr) > 0 else 0.0

    conf_dist = pd.Series(conf_labels).value_counts().to_dict()
    decision_dist = pd.Series(decisions).value_counts().to_dict()

    # Extract misclassified samples
    misclassified = []
    for idx, (true_lbl, pred_lbl, text, conf, dec) in enumerate(zip(y_true, y_pred, X_test, conf_arr, decisions)):
        if true_lbl != pred_lbl:
            misclassified.append({
                "sample_id": idx,
                "ticket_text": text,
                "true_category": true_lbl,
                "predicted_category": pred_lbl,
                "confidence": conf,
                "decision": dec,
            })

    results = {
        "accuracy": round(float(acc), 4),
        "precision_macro": round(float(p_macro), 4),
        "recall_macro": round(float(r_macro), 4),
        "f1_macro": round(float(f1_macro), 4),
        "precision_weighted": round(float(p_weighted), 4),
        "recall_weighted": round(float(r_weighted), 4),
        "f1_weighted": round(float(f1_weighted), 4),
        "confusion_matrix": cm.tolist(),
        "classes": [str(c) for c in classes],
        "classification_report": cls_report,
        "misclassified_count": len(misclassified),
        "misclassified_samples": misclassified,
        "confidence_metrics": {
            "avg_confidence": round(avg_conf, 2),
            "avg_correct_confidence": round(avg_correct_conf, 2),
            "avg_incorrect_confidence": round(avg_incorrect_conf, 2),
            "highest_confidence": round(max_conf, 2),
            "lowest_confidence": round(min_conf, 2),
            "confidence_level_distribution": conf_dist,
            "routing_decision_distribution": decision_dist,
        },
    }

    # Save confidence_report.json into reports/
    conf_report_path = os.path.join(scorer.config["reports"]["dir"], scorer.config["reports"]["confidence_report_filename"])
    save_json(results["confidence_metrics"], conf_report_path)

    logger.info(f"Evaluation metrics: Accuracy={acc:.4f}, F1-Weighted={f1_weighted:.4f}, Avg Confidence={avg_conf:.2f}%")
    return results

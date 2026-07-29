"""
Production Confidence Scoring and Human Review Triage Router.

Evaluates prediction confidence, calculates top-2 department recommendations, full probability distributions,
assigns confidence level tags (VERY HIGH, HIGH, MEDIUM, LOW), and determines routing decisions (AUTO ASSIGN vs NEEDS HUMAN REVIEW).
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from sklearn.pipeline import Pipeline

from src.utils.io_utils import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def softmax(x: np.ndarray) -> np.ndarray:
    """Computes softmax values for a 1D or 2D array of raw decision scores.

    Args:
        x (np.ndarray): Raw decision function scores.

    Returns:
        np.ndarray: Softmax normalized probability distribution.
    """
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


class ConfidenceScorer:
    """Production confidence calculation and department routing decision engine."""

    def __init__(self, config_path: str = "config/config.yaml") -> None:
        """Initializes confidence threshold configuration.

        Args:
            config_path (str): Path to YAML configuration file.
        """
        self.config = load_config(config_path)
        conf_cfg = self.config.get("confidence", {})
        self.threshold = conf_cfg.get("threshold", 0.60)
        self.very_high_threshold = conf_cfg.get("very_high", 0.95)
        self.high_threshold = conf_cfg.get("high", 0.80)
        self.medium_threshold = conf_cfg.get("medium", 0.60)

    def compute_confidence_label(self, confidence_pct: float) -> str:
        """Determines the confidence level tag based on percentage.

        Args:
            confidence_pct (float): Confidence score percentage (0 to 100).

        Returns:
            str: Level label ("VERY HIGH", "HIGH", "MEDIUM", "LOW").
        """
        conf_ratio = confidence_pct / 100.0
        if conf_ratio >= self.very_high_threshold:
            return "VERY HIGH"
        elif conf_ratio >= self.high_threshold:
            return "HIGH"
        elif conf_ratio >= self.medium_threshold:
            return "MEDIUM"
        else:
            return "LOW"

    def compute_routing_decision(self, confidence_pct: float) -> str:
        """Determines automated routing decision vs human review requirement.

        Args:
            confidence_pct (float): Confidence score percentage (0 to 100).

        Returns:
            str: Decision ("AUTO ASSIGN" vs "NEEDS HUMAN REVIEW").
        """
        conf_ratio = confidence_pct / 100.0
        if conf_ratio >= self.threshold:
            return "AUTO ASSIGN"
        else:
            logger.warning(
                f"Low prediction confidence ({confidence_pct:.2f}% < {self.threshold * 100:.0f}% threshold). "
                "Ticket routed for manual human review."
            )
            return "NEEDS HUMAN REVIEW"

    def predict_with_confidence(
        self,
        pipeline: Pipeline,
        raw_text: str,
        cleaned_text: str
    ) -> Dict[str, Any]:
        """Calculates prediction, probability distribution, confidence score, and routing decision.

        Args:
            pipeline (Pipeline): Trained scikit-learn pipeline.
            raw_text (str): Original input ticket string.
            cleaned_text (str): Cleaned/normalized ticket string.

        Returns:
            Dict[str, Any]: Structured prediction object.
        """
        classes = list(pipeline.classes_)

        # Handle empty/invalid input
        if not cleaned_text:
            all_probs = {c: round(100.0 / len(classes), 2) for c in classes}
            return {
                "ticket": raw_text,
                "cleaned_text": "",
                "prediction": "GENERAL",
                "department": "GENERAL",
                "confidence": 0.0,
                "confidence_pct": 0.0,
                "confidence_label": "LOW",
                "decision": "NEEDS HUMAN REVIEW",
                "top_predictions": [
                    {"department": "GENERAL", "confidence": 0.0},
                    {"department": "TECHNICAL", "confidence": 0.0},
                ],
                "top_2_predictions": [
                    {"department": "GENERAL", "confidence": 0.0},
                    {"department": "TECHNICAL", "confidence": 0.0},
                ],
                "all_probabilities": all_probs,
                "status": "warning_empty_input",
            }

        # Probability Estimation Strategy Detection
        classifier = pipeline.named_steps["classifier"]

        if hasattr(pipeline, "predict_proba"):
            probs = pipeline.predict_proba([cleaned_text])[0]
        elif hasattr(pipeline, "decision_function"):
            decision_scores = pipeline.decision_function([cleaned_text])[0]
            probs = softmax(decision_scores)
        else:
            top_pred = pipeline.predict([cleaned_text])[0]
            probs = np.array([1.0 if c == top_pred else 0.0 for c in classes])

        # Sort probability indices descending
        sorted_indices = np.argsort(probs)[::-1]

        top1_idx = sorted_indices[0]
        top1_dept = classes[top1_idx]
        top1_conf = round(float(probs[top1_idx]) * 100, 2)

        conf_label = self.compute_confidence_label(top1_conf)
        decision = self.compute_routing_decision(top1_conf)

        # Top 2 Predictions
        top_predictions = []
        for idx in sorted_indices[:2]:
            top_predictions.append({
                "department": classes[idx],
                "confidence": round(float(probs[idx]) * 100, 2)
            })

        # All Probabilities Mapping
        all_probabilities = {}
        for idx, cls in enumerate(classes):
            all_probabilities[cls] = round(float(probs[idx]) * 100, 2)

        return {
            "ticket": raw_text,
            "cleaned_text": cleaned_text,
            "prediction": top1_dept,
            "department": top1_dept,
            "confidence": top1_conf,
            "confidence_pct": top1_conf,
            "confidence_label": conf_label,
            "decision": decision,
            "top_predictions": top_predictions,
            "top_2_predictions": top_predictions,
            "all_probabilities": all_probabilities,
            "status": "success",
        }

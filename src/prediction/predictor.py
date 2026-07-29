"""
Ticket Categorizer Prediction Engine and Department Router.

Handles real-time single ticket classification, batch file processing, confidence scoring, priority tagging, and top-2 department recommendations.
"""

import os
from typing import Dict, Any, List, Optional
import pandas as pd
from sklearn.pipeline import Pipeline

from src.preprocessing.text_cleaner import TextCleaner
from src.prediction.confidence import ConfidenceScorer
from src.prediction.priority import PriorityTagger
from src.utils.io_utils import load_artifact, load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TicketPredictor:
    """Production predictor and routing engine for incoming support tickets."""

    def __init__(self, model_path: Optional[str] = None, config_path: str = "config/config.yaml") -> None:
        """Initializes predictor by loading config, text cleaner, confidence scorer, priority tagger, and model artifact.

        Args:
            model_path (Optional[str]): Path to trained joblib pipeline. If None, uses path from config.
            config_path (str): Path to YAML config.
        """
        self.config = load_config(config_path)
        self.cleaner = TextCleaner()
        self.scorer = ConfidenceScorer(config_path=config_path)
        self.tagger = PriorityTagger(config_path=config_path)

        if model_path is None:
            art_dir = self.config["artifacts"]["dir"]
            model_path = os.path.join(art_dir, self.config["artifacts"]["model_filename"])

        logger.info(f"Loading trained pipeline artifact from {model_path}...")
        self.pipeline: Pipeline = load_artifact(model_path)
        self.classes: List[str] = list(self.pipeline.classes_)

    def predict_single(self, text: str) -> Dict[str, Any]:
        """Classifies a single support ticket text in real time with confidence scoring and priority tagging.

        Args:
            text (str): Raw support ticket text input.

        Returns:
            Dict[str, Any]: Dictionary with prediction, confidence percentage, priority tag, top-2 department list, decision, and probabilities.
        """
        cleaned_text = self.cleaner.clean_text(text)
        res = self.scorer.predict_with_confidence(self.pipeline, text, cleaned_text)

        # Assign deterministic priority
        priority, keyword = self.tagger.assign_priority(text)
        res["priority"] = priority
        res["priority_keyword"] = keyword

        return res

    def predict_batch(self, df: pd.DataFrame, text_col: str = "ticket") -> pd.DataFrame:
        """Processes a batch dataframe of support tickets.

        Args:
            df (pd.DataFrame): Input batch dataframe.
            text_col (str): Ticket column name.

        Returns:
            pd.DataFrame: Dataframe augmented with predicted department, confidence, priority, level, and routing decision.
        """
        logger.info(f"Running batch prediction on {len(df)} records...")
        results = df.copy()

        departments = []
        confidences = []
        priorities = []
        levels = []
        decisions = []
        second_choices = []

        for text in results[text_col]:
            res = self.predict_single(str(text))
            departments.append(res["prediction"])
            confidences.append(res["confidence"])
            priorities.append(res["priority"])
            levels.append(res["confidence_label"])
            decisions.append(res["decision"])
            if len(res["top_predictions"]) > 1:
                second_choices.append(res["top_predictions"][1]["department"])
            else:
                second_choices.append("N/A")

        results["predicted_department"] = departments
        results["confidence_pct"] = confidences
        results["priority"] = priorities
        results["confidence_level"] = levels
        results["routing_decision"] = decisions
        results["secondary_department"] = second_choices

        return results

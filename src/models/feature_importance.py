"""
Feature Importance Extraction Module.

Extracts top informative TF-IDF words for each department class based on trained classifier model weights.
"""

from typing import Dict, List, Any
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.utils.logger import get_logger

logger = get_logger(__name__)


def extract_top_feature_importance(
    pipeline: Pipeline,
    top_n: int = 25
) -> Dict[str, List[Dict[str, Any]]]:
    """Extracts top_n terms per department class from trained pipeline.

    Args:
        pipeline (Pipeline): Fitted scikit-learn pipeline with 'tfidf' and 'classifier'.
        top_n (int): Number of top feature words per class. Defaults to 25.

    Returns:
        Dict[str, List[Dict[str, Any]]]: Dictionary mapping department name to list of feature dicts {feature, score}.
    """
    vectorizer = pipeline.named_steps["tfidf"]
    classifier = pipeline.named_steps["classifier"]
    feature_names = np.array(vectorizer.get_feature_names_out())
    classes = list(pipeline.classes_)

    # Unwrap CalibratedClassifierCV if applicable
    if hasattr(classifier, "calibrated_classifiers_"):
        # Average coefficients across calibrated base estimators
        coefs = np.mean([
            cal_clf.estimator.coef_ for cal_clf in classifier.calibrated_classifiers_
        ], axis=0)
    elif hasattr(classifier, "coef_"):
        coefs = classifier.coef_
    elif hasattr(classifier, "feature_log_prob_"):
        coefs = classifier.feature_log_prob_
    elif hasattr(classifier, "feature_importances_"):
        # Tree-based importance shared across classes
        importances = classifier.feature_importances_
        sorted_indices = np.argsort(importances)[::-1][:top_n]
        res_list = [{"feature": str(feature_names[i]), "score": round(float(importances[i]), 5)} for i in sorted_indices]
        return {c: res_list for c in classes}
    else:
        logger.warning("Classifier does not expose feature coefficients/importances.")
        return {}

    importance_by_class = {}

    for i, class_name in enumerate(classes):
        if coefs.ndim == 1:
            class_coefs = coefs
        else:
            class_coefs = coefs[i]

        top_indices = np.argsort(class_coefs)[::-1][:top_n]
        top_features = [
            {
                "feature": str(feature_names[idx]),
                "score": round(float(class_coefs[idx]), 5)
            }
            for idx in top_indices
        ]
        importance_by_class[class_name] = top_features

    return importance_by_class

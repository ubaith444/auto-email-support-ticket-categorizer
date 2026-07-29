"""
Unit tests for candidate model dictionary and model evaluator functions.
"""

import pytest
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from src.models.train import get_candidate_grid_models
from src.models.evaluator import evaluate_pipeline


def test_get_candidate_grid_models():
    models = get_candidate_grid_models(random_state=42)
    expected_keys = {"logistic_regression", "multinomial_nb", "linear_svm", "decision_tree", "random_forest"}
    assert set(models.keys()) == expected_keys


def test_evaluate_pipeline():
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("classifier", MultinomialNB())
    ])
    df = pd.DataFrame({
        "cleaned_ticket": ["payment failed", "cant login", "leave request", "office hours"],
        "category": ["BILLING", "TECHNICAL", "HR", "GENERAL"]
    })
    pipeline.fit(df["cleaned_ticket"], df["category"])

    metrics = evaluate_pipeline(pipeline, df, text_col="cleaned_ticket", target_col="category")
    assert "accuracy" in metrics
    assert "f1_weighted" in metrics
    assert "confusion_matrix" in metrics
    assert metrics["accuracy"] == 1.0

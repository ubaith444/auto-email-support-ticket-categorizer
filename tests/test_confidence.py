"""
Unit tests for ConfidenceScorer module and decision routing.
"""

import pytest
import numpy as np
from src.prediction.confidence import ConfidenceScorer, softmax


def test_confidence_label_classification():
    scorer = ConfidenceScorer()
    assert scorer.compute_confidence_label(96.5) == "VERY HIGH"
    assert scorer.compute_confidence_label(85.0) == "HIGH"
    assert scorer.compute_confidence_label(70.0) == "MEDIUM"
    assert scorer.compute_confidence_label(50.0) == "LOW"


def test_routing_decision():
    scorer = ConfidenceScorer()
    assert scorer.compute_routing_decision(95.0) == "AUTO ASSIGN"
    assert scorer.compute_routing_decision(65.0) == "AUTO ASSIGN"
    assert scorer.compute_routing_decision(55.0) == "NEEDS HUMAN REVIEW"


def test_softmax_function():
    scores = np.array([1.0, 2.0, 3.0])
    probs = softmax(scores)
    assert len(probs) == 3
    assert np.isclose(np.sum(probs), 1.0)
    assert probs[2] > probs[1] > probs[0]

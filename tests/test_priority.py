"""
Unit tests for PriorityTagger module, keyword overrides, threshold boundaries, and confidence logging.
"""

import pytest
from src.prediction.priority import PriorityTagger
from src.prediction.predictor import TicketPredictor


def test_priority_high_keywords():
    tagger = PriorityTagger()
    priority, kw = tagger.assign_priority("System crash in production environment! Urgent help needed.")
    assert priority == "HIGH"
    assert kw in ["crash", "production", "urgent"]


def test_priority_low_keywords():
    tagger = PriorityTagger()
    priority, kw = tagger.assign_priority("I have a quick question regarding the user guide documentation.")
    assert priority == "LOW"
    assert kw in ["question", "guide", "documentation"]


def test_priority_normal_no_keywords():
    tagger = PriorityTagger()
    priority, kw = tagger.assign_priority("Updated my contact phone number in system preferences.")
    assert priority == "NORMAL"
    assert kw is None


def test_priority_high_overrides_low():
    tagger = PriorityTagger()
    # Contains 'question' (low) AND 'urgent' (high)
    priority, kw = tagger.assign_priority("Quick question regarding urgent payment failed error.")
    assert priority == "HIGH"
    assert kw in ["urgent", "payment failed"]


def test_predictor_integration_high_confidence_high_priority():
    predictor = TicketPredictor()
    res = predictor.predict_single("My payment failed twice and this issue is urgent.")
    assert res["department"] == "BILLING"
    assert res["priority"] == "HIGH"
    assert res["confidence"] >= 60.0
    assert res["decision"] == "AUTO ASSIGN"


def test_predictor_empty_and_unknown_inputs():
    predictor = TicketPredictor()

    res_empty = predictor.predict_single("")
    assert res_empty["status"] == "warning_empty_input"
    assert res_empty["decision"] == "NEEDS HUMAN REVIEW"

    res_unknown = predictor.predict_single("xyz123 quantum particle accelerator")
    assert "priority" in res_unknown
    assert "decision" in res_unknown

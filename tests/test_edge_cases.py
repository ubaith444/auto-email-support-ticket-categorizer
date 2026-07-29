"""
Unit tests for edge cases, missing inputs, whitespace, and unknown inputs.
"""

import pytest
from src.prediction.predictor import TicketPredictor


def test_empty_string_input():
    predictor = TicketPredictor()
    res = predictor.predict_single("")
    assert res["status"] == "warning_empty_input"
    assert res["department"] == "GENERAL"


def test_whitespace_only_input():
    predictor = TicketPredictor()
    res = predictor.predict_single("     \n\t   ")
    assert res["status"] == "warning_empty_input"


def test_punctuation_and_symbols_only():
    predictor = TicketPredictor()
    res = predictor.predict_single("!@#$%^&*()_+{}[]")
    assert res["status"] == "warning_empty_input"


def test_unknown_ticket_context():
    predictor = TicketPredictor()
    res = predictor.predict_single("Quantum physics calculation in subatomic particles")
    assert "department" in res
    assert "confidence_pct" in res

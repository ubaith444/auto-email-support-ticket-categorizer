"""
Unit tests for TicketPredictor prediction engine and router.
"""

import pytest
import pandas as pd
from src.prediction.predictor import TicketPredictor


def test_predict_single_billing():
    predictor = TicketPredictor()
    res = predictor.predict_single("I was overcharged for my last invoice subscription payment.")
    assert res["department"] == "BILLING"
    assert res["confidence_pct"] > 0
    assert len(res["top_2_predictions"]) == 2


def test_predict_single_technical():
    predictor = TicketPredictor()
    res = predictor.predict_single("App keeps crashing and page is not loading API error.")
    assert res["department"] == "TECHNICAL"


def test_predict_single_hr():
    predictor = TicketPredictor()
    res = predictor.predict_single("Need my salary slip and leave request status updated.")
    assert res["department"] == "HR"


def test_predict_single_general():
    predictor = TicketPredictor()
    res = predictor.predict_single("What are your office hours and address location?")
    assert res["department"] == "GENERAL"


def test_predict_batch():
    predictor = TicketPredictor()
    batch_df = pd.DataFrame({
        "ticket": [
            "Need refund for double charge",
            "Server down cant login",
            "Offer letter pending HR",
            "How do I contact support demo"
        ]
    })
    output_df = predictor.predict_batch(batch_df)
    assert "predicted_department" in output_df.columns
    assert "confidence_pct" in output_df.columns
    assert len(output_df) == 4

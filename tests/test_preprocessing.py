"""
Unit tests for data preprocessing and text cleaner module.
"""

import pytest
import pandas as pd
from src.preprocessing.text_cleaner import TextCleaner, preprocess_dataset


def test_text_cleaner_lowercasing():
    cleaner = TextCleaner(lowercase=True)
    assert cleaner.clean_text("UNABLE TO LOGIN") == "unable to login"


def test_text_cleaner_url_removal():
    cleaner = TextCleaner(remove_urls=True)
    text = "Please check https://example.com/status for server updates"
    assert cleaner.clean_text(text) == "please check for server updates"


def test_text_cleaner_email_removal():
    cleaner = TextCleaner(remove_emails=True)
    text = "Contact user at john.doe@company.org regarding refund"
    assert cleaner.clean_text(text) == "contact user at regarding refund"


def test_text_cleaner_html_removal():
    cleaner = TextCleaner(remove_html=True)
    text = "<h1>Billing Error</h1><p>Invoice #123</p>"
    assert cleaner.clean_text(text) == "billing error invoice 123"


def test_text_cleaner_punctuation_and_special_chars():
    cleaner = TextCleaner(remove_punctuation=True)
    text = "Urgent!! Payment failed #4521..."
    assert cleaner.clean_text(text) == "urgent payment failed 4521"


def test_preprocess_dataset_pipeline():
    raw_df = pd.DataFrame({
        "ticket": ["Charged twice!", None, "Charged twice!", "   "],
        "category": ["BILLING", "BILLING", "BILLING", "BILLING"]
    })
    processed = preprocess_dataset(raw_df, text_col="ticket", target_col="category", drop_duplicates=True, drop_na=True)
    assert len(processed) == 1
    assert processed.iloc[0]["cleaned_ticket"] == "charged twice"

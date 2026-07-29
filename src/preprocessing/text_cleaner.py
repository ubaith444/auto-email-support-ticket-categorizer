"""
Text Cleaning and Normalization Module.

Provides robust text cleaning, sanitization, contraction expansion, and normalization functions for support ticket text.
"""

import re
import string
from typing import Optional, List, Dict
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Contraction mapping dictionary
CONTRACTION_MAP: Dict[str, str] = {
    "can't": "cannot",
    "cant": "cannot",
    "won't": "will not",
    "wont": "will not",
    "n't": " not",
    "'re": " are",
    "'s": " is",
    "'d": " would",
    "'ll": " will",
    "'t": " not",
    "'ve": " have",
    "'m": " am",
    "haven't": "have not",
    "hasn't": "has not",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "couldn't": "could not",
    "shouldn't": "should not",
    "wouldn't": "would not",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
}


class TextCleaner:
    """Text preprocessing pipeline for incoming customer support tickets.

    Cleaning Steps:
    1. Lowercase conversion
    2. Contraction expansion ("can't" -> "cannot", "I've" -> "I have")
    3. HTML tag removal
    4. URL removal
    5. Email removal
    6. Special character & Punctuation removal
    7. Whitespace normalization
    """

    def __init__(
        self,
        lowercase: bool = True,
        expand_contractions: bool = True,
        remove_html: bool = True,
        remove_urls: bool = True,
        remove_emails: bool = True,
        remove_punctuation: bool = True,
        remove_extra_whitespace: bool = True,
    ) -> None:
        """Initializes cleaner settings.

        Args:
            lowercase (bool): Whether to convert text to lowercase.
            expand_contractions (bool): Whether to expand english contractions.
            remove_html (bool): Whether to remove HTML tags.
            remove_urls (bool): Whether to remove web links.
            remove_emails (bool): Whether to remove email addresses.
            remove_punctuation (bool): Whether to remove punctuation and special characters.
            remove_extra_whitespace (bool): Whether to normalize whitespace.
        """
        self.lowercase = lowercase
        self.expand_contractions = expand_contractions
        self.remove_html = remove_html
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.remove_punctuation = remove_punctuation
        self.remove_extra_whitespace = remove_extra_whitespace

        # Precompile regular expressions for efficiency
        self.html_pattern = re.compile(r"<[^>]+>")
        self.url_pattern = re.compile(r"https?://\S+|www\.\S+")
        self.email_pattern = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
        self.special_char_pattern = re.compile(r"[^a-zA-Z0-9\s]")
        self.whitespace_pattern = re.compile(r"\s+")

    def expand_text_contractions(self, text: str) -> str:
        """Expands common english contractions in text.

        Args:
            text (str): Raw string.

        Returns:
            str: String with contractions expanded.
        """
        words = text.split()
        expanded_words = [CONTRACTION_MAP.get(word.lower(), word) for word in words]
        return " ".join(expanded_words)

    def clean_text(self, text: Optional[str]) -> str:
        """Cleans a single input text string.

        Args:
            text (Optional[str]): Raw input support ticket string.

        Returns:
            str: Cleaned and normalized text string.
        """
        if text is None or not isinstance(text, str) or not text.strip():
            return ""

        cleaned = text

        if self.lowercase:
            cleaned = cleaned.lower()

        if self.expand_contractions:
            cleaned = self.expand_text_contractions(cleaned)

        if self.remove_html:
            cleaned = self.html_pattern.sub(" ", cleaned)

        if self.remove_urls:
            cleaned = self.url_pattern.sub(" ", cleaned)

        if self.remove_emails:
            cleaned = self.email_pattern.sub(" ", cleaned)

        if self.remove_punctuation:
            cleaned = self.special_char_pattern.sub(" ", cleaned)

        if self.remove_extra_whitespace:
            cleaned = self.whitespace_pattern.sub(" ", cleaned).strip()

        return cleaned

    def clean_series(self, series: pd.Series) -> pd.Series:
        """Cleans a pandas Series of text strings.

        Args:
            series (pd.Series): Pandas series containing ticket texts.

        Returns:
            pd.Series: Cleaned text series.
        """
        logger.info(f"Cleaning pandas Series with {len(series)} samples...")
        return series.astype(str).apply(self.clean_text)


def preprocess_dataset(
    df: pd.DataFrame,
    text_col: str = "ticket",
    target_col: str = "category",
    drop_duplicates: bool = True,
    drop_na: bool = True,
) -> pd.DataFrame:
    """Preprocesses full dataframe by cleaning text, handling missing values, and dropping duplicate rows.

    Args:
        df (pd.DataFrame): Input dataframe.
        text_col (str): Name of ticket text column.
        target_col (str): Name of category target column.
        drop_duplicates (bool): Whether to drop duplicate rows.
        drop_na (bool): Whether to drop NA rows.

    Returns:
        pd.DataFrame: Processed dataframe ready for modeling.
    """
    logger.info("Starting dataset preprocessing pipeline...")
    cleaned_df = df.copy()

    if drop_na:
        initial_len = len(cleaned_df)
        cleaned_df = cleaned_df.dropna(subset=[text_col, target_col])
        logger.info(f"Dropped {initial_len - len(cleaned_df)} null rows.")

    cleaner = TextCleaner()
    cleaned_df["cleaned_ticket"] = cleaner.clean_series(cleaned_df[text_col])

    # Remove empty cleaned strings
    cleaned_df = cleaned_df[cleaned_df["cleaned_ticket"].str.len() > 0]

    if drop_duplicates:
        initial_len = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates(subset=["cleaned_ticket", target_col])
        logger.info(f"Dropped {initial_len - len(cleaned_df)} duplicate rows.")

    logger.info(f"Preprocessing completed. Final shape: {cleaned_df.shape}")
    return cleaned_df

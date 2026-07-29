"""
TF-IDF Feature Builder Module.

Configures and builds TfidfVectorizer instances supporting unigrams, bigrams, and trigrams.
"""

from typing import Dict, Any, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer

from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_tfidf_vectorizer(config: Dict[str, Any]) -> TfidfVectorizer:
    """Instantiates a scikit-learn TfidfVectorizer based on configuration parameters.

    Args:
        config (Dict[str, Any]): Feature engineering configuration.

    Returns:
        TfidfVectorizer: Configured vectorizer instance.
    """
    tfidf_cfg = config.get("tfidf", {})
    ngram_range_tuple = tuple(tfidf_cfg.get("ngram_range", [1, 2]))
    max_features = tfidf_cfg.get("max_features", 3000)
    min_df = tfidf_cfg.get("min_df", 1)
    max_df = tfidf_cfg.get("max_df", 0.95)
    sublinear_tf = tfidf_cfg.get("sublinear_tf", True)
    smooth_idf = tfidf_cfg.get("smooth_idf", True)
    norm = tfidf_cfg.get("norm", "l2")

    logger.info(
        f"Building TfidfVectorizer (ngram_range={ngram_range_tuple}, "
        f"max_features={max_features}, min_df={min_df}, max_df={max_df}, sublinear_tf={sublinear_tf}, norm={norm})"
    )

    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range_tuple,
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=sublinear_tf,
        smooth_idf=smooth_idf,
        norm=norm,
        stop_words="english",
    )
    return vectorizer

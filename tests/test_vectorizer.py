"""
Unit tests for TF-IDF feature builder module.
"""

import pytest
from src.features.tfidf_builder import build_tfidf_vectorizer


def test_tfidf_builder_configuration():
    config = {
        "tfidf": {
            "ngram_range": [1, 2],
            "max_features": 50,
            "min_df": 1,
            "max_df": 1.0,
            "sublinear_tf": True,
        }
    }
    vec = build_tfidf_vectorizer(config)
    assert vec.ngram_range == (1, 2)
    assert vec.max_features == 50
    assert vec.sublinear_tf is True


def test_tfidf_transform():
    config = {"tfidf": {"ngram_range": [1, 1], "max_features": 100}}
    vec = build_tfidf_vectorizer(config)
    corpus = ["payment failed invoice", "login error server down"]
    matrix = vec.fit_transform(corpus)
    assert matrix.shape[0] == 2
    assert matrix.shape[1] > 0

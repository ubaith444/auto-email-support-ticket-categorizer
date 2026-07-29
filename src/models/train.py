"""
Model Training, Hyperparameter Optimization, and Selection Pipeline Module.

Trains classical ML models (Linear SVM, Logistic Regression, Multinomial Naive Bayes, Decision Tree, Random Forest)
using scikit-learn Pipelines, performs GridSearchCV hyperparameter optimization over LinearSVC and TF-IDF parameters,
and selects the optimal model.
"""

import time
import os
from typing import Dict, Any, Tuple, List
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    GridSearchCV,
    cross_validate,
    train_test_split,
)

from src.features.tfidf_builder import build_tfidf_vectorizer
from src.preprocessing.text_cleaner import preprocess_dataset
from src.models.feature_importance import extract_top_feature_importance
from src.utils.io_utils import load_config, load_dataset, save_artifact, save_json
from src.utils.logger import get_logger

logger = get_logger(__name__)


def get_candidate_grid_models(random_state: int = 42) -> Dict[str, Dict[str, Any]]:
    """Returns candidate model instances and their hyperparameter grid search spaces.

    Args:
        random_state (int): Seed for reproducibility.

    Returns:
        Dict[str, Dict[str, Any]]: Dictionary mapping model name to classifier and param_grid.
    """
    return {
        "linear_svm": {
            "estimator": CalibratedClassifierCV(LinearSVC(random_state=random_state, max_iter=2000)),
            "param_grid": {
                "tfidf__ngram_range": [(1, 1), (1, 2)],
                "tfidf__max_features": [1000, 2000, 3000],
                "tfidf__sublinear_tf": [True, False],
                "classifier__estimator__C": [0.1, 0.5, 1.0, 2.0, 5.0],
            },
        },
        "logistic_regression": {
            "estimator": LogisticRegression(random_state=random_state, max_iter=1000, solver="lbfgs"),
            "param_grid": {
                "tfidf__ngram_range": [(1, 1), (1, 2)],
                "classifier__C": [0.1, 0.5, 1.0, 2.0, 5.0],
            },
        },
        "multinomial_nb": {
            "estimator": MultinomialNB(),
            "param_grid": {
                "tfidf__ngram_range": [(1, 1), (1, 2)],
                "classifier__alpha": [0.01, 0.1, 0.5, 1.0],
            },
        },
        "random_forest": {
            "estimator": RandomForestClassifier(random_state=random_state),
            "param_grid": {
                "classifier__n_estimators": [100, 200],
                "classifier__max_depth": [None, 15],
            },
        },
        "decision_tree": {
            "estimator": DecisionTreeClassifier(random_state=random_state),
            "param_grid": {
                "classifier__max_depth": [10, 20],
            },
        },
    }


def train_and_select_best_model(
    config: Dict[str, Any], df: pd.DataFrame
) -> Tuple[Pipeline, pd.DataFrame, pd.DataFrame, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Runs end-to-end hyperparameter tuning over LinearSVC and TF-IDF parameters, cross-validation, and selection.

    Args:
        config (Dict[str, Any]): Configuration dictionary.
        df (pd.DataFrame): Preprocessed dataframe.

    Returns:
        Tuple: (best_pipeline, df_train, df_test, summary, comparison_results, feature_importance).
    """
    text_col = "cleaned_ticket"
    target_col = config["data"].get("target_column", "category")
    test_size = config["model"].get("test_size", 0.2)
    random_state = config["model"].get("random_state", 42)

    X = df[text_col]
    y = df[target_col]

    # Train / Held-out Test split
    df_train, df_test = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=y
    )

    X_train, y_train = df_train[text_col], df_train[target_col]
    X_test, y_test = df_test[text_col], df_test[target_col]

    logger.info(f"Dataset split: Train shape={X_train.shape}, Held-out Test shape={X_test.shape}")

    candidate_specs = get_candidate_grid_models(random_state=random_state)
    comparison_results = {}

    best_model_name = None
    best_score = -1.0
    best_pipeline = None

    # Repeated Stratified K-Fold CV (5 splits x 3 repeats = 15 evaluation splits)
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=random_state)

    for name, spec in candidate_specs.items():
        logger.info(f"Performing GridSearchCV hyperparameter optimization for: {name}...")

        vectorizer = build_tfidf_vectorizer(config)
        pipe = Pipeline([("tfidf", vectorizer), ("classifier", spec["estimator"])])

        grid_search = GridSearchCV(
            pipe,
            param_grid=spec["param_grid"],
            cv=rskf,
            scoring="f1_weighted",
            n_jobs=-1,
            refit=True,
        )

        start_train_time = time.time()
        grid_search.fit(X_train, y_train)
        train_time = time.time() - start_train_time

        # Measure inference latency on test set
        fitted_pipe = grid_search.best_estimator_
        start_inf_time = time.time()
        preds = fitted_pipe.predict(X_test)
        inf_time = time.time() - start_inf_time

        # Evaluate Repeated Stratified K-Fold CV scores of best estimator
        cv_res = cross_validate(
            fitted_pipe,
            X_train,
            y_train,
            cv=rskf,
            scoring=["accuracy", "f1_weighted", "precision_weighted", "recall_weighted"],
        )

        mean_cv_f1 = float(np.mean(cv_res["test_f1_weighted"]))
        std_cv_f1 = float(np.std(cv_res["test_f1_weighted"]))
        var_cv_f1 = float(np.var(cv_res["test_f1_weighted"]))
        mean_cv_acc = float(np.mean(cv_res["test_accuracy"]))

        comparison_results[name] = {
            "best_params": {k.replace("classifier__", "").replace("tfidf__", "tfidf_"): v for k, v in grid_search.best_params_.items()},
            "cv_f1_weighted_mean": round(mean_cv_f1, 4),
            "cv_f1_weighted_std": round(std_cv_f1, 4),
            "cv_f1_weighted_var": round(var_cv_f1, 6),
            "cv_accuracy_mean": round(mean_cv_acc, 4),
            "train_time_sec": round(train_time, 4),
            "test_inference_time_sec": round(inf_time, 6),
        }

        logger.info(
            f"Algorithm [{name}] Best Params: {grid_search.best_params_} -> "
            f"Repeated 5-Fold CV F1: {mean_cv_f1:.4f} (±{std_cv_f1:.4f}), Accuracy: {mean_cv_acc:.4f}"
        )

        if mean_cv_f1 > best_score:
            best_score = mean_cv_f1
            best_model_name = name
            best_pipeline = fitted_pipe

    logger.info(
        f"Optimal Selected Classifier: '{best_model_name}' with Repeated Stratified CV F1: {best_score * 100:.2f}%"
    )

    # Extract feature importances per class
    feature_importance_dict = extract_top_feature_importance(best_pipeline, top_n=25)

    summary = {
        "best_algorithm": best_model_name,
        "best_params": comparison_results[best_model_name]["best_params"],
        "best_cv_f1_score": round(best_score, 4),
        "total_samples": len(df),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "num_classes": len(y.unique()),
        "classes": list(map(str, sorted(y.unique()))),
    }

    return best_pipeline, df_train, df_test, summary, comparison_results, feature_importance_dict


def run_training_pipeline(
    config_path: str = "config/config.yaml"
) -> Tuple[Pipeline, pd.DataFrame, pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    """Executes the full hyperparameter optimization pipeline from config to saved artifacts.

    Args:
        config_path (str): Path to config file.

    Returns:
        Tuple: (Trained pipeline, df_train, df_test, report dict, feature_importance dict).
    """
    config = load_config(config_path)
    raw_data_path = config["data"]["raw_path"]

    df_raw = load_dataset(raw_data_path)
    df_clean = preprocess_dataset(
        df_raw,
        text_col=config["data"]["text_column"],
        target_col=config["data"]["target_column"],
    )

    pipeline, df_train, df_test, summary, comparisons, feat_importance = train_and_select_best_model(
        config, df_clean
    )

    # Save artifacts
    art_dir = config["artifacts"]["dir"]
    model_path = os.path.join(art_dir, config["artifacts"]["model_filename"])
    metrics_path = os.path.join(art_dir, config["artifacts"]["metrics_filename"])

    save_artifact(pipeline, model_path)

    full_report = {
        "summary": summary,
        "model_comparisons": comparisons,
        "feature_importance": feat_importance,
    }
    save_json(full_report, metrics_path)

    logger.info("Training pipeline completed successfully.")
    return pipeline, df_train, df_test, full_report, feat_importance

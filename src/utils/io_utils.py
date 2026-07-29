"""
File I/O and Configuration Utility Module.

Provides functions for loading configurations, serializing models, and reading datasets safely.
"""

import os
import json
from typing import Any, Dict
import yaml
import joblib
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


def ensure_dir(dir_path: str) -> None:
    """Ensures that a directory exists, creating it if necessary.

    Args:
        dir_path (str): Path to the directory.
    """
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
        logger.debug(f"Created directory: {dir_path}")


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Loads YAML configuration file.

    Args:
        config_path (str): Path to config file. Defaults to 'config/config.yaml'.

    Returns:
        Dict[str, Any]: Configuration dictionary.

    Raises:
        FileNotFoundError: If configuration file is missing.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logger.info(f"Loaded configuration from {config_path}")
    return config


def load_dataset(file_path: str) -> pd.DataFrame:
    """Loads dataset from CSV file.

    Args:
        file_path (str): Path to the dataset CSV file.

    Returns:
        pd.DataFrame: Loaded DataFrame.

    Raises:
        FileNotFoundError: If dataset file is missing.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at path: {file_path}")

    df = pd.read_csv(file_path)
    logger.info(f"Loaded dataset from {file_path} with shape {df.shape}")
    return df


def save_artifact(obj: Any, file_path: str) -> None:
    """Saves a python object (model/vectorizer) using joblib.

    Args:
        obj (Any): Object to serialize.
        file_path (str): Path where the object should be saved.
    """
    ensure_dir(os.path.dirname(file_path))
    joblib.dump(obj, file_path)
    logger.info(f"Saved artifact to {file_path}")


def load_artifact(file_path: str) -> Any:
    """Loads a joblib serialized object.

    Args:
        file_path (str): Path to the joblib artifact file.

    Returns:
        Any: Deserialized python object.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Artifact file not found at: {file_path}")

    obj = joblib.load(file_path)
    logger.info(f"Loaded artifact from {file_path}")
    return obj


def save_json(data: Dict[str, Any], file_path: str) -> None:
    """Saves a dictionary as a JSON file.

    Args:
        data (Dict[str, Any]): Data dictionary.
        file_path (str): Target JSON file path.
    """
    ensure_dir(os.path.dirname(file_path))
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    logger.info(f"Saved JSON metrics to {file_path}")

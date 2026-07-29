"""
Data Visualization and Plotting Module.

Generates publication-quality charts (Confusion Matrix, Class Distribution, Ticket Length, Feature Importance, Confidence Distribution).
"""

import os
from typing import List, Dict, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.io_utils import ensure_dir
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Set global aesthetic style
sns.set_theme(style="whitegrid", palette="muted")


def plot_confusion_matrix(
    cm: List[List[int]],
    classes: List[str],
    output_path: str = "reports/confusion_matrix.png",
) -> None:
    """Generates and saves a confusion matrix heatmap."""
    ensure_dir(os.path.dirname(output_path))
    plt.figure(figsize=(7, 5))
    cm_arr = np.array(cm)

    sns.heatmap(
        cm_arr,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        cbar=True,
    )
    plt.title("Support Ticket Categorizer - Confusion Matrix", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Predicted Department", fontsize=10)
    plt.ylabel("Actual Department", fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved confusion matrix plot to {output_path}")


def plot_class_distribution(
    df: pd.DataFrame,
    target_col: str = "category",
    output_path: str = "reports/class_distribution.png",
) -> None:
    """Plots and saves department class frequency distribution."""
    ensure_dir(os.path.dirname(output_path))
    plt.figure(figsize=(7, 4))
    counts = df[target_col].value_counts().reset_index()
    counts.columns = [target_col, "count"]

    ax = sns.barplot(data=counts, x=target_col, y="count", hue=target_col, palette="viridis", legend=False)
    plt.title("Department Label Frequency Distribution", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Department Category", fontsize=10)
    plt.ylabel("Number of Tickets", fontsize=10)

    for p in ax.patches:
        ax.annotate(
            f"{int(p.get_height())}",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="center",
            xytext=(0, 5),
            textcoords="offset points",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved class distribution plot to {output_path}")


def plot_ticket_length_distribution(
    df: pd.DataFrame,
    text_col: str = "ticket",
    target_col: str = "category",
    output_path: str = "reports/ticket_length_distribution.png",
) -> None:
    """Plots ticket word count distribution grouped by department."""
    ensure_dir(os.path.dirname(output_path))
    plt.figure(figsize=(8, 5))
    df_copy = df.copy()
    df_copy["word_count"] = df_copy[text_col].astype(str).apply(lambda x: len(x.split()))

    sns.boxplot(data=df_copy, x=target_col, y="word_count", hue=target_col, palette="Set2", legend=False)
    plt.title("Ticket Word Count Distribution per Department", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Department Category", fontsize=10)
    plt.ylabel("Word Count", fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved ticket length distribution plot to {output_path}")


def plot_feature_importance(
    importance_dict: Dict[str, List[Dict[str, Any]]],
    output_dir: str = "reports",
    top_n: int = 15
) -> None:
    """Generates bar plots showing top TF-IDF features for each department."""
    ensure_dir(output_dir)

    for dept, features in importance_dict.items():
        if not features:
            continue
        df_feat = pd.DataFrame(features[:top_n])
        plt.figure(figsize=(8, 5))
        ax = sns.barplot(data=df_feat, x="score", y="feature", hue="feature", palette="Blues_r", legend=False)
        plt.title(f"Top {top_n} Informative Features for {dept}", fontsize=12, fontweight="bold", pad=12)
        plt.xlabel("Feature Score / Weight", fontsize=10)
        plt.ylabel("TF-IDF Term", fontsize=10)
        plt.tight_layout()
        out_path = os.path.join(output_dir, f"feature_importance_{dept.lower()}.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
        logger.info(f"Saved feature importance plot for {dept} to {out_path}")


def plot_confidence_distribution(
    confidences: List[float],
    output_path: str = "reports/confidence_distribution.png"
) -> None:
    """Plots histogram and KDE distribution of prediction confidence percentages."""
    ensure_dir(os.path.dirname(output_path))
    plt.figure(figsize=(7, 4))
    sns.histplot(confidences, kde=True, bins=20, color="teal")
    plt.axvline(x=60.0, color="red", linestyle="--", label="Human Review Threshold (60%)")
    plt.title("Prediction Confidence Score Distribution", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Confidence Percentage (%)", fontsize=10)
    plt.ylabel("Ticket Count", fontsize=10)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved confidence distribution plot to {output_path}")


def plot_confidence_by_department(
    df_results: pd.DataFrame,
    dept_col: str = "predicted_department",
    conf_col: str = "confidence_pct",
    output_path: str = "reports/confidence_by_department.png"
) -> None:
    """Plots prediction confidence boxplots grouped by assigned department."""
    ensure_dir(os.path.dirname(output_path))
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df_results, x=dept_col, y=conf_col, hue=dept_col, palette="Set3", legend=False)
    plt.axhline(y=60.0, color="red", linestyle="--", label="Human Review Threshold (60%)")
    plt.title("Prediction Confidence by Assigned Department", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Assigned Department", fontsize=10)
    plt.ylabel("Confidence Percentage (%)", fontsize=10)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved confidence by department plot to {output_path}")

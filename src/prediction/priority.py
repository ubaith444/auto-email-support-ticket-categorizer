"""
Deterministic Priority Tagging Module.

Assigns priority levels (HIGH, NORMAL, LOW) to incoming support tickets using configurable business rules.
This module is strictly rule-based and does NOT use machine learning.
"""

from typing import Dict, Any, List, Optional, Tuple
from src.utils.io_utils import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PriorityTagger:
    """Rule-based priority tagging engine based on keyword match precedence."""

    def __init__(self, config_path: str = "config/config.yaml") -> None:
        """Initializes PriorityTagger with configurable keyword lists from config.

        Args:
            config_path (str): Path to YAML configuration file.
        """
        self.config = load_config(config_path)
        priority_cfg = self.config.get("priority", {})
        self.high_keywords = [k.lower() for k in priority_cfg.get("high_keywords", [])]
        self.low_keywords = [k.lower() for k in priority_cfg.get("low_keywords", [])]

    def assign_priority(self, text: str) -> Tuple[str, Optional[str]]:
        """Determines ticket priority level (HIGH, NORMAL, LOW) based on keyword precedence.

        Precedence Rules:
        - HIGH overrides NORMAL and LOW if any HIGH keyword matches.
        - LOW applies if LOW keyword matches (and no HIGH keyword matches).
        - NORMAL applies if no HIGH or LOW keywords match.

        Args:
            text (str): Ticket text string.

        Returns:
            Tuple[str, Optional[str]]: (Priority Level, Triggering Keyword).
        """
        text_lower = text.lower()

        # Check for HIGH priority keywords
        for kw in self.high_keywords:
            if kw in text_lower:
                logger.info(f"Priority HIGH triggered by keyword: '{kw}'")
                return "HIGH", kw

        # Check for LOW priority keywords
        for kw in self.low_keywords:
            if kw in text_lower:
                logger.info(f"Priority LOW triggered by keyword: '{kw}'")
                return "LOW", kw

        # Default to NORMAL priority
        return "NORMAL", None

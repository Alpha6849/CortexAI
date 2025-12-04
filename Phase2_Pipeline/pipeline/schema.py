"""
schema.py

Responsible for detecting column types, identifying ID columns,
detecting target column, and building a clean schema dictionary.

Part of the CortexAI Phase 2 production pipeline.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger("SchemaDetector")
logger.setLevel(logging.INFO)

if not logger.handlers:
    console = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - [SchemaDetector] - %(levelname)s - %(message)s")
    console.setFormatter(formatter)
    logger.addHandler(console)


class SchemaDetector:
    """
    Detects schema information from a pandas DataFrame:
    - Column types
    - ID columns
    - Target column
    - Numeric / categorical / datetime grouping
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def detect(self) -> Dict:
        raise NotImplementedError("detect() not implemented yet.")

    # Column Type Detection

    def _detect_numeric_columns(self) -> List[str]:
        """Return list of numeric columns."""
        numeric_cols = self.df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        logger.info(f"Numeric columns detected: {numeric_cols}")
        return numeric_cols

    def _detect_categorical_columns(self) -> List[str]:
        """Return list of categorical columns."""
        categorical_cols = self.df.select_dtypes(include=["object"]).columns.tolist()
        logger.info(f"Categorical columns detected: {categorical_cols}")
        return categorical_cols

    def _detect_datetime_columns(self) -> List[str]:
        """Detect datetime-like columns safely (avoid converting numeric columns)."""
        datetime_cols = []

        for col in self.df.columns:
            # Skip obvious numeric columns
            if pd.api.types.is_numeric_dtype(self.df[col]):
                continue

            try:
                parsed = pd.to_datetime(self.df[col], errors="raise", infer_datetime_format=True)
                # Only consider datetime if at least 50% values parsed cleanly
                if parsed.notna().mean() > 0.5:
                    datetime_cols.append(col)
            except:
                continue

        logger.info(f"Datetime columns detected: {datetime_cols}")
        return datetime_cols

    # ID Column Detection (regex safe) (previously in phase 1 this detected even width a id since it has "id" in its spelling)

    def _detect_id_columns(self) -> List[str]:
        """
        Detects ID-like columns using:
        - regex-based name patterns (safe)
        - uniqueness ratio
        """
        import re

        id_cols = []

        # Safe ID name patterns – uses word boundaries
        id_patterns = [
            r"\bid\b",
            r"\bID\b",
            r"\bId\b",
            r"\bidentifier\b",
            r"\buuid\b",
            r"\bserial\b",
            r"\bindex\b",
            r"\bs\.?no\b",      # "s.no" or "sno"
            r"\buser[_\- ]?id\b",
            r"\bpatient[_\- ]?id\b",
            r"\btransaction[_\- ]?id\b",
        ]

        for col in self.df.columns:
            col_lower = col.lower()

            # 1) Regex match — SAFE 
            if any(re.search(pattern, col_lower) for pattern in id_patterns):
                id_cols.append(col)
                continue

            # 2) Uniqueness check
            unique_ratio = self.df[col].nunique() / len(self.df)
            if unique_ratio > 0.98:
                id_cols.append(col)

        logger.info(f"ID columns detected: {id_cols}")
        return id_cols
    
    # Target Column Detection-
    def _detect_target_column(self) -> Optional[str]:
        """
        Detect the target column using:
        - common naming patterns
        - position (last column)
        - unique value count
        """
        import re

        # Common target name patterns to coonsider (might be helpful for many csv with similar columns names idk gotta try and find out)
        target_patterns = [
            r"target", r"label", r"class", r"species", r"outcome",
            r"result", r"y", r"diagnosis", r"churn"
        ]

        for col in self.df.columns:
            col_lower = col.lower()
            if any(re.search(pattern, col_lower) for pattern in target_patterns):
                logger.info(f"Target column detected via name match: {col}")
                return col

        #  If last column is categorical (good chance it is target)
        last_col = self.df.columns[-1]
        if self.df[last_col].dtype == "object":
            logger.info(f"Target column detected as last categorical column: {last_col}")
            return last_col

        # Unique values heuristic
        unique_counts = {col: self.df[col].nunique() for col in self.df.columns}

        # columns with low unique values , since it may detect classification
        classification_candidates = [
            col for col, uniq in unique_counts.items()
            if 2 <= uniq <= 20 and self.df[col].dtype == "object"
        ]

        if classification_candidates:
            logger.info(f"Target column detected via unique-value heuristic: {classification_candidates[0]}")
            return classification_candidates[0]

        # If nothing matches
        logger.warning("No clear target column detected.")
        return None
    
    def detect(self) -> Dict:
        """
        a dictionary containing detected schema information.
        """
        logger.info("Starting schema detection...")

        schema = {
            "numeric": self._detect_numeric_columns(),
            "categorical": self._detect_categorical_columns(),
            "datetime": self._detect_datetime_columns(),
            "id_columns": self._detect_id_columns(),
            "target": self._detect_target_column()
        }

        logger.info(f"Schema detection complete: {schema}")
        return schema





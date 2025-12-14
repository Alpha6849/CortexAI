import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
import re

logger = logging.getLogger("SchemaDetector")
logger.setLevel(logging.INFO)

if not logger.handlers:
    console = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - [SchemaDetector] - %(levelname)s - %(message)s"
    )
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

    # --------------------------------------------------
    # COLUMN TYPE DETECTION
    # --------------------------------------------------
    def _detect_numeric_columns(self) -> List[str]:
        numeric_cols = self.df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        logger.info(f"Numeric columns detected: {numeric_cols}")
        return numeric_cols

    def _detect_categorical_columns(self) -> List[str]:
        categorical_cols = self.df.select_dtypes(include=["object"]).columns.tolist()
        logger.info(f"Categorical columns detected: {categorical_cols}")
        return categorical_cols

    def _detect_datetime_columns(self) -> List[str]:
        datetime_cols = []

        for col in self.df.columns:
            if pd.api.types.is_numeric_dtype(self.df[col]):
                continue
            try:
                parsed = pd.to_datetime(self.df[col], errors="raise", infer_datetime_format=True)
                if parsed.notna().mean() > 0.5:
                    datetime_cols.append(col)
            except Exception:
                continue

        logger.info(f"Datetime columns detected: {datetime_cols}")
        return datetime_cols

    # --------------------------------------------------
    # ID COLUMN DETECTION
    # --------------------------------------------------
    def _detect_id_columns(self) -> List[str]:

        id_patterns = [
            r"\bid\b", r"\bidentifier\b", r"\buuid\b", r"\bserial\b",
            r"\bindex\b", r"\bs\.?no\b",
            r"\buser[_\- ]?id\b",
            r"\bpatient[_\- ]?id\b",
            r"\btransaction[_\- ]?id\b",
        ]

        id_cols = []

        for col in self.df.columns:
            col_lower = col.lower()

            if any(re.search(p, col_lower) for p in id_patterns):
                id_cols.append(col)
                continue

            unique_ratio = self.df[col].nunique() / len(self.df)
            if unique_ratio > 0.98:
                id_cols.append(col)

        logger.info(f"ID columns detected: {id_cols}")
        return id_cols

    # --------------------------------------------------
    # TARGET COLUMN DETECTION (FIXED)
    # --------------------------------------------------
    def _detect_target_column(self) -> str:
        """
        Detect target column using:
        - Name patterns
        - Classification heuristics
        - Safe fallback (last column)
        """

        target_patterns = [
            r"target", r"label", r"class", r"species",
            r"outcome", r"result", r"diagnosis", r"churn", r"y$"
        ]

        # 1️⃣ Name-based detection
        for col in self.df.columns:
            col_lower = col.lower()
            if any(re.search(p, col_lower) for p in target_patterns):
                logger.info(f"Target detected via name match: {col}")
                return col

        # 2️⃣ Classification heuristic
        for col in self.df.columns:
            if self.df[col].dtype == "object":
                uniq = self.df[col].nunique()
                if 2 <= uniq <= 20:
                    logger.info(f"Target detected via categorical heuristic: {col}")
                    return col

        # 3️⃣ FINAL SAFE FALLBACK (IMPORTANT)
        fallback = self.df.columns[-1]
        logger.warning(
            f"No clear target detected. Falling back to last column: {fallback}"
        )
        return fallback

    # --------------------------------------------------
    # MAIN SCHEMA DETECTOR
    # --------------------------------------------------
    def detect(self) -> Dict:

        logger.info("Starting schema detection...")

        schema = {
            "numeric": self._detect_numeric_columns(),
            "categorical": self._detect_categorical_columns(),
            "datetime": self._detect_datetime_columns(),
            "id_columns": self._detect_id_columns(),
            "target": self._detect_target_column()
        }

        # Safety check
        if schema["target"] in schema["id_columns"]:
            raise ValueError(
                f"Detected target `{schema['target']}` looks like an ID column."
            )

        logger.info(f"Schema detection complete: {schema}")
        return schema
   
   






import pandas as pd
import numpy as np
from typing import Dict, List
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
        numeric_cols = self.df.select_dtypes(
            include=["int64", "float64"]
        ).columns.tolist()
        logger.info(f"Numeric columns detected: {numeric_cols}")
        return numeric_cols

    def _detect_categorical_columns(self) -> List[str]:
        categorical_cols = self.df.select_dtypes(
            include=["object"]
        ).columns.tolist()
        logger.info(f"Categorical columns detected: {categorical_cols}")
        return categorical_cols

    def _detect_datetime_columns(self) -> List[str]:
        datetime_cols = []

        for col in self.df.columns:
            if pd.api.types.is_numeric_dtype(self.df[col]):
                continue
            try:
                parsed = pd.to_datetime(
                    self.df[col],
                    errors="raise",
                    infer_datetime_format=True
                )
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
    # TARGET COLUMN DETECTION (SCORING BASED)
    # --------------------------------------------------
    def _detect_target_column(self) -> str:
        """
        Domain-agnostic target detection using structural scoring.
        Handles binary ambiguity safely.
        """

        id_cols = set(self._detect_id_columns())

        generic_target_patterns = [
            r"\btarget\b", r"\blabel\b", r"\bclass\b",
            r"\boutcome\b", r"\bresult\b", r"\by$"
        ]

        scores = {}

        for col in self.df.columns:
            if col in id_cols:
                continue

            nunique = self.df[col].nunique()
            if nunique <= 1:
                continue

            score = 0
            col_lower = col.lower()
            is_numeric = pd.api.types.is_numeric_dtype(self.df[col])
            is_object = self.df[col].dtype == "object"

            #  Weak generic name signal
            if any(re.search(p, col_lower) for p in generic_target_patterns):
                score += 3

            #  Binary dominance (core signal)
            if nunique == 2:
                score += 6

                # Penalize categorical grouping features (e.g., Sex, Gender)
                if is_object:
                    score -= 2

                # Prefer encoded numeric binary outcomes (0/1)
                if is_numeric:
                    score += 1

            # Small multi-class outcome
            elif 3 <= nunique <= 10:
                score += 2

                # Penalize ordinal numeric encodings
                if is_numeric:
                    score -= 2

            #  Penalize feature-like high cardinality
            if nunique > 50:
                score -= 3

            scores[col] = score
            logger.info(f"Target score — {col}: {score}")

        if not scores:
            raise ValueError("No valid target candidates found.")

        # Sort for transparency
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # Ambiguity detection
        if (
            len(sorted_scores) > 1 and
            sorted_scores[0][1] - sorted_scores[1][1] <= 1
        ):
            logger.warning(
                f"Ambiguous target detection between "
                f"{sorted_scores[0][0]} and {sorted_scores[1][0]}"
            )

        best_target = sorted_scores[0][0]
        logger.info(f"Selected target column: {best_target}")

        return best_target

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

        # Final safety check
        if schema["target"] in schema["id_columns"]:
            raise ValueError(
                f"Detected target `{schema['target']}` appears to be an ID column."
            )

        logger.info(f"Schema detection complete: {schema}")
        return schema

   






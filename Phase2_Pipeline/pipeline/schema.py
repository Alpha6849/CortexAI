"""
schema_detector.py

Schema validation & analysis module for CortexAI.

Design principles:
- Human-in-the-loop (target provided by UI)

"""

import pandas as pd
import logging
import re
from typing import Dict, List

# --------------------------------------------------
# LOGGER
# --------------------------------------------------
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
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.n_rows = len(df)

    # --------------------------------------------------
    # BASIC TYPE DETECTION
    # --------------------------------------------------
    def _detect_numeric_columns(self) -> List[str]:
        return self.df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    def _detect_categorical_columns(self) -> List[str]:
        return self.df.select_dtypes(include=["object"]).columns.tolist()

    def _detect_datetime_columns(self) -> List[str]:
        datetime_cols = []
        for col in self.df.columns:
            if self.df[col].dtype == "object":
                try:
                    parsed = pd.to_datetime(self.df[col], errors="raise")
                    if parsed.notna().mean() > 0.5:
                        datetime_cols.append(col)
                except Exception:
                    pass
        return datetime_cols

    # --------------------------------------------------
    # ID DETECTION (NAME-BASED ONLY)
    # --------------------------------------------------
    def _detect_id_columns(self) -> List[str]:
        id_patterns = [
            r"^id$",
            r"_id$",
            r"id$",
            r"^uuid$",
            r"^index$",
            r"^s\.?no$"
        ]

        id_cols = []
        for col in self.df.columns:
            col_lower = col.lower()
            if any(re.search(p, col_lower) for p in id_patterns):
                id_cols.append(col)

        return id_cols

    # --------------------------------------------------
    # HIGH CARDINALITY (TEXT-LIKE)
    # --------------------------------------------------
    def _detect_high_cardinality_columns(self) -> List[str]:
        high_card_cols = []

        for col in self.df.select_dtypes(include=["object"]).columns:
            unique_ratio = self.df[col].nunique() / self.n_rows
            threshold = max(0.3, 10 / self.n_rows)

            if unique_ratio > threshold:
                high_card_cols.append(col)

        return high_card_cols

    # --------------------------------------------------
    # ORDINAL NUMERIC DETECTION (TARGET-SAFE)
    # --------------------------------------------------
    def _detect_ordinal_columns(
        self,
        numeric_cols: List[str],
        target_col: str
    ) -> List[str]:

        ordinal_cols = []

        for col in numeric_cols:
            if col == target_col:
                continue

            series = self.df[col].dropna()
            if series.empty:
                continue

            if series.nunique() <= 10 and (series % 1 == 0).all():
                ordinal_cols.append(col)

        return ordinal_cols

    # --------------------------------------------------
    # CATEGORICAL SPLIT BY MISSINGNESS
    # --------------------------------------------------
    def _split_categorical_by_missing(self, cat_cols: List[str]):
        normal = []
        high_missing = []

        for col in cat_cols:
            missing_ratio = self.df[col].isna().mean()
            if missing_ratio > 0.4:
                high_missing.append(col)
            else:
                normal.append(col)

        return normal, high_missing

    # --------------------------------------------------
    # TARGET VALIDATION (USER-SELECTED)
    # --------------------------------------------------
    def _validate_target(self, target_col: str) -> Dict:
        warnings = []

        if target_col not in self.df.columns:
            raise ValueError(f"Target column `{target_col}` not found.")

        nunique = self.df[target_col].nunique()
        numeric_target = pd.api.types.is_numeric_dtype(self.df[target_col])

        if nunique <= 1:
            warnings.append("Target column is constant or near-constant.")

        if target_col in self._detect_id_columns():
            warnings.append("Target column appears to be an ID column.")

        if numeric_target and nunique <= 10:
            warnings.append(
                f"Target treated as classification due to low cardinality (n={nunique})."
            )

        task_type = (
            "regression"
            if numeric_target and nunique > 10
            else "classification"
        )

        return {
            "target": target_col,
            "task_type": task_type,
            "target_warnings": warnings
        }

    # --------------------------------------------------
    # DATASET-LEVEL WARNINGS
    # --------------------------------------------------
    def _dataset_warnings(
        self,
        numeric_cols: List[str],
        high_missing_cat: List[str]
    ) -> List[str]:

        warnings = []

        for col in high_missing_cat:
            ratio = self.df[col].isna().mean()
            warnings.append(f"High missing values in `{col}` ({ratio:.0%})")

        for col in numeric_cols:
            series = self.df[col].dropna()
            if not series.empty and series.skew() > 1.5:
                warnings.append(f"`{col}` is heavily right-skewed")

        return warnings

    # --------------------------------------------------
    # MAIN SCHEMA GENERATOR
    # --------------------------------------------------
    def detect(self, target_col: str) -> Dict:
        logger.info("Starting schema detection & validation")

        numeric_cols = self._detect_numeric_columns()
        categorical_cols = self._detect_categorical_columns()
        datetime_cols = self._detect_datetime_columns()
        id_columns = self._detect_id_columns()
        high_cardinality = self._detect_high_cardinality_columns()

        ordinal_cols = self._detect_ordinal_columns(numeric_cols, target_col)

        numeric_continuous = [
            c for c in numeric_cols
            if c not in ordinal_cols and c != target_col and c not in id_columns
        ]

        categorical_cols = [
            c for c in categorical_cols
            if c not in high_cardinality and c not in id_columns
        ]

        categorical_clean, categorical_high_missing = (
            self._split_categorical_by_missing(categorical_cols)
        )

        target_info = self._validate_target(target_col)

        schema = {
            "target": target_info["target"],
            "task_type": target_info["task_type"],

            "numeric": numeric_continuous,
            "ordinal": ordinal_cols,
            "categorical": categorical_clean,

            "high_missing_categorical": categorical_high_missing,
            "high_cardinality_columns": high_cardinality,
            "id_columns": id_columns,
            "datetime": datetime_cols,

            "warnings": (
                target_info["target_warnings"]
                + self._dataset_warnings(numeric_continuous, categorical_high_missing)
            )
        }

        logger.info("Schema detection complete")
        return schema

"""
cleaner.py

Schema-driven data cleaning for CortexAI.

RULES:
- Obeys schema strictly
- Never re-detects anything
- Never modifies target
- Safe defaults (no destructive ops)
"""

import pandas as pd
from typing import Dict
import logging

# --------------------------------------------------
# LOGGER
# --------------------------------------------------
logger = logging.getLogger("DataCleaner")
logger.setLevel(logging.INFO)

if not logger.handlers:
    console = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - [DataCleaner] - %(levelname)s - %(message)s"
    )
    console.setFormatter(formatter)
    logger.addHandler(console)


class DataCleaner:

    def __init__(self, df: pd.DataFrame, schema: Dict):
        self.original_shape = df.shape
        self.df = df.copy()
        self.schema = schema
        self.target = schema["target"]
        self.report = {}

        if self.target not in self.df.columns:
            raise ValueError("Target column missing before cleaning.")

    # --------------------------------------------------
    # MAIN PIPELINE
    # --------------------------------------------------
    def clean(self):
        logger.info("Starting schema-driven cleaning pipeline")

        self._drop_columns()
        self._handle_missing_values()
        self._fix_column_types()

        cleaned_df = self.df.reset_index(drop=True)
        self.report["final_shape"] = cleaned_df.shape
        self.report["target_preserved"] = self.target

        logger.info("Cleaning completed successfully")
        return cleaned_df, self.report

    # --------------------------------------------------
    # DROP COLUMNS (SCHEMA-DRIVEN)
    # --------------------------------------------------
    def _drop_columns(self):
        dropped = []

        for key in [
            "id_columns",
            "high_cardinality_columns",
            "high_missing_categorical"
        ]:
            cols = self.schema.get(key, [])
            cols = [c for c in cols if c in self.df.columns and c != self.target]

            if cols:
                self.df.drop(columns=cols, inplace=True)
                dropped.extend(cols)
                self.report[key] = cols

        if dropped:
            logger.info(f"Dropped columns: {dropped}")
        else:
            logger.info("No columns dropped")

    # --------------------------------------------------
    # MISSING VALUES (FEATURES ONLY)
    # --------------------------------------------------
    def _handle_missing_values(self):
        missing_info = {}

        # Numeric → median
        for col in self.schema.get("numeric", []):
            if col == self.target or col not in self.df.columns:
                continue

            if self.df[col].isna().any():
                median_val = self.df[col].median()
                self.df[col].fillna(median_val, inplace=True)
                missing_info[col] = f"filled with median ({median_val})"

        # Categorical → mode
        for col in self.schema.get("categorical", []):
            if col == self.target or col not in self.df.columns:
                continue

            if self.df[col].isna().any():
                mode_val = self.df[col].mode().iloc[0]
                self.df[col].fillna(mode_val, inplace=True)
                missing_info[col] = f"filled with mode ({mode_val})"

        self.report["missing_values"] = missing_info

    # --------------------------------------------------
    # TYPE CASTING
    # --------------------------------------------------
    def _fix_column_types(self):
        type_changes = {}

        for col in self.schema.get("numeric", []):
            if col in self.df.columns and col != self.target:
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce")
                type_changes[col] = "numeric"

        for col in self.schema.get("ordinal", []):
            if col in self.df.columns and col != self.target:
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce")
                type_changes[col] = "ordinal"

        for col in self.schema.get("categorical", []):
            if col in self.df.columns and col != self.target:
                self.df[col] = self.df[col].astype(str)
                type_changes[col] = "categorical"

        self.report["type_casting"] = type_changes





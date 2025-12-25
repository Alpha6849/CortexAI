"""
cleaner.py

Responsible for automated data cleaning based on validated schema:
- Remove ID columns
- Handle missing values (FEATURES ONLY)
- Cast column types properly
- Basic outlier handling (FEATURES ONLY)
- Detect high-cardinality categorical features

 NEVER modifies target column
 Compatible with user-selected target architecture
"""

import pandas as pd
import numpy as np
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
        self.df = df.copy()  # safe copy
        self.schema = schema
        self.target = schema.get("target")
        self.report = {}

        if self.target not in self.df.columns:
            raise ValueError("Target column missing before cleaning.")

    # --------------------------------------------------
    # MAIN PIPELINE
    # --------------------------------------------------
    def clean(self):
        logger.info("Starting data cleaning pipeline...")

        self._drop_id_columns()
        self._handle_missing_values()
        self._fix_column_types()
        self._fix_outliers()
        self._detect_high_cardinality()

        logger.info("Cleaning pipeline completed.")

        cleaned_df = self.df.reset_index(drop=True)
        report = self._generate_report()

        return cleaned_df, report

    # --------------------------------------------------
    # ID COLUMN REMOVAL
    # --------------------------------------------------
    def _drop_id_columns(self):
        id_cols = self.schema.get("id_columns", [])

        if not id_cols:
            logger.info("No ID columns to remove.")
            return

        # Never drop target even if misclassified
        id_cols = [c for c in id_cols if c != self.target]

        self.df.drop(columns=id_cols, inplace=True, errors="ignore")
        logger.info(f"Removed ID columns: {id_cols}")

        self.report["removed_id_columns"] = id_cols

    # --------------------------------------------------
    # MISSING VALUES (FEATURES ONLY)
    # --------------------------------------------------
    def _handle_missing_values(self):
        missing_info = {}

        # Numeric features → median
        for col in self.schema.get("numeric", []):
            if col == self.target or col not in self.df.columns:
                continue

            if self.df[col].isna().sum() > 0:
                median_val = self.df[col].median()
                self.df[col].fillna(median_val, inplace=True)
                missing_info[col] = f"filled with median ({median_val})"

        # Categorical features → mode
        for col in self.schema.get("categorical", []):
            if col == self.target or col not in self.df.columns:
                continue

            if self.df[col].isna().sum() > 0:
                mode_val = self.df[col].mode().iloc[0]
                self.df[col].fillna(mode_val, inplace=True)
                missing_info[col] = f"filled with mode ({mode_val})"

        if missing_info:
            logger.info(f"Missing values handled: {missing_info}")
            self.report["missing_values"] = missing_info
        else:
            logger.info("No missing values detected.")

    # --------------------------------------------------
    # TYPE CASTING (FEATURES ONLY)
    # --------------------------------------------------
    def _fix_column_types(self):
        type_changes = {}

        # Numeric
        for col in self.schema.get("numeric", []):
            if col == self.target or col not in self.df.columns:
                continue
            self.df[col] = pd.to_numeric(self.df[col], errors="coerce")
            type_changes[col] = "to_numeric"

        # Categorical
        for col in self.schema.get("categorical", []):
            if col == self.target or col not in self.df.columns:
                continue
            self.df[col] = self.df[col].astype(str)
            type_changes[col] = "to_string"

        # Datetime
        for col in self.schema.get("datetime", []):
            if col == self.target or col not in self.df.columns:
                continue
            self.df[col] = pd.to_datetime(self.df[col], errors="coerce")
            type_changes[col] = "to_datetime"

        if type_changes:
            logger.info(f"Column types fixed: {type_changes}")
            self.report["type_casting"] = type_changes
        else:
            logger.info("No type changes required.")

    # --------------------------------------------------
    # OUTLIER HANDLING (FEATURES ONLY)
    # --------------------------------------------------
    def _fix_outliers(self):
        outlier_info = {}

        for col in self.schema.get("numeric", []):
            if col == self.target or col not in self.df.columns:
                continue

            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1

            if IQR == 0:
                continue

            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR

            outliers = self.df[(self.df[col] < lower) | (self.df[col] > upper)]

            if len(outliers) > 0:
                median_val = self.df[col].median()
                self.df.loc[self.df[col] < lower, col] = median_val
                self.df.loc[self.df[col] > upper, col] = median_val

                outlier_info[col] = f"{len(outliers)} outliers replaced with median ({median_val})"

        if outlier_info:
            logger.info(f"Outliers handled: {outlier_info}")
            self.report["outliers"] = outlier_info
        else:
            logger.info("No outliers detected.")

    # --------------------------------------------------
    # HIGH CARDINALITY CHECK
    # --------------------------------------------------
    def _detect_high_cardinality(self):
        high_card_cols = {}

        for col in self.schema.get("categorical", []):
            if col == self.target or col not in self.df.columns:
                continue

            unique_count = self.df[col].nunique()
            if unique_count > 20:
                high_card_cols[col] = unique_count

        if high_card_cols:
            logger.info(f"High-cardinality categorical columns: {high_card_cols}")
            self.report["high_cardinality"] = high_card_cols
        else:
            logger.info("No high-cardinality categorical columns.")

    # --------------------------------------------------
    # REPORT
    # --------------------------------------------------
    def _generate_report(self) -> Dict:
        return {
            "removed_id_columns": self.report.get("removed_id_columns", []),
            "missing_values": self.report.get("missing_values", {}),
            "type_casting": self.report.get("type_casting", {}),
            "outliers": self.report.get("outliers", {}),
            "high_cardinality": self.report.get("high_cardinality", {})
        }






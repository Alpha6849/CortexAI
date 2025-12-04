"""
cleaner.py

Responsible for automated data cleaning based on detected schema:
- Remove ID columns
- Handle missing values
- Cast column types properly
- Basic outlier and consistency fixes

Part of the CortexAI Phase 2 production pipeline.
"""

import pandas as pd
import numpy as np
from typing import Dict, List
import logging

#logger 
logger = logging.getLogger("DataCleaner")
logger.setLevel(logging.INFO)

if not logger.handlers:
    console = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - [DataCleaner] - %(levelname)s - %(message)s")
    console.setFormatter(formatter)
    logger.addHandler(console)


class DataCleaner:

    def __init__(self, df: pd.DataFrame, schema: Dict):
        self.df = df.copy()  # safe copy
        self.schema = schema
        self.report = {}  # store cleaning summary

    def clean(self) -> pd.DataFrame:
        """
        Run cleaning steps in correct order.
        """
        logger.info("Starting data cleaning pipeline...")

        self._drop_id_columns()
        self._handle_missing_values()

        logger.info("Cleaning pipeline completed.")
        return self.df.reset_index(drop=True)




    
    def _drop_id_columns(self):
        """Remove ID columns based on schema. , or else model acc = 100% cause it will use labels to detect"""
        id_cols = self.schema.get("id_columns", [])

        if not id_cols:
            logger.info("No ID columns to remove.")
            return

        self.df.drop(columns=id_cols, inplace=True)
        logger.info(f"Removed ID columns: {id_cols}")

        # Save in cleaning report
        self.report["removed_id_columns"] = id_cols
        
        
    def _handle_missing_values(self):
        """Fill missing values based on column types."""
        missing_info = {}

        # Numeric columns → median
        for col in self.schema.get("numeric", []):
            if col not in self.df.columns:   # 👈 SKIP if column was removed
                continue

            if self.df[col].isna().sum() > 0:
                median_val = self.df[col].median()
                self.df[col].fillna(median_val, inplace=True)
                missing_info[col] = f"filled with median ({median_val})"

        # Categorical columns → mode
        for col in self.schema.get("categorical", []):
            if col not in self.df.columns:   # 👈 SKIP if column is missing
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


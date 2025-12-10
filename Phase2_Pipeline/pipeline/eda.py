"""
eda.py

Automated EDA module for CortexAI.
Generates dataset statistics, visual summaries,
and insights to support automatic ML decisions.

Part of the Phase 2 production pipeline.
"""

import os
import pandas as pd
import logging
from typing import Dict, Optional

logger = logging.getLogger("EDAEngine")
logger.setLevel(logging.INFO)


class EDAEngine:
    """
    Generates visual and statistical EDA for tabular datasets.

    Inputs:
    - Pandas DataFrame (already cleaned)
    - Schema dictionary (from SchemaDetector)

    Outputs:
    - EDA report dictionary (metadata for UI & LLM)
    - Saved plots in results folder 
    """

    def __init__(self, df: pd.DataFrame, schema: Dict, output_dir: Optional[str] = None):
        self.df = df
        self.schema = schema
        self.output_dir = output_dir or "eda_results"
        self.report = {}

        # directory to store output plots
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"EDA output directory set: {self.output_dir}")
        
    def generate_basic_statistics(self) -> Dict:
        """
        Compute key statistical info about the dataset.
        """

        stats = {
            "shape": self.df.shape,
            "data_types": self.df.dtypes.apply(lambda x: str(x)).to_dict(),
            "missing_values": self.df.isna().sum().to_dict()
        }

        # Unique counts for categorical columns
        categorical_cols = self.schema.get("categorical", [])
        stats["unique_counts"] = {
            col: self.df[col].nunique() for col in categorical_cols if col in self.df.columns
        }

        # Basic numeric stats 
        numeric_cols = self.schema.get("numeric", [])
        numeric_cols = [col for col in numeric_cols if col in self.df.columns]

        stats["numeric_summary"] = self.df[numeric_cols].describe().to_dict()

        self.report["basic_statistics"] = stats
        logger.info("Basic statistics generated.")

        return stats



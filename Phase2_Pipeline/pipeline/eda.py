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
    
    def analyze_target_column(self) -> Dict:
       
        target_col = self.schema.get("target")
        if not target_col or target_col not in self.df.columns:
            logger.warning("No target column found in dataframe.")
            return {}

        target_data = self.df[target_col]
        result = {"target_column": target_col}

        # Classification case 
        if target_data.dtype == "object" or target_data.nunique() <= 20:
            value_counts = target_data.value_counts().to_dict()
            result["type"] = "classification"
            result["class_distribution"] = value_counts

        # Regression case 
        else:
            result["type"] = "regression"
            result["summary"] = {
                "min": float(target_data.min()),
                "max": float(target_data.max()),
                "mean": float(target_data.mean()),
                "std": float(target_data.std()),
            }

        self.report["target_analysis"] = result
        logger.info(f"Target analysis complete: {result}")

        return result
    
    def analyze_numeric_columns(self) -> Dict:
        """
        (hist & boxplot suggestion).
        """
        numeric_cols = [
            col for col in self.schema.get("numeric", [])
            if col in self.df.columns
        ]

        numeric_info = {}

        for col in numeric_cols:
            col_data = self.df[col]
            numeric_info[col] = {
                "mean": float(col_data.mean()),
                "median": float(col_data.median()),
                "std": float(col_data.std()),
                "min": float(col_data.min()),
                "max": float(col_data.max()),
                "skewness": float(col_data.skew()),
                "suggest_plots": ["hist", "box"]
            }

        self.report["numeric_analysis"] = numeric_info
        logger.info("Numeric column analysis completed.")

        return numeric_info
    
    def analyze_correlations(self) -> Dict:
        """
        Analyze numeric column correlations to detect multicollinearity.
        """
        numeric_cols = [
            col for col in self.schema.get("numeric", [])
            if col in self.df.columns
        ]

        if len(numeric_cols) < 2:
            logger.info("Not enough numeric columns for correlation analysis.")
            return {}

        corr_matrix = self.df[numeric_cols].corr().round(3)
        high_corr_pairs = {}

        # Check pairs (i < j) to avoid duplicates
        for i in range(len(numeric_cols)):
            for j in range(i + 1, len(numeric_cols)):
                col1, col2 = numeric_cols[i], numeric_cols[j]
                corr_value = abs(corr_matrix.loc[col1, col2])

                if corr_value >= 0.8:  # Threshold for high correlation
                    high_corr_pairs[f"{col1} & {col2}"] = corr_matrix.loc[col1, col2]

        self.report["correlation_matrix"] = corr_matrix.to_dict()
        self.report["high_correlation_pairs"] = high_corr_pairs

        if high_corr_pairs:
            logger.info(f"High correlations detected: {high_corr_pairs}")
        else:
            logger.info("No high-correlation feature pairs found.")

        return {
            "matrix": corr_matrix.to_dict(),
            "high_pairs": high_corr_pairs
        }
        
    def refine_plot_suggestions(self):
        """
        Enhance plot suggestions using skewness + correlation insights.
        """
        numeric_info = self.report.get("numeric_analysis", {})
        high_corr = self.report.get("high_correlation_pairs", {})

        # If no numeric info exists, skip refinement
        if not numeric_info:
            return

        for col, info in numeric_info.items():
            # Skewness-based suggestions
            skew = info.get("skewness", 0)
            if abs(skew) > 1:
                info["insight"] = "Highly skewed distribution — consider transformation"
                if "box" not in info["suggest_plots"]:
                    info["suggest_plots"].append("box")

        # Correlation-based plot suggestions
        for pair, corr_val in high_corr.items():
            col1, col2 = pair.split(" & ")

            if col1 in numeric_info:
                numeric_info[col1]["suggest_plots"].append(f"scatter_with:{col2}")

            if col2 in numeric_info:
                numeric_info[col2]["suggest_plots"].append(f"scatter_with:{col1}")

        self.report["numeric_analysis"] = numeric_info
        logger.info("Plot suggestions refined using skewness & correlations.")
        return numeric_info
    
    def generate_report(self) -> Dict:
        """
        full EDA insights report.
        """
        if "basic_statistics" not in self.report:
            self.generate_basic_statistics()

        if "target_analysis" not in self.report:
            self.analyze_target_column()

        if "numeric_analysis" not in self.report:
            self.analyze_numeric_columns()

        if "correlation_matrix" not in self.report:
            self.analyze_correlations()

        # refining after the above are computed
        self.refine_plot_suggestions()

        logger.info("Final EDA report generated.")
        return self.report








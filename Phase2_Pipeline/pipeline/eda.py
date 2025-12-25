"""
eda.py

Schema-driven EDA module for CortexAI.

RULES:
- Uses CLEANED dataframe only
- Fully trusts schema (no guessing)
- Never mutates data
- Ordinal ≠ Numeric ≠ Categorical
- Target is NEVER modified
"""

import os
import pandas as pd
import logging
from typing import Dict, Optional

# --------------------------------------------------
# LOGGER
# --------------------------------------------------
logger = logging.getLogger("EDAEngine")
logger.setLevel(logging.INFO)

if not logger.handlers:
    console = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - [EDAEngine] - %(levelname)s - %(message)s"
    )
    console.setFormatter(formatter)
    logger.addHandler(console)


class EDAEngine:
    """
    Generates statistical EDA for tabular datasets.

    Inputs:
    - Cleaned DataFrame
    - Validated schema

    Output:
    - Structured EDA report for UI + LLM
    """

    def __init__(self, df: pd.DataFrame, schema: Dict, output_dir: Optional[str] = None):
        self.df = df
        self.schema = schema
        self.target = schema["target"]
        self.task_type = schema["task_type"]
        self.output_dir = output_dir or "eda_results"
        self.report = {}

        if self.target not in self.df.columns:
            raise ValueError("Target column missing in EDA input.")

        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"EDA output directory set: {self.output_dir}")

    # --------------------------------------------------
    # BASIC STATISTICS
    # --------------------------------------------------
    def generate_basic_statistics(self) -> Dict:
        stats = {
            "shape": self.df.shape,
            "data_types": self.df.dtypes.astype(str).to_dict(),
            "missing_values": self.df.isna().sum().to_dict(),
        }

        categorical_cols = [
            c for c in self.schema.get("categorical", [])
            if c in self.df.columns and c != self.target
        ]

        stats["unique_counts"] = {
            col: self.df[col].nunique()
            for col in categorical_cols
        }

        numeric_cols = [
            c for c in (
                self.schema.get("numeric", []) +
                self.schema.get("ordinal", [])
            )
            if c in self.df.columns and c != self.target
        ]

        if numeric_cols:
            stats["numeric_summary"] = self.df[numeric_cols].describe().to_dict()
        else:
            stats["numeric_summary"] = {}

        self.report["basic_statistics"] = stats
        logger.info("Basic statistics generated.")
        return stats

    # --------------------------------------------------
    # TARGET ANALYSIS
    # --------------------------------------------------
    def analyze_target_column(self) -> Dict:
        target_data = self.df[self.target]

        result = {
            "target_column": self.target,
            "task_type": self.task_type
        }

        if self.task_type == "classification":
            result["class_distribution"] = target_data.value_counts().to_dict()
        else:
            result["summary"] = {
                "min": float(target_data.min()),
                "max": float(target_data.max()),
                "mean": float(target_data.mean()),
                "std": float(target_data.std()),
                "skewness": float(target_data.skew())
            }

        self.report["target_analysis"] = result
        logger.info("Target analysis completed.")
        return result

    # --------------------------------------------------
    # NUMERIC FEATURE ANALYSIS (CONTINUOUS)
    # --------------------------------------------------
    def analyze_numeric_columns(self) -> Dict:
        numeric_cols = [
            c for c in self.schema.get("numeric", [])
            if c in self.df.columns and c != self.target
        ]

        numeric_info = {}

        for col in numeric_cols:
            data = self.df[col]
            numeric_info[col] = {
                "mean": float(data.mean()),
                "median": float(data.median()),
                "std": float(data.std()),
                "min": float(data.min()),
                "max": float(data.max()),
                "skewness": float(data.skew()),
                "suggest_plots": ["hist", "box"]
            }

        self.report["numeric_analysis"] = numeric_info
        logger.info("Numeric feature analysis completed.")
        return numeric_info

    # --------------------------------------------------
    # ORDINAL FEATURE ANALYSIS
    # --------------------------------------------------
    def analyze_ordinal_columns(self) -> Dict:
        ordinal_cols = [
            c for c in self.schema.get("ordinal", [])
            if c in self.df.columns and c != self.target
        ]

        ordinal_info = {}

        for col in ordinal_cols:
            ordinal_info[col] = {
                "value_counts": self.df[col].value_counts().to_dict(),
                "min": int(self.df[col].min()),
                "max": int(self.df[col].max())
            }

        self.report["ordinal_analysis"] = ordinal_info
        logger.info("Ordinal feature analysis completed.")
        return ordinal_info

    # --------------------------------------------------
    # CORRELATION ANALYSIS (NUMERIC + ORDINAL)
    # --------------------------------------------------
    def analyze_correlations(self) -> Dict:
        corr_cols = [
            c for c in (
                self.schema.get("numeric", []) +
                self.schema.get("ordinal", [])
            )
            if c in self.df.columns and c != self.target
        ]

        if len(corr_cols) < 2:
            logger.info("Not enough columns for correlation analysis.")
            self.report["correlation_matrix"] = {}
            self.report["high_correlation_pairs"] = {}
            return {}

        corr_matrix = self.df[corr_cols].corr().round(3)
        high_corr_pairs = {}

        for i in range(len(corr_cols)):
            for j in range(i + 1, len(corr_cols)):
                c1, c2 = corr_cols[i], corr_cols[j]
                corr_val = abs(corr_matrix.loc[c1, c2])
                if corr_val >= 0.8:
                    high_corr_pairs[f"{c1} & {c2}"] = corr_matrix.loc[c1, c2]

        self.report["correlation_matrix"] = corr_matrix.to_dict()
        self.report["high_correlation_pairs"] = high_corr_pairs

        logger.info("Correlation analysis completed.")
        return {
            "matrix": corr_matrix.to_dict(),
            "high_pairs": high_corr_pairs
        }

    # --------------------------------------------------
    # BINARY OUTCOME ANALYSIS (CATEGORICAL vs TARGET)
    # --------------------------------------------------
    def analyze_binary_outcomes(self) -> Dict:
        if self.task_type != "classification":
            self.report["binary_outcome_analysis"] = {}
            return {}

        if self.df[self.target].nunique() != 2:
            self.report["binary_outcome_analysis"] = {}
            return {}

        outcome_analysis = {}

        for col in self.schema.get("categorical", []):
            if col not in self.df.columns or col == self.target:
                continue

            try:
                rates = (
                    self.df
                    .groupby(col)[self.target]
                    .mean()
                    .round(3)
                    .to_dict()
                )

                if len(rates) > 1:
                    outcome_analysis[col] = rates

            except Exception:
                continue

        self.report["binary_outcome_analysis"] = outcome_analysis
        logger.info("Binary outcome analysis completed.")
        return outcome_analysis

    # --------------------------------------------------
    # KEY INSIGHTS (HUMAN-READABLE)
    # --------------------------------------------------
    def generate_key_insights(self):
        insights = []

        # Target distribution
        if self.task_type == "classification":
            dist = self.report.get("target_analysis", {}).get("class_distribution", {})
            if dist:
                total = sum(dist.values())
                ratios = {k: round(v / total, 2) for k, v in dist.items()}
                insights.append(f"Target distribution: {ratios}")

        # Skewed numeric features
        for col, info in self.report.get("numeric_analysis", {}).items():
            if abs(info.get("skewness", 0)) > 1:
                insights.append(f"{col} is highly skewed")

        # Strong categorical signals
        for col, rates in self.report.get("binary_outcome_analysis", {}).items():
            if rates:
                insights.append(f"{col} shows strong relationship with target")

        self.report["key_insights"] = insights
        logger.info("Key insights generated.")

    # --------------------------------------------------
    # FINAL REPORT
    # --------------------------------------------------
    def generate_report(self) -> Dict:
        self.generate_basic_statistics()
        self.analyze_target_column()
        self.analyze_numeric_columns()
        self.analyze_ordinal_columns()
        self.analyze_correlations()
        self.analyze_binary_outcomes()
        self.generate_key_insights()

        logger.info("Final EDA report generated.")
        return self.report







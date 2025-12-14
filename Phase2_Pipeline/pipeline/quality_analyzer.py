"""
quality_analyzer.py

Evaluates dataset suitability for Machine Learning.
Provides a learnability score, warnings, and recommendations.

Part of CortexAI Phase 2+ (Product Intelligence Layer).
"""

from typing import Dict, List
import numpy as np


class DatasetQualityAnalyzer:
    """
    Analyzes dataset quality and ML learnability
    using schema, EDA, and training results.
    """

    def __init__(
        self,
        schema: Dict,
        eda_report: Dict,
        training_results: Dict,
        baseline_name: str = "Baseline"
    ):
        self.schema = schema
        self.eda = eda_report
        self.training = training_results
        self.baseline_name = baseline_name

        self.score = 100
        self.reasons: List[str] = []
        self.recommendations: List[str] = []

    # --------------------------------------------------
    # PUBLIC API
    # --------------------------------------------------
    def analyze(self) -> Dict:
        """
        Run full dataset quality analysis.
        """
        self._check_id_dominance()
        self._check_target_imbalance()
        self._check_model_improvement()
        self._finalize_score()

        return {
            "learnability_score": max(self.score, 0),
            "verdict": self._verdict(),
            "reasons": self.reasons,
            "recommendations": self.recommendations
        }

    # --------------------------------------------------
    # CHECKS
    # --------------------------------------------------
    def _check_id_dominance(self):
        numeric_cols = self.schema.get("numeric", [])
        id_cols = self.schema.get("id_columns", [])

        if not numeric_cols:
            self.score -= 30
            self.reasons.append(
                "No meaningful numeric features detected."
            )
            self.recommendations.append(
                "Add real-valued features relevant to the prediction task."
            )
            return

        id_ratio = len(id_cols) / max(len(numeric_cols), 1)

        if id_ratio > 0.5:
            self.score -= 25
            self.reasons.append(
                "A large portion of features appear to be identifiers."
            )
            self.recommendations.append(
                "Remove ID-like columns or replace them with domain features."
            )

    def _check_target_imbalance(self):
        target_info = self.eda.get("target_analysis", {})

        if not target_info:
            return

        if target_info.get("type") == "classification":
            class_dist = target_info.get("class_distribution", {})
            if not class_dist:
                return

            counts = np.array(list(class_dist.values()))
            imbalance_ratio = counts.max() / max(counts.min(), 1)

            if imbalance_ratio > 10:
                self.score -= 20
                self.reasons.append(
                    f"Severe target class imbalance detected (ratio ≈ {imbalance_ratio:.1f}:1)."
                )
                self.recommendations.append(
                    "Consider resampling, class weighting, or reframing the problem."
                )

    def _check_model_improvement(self):
        if self.baseline_name not in self.training:
            return

        baseline_score = self.training[self.baseline_name]["cv_mean_score"]
        best_score = max(
            v["cv_mean_score"] for v in self.training.values()
        )

        if baseline_score <= 0:
            return

        improvement_factor = best_score / baseline_score

        if improvement_factor < 2:
            self.score -= 30
            self.reasons.append(
                "Models barely outperform the baseline."
            )
            self.recommendations.append(
                "Add stronger features or reconsider the prediction target."
            )

    # --------------------------------------------------
    # FINALIZATION
    # --------------------------------------------------
    def _finalize_score(self):
        if not self.reasons:
            self.reasons.append(
                "No major data quality issues detected."
            )

    def _verdict(self) -> str:
        if self.score >= 70:
            return "High ML potential"
        if self.score >= 40:
            return "Moderate ML potential"
        return "Low ML potential"

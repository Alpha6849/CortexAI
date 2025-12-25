"""
quality_analyzer.py

Evaluates dataset suitability for Machine Learning.
Separates:
- Strengths
- Risks
- Recommendations

CortexAI Phase 2 — Product Intelligence Layer
"""

from typing import Dict, List
import numpy as np


class DatasetQualityAnalyzer:

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

        self.score = 50  # neutral baseline

        self.risks: List[str] = []
        self.strengths: List[str] = []
        self.recommendations: List[str] = []

    # --------------------------------------------------
    # PUBLIC API
    # --------------------------------------------------
    def analyze(self) -> Dict:
        self._check_target_imbalance()
        self._check_model_improvement()
        self._check_feature_richness()
        self._finalize_score()

        return {
            "learnability_score": min(max(self.score, 0), 100),
            "verdict": self._verdict(),
            "strengths": self.strengths,
            "risks": self.risks,
            "recommendations": self.recommendations
        }

    # --------------------------------------------------
    # CHECKS
    # --------------------------------------------------
    def _check_target_imbalance(self):
        target_info = self.eda.get("target_analysis", {})

        if target_info.get("task_type") != "classification":
            return

        class_dist = target_info.get("class_distribution", {})
        if not class_dist:
            return

        counts = np.array(list(class_dist.values()))
        imbalance_ratio = counts.max() / max(counts.min(), 1)

        if imbalance_ratio > 10:
            self.score -= 15
            self.risks.append(
                f"Severe class imbalance detected (≈ {imbalance_ratio:.1f}:1)."
            )
            self.recommendations.append(
                "Consider resampling techniques or class-weighted models."
            )

    def _check_model_improvement(self):
        if self.baseline_name not in self.training:
            return

        baseline_score = self.training[self.baseline_name]["cv_mean_score"]
        best_score = max(v["cv_mean_score"] for v in self.training.values())

        absolute_gain = best_score - baseline_score

        if absolute_gain >= 0.25:
            self.score += 30
            self.strengths.append(
                f"Models significantly outperform the baseline (+{absolute_gain:.2f})."
            )

        elif absolute_gain >= 0.10:
            self.score += 15
            self.strengths.append(
                f"Models moderately outperform the baseline (+{absolute_gain:.2f})."
            )
            self.recommendations.append(
                "Feature engineering could further improve performance."
            )

        else:
            self.score -= 20
            self.risks.append(
                "Models show limited improvement over the baseline."
            )
            self.recommendations.append(
                "Consider revisiting feature selection or problem framing."
            )

    def _check_feature_richness(self):
        feature_count = (
            len(self.schema.get("numeric", [])) +
            len(self.schema.get("ordinal", [])) +
            len(self.schema.get("categorical", []))
        )

        if feature_count >= 6:
            self.score += 10
            self.strengths.append(
                "Dataset contains a diverse and informative feature set."
            )
        elif feature_count <= 2:
            self.score -= 10
            self.risks.append(
                "Very limited number of usable features detected."
            )

    # --------------------------------------------------
    # FINALIZATION
    # --------------------------------------------------
    def _finalize_score(self):
        if not self.risks and not self.strengths:
            self.strengths.append(
                "No major data quality issues detected."
            )

    def _verdict(self) -> str:
        if self.score >= 80:
            return "Strong ML potential"
        if self.score >= 60:
            return "Moderate ML potential"
        return "Low ML potential"

"""
report_adapter.py

Canonical Report Adapter for CortexAI.
This is for report data.
"""

from typing import Dict, Any


class ReportAdapter:
    def __init__(
        self,
        metadata: Dict,
        schema: Dict,
        cleaning_report: Dict,
        eda_report: Dict,
        training_results: Dict,
        training_summary: Dict,
        dataset_quality: Dict,
    ):
        self.metadata = metadata or {}
        self.schema = schema or {}
        self.cleaning = cleaning_report or {}
        self.eda = eda_report or {}
        self.training_results = training_results or {}
        self.training_summary = training_summary or {}
        self.quality = dataset_quality or {}

    # --------------------------------------------------
    # PUBLIC API
    # --------------------------------------------------
    def build(self) -> Dict[str, Any]:
        return {
     "dataset_overview": self._dataset_overview(),
     "schema_summary": self._schema_summary(),
     "cleaning_summary": self._cleaning_summary(),
     "eda_summary": self._eda_summary(),
     "model_results": self._model_results(),
     "best_model": self._best_model(),
     "dataset_quality": self._dataset_quality(),
     "strengths": self._strengths(),
     "risks": self._risks(),
     "recommendations": self._recommendations(),
       }


    # --------------------------------------------------
    # BUILDERS
    # --------------------------------------------------
    def _dataset_overview(self):
        shape = self.eda.get("basic_statistics", {}).get("shape", ("?", "?"))
        return {
            "rows": shape[0] if len(shape) == 2 else "?",
            "columns": shape[1] if len(shape) == 2 else "?",
            "target": self.schema.get("target"),
            "task_type": self.schema.get("task_type"),
        }

    def _schema_summary(self):
        return {
            "numeric": self.schema.get("numeric", []),
            "ordinal": self.schema.get("ordinal", []),
            "categorical": self.schema.get("categorical", []),
            "datetime": self.schema.get("datetime", []),
            "id_columns": self.schema.get("id_columns", []),
            "warnings": self.schema.get("warnings", []),
        }

    def _cleaning_summary(self):
        return {
            "dropped_id": self.cleaning.get("id_columns", []),
            "dropped_high_cardinality": self.cleaning.get("high_cardinality_columns", []),
            "missing_values": self.cleaning.get("missing_values", {}),
            "type_casting": self.cleaning.get("type_casting", {}),
            "final_shape": self.cleaning.get("final_shape"),
        }

    def _eda_summary(self):
        return {
            "key_insights": self.eda.get("key_insights", []),
            "target_distribution": (
                self.eda
                .get("target_analysis", {})
                .get("class_distribution", {})
            ),
        }

    def _model_results(self):
        return self.training_results

    def _best_model(self):
        return {
            "name": self.training_summary.get("best_model"),
            "score": self.training_summary.get("best_score"),
            "metric": self.training_summary.get("metric"),
        }

    def _dataset_quality(self):
        return {
            "score": self.quality.get("learnability_score"),
            "verdict": self.quality.get("verdict"),
            "reasons": self.quality.get("reasons", []),
        }

    def _risks(self):
        risks = []
        risks.extend(self.schema.get("warnings", []))
        risks.extend(self.quality.get("reasons", []))
        return list(dict.fromkeys(risks))

    def _recommendations(self):
        return self.quality.get("recommendations", [])
    
    def _dataset_quality(self):
     return {
        "score": self.quality.get("learnability_score"),
        "verdict": self.quality.get("verdict"),
    }

    def _strengths(self):
     return self.quality.get("strengths", [])

    def _risks(self):
      risks = []
      risks.extend(self.schema.get("warnings", []))
      risks.extend(self.quality.get("risks", []))
      return risks

    def _recommendations(self):
     return self.quality.get("recommendations", [])


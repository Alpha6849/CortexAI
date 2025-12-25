"""
report_generator.py

Generates a single executive-ready PDF report
combining all CortexAI pipeline outputs.

✔ Dataset-agnostic
✔ No hallucinations
✔ Client / boss friendly
"""

from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from typing import Dict
import datetime


class CortexReportGenerator:
    def __init__(self, output_path: str = "cortexai_report.pdf"):
        self.output_path = output_path
        self.styles = getSampleStyleSheet()
        self.elements = []

    # --------------------------------------------------
    # HELPERS
    # --------------------------------------------------
    def _title(self, text):
        self.elements.append(Paragraph(f"<b>{text}</b>", self.styles["Title"]))
        self.elements.append(Spacer(1, 12))

    def _h(self, text):
        self.elements.append(Paragraph(f"<b>{text}</b>", self.styles["Heading2"]))
        self.elements.append(Spacer(1, 8))

    def _p(self, text):
        self.elements.append(Paragraph(text, self.styles["BodyText"]))
        self.elements.append(Spacer(1, 6))

    def _table(self, data):
        table = Table(data, colWidths=[200, 300])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("VALIGN", (0, 0), (-1, -1), "TOP")
        ]))
        self.elements.append(table)
        self.elements.append(Spacer(1, 10))

    # --------------------------------------------------
    # SECTIONS
    # --------------------------------------------------
    def add_executive_summary(self, dataset_quality: Dict):
        self._h("1. Executive Summary")
        self._p(f"""
This report was generated automatically by CortexAI.

Learnability Score: <b>{dataset_quality['learnability_score']} / 100</b><br/>
Verdict: <b>{dataset_quality['verdict']}</b>
        """)

    def add_dataset_overview(self, meta: Dict, schema: Dict):
        self._h("2. Dataset Overview")
        self._table([
            ["Metric", "Value"],
            ["Rows", meta.get("rows")],
            ["Columns", meta.get("columns")],
            ["Target Column", schema.get("target")],
            ["Task Type", schema.get("task_type")]
        ])

    def add_schema_summary(self, schema: Dict):
        self._h("3. Feature Schema")
        self._table([
            ["Numeric Features", ", ".join(schema.get("numeric", []))],
            ["Ordinal Features", ", ".join(schema.get("ordinal", []))],
            ["Categorical Features", ", ".join(schema.get("categorical", []))],
            ["ID Columns", ", ".join(schema.get("id_columns", []))],
        ])

    def add_cleaning_summary(self, cleaning_report: Dict):
        self._h("4. Data Cleaning Summary")
        self._p("Key automated cleaning steps applied:")
        self._table([
            ["Removed ID Columns", ", ".join(cleaning_report.get("id_columns", []))],
            ["Missing Values Fixed", str(cleaning_report.get("missing_values", {}))],
            ["Type Casting", str(cleaning_report.get("type_casting", {}))],
            ["Final Dataset Shape", str(cleaning_report.get("final_shape", ""))]
        ])

    def add_eda_summary(self, eda: Dict):
        self._h("5. Exploratory Data Analysis (EDA)")
        for insight in eda.get("key_insights", []):
            self._p(f"• {insight}")

    def add_model_results(self, training_summary: Dict):
        self._h("6. Model Training Results")
        self._table([
            ["Best Model", training_summary.get("best_model")],
            ["Metric Used", training_summary.get("metric")],
            ["Best CV Score", training_summary.get("best_score")]
        ])

    def add_recommendations(self, dataset_quality: Dict):
        self._h("7. Recommendations & Risks")
        for r in dataset_quality.get("recommendations", []):
            self._p(f"• {r}")
        for w in dataset_quality.get("reasons", []):
            self._p(f"⚠ {w}")

    # --------------------------------------------------
    # BUILD
    # --------------------------------------------------
    def build(self):
        doc = SimpleDocTemplate(self.output_path, pagesize=A4)
        doc.build(self.elements)
        return self.output_path

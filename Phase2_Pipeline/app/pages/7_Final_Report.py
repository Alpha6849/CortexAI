import sys
import os
import streamlit as st

# --------------------------------------------------
# PATH SETUP
# --------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.dirname(CURRENT_DIR)
PARENT_DIR = os.path.dirname(APP_DIR)
sys.path.append(PARENT_DIR)

from pipeline.report_generator import CortexReportGenerator

st.set_page_config(page_title="Final Report — CortexAI", layout="centered")
st.title("📄 Final Report (Executive PDF)")

st.write("""
Download a **single, professional PDF report**
combining **all CortexAI analysis**.

Perfect for:
• Clients  
• Managers  
• Stakeholders  
""")

# --------------------------------------------------
# REQUIRED DATA
# --------------------------------------------------
required_keys = [
    "meta",
    "schema",
    "cleaning_report",
    "eda_report",
    "training_results",
    "dataset_quality"
]

missing = [k for k in required_keys if k not in st.session_state]
if missing:
    st.error(f"Missing pipeline steps: {', '.join(missing)}")
    st.stop()

# --------------------------------------------------
# GENERATE REPORT
# --------------------------------------------------
if st.button("📥 Generate Final PDF Report"):

    generator = CortexReportGenerator("cortexai_final_report.pdf")

    generator._title("CortexAI — Automated Data Science Report")
    generator._p(f"Generated on: {__import__('datetime').datetime.now()}")

    generator.add_executive_summary(st.session_state["dataset_quality"])
    generator.add_dataset_overview(st.session_state["meta"], st.session_state["schema"])
    generator.add_schema_summary(st.session_state["schema"])
    generator.add_cleaning_summary(st.session_state["cleaning_report"])
    generator.add_eda_summary(st.session_state["eda_report"])
    generator.add_model_results(
        st.session_state.get("training_summary", {})
    )
    generator.add_recommendations(st.session_state["dataset_quality"])

    pdf_path = generator.build()

    with open(pdf_path, "rb") as f:
        st.download_button(
            "⬇️ Download Final PDF Report",
            data=f,
            file_name="CortexAI_Report.pdf",
            mime="application/pdf"
        )

    st.success("Report generated successfully!")

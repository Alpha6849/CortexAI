import streamlit as st
from pipeline.report_adapter import ReportAdapter
from pipeline.report_generator import CortexAIReportGenerator

st.set_page_config(page_title="Final Report — CortexAI", layout="centered")
st.title("📄 Final Automated Report")

required_keys = [
    "dataset_metadata",
    "schema",
    "cleaning_report",
    "eda_report",
    "training_results",
    "training_summary",
    "dataset_quality"
]

missing = [k for k in required_keys if k not in st.session_state]
if missing:
    st.error(f"Missing pipeline outputs: {missing}")
    st.stop()

if st.button("📥 Generate Full PDF Report"):
    with st.spinner("Generating enterprise-grade report..."):

        adapter = ReportAdapter(
            metadata=st.session_state["dataset_metadata"],
            schema=st.session_state["schema"],
            cleaning_report=st.session_state["cleaning_report"],
            eda_report=st.session_state["eda_report"],
            training_results=st.session_state["training_results"],
            training_summary=st.session_state["training_summary"],
            dataset_quality=st.session_state["dataset_quality"],
        )

        payload = adapter.build()

        gen = CortexAIReportGenerator()
        path = gen.render(payload)

    with open(path, "rb") as f:
        st.download_button(
            "⬇️ Download Final PDF Report",
            data=f,
            file_name="CortexAI_Final_Report.pdf",
            mime="application/pdf"
        )

    st.success("Report generated successfully!")

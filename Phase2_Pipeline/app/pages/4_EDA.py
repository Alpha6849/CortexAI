import sys
import os
import streamlit as st
import json

# Fix import path for pipeline/
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.dirname(CURRENT_DIR)
PARENT_DIR = os.path.dirname(APP_DIR)
sys.path.append(PARENT_DIR)

from pipeline.eda import EDAEngine


st.title("📊 Exploratory Data Analysis (EDA)")

# Ensure dependencies exist
if "cleaned_df" not in st.session_state:
    st.error("No cleaned dataset found. Please run Cleaning first.")
    st.stop()

if "schema" not in st.session_state:
    st.error("Schema not found. Please run Schema Detection first.")
    st.stop()

df = st.session_state["cleaned_df"]
schema = st.session_state["schema"]

st.write("""
This step generates automated EDA:
- Basic statistics  
- Distribution analysis  
- Target analysis  
- Correlation matrix  
- Plot suggestions  
""")

# Run EDA (button-controlled)
if st.button("Run EDA Analysis"):
    engine = EDAEngine(df, schema)
    eda_report = engine.generate_report()

    # Save to session state
    st.session_state["eda_report"] = eda_report

    st.success("EDA Analysis completed!")

    # Display main sections
    st.subheader("📘 Full EDA Report")
    st.json(eda_report)

    # ---- Basic Statistics ----
    st.subheader("📊 Basic Statistics")
    st.json(eda_report.get("basic_statistics", {}))

    # ---- Target Column ----
    st.subheader("🎯 Target Analysis")
    st.json(eda_report.get("target_analysis", {}))

    # ---- Numeric Analysis ----
    st.subheader("🔢 Numeric Columns Analysis")
    st.json(eda_report.get("numeric_analysis", {}))

    # ---- Correlation Matrix ----
    st.subheader("🔗 Correlation Matrix")
    st.json(eda_report.get("correlation_matrix", {}))

    # ---- High Correlation Pairs ----
    st.subheader("⚠️ High Correlation Pairs")
    st.json(eda_report.get("high_correlation_pairs", {}))

    # ---- Download JSON ----
    st.download_button(
        "⬇️ Download EDA Report (JSON)",
        data=json.dumps(eda_report, indent=4),
        file_name="eda_report.json",
        mime="application/json"
    )

else:
    st.info("Click 'Run EDA Analysis' to begin.")

    if "eda_report" in st.session_state:
        st.warning("Previously generated EDA report found.")

st.write("EDA keys:", st.session_state["eda_report"].keys())

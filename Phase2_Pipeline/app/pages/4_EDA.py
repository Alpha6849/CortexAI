import sys
import os
import streamlit as st
import json

# --------------------------------------------------
# Fix import path
# --------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.dirname(CURRENT_DIR)
PARENT_DIR = os.path.dirname(APP_DIR)
sys.path.append(PARENT_DIR)

from pipeline.eda import EDAEngine


st.title("📊 Exploratory Data Analysis (EDA)")

# --------------------------------------------------
# Preconditions
# --------------------------------------------------
if "cleaned_df" not in st.session_state:
    st.error("No cleaned dataset found. Please run Cleaning first.")
    st.stop()

if "schema" not in st.session_state:
    st.error("Schema not found. Please run Schema Detection first.")
    st.stop()

df = st.session_state["cleaned_df"]
schema = st.session_state["schema"]

st.write("""
This step generates automated, schema-aware EDA:

• Dataset overview  
• Target analysis  
• Numeric & ordinal feature analysis  
• Categorical outcome analysis  
• Correlation insights  
""")

# --------------------------------------------------
# Run EDA
# --------------------------------------------------
if st.button("▶ Run EDA Analysis"):
    engine = EDAEngine(df, schema)
    eda_report = engine.generate_report()
    st.session_state["eda_report"] = eda_report
    st.success("EDA Analysis completed successfully!")

# --------------------------------------------------
# Display EDA (if available)
# --------------------------------------------------
if "eda_report" in st.session_state:
    eda = st.session_state["eda_report"]

    # ---- Key Insights ----
    st.subheader("🧠 Key Insights")
    insights = eda.get("key_insights", [])
    if insights:
        for i in insights:
            st.markdown(f"- {i}")
    else:
        st.info("No strong high-level insights detected.")

    # ---- Basic Statistics ----
    st.subheader("📊 Dataset Overview")
    st.json(eda.get("basic_statistics", {}))

    # ---- Target Analysis ----
    st.subheader("🎯 Target Analysis")
    st.json(eda.get("target_analysis", {}))

    # ---- Numeric Features ----
    st.subheader("🔢 Numeric Feature Analysis")
    st.json(eda.get("numeric_analysis", {}))

    # ---- Ordinal Features ----
    if eda.get("ordinal_analysis"):
        st.subheader("📐 Ordinal Feature Analysis")
        st.json(eda.get("ordinal_analysis", {}))

    # ---- Binary Outcome Analysis ----
    if eda.get("binary_outcome_analysis"):
        st.subheader("📈 Categorical vs Target Analysis")
        st.json(eda.get("binary_outcome_analysis", {}))

    # ---- Correlations ----
    st.subheader("🔗 Feature Correlations")
    st.json(eda.get("correlation_matrix", {}))

    if eda.get("high_correlation_pairs"):
        st.warning("⚠️ High Correlation Pairs Detected")
        st.json(eda.get("high_correlation_pairs", {}))
    else:
        st.info("No highly correlated feature pairs detected.")

    # ---- Download ----
    st.download_button(
        "⬇️ Download EDA Report (JSON)",
        data=json.dumps(eda, indent=4),
        file_name="eda_report.json",
        mime="application/json"
    )

else:
    st.info("Click **Run EDA Analysis** to generate insights.")

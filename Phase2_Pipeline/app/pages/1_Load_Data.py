import sys
import os
import streamlit as st

# --------------------------------------------------
# Fix import path
# --------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.dirname(CURRENT_DIR)
PARENT_DIR = os.path.dirname(APP_DIR)
sys.path.append(PARENT_DIR)

# --------------------------------------------------
# Imports
# --------------------------------------------------
from pipeline.loader import DataLoader
from pipeline.usage_manager import (
    init_plan_and_usage,
    enforce_limit,
    increment_usage,
    get_plan_limits
)

# --------------------------------------------------
# Init plan & usage
# --------------------------------------------------
init_plan_and_usage()

st.title("📂 Load Dataset")

st.write("""
Upload a CSV file to begin the AutoML pipeline.
This uses the **production DataLoader class**.
""")

# --------------------------------------------------
# Enforce upload limit BEFORE upload
# --------------------------------------------------
enforce_limit(
    key="uploads",
    message="🚫 Upload limit reached for Free plan. Upgrade to Pro to upload more datasets."
)

uploaded = st.file_uploader("Upload CSV", type=["csv"])

if uploaded:
    # --------------------------------------------------
    # Save temp file
    # --------------------------------------------------
    temp_path = os.path.join(APP_DIR, "_temp_upload.csv")
    with open(temp_path, "wb") as f:
        f.write(uploaded.getbuffer())

    try:
        loader = DataLoader(temp_path)
        df, meta = loader.load()
    except Exception as e:
        st.error(f"DataLoader failed: {e}")
        st.stop()

    # --------------------------------------------------
    # Enforce row limits AFTER load
    # --------------------------------------------------
    max_rows = get_plan_limits().get("max_rows", 0)

    if len(df) > max_rows:
        st.error(
            f"🚫 Dataset too large for your plan.\n\n"
            f"Rows: {len(df)} | Allowed: {max_rows}\n\n"
            "Upgrade to Pro to upload larger datasets."
        )
        st.stop()

    # --------------------------------------------------
    # Increment usage
    # --------------------------------------------------
    increment_usage("uploads")

    st.success("File loaded successfully using CortexAI DataLoader!")

    # --------------------------------------------------
    # Save immutable copies to session state
    # --------------------------------------------------
    st.session_state["raw_df"] = df.copy()
    st.session_state["df"] = df.copy()
    st.session_state["meta"] = meta

    # --------------------------------------------------
    # Preview
    # --------------------------------------------------
    st.subheader("🔍 Dataset Preview")
    st.dataframe(df.head(20))

    st.subheader("📘 Metadata")
    st.json(meta)

    st.info("Proceed to **Schema** page →")

else:
    st.info("Please upload a CSV file to continue.")

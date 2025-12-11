import sys
import os
import streamlit as st
import pandas as pd

# Fix import path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.dirname(CURRENT_DIR)
PARENT_DIR = os.path.dirname(APP_DIR)
sys.path.append(PARENT_DIR)

from pipeline.loader import DataLoader


st.title("📂 Load Dataset")

st.write("""
Upload a CSV file to begin the AutoML pipeline.
This uses the **production DataLoader class**.
""")


uploaded = st.file_uploader("Upload CSV", type=["csv"])

if uploaded:
    # temporary file for DataLoader
    temp_path = os.path.join(APP_DIR, "_temp_upload.csv")
    with open(temp_path, "wb") as f:
        f.write(uploaded.getbuffer())

    try:
        loader = DataLoader(temp_path)
        df, meta = loader.load()
        st.success("File loaded successfully using CortexAI DataLoader!")

    except Exception as e:
        st.error(f"DataLoader failed: {e}")
        st.stop()

    # Save to session state
    st.session_state["df"] = df
    st.session_state["meta"] = meta

    # Preview
    st.subheader("🔍 Dataset Preview")
    st.dataframe(df.head(20))

    st.subheader("📘 Metadata")
    st.json(meta)

    st.info("Proceed to **Schema** page →")

else:
    st.info("Please upload a CSV file to continue.")

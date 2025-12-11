import sys
import os
import streamlit as st

# Fix import path for pipeline/
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.dirname(CURRENT_DIR)
PARENT_DIR = os.path.dirname(APP_DIR)
sys.path.append(PARENT_DIR)

from pipeline.schema import SchemaDetector


st.title("🧬 Schema Detection")

# Ensure dataset is loaded
if "df" not in st.session_state:
    st.error("No dataset found. Please load a CSV first from 'Load Data'.")
    st.stop()

df = st.session_state["df"]

st.write("""
This page analyzes your dataset and identifies:
- Numeric columns  
- Categorical columns  
- Datetime columns  
- ID columns  
- Target column  
""")

# Run schema detection
detector = SchemaDetector(df)
schema = detector.detect()   # FIXED METHOD CALL

# Save schema for next pages (Cleaning, EDA, Training)
st.session_state["schema"] = schema

# Display schema
st.subheader("📘 Schema Overview")
st.json(schema)

# Column types table
st.subheader("📊 Column Types")

col_data = []
for col in df.columns:
    if col in schema["numeric"]:
        typ = "Numeric"
    elif col in schema["categorical"]:
        typ = "Categorical"
    elif col in schema["datetime"]:
        typ = "Datetime"
    else:
        typ = "Unknown"
    col_data.append({"Column": col, "Type": typ})

st.table(col_data)

# ID Columns
st.subheader("🆔 ID Columns")
st.write(schema["id_columns"] if schema["id_columns"] else "None detected")

# Target Column
st.subheader("🎯 Target Column")
st.write(schema["target"] if schema["target"] else "None detected")

st.info("Proceed to **Cleaning** page →")

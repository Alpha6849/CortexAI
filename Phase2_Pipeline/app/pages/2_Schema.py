import sys
import os
import streamlit as st

# --------------------------------------------------
# Fix import path for pipeline/
# --------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.dirname(CURRENT_DIR)
PARENT_DIR = os.path.dirname(APP_DIR)
sys.path.append(PARENT_DIR)

from pipeline.schema import SchemaDetector


st.title("🧬 Schema Configuration")

# --------------------------------------------------
# Ensure dataset is loaded
# --------------------------------------------------
if "df" not in st.session_state:
    st.error("No dataset found. Please load a CSV first from 'Load Data'.")
    st.stop()

df = st.session_state["df"]

st.write("""
This step configures the dataset schema.

Please **select the target column**.
Cortex will validate your selection and infer the ML task.
""")

# --------------------------------------------------
# Target Selection
# --------------------------------------------------
st.subheader("🎯 Select Target Column")

target_col = st.selectbox(
    "Choose the column you want to predict:",
    options=df.columns.tolist()
)

# --------------------------------------------------
# Helper: Column Type Resolver
# --------------------------------------------------
def resolve_column_type(col, schema):
    if col == schema["target"]:
        return "🎯 Target"

    if col in schema.get("id_columns", []):
        return "🆔 ID (Dropped)"

    if col in schema.get("ordinal", []):
        return "Ordinal"

    if col in schema.get("numeric", []):
        return "Numeric"

    if col in schema.get("categorical", []):
        return "Categorical"

    if col in schema.get("high_missing_categorical", []):
        return "High Missing (Dropped)"

    if col in schema.get("high_cardinality_columns", []):
        return "High Cardinality (Dropped)"

    if col in schema.get("datetime", []):
        return "Datetime"

    return "Unknown"


# --------------------------------------------------
# Run Schema Detection
# --------------------------------------------------
if st.button("🔍 Analyze Schema"):

    detector = SchemaDetector(df)
    schema = detector.detect(target_col)

    # Save schema for downstream pages
    st.session_state["schema"] = schema

    # --------------------------------------------------
    # Schema Overview
    # --------------------------------------------------
    st.subheader("📘 Schema Overview")
    st.json(schema)

    # --------------------------------------------------
    # Column Types Table
    # --------------------------------------------------
    st.subheader("📊 Column Types")

    col_data = []
    for col in df.columns:
        col_data.append({
            "Column": col,
            "Type": resolve_column_type(col, schema)
        })

    st.table(col_data)

    # --------------------------------------------------
    # ID Columns
    # --------------------------------------------------
    st.subheader("🆔 ID Columns")
    st.write(schema["id_columns"] if schema["id_columns"] else "None detected")

    # --------------------------------------------------
    # Task Type
    # --------------------------------------------------
    st.subheader("🧠 Inferred Task Type")
    st.success(schema["task_type"])

    # --------------------------------------------------
    # Warnings
    # --------------------------------------------------
    if schema.get("warnings"):
        st.subheader("⚠️ Warnings")
        for w in schema["warnings"]:
            st.warning(w)
    else:
        st.success("No schema warnings detected.")

    st.info("✅ Schema locked. Proceed to **Cleaning** →")

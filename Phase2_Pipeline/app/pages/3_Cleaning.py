import sys
import os
import streamlit as st

# Fix import path for pipeline/
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.dirname(CURRENT_DIR)
PARENT_DIR = os.path.dirname(APP_DIR)
sys.path.append(PARENT_DIR)

from pipeline.cleaner import DataCleaner


st.title("🧹 Data Cleaning")

# Check required session state
if "df" not in st.session_state:
    st.error("No dataset found. Please load a CSV first.")
    st.stop()

if "schema" not in st.session_state:
    st.error("Schema not found. Please run Schema Detection first.")
    st.stop()

df = st.session_state["df"]
schema = st.session_state["schema"]

st.write("""
This step cleans your dataset automatically using CortexAI's production cleaner:
- Removes ID columns  
- Fixes missing values  
- Fixes column types  
- Handles outliers  
- Detects high-cardinality columns  
""")

# Run cleaning (button so it doesn't auto-run every rerender)
if st.button("Run Cleaning Pipeline"):
    cleaner = DataCleaner(df, schema)
    cleaned_df, report = cleaner.clean()

    # Save into session_state
    st.session_state["cleaned_df"] = cleaned_df
    st.session_state["cleaning_report"] = report

    st.success("Cleaning completed successfully!")

    # Show report
    st.subheader("📝 Cleaning Report")
    st.json(report)

    # Show preview
    st.subheader("🔍 Cleaned Dataset Preview")
    st.dataframe(cleaned_df.head(20))

    # Download cleaned CSV
    st.download_button(
        "⬇️ Download Cleaned Dataset",
        cleaned_df.to_csv(index=False).encode("utf-8"),
        "cleaned_dataset.csv",
        "text/csv"
    )

else:
    st.info("Click 'Run Cleaning Pipeline' to begin.")

    if "cleaned_df" in st.session_state:
        st.warning("You have a previously cleaned dataset stored.")

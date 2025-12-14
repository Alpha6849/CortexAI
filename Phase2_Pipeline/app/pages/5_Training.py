import sys
import os
import json
import streamlit as st

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.dirname(CURRENT_DIR)
PARENT_DIR = os.path.dirname(APP_DIR)
sys.path.append(PARENT_DIR)

from pipeline.trainer import ModelTrainer

st.title("🤖 Model Training — AutoML")

# -----------------------------
# Session Guards
# -----------------------------
if "cleaned_df" not in st.session_state:
    st.error("No cleaned dataset found. Run Cleaning first.")
    st.stop()

if "schema" not in st.session_state:
    st.error("Schema not found. Run Schema Detection first.")
    st.stop()

df = st.session_state["cleaned_df"]
schema = st.session_state["schema"]

st.markdown("""
**This step performs:**
- Feature preparation  
- 5-fold cross-validation  
- Best model selection  
- Retraining on full data  
- Model & summary export  
""")

# -----------------------------
# Training Trigger
# -----------------------------
if st.button("🚀 Start AutoML Training"):

    trainer = ModelTrainer(df, schema)

    # -------- Prepare Data --------
    with st.spinner("Preparing data..."):
        prep_info = trainer.prepare_data()

    st.subheader("📦 Data Preparation")
    st.json(prep_info)

    # -------- Train Models --------
    st.subheader("📊 Cross-Validation Results")

    with st.spinner("Training models..."):
        scores = trainer.train_all_models()

    st.json(scores)

    if trainer.best_model_name is None:
        st.error("Training failed. No valid model.")
        st.stop()

    # -------- Best Model --------
    st.subheader("🏆 Best Model")
    st.write(f"**Model:** `{trainer.best_model_name}`")
    st.write(f"**Mean CV Score:** `{trainer.best_score}`")

    # -------- Retrain --------
    with st.spinner("Retraining best model..."):
        trainer.retrain_best_model()

    st.success("Best model retrained.")

    # -------- Save --------
    model_path = trainer.save_best_model("best_model.pkl")
    summary = trainer.save_training_summary("training_summary.json")

    # -------- Download --------
    st.subheader("⬇️ Downloads")

    with open(model_path, "rb") as f:
        st.download_button(
            "Download Model",
            f,
            "best_model.pkl",
            "application/octet-stream"
        )

    st.download_button(
        "Download Training Summary",
        data=json.dumps(summary, indent=4),
        file_name="training_summary.json",
        mime="application/json"
    )

else:
    st.info("Click **Start AutoML Training** to begin.")

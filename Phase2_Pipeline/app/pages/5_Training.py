import sys
import os
import json
import streamlit as st

# --------------------------------------------------
# Path setup
# --------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.dirname(CURRENT_DIR)
PARENT_DIR = os.path.dirname(APP_DIR)
sys.path.append(PARENT_DIR)

# --------------------------------------------------
# Imports
# --------------------------------------------------
from pipeline.trainer import ModelTrainer
from pipeline.quality_analyzer import DatasetQualityAnalyzer

# --------------------------------------------------
# Page Title
# --------------------------------------------------
st.title("🤖 Model Training — AutoML")

# --------------------------------------------------
# Session Guards
# --------------------------------------------------
if "cleaned_df" not in st.session_state:
    st.error("No cleaned dataset found. Run Cleaning first.")
    st.stop()

if "schema" not in st.session_state:
    st.error("Schema not found. Run Schema Detection first.")
    st.stop()

df = st.session_state["cleaned_df"]
schema = st.session_state["schema"]

# --------------------------------------------------
# Page Description
# --------------------------------------------------
st.markdown("""
**This step performs:**
- Feature preparation  
- 5-fold cross-validation  
- Best model selection  
- Retraining on full data  
- Dataset quality assessment  
- Model & summary export  
""")

# --------------------------------------------------
# Training Trigger
# --------------------------------------------------
if st.button("🚀 Start AutoML Training"):

    # -----------------------------
    # Initialize Trainer
    # -----------------------------
    trainer = ModelTrainer(df, schema)

    # -----------------------------
    # Prepare Data
    # -----------------------------
    with st.spinner("Preparing data..."):
        prep_info = trainer.prepare_data()

    st.subheader("📦 Data Preparation")
    st.json(prep_info)

    # -----------------------------
    # Train Models
    # -----------------------------
    st.subheader("📊 Cross-Validation Results")

    with st.spinner("Training models..."):
        training_results = trainer.train_all_models()

    st.session_state["training_results"] = training_results
    st.json(training_results)

    if trainer.best_model_name is None:
        st.error("Training failed. No valid model.")
        st.stop()

    # -----------------------------
    # Best Model
    # -----------------------------
    st.subheader("🏆 Best Model")
    st.write(f"**Model:** `{trainer.best_model_name}`")
    st.write(f"**Mean CV Score:** `{trainer.best_score}`")

    # -----------------------------
    # Retrain Best Model
    # -----------------------------
    with st.spinner("Retraining best model on full dataset..."):
        trainer.retrain_best_model()

    st.success("Best model retrained successfully.")

    # -----------------------------
    # Dataset Quality Analysis
    # -----------------------------
    st.subheader("🧠 Dataset Quality Assessment")

    quality_analyzer = DatasetQualityAnalyzer(
        schema=schema,
        eda_report=st.session_state.get("eda_report", {}),
        training_results=st.session_state["training_results"]
    )

    dataset_quality = quality_analyzer.analyze()
    st.session_state["dataset_quality"] = dataset_quality

    st.metric(
        label="Learnability Score",
        value=f"{dataset_quality['learnability_score']} / 100"
    )

    st.info(f"**Verdict:** {dataset_quality['verdict']}")

    if dataset_quality["reasons"]:
        st.markdown("**Reasons:**")
        for reason in dataset_quality["reasons"]:
            st.write(f"• {reason}")

    if dataset_quality["recommendations"]:
        st.markdown("**Recommendations:**")
        for rec in dataset_quality["recommendations"]:
            st.write(f"• {rec}")

    # -----------------------------
    # Save Outputs
    # -----------------------------
    model_path = trainer.save_best_model("best_model.pkl")
    summary = trainer.save_training_summary("training_summary.json")

    # -----------------------------
    # Downloads
    # -----------------------------
    st.subheader("⬇️ Downloads")

    with open(model_path, "rb") as f:
        st.download_button(
            label="Download Trained Model",
            data=f,
            file_name="best_model.pkl",
            mime="application/octet-stream"
        )

    st.download_button(
        label="Download Training Summary",
        data=json.dumps(summary, indent=4),
        file_name="training_summary.json",
        mime="application/json"
    )

else:
    st.info("Click **Start AutoML Training** to begin.")

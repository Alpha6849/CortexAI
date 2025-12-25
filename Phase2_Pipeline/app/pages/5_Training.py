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
from pipeline.usage_manager import (
    init_plan_and_usage,
    enforce_limit,
    increment_usage
)

# --------------------------------------------------
# Init plan & usage
# --------------------------------------------------
init_plan_and_usage()

# --------------------------------------------------
# Page Title
# --------------------------------------------------
st.title("🤖 Model Training — AutoML")

# --------------------------------------------------
# Session Guards
# --------------------------------------------------
if "cleaned_df" not in st.session_state:
    st.error("No cleaned dataset found. Please run **Cleaning** first.")
    st.stop()

if "schema" not in st.session_state:
    st.error("Schema not found. Please run **Schema Detection** first.")
    st.stop()

df = st.session_state["cleaned_df"]
schema = st.session_state["schema"]

# --------------------------------------------------
# Page Description
# --------------------------------------------------
st.markdown("""
This stage automatically trains multiple machine learning models
and selects the best one using cross-validation.

**What happens here:**
- Schema-driven feature preparation  
- 5-fold cross-validation  
- Best model selection  
- Retraining on full dataset  
- Dataset quality & learnability assessment  
- Model and summary export  
""")

# --------------------------------------------------
# Existing results notice
# --------------------------------------------------
if "training_results" in st.session_state:
    st.info("Previous training results detected. Re-run to update them.")

# --------------------------------------------------
# Enforce usage limits
# --------------------------------------------------
enforce_limit(
    key="pipeline_runs",
    message="🚫 Training limit reached for Free plan. Upgrade to Pro to run more AutoML trainings."
)

# --------------------------------------------------
# Training Trigger
# --------------------------------------------------
if st.button("🚀 Start AutoML Training"):

    increment_usage("pipeline_runs")

    # -----------------------------
    # Initialize Trainer
    # -----------------------------
    trainer = ModelTrainer(df, schema)

    # -----------------------------
    # Data Preparation
    # -----------------------------
    with st.spinner("Preparing features using schema..."):
        prep_info = trainer.prepare_data()

    st.subheader("📦 Data Preparation Summary")
    st.json(prep_info)

    # -----------------------------
    # Train Models
    # -----------------------------
    st.subheader("📊 Model Performance (Cross-Validation)")

    with st.spinner("Training models with cross-validation..."):
        training_results = trainer.train_all_models()

    st.session_state["training_results"] = training_results

    # --- Performance Summary ---
    st.markdown("### 📈 Mean CV Scores")
    for model, info in training_results.items():
        st.write(
            f"**{model}** → `{info['cv_mean_score']:.3f}`"
        )

    with st.expander("🔍 Full cross-validation details"):
        st.json(training_results)

    if trainer.best_model_name is None:
        st.error("Training failed. No valid model was selected.")
        st.stop()

    # -----------------------------
    # Best Model
    # -----------------------------
    st.subheader("🏆 Best Model Selected")

    st.success(
        f"**{trainer.best_model_name}** "
        f"(Mean CV Score: {trainer.best_score:.3f})"
    )

    st.caption(
        f"Evaluation metric used: **{trainer.metric_used}** "
        "(chosen automatically based on task type)"
    )

    # -----------------------------
    # Retrain Best Model
    # -----------------------------
    with st.spinner("Retraining best model on full dataset..."):
        trainer.retrain_best_model()

    st.success("Best model retrained on full dataset.")

    # -----------------------------
    # Dataset Quality Assessment
    # -----------------------------
    st.subheader("🧠 Dataset Learnability Assessment")

    quality_analyzer = DatasetQualityAnalyzer(
        schema=schema,
        eda_report=st.session_state.get("eda_report", {}),
        training_results=training_results
    )

    dataset_quality = quality_analyzer.analyze()
    st.session_state["dataset_quality"] = dataset_quality

    score = dataset_quality["learnability_score"]

    if score >= 80:
        st.success(f"Learnability Score: **{score} / 100**")
    elif score >= 60:
        st.warning(f"Learnability Score: **{score} / 100**")
    else:
        st.error(f"Learnability Score: **{score} / 100**")

    st.info(f"**Verdict:** {dataset_quality['verdict']}")

    # ---- Reasons ----
    if dataset_quality["reasons"]:
        st.markdown("### ✅ Why this dataset works")
        for reason in dataset_quality["reasons"]:
            st.write(f"✔ {reason}")

    # ---- Recommendations ----
    if dataset_quality["recommendations"]:
        st.markdown("### 🔧 How to improve further")
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

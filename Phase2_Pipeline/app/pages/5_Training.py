import sys
import os
import streamlit as st
import json

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.dirname(CURRENT_DIR)
PARENT_DIR = os.path.dirname(APP_DIR)
sys.path.append(PARENT_DIR)

from pipeline.trainer import ModelTrainer


st.title("🤖 Model Training — Cross-Validation + Best Model Retraining")

#session state :-
if "cleaned_df" not in st.session_state:
    st.error("No cleaned dataset found. Please run the Cleaning step first.")
    st.stop()

if "schema" not in st.session_state:
    st.error("Schema not found. Please run the Schema step first.")
    st.stop()

df = st.session_state["cleaned_df"]
schema = st.session_state["schema"]


st.write("""
This step performs:
- 5-Fold Cross-Validation for all ML models  
- Automatic best model selection  
- Retraining best model on full data  
- Saving model + summary  
""")


if st.button("🚀 Start AutoML Training"):

    trainer = ModelTrainer(df, schema)


    prep_info = trainer.prepare_data()
    st.subheader("📦 Data Preparation")
    st.json(prep_info)


    st.subheader("📊 Model Cross-Validation Scores")

    scores = trainer.train_all_models()  # MUST run before reading best_model_name
    st.json(scores)

    if trainer.best_model_name is None:
        st.error("Training failed. No valid model found.")
        st.stop()


    #  SHOW BEST MODEL

    st.subheader("🏆 Best Model (By Mean CV Score)")
    st.write(f"**Model:** `{trainer.best_model_name}`")
    st.write(f"**Mean CV Score:** `{trainer.best_score}`")


    #  RETRAIN BEST MODEL
    trainer.retrain_best_model()
    st.success("Best model successfully retrained on full dataset!")


    #  SAVE MODEL + SUMMARY
    model_path = trainer.save_best_model("best_model.pkl")
    summary = trainer.save_training_summary("training_summary.json")

    # Save into session state
    st.session_state["best_model_file"] = model_path
    st.session_state["training_summary"] = summary


    st.subheader("⬇️ Download Outputs")

    with open("best_model.pkl", "rb") as f:
        st.download_button(
            "Download Best Model (Pickle)",
            f,
            "best_model.pkl",
            "application/octet-stream"
        )

    st.download_button(
        "Download Training Summary (JSON)",
        data=json.dumps(summary, indent=4),
        file_name="training_summary.json",
        mime="application/json"
    )


else:
    st.info("Click **Start AutoML Training** to begin.")

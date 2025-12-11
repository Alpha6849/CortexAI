"""
trainer.py

Handles automatic model training, evaluation,
and best-model selection for CortexAI Phase 2.
Now upgraded with:
- K-Fold Cross-Validation
- Best model selection based on CV mean score
- Retraining best model on full dataset
"""

import logging
import pandas as pd
from typing import Dict, Any
from sklearn.model_selection import train_test_split, cross_val_score, KFold

logger = logging.getLogger("ModelTrainer")
logger.setLevel(logging.INFO)


class ModelTrainer:
    """
    Trains multiple ML models and selects the best one using Cross-Validation.

    Inputs:
    - cleaned DataFrame
    - schema (from SchemaDetector)
    """

    def __init__(self, df: pd.DataFrame, schema: Dict):
        self.df = df
        self.schema = schema

        # target column
        self.target = schema.get("target")
        if not self.target:
            raise ValueError("Target column not found in schema.")

        # X and y
        self.X = df.drop(columns=[self.target])
        self.y = df[self.target]

        # Detect task type
        if df[self.target].dtype == "object" or df[self.target].nunique() <= 20:
            self.task_type = "classification"
        else:
            self.task_type = "regression"

        logger.info(f"Task detected: {self.task_type}")

        self.results = {}           # CV scores per model
        self.best_model_name = None
        self.best_score = -999
        self.best_model = None
        self.label_encoder = None

    def prepare_data(self):

        logger.info("Preparing data for training...")

        # Label Encoding for classification targets
        if self.task_type == "classification":
            from sklearn.preprocessing import LabelEncoder
            self.label_encoder = LabelEncoder()
            self.y = self.label_encoder.fit_transform(self.y)
            logger.info("Target column encoded for classification.")

        return {
            "rows": self.X.shape[0],
            "columns": self.X.shape[1],
            "task_type": self.task_type
        }


    def train_all_models(self):
        """
        Train multiple models using cross-validation and select best.
        """

        logger.info("Starting CV training for all models...")

        from sklearn.linear_model import LogisticRegression, LinearRegression
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.svm import SVC
        from sklearn.neighbors import KNeighborsClassifier

        # Model list
        models = {}

        if self.task_type == "classification":
            models = {
                "LogisticRegression": LogisticRegression(max_iter=500),
                "RandomForestClassifier": RandomForestClassifier(),
                "SVC": SVC(),
                "KNN": KNeighborsClassifier()
            }

            # Try XGBoost
            try:
                from xgboost import XGBClassifier
                models["XGBoost"] = XGBClassifier(
                    eval_metric="mlogloss",
                    use_label_encoder=False
                )
                logger.info("XGBoost added.")
            except Exception as e:
                logger.warning(f"XGBoost not available: {e}")

            scoring_metric = "accuracy"

        else:
            models = {
                "LinearRegression": LinearRegression(),
                "RandomForestRegressor": RandomForestRegressor()
            }

            # Try XGBoost
            try:
                from xgboost import XGBRegressor
                models["XGBoostRegressor"] = XGBRegressor()
                logger.info("XGBRegressor added.")
            except Exception as e:
                logger.warning(f"XGBoostRegressor not available: {e}")

            scoring_metric = "r2"

        # K-Fold setup
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        # Evaluate all models
        for name, model in models.items():

            logger.info(f"Cross-validating model: {name}")

            try:
                cv_scores = cross_val_score(
                    model, self.X, self.y,
                    cv=kf, scoring=scoring_metric
                )
            except Exception as e:
                logger.error(f"Model {name} failed: {e}")
                continue

            mean_score = cv_scores.mean()
            logger.info(f"{name} CV Mean Score = {mean_score}")

            # Save results
            self.results[name] = {
                "cv_scores": cv_scores.tolist(),
                "cv_mean_score": float(mean_score)
            }

            # Track best model
            if mean_score > self.best_score:
                self.best_score = mean_score
                self.best_model_name = name
                self.best_model = model

        logger.info(f"Best model after CV: {self.best_model_name} (score: {self.best_score})")
        return self.results



    def retrain_best_model(self):

        if self.best_model is None:
            raise ValueError("No best model selected. Run train_all_models() first.")

        logger.info(f"Retraining best model on full dataset: {self.best_model_name}")

        self.best_model.fit(self.X, self.y)
        return self.best_model


    def save_best_model(self, output_path="best_model.pkl"):
        import joblib
        joblib.dump(self.best_model, output_path)
        logger.info(f"Best model saved to {output_path}")
        return output_path


    def save_training_summary(self, output_path="training_summary.json"):
        import json

        summary = {
            "task_type": self.task_type,
            "best_model": self.best_model_name,
            "best_score": float(self.best_score),
            "all_model_scores": self.results
        }

        with open(output_path, "w") as f:
            json.dump(summary, f, indent=4)

        logger.info(f"Training summary saved to {output_path}")
        return summary





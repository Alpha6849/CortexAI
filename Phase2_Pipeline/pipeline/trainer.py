"""
trainer.py

Handles automatic model training, evaluation,
and best-model selection for CortexAI Phase 2.
"""

import logging
import pandas as pd
from typing import Dict, Any
from sklearn.model_selection import train_test_split

logger = logging.getLogger("ModelTrainer")
logger.setLevel(logging.INFO)


class ModelTrainer:
    """
    Trains multiple ML models and selects the best one automatically.

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

        # task type
        if df[self.target].dtype == "object" or df[self.target].nunique() <= 20:
            self.task_type = "classification"
        else:
            self.task_type = "regression"

        logger.info(f"Task detected: {self.task_type}")

        # storing results
        self.results = {}
        self.best_model = None
        self.best_score = -1
        
        
    def prepare_data(self):
        
        logger.info("Preparing data for training...")

        # classification :- encode target
        if self.task_type == "classification":
            from sklearn.preprocessing import LabelEncoder
            self.label_encoder = LabelEncoder()
            self.y = self.label_encoder.fit_transform(self.y)
            logger.info("Target column encoded for classification.")
        else:
            self.label_encoder = None

        # Train/Test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

        logger.info(
            f"Data prepared. Train size: {self.X_train.shape}, Test size: {self.X_test.shape}"
        )

        return {
            "train_shape": self.X_train.shape,
            "test_shape": self.X_test.shape,
            "task_type": self.task_type
        }
        
    def train_all_models(self):
        """Training multiple models and find the best one."""
        logger.info("Starting model training...")

        
        from sklearn.linear_model import LogisticRegression, LinearRegression
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.svm import SVC
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.metrics import accuracy_score, r2_score

        models = {}

        # Classification Models
        if self.task_type == "classification":
            models = {
                "LogisticRegression": LogisticRegression(max_iter=500),
                "RandomForestClassifier": RandomForestClassifier(),
                "SVC": SVC(),
                "KNN": KNeighborsClassifier()
            }

            # Try adding XGBoost
            try:
                from xgboost import XGBClassifier
                models["XGBoost"] = XGBClassifier(
                    eval_metric="mlogloss",
                    use_label_encoder=False
                )
                logger.info("XGBoostClassifier added to model list.")
            except Exception as e:
                logger.warning(f"XGBoost not available: {e}")


        # Regression Models
        else:
            models = {
                "LinearRegression": LinearRegression(),
                "RandomForestRegressor": RandomForestRegressor()
            }

        for name, model in models.items():
            logger.info(f"Training model: {name}")

            model.fit(self.X_train, self.y_train)

            preds = model.predict(self.X_test)

            #  metric
            if self.task_type == "classification":
                score = accuracy_score(self.y_test, preds)
            else:
                score = r2_score(self.y_test, preds)

            logger.info(f"{name} Score: {score}")
            self.results[name] = score

            # Track best model
            if score > self.best_score:
                self.best_score = score
                self.best_model = model
                logger.info(f"New best model: {name} (score={score})")

        return self.results
    
    def save_best_model(self, output_path: str = "best_model.pkl"):
        """Saving best model as pickle."""
        import joblib
        if self.best_model is None:
            raise ValueError("No trained model to save. Run train_all_models() first.")

        joblib.dump(self.best_model, output_path)
        logger.info(f"Best model saved to {output_path}")
        return output_path


    def save_training_summary(self, output_path: str = "training_summary.json"):
        """ results as JSON."""
        import json

        summary = {
            "task_type": self.task_type,
            "scores": self.results,
            "best_score": self.best_score,
            "best_model_type": type(self.best_model).__name__
        }

        with open(output_path, "w") as f:
            json.dump(summary, f, indent=4)

        logger.info(f"Training summary saved to {output_path}")
        return summary




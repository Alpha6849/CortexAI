"""
trainer.py

Handles automatic model training, evaluation,
and best-model selection for CortexAI Phase 2.

AutoML v1 (REAL, not toy):
- Numeric-only (safe)
- Proper preprocessing pipelines
- Baseline sanity check
- Correct metrics
- Stable model selection
"""

import logging
import pandas as pd
from typing import Dict
from sklearn.model_selection import cross_val_score, KFold

logger = logging.getLogger("ModelTrainer")
logger.setLevel(logging.INFO)


class ModelTrainer:
    def __init__(self, df: pd.DataFrame, schema: Dict):
        self.df = df
        self.schema = schema

        self.target = schema.get("target")
        if not self.target:
            raise ValueError("Target column not found in schema.")

        # Task detection
        if df[self.target].dtype == "object":
            self.task_type = "classification"
        elif df[self.target].nunique() <= 15:
            self.task_type = "classification"
        else:
            self.task_type = "regression"

        self.X = None
        self.y = None

        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.best_score = -1e9
        self.label_encoder = None

    # --------------------------------------------------
    # FEATURE PREPARATION
    # --------------------------------------------------
    def _prepare_ml_features(self):

        numeric_cols = self.schema.get("numeric", [])
        id_cols = self.schema.get("id_columns", [])

        feature_cols = [
            c for c in numeric_cols
            if c != self.target and c not in id_cols and c in self.df.columns
        ]

        if not feature_cols:
            raise ValueError("No valid numeric features for ML.")

        X = self.df[feature_cols].copy()
        y = self.df[self.target].copy()

        return X, y

    def prepare_data(self):

        self.X, self.y = self._prepare_ml_features()

        if self.task_type == "classification":
            from sklearn.preprocessing import LabelEncoder
            self.label_encoder = LabelEncoder()
            self.y = self.label_encoder.fit_transform(self.y)

        return {
            "rows": self.X.shape[0],
            "features": self.X.shape[1],
            "task_type": self.task_type,
            "feature_columns": list(self.X.columns)
        }

    # --------------------------------------------------
    # TRAIN MODELS (FIXED)
    # --------------------------------------------------
    def train_all_models(self):

        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.dummy import DummyClassifier

        from sklearn.linear_model import LogisticRegression, LinearRegression
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.svm import SVC
        from sklearn.neighbors import KNeighborsClassifier

        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        # ---------------- CLASSIFICATION ----------------
        if self.task_type == "classification":

            scoring = "f1_weighted"

            models = {
                "Baseline": DummyClassifier(strategy="most_frequent"),

                "LogisticRegression": Pipeline([
                    ("scaler", StandardScaler()),
                    ("model", LogisticRegression(max_iter=1000))
                ]),

                "SVC": Pipeline([
                    ("scaler", StandardScaler()),
                    ("model", SVC(C=10, gamma="scale"))
                ]),

                "KNN": Pipeline([
                    ("scaler", StandardScaler()),
                    ("model", KNeighborsClassifier())
                ]),

                "RandomForestClassifier": RandomForestClassifier(
                    n_estimators=200,
                    random_state=42
                )
            }

        # ---------------- REGRESSION ----------------
        else:

            scoring = "r2"

            models = {
                "LinearRegression": Pipeline([
                    ("scaler", StandardScaler()),
                    ("model", LinearRegression())
                ]),

                "RandomForestRegressor": RandomForestRegressor(
                    n_estimators=200,
                    random_state=42
                )
            }

        # ---------------- TRAIN LOOP ----------------
        for name, model in models.items():
            try:
                scores = cross_val_score(
                    model,
                    self.X,
                    self.y,
                    cv=kf,
                    scoring=scoring
                )
            except Exception as e:
                logger.warning(f"{name} failed: {e}")
                continue

            mean_score = scores.mean()

            self.results[name] = {
                "cv_scores": scores.tolist(),
                "cv_mean_score": float(mean_score)
            }

            if mean_score > self.best_score:
                self.best_score = mean_score
                self.best_model = model
                self.best_model_name = name

        return self.results

    # --------------------------------------------------
    # RETRAIN + SAVE
    # --------------------------------------------------
    def retrain_best_model(self):

        if self.best_model is None:
            raise ValueError("No best model selected.")

        self.best_model.fit(self.X, self.y)
        return self.best_model

    def save_best_model(self, output_path="best_model.pkl"):
        import joblib

        joblib.dump(
            {
                "model": self.best_model,
                "label_encoder": self.label_encoder,
                "task_type": self.task_type,
                "features": list(self.X.columns)
            },
            output_path
        )
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

        return summary

"""
trainer.py

Schema-driven AutoML trainer for CortexAI.

SAFE & GENERAL:
- Binary & multiclass classification
- Regression
- Pipeline-safe scoring
- No silent model drops
"""

import logging
import pandas as pd
from typing import Dict

from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, make_scorer

logger = logging.getLogger("ModelTrainer")
logger.setLevel(logging.INFO)


class ModelTrainer:
    def __init__(self, df: pd.DataFrame, schema: Dict):
        self.df = df
        self.schema = schema

        self.target = schema["target"]
        self.task_type = schema["task_type"]

        if self.target not in df.columns:
            raise ValueError("Target column missing in trainer.")

        self.X = None
        self.y = None

        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.best_score = -1e9
        self.metric_used = None
        self.label_encoder = None

    # --------------------------------------------------
    # FEATURE PREPARATION
    # --------------------------------------------------
    def prepare_data(self):

        numeric = self.schema.get("numeric", [])
        ordinal = self.schema.get("ordinal", [])
        categorical = self.schema.get("categorical", [])
        id_cols = self.schema.get("id_columns", [])

        feature_cols = [
            c for c in (numeric + ordinal + categorical)
            if c in self.df.columns and c not in id_cols and c != self.target
        ]

        if not feature_cols:
            raise ValueError("No valid features found for training.")

        self.X = self.df[feature_cols].copy()
        y = self.df[self.target].copy()

        if self.task_type == "classification":
            self.label_encoder = LabelEncoder()
            self.y = self.label_encoder.fit_transform(y)
        else:
            self.y = y

        return {
            "rows": self.X.shape[0],
            "features": self.X.shape[1],
            "task_type": self.task_type,
            "feature_columns": feature_cols
        }

    # --------------------------------------------------
    # TRAIN MODELS
    # --------------------------------------------------
    def train_all_models(self):

        from sklearn.dummy import DummyClassifier, DummyRegressor
        from sklearn.linear_model import LogisticRegression, LinearRegression
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.svm import SVC
        from sklearn.neighbors import KNeighborsClassifier

        numeric = [
    c for c in (self.schema.get("numeric", []) + self.schema.get("ordinal", []))
    if c in self.X.columns
]

        categorical = [
    c for c in self.schema.get("categorical", [])
    if c in self.X.columns
]


        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numeric),
                ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ],
            remainder="drop"
        )

        # ---------------- CV + SCORER ----------------
        if self.task_type == "classification":
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

            n_classes = len(set(self.y))
            if n_classes > 2:
                scorer = make_scorer(f1_score, average="macro")
                self.metric_used = "f1_macro"
            else:
                scorer = make_scorer(f1_score, average="weighted")
                self.metric_used = "f1_weighted"

        else:
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
            scorer = "r2"
            self.metric_used = "r2"

        # ---------------- MODELS ----------------
        if self.task_type == "classification":
            models = {
                "Baseline": DummyClassifier(strategy="most_frequent"),

                "LogisticRegression": Pipeline([
                    ("prep", preprocessor),
                    ("model", LogisticRegression(max_iter=1000))
                ]),

                "SVC": Pipeline([
                    ("prep", preprocessor),
                    ("model", SVC(C=10, gamma="scale"))
                ]),

                "KNN": Pipeline([
                    ("prep", preprocessor),
                    ("model", KNeighborsClassifier())
                ]),

                "RandomForestClassifier": Pipeline([
                    ("prep", preprocessor),
                    ("model", RandomForestClassifier(
                        n_estimators=200,
                        random_state=42
                    ))
                ])
            }

        else:
            models = {
                "Baseline": DummyRegressor(strategy="mean"),

                "LinearRegression": Pipeline([
                    ("prep", preprocessor),
                    ("model", LinearRegression())
                ]),

                "RandomForestRegressor": Pipeline([
                    ("prep", preprocessor),
                    ("model", RandomForestRegressor(
                        n_estimators=200,
                        random_state=42
                    ))
                ])
            }

        # ---------------- TRAIN LOOP ----------------
        for name, model in models.items():
            scores = cross_val_score(
                model,
                self.X,
                self.y,
                cv=cv,
                scoring=scorer
            )

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
        self.best_model.fit(self.X, self.y)
        return self.best_model

    def save_best_model(self, output_path="best_model.pkl"):
        import joblib

        joblib.dump(
            {
                "model": self.best_model,
                "label_encoder": self.label_encoder,
                "task_type": self.task_type,
                "metric": self.metric_used,
                "features": list(self.X.columns)
            },
            output_path
        )
        return output_path

    def save_training_summary(self, output_path="training_summary.json"):
        import json

        summary = {
            "task_type": self.task_type,
            "metric": self.metric_used,
            "best_model": self.best_model_name,
            "best_score": float(self.best_score),
            "all_model_scores": self.results
        }

        with open(output_path, "w") as f:
            json.dump(summary, f, indent=4)

        return summary

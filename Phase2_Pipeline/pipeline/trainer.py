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


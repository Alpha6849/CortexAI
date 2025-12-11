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

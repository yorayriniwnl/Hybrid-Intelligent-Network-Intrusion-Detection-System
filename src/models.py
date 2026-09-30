"""
ML Models Module for Hybrid Intelligent Network Intrusion Detection System
FR3: Random Forest Baseline & XGBoost Advanced Supervised Classifier
"""

import time
import os
import joblib
from typing import Dict, Any, Tuple
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

class MLModelTrainer:
    """
    Manages initialization, training, time-profiling, and persistence
    for Random Forest and XGBoost network flow classifiers.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.rf_model: RandomForestClassifier = None
        self.xgb_model: XGBClassifier = None
        self.training_times: Dict[str, float] = {}

    def train_random_forest(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        n_estimators: int = 100,
        max_depth: int = 20
    ) -> Tuple[RandomForestClassifier, float]:
        """
        Trains Random Forest classifier and records exact training duration.
        """
        print(f"\n[+] Training Random Forest Classifier (n_estimators={n_estimators}, max_depth={max_depth})...")
        self.rf_model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=self.random_state,
            n_jobs=-1,
            class_weight='balanced'
        )

        start = time.perf_counter()
        self.rf_model.fit(X_train, y_train)
        duration = time.perf_counter() - start
        self.training_times["Random Forest"] = duration
        print(f"[+] Random Forest training completed in {duration:.2f} seconds.")
        return self.rf_model, duration

    def train_xgboost(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1
    ) -> Tuple[XGBClassifier, float]:
        """
        Trains XGBoost classifier and records exact training duration.
        """
        print(f"\n[+] Training XGBoost Classifier (n_estimators={n_estimators}, max_depth={max_depth}, lr={learning_rate})...")
        
        num_classes = len(np.unique(y_train))
        objective = 'binary:logistic' if num_classes == 2 else 'multi:softprob'
        eval_metric = 'logloss' if num_classes == 2 else 'mlogloss'

        self.xgb_model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            objective=objective,
            eval_metric=eval_metric,
            random_state=self.random_state,
            n_jobs=-1,
            tree_method='hist'  # fast histogram-based algorithm
        )

        start = time.perf_counter()
        self.xgb_model.fit(X_train, y_train)
        duration = time.perf_counter() - start
        self.training_times["XGBoost"] = duration
        print(f"[+] XGBoost training completed in {duration:.2f} seconds.")
        return self.xgb_model, duration

    def save_models(self, artifacts_dir: str):
        """Saves trained models to artifacts directory."""
        os.makedirs(artifacts_dir, exist_ok=True)
        if self.rf_model:
            rf_path = os.path.join(artifacts_dir, "random_forest_model.joblib")
            joblib.dump(self.rf_model, rf_path)
            print(f"[*] Saved Random Forest model to: {rf_path}")
        if self.xgb_model:
            xgb_path = os.path.join(artifacts_dir, "xgboost_model.joblib")
            joblib.dump(self.xgb_model, xgb_path)
            print(f"[*] Saved XGBoost model to: {xgb_path}")

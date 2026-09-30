"""
Sample Inference Module for Hybrid Intelligent Network Intrusion Detection System
Passes unseen test network flows through trained models and displays granular predictions
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any

class UnseenFlowPredictor:
    """
    Simulates real-time packet/flow stream inference on held-out test samples.
    """

    def __init__(self, preprocessor, rf_model, xgb_model, class_names: List[str]):
        self.preprocessor = preprocessor
        self.rf_model = rf_model
        self.xgb_model = xgb_model
        self.class_names = class_names

    def run_sample_predictions(
        self,
        X_test_scaled: np.ndarray,
        y_test: np.ndarray,
        raw_test_df: pd.DataFrame,
        num_samples_per_class: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Selects balanced unseen test samples and executes dual-model prediction with probabilities.
        """
        selected_indices = []
        for class_idx in range(len(self.class_names)):
            class_indices = np.where(y_test == class_idx)[0]
            if len(class_indices) > 0:
                sample_idx = np.random.choice(class_indices, size=min(num_samples_per_class, len(class_indices)), replace=False)
                selected_indices.extend(sample_idx)

        # Shuffle selected indices
        np.random.shuffle(selected_indices)

        predictions_log = []
        print("\n" + "="*85)
        print(f" LIVE INFERENCE DEMONSTRATION ON UNSEEN HELD-OUT TEST NETWORK FLOWS ({len(selected_indices)} SAMPLES)")
        print("="*85)

        for i, idx in enumerate(selected_indices, 1):
            sample_vector = X_test_scaled[idx].reshape(1, -1)
            true_class_idx = y_test[idx]
            true_label = self.class_names[true_class_idx]

            # Random Forest Inference
            rf_pred_idx = self.rf_model.predict(sample_vector)[0]
            rf_proba = self.rf_model.predict_proba(sample_vector)[0]
            rf_pred_label = self.class_names[rf_pred_idx]
            rf_conf = float(rf_proba[rf_pred_idx])

            # XGBoost Inference
            xgb_pred_idx = self.xgb_model.predict(sample_vector)[0]
            xgb_proba = self.xgb_model.predict_proba(sample_vector)[0]
            xgb_pred_label = self.class_names[xgb_pred_idx]
            xgb_conf = float(xgb_proba[xgb_pred_idx])

            # Extract sample flow summary characteristics
            flow_info = {}
            if raw_test_df is not None and idx < len(raw_test_df):
                row = raw_test_df.iloc[idx]
                candidate_cols = [
                    'Destination Port', 'Flow Duration', 'Total Fwd Packets',
                    'Total Backward Packets', 'Fwd Packet Length Mean', 'Flow Bytes/s'
                ]
                for c in candidate_cols:
                    if c in row.index:
                        flow_info[c] = round(float(row[c]), 2) if isinstance(row[c], (int, float, np.number)) else str(row[c])

            entry = {
                "sample_id": int(i),
                "true_label": true_label,
                "rf_predicted": rf_pred_label,
                "rf_confidence": round(rf_conf, 4),
                "rf_correct": bool(rf_pred_label == true_label),
                "xgb_predicted": xgb_pred_label,
                "xgb_confidence": round(xgb_conf, 4),
                "xgb_correct": bool(xgb_pred_label == true_label),
                "key_flow_features": flow_info
            }
            predictions_log.append(entry)

            status_icon = "MATCH" if (entry["rf_correct"] and entry["xgb_correct"]) else "DISCREPANCY"
            print(f"Sample #{i:02d} | True: {true_label:<10} | [{status_icon}]")
            print(f"   Key Features: {flow_info}")
            print(f"   -> Random Forest: {rf_pred_label:<10} (Confidence: {rf_conf*100:.2f}%)")
            print(f"   -> XGBoost:       {xgb_pred_label:<10} (Confidence: {xgb_conf*100:.2f}%)")
            print("-" * 85)

        return predictions_log

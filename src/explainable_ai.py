"""
Explainable AI (XAI) Module for Hybrid Intelligent Network Intrusion Detection System
FR6: SHAP Feature Attributions, Local Explanations, and Global Importance Visualizations
"""

import os
import numpy as np
import pandas as pd
import xgboost as xgb
from typing import Dict, Any, List

class NetworkExplainableAI:
    """
    Computes exact game-theoretic SHAP (SHapley Additive exPlanations)
    attributions for individual network flow predictions and global risk factors.
    """

    def __init__(self, xgb_model, feature_names: List[str], output_dir: str = "artifacts/figures"):
        self.xgb_model = xgb_model
        self.feature_names = feature_names
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def explain_single_flow(self, flow_vector: np.ndarray, top_k: int = 6) -> Dict[str, Any]:
        """
        Computes exact Tree SHAP attributions for an individual network flow.
        """
        if flow_vector.ndim == 1:
            flow_vector = flow_vector.reshape(1, -1)

        dmat = xgb.DMatrix(flow_vector, feature_names=self.feature_names)
        # XGBoost C++ native tree shap: (1, num_features + 1)
        contribs = self.xgb_model.get_booster().predict(dmat, pred_contribs=True)[0]
        
        feature_shap = contribs[:-1]
        base_value = float(contribs[-1])

        # Sort by absolute contribution
        sorted_indices = np.argsort(np.abs(feature_shap))[::-1]
        
        top_positive = []
        top_negative = []
        for idx in sorted_indices:
            feat_name = self.feature_names[idx]
            val = float(feature_shap[idx])
            feat_val = float(flow_vector[0, idx])
            item = {"feature": feat_name, "shap_value": round(val, 4), "feature_value": round(feat_val, 4)}
            if val > 0 and len(top_positive) < top_k:
                top_positive.append(item)
            elif val < 0 and len(top_negative) < top_k:
                top_negative.append(item)

        return {
            "base_value": round(base_value, 4),
            "top_attack_indicators": top_positive,
            "top_benign_indicators": top_negative,
            "all_shap_values": {self.feature_names[i]: float(feature_shap[i]) for i in range(len(self.feature_names))}
        }

    def generate_global_shap_summary(self, X_sample: np.ndarray, sample_size: int = 500) -> str:
        """
        Calculates global SHAP importance across a sample and saves a summary figure.
        """
        if len(X_sample) > sample_size:
            idx = np.random.choice(len(X_sample), sample_size, replace=False)
            X_eval = X_sample[idx]
        else:
            X_eval = X_sample

        dmat = xgb.DMatrix(X_eval, feature_names=self.feature_names)
        contribs = self.xgb_model.get_booster().predict(dmat, pred_contribs=True)
        feature_contribs = contribs[:, :-1]

        # Mean absolute SHAP per feature
        mean_abs_shap = np.mean(np.abs(feature_contribs), axis=0)
        top_indices = np.argsort(mean_abs_shap)[::-1][:12]

        top_names = [self.feature_names[i] for i in top_indices][::-1]
        top_scores = mean_abs_shap[top_indices][::-1]

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(top_names, top_scores, color='#e74c3c', edgecolor='black', linewidth=1.1)
        ax.set_title("Global Explainable AI: Mean |SHAP| Feature Attributions", fontsize=13, fontweight='bold', pad=12)
        ax.set_xlabel("Mean Absolute SHAP Value (Impact on Intrusion Prediction)", fontsize=11, fontweight='bold')
        ax.grid(axis='x', linestyle='--', alpha=0.7)

        for bar in bars:
            w = bar.get_width()
            ax.text(w + 0.01, bar.get_y() + bar.get_height()/2, f"{w:.3f}", va='center', fontsize=9.5, fontweight='bold')

        plt.tight_layout()
        filepath = os.path.join(self.output_dir, "shap_summary.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[*] Saved SHAP summary plot to: {filepath}")
        return filepath

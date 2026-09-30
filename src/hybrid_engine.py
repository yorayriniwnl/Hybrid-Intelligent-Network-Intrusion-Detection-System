"""
Hybrid Decision Fusion Engine for Network Intrusion Detection
FR5: Fuses Supervised Known-Attack Classification with Autoencoder Anomaly Scoring
"""

import numpy as np
from typing import Dict, Any, List

class HybridDecisionEngine:
    """
    Fuses Supervised Classification Probabilities with Unsupervised Autoencoder Anomaly Scores.
    Solves the closed-world assumption by categorizing flows into:
      1. Normal (BENIGN)
      2. Known Attack (Supervised detection)
      3. Suspicious / Held-Out Attack (Out-of-distribution anomaly detection)
    """

    def __init__(self, supervised_model, autoencoder, class_names: List[str], confidence_threshold: float = 0.85):
        self.supervised_model = supervised_model
        self.autoencoder = autoencoder
        self.class_names = class_names
        self.confidence_threshold = confidence_threshold

    def evaluate_flow(self, flow_vector: np.ndarray) -> Dict[str, Any]:
        """
        Runs dual-stream inference on a single or batch feature vector.
        """
        if flow_vector.ndim == 1:
            flow_vector = flow_vector.reshape(1, -1)

        # 1. Supervised Stream
        sup_probs = self.supervised_model.predict_proba(flow_vector)[0]
        sup_class_idx = int(np.argmax(sup_probs))
        sup_label = self.class_names[sup_class_idx]
        sup_conf = float(sup_probs[sup_class_idx])

        # 2. Unsupervised Anomaly Stream
        is_anom, recon_errors = self.autoencoder.predict_anomalies(flow_vector)
        anomaly_flag = bool(is_anom[0])
        recon_error = float(recon_errors[0])
        threshold = float(self.autoencoder.anomaly_threshold)

        # 3. Hybrid Fusion Logic (FR5)
        if sup_label != "BENIGN" and sup_conf >= self.confidence_threshold:
            # Case A: High-confidence known attack
            final_status = "Known Attack"
            sub_category = sup_label
            rationale = f"Matched supervised signature for {sup_label} ({sup_conf*100:.1f}% confidence)."
            severity = "HIGH"
        elif sup_label == "BENIGN" and not anomaly_flag and sup_conf >= self.confidence_threshold:
            # Case B: Concordant legitimate traffic
            final_status = "Normal"
            sub_category = "BENIGN"
            rationale = f"Normal verified flow (Reconstruction error {recon_error:.4f} <= threshold {threshold:.4f})."
            severity = "LOW"
        elif anomaly_flag:
            # Case C: High reconstruction error -> Out-of-Distribution / Zero-Day / Held-Out attack
            final_status = "Suspicious / Held-Out Attack"
            sub_category = f"Anomalous (Error {recon_error:.3f} > {threshold:.3f})"
            rationale = f"Supervised labeled as '{sup_label}', but Autoencoder detected abnormal flow reconstruction spike."
            severity = "CRITICAL"
        else:
            # Case D: Ambiguous classification
            final_status = "Suspicious / Anomalous"
            sub_category = "Low-Confidence Traffic"
            rationale = f"Supervised confidence ({sup_conf*100:.1f}%) below threshold {self.confidence_threshold*100:.0f}%."
            severity = "MEDIUM"

        return {
            "final_verdict": final_status,
            "sub_category": sub_category,
            "severity": severity,
            "supervised_label": sup_label,
            "supervised_confidence": round(sup_conf, 4),
            "anomaly_score": round(recon_error, 5),
            "anomaly_threshold": round(threshold, 5),
            "is_anomaly": anomaly_flag,
            "rationale": rationale
        }

    def evaluate_batch(self, X_batch: np.ndarray) -> List[Dict[str, Any]]:
        """Evaluates batch of network flows."""
        return [self.evaluate_flow(X_batch[i]) for i in range(len(X_batch))]

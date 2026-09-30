"""
Evaluation Module for Hybrid Intelligent Network Intrusion Detection System
Reports genuine test metrics: Precision, Recall, F1, Accuracy, Confusion Matrix, FPR, FNR, and Inference Latency
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

class ModelEvaluator:
    """
    Computes rigorous cybersecurity-focused evaluation metrics
    from actual holdout test predictions.
    """

    def __init__(self, class_names: List[str]):
        self.class_names = class_names

    def evaluate_model(
        self,
        model_name: str,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        training_time: float
    ) -> Dict[str, Any]:
        """
        Runs inference on X_test, measures latency, and calculates all benchmark metrics.
        """
        print(f"\n[*] Evaluating {model_name} on {len(X_test):,} holdout test samples...")

        # Measure inference latency
        start_time = time.perf_counter()
        y_pred = model.predict(X_test)
        inference_duration = time.perf_counter() - start_time
        latency_per_sample_us = (inference_duration / len(X_test)) * 1_000_000
        throughput = len(X_test) / inference_duration

        # Predict probabilities if supported
        y_proba = None
        roc_auc = None
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)
            try:
                if len(self.class_names) == 2:
                    roc_auc = float(roc_auc_score(y_test, y_proba[:, 1]))
                else:
                    roc_auc = float(roc_auc_score(y_test, y_proba, multi_class='ovr'))
            except Exception as e:
                roc_auc = None

        # Standard metrics
        acc = float(accuracy_score(y_test, y_pred))
        prec_macro = float(precision_score(y_test, y_pred, average='macro', zero_division=0))
        rec_macro = float(recall_score(y_test, y_pred, average='macro', zero_division=0))
        f1_macro = float(f1_score(y_test, y_pred, average='macro', zero_division=0))

        prec_weighted = float(precision_score(y_test, y_pred, average='weighted', zero_division=0))
        rec_weighted = float(recall_score(y_test, y_pred, average='weighted', zero_division=0))
        f1_weighted = float(f1_score(y_test, y_pred, average='weighted', zero_division=0))

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Calculate FPR and FNR for binary or class 0 (Benign) vs class 1 (Attack)
        fpr = None
        fnr = None
        if len(self.class_names) == 2:
            tn, fp, fn, tp = cm.ravel()
            fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
            fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

        # Detailed per-class report
        clf_report_dict = classification_report(
            y_test,
            y_pred,
            target_names=self.class_names,
            output_dict=True,
            zero_division=0
        )

        results = {
            "model_name": model_name,
            "test_samples": len(y_test),
            "training_time_seconds": round(training_time, 3),
            "inference_time_seconds": round(inference_duration, 4),
            "latency_microseconds_per_flow": round(latency_per_sample_us, 2),
            "throughput_flows_per_sec": round(throughput, 1),
            "accuracy": round(acc, 5),
            "precision_macro": round(prec_macro, 5),
            "recall_macro": round(rec_macro, 5),
            "f1_score_macro": round(f1_macro, 5),
            "precision_weighted": round(prec_weighted, 5),
            "recall_weighted": round(rec_weighted, 5),
            "f1_score_weighted": round(f1_weighted, 5),
            "false_positive_rate": round(fpr, 5) if fpr is not None else "N/A",
            "false_negative_rate": round(fnr, 5) if fnr is not None else "N/A",
            "roc_auc": round(roc_auc, 5) if roc_auc is not None else "N/A",
            "confusion_matrix": cm.tolist(),
            "classification_report": clf_report_dict
        }

        print(f"    - Accuracy:  {acc*100:.2f}%")
        print(f"    - Precision: {prec_macro*100:.2f}% (macro) | {prec_weighted*100:.2f}% (weighted)")
        print(f"    - Recall:    {rec_macro*100:.2f}% (macro) | {rec_weighted*100:.2f}% (weighted)")
        print(f"    - F1-Score:  {f1_macro*100:.2f}% (macro) | {f1_weighted*100:.2f}% (weighted)")
        if fpr is not None:
            print(f"    - False Positive Rate (FPR): {fpr*100:.4f}%")
            print(f"    - False Negative Rate (FNR): {fnr*100:.4f}%")
        if roc_auc is not None:
            print(f"    - ROC-AUC:   {roc_auc:.4f}")
        print(f"    - Latency:   {latency_per_sample_us:.2f} µs/flow ({throughput:,.0f} flows/sec)")

        return results

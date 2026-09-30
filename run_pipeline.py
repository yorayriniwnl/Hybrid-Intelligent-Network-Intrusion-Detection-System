"""
End-to-End Pipeline Runner for Hybrid Intelligent Network Intrusion Detection System
Next-Week Faculty Evaluation Prototype Execution Script
"""

import os
import sys
import json
import argparse
import pandas as pd
import numpy as np

# Ensure src module is on sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.data_loader import NetworkDataLoader
from src.preprocessor import NetworkDataPreprocessor
from src.models import MLModelTrainer
from src.evaluate import ModelEvaluator
from src.visualize import NetworkVisualizer
from src.sample_inference import UnseenFlowPredictor

def run_evaluation_prototype(
    dataset_path: str = "data/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    sample_size: int = 100000,
    test_size: float = 0.20,
    random_state: int = 42
):
    print("=" * 90)
    print(" HYBRID INTELLIGENT NETWORK INTRUSION DETECTION SYSTEM (H-NIDS)")
    print(" PHASE 1 EVALUATION-READY ML PROTOTYPE — ACADEMIC FACULTY BENCHMARK")
    print("=" * 90)

    # Output directories
    artifacts_dir = "artifacts"
    models_dir = os.path.join(artifacts_dir, "models")
    figures_dir = os.path.join(artifacts_dir, "figures")
    results_dir = os.path.join(artifacts_dir, "results")
    for d in [models_dir, figures_dir, results_dir]:
        os.makedirs(d, exist_ok=True)

    # ---------------------------------------------------------
    # STEP 1: DATASET LOADING AND INSPECTION
    # ---------------------------------------------------------
    print("\n" + "#" * 40 + " [STEP 1: DATASET INGESTION & INSPECTION] " + "#" * 40)
    loader = NetworkDataLoader(dataset_path)
    df_raw = loader.load_data(sample_size=sample_size, random_state=random_state)
    inspection = loader.inspect_dataset()

    print(f"\n[*] Dataset Shape: {df_raw.shape[0]:,} flows across {df_raw.shape[1]} columns")
    print(f"[*] Total Network Flow Features: {inspection['total_features']}")
    print(f"[*] Label Column Identified: '{inspection['label_column']}'")
    print("\n[*] Initial Class Distribution:")
    for label, stat in inspection['class_distribution'].items():
        print(f"    - {label:<15}: {stat['count']:,} flows ({stat['percentage']}%)")

    print(f"[*] Missing Values Detected: {inspection['total_null_values']:,} across {len(inspection['columns_with_nulls'])} columns")
    print(f"[*] Infinite Values Detected: {inspection['total_infinite_values']:,} across {len(inspection['columns_with_infs'])} columns")

    # ---------------------------------------------------------
    # STEP 2: DATA CLEANING & PREPROCESSING (Strict Featurization)
    # ---------------------------------------------------------
    print("\n" + "#" * 40 + " [STEP 2: DATA CLEANING & PREPROCESSING] " + "#" * 40)
    preprocessor = NetworkDataPreprocessor(target_col=inspection['label_column'], scaler_type='robust')
    df_clean, cleaning_stats = preprocessor.clean_raw_dataframe(df_raw)

    print(f"[*] Initial records: {cleaning_stats['initial_rows']:,}")
    print(f"[*] Duplicate flows removed: {cleaning_stats['duplicates_removed']:,}")
    print(f"[*] Remaining clean flows: {cleaning_stats['deduplicated_rows']:,}")
    print(f"[*] Infinite values scrubbed: {cleaning_stats['infinite_values_cleaned']:,}")

    # Generate Class Distribution Visualization
    visualizer = NetworkVisualizer(figures_dir)
    clean_dist = {
        k: {
            "count": int(v),
            "percentage": round(v / len(df_clean) * 100, 2)
        }
        for k, v in df_clean[inspection['label_column']].value_counts().items()
    }
    dist_plot_path = visualizer.plot_class_distribution(clean_dist)

    # Leakage-Free Stratified Train/Test Split
    X_train_scaled, X_test_scaled, y_train, y_test, split_meta = preprocessor.prepare_train_test_split(
        df_clean,
        test_size=test_size,
        random_state=random_state
    )

    # Save Preprocessor Artifact
    preprocessor_path = os.path.join(models_dir, "preprocessor.joblib")
    preprocessor.save(preprocessor_path)

    # ---------------------------------------------------------
    # STEP 3: MODEL TRAINING (RANDOM FOREST & XGBOOST)
    # ---------------------------------------------------------
    print("\n" + "#" * 40 + " [STEP 3: SUPERVISED MODEL TRAINING] " + "#" * 40)
    trainer = MLModelTrainer(random_state=random_state)
    
    # 3A. Random Forest
    rf_model, rf_train_time = trainer.train_random_forest(
        X_train_scaled, y_train, n_estimators=100, max_depth=20
    )

    # 3B. XGBoost
    xgb_model, xgb_train_time = trainer.train_xgboost(
        X_train_scaled, y_train, n_estimators=100, max_depth=6, learning_rate=0.1
    )

    trainer.save_models(models_dir)

    # ---------------------------------------------------------
    # STEP 4: MODEL EVALUATION ON HOLDOUT TEST SET
    # ---------------------------------------------------------
    print("\n" + "#" * 40 + " [STEP 4: MODEL EVALUATION & METRICS REPORTING] " + "#" * 40)
    evaluator = ModelEvaluator(class_names=preprocessor.classes_)
    
    rf_eval = evaluator.evaluate_model(
        "Random Forest", rf_model, X_test_scaled, y_test, rf_train_time
    )
    xgb_eval = evaluator.evaluate_model(
        "XGBoost", xgb_model, X_test_scaled, y_test, xgb_train_time
    )

    # Save Evaluation Metrics JSON
    metrics_summary = {
        "dataset_metadata": {
            "dataset_source": dataset_path,
            "total_samples": len(df_raw),
            "cleaned_samples": len(df_clean),
            "train_samples": len(X_train_scaled),
            "test_samples": len(X_test_scaled),
            "features_used": len(preprocessor.feature_columns),
            "classes": preprocessor.classes_
        },
        "models": {
            "Random Forest": rf_eval,
            "XGBoost": xgb_eval
        }
    }
    metrics_file = os.path.join(results_dir, "evaluation_metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"\n[*] Full evaluation metrics saved to: {metrics_file}")

    # ---------------------------------------------------------
    # STEP 5: VISUALIZATIONS & CONFUSION MATRICES
    # ---------------------------------------------------------
    print("\n" + "#" * 40 + " [STEP 5: VISUALIZATIONS & ARTIFACT GENERATION] " + "#" * 40)
    cm_rf = np.array(rf_eval["confusion_matrix"])
    cm_xgb = np.array(xgb_eval["confusion_matrix"])
    
    cm_rf_plot = visualizer.plot_confusion_matrix(cm_rf, preprocessor.classes_, "Random Forest")
    cm_xgb_plot = visualizer.plot_confusion_matrix(cm_xgb, preprocessor.classes_, "XGBoost")
    comp_plot = visualizer.plot_model_comparison(rf_eval, xgb_eval)
    feat_plot = visualizer.plot_feature_importances(preprocessor.feature_columns, rf_model, xgb_model, top_k=12)

    # ---------------------------------------------------------
    # STEP 6: UNSEEN TEST SAMPLES LIVE INFERENCE
    # ---------------------------------------------------------
    print("\n" + "#" * 40 + " [STEP 6: UNSEEN TEST SAMPLES PREDICTION DEMONSTRATION] " + "#" * 40)
    predictor = UnseenFlowPredictor(
        preprocessor, rf_model, xgb_model, preprocessor.classes_
    )
    # Reconstruct raw test dataframe view for feature display
    raw_test_df = df_clean.iloc[-len(y_test):].reset_index(drop=True)
    unseen_preds = predictor.run_sample_predictions(
        X_test_scaled, y_test, raw_test_df, num_samples_per_class=4
    )
    
    preds_file = os.path.join(results_dir, "unseen_test_predictions.json")
    with open(preds_file, "w") as f:
        json.dump(unseen_preds, f, indent=2)
    print(f"[*] Unseen test predictions saved to: {preds_file}")

    # ---------------------------------------------------------
    # STEP 7: SIDE-BY-SIDE MODEL COMPARISON TABLE
    # ---------------------------------------------------------
    print("\n" + "#" * 40 + " [STEP 7: RANDOM FOREST vs XGBOOST COMPARISON] " + "#" * 40)
    comparison_table = pd.DataFrame([
        {
            "Metric": "Accuracy",
            "Random Forest": f"{rf_eval['accuracy']*100:.3f}%",
            "XGBoost": f"{xgb_eval['accuracy']*100:.3f}%",
            "Delta (XGB - RF)": f"{(xgb_eval['accuracy'] - rf_eval['accuracy'])*100:+.3f}%"
        },
        {
            "Metric": "Precision (Macro)",
            "Random Forest": f"{rf_eval['precision_macro']*100:.3f}%",
            "XGBoost": f"{xgb_eval['precision_macro']*100:.3f}%",
            "Delta (XGB - RF)": f"{(xgb_eval['precision_macro'] - rf_eval['precision_macro'])*100:+.3f}%"
        },
        {
            "Metric": "Recall (Macro)",
            "Random Forest": f"{rf_eval['recall_macro']*100:.3f}%",
            "XGBoost": f"{xgb_eval['recall_macro']*100:.3f}%",
            "Delta (XGB - RF)": f"{(xgb_eval['recall_macro'] - rf_eval['recall_macro'])*100:+.3f}%"
        },
        {
            "Metric": "F1-Score (Macro)",
            "Random Forest": f"{rf_eval['f1_score_macro']*100:.3f}%",
            "XGBoost": f"{xgb_eval['f1_score_macro']*100:.3f}%",
            "Delta (XGB - RF)": f"{(xgb_eval['f1_score_macro'] - rf_eval['f1_score_macro'])*100:+.3f}%"
        },
        {
            "Metric": "False Positive Rate (FPR)",
            "Random Forest": f"{rf_eval['false_positive_rate']*100:.4f}%" if isinstance(rf_eval['false_positive_rate'], float) else "N/A",
            "XGBoost": f"{xgb_eval['false_positive_rate']*100:.4f}%" if isinstance(xgb_eval['false_positive_rate'], float) else "N/A",
            "Delta (XGB - RF)": f"{(xgb_eval['false_positive_rate'] - rf_eval['false_positive_rate'])*100:+.4f}%" if isinstance(rf_eval['false_positive_rate'], float) else "N/A"
        },
        {
            "Metric": "False Negative Rate (FNR)",
            "Random Forest": f"{rf_eval['false_negative_rate']*100:.4f}%" if isinstance(rf_eval['false_negative_rate'], float) else "N/A",
            "XGBoost": f"{xgb_eval['false_negative_rate']*100:.4f}%" if isinstance(xgb_eval['false_negative_rate'], float) else "N/A",
            "Delta (XGB - RF)": f"{(xgb_eval['false_negative_rate'] - rf_eval['false_negative_rate'])*100:+.4f}%" if isinstance(rf_eval['false_negative_rate'], float) else "N/A"
        },
        {
            "Metric": "Training Time",
            "Random Forest": f"{rf_eval['training_time_seconds']:.2f} s",
            "XGBoost": f"{xgb_eval['training_time_seconds']:.2f} s",
            "Delta (XGB - RF)": f"{(xgb_eval['training_time_seconds'] - rf_eval['training_time_seconds']):+.2f} s"
        },
        {
            "Metric": "Inference Latency per Flow",
            "Random Forest": f"{rf_eval['latency_microseconds_per_flow']:.2f} µs",
            "XGBoost": f"{xgb_eval['latency_microseconds_per_flow']:.2f} µs",
            "Delta (XGB - RF)": f"{(xgb_eval['latency_microseconds_per_flow'] - rf_eval['latency_microseconds_per_flow']):+.2f} µs"
        },
        {
            "Metric": "Throughput (Flows/sec)",
            "Random Forest": f"{rf_eval['throughput_flows_per_sec']:,.0f}",
            "XGBoost": f"{xgb_eval['throughput_flows_per_sec']:,.0f}",
            "Delta (XGB - RF)": f"{(xgb_eval['throughput_flows_per_sec'] - rf_eval['throughput_flows_per_sec']):+,.0f}"
        }
    ])
    print(comparison_table.to_string(index=False))

    # ---------------------------------------------------------
    # STEP 8: ARCHITECTURAL TRANSITION TO NEXT PHASE
    # ---------------------------------------------------------
    print("\n" + "#" * 40 + " [STEP 8: NEXT PHASE ARCHITECTURE EXPLANATION] " + "#" * 40)
    print("Why Phase 1 establishes the baseline and what Phase 2 adds:")
    print("  1. Current Supervised ML (RF + XGBoost):")
    print("     - High discriminative accuracy on known, pre-labeled attack patterns.")
    print("     - Sensitive to closed-world assumption (cannot detect unseen attack vectors without high confidence errors).")
    print("  2. Proposed Deep Learning Module (TabTransformer):")
    print("     - Attention-based contextual feature embeddings for tabular network flow features.")
    print("     - Superior modeling of non-linear correlations and cross-feature interactions.")
    print("  3. Proposed Unsupervised Module (Autoencoder):")
    print("     - Trained strictly on normal/benign network traffic.")
    print("     - Reconstructs normal flows; high reconstruction error on novel/zero-day attacks flags them as anomalous.")
    print("  4. Hybrid Decision Engine:")
    print("     - Fuses supervised attack probabilities with unsupervised anomaly scores.")
    print("     - Outputs 3 distinct states: Normal, Known Attack, or Suspicious/Anomalous (Held-Out).")
    print("  5. Explainable AI (SHAP):")
    print("     - Provides flow-level game-theoretic explanations for security operations center (SOC) analysts.")

    print("\n" + "=" * 90)
    print(" SUGGESTED EXPLANATION TO FACULTY EVALUATION COMMITTEE:")
    print("=" * 90)
    print('“In the first phase, we implemented supervised machine learning for network intrusion')
    print('detection using Random Forest and XGBoost. We are evaluating the models using precision,')
    print('recall, F1-score and confusion matrices. The proposed next phase adds TabTransformer for')
    print('attention-based classification and an Autoencoder for anomaly detection of held-out or')
    print('suspicious attack behavior, followed by hybrid decision fusion and Explainable AI.”')
    print("=" * 90)

    return metrics_summary, comparison_table

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run H-NIDS Phase 1 Prototype Pipeline")
    parser.add_argument("--dataset", type=str, default="data/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv")
    parser.add_argument("--sample_size", type=int, default=100000)
    parser.add_argument("--test_size", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_evaluation_prototype(
        dataset_path=args.dataset,
        sample_size=args.sample_size,
        test_size=args.test_size,
        random_state=args.seed
    )

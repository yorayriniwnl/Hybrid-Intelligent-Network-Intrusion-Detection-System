"""
Master Advanced Pipeline Runner (Phases 2 & 3)
Trains TabTransformer, Deep Autoencoder, runs Held-Out Experiment, and generates SHAP explanations.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.data_loader import NetworkDataLoader
from src.preprocessor import NetworkDataPreprocessor
from src.tab_transformer import TabTransformerTrainer
from src.autoencoder import AutoencoderTrainer
from src.hybrid_engine import HybridDecisionEngine
from src.heldout_experiment import HeldOutAttackExperiment
from src.explainable_ai import NetworkExplainableAI
from src.evaluate import ModelEvaluator

def run_advanced_ai_pipeline():
    print("=" * 90)
    print(" HYBRID INTELLIGENT NIDS — PHASES 2 & 3 ADVANCED AI & EXPLAINABILITY ENGINE")
    print("=" * 90)

    models_dir = "artifacts/models"
    results_dir = "artifacts/results"
    figures_dir = "artifacts/figures"
    for d in [models_dir, results_dir, figures_dir]:
        os.makedirs(d, exist_ok=True)

    # 1. Load Preprocessor and Existing Supervised Baseline
    print("\n[*] Loading Phase 1 baseline artifacts...")
    preprocessor = NetworkDataPreprocessor.load(os.path.join(models_dir, "preprocessor.joblib"))
    xgb_model = joblib.load(os.path.join(models_dir, "xgboost_model.joblib"))

    # Reload clean data
    loader = NetworkDataLoader("data/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv")
    df_raw = loader.load_data(sample_size=60000, random_state=42)
    df_clean, _ = preprocessor.clean_raw_dataframe(df_raw)

    X_train, X_test, y_train, y_test, _ = preprocessor.prepare_train_test_split(df_clean, test_size=0.20, random_state=42)
    num_features = X_train.shape[1]

    # 2. Phase 2A: Train TabTransformer
    print("\n" + "#" * 40 + " [PHASE 2A: TABTRANSFORMER ATTENTION CLASSIFIER] " + "#" * 40)
    tab_trainer = TabTransformerTrainer(num_features=num_features, num_classes=len(preprocessor.classes_))
    tab_train_time = tab_trainer.fit(X_train, y_train, epochs=4, batch_size=256)
    tab_trainer.save(os.path.join(models_dir, "tab_transformer.pt"))

    # Evaluate TabTransformer on holdout test set
    evaluator = ModelEvaluator(class_names=preprocessor.classes_)
    tab_eval = evaluator.evaluate_model("TabTransformer", tab_trainer, X_test, y_test, tab_train_time)

    # 3. Phase 2B: Train Autoencoder on BENIGN flows only
    print("\n" + "#" * 40 + " [PHASE 2B: DEEP AUTOENCODER ANOMALY DETECTOR] " + "#" * 40)
    benign_idx = np.where(y_train == 0)[0]  # Class 0 = BENIGN
    X_benign_train = X_train[benign_idx]
    
    # Split benign train into train and validation for thresholding
    val_split = int(len(X_benign_train) * 0.85)
    X_b_tr = X_benign_train[:val_split]
    X_b_val = X_benign_train[val_split:]

    ae_trainer = AutoencoderTrainer(input_dim=num_features, latent_dim=12)
    ae_stats = ae_trainer.fit_on_benign(X_b_tr, X_b_val, epochs=5, batch_size=256, percentile_threshold=98.5)
    ae_trainer.save(os.path.join(models_dir, "autoencoder.pt"))

    # 4. Phase 2C: Hybrid Decision Engine
    print("\n" + "#" * 40 + " [PHASE 2C: HYBRID DECISION ENGINE] " + "#" * 40)
    hybrid_engine = HybridDecisionEngine(supervised_model=xgb_model, autoencoder=ae_trainer, class_names=preprocessor.classes_)
    sample_decisions = hybrid_engine.evaluate_batch(X_test[:5])
    print("[*] Sample Hybrid Engine Decisions on Test Flows:")
    for i, dec in enumerate(sample_decisions, 1):
        print(f"    Flow #{i}: Verdict = {dec['final_verdict']:<25} | Severity = {dec['severity']:<8} | {dec['rationale']}")

    # 5. Phase 2D: Held-Out Attack Experiment (Zero-Day Simulation with DDoS)
    print("\n" + "#" * 40 + " [PHASE 2D: HELD-OUT ATTACK EXPERIMENT] " + "#" * 40)
    heldout_exp = HeldOutAttackExperiment(preprocessor, xgb_model, ae_trainer, hybrid_engine)
    heldout_results = heldout_exp.run_experiment(
        heldout_csv_path="data/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
        num_samples=1000
    )
    heldout_file = os.path.join(results_dir, "heldout_experiment_results.json")
    with open(heldout_file, "w") as f:
        json.dump(heldout_results, f, indent=2)

    # 6. Phase 3: Explainable AI (SHAP Feature Attributions)
    print("\n" + "#" * 40 + " [PHASE 3: EXPLAINABLE AI (SHAP)] " + "#" * 40)
    xai = NetworkExplainableAI(xgb_model, preprocessor.feature_columns, output_dir=figures_dir)
    shap_plot_path = xai.generate_global_shap_summary(X_test, sample_size=500)
    
    # Explain a single attack flow
    sample_attack_idx = np.where(y_test == 1)[0][0]
    single_explanation = xai.explain_single_flow(X_test[sample_attack_idx])
    print(f"\n[*] Sample SHAP Local Explanation (Attribution for Flow #{sample_attack_idx}):")
    print(f"    Base Expectation E[f(x)]: {single_explanation['base_value']}")
    print(f"    Top Attack Indicators (Positive SHAP):")
    for item in single_explanation['top_attack_indicators'][:4]:
        print(f"      + {item['feature']:<30}: +{item['shap_value']:.4f} (Value: {item['feature_value']})")

    # Save summary of advanced results
    advanced_summary = {
        "tab_transformer_evaluation": tab_eval,
        "autoencoder_stats": ae_stats,
        "heldout_attack_experiment": heldout_results,
        "sample_shap_explanation": {
            "base_value": single_explanation["base_value"],
            "top_attack_indicators": single_explanation["top_attack_indicators"],
            "top_benign_indicators": single_explanation["top_benign_indicators"]
        }
    }
    adv_file = os.path.join(results_dir, "advanced_pipeline_results.json")
    with open(adv_file, "w") as f:
        json.dump(advanced_summary, f, indent=2)
    print(f"\n[*] Advanced AI results saved to: {adv_file}")
    print("=" * 90)
    print(" PHASES 2 & 3 ADVANCED AI & EXPLAINABILITY ENGINE COMPLETED SUCCESSFULLY!")
    print("=" * 90)

if __name__ == "__main__":
    run_advanced_ai_pipeline()

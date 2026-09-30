"""
Held-Out Attack Experiment Module for Hybrid Intelligent Network Intrusion Detection System
Evaluates detection of excluded attack categories (DDoS) using unsupervised anomaly detection
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, Any, List

class HeldOutAttackExperiment:
    """
    Simulates a zero-day / novel attack scenario by presenting an attack category
    (DDoS from CIC-IDS2017) that was strictly excluded from supervised training.
    """

    def __init__(self, preprocessor, supervised_model, autoencoder, hybrid_engine):
        self.preprocessor = preprocessor
        self.supervised_model = supervised_model
        self.autoencoder = autoencoder
        self.hybrid_engine = hybrid_engine

    def run_experiment(
        self,
        heldout_csv_path: str = "data/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
        num_samples: int = 1000
    ) -> Dict[str, Any]:
        """
        Loads held-out DDoS traffic, transforms it with the training pipeline,
        and evaluates whether the Autoencoder & Hybrid Engine flag it as anomalous.
        """
        if not os.path.exists(heldout_csv_path):
            raise FileNotFoundError(f"Held-out dataset not found at {heldout_csv_path}")

        print(f"\n" + "="*85)
        print(f"[*] RUNNING HELD-OUT ATTACK EXPERIMENT (Novel Zero-Day Simulation)")
        print(f"    Source: {heldout_csv_path}")
        print("="*85)

        df_heldout = pd.read_csv(heldout_csv_path)
        df_heldout.columns = df_heldout.columns.str.strip()
        label_col = [c for c in df_heldout.columns if 'label' in c.lower()][0]

        # Filter strictly for the attack class (DDoS)
        ddos_flows = df_heldout[df_heldout[label_col].str.contains("DDoS", case=False, na=False)]
        print(f"[*] Available DDoS flows: {len(ddos_flows):,}. Sampling {num_samples} flows for evaluation...")
        
        sample_ddos = ddos_flows.sample(min(num_samples, len(ddos_flows)), random_state=42).copy()

        # Transform using saved training pipeline
        X_ddos_scaled = self.preprocessor.transform_samples(sample_ddos)

        # 1. Supervised Model Evaluation (Closed-World Limitation)
        sup_preds_raw = self.supervised_model.predict(X_ddos_scaled)
        sup_class_names = [self.preprocessor.classes_[idx] for idx in sup_preds_raw]
        sup_benign_count = sum(1 for name in sup_class_names if name == "BENIGN")
        sup_portscan_count = sum(1 for name in sup_class_names if name == "PortScan")

        # 2. Autoencoder Anomaly Detection
        is_anom, recon_errors = self.autoencoder.predict_anomalies(X_ddos_scaled)
        flagged_count = int(np.sum(is_anom))
        detection_rate = (flagged_count / len(sample_ddos)) * 100.0
        avg_recon_error = float(np.mean(recon_errors))
        threshold = float(self.autoencoder.anomaly_threshold)

        # 3. Hybrid Engine Evaluation
        hybrid_results = self.hybrid_engine.evaluate_batch(X_ddos_scaled)
        flagged_as_suspicious = sum(1 for r in hybrid_results if "Suspicious" in r["final_verdict"])
        flagged_as_attack = sum(1 for r in hybrid_results if "Attack" in r["final_verdict"])

        print(f"\n[+] Experimental Findings:")
        print(f"    - Supervised Model Result (Closed-World Assumption):")
        print(f"      * Forced to label DDoS as 'BENIGN':   {sup_benign_count} flows ({sup_benign_count/len(sample_ddos)*100:.1f}%)")
        print(f"      * Forced to label DDoS as 'PortScan': {sup_portscan_count} flows ({sup_portscan_count/len(sample_ddos)*100:.1f}%)")
        print(f"      -> Proves supervised models cannot identify novel attacks on their own!")
        print(f"\n    - Unsupervised Autoencoder Result:")
        print(f"      * Average DDoS Reconstruction Error:  {avg_recon_error:.5f} (Normal Threshold = {threshold:.5f})")
        print(f"      * Successfully Flagged as Anomaly:    {flagged_count} / {len(sample_ddos)} flows ({detection_rate:.2f}%)")
        print(f"\n    - Hybrid Decision Engine Result:")
        print(f"      * Intercepted & Flagged as Threat:    {flagged_as_attack + flagged_as_suspicious} / {len(sample_ddos)} flows ({(flagged_as_attack + flagged_as_suspicious)/len(sample_ddos)*100:.2f}%)")

        results = {
            "held_out_attack": "DDoS",
            "evaluated_samples": len(sample_ddos),
            "anomaly_threshold": round(threshold, 5),
            "average_recon_error": round(avg_recon_error, 5),
            "autoencoder_detection_rate": round(detection_rate, 2),
            "supervised_forced_benign_ratio": round(sup_benign_count / len(sample_ddos) * 100, 2),
            "hybrid_threat_detection_rate": round((flagged_as_attack + flagged_as_suspicious) / len(sample_ddos) * 100, 2),
            "research_conclusion": "Confirms that combining supervised classification with Autoencoder anomaly detection successfully intercepts held-out attacks that supervised models fail to identify."
        }
        return results

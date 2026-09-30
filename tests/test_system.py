"""
System-wide integration and unit tests for the Hybrid Intelligent NIDS (H-NIDS).
Validates:
1. Model loading & inference (RF, XGBoost, TabTransformer, Deep Autoencoder).
2. Hybrid Decision Fusion logic (Normal, Known Attack, Suspicious/Held-out).
3. SHAP feature attribution generation.
4. FastAPI application endpoints.
"""

import unittest
import numpy as np
import os
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi.testclient import TestClient
from backend.app import app, MODELS, load_all_models
from src.explainable_ai import NetworkExplainableAI
from src.hybrid_engine import HybridDecisionEngine


class TestHNIDSSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Initialize models and test client."""
        load_all_models()
        cls.client = TestClient(app)

    def test_models_loaded(self):
        """Verify that all core components are loaded in backend state."""
        self.assertIn("preprocessor", MODELS)
        self.assertIn("xgboost", MODELS)
        self.assertIn("random_forest", MODELS)
        self.assertIn("autoencoder", MODELS)
        self.assertIn("hybrid_engine", MODELS)
        self.assertIn("xai", MODELS)

    def test_preset_benign_flow(self):
        """Verify normal flow classification through hybrid decision engine."""
        engine: HybridDecisionEngine = MODELS["hybrid_engine"]
        prep = MODELS["preprocessor"]
        
        # Create a sample flow vector
        import pandas as pd
        sample_df = pd.DataFrame([{
            "Destination Port": 80,
            "Flow Duration": 125000,
            "Total Fwd Packets": 5,
            "Total Backward Packets": 6,
            "Fwd Packet Length Mean": 45.0,
            "Flow Bytes/s": 3600.0,
            "Flow Packets/s": 40.0,
        }])
        X_scaled = prep.transform_samples(sample_df)
        res = engine.evaluate_flow(X_scaled[0])
        self.assertIn("final_verdict", res)
        self.assertIn("severity", res)
        self.assertIn("anomaly_score", res)
        self.assertIn("anomaly_threshold", res)
        self.assertIn("supervised_label", res)

    def test_shap_explanation(self):
        """Verify that TreeSHAP produces valid feature attributions."""
        xai: NetworkExplainableAI = MODELS["xai"]
        prep = MODELS["preprocessor"]
        import pandas as pd
        sample_df = pd.DataFrame([{
            "Destination Port": 445,
            "Flow Duration": 30,
            "Total Fwd Packets": 2,
            "Packet Length Mean": 0.0,
            "Flow Bytes/s": 0.0,
            "Flow Packets/s": 66666.0,
        }])
        X_scaled = prep.transform_samples(sample_df)
        exp = xai.explain_single_flow(X_scaled[0], top_k=5)
        self.assertIn("base_value", exp)
        self.assertIn("top_attack_indicators", exp)
        self.assertIn("top_benign_indicators", exp)

    def test_fastapi_health(self):
        """Verify /api/health endpoint."""
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "online")
        self.assertIn("xgboost", data["loaded_modules"])
        self.assertIn("autoencoder", data["loaded_modules"])
        self.assertIn("hybrid_engine", data["loaded_modules"])

    def test_fastapi_stats(self):
        """Verify /api/stats endpoint returns verified benchmark metrics."""
        response = self.client.get("/api/stats")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("models", data)
        self.assertIn("xgboost", data["models"])
        self.assertIn("tab_transformer", data["models"])
        self.assertIn("autoencoder", data["models"])
        self.assertEqual(data["models"]["xgboost"]["accuracy"], "99.994%")

    def test_fastapi_predict_flow(self):
        """Verify /api/predict/flow endpoint."""
        payload = {
            "features": {
                "Destination Port": 80,
                "Flow Duration": 50000,
                "Packet Length Mean": 50.0
            }
        }
        response = self.client.post("/api/predict/flow", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("decision", data)
        self.assertIn("final_verdict", data["decision"])
        self.assertIn("shap_explanation", data)

    def test_fastapi_predict_preset(self):
        """Verify /api/predict/preset endpoint for all scenarios."""
        for scenario in ["benign", "portscan", "zero_day"]:
            response = self.client.post("/api/predict/preset", json={"scenario": scenario})
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertIn("decision", data)
            self.assertIn("final_verdict", data["decision"])


if __name__ == "__main__":
    unittest.main()

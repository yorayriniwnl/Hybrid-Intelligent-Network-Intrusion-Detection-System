import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "artifacts" / "results"


class TestEvidenceContract(unittest.TestCase):
    def load_json(self, name: str):
        return json.loads((RESULTS / name).read_text(encoding="utf-8"))

    def test_supervised_metrics_are_backed_by_artifacts(self):
        data = self.load_json("evaluation_metrics.json")
        self.assertEqual(data["dataset_metadata"]["test_samples"], 17248)

        rf = data["models"]["Random Forest"]
        xgb = data["models"]["XGBoost"]

        self.assertAlmostEqual(rf["accuracy"], 0.99988, places=5)
        self.assertAlmostEqual(rf["f1_score_macro"], 0.99988, places=5)
        self.assertAlmostEqual(xgb["accuracy"], 0.99994, places=5)
        self.assertAlmostEqual(xgb["f1_score_macro"], 0.99994, places=5)

    def test_heldout_result_is_explicitly_a_proxy_experiment(self):
        heldout = self.load_json("heldout_experiment_results.json")
        self.assertEqual(heldout["held_out_attack"], "DDoS")
        self.assertEqual(heldout["evaluated_samples"], 1000)
        self.assertEqual(heldout["supervised_forced_benign_ratio"], 42.3)
        self.assertEqual(heldout["hybrid_threat_detection_rate"], 71.1)

        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("Zero-Day Proxy", readme)
        self.assertIn("not evidence that every novel or real-world zero-day attack will be detected", readme)
        self.assertNotIn("Production-Ready Hybrid", readme)

    def test_readme_does_not_claim_unbacked_hybrid_accuracy(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        hybrid_rows = [
            line for line in readme.splitlines()
            if line.startswith("| **H-NIDS Hybrid Engine**")
        ]
        self.assertEqual(hybrid_rows, [])
        self.assertNotRegex(readme, re.compile(r"H-NIDS Hybrid Engine.*99\.994%", re.I))

    def test_readme_does_not_advertise_missing_license(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertFalse((ROOT / "LICENSE").exists())
        self.assertNotIn("License-MIT", readme)


if __name__ == "__main__":
    unittest.main()

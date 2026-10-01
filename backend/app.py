"""
FastAPI Backend Application for Hybrid Intelligent Network Intrusion Detection System (H-NIDS)
Phase 4: REST API, Live Dual-Stream Inference, CSV Batch Upload, and Dashboard Backend
Optimized for high-performance serverless deployment on Vercel (<60MB runtime footprint).
"""

import os
import sys
import io
import csv
import json
import time
import numpy as np
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel

# Ensure src is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.serverless_engine import PureNetworkPreprocessor, PureXGBoostClassifier
from src.autoencoder import NumpyAutoencoder
from src.hybrid_engine import HybridDecisionEngine

app = FastAPI(
    title="Hybrid Intelligent NIDS API",
    description="Research prototype for dual-stream intrusion classification and held-out anomaly experiments",
    version="2.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model state
MODELS: Dict[str, Any] = {}

def load_all_models():
    if MODELS:
        return
    print("[*] Initializing H-NIDS API: Loading models and preprocessor artifacts...")
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "artifacts", "models"))
    
    # 1. Preprocessor (Pure NumPy/JSON)
    prep_json = os.path.join(models_dir, "preprocessor.json")
    prep_joblib = os.path.join(models_dir, "preprocessor.joblib")
    if os.path.exists(prep_json):
        MODELS["preprocessor"] = PureNetworkPreprocessor(prep_json)
    elif os.path.exists(prep_joblib):
        try:
            from src.preprocessor import NetworkDataPreprocessor
            MODELS["preprocessor"] = NetworkDataPreprocessor.load(prep_joblib)
        except Exception as e:
            print("[!] Failed to load joblib preprocessor:", e)

    # 2. XGBoost Baseline (Pure GBDT / JSON)
    xgb_json = os.path.join(models_dir, "xgboost_model.json")
    xgb_joblib = os.path.join(models_dir, "xgboost_model.joblib")
    classes = MODELS["preprocessor"].classes_ if "preprocessor" in MODELS else ["BENIGN", "PortScan"]
    feat_names = MODELS["preprocessor"].feature_columns if "preprocessor" in MODELS else []
    if os.path.exists(xgb_json):
        MODELS["xgboost"] = PureXGBoostClassifier(xgb_json, classes, feature_names=feat_names)
    elif os.path.exists(xgb_joblib):
        try:
            import joblib
            MODELS["xgboost"] = joblib.load(xgb_joblib)
        except Exception as e:
            print("[!] Failed to load joblib xgboost:", e)

    # 3. Random Forest Baseline
    rf_joblib = os.path.join(models_dir, "random_forest_model.joblib")
    if os.path.exists(rf_joblib):
        try:
            import joblib
            MODELS["random_forest"] = joblib.load(rf_joblib)
        except Exception:
            pass

    # 4. Autoencoder (Pure NumPy Engine)
    ae_npz = os.path.join(models_dir, "autoencoder_weights.npz")
    if os.path.exists(ae_npz):
        MODELS["autoencoder"] = NumpyAutoencoder(ae_npz)

    # 5. Hybrid Decision Engine
    if "xgboost" in MODELS and "autoencoder" in MODELS and "preprocessor" in MODELS:
        MODELS["hybrid_engine"] = HybridDecisionEngine(
            supervised_model=MODELS["xgboost"],
            autoencoder=MODELS["autoencoder"],
            class_names=MODELS["preprocessor"].classes_
        )

    # 6. Explainable AI (XAI)
    if "xgboost" in MODELS:
        MODELS["xai"] = MODELS["xgboost"]

    print(f"[+] H-NIDS API Ready! Loaded components: {list(MODELS.keys())}")

def ensure_models_loaded():
    if not MODELS:
        load_all_models()

@app.on_event("startup")
def startup_event():
    load_all_models()

# Pydantic Schemas
class FlowInput(BaseModel):
    features: Dict[str, float]

class PresetFlowRequest(BaseModel):
    scenario: str  # "benign", "portscan", or "zero_day"

@app.get("/api/health")
def health_check():
    ensure_models_loaded()
    return {
        "status": "online",
        "system": "Hybrid Intelligent NIDS",
        "loaded_modules": list(MODELS.keys()),
        "classes": MODELS["preprocessor"].classes_ if "preprocessor" in MODELS else [],
        "anomaly_threshold": MODELS["autoencoder"].anomaly_threshold if "autoencoder" in MODELS else None
    }

@app.get("/api/stats")
def get_system_stats():
    ensure_models_loaded()

    results_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "artifacts", "results")
    )

    def load_json(name: str) -> Dict[str, Any]:
        path = os.path.join(results_dir, name)
        if not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    evaluation = load_json("evaluation_metrics.json")
    advanced = load_json("advanced_pipeline_results.json")
    heldout = load_json("heldout_experiment_results.json")

    models = evaluation.get("models", {})
    rf = models.get("Random Forest", {})
    xgb = models.get("XGBoost", {})
    tab = advanced.get("tab_transformer_evaluation", {})
    ae = advanced.get("autoencoder_stats", {})

    def pct(value: Any) -> Optional[str]:
        return f"{float(value) * 100:.3f}%" if isinstance(value, (int, float)) else None

    def supervised_summary(record: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "accuracy": pct(record.get("accuracy")),
            "precision": pct(record.get("precision_macro")),
            "recall": pct(record.get("recall_macro")),
            "f1_score": pct(record.get("f1_score_macro")),
            "false_positive_rate": pct(record.get("false_positive_rate")),
            "false_negative_rate": pct(record.get("false_negative_rate")),
            "latency_us": record.get("latency_microseconds_per_flow"),
            "throughput_flows_sec": record.get("throughput_flows_per_sec"),
            "test_samples": record.get("test_samples"),
        }

    dataset = evaluation.get("dataset_metadata", {})
    return {
        "evidence_status": "committed_experiment_artifacts",
        "scope_note": (
            "Supervised metrics describe held-out BENIGN/PortScan evaluation. "
            "The DDoS result is a separate held-out-category proxy experiment, "
            "not a guarantee of real-world zero-day detection."
        ),
        "total_flows_inspected": dataset.get("total_samples"),
        "clean_unique_flows": dataset.get("cleaned_samples"),
        "training_flows": dataset.get("train_samples"),
        "test_flows_evaluated": dataset.get("test_samples"),
        "models": {
            "xgboost": supervised_summary(xgb),
            "random_forest": supervised_summary(rf),
            "tab_transformer": supervised_summary(tab),
            "autoencoder": {
                "anomaly_threshold": ae.get("anomaly_threshold"),
                "threshold_percentile": ae.get("threshold_percentile"),
            },
        },
        "heldout_experiment": {
            "attack_category": heldout.get("held_out_attack"),
            "evaluated_samples": heldout.get("evaluated_samples"),
            "autoencoder_detection_rate_percent": heldout.get("autoencoder_detection_rate"),
            "supervised_forced_benign_percent": heldout.get("supervised_forced_benign_ratio"),
            "hybrid_threat_detection_rate_percent": heldout.get("hybrid_threat_detection_rate"),
        },
    }

@app.post("/api/predict/flow")
def predict_single_flow(payload: FlowInput):
    ensure_models_loaded()
    if "hybrid_engine" not in MODELS or "preprocessor" not in MODELS:
        raise HTTPException(status_code=503, detail="Models not initialized")

    preprocessor = MODELS["preprocessor"]
    hybrid_engine = MODELS["hybrid_engine"]
    xai = MODELS.get("xai")

    # Scaled feature vector
    X_scaled = preprocessor.transform_samples(payload.features)

    # Hybrid Decision
    decision = hybrid_engine.evaluate_flow(X_scaled[0])

    # SHAP Explanation
    shap_explanation = {}
    if xai and hasattr(xai, "explain_single_flow"):
        shap_explanation = xai.explain_single_flow(X_scaled[0], preprocessor.feature_columns, top_k=5)

    return {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "decision": decision,
        "shap_explanation": {
            "top_attack_indicators": shap_explanation.get("top_attack_indicators", []),
            "top_benign_indicators": shap_explanation.get("top_benign_indicators", []),
            "base_value": shap_explanation.get("base_value", 0.0)
        }
    }

@app.post("/api/predict/preset")
def predict_preset_scenario(req: PresetFlowRequest):
    """Generates realistic simulation flows for interactive UI demonstrations."""
    ensure_models_loaded()
    if req.scenario.lower() == "benign":
        sample = {
            "Destination Port": 443, "Flow Duration": 5696503, "Total Fwd Packets": 8, "Total Backward Packets": 7,
            "Total Length of Fwd Packets": 621.0, "Total Length of Bwd Packets": 4673.0, "Fwd Packet Length Mean": 77.6,
            "Flow Bytes/s": 929.0, "Flow Packets/s": 2.6, "PSH Flag Count": 1, "Init_Win_bytes_forward": 8192
        }
    elif req.scenario.lower() == "portscan":
        sample = {
            "Destination Port": 84, "Flow Duration": 44, "Total Fwd Packets": 1, "Total Backward Packets": 1,
            "Total Length of Fwd Packets": 0.0, "Total Length of Bwd Packets": 6.0, "Packet Length Mean": 2.0,
            "Fwd Packet Length Mean": 0.0, "Flow Bytes/s": 136363.6, "Flow Packets/s": 45454.5,
            "Bwd Packets/s": 22727.27, "Flow IAT Max": 44.0, "Flow IAT Min": 44.0, "PSH Flag Count": 1,
            "Init_Win_bytes_forward": 29200, "Init_Win_bytes_backward": 0
        }
    else:  # zero_day / novel DDoS
        sample = {
            "Destination Port": 80, "Flow Duration": 120000000, "Total Fwd Packets": 10000, "Total Backward Packets": 0,
            "Idle Min": 120000000, "Idle Max": 120000000, "Idle Mean": 120000000,
            "Flow Bytes/s": 5000000.0, "Flow Packets/s": 2000.0, "PSH Flag Count": 0
        }

    return predict_single_flow(FlowInput(features=sample))

@app.post("/api/predict/batch")
async def predict_batch_csv(file: UploadFile = File(...)):
    """Accepts uploaded network flow CSV file and runs full Hybrid batch detection."""
    ensure_models_loaded()
    if "hybrid_engine" not in MODELS or "preprocessor" not in MODELS:
        raise HTTPException(status_code=503, detail="Models not initialized")

    preprocessor = MODELS["preprocessor"]
    hybrid_engine = MODELS["hybrid_engine"]

    contents = await file.read()
    try:
        text_stream = io.StringIO(contents.decode("utf-8", errors="ignore"))
        reader = csv.DictReader(text_stream)
        rows: List[Dict[str, Any]] = []
        for i, row in enumerate(reader):
            if i >= 1000:
                break
            clean_row = {}
            for k, v in row.items():
                if k:
                    k_clean = k.strip()
                    try:
                        clean_row[k_clean] = float(v)
                    except (ValueError, TypeError):
                        clean_row[k_clean] = v
            rows.append(clean_row)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV file: {str(e)}")

    if not rows:
        raise HTTPException(status_code=400, detail="CSV file contains no valid data rows.")

    X_scaled = preprocessor.transform_samples(rows)
    decisions = hybrid_engine.evaluate_batch(X_scaled)

    normal_count = sum(1 for d in decisions if d["final_verdict"] == "Normal")
    known_attack_count = sum(1 for d in decisions if d["final_verdict"] == "Known Attack")
    suspicious_count = sum(1 for d in decisions if "Suspicious" in d["final_verdict"])

    return {
        "filename": file.filename,
        "total_flows_processed": len(rows),
        "summary": {
            "normal_flows": normal_count,
            "known_attacks": known_attack_count,
            "suspicious_zero_day_flows": suspicious_count,
            "threat_ratio_percent": round((known_attack_count + suspicious_count) / len(rows) * 100, 2)
        },
        "sample_flagged_alerts": [d for d in decisions if d["final_verdict"] != "Normal"][:10]
    }

# Mount static frontend
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
public_dir = os.path.join(base_dir, "public")
static_dir = os.path.join(os.path.dirname(__file__), "static")

if os.path.exists(public_dir):
    app.mount("/public", StaticFiles(directory=public_dir), name="public")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/favicon.ico", include_in_schema=False)
def get_favicon():
    svg_icon = """<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>🛡️</text></svg>"""
    return Response(content=svg_icon, media_type="image/svg+xml")

@app.get("/", response_class=HTMLResponse)
def serve_dashboard():
    for fpath in [os.path.join(public_dir, "index.html"), os.path.join(static_dir, "index.html")]:
        if os.path.exists(fpath):
            with open(fpath, "r", encoding="utf-8") as f:
                return f.read()
    return "<h1>H-NIDS API is Running.</h1>"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app:app", host="127.0.0.1", port=8000, reload=False)
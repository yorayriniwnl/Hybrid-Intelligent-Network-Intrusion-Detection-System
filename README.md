<div align="center">

![Hybrid Intelligent NIDS Banner](artifacts/figures/banner.png)

# 🛡️ Hybrid Intelligent Network Intrusion Detection System (H-NIDS)
> **7th Semester Major Project — End-to-End Research Prototype**  
> *A hybrid machine-learning and deep-learning NIDS prototype for known-attack classification and held-out attack anomaly experiments, with explainability and an interactive SOC dashboard.*

[![Python Version](https://img.shields.io/badge/Python-3.11%20%7C%203.12-blue?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![CI](https://github.com/yorayriniwnl/Hybrid-Intelligent-Network-Intrusion-Detection-System/actions/workflows/ci.yml/badge.svg)](https://github.com/yorayriniwnl/Hybrid-Intelligent-Network-Intrusion-Detection-System/actions/workflows/ci.yml)

</div>

---

## 📌 Executive Summary & Academic Objectives

Traditional Network Intrusion Detection Systems (NIDS) face a fundamental flaw known as the **Closed-World Assumption**: supervised classifiers (Random Forest, XGBoost, TabTransformer) can accurately detect known attack signatures, but they catastrophically misclassify novel, held-out, or zero-day attacks as normal traffic or coerce them into erroneous categories.

**H-NIDS** resolves this limitation by introducing a **Dual-Stream Hybrid Fusion Architecture**:
1. **Supervised Stream (XGBoost & TabTransformer):** Contextual multi-head self-attention and gradient boosting for microsecond classification of known network attack vectors.
2. **Unsupervised Anomaly Stream (Deep Autoencoder):** Symmetric bottleneck neural network trained exclusively on benign traffic baselines; detects deviations via statistical reconstruction error ($L_2$ error thresholding).
3. **Hybrid Decision Fusion Engine:** Reconciles supervised probabilities and autoencoder anomaly scores to output 3 unambiguous operational states: **`Normal`**, **`Known Attack`**, or **`Suspicious / Held-Out Zero-Day Attack`**.
4. **Explainable AI (SHAP):** Game-theoretic feature attributions pinpointing exact flow telemetry metrics responsible for intrusion flags.
5. **Real-Time SOC Dashboard & REST API:** High-throughput FastAPI backend serving a Cyberpunk-styled dark/red SOC monitoring console with live telemetry, scenario simulations, and drag-and-drop CSV batch scanning.

---

## 🏗️ System Architecture

```
                                  INCOMING NETWORK FLOW
                                           │
                                           ▼
                   ┌────────────────────────────────────────────────┐
                   │       Strict Featurization & Preprocessing     │
                   │   (Cleaning, Infinity Handling, Robust Scaling) │
                   └───────────────────────┬────────────────────────┘
                                           │
                    ┌──────────────────────┴──────────────────────┐
                    │                                             │
                    ▼                                             ▼
     ┌─────────────────────────────┐               ┌─────────────────────────────┐
     │   SUPERVISED INFERENCE      │               │   UNSUPERVISED ANOMALY      │
     │   • XGBoost (0.4 µs)        │               │   • Deep Autoencoder        │
     │   • TabTransformer (Attn)   │               │   • Statistical Threshold   │
     │   Output: P(Known Attack)   │               │   Output: MSE Recon Error   │
     └──────────────┬──────────────┘               └──────────────┬──────────────┘
                    │                                             │
                    └──────────────────────┬──────────────────────┘
                                           │
                                           ▼
                   ┌────────────────────────────────────────────────┐
                   │          HYBRID DECISION FUSION ENGINE         │
                   │                                                │
                   │   • If P(Attack) ≥ 85%     ──► Known Attack    │
                   │   • If P(Benign) & Low MSE ──► Normal Traffic  │
                   │   • If High MSE Recon      ──► Held-Out Threat │
                   └───────────────────────┬────────────────────────┘
                                           │
                    ┌──────────────────────┴──────────────────────┐
                    ▼                                             ▼
     ┌─────────────────────────────┐               ┌─────────────────────────────┐
     │      EXPLAINABLE AI         │               │     REST API & DASHBOARD    │
     │   • TreeSHAP Attributions   │               │   • FastAPI Async Backend   │
     │   • Local & Global Factors  │               │   • Cyberpunk Red Theme SOC │
     └─────────────────────────────┘               └─────────────────────────────┘
```

---

## 📊 Comprehensive Experimental Benchmark Results

Committed evaluation artifacts report the following results on **CIC-IDS2017**. The supervised metrics come from held-out BENIGN/PortScan splits; the DDoS experiment is a separate held-out-category simulation and must not be interpreted as proof of general real-world zero-day detection.

| Architecture | Evaluation scope | Accuracy | Macro F1 | FPR | FNR | Measured latency | Measured throughput |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | BENIGN vs PortScan holdout (17,248 flows) | **99.988%** | **99.988%** | 0.0000% | 0.0230% | 2.51 µs / flow | 398,917 flows/s |
| **XGBoost** | BENIGN vs PortScan holdout (17,248 flows) | **99.994%** | **99.994%** | **0.0000%** | **0.0120%** | **0.41 µs / flow** | **2,412,004 flows/s** |
| **TabTransformer** | BENIGN vs PortScan holdout (10,833 flows) | **99.945%** | **99.945%** | 0.0190% | 0.0900% | 177.53 µs / flow | 5,633 flows/s |
| **Deep Autoencoder + Hybrid** | Separate 1,000-flow held-out DDoS simulation | N/A | N/A | N/A | N/A | Not reported in committed held-out artifact | **71.1% hybrid threat-detection rate** |

The repository does **not** currently contain a directly comparable aggregate accuracy/F1/latency benchmark for the fused hybrid engine. Those values are therefore not claimed here. Source artifacts: `artifacts/results/evaluation_metrics.json`, `advanced_pipeline_results.json`, and `heldout_experiment_results.json`.

---

## 🧪 Held-Out Attack Experiment (Zero-Day Proxy)

To probe the **closed-world assumption**, the project conducts a held-out-category experiment. DDoS is excluded from supervised training and then evaluated as an unseen attack category. This is a useful proxy experiment, not evidence that every novel or real-world zero-day attack will be detected.
- The supervised models were trained **only** on `BENIGN` and `PortScan` flows.
- A completely held-out, unseen attack category (**`DDoS`**) was passed into both systems without retraining.

### Experimental Outcome:
1. **Closed-World Supervised Failure:**
   - XGBoost and Random Forest completely failed to detect the novel nature of the attack:
   - **42.3%** of DDoS attack flows were misclassified as **`BENIGN`** (silent security breach).
   - **57.7%** were forcibly misclassified as `PortScan`.
2. **H-NIDS Hybrid Fusion Success:**
   - The Deep Autoencoder flagged an average reconstruction error spike of **$4.09 \times 10^{13}$**, surpassing the 98.5th percentile statistical threshold.
   - The Hybrid Decision Engine flagged **71.1% of the 1,000 held-out DDoS sample flows** as a threat in the committed experiment.

---

## 🔎 Evidence Boundary

- The headline supervised metrics are read from committed JSON result artifacts, not recomputed on every web request.
- The held-out DDoS result is a **simulation on one excluded CIC-IDS2017 category**. It is not a guarantee of real-world zero-day detection.
- Latency and throughput are machine/run dependent and should be interpreted only in the context of the recorded experiment.
- The repository is a research/academic prototype. Production SOC deployment, continuous packet capture, adversarial robustness, calibration across networks, and operational incident-response validation are outside the currently demonstrated scope.
- No software license is asserted in this README until a root license file is deliberately chosen and added.

---

## 🔬 Visualizations & Explainable AI (SHAP)

### 1. Global Explainable AI Feature Attribution
Exact Tree SHAP game-theoretic Shapley values calculated across unseen flows reveal the most critical network traffic signatures driving intrusion verdicts:

![SHAP Summary Plot](artifacts/figures/shap_summary.png)

- **Total Length of Fwd Packets & Flow Bytes/s:** Overwhelmingly separate volumetric attack traffic from legitimate HTTP/S sessions.
- **PSH / SYN Flag Counts:** Instantly signal reconnaissance port scanning and SYN flood attempts.
- **Flow IAT (Inter-Arrival Time) & Packet Length Mean:** Differentiate automated script probes from human browsing dynamics.

### 2. Supervised Baseline Performance & Confusion Matrices
| Model Comparison | Class Distribution |
| :---: | :---: |
| ![Model Comparison](artifacts/figures/model_comparison.png) | ![Class Distribution](artifacts/figures/class_distribution.png) |

| Random Forest Confusion Matrix | XGBoost Confusion Matrix |
| :---: | :---: |
| ![RF Confusion Matrix](artifacts/figures/confusion_matrix_random_forest.png) | ![XGBoost Confusion Matrix](artifacts/figures/confusion_matrix_xgboost.png) |

---

## 🖥️ Cyberpunk SOC Monitoring Dashboard (Phase 4)

The project includes an interactive, high-contrast **Cyberpunk Dark/Red Theme** web console:

- **Telemetry & KPI HUD:** Displays metrics from committed experiment artifacts and the active anomaly threshold.
- **One-Click Attack Simulator:** Test pre-configured simulation scenarios (`Normal HTTPS Browsing`, `Aggressive PortScan Probe`, `Novel DDoS Volumetric Flood`) with real-time verdicts.
- **Per-flow attribution visualizer:** Uses the deployed XGBoost contribution engine for individual-flow feature attributions.
- **Batch CSV Drag & Drop Upload:** Drop any raw CIC-IDS2017 or firewall capture CSV file to run automated batch triage and download threat intelligence summaries.

---

## 🚀 Quickstart & Setup Guide

### 1. Clone & Set Up Local Environment
```bash
git clone https://github.com/yorayriniwnl/Hybrid-Intelligent-Network-Intrusion-Detection-System.git
cd Hybrid-Intelligent-Network-Intrusion-Detection-System

# Create virtual environment
python -m venv .venv
.\.venv\Scripts\activate

# Install dependencies (CPU PyTorch + Scikit-Learn + XGBoost + FastAPI)
pip install -r requirements.txt
```

### 2. Run Integration Test Suite
```bash
python -m unittest tests/test_system.py
```
> The integration suite is also enforced by GitHub Actions. Test counts and runtime may evolve with the project.

### 3. Launch the Cyberpunk Dashboard & REST API
```bash
uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload
```
Open **[http://localhost:8000](http://localhost:8000)** in your browser.

---

## 🐳 Docker Deployment

Deploy the entire system with zero host dependencies using Docker or Docker Compose:

```bash
# Build and run using Docker Compose
docker compose up --build -d

# Or build standalone Docker container
docker build -t hnids-system .
docker run -p 8000:8000 hnids-system
```
Access the web dashboard at `http://localhost:8000` or inspect API docs at `http://localhost:8000/docs`.

---

## ☁️ Vercel Serverless Deployment

The repository includes a lightweight Vercel deployment path for the dashboard and NumPy/JSON inference runtime. Treat it as a demonstration deployment; the full local training stack is not executed on Vercel.

### Method A: One-Click Git Deployment (Recommended)
1. Push this repository to GitHub:
   ```bash
   git push origin main
   ```
2. Go to **[vercel.com/new](https://vercel.com/new)** and sign in.
3. Import the `Hybrid-Intelligent-Network-Intrusion-Detection-System` repository.
4. Leave build settings as default (Framework Preset: **Other**) and click **Deploy**.
5. Vercel automatically:
   - Deploys static dashboard assets in `/public` to the global edge CDN.
   - Deploys `/api/index.py` as a Python serverless function; latency depends on platform/runtime conditions.
   - Ingests lightweight serialized models (`autoencoder_weights.npz`, `xgboost_model.joblib`, `preprocessor.joblib`).

### Method B: Vercel CLI Deployment
```bash
# 1. Login to Vercel (first time only)
npx vercel login

# 2. Deploy directly to production
npx vercel --prod
```

---

## 📁 Repository Directory Structure

```
Hybrid-Intelligent-Network-Intrusion-Detection-System/
├── api/
│   └── index.py                  # Vercel Serverless Function entry point
├── public/
│   └── index.html                # Edge CDN-optimized interactive SOC dashboard
├── artifacts/
│   ├── figures/                  # Publication figures (Confusion Matrices, SHAP, etc.)
│   ├── models/                   # Serialized ML & PyTorch checkpoints (.joblib, .pt, .npz)
│   └── results/                  # Experimental JSON metrics & benchmarks
├── backend/
│   ├── app.py                    # High-throughput FastAPI REST API
│   └── static/
│       └── index.html            # Cyberpunk dark/red interactive dashboard
├── src/
│   ├── data_loader.py            # Stream-based chunking & automated data ingestion
│   ├── preprocessor.py           # Robust scaler, inf/nan handling, zero-leakage fit
│   ├── models.py                 # Random Forest & XGBoost model wrappers
│   ├── evaluate.py               # Precision, Recall, F1, Latency, Confusion Matrix
│   ├── tab_transformer.py        # PyTorch multi-head self-attention tabular model
│   ├── autoencoder.py            # Deep autoencoder & pure NumPy serverless engine
│   ├── hybrid_engine.py          # Dual-stream decision fusion engine
│   ├── heldout_experiment.py     # Zero-day held-out attack simulation suite
│   └── explainable_ai.py         # Exact Tree SHAP attribution engine
├── tests/
│   └── test_system.py            # End-to-end unit & integration test suite
├── faculty_evaluation_prototype.ipynb  # Interactive Jupyter walkthrough
├── run_pipeline.py               # Phase 1 supervised pipeline runner
├── run_advanced_pipeline.py      # Phases 2 & 3 deep learning & hybrid runner
├── Dockerfile                    # Multi-stage production container configuration
├── docker-compose.yml            # Container orchestration manifest
├── vercel.json                   # Vercel routing and serverless function packaging
├── .vercelignore                 # Excludes raw datasets & dev packages from Vercel upload
├── requirements.txt              # Lean production serverless dependencies
├── requirements-dev.txt          # Full local deep learning training dependencies
└── README.md                     # Comprehensive technical documentation
```

---

## 🗣️ Faculty Evaluation Presentation Script

When presenting to the evaluation committee:

> *"Good morning/afternoon, professors. In this project, we address the critical challenge of the Closed-World Assumption in Network Intrusion Detection Systems.*
>
> *In our committed BENIGN-versus-PortScan holdout, Random Forest and XGBoost exceed 99.99% supervised accuracy. In a separate proxy experiment where DDoS is withheld from supervised training, 42.3% of those held-out DDoS flows are classified as BENIGN. This illustrates the closed-world limitation in this experiment; it does not establish performance on arbitrary real-world zero-day attacks.*
>
> *Our proposed Dual-Stream Hybrid Fusion Framework combines supervised TabTransformer/XGBoost outputs with a Deep Autoencoder trained on benign traffic. In the committed 1,000-flow held-out DDoS proxy experiment, the hybrid decision engine flags 71.1% of those flows as threats without retraining. That result is specific to this dataset and held-out category, not a claim of general zero-day detection.*
>
> *Finally, we use Tree SHAP for feature attribution and expose the experimental pipeline through a responsive SOC-style dashboard and REST API. Reported latency and throughput values belong to the recorded benchmark runs and should not be generalized to every deployment environment."*

---

## 📜 Academic Integrity & Citation

This project is submitted in partial fulfillment of the requirements for the **Bachelor of Technology (B.Tech) Degree in Computer Science and Engineering**. 

Benchmark Dataset: *Iman Sharafaldin, Arash Habibi Lashkari, and Ali A. Ghorbani, "Toward Generating a New Dataset for Intrusion Detection: A Realistic Protocol and Attack Analysis", Canadian Institute for Cybersecurity (CIC), 2018.*
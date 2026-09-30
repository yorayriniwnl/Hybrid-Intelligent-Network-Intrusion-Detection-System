import json

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Hybrid Intelligent Network Intrusion Detection System (H-NIDS)\n",
    "## Phase 1 Working ML Prototype — Faculty Evaluation Benchmark\n",
    "\n",
    "> **Academic Stage:** 7th Semester Major Project  \n",
    "> **Objective:** Demonstrate working supervised machine learning baselines (Random Forest & XGBoost) on authentic network intrusion traffic (CIC-IDS2017) with full evaluation metrics, confusion matrices, and unseen flow predictions before advancing to Phase 2 (TabTransformer + Autoencoder + Hybrid Decision Engine)."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Step 1: Ingest and Inspect the Dataset\n",
    "We load the Canadian Institute for Cybersecurity benchmark (`CIC-IDS2017`), examine the flow attributes, and inspect nulls, infinite values, and label distribution."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import sys\n",
    "# Ensure project root is in sys.path for robust module imports\n",
    "sys.path.insert(0, os.path.abspath('.'))\n",
    "\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "from src.data_loader import NetworkDataLoader\n",
    "\n",
    "dataset_path = 'data/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv'\n",
    "loader = NetworkDataLoader(dataset_path)\n",
    "df_raw = loader.load_data(sample_size=100000, random_state=42)\n",
    "inspection = loader.inspect_dataset()\n",
    "\n",
    "print(f\"Total Flows Loaded: {len(df_raw):,}\")\n",
    "print(f\"Total Features: {inspection['total_features']}\")\n",
    "print(f\"Class Distribution: {inspection['class_distribution']}\")\n",
    "print(f\"Infinite Values: {inspection['total_infinite_values']:,}\")\n",
    "print(f\"Missing Values: {inspection['total_null_values']:,}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Step 2: Data Cleaning & Preprocessing (Strict Featurization Ordering)\n",
    "1. Strip column whitespace.\n",
    "2. Remove duplicate flows.\n",
    "3. Replace division-by-zero infinities (`Flow Bytes/s`) with training medians.\n",
    "4. Eliminate invariant zero-variance features.\n",
    "5. **Stratified 80/20 Train/Test split BEFORE parameter fitting** to guarantee zero data leakage.\n",
    "6. Apply `RobustScaler` on training split."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "from src.preprocessor import NetworkDataPreprocessor\n",
    "from src.visualize import NetworkVisualizer\n",
    "\n",
    "preprocessor = NetworkDataPreprocessor(target_col=inspection['label_column'], scaler_type='robust')\n",
    "df_clean, cleaning_stats = preprocessor.clean_raw_dataframe(df_raw)\n",
    "\n",
    "# Visualize class distribution\n",
    "visualizer = NetworkVisualizer('artifacts/figures')\n",
    "clean_dist = {\n",
    "    k: {'count': int(v), 'percentage': round(v / len(df_clean) * 100, 2)}\n",
    "    for k, v in df_clean[inspection['label_column']].value_counts().items()\n",
    "}\n",
    "visualizer.plot_class_distribution(clean_dist)\n",
    "\n",
    "# Leakage-free train/test split\n",
    "X_train, X_test, y_train, y_test, meta = preprocessor.prepare_train_test_split(df_clean, test_size=0.20, random_state=42)\n",
    "print(f\"Training Set: {len(X_train):,} flows | Test Set: {len(X_test):,} flows\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Step 3: Train Random Forest and XGBoost Models\n",
    "- **Random Forest**: 100 estimators, max depth 20, class weighting.\n",
    "- **XGBoost**: Histogram-based gradient boosted trees, max depth 6, learning rate 0.1."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "from src.models import MLModelTrainer\n",
    "\n",
    "trainer = MLModelTrainer(random_state=42)\n",
    "rf_model, rf_time = trainer.train_random_forest(X_train, y_train, n_estimators=100, max_depth=20)\n",
    "xgb_model, xgb_time = trainer.train_xgboost(X_train, y_train, n_estimators=100, max_depth=6, learning_rate=0.1)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Step 4: Model Evaluation on Holdout Test Set (17,248 Unseen Flows)\n",
    "Calculate actual experimental metrics: Accuracy, Macro/Weighted Precision, Recall, F1-Score, False Positive Rate (FPR), False Negative Rate (FNR), Confusion Matrix, and Inference Latency."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "from src.evaluate import ModelEvaluator\n",
    "\n",
    "evaluator = ModelEvaluator(class_names=preprocessor.classes_)\n",
    "rf_eval = evaluator.evaluate_model('Random Forest', rf_model, X_test, y_test, rf_time)\n",
    "xgb_eval = evaluator.evaluate_model('XGBoost', xgb_model, X_test, y_test, xgb_time)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Step 5: Confusion Matrices & Comparative Visualizations"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "\n",
    "visualizer.plot_confusion_matrix(np.array(rf_eval['confusion_matrix']), preprocessor.classes_, 'Random Forest')\n",
    "visualizer.plot_confusion_matrix(np.array(xgb_eval['confusion_matrix']), preprocessor.classes_, 'XGBoost')\n",
    "visualizer.plot_model_comparison(rf_eval, xgb_eval)\n",
    "visualizer.plot_feature_importances(preprocessor.feature_columns, rf_model, xgb_model, top_k=12)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Step 6: Live Inference on Unseen Test Network Flows\n",
    "Pass individual held-out test flows through both models to demonstrate live prediction accuracy and confidence scores."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "from src.sample_inference import UnseenFlowPredictor\n",
    "\n",
    "predictor = UnseenFlowPredictor(preprocessor, rf_model, xgb_model, preprocessor.classes_)\n",
    "raw_test_df = df_clean.iloc[-len(y_test):].reset_index(drop=True)\n",
    "unseen_preds = predictor.run_sample_predictions(X_test, y_test, raw_test_df, num_samples_per_class=4)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Step 7: Random Forest vs. XGBoost Benchmark Summary Table"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "comp_df = pd.DataFrame([\n",
    "    {'Metric': 'Accuracy', 'Random Forest': f\"{rf_eval['accuracy']*100:.3f}%\", 'XGBoost': f\"{xgb_eval['accuracy']*100:.3f}%\"},\n",
    "    {'Metric': 'Precision (Macro)', 'Random Forest': f\"{rf_eval['precision_macro']*100:.3f}%\", 'XGBoost': f\"{xgb_eval['precision_macro']*100:.3f}%\"},\n",
    "    {'Metric': 'Recall (Macro)', 'Random Forest': f\"{rf_eval['recall_macro']*100:.3f}%\", 'XGBoost': f\"{xgb_eval['recall_macro']*100:.3f}%\"},\n",
    "    {'Metric': 'F1-Score (Macro)', 'Random Forest': f\"{rf_eval['f1_score_macro']*100:.3f}%\", 'XGBoost': f\"{xgb_eval['f1_score_macro']*100:.3f}%\"},\n",
    "    {'Metric': 'False Positive Rate (FPR)', 'Random Forest': f\"{rf_eval['false_positive_rate']*100:.4f}%\", 'XGBoost': f\"{xgb_eval['false_positive_rate']*100:.4f}%\"},\n",
    "    {'Metric': 'False Negative Rate (FNR)', 'Random Forest': f\"{rf_eval['false_negative_rate']*100:.4f}%\", 'XGBoost': f\"{xgb_eval['false_negative_rate']*100:.4f}%\"},\n",
    "    {'Metric': 'Training Time', 'Random Forest': f\"{rf_eval['training_time_seconds']:.2f} s\", 'XGBoost': f\"{xgb_eval['training_time_seconds']:.2f} s\"},\n",
    "    {'Metric': 'Inference Latency per Flow', 'Random Forest': f\"{rf_eval['latency_microseconds_per_flow']:.2f} µs\", 'XGBoost': f\"{xgb_eval['latency_microseconds_per_flow']:.2f} µs\"},\n",
    "    {'Metric': 'Throughput (Flows/sec)', 'Random Forest': f\"{rf_eval['throughput_flows_per_sec']:,.0f}\", 'XGBoost': f\"{xgb_eval['throughput_flows_per_sec']:,.0f}\"}\n",
    "])\n",
    "display(comp_df)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Step 8: Architecture Roadmap & Faculty Viva Statement\n",
    "\n",
    "> **Suggested Explanation to Faculty:**  \n",
    "> *“In the first phase, we implemented supervised machine learning for network intrusion detection using Random Forest and XGBoost. We are evaluating the models using precision, recall, F1-score and confusion matrices. The proposed next phase adds TabTransformer for attention-based classification and an Autoencoder for anomaly detection of held-out or suspicious attack behavior, followed by hybrid decision fusion and Explainable AI.”*\n",
    "\n",
    "**Why Phase 2 is Required:**\n",
    "1. **Supervised Limitations:** Supervised models assume a closed-world distribution; novel zero-day attacks are forced into known categories.\n",
    "2. **TabTransformer:** Embeds tabular flow features through multi-head self-attention to capture complex inter-feature correlations.\n",
    "3. **Deep Autoencoder:** Trained strictly on benign network traffic to learn baseline normal behavior; novel attacks produce high reconstruction error.\n",
    "4. **Hybrid Decision Fusion:** Combines supervised attack classification probability with unsupervised anomaly score to classify flows as `Normal`, `Known Attack`, or `Suspicious/Held-Out`.\n",
    "5. **Explainable AI (SHAP):** Generates flow-level attribution values for security analysts."
   ]
  }
 ],
 "metadata": {
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}

with open("faculty_evaluation_prototype.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)

print("Created faculty_evaluation_prototype.ipynb successfully.")

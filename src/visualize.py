"""
Visualization Module for Hybrid Intelligent Network Intrusion Detection System
Plots Attack Distribution, Confusion Matrices, and Model Comparison
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless server/cli execution
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List

def _display_if_notebook(filepath: str):
    """Safely renders saved figure inline if running in an interactive notebook."""
    try:
        from IPython.display import Image, display
        from IPython import get_ipython
        if get_ipython() is not None:
            display(Image(filepath))
    except Exception:
        pass

class NetworkVisualizer:
    """
    Generates presentation-ready publication-quality figures
    for academic faculty evaluation.
    """

    def __init__(self, output_dir: str = "artifacts/figures"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        sns.set_theme(style="whitegrid", palette="muted")
        plt.rcParams.update({'font.sans-serif': 'Arial', 'font.size': 11})

    def plot_class_distribution(self, class_distribution: Dict[str, Dict[str, Any]]) -> str:
        """
        Visualizes the class/attack distribution of the network flow dataset.
        """
        labels = list(class_distribution.keys())
        counts = [class_distribution[lbl]["count"] for lbl in labels]
        percentages = [class_distribution[lbl]["percentage"] for lbl in labels]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

        # Bar plot
        bars = ax1.bar(labels, counts, color=['#2b5c8f', '#d95f02'][:len(labels)], width=0.5, edgecolor='black', linewidth=1.2)
        ax1.set_title("Network Flow Class Distribution (Counts)", fontsize=13, fontweight='bold', pad=12)
        ax1.set_xlabel("Traffic Category", fontsize=11, fontweight='bold')
        ax1.set_ylabel("Total Number of Flows", fontsize=11, fontweight='bold')
        ax1.grid(axis='y', linestyle='--', alpha=0.7)

        for bar, count, pct in zip(bars, counts, percentages):
            yval = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2.0, yval + (max(counts) * 0.015), f"{count:,}\n({pct}%)",
                     ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax1.set_ylim(0, max(counts) * 1.15)

        # Pie chart
        colors = ['#4a7bb0', '#e67e22', '#2ecc71', '#9b59b6'][:len(labels)]
        ax2.pie(counts, labels=labels, autopct='%1.2f%%', startangle=140, colors=colors,
                wedgeprops={'edgecolor': 'black', 'linewidth': 1.2},
                textprops={'fontsize': 11, 'fontweight': 'bold'})
        ax2.set_title("Class Composition Ratio", fontsize=13, fontweight='bold', pad=12)

        plt.suptitle("Dataset Traffic & Attack Profile (CIC-IDS2017)", fontsize=15, fontweight='bold', y=1.02)
        plt.tight_layout()

        filepath = os.path.join(self.output_dir, "class_distribution.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[*] Saved class distribution plot to: {filepath}")
        _display_if_notebook(filepath)
        return filepath

    def plot_confusion_matrix(self, cm: np.ndarray, class_names: List[str], model_name: str) -> str:
        """
        Visualizes the confusion matrix with counts and percentages.
        """
        fig, ax = plt.subplots(figsize=(7, 6))

        # Normalize confusion matrix for percentages
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        # Combine annotations
        annot = np.empty_like(cm, dtype=object)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annot[i, j] = f"{cm[i, j]:,}\n({cm_normalized[i, j]*100:.2f}%)"

        sns.heatmap(
            cm,
            annot=annot,
            fmt='',
            cmap="Blues" if "Random Forest" in model_name else "YlGnBu",
            xticklabels=class_names,
            yticklabels=class_names,
            cbar=True,
            ax=ax,
            linewidths=1.5,
            linecolor='gray',
            annot_kws={"size": 11, "weight": "bold"}
        )

        ax.set_title(f"Confusion Matrix — {model_name}\n(Holdout Test Set)", fontsize=13, fontweight='bold', pad=12)
        ax.set_xlabel("Predicted Class", fontsize=11, fontweight='bold', labelpad=8)
        ax.set_ylabel("True Class", fontsize=11, fontweight='bold', labelpad=8)

        plt.tight_layout()
        filename = f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[*] Saved confusion matrix plot to: {filepath}")
        _display_if_notebook(filepath)
        return filepath

    def plot_model_comparison(self, rf_metrics: Dict[str, Any], xgb_metrics: Dict[str, Any]) -> str:
        """
        Plots side-by-side bar chart comparing Random Forest vs XGBoost across key metrics.
        """
        metric_keys = [
            ("Accuracy", "accuracy"),
            ("Precision (Macro)", "precision_macro"),
            ("Recall (Macro)", "recall_macro"),
            ("F1-Score (Macro)", "f1_score_macro")
        ]

        labels = [m[0] for m in metric_keys]
        rf_vals = [rf_metrics[m[1]] * 100 for m in metric_keys]
        xgb_vals = [xgb_metrics[m[1]] * 100 for m in metric_keys]

        x = np.arange(len(labels))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 6))

        rects1 = ax.bar(x - width/2, rf_vals, width, label='Random Forest', color='#2b5c8f', edgecolor='black')
        rects2 = ax.bar(x + width/2, xgb_vals, width, label='XGBoost', color='#2ca02c', edgecolor='black')

        ax.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
        ax.set_title('Supervised Baseline Comparison: Random Forest vs. XGBoost', fontsize=14, fontweight='bold', pad=14)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=11, fontweight='bold')
        ax.legend(frameon=True, fontsize=11)
        ax.set_ylim(90, 101)  # Focus on top percentage range
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        # Value labels on top of bars
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.2f}%',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 4),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=9.5, fontweight='bold')

        autolabel(rects1)
        autolabel(rects2)

        plt.tight_layout()
        filepath = os.path.join(self.output_dir, "model_comparison.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[*] Saved model comparison plot to: {filepath}")
        _display_if_notebook(filepath)
        return filepath

    def plot_feature_importances(
        self,
        feature_names: List[str],
        rf_model: Any,
        xgb_model: Any,
        top_k: int = 12
    ) -> str:
        """
        Plots the top K most decisive network flow features learned by both models.
        """
        rf_importances = rf_model.feature_importances_
        xgb_importances = xgb_model.feature_importances_

        rf_idx = np.argsort(rf_importances)[::-1][:top_k]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Random Forest Top Features
        ax1.barh(np.array(feature_names)[rf_idx][::-1], rf_importances[rf_idx][::-1], color='#2b5c8f', edgecolor='black')
        ax1.set_title(f"Top {top_k} Features — Random Forest", fontsize=12, fontweight='bold')
        ax1.set_xlabel("Gini Importance", fontsize=11)

        # XGBoost Top Features
        xgb_idx = np.argsort(xgb_importances)[::-1][:top_k]
        ax2.barh(np.array(feature_names)[xgb_idx][::-1], xgb_importances[xgb_idx][::-1], color='#2ca02c', edgecolor='black')
        ax2.set_title(f"Top {top_k} Features — XGBoost", fontsize=12, fontweight='bold')
        ax2.set_xlabel("Gain / Weight Importance", fontsize=11)

        plt.suptitle("Cybersecurity Feature Significance (Network Flow Attribution)", fontsize=14, fontweight='bold', y=1.01)
        plt.tight_layout()

        filepath = os.path.join(self.output_dir, "feature_importance.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[*] Saved feature importance plot to: {filepath}")
        _display_if_notebook(filepath)
        return filepath

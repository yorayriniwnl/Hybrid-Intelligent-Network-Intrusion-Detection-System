"""
Data Loader Module for Hybrid Intelligent Network Intrusion Detection System
Academic Phase 1 Prototype: Dataset Loading and Inspection
"""

import os
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any

class NetworkDataLoader:
    """
    Handles loading, initial inspection, and schema verification
    for network-flow intrusion detection datasets (CIC-IDS2017).
    """

    def __init__(self, data_path: str):
        self.data_path = data_path
        self.raw_df: pd.DataFrame = None

    def load_data(self, sample_size: int = None, random_state: int = 42) -> pd.DataFrame:
        """
        Loads CSV dataset with automatic column stripping and inspection.
        Optionally takes a stratified or random sample for rapid prototyping.
        """
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset not found at {self.data_path}")

        print(f"[*] Loading network traffic dataset from: {self.data_path}")
        df = pd.read_csv(self.data_path)
        
        # Standardize column names by stripping leading/trailing whitespace
        df.columns = df.columns.str.strip()
        
        if sample_size and sample_size < len(df):
            print(f"[*] Sampling {sample_size:,} records from {len(df):,} total flows (stratified)...")
            label_col = [c for c in df.columns if c.lower() == 'label'][0]
            # Stratified sample by label
            df = df.groupby(label_col, group_keys=False).apply(
                lambda x: x.sample(int(np.rint(sample_size * len(x) / len(df))), random_state=random_state)
            ).reset_index(drop=True)

        self.raw_df = df
        return df

    def inspect_dataset(self) -> Dict[str, Any]:
        """
        Performs thorough exploratory inspection of the dataset.
        Returns summary statistics, missing counts, infinite counts, and class distribution.
        """
        if self.raw_df is None:
            raise ValueError("Data has not been loaded. Call load_data() first.")

        label_col = [c for c in self.raw_df.columns if c.lower() == 'label'][0]
        numeric_cols = self.raw_df.select_dtypes(include=[np.number]).columns
        
        # Count missing and infinite values
        null_counts = self.raw_df.isnull().sum()
        cols_with_nulls = null_counts[null_counts > 0].to_dict()
        
        # Count infinite values in numeric columns
        inf_counts = np.isinf(self.raw_df[numeric_cols]).sum()
        cols_with_infs = inf_counts[inf_counts > 0].to_dict()

        # Class counts and percentages
        class_counts = self.raw_df[label_col].value_counts().to_dict()
        class_percentages = (self.raw_df[label_col].value_counts(normalize=True) * 100).round(2).to_dict()

        inspection_report = {
            "total_samples": len(self.raw_df),
            "total_features": len(self.raw_df.columns) - 1,
            "label_column": label_col,
            "class_distribution": {
                label: {"count": class_counts[label], "percentage": class_percentages[label]}
                for label in class_counts
            },
            "columns_with_nulls": cols_with_nulls,
            "total_null_values": int(null_counts.sum()),
            "columns_with_infs": cols_with_infs,
            "total_infinite_values": int(inf_counts.sum()),
            "sample_columns": list(self.raw_df.columns[:8])
        }

        return inspection_report

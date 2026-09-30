"""
Data Preprocessing Module for Hybrid Intelligent Network Intrusion Detection System
FR2: Cleaning, Outlier/Inf/Null Handling, Train/Test Split Without Leakage, Scaling
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
import joblib
import os
from typing import Tuple, Dict, Any, List

class NetworkDataPreprocessor:
    """
    Implements network traffic cleaning, leakage-free featurization,
    standardization, and persistent pipeline artifact management.
    """

    def __init__(self, target_col: str = 'Label', scaler_type: str = 'robust'):
        self.target_col = target_col
        self.scaler_type = scaler_type
        self.label_encoder = LabelEncoder()
        self.scaler = RobustScaler() if scaler_type == 'robust' else StandardScaler()
        self.feature_columns: List[str] = []
        self.median_imputer_values: Dict[str, float] = {}
        self.constant_columns: List[str] = []
        self.classes_: List[str] = []

    def clean_raw_dataframe(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Cleans column names, converts inf to nan, and removes duplicate flows.
        """
        df_clean = df.copy()
        
        # 1. Strip whitespace from column names
        df_clean.columns = df_clean.columns.str.strip()

        # 2. Check and record duplicate rows
        initial_rows = len(df_clean)
        duplicates_count = int(df_clean.duplicated().sum())
        if duplicates_count > 0:
            df_clean = df_clean.drop_duplicates().reset_index(drop=True)
            print(f"[*] Removed {duplicates_count:,} duplicate flows ({duplicates_count/initial_rows*100:.2f}%).")

        # 3. Replace inf and -inf with NaN for statistical imputation
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        inf_mask = np.isinf(df_clean[numeric_cols])
        inf_count = int(inf_mask.sum().sum())
        if inf_count > 0:
            df_clean[numeric_cols] = df_clean[numeric_cols].replace([np.inf, -np.inf], np.nan)
            print(f"[*] Replaced {inf_count:,} infinite values with NaN for robust imputation.")

        cleaning_stats = {
            "initial_rows": initial_rows,
            "deduplicated_rows": len(df_clean),
            "duplicates_removed": duplicates_count,
            "infinite_values_cleaned": inf_count
        }
        return df_clean, cleaning_stats

    def prepare_train_test_split(
        self,
        df_clean: pd.DataFrame,
        test_size: float = 0.20,
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Executes stratified train/test split BEFORE any scaling or imputation
        to strictly prevent data leakage between evaluation and training sets.
        """
        if self.target_col not in df_clean.columns:
            raise KeyError(f"Target column '{self.target_col}' not found in columns: {list(df_clean.columns)}")

        X_raw = df_clean.drop(columns=[self.target_col])
        y_raw = df_clean[self.target_col].astype(str)

        # 1. Fit LabelEncoder on target classes
        y_encoded = self.label_encoder.fit_transform(y_raw)
        self.classes_ = list(self.label_encoder.classes_)
        print(f"[*] Encoded Target Classes: {dict(enumerate(self.classes_))}")

        # 2. Identify zero-variance / constant columns on training candidate
        std_per_col = X_raw.std(numeric_only=True)
        self.constant_columns = list(std_per_col[std_per_col == 0].index)
        if self.constant_columns:
            print(f"[*] Dropping {len(self.constant_columns)} constant/zero-variance columns: {self.constant_columns}")
            X_raw = X_raw.drop(columns=self.constant_columns)

        self.feature_columns = list(X_raw.columns)

        # 3. Stratified Split to guarantee identical attack representation
        X_train_df, X_test_df, y_train, y_test = train_test_split(
            X_raw,
            y_encoded,
            test_size=test_size,
            stratify=y_encoded,
            random_state=random_state
        )

        print(f"[*] Train set: {len(X_train_df):,} flows | Test set: {len(X_test_df):,} flows (test_size={test_size*100:.0f}%)")

        # 4. Compute Median Imputation ONLY on Train Split (Strict Featurization Ordering)
        for col in self.feature_columns:
            med_val = X_train_df[col].median()
            self.median_imputer_values[col] = float(0.0 if np.isnan(med_val) else med_val)

        # Impute missing values with training medians
        X_train_imputed = X_train_df.fillna(self.median_imputer_values)
        X_test_imputed = X_test_df.fillna(self.median_imputer_values)

        # 5. Fit Scaler ONLY on Training Split, then transform Test
        X_train_scaled = self.scaler.fit_transform(X_train_imputed)
        X_test_scaled = self.scaler.transform(X_test_imputed)

        split_summary = {
            "train_samples": len(X_train_df),
            "test_samples": len(X_test_df),
            "num_features": len(self.feature_columns),
            "feature_names": self.feature_columns,
            "classes": self.classes_,
            "dropped_constant_columns": self.constant_columns
        }

        return X_train_scaled, X_test_scaled, y_train, y_test, split_summary

    def transform_samples(self, samples_df: pd.DataFrame) -> np.ndarray:
        """
        Transforms raw unseen network flow dataframe using saved pipeline transformations.
        Used for real-time inference demonstration.
        """
        df_sample = samples_df.copy()
        df_sample.columns = df_sample.columns.str.strip()

        # Remove target column if present
        if self.target_col in df_sample.columns:
            df_sample = df_sample.drop(columns=[self.target_col])

        # Drop constant columns
        if self.constant_columns:
            df_sample = df_sample.drop(columns=[c for c in self.constant_columns if c in df_sample.columns], errors='ignore')

        # Replace inf with nan
        numeric_cols = df_sample.select_dtypes(include=[np.number]).columns
        df_sample[numeric_cols] = df_sample[numeric_cols].replace([np.inf, -np.inf], np.nan)

        # Align columns
        for col in self.feature_columns:
            if col not in df_sample.columns:
                df_sample[col] = self.median_imputer_values.get(col, 0.0)

        df_aligned = df_sample[self.feature_columns]

        # Impute with stored training medians
        df_imputed = df_aligned.fillna(self.median_imputer_values)

        # Scale using fitted scaler
        return self.scaler.transform(df_imputed)

    def save(self, filepath: str):
        """Persists fitted preprocessor to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)
        print(f"[*] Preprocessing pipeline saved to: {filepath}")

    @staticmethod
    def load(filepath: str) -> 'NetworkDataPreprocessor':
        """Loads fitted preprocessor from disk."""
        return joblib.load(filepath)

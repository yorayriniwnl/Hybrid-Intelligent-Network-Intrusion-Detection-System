"""
Autoencoder Anomaly Detection Module for Hybrid Intelligent Network Intrusion Detection System
FR4: Normal-traffic reconstruction, statistical anomaly threshold selection, and out-of-distribution detection
"""

import time
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from typing import Tuple, Dict, Any

class DeepAutoencoder(nn.Module):
    """
    Symmetric Deep Autoencoder for Network Flow Anomaly Detection.
    Compresses flow features into a compact latent bottleneck and reconstructs them.
    """
    def __init__(self, input_dim: int, latent_dim: int = 12):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 48),
            nn.BatchNorm1d(48),
            nn.LeakyReLU(0.1),
            nn.Linear(48, 24),
            nn.BatchNorm1d(24),
            nn.LeakyReLU(0.1),
            nn.Linear(24, latent_dim),
            nn.LeakyReLU(0.1)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 24),
            nn.BatchNorm1d(24),
            nn.LeakyReLU(0.1),
            nn.Linear(24, 48),
            nn.BatchNorm1d(48),
            nn.LeakyReLU(0.1),
            nn.Linear(48, input_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon

class AutoencoderTrainer:
    """
    Manages Autoencoder training on benign traffic,
    statistical threshold determination, and anomaly scoring.
    """
    def __init__(self, input_dim: int, latent_dim: int = 12, lr: float = 0.002):
        self.device = torch.device("cpu")
        self.model = DeepAutoencoder(input_dim, latent_dim).to(self.device)
        self.criterion = nn.MSELoss(reduction='none')
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-5)
        self.anomaly_threshold: float = 0.0
        self.val_recon_stats: Dict[str, float] = {}

    def fit_on_benign(
        self,
        X_benign_train: np.ndarray,
        X_benign_val: np.ndarray,
        epochs: int = 6,
        batch_size: int = 256,
        percentile_threshold: float = 98.5
    ) -> Dict[str, Any]:
        """
        Trains exclusively on normal/benign network traffic and determines
        the empirical anomaly threshold from holdout validation flows.
        """
        print(f"\n[+] Training Autoencoder on {len(X_benign_train):,} BENIGN flows (epochs={epochs})...")
        self.model.train()
        start = time.perf_counter()

        X_t = torch.tensor(X_benign_train, dtype=torch.float32)
        dataset = TensorDataset(X_t, X_t)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            for batch_X, _ in loader:
                batch_X = batch_X.to(self.device)
                self.optimizer.zero_grad()
                recon = self.model(batch_X)
                loss = self.criterion(recon, batch_X).mean()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item() * len(batch_X)

            epoch_loss = total_loss / len(X_benign_train)
            if epoch % 1 == 0 or epoch == epochs:
                print(f"    Epoch {epoch}/{epochs} - Benign Reconstruction MSE: {epoch_loss:.5f}")

        training_time = time.perf_counter() - start

        # Statistical Threshold Selection on validation benign flows (FR4 Requirement)
        val_errors = self.get_reconstruction_errors(X_benign_val)
        mean_err = float(np.mean(val_errors))
        std_err = float(np.std(val_errors))
        self.anomaly_threshold = float(np.percentile(val_errors, percentile_threshold))

        self.val_recon_stats = {
            "mean_val_error": round(mean_err, 5),
            "std_val_error": round(std_err, 5),
            "anomaly_threshold": round(self.anomaly_threshold, 5),
            "threshold_percentile": percentile_threshold,
            "training_time": round(training_time, 2)
        }
        print(f"[+] Statistical Anomaly Threshold Determined: {self.anomaly_threshold:.5f} ({percentile_threshold}th percentile on validation benign traffic)")
        return self.val_recon_stats

    def get_reconstruction_errors(self, X: np.ndarray) -> np.ndarray:
        """Calculates per-flow Mean Squared Reconstruction Error."""
        self.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
            recon = self.model(X_t)
            # MSE per sample across features
            errors = ((recon - X_t) ** 2).mean(dim=1).cpu().numpy()
        return errors

    def predict_anomalies(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns binary anomaly flag (True = anomalous/suspicious)
        and continuous reconstruction error score.
        """
        errors = self.get_reconstruction_errors(X)
        is_anomaly = errors > self.anomaly_threshold
        return is_anomaly, errors

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'anomaly_threshold': self.anomaly_threshold,
            'val_recon_stats': self.val_recon_stats
        }, filepath)
        print(f"[*] Saved Autoencoder checkpoint to: {filepath}")

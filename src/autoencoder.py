"""
Autoencoder Anomaly Detection Module for Hybrid Intelligent Network Intrusion Detection System
FR4: Normal-traffic reconstruction, statistical anomaly threshold selection, and out-of-distribution detection
"""

import time
import os
import numpy as np
from typing import Tuple, Dict, Any

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    nn = object  # dummy fallback

class NumpyAutoencoder:
    """
    Lightweight, high-throughput NumPy inference engine for DeepAutoencoder.
    Runs with zero PyTorch dependencies for ultra-fast serverless execution (<0.2ms latency).
    """
    def __init__(self, weights_path: str):
        data = np.load(weights_path)
        self.weights = {k: data[k] for k in data.files}
        self.anomaly_threshold = float(self.weights.get("anomaly_threshold", [206885990957056.0])[0])

    def _bn_forward(self, x: np.ndarray, weight: np.ndarray, bias: np.ndarray, running_mean: np.ndarray, running_var: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        return (x - running_mean) / np.sqrt(running_var + eps) * weight + bias

    def _lrelu(self, x: np.ndarray, alpha: float = 0.1) -> np.ndarray:
        return np.where(x > 0, x, x * alpha)

    def forward(self, x: np.ndarray) -> np.ndarray:
        w = self.weights
        # Encoder
        h = np.dot(x, w["encoder.0.weight"].T) + w["encoder.0.bias"]
        h = self._bn_forward(h, w["encoder.1.weight"], w["encoder.1.bias"], w["encoder.1.running_mean"], w["encoder.1.running_var"])
        h = self._lrelu(h)
        h = np.dot(h, w["encoder.3.weight"].T) + w["encoder.3.bias"]
        h = self._bn_forward(h, w["encoder.4.weight"], w["encoder.4.bias"], w["encoder.4.running_mean"], w["encoder.4.running_var"])
        h = self._lrelu(h)
        z = np.dot(h, w["encoder.6.weight"].T) + w["encoder.6.bias"]
        z = self._lrelu(z)

        # Decoder
        h = np.dot(z, w["decoder.0.weight"].T) + w["decoder.0.bias"]
        h = self._bn_forward(h, w["decoder.1.weight"], w["decoder.1.bias"], w["decoder.1.running_mean"], w["decoder.1.running_var"])
        h = self._lrelu(h)
        h = np.dot(h, w["decoder.3.weight"].T) + w["decoder.3.bias"]
        h = self._bn_forward(h, w["decoder.4.weight"], w["decoder.4.bias"], w["decoder.4.running_mean"], w["decoder.4.running_var"])
        h = self._lrelu(h)
        return np.dot(h, w["decoder.6.weight"].T) + w["decoder.6.bias"]

    def predict_anomalies(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if X.ndim == 1:
            X = X.reshape(1, -1)
        X_recon = self.forward(X)
        errors = np.mean(np.square(X - X_recon), axis=1)
        return errors > self.anomaly_threshold, errors


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

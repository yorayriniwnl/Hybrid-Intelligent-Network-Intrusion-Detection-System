"""
TabTransformer Module for Hybrid Intelligent Network Intrusion Detection System
Phase 2: Attention-Based Deep Learning for Tabular Network Flows
"""

import time
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from typing import Tuple, Dict, Any, List

class TabTransformerBlock(nn.Module):
    """Multi-Head Self-Attention block with residual connection and LayerNorm."""
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 64, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Multi-head self-attention
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + attn_out)
        # Feed-forward network
        ff_out = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = self.norm2(x + ff_out)
        return x

class TabTransformer(nn.Module):
    """
    Attention-based Tabular Deep Learning Architecture
    Maps network flow features into multi-dimensional tokens,
    passes them through self-attention, and classifies via MLP head.
    """
    def __init__(
        self,
        num_features: int,
        num_classes: int = 2,
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 64,
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_features = num_features
        self.d_model = d_model

        # Linear projection of each continuous feature into an embedding dimension
        self.feature_projectors = nn.ModuleList([
            nn.Linear(1, d_model) for _ in range(num_features)
        ])

        # Stacked Transformer attention layers
        self.transformer_blocks = nn.ModuleList([
            TabTransformerBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

        # Classification MLP head
        self.mlp_head = nn.Sequential(
            nn.Linear(num_features * d_model, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        # Project each feature into (batch_size, 1, d_model)
        tokens = [
            self.feature_projectors[i](x[:, i:i+1]).unsqueeze(1)
            for i in range(self.num_features)
        ]
        # Concatenate tokens along sequence dimension -> (batch_size, num_features, d_model)
        tokens = torch.cat(tokens, dim=1)

        # Pass through attention layers
        for block in self.transformer_blocks:
            tokens = block(tokens)

        # Flatten tokens and classify
        flattened = tokens.reshape(batch_size, -1)
        logits = self.mlp_head(flattened)
        return logits

class TabTransformerTrainer:
    """Trains and manages TabTransformer PyTorch model."""
    def __init__(self, num_features: int, num_classes: int = 2, lr: float = 0.001):
        self.device = torch.device("cpu")
        self.model = TabTransformer(num_features, num_classes).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int = 5,
        batch_size: int = 256
    ) -> float:
        print(f"\n[+] Training TabTransformer (Attention-Based Deep Learning, epochs={epochs}, batch_size={batch_size})...")
        self.model.train()
        start = time.perf_counter()

        X_t = torch.tensor(X_train, dtype=torch.float32)
        y_t = torch.tensor(y_train, dtype=torch.long)
        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            for batch_X, batch_y in loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item() * len(batch_y)

            epoch_loss = total_loss / len(X_train)
            if epoch % 1 == 0 or epoch == epochs:
                print(f"    Epoch {epoch}/{epochs} - CrossEntropy Loss: {epoch_loss:.4f}")

        duration = time.perf_counter() - start
        print(f"[+] TabTransformer training complete in {duration:.2f} seconds.")
        return duration

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
            logits = self.model(X_t)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
        return preds

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
            logits = self.model(X_t)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'num_features': self.model.num_features
        }, filepath)
        print(f"[*] Saved TabTransformer checkpoint to: {filepath}")

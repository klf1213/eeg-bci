from __future__ import annotations

import torch
from torch import nn


class TemporalAttention(nn.Module):
    def __init__(self, n_features: int, hidden_size: int = 16):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(n_features, hidden_size), nn.Tanh(), nn.Linear(hidden_size, 1))

    def forward(self, x: torch.Tensor):
        sequence = x.transpose(1, 2)
        weights = torch.softmax(self.score(sequence).squeeze(-1), dim=1)
        context = torch.sum(sequence * weights.unsqueeze(-1), dim=1)
        return context, weights


class EEGNetTemporalAttention(nn.Module):
    def __init__(self, n_channels=22, n_classes=4, f1=8, depth_multiplier=2, f2=16,
                 temporal_kernel_length=32, separable_kernel_length=16, dropout=0.5,
                 attention_hidden_size=16, n_time_points=256, second_pool_size=8,
                 use_position=False, residual_mean=False):
        super().__init__()
        features = f1 * depth_multiplier
        self.features = nn.Sequential(
            nn.Conv2d(1, f1, (1, temporal_kernel_length), padding="same", bias=False), nn.BatchNorm2d(f1),
            nn.Conv2d(f1, features, (n_channels, 1), groups=f1, bias=False), nn.BatchNorm2d(features),
            nn.ELU(), nn.AvgPool2d((1, 4)), nn.Dropout(dropout),
            nn.Conv2d(features, features, (1, separable_kernel_length), padding="same", groups=features, bias=False),
            nn.Conv2d(features, f2, (1, 1), bias=False), nn.BatchNorm2d(f2),
            nn.ELU(), nn.AvgPool2d((1, second_pool_size)), nn.Dropout(dropout),
        )
        self.attention = TemporalAttention(f2, attention_hidden_size)
        self.position = (
            nn.Parameter(torch.zeros(1, n_time_points // (4 * second_pool_size), f2))
            if use_position else None
        )
        self.residual_mean = residual_mean
        self.attention_gate = nn.Parameter(torch.tensor(-2.0)) if residual_mean else None
        self.classifier = nn.Linear(f2, n_classes)

    def forward(self, x: torch.Tensor, return_attention=False):
        features = self.features(x)
        if features.shape[2] != 1:
            raise ValueError(f"Expected one spatial dimension, got {features.shape}.")
        sequence = features.squeeze(2)
        if self.position is not None:
            sequence = sequence + self.position[:, :sequence.shape[-1]].transpose(1, 2)
        context, weights = self.attention(sequence)
        if self.residual_mean:
            gate = torch.sigmoid(self.attention_gate)
            context = sequence.mean(dim=-1) + gate * (context - sequence.mean(dim=-1))
        logits = self.classifier(context)
        return (logits, weights) if return_attention else logits


class EEGNetTemporalMean(nn.Module):
    """EEGNet feature extractor with parameter-free temporal mean pooling."""

    def __init__(self, n_channels=22, n_classes=4, f1=8, depth_multiplier=2, f2=16,
                 temporal_kernel_length=32, separable_kernel_length=16, dropout=0.5,
                 second_pool_size=8):
        super().__init__()
        features = f1 * depth_multiplier
        self.features = nn.Sequential(
            nn.Conv2d(1, f1, (1, temporal_kernel_length), padding="same", bias=False), nn.BatchNorm2d(f1),
            nn.Conv2d(f1, features, (n_channels, 1), groups=f1, bias=False), nn.BatchNorm2d(features),
            nn.ELU(), nn.AvgPool2d((1, 4)), nn.Dropout(dropout),
            nn.Conv2d(features, features, (1, separable_kernel_length), padding="same", groups=features, bias=False),
            nn.Conv2d(features, f2, (1, 1), bias=False), nn.BatchNorm2d(f2),
            nn.ELU(), nn.AvgPool2d((1, second_pool_size)), nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(f2, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        if features.shape[2] != 1:
            raise ValueError(f"Expected one spatial dimension, got {features.shape}.")
        return self.classifier(features.squeeze(2).mean(dim=-1))

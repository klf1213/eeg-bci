from __future__ import annotations

import torch
from torch import nn


class EEGNet(nn.Module):
    def __init__(self, n_channels=22, n_time_points=256, n_classes=4, f1=8,
                 depth_multiplier=2, f2=16, temporal_kernel_length=32,
                 separable_kernel_length=16, dropout=0.5):
        super().__init__()
        features = f1 * depth_multiplier
        self.features = nn.Sequential(
            nn.Conv2d(1, f1, (1, temporal_kernel_length), padding="same", bias=False),
            nn.BatchNorm2d(f1),
            nn.Conv2d(f1, features, (n_channels, 1), groups=f1, bias=False),
            nn.BatchNorm2d(features), nn.ELU(), nn.AvgPool2d((1, 4)), nn.Dropout(dropout),
            nn.Conv2d(features, features, (1, separable_kernel_length), padding="same", groups=features, bias=False),
            nn.Conv2d(features, f2, (1, 1), bias=False),
            nn.BatchNorm2d(f2), nn.ELU(), nn.AvgPool2d((1, 8)), nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(f2 * (n_time_points // 32), n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(torch.flatten(self.features(x), start_dim=1))


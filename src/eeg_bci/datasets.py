from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class EEGDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        if x.ndim != 3 or len(x) != len(y):
            raise ValueError("Expected matching X (trials, channels, time) and y.")
        self.x = torch.from_numpy(x[:, None]).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


def make_loader(x, y, batch_size, shuffle=False):
    return DataLoader(EEGDataset(x, y), batch_size=batch_size, shuffle=shuffle, num_workers=0)


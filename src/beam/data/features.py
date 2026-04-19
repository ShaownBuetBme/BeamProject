from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Standardizer:
    mean: np.ndarray
    std: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.std


def fit_standardizer(values: np.ndarray) -> Standardizer:
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return Standardizer(mean=mean, std=std)

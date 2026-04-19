from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class BeamData:
    x_img: np.ndarray
    x_num: np.ndarray
    y: np.ndarray
    beam_id: np.ndarray
    image_path: np.ndarray
    fold_id: np.ndarray
    fold_group_ids: np.ndarray
    numeric_columns: np.ndarray
    target_columns: np.ndarray

    @property
    def num_folds(self) -> int:
        if self.fold_group_ids.size > 0:
            return int(self.fold_group_ids.shape[0])
        if self.fold_id.size > 0:
            return int(np.max(self.fold_id) + 1)
        return 0


def load_beam_npz(npz_path: str | Path) -> BeamData:
    npz_path = Path(npz_path)
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ not found: {npz_path}")

    with np.load(npz_path, allow_pickle=True) as data:
        required = [
            "X_img",
            "X_num",
            "Y",
            "beam_id",
            "image_path",
            "fold_id",
            "fold_group_ids",
            "numeric_columns",
            "target_columns",
        ]
        missing = [k for k in required if k not in data]
        if missing:
            raise ValueError(f"Missing required NPZ arrays: {missing}")

        return BeamData(
            x_img=data["X_img"],
            x_num=data["X_num"],
            y=data["Y"],
            beam_id=data["beam_id"],
            image_path=data["image_path"],
            fold_id=data["fold_id"],
            fold_group_ids=data["fold_group_ids"],
            numeric_columns=data["numeric_columns"],
            target_columns=data["target_columns"],
        )


def get_fold_indices(fold_id: np.ndarray, test_fold: int) -> tuple[np.ndarray, np.ndarray]:
    if test_fold < 0:
        raise ValueError("test_fold must be >= 0")

    test_mask = fold_id == test_fold
    train_mask = ~test_mask

    train_idx = np.where(train_mask)[0]
    test_idx = np.where(test_mask)[0]
    if test_idx.size == 0:
        raise ValueError(f"Fold {test_fold} has no samples")

    return train_idx, test_idx

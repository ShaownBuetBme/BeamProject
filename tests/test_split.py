import numpy as np

from beam.data.dataset import get_fold_indices


def test_get_fold_indices_no_overlap() -> None:
    fold_id = np.array([0, 0, 1, 1, 2, 2], dtype=np.int16)
    train_idx, test_idx = get_fold_indices(fold_id=fold_id, test_fold=1)

    assert set(train_idx.tolist()).isdisjoint(set(test_idx.tolist()))
    assert set(test_idx.tolist()) == {2, 3}

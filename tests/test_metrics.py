import numpy as np

from beam.evaluation.metrics import compute_regression_metrics


def test_compute_regression_metrics_perfect_prediction() -> None:
    y_true = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    y_pred = y_true.copy()

    metrics = compute_regression_metrics(y_true=y_true, y_pred=y_pred)

    assert metrics["mae_macro"] == 0.0
    assert metrics["rmse_macro"] == 0.0
    assert metrics["r2_macro"] == 1.0

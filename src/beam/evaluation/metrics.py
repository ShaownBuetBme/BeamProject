from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

TARGET_ORDER = ["load_capacity_kn", "max_deflection_mm"]


def _safe_r2_per_target(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    values: list[float] = []
    for idx in range(y_true.shape[1]):
        y_t = y_true[:, idx]
        y_p = y_pred[:, idx]
        if float(np.var(y_t)) < 1e-12:
            values.append(float("nan"))
        else:
            values.append(float(r2_score(y_t, y_p)))
    return np.array(values, dtype=np.float64)


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    mae_per_target = mean_absolute_error(y_true, y_pred, multioutput="raw_values")
    rmse_per_target = np.sqrt(mean_squared_error(y_true, y_pred, multioutput="raw_values"))
    r2_per_target = _safe_r2_per_target(y_true=y_true, y_pred=y_pred)

    metrics: dict[str, float] = {
        "mae_macro": float(np.mean(mae_per_target)),
        "rmse_macro": float(np.mean(rmse_per_target)),
        "r2_macro": float(np.nanmean(r2_per_target)),
    }

    for idx, target in enumerate(TARGET_ORDER):
        metrics[f"mae_{target}"] = float(mae_per_target[idx])
        metrics[f"rmse_{target}"] = float(rmse_per_target[idx])
        metrics[f"r2_{target}"] = float(r2_per_target[idx])

    return metrics


def aggregate_fold_metrics(
    fold_metrics: list[dict[str, float]], fold_sizes: list[int]
) -> dict[str, float]:
    if not fold_metrics:
        return {}

    keys = sorted(k for k in fold_metrics[0].keys() if k != "fold")
    unweighted: dict[str, float] = {}
    weighted: dict[str, float] = {}

    size_arr = np.array(fold_sizes, dtype=np.float64)
    size_arr = size_arr / size_arr.sum()

    for key in keys:
        values = np.array([m[key] for m in fold_metrics], dtype=np.float64)
        unweighted[f"mean_{key}"] = float(np.nanmean(values))
        unweighted[f"std_{key}"] = float(np.nanstd(values, ddof=0))

        valid = ~np.isnan(values)
        if np.any(valid):
            weights_valid = size_arr[valid]
            weights_valid = weights_valid / weights_valid.sum()
            weighted[f"weighted_mean_{key}"] = float(np.sum(values[valid] * weights_valid))
        else:
            weighted[f"weighted_mean_{key}"] = float("nan")

    return {**unweighted, **weighted}

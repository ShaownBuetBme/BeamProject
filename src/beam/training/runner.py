from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from beam.data.dataset import get_fold_indices, load_beam_npz
from beam.data.features import fit_standardizer
from beam.evaluation.metrics import aggregate_fold_metrics, compute_regression_metrics
from beam.evaluation.plots import save_fold_metric_bar, save_residual_histograms, save_scatter_plots
from beam.models.factory import create_model
from beam.utils.io import make_run_dir, write_csv_dicts, write_json
from beam.utils.seed import set_seed


@dataclass
class TrainConfig:
    experiment_name: str
    dataset_path: str
    output_dir: str
    model_name: str
    feature_mode: str
    run_mode: str
    test_fold: int
    random_seed: int
    alpha: float
    n_estimators: int


def _build_features(
    data_x_img: np.ndarray, data_x_num: np.ndarray, feature_mode: str
) -> np.ndarray:
    mode = feature_mode.lower()
    if mode == "numeric":
        return data_x_num

    if mode == "image":
        return data_x_img.reshape(data_x_img.shape[0], -1)

    if mode == "image_numeric":
        flat_img = data_x_img.reshape(data_x_img.shape[0], -1)
        return np.concatenate([flat_img, data_x_num], axis=1)

    raise ValueError(
        "feature_mode must be one of: numeric, image, image_numeric"
    )


def _run_one_fold(
    *,
    x: np.ndarray,
    y: np.ndarray,
    fold_id: np.ndarray,
    fold_idx: int,
    model_name: str,
    alpha: float,
    n_estimators: int,
    random_seed: int,
    target_names: list[str],
    fold_output_dir: Path,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    train_idx, test_idx = get_fold_indices(fold_id=fold_id, test_fold=fold_idx)

    x_train = x[train_idx]
    x_test = x[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    scaler = fit_standardizer(x_train)
    x_train_scaled = scaler.transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    model = create_model(
        model_name=model_name,
        random_seed=random_seed,
        alpha=alpha,
        n_estimators=n_estimators,
    )
    model.fit(x_train_scaled, y_train)
    y_pred = model.predict(x_test_scaled)

    fold_metrics = compute_regression_metrics(y_true=y_test, y_pred=y_pred)

    plots_dir = fold_output_dir / "plots"
    save_scatter_plots(y_true=y_test, y_pred=y_pred, out_dir=plots_dir, target_names=target_names)
    save_residual_histograms(
        y_true=y_test,
        y_pred=y_pred,
        out_dir=plots_dir,
        target_names=target_names,
    )

    prediction_rows: list[dict[str, float]] = []
    for i in range(y_test.shape[0]):
        row = {
            "fold": fold_idx,
            "sample_index": int(test_idx[i]),
            "y_true_0": float(y_test[i, 0]),
            "y_pred_0": float(y_pred[i, 0]),
            "y_true_1": float(y_test[i, 1]),
            "y_pred_1": float(y_pred[i, 1]),
        }
        prediction_rows.append(row)

    write_json(fold_metrics, fold_output_dir / "metrics.json")
    write_csv_dicts(prediction_rows, fold_output_dir / "predictions.csv")

    return fold_metrics, prediction_rows


def run_training(config: TrainConfig) -> Path:
    set_seed(config.random_seed)

    data = load_beam_npz(config.dataset_path)
    x = _build_features(
        data_x_img=data.x_img,
        data_x_num=data.x_num,
        feature_mode=config.feature_mode,
    )
    y = data.y
    target_names = [str(name) for name in data.target_columns.tolist()]

    run_dir = make_run_dir(base_dir=config.output_dir, experiment_name=config.experiment_name)
    write_json(config.__dict__, run_dir / "resolved_config.json")

    fold_metrics_list: list[dict[str, float]] = []
    fold_sizes: list[int] = []
    all_predictions: list[dict[str, float]] = []

    if config.run_mode == "single_fold":
        fold_indices = [config.test_fold]
    elif config.run_mode == "cv12":
        fold_indices = list(range(data.num_folds))
    else:
        raise ValueError("run_mode must be one of: single_fold, cv12")

    for fold_idx in fold_indices:
        fold_output_dir = run_dir / f"fold_{fold_idx}"
        fold_output_dir.mkdir(parents=True, exist_ok=True)

        fold_metrics, predictions = _run_one_fold(
            x=x,
            y=y,
            fold_id=data.fold_id,
            fold_idx=fold_idx,
            model_name=config.model_name,
            alpha=config.alpha,
            n_estimators=config.n_estimators,
            random_seed=config.random_seed,
            target_names=target_names,
            fold_output_dir=fold_output_dir,
        )
        fold_metrics["fold"] = float(fold_idx)
        fold_metrics_list.append(fold_metrics)
        fold_sizes.append(len(predictions))
        all_predictions.extend(predictions)

    aggregate = aggregate_fold_metrics(fold_metrics=fold_metrics_list, fold_sizes=fold_sizes)

    rows_for_csv = []
    for fm in fold_metrics_list:
        row = {k: v for k, v in fm.items()}
        rows_for_csv.append(row)

    write_json(aggregate, run_dir / "metrics_summary.json")
    write_csv_dicts(rows_for_csv, run_dir / "fold_metrics.csv")
    write_csv_dicts(all_predictions, run_dir / "all_predictions.csv")

    if fold_metrics_list:
        save_fold_metric_bar(
            fold_scores=fold_metrics_list,
            out_path=run_dir / "plots" / "fold_rmse_macro.png",
            metric_key="rmse_macro",
        )

    return run_dir

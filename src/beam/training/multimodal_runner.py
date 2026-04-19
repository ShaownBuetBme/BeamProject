from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from beam.data.dataset import get_fold_indices, load_beam_npz
from beam.data.features import fit_standardizer
from beam.evaluation.metrics import aggregate_fold_metrics, compute_regression_metrics
from beam.evaluation.plots import save_residual_histograms, save_scatter_plots
from beam.models.multimodal_torch import MultimodalRegressor
from beam.utils.io import make_run_dir, write_csv_dicts, write_json
from beam.utils.seed import set_seed


class BeamTorchDataset(Dataset):
    def __init__(self, x_img: np.ndarray, x_num: np.ndarray, y: np.ndarray) -> None:
        self.x_img = torch.from_numpy(np.transpose(x_img, (0, 3, 1, 2))).float()
        self.x_num = torch.from_numpy(x_num).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.x_img[idx], self.x_num[idx], self.y[idx]


@dataclass
class MultimodalTrainConfig:
    experiment_name: str
    dataset_path: str
    output_dir: str
    run_mode: str
    test_fold: int
    random_seed: int
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    pretrained_backbone: bool


def _scale_targets(
    y_train: np.ndarray, y_other: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    y_mean = y_train.mean(axis=0)
    y_std = y_train.std(axis=0)
    y_std = np.where(y_std < 1e-8, 1.0, y_std)
    y_train_scaled = (y_train - y_mean) / y_std
    y_other_scaled = (y_other - y_mean) / y_std
    return y_train_scaled, y_other_scaled, y_mean, y_std


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    y_mean: np.ndarray,
    y_std: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    criterion = nn.MSELoss()

    total_loss = 0.0
    all_pred: list[np.ndarray] = []
    all_true: list[np.ndarray] = []

    with torch.no_grad():
        for x_img, x_num, y in loader:
            x_img = x_img.to(device)
            x_num = x_num.to(device)
            y = y.to(device)

            pred = model(x_img, x_num)
            loss = criterion(pred, y)
            total_loss += float(loss.item()) * y.shape[0]

            pred_np = pred.cpu().numpy() * y_std + y_mean
            y_np = y.cpu().numpy() * y_std + y_mean
            all_pred.append(pred_np)
            all_true.append(y_np)

    y_pred = np.concatenate(all_pred, axis=0)
    y_true = np.concatenate(all_true, axis=0)
    avg_loss = total_loss / len(loader.dataset)
    return y_true, y_pred, avg_loss


def _build_dataloaders(
    x_img: np.ndarray,
    x_num: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    batch_size: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_ds = BeamTorchDataset(x_img[train_idx], x_num[train_idx], y[train_idx])
    val_ds = BeamTorchDataset(x_img[val_idx], x_num[val_idx], y[val_idx])
    test_ds = BeamTorchDataset(x_img[test_idx], x_num[test_idx], y[test_idx])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def _one_fold(
    *,
    fold_idx: int,
    num_folds: int,
    x_img: np.ndarray,
    x_num: np.ndarray,
    y: np.ndarray,
    fold_id: np.ndarray,
    cfg: MultimodalTrainConfig,
    target_names: list[str],
    run_dir: Path,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    _, test_idx = get_fold_indices(fold_id=fold_id, test_fold=fold_idx)
    val_fold = (fold_idx + 1) % num_folds
    _, val_idx = get_fold_indices(fold_id=fold_id, test_fold=val_fold)

    train_mask = np.ones(shape=(len(fold_id),), dtype=bool)
    train_mask[test_idx] = False
    train_mask[val_idx] = False
    train_idx = np.where(train_mask)[0]

    x_scaler = fit_standardizer(x_num[train_idx])
    x_num_scaled = x_num.copy()
    x_num_scaled[train_idx] = x_scaler.transform(x_num[train_idx])
    x_num_scaled[val_idx] = x_scaler.transform(x_num[val_idx])
    x_num_scaled[test_idx] = x_scaler.transform(x_num[test_idx])

    y_train_scaled, y_val_scaled, y_mean, y_std = _scale_targets(y[train_idx], y[val_idx])
    _, y_test_scaled, _, _ = _scale_targets(y[train_idx], y[test_idx])

    y_scaled = y.copy()
    y_scaled[train_idx] = y_train_scaled
    y_scaled[val_idx] = y_val_scaled
    y_scaled[test_idx] = y_test_scaled

    train_loader, val_loader, test_loader = _build_dataloaders(
        x_img=x_img,
        x_num=x_num_scaled,
        y=y_scaled,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        batch_size=cfg.batch_size,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultimodalRegressor(
        num_numeric_features=x_num.shape[1],
        output_dim=y.shape[1],
        pretrained_backbone=cfg.pretrained_backbone,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    criterion = nn.MSELoss()

    history_rows: list[dict[str, float]] = []

    for epoch in range(cfg.epochs):
        model.train()
        epoch_loss = 0.0

        for x_img_b, x_num_b, y_b in train_loader:
            x_img_b = x_img_b.to(device)
            x_num_b = x_num_b.to(device)
            y_b = y_b.to(device)

            optimizer.zero_grad(set_to_none=True)
            pred = model(x_img_b, x_num_b)
            loss = criterion(pred, y_b)
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item()) * y_b.shape[0]

        train_loss = epoch_loss / len(train_loader.dataset)
        _, _, val_loss = _evaluate(
            model=model,
            loader=val_loader,
            device=device,
            y_mean=y_mean,
            y_std=y_std,
        )
        history_rows.append(
            {
                "fold": fold_idx,
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
        )

    y_true, y_pred, _ = _evaluate(
        model=model,
        loader=test_loader,
        device=device,
        y_mean=y_mean,
        y_std=y_std,
    )

    fold_metrics = compute_regression_metrics(y_true=y_true, y_pred=y_pred)
    fold_metrics["fold"] = float(fold_idx)

    fold_dir = run_dir / f"fold_{fold_idx}"
    write_json(fold_metrics, fold_dir / "metrics.json")
    write_csv_dicts(history_rows, fold_dir / "training_history.csv")

    plot_dir = fold_dir / "plots"
    save_scatter_plots(y_true=y_true, y_pred=y_pred, out_dir=plot_dir, target_names=target_names)
    save_residual_histograms(
        y_true=y_true,
        y_pred=y_pred,
        out_dir=plot_dir,
        target_names=target_names,
    )

    predictions: list[dict[str, float]] = []
    for i in range(y_true.shape[0]):
        predictions.append(
            {
                "fold": float(fold_idx),
                "target_0_true": float(y_true[i, 0]),
                "target_0_pred": float(y_pred[i, 0]),
                "target_1_true": float(y_true[i, 1]),
                "target_1_pred": float(y_pred[i, 1]),
            }
        )

    return fold_metrics, predictions


def run_multimodal_training(cfg: MultimodalTrainConfig) -> Path:
    set_seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)
    torch.cuda.manual_seed_all(cfg.random_seed)

    data = load_beam_npz(cfg.dataset_path)
    x_img = data.x_img.astype(np.float32)
    x_num = data.x_num.astype(np.float32)
    y = data.y.astype(np.float32)

    run_dir = make_run_dir(base_dir=cfg.output_dir, experiment_name=cfg.experiment_name)
    write_json(cfg.__dict__, run_dir / "resolved_config.json")

    target_names = [str(v) for v in data.target_columns.tolist()]

    if cfg.run_mode == "single_fold":
        folds = [cfg.test_fold]
    elif cfg.run_mode == "cv12":
        folds = list(range(data.num_folds))
    else:
        raise ValueError("run_mode must be one of: single_fold, cv12")

    fold_metrics: list[dict[str, float]] = []
    fold_sizes: list[int] = []
    all_predictions: list[dict[str, float]] = []

    for fold_idx in folds:
        metrics, predictions = _one_fold(
            fold_idx=fold_idx,
            num_folds=data.num_folds,
            x_img=x_img,
            x_num=x_num,
            y=y,
            fold_id=data.fold_id,
            cfg=cfg,
            target_names=target_names,
            run_dir=run_dir,
        )
        fold_metrics.append(metrics)
        fold_sizes.append(len(predictions))
        all_predictions.extend(predictions)

    summary = aggregate_fold_metrics(fold_metrics=fold_metrics, fold_sizes=fold_sizes)
    fold_rows = [{k: v for k, v in m.items()} for m in fold_metrics]

    write_json(summary, run_dir / "metrics_summary.json")
    write_csv_dicts(fold_rows, run_dir / "fold_metrics.csv")
    write_csv_dicts(all_predictions, run_dir / "all_predictions.csv")

    return run_dir

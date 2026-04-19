from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from beam.training.multimodal_runner import MultimodalTrainConfig, run_multimodal_training
from beam.utils.config import load_yaml_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train multimodal CNN+MLP baseline (report-only)")
    parser.add_argument("--config", type=Path, default=Path("configs/multimodal_baseline.yaml"))

    parser.add_argument("--experiment-name", type=str)
    parser.add_argument("--dataset-path", type=str)
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--run-mode", type=str, choices=["single_fold", "cv12"])
    parser.add_argument("--test-fold", type=int)
    parser.add_argument("--random-seed", type=int)

    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--pretrained-backbone", type=str, choices=["true", "false"])

    return parser.parse_args()


def merge_config(base: dict, args: argparse.Namespace) -> dict:
    pretrained = None
    if args.pretrained_backbone is not None:
        pretrained = args.pretrained_backbone.lower() == "true"

    cli = {
        "experiment_name": args.experiment_name,
        "dataset_path": args.dataset_path,
        "output_dir": args.output_dir,
        "run_mode": args.run_mode,
        "test_fold": args.test_fold,
        "random_seed": args.random_seed,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "pretrained_backbone": pretrained,
    }

    merged = dict(base)
    for key, value in cli.items():
        if value is not None:
            merged[key] = value
    return merged


def main() -> None:
    args = parse_args()
    cfg = merge_config(load_yaml_config(args.config), args)

    required = [
        "experiment_name",
        "dataset_path",
        "output_dir",
        "run_mode",
        "test_fold",
        "random_seed",
        "batch_size",
        "epochs",
        "learning_rate",
        "weight_decay",
        "pretrained_backbone",
    ]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"Missing config keys: {missing}")

    run_dir = run_multimodal_training(MultimodalTrainConfig(**cfg))
    print(f"Run completed: {run_dir}")


if __name__ == "__main__":
    main()

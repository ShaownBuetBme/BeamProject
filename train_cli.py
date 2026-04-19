from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from beam.training.runner import TrainConfig, run_training
from beam.utils.config import load_yaml_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train beam models with CLI hyperparameters")
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))

    parser.add_argument("--experiment-name", type=str)
    parser.add_argument("--dataset-path", type=str)
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--model-name", type=str)
    parser.add_argument("--feature-mode", type=str, choices=["numeric", "image", "image_numeric"])
    parser.add_argument("--run-mode", type=str, choices=["single_fold", "cv12"])
    parser.add_argument("--test-fold", type=int)
    parser.add_argument("--random-seed", type=int)
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--n-estimators", type=int)

    return parser.parse_args()


def merge_config(base: dict, args: argparse.Namespace) -> dict:
    cli = {
        "experiment_name": args.experiment_name,
        "dataset_path": args.dataset_path,
        "output_dir": args.output_dir,
        "model_name": args.model_name,
        "feature_mode": args.feature_mode,
        "run_mode": args.run_mode,
        "test_fold": args.test_fold,
        "random_seed": args.random_seed,
        "alpha": args.alpha,
        "n_estimators": args.n_estimators,
    }

    merged = dict(base)
    for key, value in cli.items():
        if value is not None:
            merged[key] = value
    return merged


def main() -> None:
    args = parse_args()
    base_cfg = load_yaml_config(args.config)
    cfg = merge_config(base_cfg, args)

    required = [
        "experiment_name",
        "dataset_path",
        "output_dir",
        "model_name",
        "feature_mode",
        "run_mode",
        "test_fold",
        "random_seed",
        "alpha",
        "n_estimators",
    ]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"Missing config keys: {missing}")

    run_dir = run_training(TrainConfig(**cfg))
    print(f"Run completed: {run_dir}")


if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd

try:
    from PIL import Image
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Pillow is required to load images. Install it with: pip install Pillow"
    ) from exc


ROOT = Path(__file__).resolve().parent
DEFAULT_CSV = ROOT / "dataset" / "beam_dataset.csv"
DEFAULT_OUT = ROOT / "dataset" / "beam_multimodal.npz"
DEFAULT_META = ROOT / "dataset" / "beam_multimodal_meta.json"


NUMERIC_COLUMNS = ["beam_width", "beam_depth", "wa_content"]
TARGET_COLUMNS = ["target_load_capacity_kn", "target_max_deflection_mm"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a compressed NPZ multimodal dataset from beam_dataset.csv"
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Input CSV path")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output NPZ path")
    parser.add_argument(
        "--meta",
        type=Path,
        default=DEFAULT_META,
        help="Output metadata JSON path",
    )
    parser.add_argument("--height", type=int, default=224, help="Image height")
    parser.add_argument("--width", type=int, default=224, help="Image width")
    parser.add_argument(
        "--split-strategy",
        type=str,
        default="leave-one-group-out",
        choices=["leave-one-group-out", "random-group"],
        help="Dataset splitting strategy",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Validation fraction from unique beam_id groups",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Test fraction from unique beam_id groups",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.height <= 0 or args.width <= 0:
        raise ValueError("height and width must be positive integers")
    if args.split_strategy == "random-group":
        if args.val_ratio < 0 or args.test_ratio < 0:
            raise ValueError("val-ratio and test-ratio must be non-negative")
        if args.val_ratio + args.test_ratio >= 1.0:
            raise ValueError("val-ratio + test-ratio must be < 1.0")


def load_image(image_path: Path, width: int, height: int) -> np.ndarray:
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img = img.resize((width, height), Image.Resampling.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def make_group_split(
    beam_ids: np.ndarray,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> np.ndarray:
    unique_ids = np.unique(beam_ids)
    rng = np.random.default_rng(seed)
    shuffled = unique_ids.copy()
    rng.shuffle(shuffled)

    n_groups = len(shuffled)
    n_test = int(round(n_groups * test_ratio))
    n_val = int(round(n_groups * val_ratio))
    n_test = min(n_test, n_groups)
    n_val = min(n_val, n_groups - n_test)

    test_ids = set(shuffled[:n_test])
    val_ids = set(shuffled[n_test : n_test + n_val])

    split = np.full(shape=(len(beam_ids),), fill_value="train", dtype=object)
    for i, bid in enumerate(beam_ids):
        if bid in test_ids:
            split[i] = "test"
        elif bid in val_ids:
            split[i] = "val"
    return split


def make_leave_one_group_out_folds(beam_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unique_ids = np.unique(beam_ids)
    sorted_ids = np.sort(unique_ids)
    group_to_fold = {group_id: i for i, group_id in enumerate(sorted_ids.tolist())}
    fold_id = np.array([group_to_fold[bid] for bid in beam_ids], dtype=np.int16)
    return fold_id, sorted_ids


def main() -> None:
    args = parse_args()
    validate_args(args)

    if not args.csv.exists():
        raise FileNotFoundError(f"CSV not found: {args.csv}")

    df = pd.read_csv(args.csv)
    required = ["image_path", "beam_id", *NUMERIC_COLUMNS, *TARGET_COLUMNS]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    image_paths = df["image_path"].astype(str).to_numpy()
    beam_ids = df["beam_id"].astype(str).to_numpy()
    x_num = df[NUMERIC_COLUMNS].astype(np.float32).to_numpy()
    y = df[TARGET_COLUMNS].astype(np.float32).to_numpy()

    x_img = np.empty((len(df), args.height, args.width, 3), dtype=np.float32)
    for i, rel_path in enumerate(image_paths):
        abs_path = ROOT / rel_path
        if not abs_path.exists():
            raise FileNotFoundError(f"Missing image: {abs_path}")
        x_img[i] = load_image(abs_path, width=args.width, height=args.height)

    if args.split_strategy == "leave-one-group-out":
        fold_id, fold_group_ids = make_leave_one_group_out_folds(beam_ids=beam_ids)
        split = np.array([f"fold_{f}" for f in fold_id], dtype=object)
        split_counts: dict[str, int] = {}
        group_counts: dict[str, int] = {}
        for f_idx, group_id in enumerate(fold_group_ids.tolist()):
            key = f"fold_{f_idx}"
            split_counts[key] = int(np.sum(fold_id == f_idx))
            group_counts[key] = 1
    else:
        split = make_group_split(
            beam_ids=beam_ids,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )
        fold_id = np.full(shape=(len(df),), fill_value=-1, dtype=np.int16)
        fold_group_ids = np.array([], dtype=object)
        split_counts = {name: int(np.sum(split == name)) for name in ["train", "val", "test"]}
        group_counts = {
            name: int(len(np.unique(beam_ids[split == name]))) for name in ["train", "val", "test"]
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        X_img=x_img,
        X_num=x_num,
        Y=y,
        beam_id=beam_ids,
        image_path=image_paths,
        split=split,
        fold_id=fold_id,
        fold_group_ids=fold_group_ids,
        numeric_columns=np.array(NUMERIC_COLUMNS, dtype=object),
        target_columns=np.array(TARGET_COLUMNS, dtype=object),
    )

    metadata = {
        "rows": int(len(df)),
        "image_shape": [int(args.height), int(args.width), 3],
        "numeric_columns": NUMERIC_COLUMNS,
        "target_columns": TARGET_COLUMNS,
        "split_strategy": args.split_strategy,
        "split_counts": split_counts,
        "split_group_counts": group_counts,
        "seed": int(args.seed),
        "val_ratio": float(args.val_ratio),
        "test_ratio": float(args.test_ratio),
    }
    if args.split_strategy == "leave-one-group-out":
        metadata["num_folds"] = int(len(fold_group_ids))
        metadata["fold_to_group"] = {
            f"fold_{i}": str(group_id) for i, group_id in enumerate(fold_group_ids.tolist())
        }

    args.meta.parent.mkdir(parents=True, exist_ok=True)
    args.meta.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Created NPZ: {args.out}")
    print(f"Created metadata: {args.meta}")
    print(f"Rows: {metadata['rows']}")
    print(f"Split rows: {split_counts}")
    print(f"Split unique beam_ids: {group_counts}")
    if args.split_strategy == "leave-one-group-out":
        print(f"Number of folds: {metadata['num_folds']}")


if __name__ == "__main__":
    main()

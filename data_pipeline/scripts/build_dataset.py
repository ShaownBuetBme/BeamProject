from __future__ import annotations

from pathlib import Path
import csv
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = PROJECT_ROOT / "data_pipeline" / "raw"
IMAGE_ROOT = RAW_ROOT / "Beam Image-12"
EXCEL_PATH = RAW_ROOT / "Beam Data.xlsx"
OUTPUT_DIR = PROJECT_ROOT / "data_pipeline" / "artifacts"
DATASET_CSV = OUTPUT_DIR / "beam_dataset.csv"
DROPPED_CSV = OUTPUT_DIR / "beam_dropped_log.csv"


REQUIRED_COLUMNS = [
    "Beam ID",
    "Beam Width",
    "Beam Depth",
    "WA Content",
    "Load Capacity (KN)",
    "Max. Deflection (mm)",
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def fail(message: str) -> None:
    print(f"ERROR: {message}")
    sys.exit(1)


def resolve_excel_beam_id(folder_beam_id: str, available_ids: set[str]) -> tuple[str | None, str]:
    if folder_beam_id in available_ids:
        return folder_beam_id, "exact"

    return None, "missing"


def main() -> None:
    if not EXCEL_PATH.exists():
        fail(f"Excel file not found: {EXCEL_PATH}")
    if not IMAGE_ROOT.exists():
        fail(f"Image root folder not found: {IMAGE_ROOT}")

    df = pd.read_excel(EXCEL_PATH)
    missing_columns = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_columns:
        fail(f"Missing required Excel columns: {missing_columns}")

    df = df[REQUIRED_COLUMNS].copy()
    df["Beam ID"] = df["Beam ID"].astype(str).str.strip()

    duplicated_ids = sorted(df[df["Beam ID"].duplicated()]["Beam ID"].unique().tolist())
    if duplicated_ids:
        fail(f"Duplicate Beam ID values found in Excel: {duplicated_ids}")

    row_map = df.set_index("Beam ID").to_dict(orient="index")
    available_ids = set(row_map.keys())

    records: list[dict[str, object]] = []
    dropped: list[dict[str, str]] = []

    folders = sorted([p for p in IMAGE_ROOT.iterdir() if p.is_dir()])
    for folder in folders:
        beam_id = folder.name.strip()
        excel_beam_id, match_mode = resolve_excel_beam_id(beam_id, available_ids)
        if excel_beam_id is None:
            dropped.append(
                {
                    "reason": "missing_beam_id_in_excel",
                    "beam_id": beam_id,
                    "image_path": "",
                }
            )
            continue

        excel_row = row_map[excel_beam_id]

        images = sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS])
        if not images:
            dropped.append(
                {
                    "reason": "no_images_found_in_folder",
                    "beam_id": beam_id,
                    "image_path": "",
                }
            )
            continue

        for image_path in images:
            rel_image_path = image_path.relative_to(PROJECT_ROOT).as_posix()
            records.append(
                {
                    "image_path": rel_image_path,
                    "beam_id": beam_id,
                    "excel_beam_id": excel_beam_id,
                    "id_match_mode": match_mode,
                    "beam_width": excel_row["Beam Width"],
                    "beam_depth": excel_row["Beam Depth"],
                    "wa_content": excel_row["WA Content"],
                    "target_load_capacity_kn": excel_row["Load Capacity (KN)"],
                    "target_max_deflection_mm": excel_row["Max. Deflection (mm)"],
                }
            )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset_columns = [
        "image_path",
        "beam_id",
        "excel_beam_id",
        "id_match_mode",
        "beam_width",
        "beam_depth",
        "wa_content",
        "target_load_capacity_kn",
        "target_max_deflection_mm",
    ]
    dropped_columns = ["reason", "beam_id", "image_path"]

    with DATASET_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=dataset_columns)
        writer.writeheader()
        writer.writerows(records)

    with DROPPED_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=dropped_columns)
        writer.writeheader()
        writer.writerows(dropped)

    print(f"Created dataset: {DATASET_CSV}")
    print(f"Created dropped log: {DROPPED_CSV}")
    print(f"Total dataset rows: {len(records)}")
    print(f"Dropped entries: {len(dropped)}")


if __name__ == "__main__":
    main()

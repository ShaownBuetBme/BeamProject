from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


def make_run_dir(base_dir: str | Path, experiment_name: str) -> Path:
    base_dir = Path(base_dir)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / f"{stamp}_{experiment_name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_json(data: dict, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def write_csv_dicts(rows: list[dict], out_path: str | Path) -> None:
    import csv

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        out_path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

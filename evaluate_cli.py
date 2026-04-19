from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect an experiment run summary")
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_path = args.run_dir / "metrics_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"metrics_summary.json not found under: {args.run_dir}")

    data = json.loads(summary_path.read_text(encoding="utf-8"))
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()

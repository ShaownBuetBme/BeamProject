from __future__ import annotations

from collections import Counter
from pathlib import Path
import csv


ROOT = Path(__file__).resolve().parent
IMAGE_ROOT = ROOT / "Beam Image-12"
OUTPUT_DIR = ROOT / "dataset"
OUTPUT_CSV = OUTPUT_DIR / "folder_image_stats.csv"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def kb(value_in_bytes: int) -> float:
    return value_in_bytes / 1024.0


def mb(value_in_bytes: int) -> float:
    return value_in_bytes / (1024.0 * 1024.0)


def main() -> None:
    if not IMAGE_ROOT.exists():
        raise FileNotFoundError(f"Image folder not found: {IMAGE_ROOT}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    total_images = 0
    total_bytes = 0

    folders = sorted([p for p in IMAGE_ROOT.iterdir() if p.is_dir()])
    for folder in folders:
        files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
        sizes = [p.stat().st_size for p in files]
        ext_counts = Counter(p.suffix.lower() for p in files)

        image_count = len(files)
        folder_bytes = sum(sizes)
        total_images += image_count
        total_bytes += folder_bytes

        rows.append(
            {
                "folder": folder.name,
                "image_count": image_count,
                "total_size_mb": round(mb(folder_bytes), 3),
                "avg_size_kb": round(kb(folder_bytes) / image_count, 3) if image_count else 0.0,
                "min_size_kb": round(kb(min(sizes)), 3) if sizes else 0.0,
                "max_size_kb": round(kb(max(sizes)), 3) if sizes else 0.0,
                "jpeg_count": ext_counts.get(".jpeg", 0) + ext_counts.get(".jpg", 0),
                "png_count": ext_counts.get(".png", 0),
                "bmp_count": ext_counts.get(".bmp", 0),
                "tif_tiff_count": ext_counts.get(".tif", 0) + ext_counts.get(".tiff", 0),
                "webp_count": ext_counts.get(".webp", 0),
            }
        )

    rows.append(
        {
            "folder": "__TOTAL__",
            "image_count": total_images,
            "total_size_mb": round(mb(total_bytes), 3),
            "avg_size_kb": round(kb(total_bytes) / total_images, 3) if total_images else 0.0,
            "min_size_kb": "",
            "max_size_kb": "",
            "jpeg_count": "",
            "png_count": "",
            "bmp_count": "",
            "tif_tiff_count": "",
            "webp_count": "",
        }
    )

    fieldnames = [
        "folder",
        "image_count",
        "total_size_mb",
        "avg_size_kb",
        "min_size_kb",
        "max_size_kb",
        "jpeg_count",
        "png_count",
        "bmp_count",
        "tif_tiff_count",
        "webp_count",
    ]

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Created: {OUTPUT_CSV}")
    print(f"Folders processed: {len(folders)}")
    print(f"Total images: {total_images}")


if __name__ == "__main__":
    main()

"""
Prepare raw TartanAir downloads for DVD's TartanAir video loader.

DVD expects:
- image folders named like `image_lcam_front` containing `.png` frames
- depth folders named like `depth_lcam_front` containing `.npy` depth frames

This script:
1. finds TartanAir zip files under a source root
2. extracts them into a destination root
3. converts depth PNG frames into float32 `.npy` files

The source root can be the raw copied download folder, for example:
    /path/to/tartanair

The destination root is what you should point DVD's `train_data_dir_ttr_vid` at.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import zipfile

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare raw TartanAir zips for DVD training."
    )
    parser.add_argument(
        "--src-root",
        required=True,
        help="Root directory containing raw TartanAir zip files.",
    )
    parser.add_argument(
        "--dst-root",
        required=True,
        help="Output directory for extracted DVD-ready TartanAir data.",
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Skip zip extraction and only convert depth PNGs already in dst-root.",
    )
    parser.add_argument(
        "--keep-depth-png",
        action="store_true",
        help="Keep original depth PNGs after writing `.npy` files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing `.npy` depth files.",
    )
    return parser.parse_args()


def find_zip_files(src_root: Path) -> list[Path]:
    return sorted(src_root.rglob("*.zip"))


def extract_archives(zip_files: list[Path], dst_root: Path) -> None:
    for zip_path in zip_files:
        print(f"Extracting {zip_path}")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dst_root)


def decode_tartanair_depth_png(depth_png: Path) -> np.ndarray:
    with Image.open(depth_png) as img:
        rgba = np.asarray(img, dtype=np.uint8)

    if rgba.ndim != 3 or rgba.shape[2] != 4:
        raise ValueError(f"Expected RGBA depth PNG, got shape {rgba.shape} for {depth_png}")

    rgba = np.ascontiguousarray(rgba)
    height, width, _ = rgba.shape
    depth = rgba.view("<f4").reshape(height, width)
    return depth.copy()


def convert_depth_dirs(dst_root: Path, keep_depth_png: bool, force: bool) -> tuple[int, int]:
    converted = 0
    skipped = 0

    for depth_dir in sorted(p for p in dst_root.rglob("depth_*") if p.is_dir()):
        image_dir = depth_dir.parent / depth_dir.name.replace("depth_", "image_", 1)
        if not image_dir.is_dir():
            print(f"Skipping {depth_dir}: missing sibling image dir {image_dir}")
            continue

        for depth_png in sorted(depth_dir.glob("*.png")):
            depth_npy = depth_png.with_suffix(".npy")
            if depth_npy.exists() and not force:
                skipped += 1
                continue

            depth = decode_tartanair_depth_png(depth_png)
            np.save(depth_npy, depth)
            converted += 1

            if not keep_depth_png:
                depth_png.unlink()

    return converted, skipped


def main() -> int:
    args = parse_args()
    src_root = Path(args.src_root)
    dst_root = Path(args.dst_root)

    if not src_root.exists():
        print(f"Source root does not exist: {src_root}", file=sys.stderr)
        return 1

    dst_root.mkdir(parents=True, exist_ok=True)

    if not args.skip_extract:
        zip_files = find_zip_files(src_root)
        if not zip_files:
            print(f"No zip files found under {src_root}", file=sys.stderr)
            return 1
        extract_archives(zip_files, dst_root)

    converted, skipped = convert_depth_dirs(
        dst_root, keep_depth_png=args.keep_depth_png, force=args.force
    )

    print(f"Converted depth PNGs to NPY: {converted}")
    print(f"Skipped existing depth NPYs: {skipped}")
    print(f"DVD train_data_dir_ttr_vid should point to: {dst_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

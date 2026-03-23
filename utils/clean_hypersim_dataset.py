#!/usr/bin/env python3
"""Clean a Hypersim dataset directory for DVD training.

Walks the ``train/`` split, builds the exact file list used by
``HypersimDataset``, validates every depth sample (nan / inf / non-positive),
and then:

1. Deletes invalid datapoints (depth **and** its accompanying RGB and normal).
2. Deletes files that are not part of any training sample.
3. Removes empty directories left behind.

Usage::

    python utils/clean_hypersim_dataset.py /path/to/hypersim
    python utils/clean_hypersim_dataset.py /path/to/hypersim --dry-run
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import h5py
import numpy as np


# ---------------------------------------------------------------------------
# Depth helpers (mirrors examples/dataset/hypersim_dataset.py)
# ---------------------------------------------------------------------------

def hypersim_distance_to_depth(distance: np.ndarray) -> np.ndarray:
    int_width = 1024
    int_height = 768
    focal = 886.81

    imageplane_x = (
        np.linspace((-0.5 * int_width) + 0.5, (0.5 * int_width) - 0.5, int_width)
        .reshape(1, int_width)
        .repeat(int_height, 0)
        .astype(np.float32)[:, :, None]
    )
    imageplane_y = (
        np.linspace((-0.5 * int_height) + 0.5, (0.5 * int_height) - 0.5, int_height)
        .reshape(int_height, 1)
        .repeat(int_width, 1)
        .astype(np.float32)[:, :, None]
    )
    imageplane_z = np.full([int_height, int_width, 1], focal, np.float32)
    imageplane = np.concatenate([imageplane_x, imageplane_y, imageplane_z], axis=2)
    return distance / np.linalg.norm(imageplane, 2, 2) * focal


def validate_depth(dep_path: Path) -> str | None:
    """Return an error string if the depth sample is invalid, else ``None``."""
    try:
        with h5py.File(dep_path, "r") as f:
            distance = np.array(f["dataset"])
    except Exception as exc:
        return f"failed to read hdf5: {exc}"

    if not np.isfinite(distance).all():
        return "distance contains NaN or inf"

    depth = hypersim_distance_to_depth(distance)
    if not np.isfinite(depth).all():
        return "depth contains NaN or inf after conversion"
    if (depth <= 0).any():
        return "depth contains non-positive values"

    return None


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def discover_samples(train_dir: Path) -> list[tuple[Path, Path, Path]]:
    """Return ``(rgb, depth, normal)`` triples found under *train_dir*.

    Mirrors the discovery logic in ``HypersimDataset.__init__``.
    """
    samples: list[tuple[Path, Path, Path]] = []
    for root, _dirs, files in os.walk(train_dir):
        for fname in files:
            if fname.endswith("tonemap.jpg"):
                img = Path(root) / fname
                dep = Path(
                    str(img)
                    .replace("final_preview", "geometry_hdf5")
                    .replace("tonemap.jpg", "depth_meters.hdf5")
                )
                nor = Path(
                    str(img)
                    .replace("final_preview", "geometry_hdf5")
                    .replace("tonemap.jpg", "normal_cam.hdf5")
                )
                samples.append((img, dep, nor))
    samples.sort()
    return samples


def collect_all_files(train_dir: Path) -> set[Path]:
    """Return every file under *train_dir*."""
    all_files: set[Path] = set()
    for root, _dirs, files in os.walk(train_dir):
        for fname in files:
            all_files.add(Path(root) / fname)
    return all_files


def remove_empty_dirs(root: Path) -> int:
    """Bottom-up removal of empty directories. Returns count removed."""
    removed = 0
    for dirpath, dirnames, filenames in os.walk(root, topdown=False):
        if not filenames and not dirnames:
            try:
                os.rmdir(dirpath)
                removed += 1
            except OSError:
                pass
    return removed


def clean(dataset_root: Path, *, dry_run: bool = False) -> None:
    train_dir = dataset_root / "train"
    if not train_dir.is_dir():
        raise SystemExit(f"ERROR: train directory not found: {train_dir}")

    # 1. Discover samples the way the training loader does.
    samples = discover_samples(train_dir)
    print(f"Found {len(samples)} samples (tonemap.jpg entries)")
    if not samples:
        raise SystemExit("No samples found — nothing to do.")

    # 2. Build set of needed files (only files that actually exist on disk).
    needed_files: set[Path] = set()
    for img, dep, nor in samples:
        needed_files.add(img)
        needed_files.add(dep)
        if nor.is_file():
            needed_files.add(nor)

    # 3. Validate depth and mark invalid samples for deletion.
    invalid_files: set[Path] = set()
    n_invalid = 0
    for img, dep, nor in samples:
        if not dep.is_file():
            reason = "depth file missing"
        else:
            reason = validate_depth(dep)

        if reason is not None:
            n_invalid += 1
            print(f"  INVALID ({reason}): {dep}")
            invalid_files.add(img)
            invalid_files.add(dep)
            invalid_files.add(nor)

    print(f"Invalid samples: {n_invalid} / {len(samples)}")

    # 4. Identify unused files (exist on disk but not needed by any sample).
    all_files = collect_all_files(train_dir)
    unused_files = all_files - needed_files

    # Files to actually delete = invalid sample files (that exist) + unused.
    to_delete = (invalid_files & all_files) | unused_files
    print(f"Unused files:  {len(unused_files)}")
    print(f"Total files to delete: {len(to_delete)}")

    if not to_delete:
        print("Dataset is clean — nothing to delete.")
        return

    if dry_run:
        print("\n[DRY RUN] Would delete the following files:")
        for p in sorted(to_delete):
            print(f"  {p}")
        print(f"\n[DRY RUN] {len(to_delete)} file(s) would be removed.")
        return

    # 5. Delete.
    deleted = 0
    for p in sorted(to_delete):
        try:
            p.unlink()
            deleted += 1
        except FileNotFoundError:
            pass
        except Exception as exc:
            print(f"  WARNING: could not delete {p}: {exc}")

    print(f"Deleted {deleted} file(s).")

    # 6. Remove empty directories.
    n_dirs = remove_empty_dirs(train_dir)
    if n_dirs:
        print(f"Removed {n_dirs} empty director{'y' if n_dirs == 1 else 'ies'}.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean a Hypersim dataset directory for DVD training."
    )
    parser.add_argument(
        "dataset_dir",
        type=Path,
        help="Root of the Hypersim dataset (must contain a train/ subdirectory).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print what would be deleted; do not remove anything.",
    )
    args = parser.parse_args()
    clean(args.dataset_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()

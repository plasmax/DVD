#!/usr/bin/env python3
"""Validate the datasets used by DVD training.

This script is intentionally conservative: it mirrors the core file loading and
sanity checks used by the active training datasets, then reports whether the
configured dataset roots appear safe for a long training run.

By default it checks only datasets with non-zero sampling probability in the
given config. Use `--include-zero-prob` to also validate configured but inactive
datasets.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import h5py
import numpy as np
from omegaconf import OmegaConf
from PIL import Image

try:
    import Imath
    import OpenEXR
except ImportError:
    Imath = None
    OpenEXR = None


ACTIVE_DATASET_ORDER = [
    "hypersim_image",
    "synthhuman_image",
    "infinigen_image",
    "vkitti_image",
    "tartanair_video",
    "vkitti_video",
]


@dataclass
class DatasetReport:
    name: str
    checked: int = 0
    valid: int = 0
    fatal: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: Counter = field(default_factory=Counter)

    def add_error(self, reason: str) -> None:
        self.errors[reason] += 1
        self.checked += 1

    def add_valid(self) -> None:
        self.valid += 1
        self.checked += 1

    @property
    def invalid(self) -> int:
        return self.checked - self.valid

    @property
    def ok(self) -> bool:
        return not self.fatal and self.valid > 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate DVD training datasets.")
    parser.add_argument(
        "--config",
        default="train_config/normal_config/video_config_new.yaml",
        help="Path to the training config yaml.",
    )
    parser.add_argument(
        "--include-zero-prob",
        action="store_true",
        help="Also validate datasets whose sampling weight is zero.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of samples/scenes checked per dataset.",
    )
    return parser.parse_args()


def resolve_repo_path(repo_root: Path, path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def hypersim_distance_to_depth(distance: np.ndarray) -> np.ndarray:
    int_width = 1024
    int_height = 768
    focal = 886.81

    imageplane_x = np.linspace(
        (-0.5 * int_width) + 0.5,
        (0.5 * int_width) - 0.5,
        int_width,
        dtype=np.float32,
    ).reshape(1, int_width).repeat(int_height, 0)[:, :, None]
    imageplane_y = np.linspace(
        (-0.5 * int_height) + 0.5,
        (0.5 * int_height) - 0.5,
        int_height,
        dtype=np.float32,
    ).reshape(int_height, 1).repeat(int_width, 1)[:, :, None]
    imageplane_z = np.full([int_height, int_width, 1], focal, np.float32)
    imageplane = np.concatenate([imageplane_x, imageplane_y, imageplane_z], axis=2)
    return distance / np.linalg.norm(imageplane, 2, 2) * focal


def load_synthhuman_depth_exr(depth_path: Path) -> np.ndarray:
    if OpenEXR is None or Imath is None:
        raise RuntimeError(
            "OpenEXR Python bindings are not installed; cannot validate SynthHuman EXR depth."
        )
    exr_file = OpenEXR.InputFile(str(depth_path))
    try:
        header = exr_file.header()
        data_window = header["dataWindow"]
        width = data_window.max.x - data_window.min.x + 1
        height = data_window.max.y - data_window.min.y + 1
        channels = list(header["channels"].keys())
        preferred = ("Z", "Y", "R", "G", "B")
        channel = next((name for name in preferred if name in channels), sorted(channels)[0])
        pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)
        raw = exr_file.channel(channel, pixel_type)
        return np.frombuffer(raw, dtype=np.float32).reshape(height, width)
    finally:
        exr_file.close()


def sanitize_infinigen_depth(depth: np.ndarray) -> np.ndarray:
    finite_positive = np.isfinite(depth) & (depth > 0)
    if not finite_positive.any():
        raise ValueError("depth has no finite positive values")
    sanitized = depth.astype(np.float32, copy=True)
    fill_value = np.max(sanitized[finite_positive])
    sanitized[~finite_positive] = fill_value
    return sanitized


def validate_hypersim(root: Path, limit: int | None) -> DatasetReport:
    report = DatasetReport("hypersim_image")
    split_dir = root / "train"
    if not split_dir.is_dir():
        report.fatal.append(f"Missing Hypersim train directory: {split_dir}")
        return report

    samples = []
    for current_root, _, files in os.walk(split_dir):
        for file in files:
            if file.endswith("tonemap.jpg"):
                img = Path(current_root) / file
                dep = Path(str(img).replace("final_preview", "geometry_hdf5").replace("tonemap.jpg", "depth_meters.hdf5"))
                samples.append((img, dep))
    samples.sort()
    if limit is not None:
        samples = samples[:limit]
    if not samples:
        report.fatal.append(f"No Hypersim training samples found under {split_dir}")
        return report

    for img_path, dep_path in samples:
        try:
            with Image.open(img_path) as image:
                image.convert("RGB")
            if not dep_path.is_file():
                raise FileNotFoundError(f"missing depth file {dep_path}")
            with h5py.File(dep_path, "r") as handle:
                distance = np.array(handle["dataset"])
            if not np.isfinite(distance).all():
                raise ValueError("distance contains NaN or inf")
            depth = hypersim_distance_to_depth(distance)
            if not np.isfinite(depth).all():
                raise ValueError("depth contains NaN or inf after conversion")
            if (depth <= 0).any():
                raise ValueError("depth contains non-positive values")
            report.add_valid()
        except Exception as exc:
            report.add_error(str(exc))
    if report.valid == 0:
        report.fatal.append("No valid Hypersim samples found.")
    return report


def validate_synthhuman(root: Path, limit: int | None) -> DatasetReport:
    report = DatasetReport("synthhuman_image")
    if not root.is_dir():
        report.fatal.append(f"Missing SynthHuman root: {root}")
        return report
    if OpenEXR is None or Imath is None:
        report.fatal.append(
            "OpenEXR Python bindings are not installed; SynthHuman training will fail."
        )
        return report

    samples = []
    for current_root, _, files in os.walk(root):
        for file in files:
            if file.startswith("rgb_") and file.endswith(".png"):
                img = Path(current_root) / file
                dep = img.with_name(file.replace("rgb_", "depth_", 1)).with_suffix(".exr")
                samples.append((img, dep))
    samples.sort()
    if limit is not None:
        samples = samples[:limit]
    if not samples:
        report.fatal.append(f"No SynthHuman RGB/depth pairs found under {root}")
        return report

    for img_path, dep_path in samples:
        try:
            with Image.open(img_path) as image:
                image.convert("RGB")
            if not dep_path.is_file():
                raise FileNotFoundError(f"missing depth file {dep_path}")
            depth = load_synthhuman_depth_exr(dep_path) / 100.0
            if not np.isfinite(depth).all():
                raise ValueError("depth contains NaN or inf")
            if (depth <= 0).any():
                raise ValueError("depth contains non-positive values")
            report.add_valid()
        except Exception as exc:
            report.add_error(str(exc))
    if report.valid == 0:
        report.fatal.append("No valid SynthHuman samples found.")
    return report


def validate_infinigen(root: Path, limit: int | None) -> DatasetReport:
    report = DatasetReport("infinigen_image")
    if not root.is_dir():
        report.fatal.append(f"Missing Infinigen root: {root}")
        return report

    pairs = []
    for image_path in sorted(root.rglob("Image*.png")):
        depth_name = image_path.name.replace("Image", "Depth", 1).rsplit(".", 1)[0] + ".npy"
        if "/Image/" in str(image_path):
            depth_path = Path(str(image_path).replace("/Image/", "/Depth/")).with_name(depth_name)
        else:
            depth_path = image_path.with_name(depth_name)
        pairs.append((image_path, depth_path))
    if limit is not None:
        pairs = pairs[:limit]
    if not pairs:
        report.fatal.append(f"No Infinigen RGB/depth pairs found under {root}")
        return report

    replaced_nonfinite = 0
    for image_path, depth_path in pairs:
        try:
            with Image.open(image_path) as image:
                image.convert("RGB")
            if not depth_path.is_file():
                raise FileNotFoundError(f"missing depth file {depth_path}")
            depth = np.load(depth_path)
            if depth.ndim != 2:
                raise ValueError(f"expected HxW depth, got shape {depth.shape}")
            if not np.isfinite(depth).all():
                replaced_nonfinite += 1
            depth = sanitize_infinigen_depth(depth)
            report.add_valid()
        except Exception as exc:
            report.add_error(str(exc))
    if replaced_nonfinite:
        report.warnings.append(
            f"{replaced_nonfinite} checked sample(s) contained non-finite depth; validation filled them with the farthest finite depth, matching the training loader."
        )
    if report.valid == 0:
        report.fatal.append("No valid Infinigen samples found.")
    return report


def validate_tartanair(root: Path, min_num_frame: int, limit: int | None) -> DatasetReport:
    report = DatasetReport("tartanair_video")
    if not root.is_dir():
        report.fatal.append(f"Missing TartanAir root: {root}")
        return report

    scenes = []
    for current_root, dirs, _ in os.walk(root):
        for dirname in dirs:
            if dirname.startswith("depth_"):
                depth_dir = Path(current_root) / dirname
                rgb_dir = Path(current_root) / dirname.replace("depth_", "image_", 1)
                scenes.append((rgb_dir, depth_dir))
    scenes.sort()
    if limit is not None:
        scenes = scenes[:limit]
    if not scenes:
        report.fatal.append(f"No TartanAir depth directories found under {root}")
        return report

    for rgb_dir, depth_dir in scenes:
        try:
            if not rgb_dir.is_dir():
                raise FileNotFoundError(f"missing image dir {rgb_dir}")
            rgb_files = sorted(path for path in rgb_dir.iterdir() if path.suffix == ".png")
            depth_files = sorted(path for path in depth_dir.iterdir() if path.suffix == ".npy")
            if len(rgb_files) != len(depth_files):
                raise ValueError(
                    f"rgb/depth count mismatch: {len(rgb_files)} rgb vs {len(depth_files)} depth"
                )
            if len(rgb_files) < min_num_frame:
                raise ValueError(
                    f"not enough frames: need at least {min_num_frame}, found {len(rgb_files)}"
                )
            sample_rgb = cv2.imread(str(rgb_files[0]))
            if sample_rgb is None:
                raise ValueError(f"failed to read image {rgb_files[0]}")
            sample_depth = np.load(depth_files[0])
            if not np.isfinite(sample_depth).all():
                raise ValueError(f"depth contains NaN or inf: {depth_files[0]}")
            report.add_valid()
        except Exception as exc:
            report.add_error(str(exc))
    if report.valid == 0:
        report.fatal.append("No valid TartanAir scenes found.")
    return report


def validate_vkitti_image(root: Path, limit: int | None) -> DatasetReport:
    report = DatasetReport("vkitti_image")
    if not root.is_dir():
        report.fatal.append(f"Missing VKITTI root: {root}")
        return report

    scenes = ["02", "06", "18", "20"]
    conditions = [
        "15-deg-left", "15-deg-right", "30-deg-left", "30-deg-right", "clone",
        "fog", "morning", "overcast", "rain", "sunset",
    ]
    cameras = ["0", "1"]

    pairs = []
    for scene in scenes:
        for condition in conditions:
            for camera in cameras:
                image_dir = root / f"Scene{scene}/{condition}/frames/rgb/Camera_{camera}"
                if not image_dir.is_dir():
                    report.add_error(f"missing image dir {image_dir}")
                    continue
                for image_path in sorted(image_dir.iterdir()):
                    if image_path.suffix != ".jpg":
                        continue
                    depth_path = Path(str(image_path).replace("rgb", "depth").replace(".jpg", ".png"))
                    pairs.append((image_path, depth_path))
    if limit is not None:
        pairs = pairs[:limit]
    if not pairs:
        report.fatal.append(f"No VKITTI image samples found under {root}")
        return report

    for image_path, depth_path in pairs:
        try:
            with Image.open(image_path) as image:
                image.convert("RGB")
            depth = cv2.imread(str(depth_path), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            if depth is None:
                raise ValueError(f"failed to read depth {depth_path}")
            depth = depth / 100.0
            if not np.isfinite(depth).all():
                raise ValueError("depth contains NaN or inf")
            if not (depth > 0).any():
                raise ValueError("depth has no positive values")
            report.add_valid()
        except Exception as exc:
            report.add_error(str(exc))
    if report.valid == 0:
        report.fatal.append("No valid VKITTI image samples found.")
    return report


def validate_vkitti_video(root: Path, min_num_frame: int, limit: int | None) -> DatasetReport:
    report = DatasetReport("vkitti_video")
    if not root.is_dir():
        report.fatal.append(f"Missing VKITTI root: {root}")
        return report

    scenes = ["02", "06", "18", "20"]
    conditions = [
        "15-deg-left", "15-deg-right", "30-deg-left", "30-deg-right", "clone",
        "fog", "morning", "overcast", "rain", "sunset",
    ]
    cameras = ["0", "1"]

    dirs = []
    for scene in scenes:
        for condition in conditions:
            for camera in cameras:
                rgb_dir = root / f"Scene{scene}/{condition}/frames/rgb/Camera_{camera}"
                depth_dir = root / f"Scene{scene}/{condition}/frames/depth/Camera_{camera}"
                dirs.append((rgb_dir, depth_dir))
    if limit is not None:
        dirs = dirs[:limit]
    if not dirs:
        report.fatal.append(f"No VKITTI video directories found under {root}")
        return report

    for rgb_dir, depth_dir in dirs:
        try:
            if not rgb_dir.is_dir():
                raise FileNotFoundError(f"missing rgb dir {rgb_dir}")
            if not depth_dir.is_dir():
                raise FileNotFoundError(f"missing depth dir {depth_dir}")
            rgb_files = sorted(path for path in rgb_dir.iterdir() if path.suffix == ".jpg")
            depth_files = sorted(path for path in depth_dir.iterdir() if path.suffix == ".png")
            if len(rgb_files) != len(depth_files):
                raise ValueError(
                    f"rgb/depth count mismatch: {len(rgb_files)} rgb vs {len(depth_files)} depth"
                )
            if len(rgb_files) < min_num_frame:
                raise ValueError(
                    f"not enough frames: need at least {min_num_frame}, found {len(rgb_files)}"
                )
            if cv2.imread(str(rgb_files[0])) is None:
                raise ValueError(f"failed to read image {rgb_files[0]}")
            sample_depth = cv2.imread(
                str(depth_files[0]), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH
            )
            if sample_depth is None:
                raise ValueError(f"failed to read depth {depth_files[0]}")
            sample_depth = sample_depth / 100.0
            if not np.isfinite(sample_depth).all():
                raise ValueError(f"depth contains NaN or inf: {depth_files[0]}")
            report.add_valid()
        except Exception as exc:
            report.add_error(str(exc))
    if report.valid == 0:
        report.fatal.append("No valid VKITTI video scenes found.")
    return report


def summarize(report: DatasetReport) -> str:
    lines = [
        f"[{report.name}] checked={report.checked} valid={report.valid} invalid={report.invalid}",
    ]
    for fatal in report.fatal:
        lines.append(f"  FATAL: {fatal}")
    for warning in report.warnings:
        lines.append(f"  WARN: {warning}")
    for reason, count in report.errors.most_common(10):
        lines.append(f"  error[{count}]: {reason}")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    cfg = OmegaConf.load(repo_root / args.config)
    probs = list(cfg.get("prob", []))
    active = {
        name for name, prob in zip(ACTIVE_DATASET_ORDER, probs) if prob > 0
    }
    if args.include_zero_prob:
        active = set(ACTIVE_DATASET_ORDER[:len(probs)])

    reports = []
    if "hypersim_image" in active:
        reports.append(
            validate_hypersim(
                resolve_repo_path(repo_root, cfg.train_data_dir_hypersim),
                args.limit,
            )
        )
    if "synthhuman_image" in active:
        reports.append(
            validate_synthhuman(
                resolve_repo_path(repo_root, cfg.train_data_dir_synthhuman),
                args.limit,
            )
        )
    if "infinigen_image" in active:
        reports.append(
            validate_infinigen(
                resolve_repo_path(repo_root, cfg.train_data_dir_infinigen),
                args.limit,
            )
        )
    if "vkitti_image" in active:
        reports.append(
            validate_vkitti_image(
                resolve_repo_path(repo_root, cfg.train_data_dir_vkitti),
                args.limit,
            )
        )
    if "tartanair_video" in active:
        reports.append(
            validate_tartanair(
                resolve_repo_path(repo_root, cfg.train_data_dir_ttr_vid),
                int(cfg.min_num_frame),
                args.limit,
            )
        )
    if "vkitti_video" in active:
        reports.append(
            validate_vkitti_video(
                resolve_repo_path(repo_root, cfg.train_data_dir_vkitti_vid),
                int(cfg.min_num_frame),
                args.limit,
            )
        )

    print(f"Using config: {repo_root / args.config}")
    print(f"Active datasets: {', '.join(sorted(active))}")
    print()
    for report in reports:
        print(summarize(report))
        print()

    fatal = any(report.fatal for report in reports)
    total_valid = sum(report.valid for report in reports)
    if fatal:
        print("Validation result: FATAL")
        return 2
    if total_valid == 0:
        print("Validation result: FATAL (no valid samples found)")
        return 2
    print("Validation result: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
Convert TartanAir/DVD depth `.npy` files into EXR files.

The input depth is written unchanged as float32 to the EXR red channel only.
The script accepts either a single `.npy` file or a directory containing `.npy`
files and converts each file in that directory.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

try:
    import OpenEXR
except ImportError as exc:
    raise SystemExit(
        "OpenEXR Python bindings are required. Install the package that provides `OpenEXR`."
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a TartanAir depth .npy file or directory to EXR."
    )
    parser.add_argument(
        "input_path",
        help="Path to an input depth .npy file or a directory of .npy files.",
    )
    parser.add_argument(
        "--output",
        help="Output EXR path for single-file input. Defaults to the input path with .exr suffix.",
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory for directory input. Defaults to the input directory.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing EXR files.",
    )
    parser.add_argument(
        "--dvd-preprocess",
        action="store_true",
        help=(
            "Apply the same TartanAir depth preprocessing DVD uses for training "
            "(clamp, optional resize, truncated disparity normalization)."
        ),
    )
    parser.add_argument(
        "--norm-type",
        default="trunc_disparity",
        choices=["instnorm", "truncnorm", "perscene_norm", "disparity", "trunc_disparity"],
        help="DVD normalization mode to apply when --dvd-preprocess is set.",
    )
    parser.add_argument(
        "--truncnorm-min",
        type=float,
        default=0.02,
        help="Lower quantile for truncated normalization when --dvd-preprocess is set.",
    )
    parser.add_argument(
        "--d-max",
        type=float,
        default=50.0,
        help="Maximum depth clamp in meters for DVD preprocessing.",
    )
    parser.add_argument(
        "--resize-height",
        type=int,
        default=480,
        help="Target height for DVD preprocessing resize.",
    )
    parser.add_argument(
        "--resize-width",
        type=int,
        default=640,
        help="Target width for DVD preprocessing resize.",
    )
    parser.add_argument(
        "--no-resize",
        action="store_true",
        help="Skip DVD's training-time resize and keep the original resolution.",
    )
    return parser.parse_args()


def validate_depth(depth: np.ndarray, input_path: Path) -> np.ndarray:
    if depth.ndim != 2:
        raise ValueError(f"Expected HxW depth array, got shape {depth.shape} for {input_path}")
    if depth.dtype != np.float32:
        raise ValueError(f"Expected float32 depth array, got {depth.dtype} for {input_path}")
    return np.ascontiguousarray(depth)


def apply_dvd_preprocess(
    depth: np.ndarray,
    *,
    norm_type: str,
    truncnorm_min: float,
    d_max: float,
    resize_height: int,
    resize_width: int,
    do_resize: bool,
) -> np.ndarray:
    depth_t = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0)
    depth_t = torch.clamp(depth_t, 0, d_max)

    if do_resize:
        depth_t = F.interpolate(
            depth_t, size=(resize_height, resize_width), mode="nearest"
        )

    depth_t = depth_t.squeeze(0).squeeze(0)
    truncnorm_max = 1.0 - truncnorm_min

    if norm_type == "instnorm":
        dmin = depth_t.min()
        dmax = depth_t.max()
        depth_norm = (depth_t - dmin) / (dmax - dmin + 1e-5)
    elif norm_type == "truncnorm":
        dmin = torch.quantile(depth_t, truncnorm_min)
        dmax = torch.quantile(depth_t, truncnorm_max)
        depth_norm = (depth_t - dmin) / (dmax - dmin + 1e-5)
    elif norm_type == "perscene_norm":
        depth_norm = depth_t / d_max
    elif norm_type == "disparity":
        safe_depth = torch.clamp(depth_t, min=1e-5)
        depth_norm = 1.0 / safe_depth
    elif norm_type == "trunc_disparity":
        safe_depth = torch.clamp(depth_t, min=1e-5)
        disparity = 1.0 / safe_depth
        disparity_min = torch.quantile(disparity, truncnorm_min)
        disparity_max = torch.quantile(disparity, truncnorm_max)
        depth_norm = (disparity - disparity_min) / (disparity_max - disparity_min + 1e-5)
    else:
        raise ValueError(f"Unsupported norm type: {norm_type}")

    depth_norm = depth_norm.clip(0, 1)
    return depth_norm.to(torch.float32).cpu().numpy()


def save_depth_exr(output_path: Path, depth: np.ndarray) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = {
        "compression": OpenEXR.ZIP_COMPRESSION,
        "type": OpenEXR.scanlineimage,
    }
    channels = {"R": depth}
    with OpenEXR.File(header, channels) as exr_file:
        exr_file.write(str(output_path))


def convert_file(input_path: Path, output_path: Path, args: argparse.Namespace) -> bool:
    if output_path.exists() and not args.force:
        print(f"Skipping existing file: {output_path}")
        return False

    depth = validate_depth(np.load(input_path), input_path)
    if args.dvd_preprocess:
        depth = apply_dvd_preprocess(
            depth,
            norm_type=args.norm_type,
            truncnorm_min=args.truncnorm_min,
            d_max=args.d_max,
            resize_height=args.resize_height,
            resize_width=args.resize_width,
            do_resize=not args.no_resize,
        )
    save_depth_exr(output_path, depth)
    print(f"Saved {output_path}")
    return True


def convert_directory(input_dir: Path, output_dir: Path, args: argparse.Namespace) -> int:
    npy_files = sorted(path for path in input_dir.glob("*.npy") if path.is_file())
    if not npy_files:
        raise ValueError(f"No .npy files found in {input_dir}")

    converted = 0
    for input_path in npy_files:
        output_path = output_dir / input_path.with_suffix(".exr").name
        if convert_file(input_path, output_path, args=args):
            converted += 1
    return converted


def main() -> int:
    args = parse_args()
    input_path = Path(args.input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    if input_path.is_dir():
        if args.output:
            raise ValueError("--output is only valid for single-file input")
        output_dir = Path(args.output_dir) if args.output_dir else input_path
        output_dir.mkdir(parents=True, exist_ok=True)
        converted = convert_directory(input_path, output_dir, args=args)
        print(f"Converted {converted} file(s) from {input_path} to {output_dir}")
        return 0

    if input_path.suffix.lower() != ".npy":
        raise ValueError(f"Expected a .npy file, got {input_path}")
    if args.output_dir:
        raise ValueError("--output-dir is only valid for directory input")

    output_path = Path(args.output) if args.output else input_path.with_suffix(".exr")
    convert_file(input_path, output_path, args=args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

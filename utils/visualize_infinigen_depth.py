"""
Convert Infinigen depth `.npy` files into EXR files for inspection.

This utility is format-focused rather than tied to a DVD dataset loader, since
the current DVD repo does not yet contain an Infinigen dataset path.

By default, it reads the `.npy` depth map and writes it unchanged to the EXR red
channel. Infinigen depth consumers treat `inf` as invalid background, so the
script preserves those values unless you explicitly request replacement.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

try:
    import OpenEXR
except ImportError as exc:
    raise SystemExit(
        "OpenEXR Python bindings are required. Install the package that provides `OpenEXR`."
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert an Infinigen depth .npy file or directory to EXR."
    )
    parser.add_argument(
        "input_path",
        help="Path to an input depth .npy file or a directory containing Infinigen depth maps.",
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
        "--replace-inf-with",
        type=float,
        help="Replace `inf` / `-inf` depth values before writing, for easier downstream viewing.",
    )
    parser.add_argument(
        "--replace-nan-with",
        type=float,
        help="Replace NaN depth values before writing.",
    )
    parser.add_argument(
        "--report-stats",
        action="store_true",
        help="Print summary stats, including counts of inf and NaN values.",
    )
    return parser.parse_args()


def load_depth(input_path: Path) -> np.ndarray:
    depth = np.load(input_path)
    if depth.ndim != 2:
        raise ValueError(f"Expected HxW depth array, got shape {depth.shape} for {input_path}")
    if depth.dtype != np.float32:
        depth = depth.astype(np.float32)
    return np.ascontiguousarray(depth)


def sanitize_depth(depth: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    result = depth.copy()

    if args.replace_nan_with is not None:
        result[np.isnan(result)] = np.float32(args.replace_nan_with)

    if args.replace_inf_with is not None:
        inf_mask = np.isinf(result)
        result[inf_mask] = np.float32(args.replace_inf_with)

    return np.ascontiguousarray(result)


def describe_depth(label: str, depth: np.ndarray) -> str:
    finite = depth[np.isfinite(depth)]
    nan_count = int(np.count_nonzero(np.isnan(depth)))
    pos_inf_count = int(np.count_nonzero(np.isposinf(depth)))
    neg_inf_count = int(np.count_nonzero(np.isneginf(depth)))

    if finite.size == 0:
        finite_stats = "finite=none"
    else:
        percentiles = np.percentile(finite, [0, 2, 50, 98, 100])
        finite_stats = (
            f"finite_min={percentiles[0]:.6g}, p2={percentiles[1]:.6g}, "
            f"median={percentiles[2]:.6g}, p98={percentiles[3]:.6g}, finite_max={percentiles[4]:.6g}"
        )

    return (
        f"{label}: shape={depth.shape}, {finite_stats}, "
        f"nan={nan_count}, pos_inf={pos_inf_count}, neg_inf={neg_inf_count}"
    )


def save_depth_exr(output_path: Path, depth: np.ndarray) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = {
        "compression": OpenEXR.ZIP_COMPRESSION,
        "type": OpenEXR.scanlineimage,
    }
    channels = {"R": np.ascontiguousarray(depth.astype(np.float32))}
    with OpenEXR.File(header, channels) as exr_file:
        exr_file.write(str(output_path))


def convert_file(input_path: Path, output_path: Path, args: argparse.Namespace) -> bool:
    if output_path.exists() and not args.force:
        print(f"Skipping existing file: {output_path}")
        return False

    raw_depth = load_depth(input_path)
    export_depth = sanitize_depth(raw_depth, args)

    save_depth_exr(output_path, export_depth)
    print(f"Saved {output_path}")

    if args.report_stats:
        print(describe_depth("raw_depth", raw_depth))
        if args.replace_inf_with is not None or args.replace_nan_with is not None:
            print(describe_depth("exported", export_depth))

    return True


def convert_directory(input_dir: Path, output_dir: Path, args: argparse.Namespace) -> int:
    npy_files = sorted(
        path
        for path in input_dir.rglob("Depth*.npy")
        if path.is_file()
    )
    if not npy_files:
        npy_files = sorted(path for path in input_dir.rglob("*.npy") if path.is_file())
    if not npy_files:
        raise ValueError(f"No .npy files found under {input_dir}")

    converted = 0
    for input_path in npy_files:
        relative_parent = input_path.parent.relative_to(input_dir)
        output_path = output_dir / relative_parent / input_path.with_suffix(".exr").name
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

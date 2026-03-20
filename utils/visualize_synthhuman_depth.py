"""
Convert SynthHuman depth EXR files into EXR files for inspection.

By default, the script reads the depth EXR the same way DVD does and writes the
metric depth (meters) unchanged to the EXR red channel.

With `--dvd-preprocess`, it applies the same SynthHuman depth preprocessing used
by training in this repo: convert EXR units to meters, optional resize, then
normalize with the configured depth/disparity transform. This mirrors the
`disparity` tensor, not the auxiliary clamped `raw_depth` tensor.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

try:
    import Imath
    import OpenEXR
except ImportError as exc:
    raise SystemExit(
        "OpenEXR Python bindings are required. Install the packages that provide `OpenEXR` and `Imath`."
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a SynthHuman depth .exr file or directory to EXR."
    )
    parser.add_argument(
        "input_path",
        help="Path to an input depth .exr file or a directory of .exr files.",
    )
    parser.add_argument(
        "--output",
        help="Output EXR path for single-file input. Defaults to the input path with .dvd.exr suffix.",
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
            "Apply the same SynthHuman depth preprocessing DVD uses for training "
            "(EXR load, cm->m, optional resize, normalization)."
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
    parser.add_argument(
        "--unit-scale",
        type=float,
        default=0.01,
        help="Scale factor applied after EXR load. DVD uses 0.01 to convert cm to m.",
    )
    parser.add_argument(
        "--perscene-d-max",
        type=float,
        default=65.0,
        help="Maximum depth used only by `perscene_norm`.",
    )
    parser.add_argument(
        "--report-stats",
        action="store_true",
        help="Print summary stats for the raw metric depth and the exported array.",
    )
    parser.add_argument(
        "--raw-depth-output",
        help=(
            "Optional extra EXR path for a clamped raw-depth export matching the dataset's "
            "auxiliary `raw_depth` tensor behavior (resize + clamp to [1e-3, 65])."
        ),
    )
    return parser.parse_args()


def load_synthhuman_depth_exr(input_path: Path) -> np.ndarray:
    exr_file = OpenEXR.InputFile(str(input_path))
    try:
        header = exr_file.header()
        data_window = header["dataWindow"]
        width = data_window.max.x - data_window.min.x + 1
        height = data_window.max.y - data_window.min.y + 1
        channel_map = header["channels"]
        channel_names = list(channel_map.keys())
        preferred = ("Z", "Y", "R", "G", "B")
        channel_name = next(
            (name for name in preferred if name in channel_names),
            sorted(channel_names)[0],
        )
        pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)
        raw = exr_file.channel(channel_name, pixel_type)
        depth = np.frombuffer(raw, dtype=np.float32).reshape(height, width)
        return np.ascontiguousarray(depth)
    finally:
        exr_file.close()


def validate_depth(depth: np.ndarray, input_path: Path) -> np.ndarray:
    if depth.ndim != 2:
        raise ValueError(f"Expected HxW depth array, got shape {depth.shape} for {input_path}")
    if depth.dtype != np.float32:
        raise ValueError(f"Expected float32 depth array, got {depth.dtype} for {input_path}")
    if not np.isfinite(depth).all():
        raise ValueError(f"Depth contains NaN or inf for {input_path}")
    if (depth <= 0).any():
        raise ValueError(f"Depth contains non-positive values for {input_path}")
    return np.ascontiguousarray(depth)


def apply_synthhuman_dvd_preprocess(
    depth: np.ndarray,
    *,
    norm_type: str,
    truncnorm_min: float,
    resize_height: int,
    resize_width: int,
    do_resize: bool,
    perscene_d_max: float,
) -> np.ndarray:
    depth_t = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0)

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
        depth_norm = depth_t / perscene_d_max
    elif norm_type == "disparity":
        disparity = 1.0 / depth_t
        depth_norm = disparity
    elif norm_type == "trunc_disparity":
        disparity = 1.0 / depth_t
        disparity_min = torch.quantile(disparity, truncnorm_min)
        disparity_max = torch.quantile(disparity, truncnorm_max)
        depth_norm = (disparity - disparity_min) / (disparity_max - disparity_min + 1e-5)
    else:
        raise ValueError(f"Unsupported norm type: {norm_type}")

    depth_norm = depth_norm.clip(0, 1)
    return depth_norm.to(torch.float32).cpu().numpy()


def build_raw_depth_export(
    depth: np.ndarray,
    *,
    resize_height: int,
    resize_width: int,
    do_resize: bool,
) -> np.ndarray:
    depth_t = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0)
    if do_resize:
        depth_t = F.interpolate(
            depth_t, size=(resize_height, resize_width), mode="nearest"
        )
    depth_t = depth_t.squeeze(0).squeeze(0)
    depth_t = torch.clamp(depth_t, 1e-3, 65.0)
    return depth_t.to(torch.float32).cpu().numpy()


def describe_depth(label: str, depth: np.ndarray) -> str:
    percentiles = np.percentile(depth, [0, 2, 50, 98, 100])
    return (
        f"{label}: shape={depth.shape}, min={percentiles[0]:.6g}, p2={percentiles[1]:.6g}, "
        f"median={percentiles[2]:.6g}, p98={percentiles[3]:.6g}, max={percentiles[4]:.6g}"
    )


def sentinel_summary(raw_exr: np.ndarray, unit_scale: float) -> str:
    sentinel = np.float32(65504.0)
    sentinel_count = int(np.count_nonzero(raw_exr == sentinel))
    total = int(raw_exr.size)
    ratio = sentinel_count / total if total else 0.0
    return (
        f"sentinel_65504: count={sentinel_count}, frac={ratio:.6%}, "
        f"metric_value={float(sentinel * unit_scale):.6g}m"
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


def default_output_path(input_path: Path, dvd_preprocess: bool) -> Path:
    new_name = input_path.stem + (".dvd.exr" if dvd_preprocess else ".meters.exr")
    return input_path.with_name(new_name)


def convert_file(input_path: Path, output_path: Path, args: argparse.Namespace) -> bool:
    if output_path.exists() and not args.force:
        print(f"Skipping existing file: {output_path}")
        return False

    raw_exr = load_synthhuman_depth_exr(input_path)
    depth_m = validate_depth(raw_exr * args.unit_scale, input_path)

    exported = depth_m
    if args.dvd_preprocess:
        exported = apply_synthhuman_dvd_preprocess(
            depth_m,
            norm_type=args.norm_type,
            truncnorm_min=args.truncnorm_min,
            resize_height=args.resize_height,
            resize_width=args.resize_width,
            do_resize=not args.no_resize,
            perscene_d_max=args.perscene_d_max,
        )

    save_depth_exr(output_path, exported)
    print(f"Saved {output_path}")

    if args.raw_depth_output:
        raw_depth_output = Path(args.raw_depth_output)
        raw_depth = build_raw_depth_export(
            depth_m,
            resize_height=args.resize_height,
            resize_width=args.resize_width,
            do_resize=not args.no_resize,
        )
        save_depth_exr(raw_depth_output, raw_depth)
        print(f"Saved {raw_depth_output}")

    if args.report_stats:
        print(sentinel_summary(raw_exr, args.unit_scale))
        print(describe_depth("raw_depth_m", depth_m))
        print(describe_depth("exported", exported))
        if args.raw_depth_output:
            print(describe_depth("raw_depth_clamped", raw_depth))

    return True


def convert_directory(input_dir: Path, output_dir: Path, args: argparse.Namespace) -> int:
    exr_files = sorted(
        path
        for path in input_dir.rglob("depth_*.exr")
        if path.is_file()
    )
    if not exr_files:
        raise ValueError(f"No depth_*.exr files found under {input_dir}")

    converted = 0
    for input_path in exr_files:
        relative_parent = input_path.parent.relative_to(input_dir)
        output_name = default_output_path(input_path, args.dvd_preprocess).name
        output_path = output_dir / relative_parent / output_name
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
        if args.raw_depth_output:
            raise ValueError("--raw-depth-output is only valid for single-file input")
        output_dir = Path(args.output_dir) if args.output_dir else input_path
        output_dir.mkdir(parents=True, exist_ok=True)
        converted = convert_directory(input_path, output_dir, args=args)
        print(f"Converted {converted} file(s) from {input_path} to {output_dir}")
        return 0

    if input_path.suffix.lower() != ".exr":
        raise ValueError(f"Expected a .exr file, got {input_path}")
    if args.output_dir:
        raise ValueError("--output-dir is only valid for directory input")

    output_path = Path(args.output) if args.output else default_output_path(input_path, args.dvd_preprocess)
    convert_file(input_path, output_path, args=args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
Convert a TartanAir/DVD depth `.npy` file into a viewable PNG.

This is for inspection only. It normalizes depth values for display and does
not preserve metric depth in the output image.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize a TartanAir depth .npy file as a PNG."
    )
    parser.add_argument("input_npy", help="Path to the input depth .npy file.")
    parser.add_argument(
        "--output",
        help="Output PNG path. Defaults to the input path with .png suffix.",
    )
    parser.add_argument(
        "--clip-min",
        type=float,
        default=None,
        help="Optional lower clip bound in depth units.",
    )
    parser.add_argument(
        "--clip-max",
        type=float,
        default=None,
        help="Optional upper clip bound in depth units.",
    )
    parser.add_argument(
        "--percentile-min",
        type=float,
        default=2.0,
        help="Lower percentile used when clip-min is not set.",
    )
    parser.add_argument(
        "--percentile-max",
        type=float,
        default=98.0,
        help="Upper percentile used when clip-max is not set.",
    )
    parser.add_argument(
        "--invert",
        action="store_true",
        help="Invert the grayscale mapping so nearer points appear brighter.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input_npy)
    output_path = Path(args.output) if args.output else input_path.with_suffix(".png")

    depth = np.load(input_path)
    if depth.ndim != 2:
        raise ValueError(f"Expected HxW depth array, got shape {depth.shape}")

    finite = depth[np.isfinite(depth)]
    if finite.size == 0:
        raise ValueError(f"No finite values found in {input_path}")

    clip_min = args.clip_min
    clip_max = args.clip_max
    if clip_min is None:
        clip_min = float(np.percentile(finite, args.percentile_min))
    if clip_max is None:
        clip_max = float(np.percentile(finite, args.percentile_max))

    if clip_max <= clip_min:
        raise ValueError(
            f"Invalid display range: clip_min={clip_min}, clip_max={clip_max}"
        )

    depth_vis = np.clip(depth, clip_min, clip_max)
    depth_vis = (depth_vis - clip_min) / (clip_max - clip_min)
    if args.invert:
        depth_vis = 1.0 - depth_vis

    depth_u8 = np.round(depth_vis * 255.0).astype(np.uint8)
    Image.fromarray(depth_u8, mode="L").save(output_path)

    print(f"Saved visualization to {output_path}")
    print(f"Display range used: min={clip_min:.6f}, max={clip_max:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

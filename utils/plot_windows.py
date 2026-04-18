import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


BEGINNING_PATCH_SIZE = 45


def get_window_index(total_frames, window_size, overlap):
    if total_frames <= window_size:
        return [(0, total_frames)]

    stride = window_size - overlap
    n_windows = math.ceil((total_frames - window_size) / stride) + 1
    if n_windows == 1:
        return [(0, min(window_size, total_frames))]

    span = total_frames - window_size
    starts = [round(i * span / (n_windows - 1)) for i in range(n_windows)]
    return [(s, min(s + window_size, total_frames)) for s in starts]


def get_overlap_regions(windows):
    regions = []
    for i in range(1, len(windows)):
        overlap_start = windows[i][0]
        overlap_end = min(windows[i - 1][1], windows[i][1])
        if overlap_end > overlap_start:
            regions.append((overlap_start, overlap_end))
    return regions


def get_overlap_patch_windows(total_frames, target_windows, patch_margin):
    windows = []
    for start, end in get_overlap_regions(target_windows):
        windows.append(
            (max(0, start - patch_margin), min(total_frames, end + patch_margin))
        )
    return windows


def get_beginning_patch_window(total_frames, patch_size=BEGINNING_PATCH_SIZE):
    if patch_size <= 0:
        return []
    if total_frames < patch_size:
        raise ValueError(
            f"Need at least {patch_size} total frames for the beginning offset patch, "
            f"but got {total_frames}."
        )
    return [(0, patch_size)]


def overlap_lengths(windows):
    return [
        max(0, windows[i - 1][1] - windows[i][0])
        for i in range(1, len(windows))
    ]


def format_patch_coverage(target_windows, patch_windows):
    lines = []
    targets = get_overlap_regions(target_windows)
    for i, ((target_start, target_end), (patch_start, patch_end)) in enumerate(
        zip(targets, patch_windows), start=1
    ):
        covered = patch_start <= target_start and patch_end >= target_end
        status = "covered" if covered else "NOT covered"
        lines.append(
            f"{i}: normal overlap [{target_start}, {target_end}) "
            f"inside patch [{patch_start}, {patch_end}) -> {status}"
        )
    return lines


def format_windows(windows, first_frame):
    lines = []
    for i, (start, end) in enumerate(windows, start=1):
        source_start = first_frame + start
        source_end_exclusive = first_frame + end
        lines.append(
            f"{i}: local [{start}, {end}) len={end - start}, "
            f"source [{source_start}, {source_end_exclusive})"
        )
    return lines


def draw_timeline(
    ax,
    y,
    windows,
    color,
    overlap_color,
    label,
    first_frame,
):
    height = 0.42
    ax.broken_barh(
        [(start, end - start) for start, end in windows],
        (y - height / 2, height),
        facecolors=color,
        edgecolors="black",
        linewidth=1.1,
        alpha=0.86,
    )

    overlap_regions = [
        (start, end - start) for start, end in get_overlap_regions(windows)
    ]

    if overlap_regions:
        ax.broken_barh(
            overlap_regions,
            (y - height / 2, height),
            facecolors=overlap_color,
            edgecolors="black",
            linewidth=1.2,
            hatch="////",
            alpha=0.95,
        )

    for i, (start, end) in enumerate(windows, start=1):
        length = end - start
        center = start + length / 2
        text = f"W{i}\n{length}f"
        if length >= 50:
            text += f"\n{first_frame + start}-{first_frame + end - 1}"
        ax.text(
            center,
            y,
            text,
            ha="center",
            va="center",
            fontsize=8,
            color="black",
        )

    ax.text(
        -5,
        y,
        label,
        ha="right",
        va="center",
        fontsize=11,
        fontweight="bold",
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot DVD inference windows and independent overlap repair patches."
    )
    parser.add_argument("--total_frames", type=int, default=291)
    parser.add_argument("--first_frame", type=int, default=977)
    parser.add_argument("--end_frame", type=int, default=1268)
    parser.add_argument("--overlap", type=int, default=21)
    parser.add_argument("--window_size", type=int, default=81)
    parser.add_argument("--double", action="store_true", default=True)
    parser.add_argument(
        "--patch_margin",
        type=int,
        default=5,
        help="Extra frames on each side of each normal overlap for overlap-patches.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("utils/window_timelines.png"),
        help="Output plot path.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    expected_end = args.first_frame + args.total_frames
    if args.end_frame != expected_end:
        print(
            "Note: using total_frames for plotting. "
            f"first_frame + total_frames = {expected_end}, "
            f"but --end_frame is {args.end_frame}."
        )

    depth_windows = get_window_index(args.total_frames, args.window_size, args.overlap)
    output_windows = depth_windows.copy()
    if args.double:
        beginning_patch_windows = get_beginning_patch_window(args.total_frames)
        overlap_patch_windows = get_overlap_patch_windows(
            args.total_frames, depth_windows, args.patch_margin
        )
        offset_windows = beginning_patch_windows + overlap_patch_windows
    else:
        beginning_patch_windows = []
        overlap_patch_windows = []
        offset_windows = []

    print("Normal pass inference/output windows:")
    print("\n".join(format_windows(depth_windows, args.first_frame)))
    print(f"Normal pass overlaps: {overlap_lengths(depth_windows)}")

    if offset_windows:
        print("\nDouble overlap repair patches:")
        print("\n".join(format_windows(offset_windows, args.first_frame)))
        if beginning_patch_windows:
            print(f"Beginning artifact patch size: {BEGINNING_PATCH_SIZE}")
        print(f"Patch margin: {args.patch_margin}")
        if overlap_patch_windows:
            print("Normal overlap coverage by independent offset patches:")
            print("\n".join(format_patch_coverage(depth_windows, overlap_patch_windows)))

    fig, ax = plt.subplots(figsize=(16, 4.8))
    for start, end in get_overlap_regions(output_windows):
        ax.axvspan(
            start,
            end,
            facecolor="#f05a28",
            edgecolor="#f05a28",
            alpha=0.12,
            linewidth=1.0,
        )

    draw_timeline(
        ax,
        1.3,
        output_windows,
        "#7db7d8",
        "#f05a28",
        "normal pass",
        args.first_frame,
    )
    if offset_windows:
        draw_timeline(
            ax,
            0.45,
            offset_windows,
            "#f2c46d",
            "#4b49ac",
            "offset patches",
            args.first_frame,
        )

    ax.set_xlim(0, args.total_frames)
    ax.set_ylim(0, 1.85)
    ax.set_yticks([])
    ax.set_xlabel(
        f"Local frame index, source frames [{args.first_frame}, {expected_end})"
    )
    ax.set_title(
        "DVD inference windows: "
        f"T={args.total_frames}, window_size={args.window_size}, "
        f"overlap={args.overlap}, double={args.double}, "
        f"patch_margin={args.patch_margin}, "
        f"beginning_patch_size={BEGINNING_PATCH_SIZE}"
    )
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    legend_handles = [
        Patch(facecolor="#7db7d8", edgecolor="black", label="normal pass"),
        Patch(
            facecolor="#f2c46d",
            edgecolor="black",
            label="offset patches",
        ),
        Patch(
            facecolor="#f05a28",
            edgecolor="black",
            hatch="////",
            label="normal overlap",
        ),
        Patch(
            facecolor="#f05a28",
            edgecolor="#f05a28",
            alpha=0.12,
            label="normal overlap projection",
        ),
    ]
    ax.legend(handles=legend_handles, loc="upper right")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    print(f"\nSaved plot to {args.output}")


if __name__ == "__main__":
    main()

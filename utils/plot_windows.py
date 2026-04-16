import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


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


def get_trimmed_window_plan(total_frames, window_size, overlap, trim_frames):
    if trim_frames == 0:
        inference_windows = get_window_index(total_frames, window_size, overlap)
        return inference_windows, inference_windows.copy()

    usable_frames = window_size - trim_frames
    stride = usable_frames - overlap
    if usable_frames <= 0:
        raise ValueError(
            "trim_frames must be smaller than window_size, got "
            f"trim_frames={trim_frames}, window_size={window_size}"
        )
    if stride <= 0:
        raise ValueError(
            "overlap must be smaller than the usable frames after trimming, got "
            f"overlap={overlap}, usable_frames={usable_frames}"
        )

    if total_frames <= window_size:
        return [(0, total_frames)], [(0, total_frames)]

    n_windows = math.ceil((total_frames - window_size) / stride) + 1
    span = total_frames - window_size
    starts = [0] if n_windows == 1 else [
        round(i * span / (n_windows - 1)) for i in range(n_windows)
    ]

    inference_windows = []
    output_windows = []
    for i, start in enumerate(starts):
        end = min(start + window_size, total_frames)
        inference_windows.append((start, end))
        if i == 0:
            output_windows.append((start, end))
        else:
            output_windows.append((start + trim_frames, end))

    return inference_windows, output_windows


def get_double_offset_windows(total_frames, window_size, overlap):
    offset = window_size // 2
    if offset <= 0 or offset >= total_frames:
        return []

    windows = [(0, offset)]
    remaining_frames = total_frames - offset
    if remaining_frames > 0:
        later = get_window_index(remaining_frames, window_size, overlap)
        windows.extend(
            (start + offset, min(end + offset, total_frames))
            for start, end in later
        )
    return windows


def get_overlap_regions(windows):
    regions = []
    for i in range(1, len(windows)):
        overlap_start = windows[i][0]
        overlap_end = min(windows[i - 1][1], windows[i][1])
        if overlap_end > overlap_start:
            regions.append((overlap_start, overlap_end))
    return regions


def get_centered_overlap_windows(total_frames, window_size, target_windows):
    windows = []
    seen = set()
    for start, end in get_overlap_regions(target_windows):
        center = (start + end) / 2
        window_start = int(math.floor(center - window_size / 2 + 0.5))
        window_start = max(0, min(window_start, total_frames - window_size))
        window = (window_start, min(window_start + window_size, total_frames))
        if window not in seen:
            windows.append(window)
            seen.add(window)
    return windows


def choose_center_band_width(window_size, overlap, target_windows):
    max_target_len = max(
        (end - start for start, end in get_overlap_regions(target_windows)), default=0
    )
    natural_center_len = window_size - 2 * overlap
    return max(max_target_len, natural_center_len, 1)


def get_covering_offset_windows(
    total_frames, window_size, center_band_width, target_windows, coverage_margin
):
    targets = get_overlap_regions(target_windows)
    if not targets:
        return []

    edge_margin = (window_size - center_band_width) / 2
    max_start = total_frames - window_size
    feasible_starts = []
    for target_start, target_end in targets:
        # The target must fit entirely inside the central low-artifact band:
        # [window_start + edge_margin, window_end - edge_margin), with an
        # extra margin so coverage is not sitting exactly on a boundary.
        start_min = math.ceil(
            target_end + coverage_margin - (window_size - edge_margin)
        )
        start_max = math.floor(target_start - coverage_margin - edge_margin)
        start_min = max(0, start_min)
        start_max = min(max_start, start_max)
        if start_min > start_max:
            target_len = target_end - target_start
            raise ValueError(
                "Cannot cover normal overlap with the requested center band: "
                f"target=({target_start}, {target_end}), target_len={target_len}, "
                f"center_band_width={center_band_width}, "
                f"coverage_margin={coverage_margin}"
            )
        feasible_starts.append(list(range(start_min, start_max + 1)))

    states = {}
    first_target = targets[0]
    first_center = (first_target[0] + first_target[1]) / 2
    for start in feasible_starts[0]:
        band_center = start + window_size / 2
        center_penalty = abs(band_center - first_center)
        states[start] = ((0, 0, center_penalty), [start])

    for target, starts in zip(targets[1:], feasible_starts[1:]):
        target_center = (target[0] + target[1]) / 2
        next_states = {}
        for start in starts:
            band_center = start + window_size / 2
            center_penalty = abs(band_center - target_center)
            best = None
            for prev_start, (prev_cost, prev_path) in states.items():
                overlap_len = max(0, prev_start + window_size - start)
                total_overlap = prev_cost[0] + overlap_len
                max_overlap = max(prev_cost[1], overlap_len)
                penalty = prev_cost[2] + center_penalty
                cost = (total_overlap, max_overlap, penalty)
                if best is None or cost < best[0]:
                    best = (cost, prev_path + [start])
            next_states[start] = best
        states = next_states

    _, starts = min(states.values(), key=lambda item: item[0])
    return [(start, start + window_size) for start in starts]


def get_overlap_patch_windows(total_frames, target_windows, patch_margin):
    windows = []
    for start, end in get_overlap_regions(target_windows):
        windows.append((max(0, start - patch_margin), min(total_frames, end + patch_margin)))
    return windows


def overlap_lengths(windows):
    return [
        max(0, windows[i - 1][1] - windows[i][0])
        for i in range(1, len(windows))
    ]


def get_center_band_regions(windows, center_band_width):
    regions = []
    for start, end in windows:
        center = (start + end) / 2
        band_start = max(start, center - center_band_width / 2)
        band_end = min(end, center + center_band_width / 2)
        regions.append((band_start, band_end))
    return regions


def format_overlap_coverage(target_windows, offset_windows, center_band_width):
    center_bands = get_center_band_regions(offset_windows, center_band_width)
    lines = []
    for i, (target_start, target_end) in enumerate(
        get_overlap_regions(target_windows), start=1
    ):
        covered_by = [
            j
            for j, (band_start, band_end) in enumerate(center_bands, start=1)
            if band_start <= target_start and band_end >= target_end
        ]
        status = "covered" if covered_by else "NOT covered"
        suffix = f" by offset W{covered_by[0]}" if covered_by else ""
        lines.append(
            f"{i}: local [{target_start}, {target_end}) "
            f"len={target_end - target_start} -> {status}{suffix}"
        )
    return lines


def format_margin_coverage(
    target_windows, offset_windows, center_band_width, coverage_margin
):
    center_bands = get_center_band_regions(offset_windows, center_band_width)
    lines = []
    for i, (target_start, target_end) in enumerate(
        get_overlap_regions(target_windows), start=1
    ):
        target_start_with_margin = target_start - coverage_margin
        target_end_with_margin = target_end + coverage_margin
        covered_by = [
            j
            for j, (band_start, band_end) in enumerate(center_bands, start=1)
            if band_start <= target_start_with_margin
            and band_end >= target_end_with_margin
        ]
        status = "covered" if covered_by else "NOT covered"
        suffix = f" by offset W{covered_by[0]}" if covered_by else ""
        lines.append(
            f"{i}: local [{target_start_with_margin}, {target_end_with_margin}) "
            f"target+margin -> {status}{suffix}"
        )
    return lines


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
    center_band_width=None,
    center_band_color=None,
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

    if center_band_width is not None and center_band_color is not None:
        center_regions = [
            (start, end - start)
            for start, end in get_center_band_regions(windows, center_band_width)
        ]
        ax.broken_barh(
            center_regions,
            (y - height / 2, height),
            facecolors=center_band_color,
            edgecolors="black",
            linewidth=0.9,
            alpha=0.92,
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
    parser.add_argument(
        "--offset_overlap",
        type=int,
        default=None,
        help=(
            "Overlap/edge width used to define the reliable center band for "
            "offset windows. Defaults to half of --overlap."
        ),
    )
    parser.add_argument("--window_size", type=int, default=81)
    parser.add_argument("--double", action="store_true", default=True)
    parser.add_argument(
        "--double_strategy",
        choices=(
            "legacy-offset",
            "centered-overlaps",
            "cover-overlaps",
            "overlap-patches",
        ),
        default="overlap-patches",
        help="How to place the double-pass windows in the plot.",
    )
    parser.add_argument(
        "--center_band_width",
        type=int,
        default=None,
        help=(
            "Reliable center width for offset windows. Defaults to "
            "max(max normal overlap, window_size - 2 * overlap)."
        ),
    )
    parser.add_argument(
        "--coverage_margin",
        type=int,
        default=2,
        help="Extra frames required on each side of each normal overlap inside the offset center band.",
    )
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
    offset_overlap = args.offset_overlap
    if offset_overlap is None:
        offset_overlap = max(0, args.overlap // 2)

    center_band_width = args.center_band_width
    if center_band_width is None:
        center_band_width = choose_center_band_width(
            args.window_size, offset_overlap, depth_windows
        )

    if args.double and args.double_strategy == "legacy-offset":
        offset_windows = get_double_offset_windows(
            args.total_frames, args.window_size, args.overlap
        )
    elif args.double and args.double_strategy == "centered-overlaps":
        offset_windows = get_centered_overlap_windows(
            args.total_frames, args.window_size, depth_windows
        )
    elif args.double and args.double_strategy == "cover-overlaps":
        offset_windows = get_covering_offset_windows(
            args.total_frames,
            args.window_size,
            center_band_width,
            depth_windows,
            args.coverage_margin,
        )
    elif args.double and args.double_strategy == "overlap-patches":
        offset_windows = get_overlap_patch_windows(
            args.total_frames, depth_windows, args.patch_margin
        )
    else:
        offset_windows = []

    print("Normal pass inference/output windows:")
    print("\n".join(format_windows(depth_windows, args.first_frame)))
    print(f"Normal pass overlaps: {overlap_lengths(depth_windows)}")

    if offset_windows:
        print(f"\nDouble offset pass windows ({args.double_strategy}):")
        print("\n".join(format_windows(offset_windows, args.first_frame)))
        if args.double_strategy == "overlap-patches":
            print(f"Patch margin: {args.patch_margin}")
            print("Normal overlap coverage by independent offset patches:")
            print(
                "\n".join(format_patch_coverage(depth_windows, offset_windows))
            )
        else:
            print(f"Offset pass overlaps: {overlap_lengths(offset_windows)}")
            print(f"Offset overlap setting: {offset_overlap}")
            print(f"Offset center band width: {center_band_width}")
            print("Normal overlap coverage by offset center bands:")
            print(
                "\n".join(
                    format_overlap_coverage(
                        depth_windows, offset_windows, center_band_width
                    )
                )
            )
            if args.coverage_margin > 0:
                print(f"Normal overlap coverage with +/-{args.coverage_margin} frame margin:")
                print(
                    "\n".join(
                        format_margin_coverage(
                            depth_windows,
                            offset_windows,
                            center_band_width,
                            args.coverage_margin,
                        )
                    )
                )

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
        offset_center_band_width = None
        if args.double_strategy != "overlap-patches":
            offset_center_band_width = center_band_width
        draw_timeline(
            ax,
            0.45,
            offset_windows,
            "#f2c46d",
            "#4b49ac",
            "offset patches" if args.double_strategy == "overlap-patches" else "offset pass",
            args.first_frame,
            center_band_width=offset_center_band_width,
            center_band_color="#56b870",
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
        f"strategy={args.double_strategy}, patch_margin={args.patch_margin}"
    )
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    legend_handles = [
        Patch(facecolor="#7db7d8", edgecolor="black", label="normal pass"),
        Patch(
            facecolor="#f2c46d",
            edgecolor="black",
            label="offset patches"
            if args.double_strategy == "overlap-patches"
            else "offset pass",
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
    if args.double_strategy != "overlap-patches":
        legend_handles.extend(
            [
                Patch(
                    facecolor="#4b49ac",
                    edgecolor="black",
                    hatch="////",
                    label="offset overlap",
                ),
                Patch(
                    facecolor="#56b870",
                    edgecolor="black",
                    label="offset center band",
                ),
            ]
        )
    ax.legend(handles=legend_handles, loc="upper right")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    print(f"\nSaved plot to {args.output}")


if __name__ == "__main__":
    main()

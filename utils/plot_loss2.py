import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatterSciNotation, LogLocator


def positive_int(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return ivalue


def ema_alpha(value):
    fvalue = float(value)
    if not 0.0 < fvalue <= 1.0:
        raise argparse.ArgumentTypeError("must be in the range (0, 1]")
    return fvalue


def parse_args():
    parser = argparse.ArgumentParser(description="Plot training loss from a CSV file.")
    parser.add_argument("csv_path", type=Path, help="Path to loss_log.csv")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output image path. Defaults to <csv_dir>/loss_plot.png",
    )
    parser.add_argument(
        "--log",
        action="store_true",
        help="Use a logarithmic scale for the y-axis",
    )
    smoothing_group = parser.add_mutually_exclusive_group()
    smoothing_group.add_argument(
        "--smooth-window",
        type=positive_int,
        metavar="N",
        help="Apply a centered moving average with window size N",
    )
    smoothing_group.add_argument(
        "--smooth-ema",
        type=ema_alpha,
        metavar="ALPHA",
        help="Apply exponential moving average smoothing with weight ALPHA",
    )
    return parser.parse_args()


def read_loss_csv(csv_path: Path):
    steps = []
    depth_losses = []
    grad_losses = []

    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        required_fields = {"global_step", "depth_loss", "grad_loss"}
        missing_fields = required_fields - set(reader.fieldnames or [])
        if missing_fields:
            missing = ", ".join(sorted(missing_fields))
            raise ValueError(f"Missing required CSV columns: {missing}")

        for row in reader:
            try:
                steps.append(int(row["global_step"]))
                depth_losses.append(float(row["depth_loss"]))
                grad_losses.append(float(row["grad_loss"]))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid row in {csv_path}: {row}") from exc

    if not steps:
        raise ValueError(f"No loss rows found in {csv_path}")

    rows = sorted(zip(steps, depth_losses, grad_losses))
    steps, depth_losses, grad_losses = zip(*rows)
    return steps, depth_losses, grad_losses


def moving_average(values, window_size):
    half_window = window_size // 2
    smoothed = []
    for idx in range(len(values)):
        start = max(0, idx - half_window)
        end = min(len(values), idx + half_window + 1)
        smoothed.append(sum(values[start:end]) / (end - start))
    return smoothed


def exponential_moving_average(values, alpha):
    smoothed = [values[0]]
    for value in values[1:]:
        smoothed.append(alpha * value + (1.0 - alpha) * smoothed[-1])
    return smoothed


def maybe_smooth(values, args):
    if args.smooth_window:
        return moving_average(values, args.smooth_window), f"moving average (window={args.smooth_window})"
    if args.smooth_ema:
        return exponential_moving_average(values, args.smooth_ema), f"EMA (alpha={args.smooth_ema})"
    return list(values), None


def main():
    args = parse_args()
    csv_path = args.csv_path
    output_path = args.output or csv_path.with_name("loss_plot.png")

    steps, depth_losses, grad_losses = read_loss_csv(csv_path)
    depth_losses, smoothing_label = maybe_smooth(depth_losses, args)
    grad_losses, _ = maybe_smooth(grad_losses, args)

    plt.figure(figsize=(12, 6))
    plt.plot(steps, depth_losses, label="depth_loss")
    plt.plot(steps, grad_losses, label="grad_loss")
    plt.xlabel("Global Step")
    plt.ylabel("Loss")
    title = "Training Loss"
    if smoothing_label:
        title += f" ({smoothing_label})"
    plt.title(title)
    ax = plt.gca()
    if args.log:
        ax.set_yscale("log")
        # Label intermediate log ticks so small changes are easier to read.
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0,)))
        ax.yaxis.set_minor_locator(LogLocator(base=10, subs=range(2, 10)))
        ax.yaxis.set_major_formatter(LogFormatterSciNotation(base=10, labelOnlyBase=False))
        ax.yaxis.set_minor_formatter(LogFormatterSciNotation(base=10, labelOnlyBase=False))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)

    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()

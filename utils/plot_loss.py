import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # must be before importing pyplot
import matplotlib.pyplot as plt


PATTERN = re.compile(r"(\d+)\s+total loss:\s+([0-9.eE+-]+)")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot training loss from all .log files in a directory."
    )
    parser.add_argument("log_dir", type=Path, help="Directory containing .log files")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output image path. Defaults to <log_dir>/loss.png",
    )
    parser.add_argument(
        "--log",
        action="store_true",
        help="Use a logarithmic scale for the y-axis",
    )
    return parser.parse_args()


def collect_loss_points(log_dir: Path):
    if not log_dir.is_dir():
        raise ValueError(f"Not a directory: {log_dir}")

    log_files = sorted(log_dir.glob("*.log"))
    if not log_files:
        raise ValueError(f"No .log files found in {log_dir}")

    steps = []
    losses = []

    for log_file in log_files:
        with log_file.open("r", errors="ignore") as f:
            for line in f:
                match = PATTERN.search(line)
                if match:
                    steps.append(int(match.group(1)))
                    losses.append(float(match.group(2)))

    if not steps:
        raise ValueError(f"No loss entries found in .log files under {log_dir}")

    steps, losses = zip(*sorted(zip(steps, losses)))
    return steps, losses


def main():
    args = parse_args()
    output_path = args.output or args.log_dir / "loss.png"

    steps, losses = collect_loss_points(args.log_dir)

    plt.figure(figsize=(50, 5))
    plt.plot(steps, losses)
    plt.xlabel("Small batch step")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    if args.log:
        plt.yscale("log")
    plt.grid(True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")

    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()

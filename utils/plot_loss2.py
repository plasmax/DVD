import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Plot training loss from a CSV file.")
    parser.add_argument("csv_path", type=Path, help="Path to loss_log.csv")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output image path. Defaults to <csv_dir>/loss_plot.png",
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


def main():
    args = parse_args()
    csv_path = args.csv_path
    output_path = args.output or csv_path.with_name("loss_plot.png")

    steps, depth_losses, grad_losses = read_loss_csv(csv_path)

    plt.figure(figsize=(12, 6))
    plt.plot(steps, depth_losses, label="depth_loss")
    plt.plot(steps, grad_losses, label="grad_loss")
    plt.xlabel("Global Step")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)

    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()

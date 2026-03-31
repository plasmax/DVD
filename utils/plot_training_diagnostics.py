import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_WINDOWS = (25, 50, 100)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate training-diagnostics charts from loss_log.csv."
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        default="saves/v005/loss_log.csv",
        help="Path to loss_log.csv",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for generated figures. Defaults to <csv_dir>/analysis_plots",
    )
    parser.add_argument(
        "--tail-points",
        type=int,
        default=100,
        help="How many recent points to show in the endgame zoom chart.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Figure DPI.",
    )
    return parser.parse_args()


def load_df(csv_path):
    df = pd.read_csv(csv_path)
    required = {"global_step", "depth_loss", "grad_loss"}
    missing = required - set(df.columns)
    if missing:
        missing_cols = ", ".join(sorted(missing))
        raise ValueError(f"Missing required columns: {missing_cols}")
    return df.sort_values("global_step").reset_index(drop=True)


def add_rolling_columns(df, columns, windows):
    for col in columns:
        for window in windows:
            df[f"{col}_roll_{window}"] = df[col].rolling(window=window).mean()
    return df


def ensure_output_dir(csv_path, output_dir):
    if output_dir is None:
        output_dir = Path(csv_path).resolve().parent / "analysis_plots"
    else:
        output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_loss_dashboard(df, loss_cols, output_dir, dpi):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    steps = df["global_step"]
    colors = {"depth_loss": "#1f77b4", "grad_loss": "#d62728"}

    for col in loss_cols:
        axes[0, 0].plot(steps, df[col], alpha=0.25, linewidth=1.2, color=colors[col], label=f"{col} raw")
        axes[0, 0].plot(steps, df[f"{col}_roll_50"], linewidth=2.0, color=colors[col], label=f"{col} roll50")
    axes[0, 0].set_title("Raw Losses with Rolling-50")
    axes[0, 0].set_xlabel("Global Step")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].legend()

    for col in loss_cols:
        for window, style in zip(DEFAULT_WINDOWS, ["-", "--", ":"]):
            axes[0, 1].plot(
                steps,
                df[f"{col}_roll_{window}"],
                linestyle=style,
                linewidth=2.0,
                color=colors[col],
                label=f"{col} roll{window}",
            )
    axes[0, 1].set_title("Rolling Means")
    axes[0, 1].set_xlabel("Global Step")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].legend(ncol=2, fontsize=9)

    scatter = axes[1, 0].scatter(
        df["depth_loss"],
        df["grad_loss"],
        c=steps,
        cmap="viridis",
        s=22,
        alpha=0.8,
    )
    axes[1, 0].set_title("Loss Tradeoff Over Time")
    axes[1, 0].set_xlabel("Depth Loss")
    axes[1, 0].set_ylabel("Grad Loss")
    # Losses stay positive, so log axes make the late-training cluster readable
    # instead of being compressed by the large early-run values.
    axes[1, 0].set_xscale("log")
    axes[1, 0].set_yscale("log")
    axes[1, 0].grid(alpha=0.3)
    cbar = fig.colorbar(scatter, ax=axes[1, 0])
    cbar.set_label("Global Step")

    summary_lines = [
        f"Latest step: {int(df.iloc[-1]['global_step'])}",
        f"Best depth step: {int(df.loc[df['depth_loss'].idxmin(), 'global_step'])}",
        f"Best grad step: {int(df.loc[df['grad_loss'].idxmin(), 'global_step'])}",
        f"Depth last50: {df['depth_loss_roll_50'].iloc[-1]:.6f}",
        f"Grad last50: {df['grad_loss_roll_50'].iloc[-1]:.6f}",
        f"Correlation: {df[['depth_loss', 'grad_loss']].corr().iloc[0, 1]:.4f}",
    ]
    axes[1, 1].axis("off")
    axes[1, 1].set_title("Run Summary")
    axes[1, 1].text(
        0.02,
        0.98,
        "\n".join(summary_lines),
        va="top",
        ha="left",
        fontsize=12,
        family="monospace",
    )

    fig.tight_layout()
    out_path = output_dir / "training_dashboard.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_endgame_zoom(df, loss_cols, output_dir, tail_points, dpi):
    tail_df = df.tail(min(tail_points, len(df))).copy()
    fig, axes = plt.subplots(len(loss_cols), 1, figsize=(14, 8), sharex=True)
    if len(loss_cols) == 1:
        axes = [axes]

    colors = {"depth_loss": "#1f77b4", "grad_loss": "#d62728"}
    for ax, col in zip(axes, loss_cols):
        ax.plot(tail_df["global_step"], tail_df[col], color=colors[col], alpha=0.35, linewidth=1.3, label=f"{col} raw")
        ax.plot(tail_df["global_step"], tail_df[f"{col}_roll_25"], color=colors[col], linewidth=2.0, label=f"{col} roll25")
        ax.plot(tail_df["global_step"], tail_df[f"{col}_roll_50"], color="black", linestyle="--", linewidth=1.8, label=f"{col} roll50")
        best_idx = tail_df[col].idxmin()
        ax.scatter(
            [df.loc[best_idx, "global_step"]],
            [df.loc[best_idx, col]],
            color="#2ca02c",
            s=40,
            zorder=5,
            label="best in window",
        )
        ax.set_ylabel(col)
        ax.grid(alpha=0.3)
        ax.legend()

    axes[-1].set_xlabel("Global Step")
    fig.suptitle(f"Late Training Zoom (last {len(tail_df)} points)")
    fig.tight_layout()
    out_path = output_dir / "endgame_zoom.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_normalized_progress(df, loss_cols, output_dir, dpi):
    fig, ax = plt.subplots(figsize=(14, 6))
    steps = df["global_step"]
    for col, color in zip(loss_cols, ["#1f77b4", "#d62728"]):
        values = df[col].to_numpy()
        denom = values[0] - values.min()
        if denom <= 0:
            normalized = np.zeros_like(values)
        else:
            normalized = (values[0] - values) / denom
        ax.plot(steps, normalized, linewidth=2.2, color=color, label=f"{col} progress")

    ax.set_title("Normalized Optimization Progress")
    ax.set_xlabel("Global Step")
    ax.set_ylabel("Progress toward best observed value")
    ax.set_ylim(-0.02, 1.05)
    ax.grid(alpha=0.3)
    ax.legend()

    out_path = output_dir / "normalized_progress.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_volatility_chart(df, loss_cols, output_dir, dpi):
    fig, ax = plt.subplots(figsize=(14, 6))
    steps = df["global_step"]
    for col, color in zip(loss_cols, ["#1f77b4", "#d62728"]):
        rolling_std = df[col].rolling(window=50).std()
        ax.plot(steps, rolling_std, linewidth=2.0, color=color, label=f"{col} rolling std (50)")

    ax.set_title("Late-Stage Volatility")
    ax.set_xlabel("Global Step")
    ax.set_ylabel("Rolling Standard Deviation")
    ax.grid(alpha=0.3)
    ax.legend()

    out_path = output_dir / "late_volatility.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    args = parse_args()
    df = load_df(args.csv_path)
    loss_cols = ["depth_loss", "grad_loss"]
    add_rolling_columns(df, loss_cols, DEFAULT_WINDOWS)
    output_dir = ensure_output_dir(args.csv_path, args.output_dir)

    outputs = [
        save_loss_dashboard(df, loss_cols, output_dir, args.dpi),
        save_endgame_zoom(df, loss_cols, output_dir, args.tail_points, args.dpi),
        save_normalized_progress(df, loss_cols, output_dir, args.dpi),
        save_volatility_chart(df, loss_cols, output_dir, args.dpi),
    ]

    print("Saved figures:")
    for out_path in outputs:
        print(out_path)


if __name__ == "__main__":
    main()

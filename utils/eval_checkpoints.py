"""Evaluate training losses across multiple checkpoints and produce a comparison plot.

Named checkpoints (e.g. ckpt/dvd_1.1) are shown as horizontal reference lines so you
can see at which training step the fine-tuned model crosses or matches the baseline.

Usage:
  python utils/eval_checkpoints.py \\
      --ckpts ckpt/model ckpt/dvd_1.1 "saves/v005/checkpoint-step-*" \\
      --config train_config/normal_config/video_config_new.yaml \\
      --dataset infinigen --num_batches 10 --batch_size 1 \\
      --output eval_results.png
"""

import argparse
import glob
import os
import re
import subprocess
import sys

import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_ckpt_dir(path):
    """Accept either a checkpoint dir or a path ending in model.safetensors."""
    if path.endswith("model.safetensors"):
        return os.path.dirname(path)
    return path


def extract_step(ckpt_dir):
    """Return the step number from e.g. .../checkpoint-step-3000, or None."""
    m = re.search(r"checkpoint-step-(\d+)", ckpt_dir)
    return int(m.group(1)) if m else None


def short_label(ckpt_dir):
    parts = ckpt_dir.rstrip("/").split("/")
    step = extract_step(ckpt_dir)
    if step is not None:
        return f"step {step}"
    return "/".join(parts[-2:]) if len(parts) >= 2 else parts[-1]


def run_eval(ckpt_dir, config, dataset, num_batches, batch_size):
    script = os.path.join(os.path.dirname(__file__), "eval_loss.py")
    cmd = [
        sys.executable, script,
        "--ckpt", ckpt_dir,
        "--config", config,
        "--dataset", dataset,
        "--num_batches", str(num_batches),
    ]
    if batch_size is not None:
        cmd += ["--batch_size", str(batch_size)]

    print(f"\n--- Evaluating: {ckpt_dir} ---", flush=True)
    # Let stderr (tqdm, warnings) go to the terminal live; capture stdout for parsing.
    result = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)
    print(result.stdout)

    losses = {}
    for key in ("depth_loss", "grad_loss", "total_loss"):
        m = re.search(rf"{key}:\s+([\d.]+)", result.stdout)
        if m:
            losses[key] = float(m.group(1))

    m = re.search(r"Batches:\s+(\d+)", result.stdout)
    losses["num_batches"] = int(m.group(1)) if m else None
    return losses


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate and plot losses across checkpoints"
    )
    parser.add_argument(
        "--ckpts", nargs="+", required=True,
        help="Checkpoint dirs or glob patterns (quote globs to prevent shell expansion)",
    )
    parser.add_argument("--config", required=True, help="Training config YAML")
    parser.add_argument(
        "--dataset", default="infinigen",
        choices=["hypersim", "tartanair", "infinigen"],
    )
    parser.add_argument("--num_batches", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--output", default="eval_results.png")
    args = parser.parse_args()

    # Expand globs, normalize, deduplicate
    raw_dirs = []
    for pattern in args.ckpts:
        expanded = sorted(glob.glob(pattern))
        raw_dirs.extend(expanded if expanded else [pattern])

    seen = set()
    ckpt_dirs = []
    for p in (normalize_ckpt_dir(p) for p in raw_dirs):
        if p not in seen:
            seen.add(p)
            ckpt_dirs.append(p)

    # Sort: named (no step) first, then by step number
    ckpt_dirs.sort(key=lambda d: (0, 0) if extract_step(d) is None else (1, extract_step(d)))

    # Run evals
    results = []
    for ckpt_dir in ckpt_dirs:
        losses = run_eval(ckpt_dir, args.config, args.dataset, args.num_batches, args.batch_size)
        results.append({
            "ckpt": ckpt_dir,
            "label": short_label(ckpt_dir),
            "step": extract_step(ckpt_dir),
            **losses,
        })

    step_results = [r for r in results if r["step"] is not None]
    named_results = [r for r in results if r["step"] is None]

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print(f"{'Checkpoint':<30} {'depth':>10} {'grad':>10} {'total':>10}")
    print("=" * 65)
    for r in results:
        print(
            f"{r['label']:<30} "
            f"{r.get('depth_loss', float('nan')):>10.6f} "
            f"{r.get('grad_loss', float('nan')):>10.6f} "
            f"{r.get('total_loss', float('nan')):>10.6f}"
        )
    print("=" * 65)

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    METRICS = [
        ("depth_loss", "Depth MSE", "#1f77b4"),
        ("grad_loss",  "Grad loss", "#ff7f0e"),
        ("total_loss", "Total loss", "#2ca02c"),
    ]

    fig, ax = plt.subplots(figsize=(13, 6))

    # Step-based checkpoints as connected lines
    if step_results:
        steps = [r["step"] for r in step_results]
        for key, label, color in METRICS:
            vals = [r.get(key) for r in step_results]
            if any(v is not None for v in vals):
                ax.plot(steps, vals, marker="o", label=label, color=color, linewidth=1.8)

    # Named checkpoints as horizontal dashed reference lines
    # (makes it easy to read off "at which step does training reach this baseline?")
    ref_linestyles = ["--", "-.", ":"]
    for i, r in enumerate(named_results):
        ls = ref_linestyles[i % len(ref_linestyles)]
        for key, _, color in METRICS:
            val = r.get(key)
            if val is not None:
                ax.axhline(
                    val, color=color, linestyle=ls, linewidth=1.2, alpha=0.75,
                    label=f"{r['label']} – {key.replace('_', ' ')}",
                )

    ax.set_xlabel("Training step", fontsize=11)
    ax.set_ylabel("Loss", fontsize=11)
    ax.set_title(
        f"Checkpoint evaluation  |  dataset: {args.dataset}  |  batches: {args.num_batches}",
        fontsize=12,
    )
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"\nPlot saved to {args.output}")


if __name__ == "__main__":
    main()

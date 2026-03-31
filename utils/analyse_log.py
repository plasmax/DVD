import argparse

import numpy as np
import pandas as pd


DEFAULT_PATH = "saves/v005/loss_log.csv"
DEFAULT_WINDOWS = (25, 50, 100)
DEFAULT_DOWNSAMPLE = 20


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize a training loss log and highlight plateau/noise trends."
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=DEFAULT_PATH,
        help=f"Path to loss_log.csv (default: {DEFAULT_PATH})",
    )
    parser.add_argument(
        "--tail",
        type=int,
        default=10,
        help="How many recent rows to print in the LAST section.",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=DEFAULT_DOWNSAMPLE,
        help="How many rows to show in the downsampled table.",
    )
    return parser.parse_args()


def infer_loss_columns(df):
    return [c for c in df.columns if c not in ["global_step", "learning_rate"]]


def latest_window(series, window):
    if len(series) < window:
        return None
    return series.tail(window).mean()


def best_window(df, col, window):
    if len(df) < window:
        return None

    rolling = df[col].rolling(window=window).mean()
    best_end = rolling.idxmin()
    best_start = best_end - window + 1
    return {
        "window": window,
        "mean": rolling.loc[best_end],
        "start_step": int(df.iloc[best_start]["global_step"]),
        "end_step": int(df.iloc[best_end]["global_step"]),
    }


def estimate_plateau_step(df, col, window=50, tolerance_ratio=0.02):
    if len(df) < window:
        return None

    rolling = df[col].rolling(window=window).mean()
    valid = rolling.dropna()
    if valid.empty:
        return None

    best = valid.min()
    threshold = best * (1 + tolerance_ratio)
    plateau_candidates = valid[valid <= threshold]
    if plateau_candidates.empty:
        return None

    plateau_end = plateau_candidates.index[0]
    plateau_start = plateau_end - window + 1
    return {
        "window": window,
        "threshold": threshold,
        "step": int(df.iloc[plateau_start]["global_step"]),
        "rolling_mean": rolling.loc[plateau_end],
    }


def volatility_report(series, recent=50, previous=50):
    recent_slice = series.tail(recent)
    if len(series) < recent + previous:
        previous_slice = series.iloc[:-recent]
    else:
        previous_slice = series.iloc[-(recent + previous):-recent]

    if len(recent_slice) < 2 or len(previous_slice) < 2:
        return None

    recent_std = recent_slice.std()
    previous_std = previous_slice.std()
    delta = recent_std - previous_std
    ratio = recent_std / previous_std if previous_std > 0 else np.nan
    return {
        "recent_std": recent_std,
        "previous_std": previous_std,
        "delta": delta,
        "ratio": ratio,
    }


def classify_metric(df, col):
    series = df[col]
    last10 = latest_window(series, 10)
    last20 = latest_window(series, 20)
    last50 = latest_window(series, 50)
    best_value = series.min()
    latest = series.iloc[-1]

    if last10 is None or last20 is None or last50 is None:
        return "not_enough_data"

    close_to_best = last10 <= best_value * 1.05
    improving = last10 < last20 < last50

    if improving and close_to_best:
        return "improving_near_best"
    if improving:
        return "improving"
    if close_to_best:
        return "stable_near_best"
    return "plateau_or_regression" if latest > last20 else "plateau"


def overall_recommendation(statuses):
    values = set(statuses.values())
    if values == {"improving_near_best"} or values == {"improving"}:
        return "continue; losses are still trending down."
    if "plateau_or_regression" in values:
        return "reduce LR or validate checkpoints; some metrics are no longer improving cleanly."
    if "stable_near_best" in values or "plateau" in values:
        return "consider LR decay or early stop; training looks close to a plateau."
    return "continue cautiously; signal is mixed."


def format_downsample(df, sample_count):
    idx = np.linspace(0, len(df) - 1, min(sample_count, len(df))).round().astype(int)
    return df.iloc[np.unique(idx)].to_string(index=False)


def main():
    args = parse_args()
    df = pd.read_csv(args.path)
    loss_cols = infer_loss_columns(df)

    print("ROWS", len(df))
    print("COLUMNS", list(df.columns))
    print("\nFIRST 5")
    print(df.head(5).to_string(index=False))
    print(f"\nLAST {args.tail}")
    print(df.tail(args.tail).to_string(index=False))

    print("\nSUMMARY")
    print(df[loss_cols].describe().to_string())

    print("\nBEST STEPS")
    for col in loss_cols:
        i = df[col].idxmin()
        print(f"{col}: best={df.loc[i, col]:.8f} at step={int(df.loc[i, 'global_step'])}")

    print("\nROLLING MEANS")
    for col in loss_cols:
        parts = []
        for window in DEFAULT_WINDOWS:
            mean_value = latest_window(df[col], window)
            if mean_value is not None:
                parts.append(f"last{window}={mean_value:.8f}")
        print(f"{col}: " + " ".join(parts))

    print("\nBEST WINDOWS")
    for col in loss_cols:
        parts = []
        for window in DEFAULT_WINDOWS:
            best = best_window(df, col, window)
            if best is not None:
                parts.append(
                    f"{window}:mean={best['mean']:.8f}@[{best['start_step']},{best['end_step']}]"
                )
        print(f"{col}: " + " ".join(parts))

    print("\nEND TREND")
    for col in loss_cols:
        s10 = latest_window(df[col], 10)
        s20 = latest_window(df[col], 20)
        s50 = latest_window(df[col], 50)
        delta_10_20 = s10 - s20 if s10 is not None and s20 is not None else np.nan
        delta_20_50 = s20 - s50 if s20 is not None and s50 is not None else np.nan
        print(
            f"{col}: "
            f"last10_mean={s10:.8f} "
            f"last20_mean={s20:.8f} "
            f"last50_mean={s50:.8f} "
            f"delta10_20={delta_10_20:+.8f} "
            f"delta20_50={delta_20_50:+.8f}"
        )

    print("\nLATE VOLATILITY")
    for col in loss_cols:
        report = volatility_report(df[col], recent=50, previous=50)
        if report is None:
            print(f"{col}: not_enough_data")
            continue
        print(
            f"{col}: "
            f"recent50_std={report['recent_std']:.8f} "
            f"prev50_std={report['previous_std']:.8f} "
            f"delta={report['delta']:+.8f} "
            f"ratio={report['ratio']:.3f}"
        )

    print("\nPLATEAU ESTIMATE")
    for col in loss_cols:
        plateau = estimate_plateau_step(df, col, window=50, tolerance_ratio=0.02)
        if plateau is None:
            print(f"{col}: not_enough_data")
            continue
        print(
            f"{col}: "
            f"within_2pct_of_best_by_step={plateau['step']} "
            f"rolling50={plateau['rolling_mean']:.8f}"
        )

    print("\nTRADEOFF")
    if len(loss_cols) >= 2:
        corr = df[loss_cols].corr().iloc[0, 1]
        print(f"{loss_cols[0]} vs {loss_cols[1]} correlation={corr:.4f}")
    else:
        print("Need at least two loss columns for a tradeoff check.")

    statuses = {col: classify_metric(df, col) for col in loss_cols}
    print("\nRECOMMENDATION")
    for col, status in statuses.items():
        print(f"{col}: {status}")
    print("overall:", overall_recommendation(statuses))

    print("\nDOWNSAMPLED")
    print(format_downsample(df, args.sample_count))


if __name__ == "__main__":
    main()

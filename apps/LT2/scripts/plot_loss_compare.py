"""Plot loss curves from compare_rwkv7_gdn_tiny.py JSON output."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def moving_average(values, window: int):
    if window <= 1:
        return values
    smoothed = []
    total = 0.0
    queue = []
    for value in values:
        queue.append(float(value))
        total += float(value)
        if len(queue) > window:
            total -= queue.pop(0)
        smoothed.append(total / len(queue))
    return smoothed


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot mixer loss curves from comparison JSON.")
    parser.add_argument("--json", required=True, help="Path to comparison JSON.")
    parser.add_argument("--out-prefix", required=True, help="Output path prefix without extension.")
    parser.add_argument("--title", default="LT2 Mixer Loss Comparison")
    parser.add_argument("--smooth-window", type=int, default=1, help="Trailing moving-average window.")
    parser.add_argument("--show-raw", action="store_true", help="Show faint raw losses behind smoothed curves.")
    args = parser.parse_args()

    with open(args.json) as f:
        data = json.load(f)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 5.6), dpi=160)
    colors = {
        "gdn": "#4C78A8",
        "rwkv7": "#F58518",
        "rwkv7_native": "#54A24B",
    }

    for result in data["results"]:
        losses = result["losses"]
        steps = list(range(1, len(losses) + 1))
        plotted_losses = moving_average(losses, args.smooth_window)
        label = (
            f"{result['mixer']} "
            f"({result['params']:,} params, {result['tokens_per_second']:.0f} tok/s)"
        )
        color = colors.get(result["mixer"])
        if args.show_raw and args.smooth_window > 1:
            ax.plot(
                steps,
                losses,
                linewidth=0.7,
                alpha=0.16,
                color=color,
            )
        ax.plot(
            steps,
            plotted_losses,
            linewidth=2.4,
            label=label,
            color=color,
        )

    ax.set_title(args.title, fontsize=14, pad=12)
    xlabel = "Timed training step"
    ylabel = "Cross entropy loss"
    if args.smooth_window > 1:
        ylabel += f" ({args.smooth_window}-step moving average)"
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(frameon=True, fontsize=9)
    ax.margins(x=0.02)
    fig.tight_layout()

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg", "pdf"):
        fig.savefig(out_prefix.with_suffix(f".{ext}"), bbox_inches="tight")


if __name__ == "__main__":
    main()

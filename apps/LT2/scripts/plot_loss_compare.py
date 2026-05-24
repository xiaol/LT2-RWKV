"""Plot loss curves from compare_rwkv7_gdn_tiny.py JSON output."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot mixer loss curves from comparison JSON.")
    parser.add_argument("--json", required=True, help="Path to comparison JSON.")
    parser.add_argument("--out-prefix", required=True, help="Output path prefix without extension.")
    parser.add_argument("--title", default="LT2 Mixer Loss Comparison")
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
        label = (
            f"{result['mixer']} "
            f"({result['params']:,} params, {result['tokens_per_second']:.0f} tok/s)"
        )
        ax.plot(
            steps,
            losses,
            marker="o",
            linewidth=2.0,
            markersize=3.8,
            label=label,
            color=colors.get(result["mixer"]),
        )

    ax.set_title(args.title, fontsize=14, pad=12)
    ax.set_xlabel("Timed training step")
    ax.set_ylabel("Cross entropy loss")
    ax.legend(frameon=True, fontsize=9)
    ax.margins(x=0.02)
    fig.tight_layout()

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg", "pdf"):
        fig.savefig(out_prefix.with_suffix(f".{ext}"), bbox_inches="tight")


if __name__ == "__main__":
    main()

"""Chart 12: Cloud vs Local trade-offs chart.

Replaces the cloud-vs-local ASCII trade-off block in Discussion 4.4.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

OUTPUT = Path(__file__).resolve().parent.parent / "images" / "12_cloud_local_tradeoffs.png"

NAVY = "#1F4E79"
GREEN = "#2C7A4D"
DARK = "#1A1A1A"

ROWS = [
    ("Wall-clock speed", "Highest", "Cloud high, local slower"),
    ("1M token context", "Yes (Claude Opus)", "Yes (cloud) / partial (local)"),
    ("Reproducibility", "Cloud commit log", "Cloud + local rerun parity"),
    ("Hardware floor", "Data-center scale", "i5-6200U / 4 GB demonstrated"),
    ("PHI / data security", "Cloud transit risk", "Local processing + cloud meta"),
    ("Audit trail", "GitHub commits", "GitHub commits + local hash"),
    ("Site adoption cost", "Cloud subscription", "Off-the-shelf laptop"),
]


def _pill(ax, x, y, w, h, color, text):
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.10",
            linewidth=1.0,
            edgecolor=color,
            facecolor="white",
        )
    )
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9, color=DARK)


def main() -> None:
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    ax.text(
        5.0,
        5.65,
        "Cloud-Only vs Cloud-Plus-Local Verification for RTCT-Aligned Oncology Trial AI",
        fontsize=12.5,
        fontweight="bold",
        ha="center",
        color=NAVY,
    )

    ax.add_patch(mpatches.Rectangle((3.05, 5.10), 3.0, 0.40, color=NAVY))
    ax.add_patch(mpatches.Rectangle((6.20, 5.10), 3.0, 0.40, color=GREEN))
    ax.text(4.55, 5.30, "Cloud Compute Only", ha="center", va="center", color="white", fontweight="bold", fontsize=10)
    ax.text(
        7.70,
        5.30,
        "Cloud + Local Verification",
        ha="center",
        va="center",
        color="white",
        fontweight="bold",
        fontsize=10,
    )

    row_h = 0.55
    for i, (prop, cloud_text, local_text) in enumerate(ROWS):
        y = 4.60 - i * row_h
        ax.text(0.20, y + row_h / 2, prop, fontsize=9.5, fontweight="bold", color=NAVY, va="center")
        _pill(ax, 3.05, y, 3.0, row_h - 0.10, NAVY, cloud_text)
        _pill(ax, 6.20, y, 3.0, row_h - 0.10, GREEN, local_text)

    ax.add_patch(
        mpatches.FancyBboxPatch(
            (0.20, 0.10),
            9.60,
            0.45,
            boxstyle="round,pad=0.02,rounding_size=0.04",
            linewidth=1.2,
            edgecolor=NAVY,
            facecolor="#FFF8E1",
        )
    )
    ax.text(
        5.0,
        0.32,
        "Sims 1, 2, 3 are cloud only; Sim 4 is cloud plus local on Core i5-6200U. Hybrid is the natural endpoint.",
        ha="center",
        va="center",
        fontsize=9.5,
        color=DARK,
        fontweight="bold",
    )

    fig.savefig(OUTPUT, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

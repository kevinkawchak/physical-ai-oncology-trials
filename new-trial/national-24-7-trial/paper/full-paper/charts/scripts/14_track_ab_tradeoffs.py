"""Chart 14: Track A vs Track B trade-offs.

Replaces tab:future-tracks in Limitations 5.2.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

OUTPUT = Path(__file__).resolve().parent.parent / "images" / "14_track_ab_tradeoffs.png"

NAVY = "#1F4E79"
TEAL = "#2C7A7A"
DARK = "#1A1A1A"

ROWS = [
    ("Architecture", "1 model, 1 inference loop", "1 orchestrator + N specialist agents"),
    ("Inference cost", "Higher per decision", "Lower per site, higher overall design"),
    ("Reg. accountability", "Single surface", "Distributed; per-agent floor"),
    ("PHI handling", "Cloud transit required", "Local agents handle PHI on-site"),
    ("Hardware floor", "Cloud (data-center)", "Site (Core i5-6200U / 4 GB feasible)"),
    ("Update cycle", "Single model upgrade", "Per-agent upgrade + orchestrator swap"),
    ("Extension target", "Simulation 4 + clinical link", "Simulation 1 site network + per-task agents"),
]


def _card(ax, x, y, w, h, color, text):
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.04",
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
        "Track A vs Track B for a Future RTCT-Aligned Oncology Trial AI",
        fontsize=13,
        fontweight="bold",
        ha="center",
        color=NAVY,
    )
    ax.text(
        5.0,
        5.30,
        "Single big model performing all tasks vs single big model that creates smaller agents performing smaller tasks locally",
        fontsize=9,
        ha="center",
        color=DARK,
        style="italic",
    )

    ax.add_patch(mpatches.Rectangle((3.05, 4.75), 3.0, 0.40, color=NAVY))
    ax.add_patch(mpatches.Rectangle((6.20, 4.75), 3.0, 0.40, color=TEAL))
    ax.text(
        4.55, 4.95, "Track A: Big Single Model", ha="center", va="center", color="white", fontweight="bold", fontsize=10
    )
    ax.text(
        7.70,
        4.95,
        "Track B: Big Model + Small Agents",
        ha="center",
        va="center",
        color="white",
        fontweight="bold",
        fontsize=10,
    )

    row_h = 0.55
    for i, (prop, a_text, b_text) in enumerate(ROWS):
        y = 4.30 - i * row_h
        ax.text(0.20, y + row_h / 2, prop, fontsize=9.5, fontweight="bold", color=NAVY, va="center")
        _card(ax, 3.05, y, 3.0, row_h - 0.10, NAVY, a_text)
        _card(ax, 6.20, y, 3.0, row_h - 0.10, TEAL, b_text)

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
        "Both tracks should be prototyped before a real-patient trial; the comparison itself becomes a research output.",
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

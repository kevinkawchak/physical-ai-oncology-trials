"""Chart 10: FDA RTCT extension chart (full page).

Replaces tab:disc-fda-extension in Discussion 4.1.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

OUTPUT = Path(__file__).resolve().parent.parent / "images" / "10_fda_extension_chart.png"

NAVY = "#1F4E79"
FDA_COLOR = "#4A7BAA"
SIM_COLOR = "#2C7A4D"
DARK = "#1A1A1A"

ROWS = [
    ("Trials", "2 (TRAVERSE, STREAM-SCLC)", "4 author simulations covering site and sponsor"),
    ("Sites", "MD Anderson, Penn (TRAVERSE)", "4 sites in Simulation 1, distributed in others"),
    ("Robots", "Not specified in the announcement", "116 robot instances in Simulation 1"),
    ("Real-time signals", "Endpoint and safety signals only", "Endpoint, safety, robot authorization, agent decision"),
    ("Cadence", "Real-time as defined per trial", "Per-hour minute-resolution + per-stage decision"),
    ("Predictive layer", "Per-trial supervised endpoints", "Repository-scale 1M-token context across 4 simulations"),
    ("Local-side compute", "Not addressed", "Verified on Core i5-6200U / 4 GB / Win 10 (Sim 4)"),
]


def _card(ax, x, y, w, h, color, text, light=False):
    fill = "white" if light else color
    edge = color
    text_color = DARK if light else "white"
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.04",
            linewidth=1.2,
            edgecolor=edge,
            facecolor=fill,
            alpha=0.95 if not light else 1.0,
        )
    )
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9, color=text_color, wrap=True)


def main() -> None:
    fig, ax = plt.subplots(figsize=(11, 8.5), constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 8.5)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.text(
        5.5,
        8.10,
        "Capability Extension over the FDA RTCT 28 April 2026 Proof of Concept",
        fontsize=15,
        fontweight="bold",
        ha="center",
        color=NAVY,
    )
    ax.text(
        5.5,
        7.78,
        "Seven Operational Dimensions, Two Programs Side by Side",
        fontsize=10.5,
        ha="center",
        color=DARK,
        style="italic",
    )

    ax.add_patch(mpatches.Rectangle((0.4, 7.05), 4.0, 0.50, color=FDA_COLOR))
    ax.add_patch(mpatches.Rectangle((6.6, 7.05), 4.0, 0.50, color=SIM_COLOR))
    ax.text(2.4, 7.30, "FDA RTCT 2026 PoC", ha="center", va="center", color="white", fontweight="bold", fontsize=11)
    ax.text(8.6, 7.30, "Four Simulations Here", ha="center", va="center", color="white", fontweight="bold", fontsize=11)

    row_height = 0.78
    for i, (dim, fda_text, sim_text) in enumerate(ROWS):
        y = 6.20 - i * row_height
        ax.text(5.5, y + row_height / 2, dim, ha="center", va="center", fontsize=10, fontweight="bold", color=NAVY)
        _card(ax, 0.4, y, 4.0, row_height - 0.12, FDA_COLOR, fda_text, light=True)
        _card(ax, 6.6, y, 4.0, row_height - 0.12, SIM_COLOR, sim_text, light=True)

    ax.add_patch(
        mpatches.FancyBboxPatch(
            (0.4, 0.10),
            10.2,
            0.60,
            boxstyle="round,pad=0.02,rounding_size=0.04",
            linewidth=1.4,
            edgecolor=NAVY,
            facecolor="#FFF8E1",
        )
    )
    ax.text(
        5.5,
        0.52,
        "Three crucial differentiators for patient safety and efficacy:",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        color=NAVY,
    )
    ax.text(
        5.5,
        0.25,
        "116 robot instances (Sim 1)   |   1M token context (all sims)   |   Core i5-6200U local verification (Sim 4)",
        ha="center",
        va="center",
        fontsize=9.5,
        color=DARK,
    )

    fig.savefig(OUTPUT, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

"""Chart 15: Cross-simulation synthesis (full page).

Replaces the cross-simulation synthesis ASCII block in Results 3.5.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

OUTPUT = Path(__file__).resolve().parent.parent / "images" / "15_cross_simulation_synthesis.png"

NAVY = "#1F4E79"
TEAL = "#2C7A7A"
GOLD = "#B45424"
PURPLE = "#6A4C8C"
DARK = "#1A1A1A"
LIGHT = "#F0F0F0"

COLUMNS = [
    ("Sim 1 (site)", NAVY, "Continuous RTCT 56h"),
    ("Sim 2 (site)", TEAL, "Single patient 10st"),
    ("Sim 3 (sponsor)", GOLD, "24-hour sponsor"),
    ("Sim 4 (sponsor)", PURPLE, "168-hour sponsor"),
]

ROWS = [
    ("Hours", ["56", "1,120 days", "24", "168"]),
    ("Patients", ["168 cumul.", "1", "5 peak", "1,336 cum."]),
    ("Robots", ["116", "2 (Da Vinci, Franka)", "0 robots, 53 agents", "0 robots, 53 agents"]),
    ("ASCII diagrams", ["168 (3/hr)", "30 progress", "75", "525"]),
    ("Python scripts", ["0", "12 modules", "24 hourly + 53 agents", "168 hourly (+ shared)"]),
    ("JSON outputs", ["0", "per-stage", "24", "168"]),
    ("GitHub commits", ["56 (1/hr)", "per-stage", "24 (1/hr)", "168 (1/hr)"]),
    ("Local verification", ["n/a", "local re-run", "n/a", "i5-6200U / 4 GB OK"]),
]


def main() -> None:
    fig, ax = plt.subplots(figsize=(11, 8.5), constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 8.5)
    ax.axis("off")

    ax.text(
        5.5,
        8.18,
        "Cross-Simulation Synthesis - Site (Sims 1, 2) and Sponsor (Sims 3, 4)",
        fontsize=14,
        fontweight="bold",
        ha="center",
        color=NAVY,
    )

    ax.text(2.45, 7.65, "Sites", fontsize=11, fontweight="bold", color=NAVY, ha="center")
    ax.text(8.05, 7.65, "Sponsors", fontsize=11, fontweight="bold", color=GOLD, ha="center")
    ax.plot([5.25, 5.25], [7.40, 0.80], color=DARK, linestyle="--", linewidth=0.8)

    col_x = [2.45, 4.55, 6.65, 8.75]
    col_w = 1.85
    header_y = 6.95
    for (title, color, scope), x in zip(COLUMNS, col_x):
        ax.add_patch(mpatches.Rectangle((x - col_w / 2, header_y), col_w, 0.55, color=color))
        ax.text(x, header_y + 0.36, title, ha="center", color="white", fontweight="bold", fontsize=10)
        ax.text(x, header_y + 0.14, scope, ha="center", color="white", fontsize=8.5, style="italic")

    row_h = 0.70
    for i, (prop, cells) in enumerate(ROWS):
        y = 6.20 - i * row_h
        if i % 2 == 0:
            ax.add_patch(mpatches.Rectangle((0.30, y), 10.40, row_h - 0.05, color=LIGHT, zorder=0))
        ax.text(0.50, y + row_h / 2, prop, fontsize=10, fontweight="bold", color=NAVY, va="center")
        for cell, x in zip(cells, col_x):
            ax.text(x, y + row_h / 2, cell, ha="center", va="center", fontsize=9, color=DARK)

    ax.add_patch(
        mpatches.FancyBboxPatch(
            (0.30, 0.10),
            10.40,
            0.55,
            boxstyle="round,pad=0.02,rounding_size=0.04",
            linewidth=1.2,
            edgecolor=NAVY,
            facecolor="#FFF8E1",
        )
    )
    ax.text(
        5.5,
        0.45,
        "Repository scale 1M token context + hourly commit cadence + ASCII + Markdown + Python + JSON artifacts",
        ha="center",
        va="center",
        fontsize=9.5,
        color=DARK,
        fontweight="bold",
    )
    ax.text(
        5.5,
        0.21,
        "is the computational signature shared across all four simulations.",
        ha="center",
        va="center",
        fontsize=9,
        color=DARK,
    )

    fig.savefig(OUTPUT, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

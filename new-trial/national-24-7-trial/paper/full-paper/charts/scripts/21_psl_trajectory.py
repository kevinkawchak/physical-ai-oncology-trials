"""Chart 21: PSL trajectory chart (NEW).

Adds a NEW PSL trajectory line chart to Results 3.4 showing the climb
from 63.4 to 70.0 across the Sim 4 168-hour 7-day run.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

OUTPUT = Path(__file__).resolve().parent.parent / "images" / "21_psl_trajectory.png"

NAVY = "#1F4E79"
GREEN = "#2C7A4D"
LIGHT_GREEN = "#E5F0E8"
GOLD = "#B45424"
DARK = "#1A1A1A"
GRID = "#D8D8D8"

ANCHORS = [(0, 63.4), (23, 64.8), (47, 65.9), (71, 67.0), (95, 67.9), (119, 68.8), (143, 69.5), (167, 70.0)]


def main() -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for d in (24, 48, 72, 96, 120, 144):
        ax.axvline(d, color=GRID, linewidth=0.7, linestyle="--")
        ax.text(d, 60.3, f"D{d // 24 + 1}", fontsize=8, ha="center", color="#7C7C7C")

    xs = [a[0] for a in ANCHORS]
    ys = [a[1] for a in ANCHORS]
    ax.fill_between(xs, [60] * len(xs), ys, color=LIGHT_GREEN, alpha=0.6)
    ax.plot(xs, ys, color=GREEN, linewidth=2.6, marker="o", markersize=7, zorder=4)
    for x, y in ANCHORS:
        ax.annotate(
            f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, 8), fontsize=8.5, ha="center", color=DARK
        )

    ax.scatter([0], [63.4], color=NAVY, s=120, zorder=5, edgecolor="white", linewidth=1.5)
    ax.scatter([167], [70.0], color=GREEN, s=120, zorder=5, edgecolor="white", linewidth=1.5)

    ax.annotate(
        "Day 3 - Safety day, 22 escalations",
        xy=(60, 66.5),
        xytext=(80, 64.2),
        fontsize=9,
        color=GOLD,
        arrowprops={"arrowstyle": "->", "color": GOLD, "lw": 1.0},
    )
    ax.annotate(
        "Day 5 - Analysis day, biostats peak",
        xy=(108, 68.4),
        xytext=(110, 65.5),
        fontsize=9,
        color=GOLD,
        arrowprops={"arrowstyle": "->", "color": GOLD, "lw": 1.0},
    )
    ax.annotate(
        "Day 7 - Closeout, audit signoff",
        xy=(167, 70.0),
        xytext=(125, 71.2),
        fontsize=9,
        color=GREEN,
        fontweight="bold",
        arrowprops={"arrowstyle": "->", "color": GREEN, "lw": 1.0},
    )

    ax.set_xlim(-5, 175)
    ax.set_ylim(60, 73)
    ax.set_xticks([0, 24, 48, 72, 96, 120, 144, 167])
    ax.set_xticklabels(["H0", "H24", "H48", "H72", "H96", "H120", "H144", "H167"], fontsize=9)
    ax.set_xlabel("Hour", fontsize=10)
    ax.set_ylabel("PSL Score", fontsize=10)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    ax.set_title(
        "Simulation 4 PSL Trajectory: 63.4 to 70.0 (+6.6) across 168 Hours",
        fontsize=13,
        fontweight="bold",
        color=NAVY,
    )

    delta_box = mpatches.FancyBboxPatch(
        (15, 71.3),
        45,
        1.4,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=1.0,
        edgecolor=GREEN,
        facecolor="#FFF8E1",
    )
    ax.add_patch(delta_box)
    ax.text(37.5, 72.0, "Weekly delta +6.6 PSL points", ha="center", fontsize=9, fontweight="bold", color=GREEN)

    fig.savefig(OUTPUT, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

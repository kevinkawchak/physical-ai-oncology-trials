"""Chart 03: Simulation 1 hour 23 robot status heatmap.

Replaces the hour-23 robot status timeline ASCII block in Results 3.1 with
a minute-resolution heatmap across the 116 robot instances at four sites.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

OUTPUT = Path(__file__).resolve().parent.parent / "images" / "03_sim1_robot_status_heatmap.png"

STANDBY = 0
MONITORING = 1
ACTIVE = 2
IDLE = 3

PALETTE = ["#E6E6E6", "#4A7BAA", "#2C7A4D", "#FFFFFF"]
EDGE = "#9CA3AF"

ROWS = [
    ("SITE-A COMPN-03", [MONITORING] * 60),
    ("SITE-A Other 28 robots", [STANDBY] * 60),
    ("SITE-B All 29 robots", [STANDBY] * 60),
    ("SITE-C All 29 robots", [STANDBY] * 60),
    ("SITE-D IMAGE-01", [IDLE] * 30 + [ACTIVE] * 15 + [IDLE] * 15),
    ("SITE-D COMPN-02", [MONITORING] * 60),
    ("SITE-D Other 27 robots", [STANDBY] * 60),
]


def main() -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    matrix = np.array([row[1] for row in ROWS])
    cmap = mcolors.ListedColormap(PALETTE)
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    ax.imshow(matrix, aspect="auto", cmap=cmap, norm=norm, interpolation="none")

    for x in range(matrix.shape[1] + 1):
        ax.axvline(x - 0.5, color=EDGE, linewidth=0.3)
    for y in range(matrix.shape[0] + 1):
        ax.axhline(y - 0.5, color=EDGE, linewidth=0.3)

    ax.set_yticks(range(len(ROWS)))
    ax.set_yticklabels([row[0] for row in ROWS], fontsize=9.5)
    ax.set_xticks(range(0, 60, 5))
    ax.set_xticklabels([f"{m:02d}" for m in range(0, 60, 5)], fontsize=9)
    ax.set_xlabel("Minute (23:00 to 23:59 UTC)", fontsize=10)
    for spine in ax.spines.values():
        spine.set_color("#1F4E79")

    ax.set_title(
        "Hour 23 Robot Status Timeline - Day 1 Closing (23:00 to 23:59 UTC)",
        fontsize=13,
        fontweight="bold",
        color="#1F4E79",
    )

    legend_items = [
        mpatches.Patch(color=PALETTE[STANDBY], label="Standby (SS)"),
        mpatches.Patch(color=PALETTE[MONITORING], label="Monitoring (MM)"),
        mpatches.Patch(color=PALETTE[ACTIVE], label="Active (AA)"),
        mpatches.Patch(facecolor=PALETTE[IDLE], edgecolor=EDGE, label="Idle"),
    ]
    ax.legend(
        handles=legend_items,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.32),
        ncol=4,
        frameon=False,
        fontsize=9,
    )

    ax.text(
        30,
        -1.2,
        "Hour-23 active robot-min: 18   Util 0.26 percent   Peak 1   Errors 0   "
        "Day 1 cumulative ~2,500 active robot-min",
        fontsize=9,
        ha="center",
        style="italic",
        color="#1A1A1A",
    )

    fig.savefig(OUTPUT, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

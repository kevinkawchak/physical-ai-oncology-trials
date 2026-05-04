"""Chart 19: 1M token value proposition wheel (NEW).

Adds a NEW value proposition wheel to Introduction 1.3 that visualizes
the six advantage dimensions of repository-scale 1M token context.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

OUTPUT = Path(__file__).resolve().parent.parent / "images" / "19_value_proposition_wheel.png"

DARK = "#1A1A1A"
HUB_NAVY = "#1F4E79"

WEDGES = [
    {
        "name": "Cross-modal",
        "color": "#1F4E79",
        "blurb": "Reasons across protocol,\nEHR, robot telemetry,\nregulatory, and ASCII\nin a single pass",
    },
    {
        "name": "Hourly cadence",
        "color": "#2C7A7A",
        "blurb": "1 commit per hour;\nsupervised models\nrelease per cohort\nor per visit",
    },
    {
        "name": "Repository scale",
        "color": "#2C7A4D",
        "blurb": "Holds the entire trial\nrepository in context;\nnarrow models hold\nonly training features",
    },
    {
        "name": "Multi-perspective",
        "color": "#B45424",
        "blurb": "Operates as site\ncoordinator, patient\norchestrator, or sponsor\ndecision agent",
    },
    {
        "name": "Local-friendly",
        "color": "#6A4C8C",
        "blurb": "Local re-run on Core\ni5-6200U with 4 GB RAM;\nsupervised often\nneeds GPU clusters",
    },
    {
        "name": "Auditable trail",
        "color": "#7A1F1F",
        "blurb": "GitHub commit log per\nhour; supervised models\ntypically release a\nsingle weight artifact",
    },
]


def main() -> None:
    fig, ax = plt.subplots(figsize=(9, 9), constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)

    n = len(WEDGES)
    angle_per = 360 / n

    for i, wedge in enumerate(WEDGES):
        theta1 = 90 - (i + 1) * angle_per
        theta2 = 90 - i * angle_per
        wedge_patch = mpatches.Wedge(
            (0, 0),
            0.95,
            theta1,
            theta2,
            width=0.55,
            facecolor=wedge["color"],
            edgecolor="white",
            linewidth=2.0,
            alpha=0.92,
        )
        ax.add_patch(wedge_patch)

        mid_angle = np.deg2rad((theta1 + theta2) / 2)
        label_r = 0.65
        ax.text(
            label_r * np.cos(mid_angle),
            label_r * np.sin(mid_angle),
            wedge["name"],
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            color="white",
        )

        text_r = 1.20
        text_x = text_r * np.cos(mid_angle)
        text_y = text_r * np.sin(mid_angle)
        ha = "left" if np.cos(mid_angle) > 0 else "right" if np.cos(mid_angle) < -0.1 else "center"
        ax.text(text_x, text_y, wedge["blurb"], ha=ha, va="center", fontsize=8, color=DARK)

    hub = mpatches.Circle((0, 0), 0.40, facecolor="white", edgecolor=HUB_NAVY, linewidth=2.0)
    ax.add_patch(hub)
    ax.text(0, 0.10, "1M Token", ha="center", va="center", fontsize=12, fontweight="bold", color=HUB_NAVY)
    ax.text(0, -0.10, "Repository\nContext", ha="center", va="center", fontsize=9, color=DARK)

    ax.set_title(
        "Value Proposition Wheel - 1M Token Repository Scale Context",
        fontsize=13,
        fontweight="bold",
        color=HUB_NAVY,
        pad=18,
    )

    fig.savefig(OUTPUT, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

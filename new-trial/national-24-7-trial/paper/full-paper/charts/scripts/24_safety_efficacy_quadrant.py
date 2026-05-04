"""Chart 24: Safety vs Efficacy quadrant (full page, NEW).

Adds a NEW 2x2 quadrant chart to Discussion 4.3 positioning the four
LLM simulations against named supervised baselines and multimodal
foundation models.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

OUTPUT = Path(__file__).resolve().parent.parent / "images" / "24_safety_efficacy_quadrant.png"

NAVY = "#1F4E79"
SIM_COLOR = "#2C7A4D"
SUPER = "#7C7C7C"
FOUND = "#6A4C8C"
DARK = "#1A1A1A"

POINTS = [
    ("Manz 2020", 3.5, 1.5, "circle", SUPER),
    ("SHIELD-RT 2020", 4.0, 2.0, "circle", SUPER),
    ("SCORPIO 2025", 2.5, 3.5, "circle", SUPER),
    ("PROGPATH 2025", 2.0, 4.0, "triangle", FOUND),
    ("AIM-LCpro 2025", 2.5, 4.0, "triangle", FOUND),
    ("Huang 2025 null", 1.5, 1.5, "circle", SUPER),
    ("Sim 1 (site)", 4.5, 4.0, "square", SIM_COLOR),
    ("Sim 2 (site)", 4.0, 4.5, "square", SIM_COLOR),
    ("Sim 3 (sponsor)", 4.5, 4.5, "square", SIM_COLOR),
    ("Sim 4 (sponsor)", 5.0, 4.5, "square", SIM_COLOR),
]


def main() -> None:
    fig, ax = plt.subplots(figsize=(9, 9), constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 5.5)
    ax.set_ylim(0, 5.5)
    ax.set_aspect("equal")

    ax.axhline(2.75, color="#7C7C7C", linewidth=0.8, linestyle="--")
    ax.axvline(2.75, color="#7C7C7C", linewidth=0.8, linestyle="--")

    quadrants = [
        (4.4, 5.2, "High safety + High efficacy\n(target zone)", "#E5F0E8"),
        (1.0, 5.2, "Low safety + High efficacy", "#F5EFEA"),
        (4.4, 0.30, "High safety + Low efficacy", "#EAF1F8"),
        (1.0, 0.30, "Low safety + Low efficacy", "#F5F0F0"),
    ]
    bg_quadrants = [
        ((2.75, 2.75), 2.75, 2.75, "#E5F0E8"),
        ((0, 2.75), 2.75, 2.75, "#F5EFEA"),
        ((2.75, 0), 2.75, 2.75, "#EAF1F8"),
        ((0, 0), 2.75, 2.75, "#F5F0F0"),
    ]
    for (x0, y0), w, h, fill in bg_quadrants:
        ax.add_patch(mpatches.Rectangle((x0, y0), w, h, facecolor=fill, edgecolor="none", alpha=0.6, zorder=0))
    for x, y, text, _ in quadrants:
        ax.text(x, y, text, fontsize=8.5, ha="center", color="#7C7C7C", style="italic", zorder=1)

    marker_map = {"circle": "o", "triangle": "^", "square": "s"}
    for label, x, y, shape, color in POINTS:
        ax.scatter(x, y, marker=marker_map[shape], s=180, color=color, edgecolor=DARK, linewidth=1.0, zorder=4)
        offset = (0.18, 0.18)
        if "Sim 4" in label:
            offset = (0.18, -0.18)
        elif "Sim 3" in label:
            offset = (0.18, 0.05)
        elif "PROGPATH" in label:
            offset = (-0.15, 0.18)
        ax.text(
            x + offset[0],
            y + offset[1],
            label,
            fontsize=9,
            color=DARK,
            fontweight="bold" if "Sim" in label else "normal",
            zorder=5,
        )

    ax.set_xlabel("Safety prediction depth (0 to 5)", fontsize=11, color=NAVY)
    ax.set_ylabel("Efficacy prediction depth (0 to 5)", fontsize=11, color=NAVY)
    ax.set_xticks(range(0, 6))
    ax.set_yticks(range(0, 6))
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    ax.set_title(
        "Safety vs Efficacy Prediction Depth\nFour LLM Simulations vs Supervised and Foundation Baselines",
        fontsize=13,
        fontweight="bold",
        color=NAVY,
    )

    legend_handles = [
        plt.Line2D(
            [0], [0], marker="o", color="white", markerfacecolor=SUPER, markersize=11, label="Supervised baselines"
        ),
        plt.Line2D(
            [0], [0], marker="^", color="white", markerfacecolor=FOUND, markersize=11, label="Multimodal foundation"
        ),
        plt.Line2D(
            [0], [0], marker="s", color="white", markerfacecolor=SIM_COLOR, markersize=11, label="Four LLM simulations"
        ),
    ]
    ax.legend(handles=legend_handles, loc="lower left", frameon=False, fontsize=9.5)

    fig.savefig(OUTPUT, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

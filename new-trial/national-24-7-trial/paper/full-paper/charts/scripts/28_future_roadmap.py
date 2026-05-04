"""Chart 28: Future deliverables roadmap (NEW)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

OUTPUT = Path(__file__).resolve().parent.parent / "images" / "28_future_roadmap.png"

NAVY = "#1F4E79"
TEAL = "#2C7A7A"
GOLD = "#B45424"
GREEN = "#2C7A4D"
DARK = "#1A1A1A"
GRID = "#E6E6E6"


def main() -> None:
    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax.set_xlim(-0.5, 12.5)
    ax.set_ylim(0, 4.4)

    for m in range(0, 13):
        ax.axvline(m, color=GRID, linewidth=0.6)

    ax.add_patch(
        mpatches.FancyBboxPatch(
            (-0.4, 2.45),
            0.6,
            1.65,
            boxstyle="round,pad=0.02,rounding_size=0.04",
            linewidth=1.2,
            edgecolor=NAVY,
            facecolor=NAVY,
        )
    )
    ax.text(
        -0.10, 3.27, "Track A", color="white", rotation=90, fontsize=11, fontweight="bold", ha="center", va="center"
    )

    ax.add_patch(
        mpatches.FancyBboxPatch(
            (-0.4, 0.50),
            0.6,
            1.65,
            boxstyle="round,pad=0.02,rounding_size=0.04",
            linewidth=1.2,
            edgecolor=TEAL,
            facecolor=TEAL,
        )
    )
    ax.text(
        -0.10, 1.32, "Track B", color="white", rotation=90, fontsize=11, fontweight="bold", ha="center", va="center"
    )

    ax.add_patch(
        mpatches.FancyBboxPatch(
            (0.20, 3.30),
            5.80,
            0.55,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            linewidth=1.0,
            edgecolor=NAVY,
            facecolor="#E6EEF7",
        )
    )
    ax.text(
        3.10,
        3.58,
        "D1: TRIPOD+AI retrospective validation (Sim 1 hour-resolution narrative)",
        ha="center",
        va="center",
        fontsize=9,
        color=DARK,
    )

    ax.add_patch(
        mpatches.FancyBboxPatch(
            (8.00, 2.65),
            4.20,
            0.55,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            linewidth=1.0,
            edgecolor=NAVY,
            facecolor="#E6EEF7",
        )
    )
    ax.text(
        10.10, 2.93, "D3: FDA RTCT pilot submission (Track A path)", ha="center", va="center", fontsize=9, color=DARK
    )

    ax.add_patch(
        mpatches.FancyBboxPatch(
            (2.00, 1.20),
            7.00,
            0.55,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            linewidth=1.0,
            edgecolor=TEAL,
            facecolor="#E6F0EF",
        )
    )
    ax.text(
        5.50,
        1.48,
        "D2: Public benchmark vs supervised baselines (Manz, SHIELD-RT, SCORPIO, PROGPATH)",
        ha="center",
        va="center",
        fontsize=9,
        color=DARK,
    )

    ax.add_patch(
        mpatches.FancyBboxPatch(
            (8.00, 0.60),
            4.20,
            0.55,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            linewidth=1.0,
            edgecolor=TEAL,
            facecolor="#E6F0EF",
        )
    )
    ax.text(
        10.10, 0.88, "D3: FDA RTCT pilot submission (Track B path)", ha="center", va="center", fontsize=9, color=DARK
    )

    milestones = [(0, "M0 Kickoff"), (3, "M1 Cohort"), (5, "M2 Protocol"), (12, "M3 Submission")]
    for m, label in milestones:
        ax.plot(m, 4.20, marker="D", color=GREEN, markersize=10)
        ax.text(m, 4.30, label, ha="center", fontsize=8, fontweight="bold", color=GREEN)

    ax.set_xticks(range(0, 13))
    ax.set_xticklabels([f"M{i}" for i in range(0, 13)], fontsize=8.5)
    ax.set_xlabel("Months from May 2026", fontsize=10)
    ax.set_yticks([])
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)

    ax.set_title(
        "Track A and Track B Future Deliverables Roadmap (12 Months)",
        fontsize=13,
        fontweight="bold",
        color=NAVY,
    )

    fig.savefig(OUTPUT, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

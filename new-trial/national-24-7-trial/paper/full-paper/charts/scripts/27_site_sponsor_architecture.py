"""Chart 27: Site / Sponsor / FDA RTCT architecture (full page, NEW)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

OUTPUT = Path(__file__).resolve().parent.parent / "images" / "27_site_sponsor_architecture.png"

NAVY = "#1F4E79"
GOLD = "#B45424"
TEAL = "#2C7A7A"
GREEN_DARK = "#1F4E2C"
DARK = "#1A1A1A"


def _box(ax, x, y, w, h, title, body, color):
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.04",
            linewidth=1.6,
            edgecolor=color,
            facecolor="white",
        )
    )
    ax.add_patch(mpatches.Rectangle((x, y + h - 0.50), w, 0.50, color=color))
    ax.text(x + w / 2, y + h - 0.25, title, ha="center", va="center", color="white", fontsize=11, fontweight="bold")
    ax.text(x + w / 2, y + (h - 0.50) / 2, body, ha="center", va="center", fontsize=9, color=DARK)


def _arrow(ax, x1, y1, x2, y2, color=DARK, lw=1.4):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops={"arrowstyle": "->", "color": color, "lw": lw})


def main() -> None:
    fig, ax = plt.subplots(figsize=(11, 8.5), constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 8.5)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.text(
        5.5,
        8.20,
        "Site / Sponsor / FDA RTCT Architecture - Four Simulations Plus Two Connection Buses",
        fontsize=13.5,
        fontweight="bold",
        ha="center",
        color=NAVY,
    )

    _box(ax, 0.3, 7.10, 10.4, 0.85, "FDA RTCT Real-Time API", "Median ack latency 13 s", GREEN_DARK)

    _box(
        ax,
        0.3,
        4.40,
        3.4,
        2.10,
        "Sim 1 - Continuous National 24/7 RTCT",
        "4 sites (Houston, Philadelphia,\nBoston, Texas Med Ctr)\n116 robot instances\nMinute-resolution narrative",
        NAVY,
    )
    _box(
        ax,
        0.3,
        1.50,
        3.4,
        2.10,
        "Sim 2 - Single-Patient 10-Stage Journey",
        "PAT-2026-0042 (58F NSCLC)\n1,120 days, 10 stages\nUSL 86.0 to 88.75",
        NAVY,
    )
    ax.text(2.0, 7.00, "Sites - Signal Origination", fontsize=11, fontweight="bold", color=NAVY, ha="center")

    _box(
        ax,
        4.20,
        3.50,
        2.6,
        2.50,
        "Paradigm Health\nAggregator",
        "TRAVERSE\nSTREAM-SCLC\nTRAVERSE-PED\nNETWORK-OBS",
        TEAL,
    )

    _box(
        ax,
        7.30,
        4.40,
        3.4,
        2.10,
        "Sim 3 - 24-Hour Autonomous Sponsor",
        "53 agents in 4 layers\n24 JSON outputs, 75 ASCII\n5 FastAPI endpoints",
        GOLD,
    )
    _box(
        ax,
        7.30,
        1.50,
        3.4,
        2.10,
        "Sim 4 - 168-Hour 7-Day Sponsor Extension",
        "168 scripts, 168 JSON, 525 ASCII\n7 daily summaries, 168 commits\nLocal verif on Core i5-6200U",
        GOLD,
    )
    ax.text(9.0, 7.00, "Sponsors - Decision Origination", fontsize=11, fontweight="bold", color=GOLD, ha="center")

    _arrow(ax, 3.70, 5.45, 4.20, 4.75, color=NAVY)
    _arrow(ax, 3.70, 2.55, 4.20, 4.45, color=NAVY)
    _arrow(ax, 6.80, 4.75, 7.30, 5.45, color=GOLD)
    _arrow(ax, 6.80, 4.45, 7.30, 2.55, color=GOLD)
    _arrow(ax, 5.50, 6.00, 5.50, 7.10, color=GREEN_DARK, lw=2.0)
    _arrow(ax, 9.00, 6.50, 9.00, 7.10, color=GREEN_DARK, lw=1.6)
    _arrow(ax, 9.00, 3.60, 9.00, 7.10, color=GREEN_DARK, lw=1.0)

    ax.add_patch(
        mpatches.FancyBboxPatch(
            (0.30, 0.10),
            10.40,
            1.10,
            boxstyle="round,pad=0.02,rounding_size=0.04",
            linewidth=1.0,
            edgecolor=NAVY,
            facecolor="#FFF8E1",
        )
    )
    ax.text(
        5.5,
        0.85,
        "Sites generate the patient and robot signal stream;",
        ha="center",
        fontsize=10,
        fontweight="bold",
        color=NAVY,
    )
    ax.text(
        5.5,
        0.55,
        "sponsors generate the governance and regulatory decision stream;",
        ha="center",
        fontsize=10,
        color=DARK,
    )
    ax.text(
        5.5,
        0.27,
        "the FDA RTCT pilot ingests both as a single real-time view.",
        ha="center",
        fontsize=10,
        color=DARK,
        fontweight="bold",
    )

    fig.savefig(OUTPUT, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

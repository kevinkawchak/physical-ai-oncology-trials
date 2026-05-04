"""Chart 30: RTCT signal flow (NEW)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

OUTPUT = Path(__file__).resolve().parent.parent / "images" / "30_rtct_signal_flow.png"

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
            linewidth=1.4,
            edgecolor=color,
            facecolor="white",
        )
    )
    ax.add_patch(mpatches.Rectangle((x, y + h - 0.40), w, 0.40, color=color))
    ax.text(x + w / 2, y + h - 0.20, title, ha="center", va="center", color="white", fontsize=10, fontweight="bold")
    if body:
        ax.text(x + w / 2, y + (h - 0.40) / 2, body, ha="center", va="center", fontsize=8.5, color=DARK)


def _arrow(ax, x1, y1, x2, y2, color=DARK, lw=1.2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops={"arrowstyle": "->", "color": color, "lw": lw})


def main() -> None:
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    ax.text(
        5.0,
        5.65,
        "RTCT Signal Flow - Sites and Sponsors Through Paradigm Health to FDA RTCT API",
        fontsize=12.5,
        fontweight="bold",
        ha="center",
        color=NAVY,
    )

    sites = [
        ("SITE-A Houston", "29 robots"),
        ("SITE-B Philadelphia", "29 robots"),
        ("SITE-C Boston", "29 robots"),
        ("SITE-D Texas Med Ctr", "29 robots"),
    ]
    for i, (name, body) in enumerate(sites):
        y = 4.40 - i * 1.00
        _box(ax, 0.20, y, 2.00, 0.85, name, body, NAVY)
        _arrow(ax, 2.20, y + 0.42, 4.10, 3.00, color=NAVY)

    bus = mpatches.FancyBboxPatch(
        (4.10, 0.60),
        1.80,
        4.65,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=1.4,
        edgecolor=TEAL,
        facecolor="white",
    )
    ax.add_patch(bus)
    ax.add_patch(mpatches.Rectangle((4.10, 4.85), 1.80, 0.40, color=TEAL))
    ax.text(5.0, 5.05, "Paradigm Health", ha="center", color="white", fontweight="bold", fontsize=10)
    channels = [
        ("TRAVERSE", 4.40),
        ("STREAM-SCLC", 3.40),
        ("TRAVERSE-PED", 2.40),
        ("NETWORK-OBS", 1.40),
    ]
    for label, y in channels:
        ax.add_patch(
            mpatches.FancyBboxPatch(
                (4.30, y - 0.20),
                1.40,
                0.60,
                boxstyle="round,pad=0.02,rounding_size=0.06",
                linewidth=0.8,
                edgecolor=TEAL,
                facecolor="#E6F0EF",
            )
        )
        ax.text(5.00, y + 0.10, label, ha="center", va="center", fontsize=8.5, fontweight="bold", color=TEAL)
        ax.text(5.00, y - 0.06, "[UP]", ha="center", va="center", fontsize=7.5, color=GREEN_DARK)

    _box(ax, 7.40, 2.40, 2.50, 1.30, "FDA RTCT API", "Median ack latency 13 s", GREEN_DARK)
    _arrow(ax, 5.90, 3.05, 7.40, 3.05, color=GREEN_DARK, lw=1.6)

    sponsors = [("Sim 3 (24h sponsor)", 0.40), ("Sim 4 (168h sponsor)", 4.30)]
    for label, x in sponsors:
        _box(ax, x, 0.05, 3.30, 0.55, label, "", GOLD)
        _arrow(ax, x + 1.65, 0.60, 5.0, 1.20, color=GOLD)

    ax.text(8.65, 2.10, "Median 13 s ack", fontsize=8, color=GREEN_DARK, ha="center", style="italic")

    fig.savefig(OUTPUT, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

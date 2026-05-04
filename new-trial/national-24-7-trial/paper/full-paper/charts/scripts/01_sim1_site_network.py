"""Chart 01: Simulation 1 site network diagram (Hour 00 cold start, full page).

Replaces the hour-00 facility ASCII block and Table sim1-hour00-network in
Results 3.1. Renders four site cards with robot inventory, the Paradigm
Health aggregator, and the FDA real-time API channel at 300 DPI.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

OUTPUT = Path(__file__).resolve().parent.parent / "images" / "01_sim1_site_network.png"

NAVY = "#1F4E79"
MID_BLUE = "#4A7BAA"
PALE_BLUE = "#B6CFE6"
ACCENT_RED = "#C9302C"
DARK_TEXT = "#1A1A1A"
LIGHT_FILL = "#F5F8FB"


def _site_card(ax, x, y, w, h, title, lines, accent):
    box = mpatches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=1.2,
        edgecolor=NAVY,
        facecolor=LIGHT_FILL,
    )
    ax.add_patch(box)
    accent_bar = mpatches.Rectangle((x, y + h - 0.55), w, 0.55, color=accent, alpha=0.85)
    ax.add_patch(accent_bar)
    ax.text(
        x + 0.18,
        y + h - 0.28,
        title,
        fontsize=11,
        fontweight="bold",
        color="white",
        va="center",
    )
    for i, line in enumerate(lines):
        ax.text(
            x + 0.18,
            y + h - 0.85 - 0.30 * i,
            line,
            fontsize=8.5,
            color=DARK_TEXT,
            va="center",
        )


def main() -> None:
    fig, ax = plt.subplots(figsize=(11, 8.5), constrained_layout=True)
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 8.5)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(
        5.5,
        8.05,
        "National 24/7 Continuous RTCT - Hour 00 Network Status",
        fontsize=16,
        fontweight="bold",
        ha="center",
        color=NAVY,
    )
    ax.text(
        5.5,
        7.70,
        "00:00 to 00:59 UTC, Day 1 Cold Start",
        fontsize=11,
        ha="center",
        color=DARK_TEXT,
    )

    site_a_lines = [
        "Patients on site: 6   Active: 2 (00:09, 00:42)",
        "Robots: 29   COBOT-01 ACTV   RTPOS-02 ACTV",
        "TRACK-02 ACTV    SURG, NEEDLE, IMAGE standby",
        "Signal link UP   latency 142 ms",
    ]
    site_b_lines = [
        "Patients on site: 4   Active: 1 (00:57)",
        "Robots: 29   COBOT-02 ACTV",
        "All other stations standby",
        "Signal link UP   latency 168 ms",
    ]
    site_c_lines = [
        "Patients on site: 3   Active: 1 (00:23)",
        "Robots: 29   IMAGE-01 ACTV",
        "All other stations standby",
        "Signal link UP   latency 201 ms",
    ]
    site_d_lines = [
        "Patients on site: 2   Active: 0",
        "Robots: 29   NEEDLE-02 calibration 00:06 to 00:25",
        "COMPN-02 pediatric passive monitoring",
        "Signal link UP   latency 119 ms",
    ]

    _site_card(ax, 0.4, 4.4, 5.05, 2.8, "SITE-A Houston (TRAVERSE)", site_a_lines, NAVY)
    _site_card(ax, 5.55, 4.4, 5.05, 2.8, "SITE-B Philadelphia (TRAVERSE)", site_b_lines, MID_BLUE)
    _site_card(ax, 0.4, 1.5, 5.05, 2.8, "SITE-C Boston (STREAM-SCLC)", site_c_lines, NAVY)
    _site_card(ax, 5.55, 1.5, 5.05, 2.8, "SITE-D Texas Med Ctr (TRAVERSE-PED)", site_d_lines, MID_BLUE)

    aggregator = mpatches.FancyBboxPatch(
        (0.4, 0.25),
        10.20,
        1.05,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=1.4,
        edgecolor=ACCENT_RED,
        facecolor=PALE_BLUE,
    )
    ax.add_patch(aggregator)
    ax.text(
        5.5,
        1.05,
        "Paradigm Health Aggregator -> FDA Real-Time API",
        fontsize=12,
        fontweight="bold",
        ha="center",
        color=NAVY,
    )
    ax.text(
        5.5,
        0.72,
        "Channels: TRAVERSE [UP]   STREAM-SCLC [UP]   TRAVERSE-PED [UP]   NETWORK-OBS [UP]",
        fontsize=9.5,
        ha="center",
        color=DARK_TEXT,
    )
    ax.text(
        5.5,
        0.42,
        "Median FDA acknowledgement latency 13 s   3 endpoints validated + 1 in progress",
        fontsize=9,
        ha="center",
        color=DARK_TEXT,
    )

    ax.text(
        5.5,
        7.40,
        "Network: 15 patients   4 active procedures (3 done + 1 ongoing)   116 robot instances   4 RTCT endpoints",
        fontsize=9.5,
        ha="center",
        style="italic",
        color=DARK_TEXT,
    )

    fig.savefig(OUTPUT, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

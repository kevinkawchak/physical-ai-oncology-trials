"""Chart 08: Simulation 4 daily metrics dashboard (full page).

Replaces tab:sim4-daily in Results 3.4 with a full-page dashboard.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

OUTPUT = Path(__file__).resolve().parent.parent / "images" / "08_sim4_daily_metrics_dashboard.png"

NAVY = "#1F4E79"
GREEN = "#2C7A4D"
GOLD = "#B45424"
LIGHT = "#F0F0F0"
DARK = "#1A1A1A"

DAYS = ["Day 1", "Day 2", "Day 3", "Day 4", "Day 5", "Day 6", "Day 7"]
HOURS = ["H000-H023", "H024-H047", "H048-H071", "H072-H095", "H096-H119", "H120-H143", "H144-H167"]
PATIENTS = [168, 195, 218, 200, 195, 210, 150]
PSL_START = [63.4, 64.8, 66.0, 67.0, 68.0, 68.8, 69.5]
PSL_END = [64.8, 65.9, 67.0, 67.9, 68.8, 69.5, 70.0]
DECISIONS = [288, 285, 290, 290, 295, 290, 278]
EVENTS = [
    "Initialization cohort",
    "Enrollment ramp",
    "Safety day, 22 escalations",
    "Mid-week steady state",
    "Analysis day, biostats peak",
    "Pre-closeout audit",
    "Closeout, audit signoff",
]


def _kpi_card(ax, x, y, w, h, label, value):
    box = mpatches.FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.04", linewidth=1.4, edgecolor=NAVY, facecolor=LIGHT
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h * 0.62, value, fontsize=22, fontweight="bold", ha="center", va="center", color=NAVY)
    ax.text(x + w / 2, y + h * 0.22, label, fontsize=10, ha="center", va="center", color=DARK)


def main() -> None:
    fig = plt.figure(figsize=(11, 8.5), constrained_layout=True)
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(4, 6, height_ratios=[0.7, 1.6, 1.6, 1.0])

    fig.suptitle(
        "Simulation 4 - 168 Hour 7-Day Sponsor Extension Daily Metrics",
        fontsize=15,
        fontweight="bold",
        color=NAVY,
    )

    ax_kpi = fig.add_subplot(gs[0, :])
    ax_kpi.set_xlim(0, 12)
    ax_kpi.set_ylim(0, 1)
    ax_kpi.axis("off")
    _kpi_card(ax_kpi, 0.2, 0.1, 3.6, 0.8, "Sponsor Decisions", "2,016")
    _kpi_card(ax_kpi, 4.2, 0.1, 3.6, 0.8, "Patients Processed", "1,336")
    _kpi_card(ax_kpi, 8.2, 0.1, 3.6, 0.8, "Escalations (Day 3 peak)", "125")

    ax_bar = fig.add_subplot(gs[1, :3])
    x = np.arange(len(DAYS))
    width = 0.35
    ax_bar.bar(x - width / 2, PATIENTS, width, color=NAVY, label="Patients in scope")
    ax_bar.set_ylabel("Patients", color=NAVY, fontsize=10)
    ax_bar.tick_params(axis="y", labelcolor=NAVY)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(DAYS, fontsize=9)
    ax_bar2 = ax_bar.twinx()
    ax_bar2.bar(x + width / 2, DECISIONS, width, color=GOLD, label="Daily decisions")
    ax_bar2.set_ylabel("Daily decisions", color=GOLD, fontsize=10)
    ax_bar2.tick_params(axis="y", labelcolor=GOLD)
    ax_bar.set_title("Daily Patients and Decisions", fontsize=11, fontweight="bold", color=DARK)
    for spine in ("top",):
        ax_bar.spines[spine].set_visible(False)
        ax_bar2.spines[spine].set_visible(False)

    ax_psl = fig.add_subplot(gs[1, 3:])
    psl_mid = [(s + e) / 2 for s, e in zip(PSL_START, PSL_END)]
    ax_psl.plot(x, psl_mid, color=GREEN, marker="o", linewidth=2.4, markersize=8)
    ax_psl.fill_between(x, PSL_START, PSL_END, color=GREEN, alpha=0.18)
    ax_psl.set_xticks(x)
    ax_psl.set_xticklabels(DAYS, fontsize=9)
    ax_psl.set_ylim(60, 72)
    ax_psl.set_ylabel("PSL Score", fontsize=10)
    ax_psl.set_title("PSL Trajectory: 63.4 -> 70.0 (+6.6)", fontsize=11, fontweight="bold", color=GREEN)
    for spine in ("top", "right"):
        ax_psl.spines[spine].set_visible(False)
    ax_psl.annotate(
        "Start 63.4",
        (0, PSL_START[0]),
        textcoords="offset points",
        xytext=(8, -10),
        fontsize=8,
        color=DARK,
    )
    ax_psl.annotate(
        "End 70.0",
        (6, PSL_END[6]),
        textcoords="offset points",
        xytext=(-30, 6),
        fontsize=8,
        fontweight="bold",
        color=GREEN,
    )

    ax_table = fig.add_subplot(gs[2:, :])
    ax_table.axis("off")
    ax_table.set_xlim(0, 10)
    ax_table.set_ylim(0, 9)
    headers = ["Day", "Hour Range", "Patients", "PSL Start", "PSL End", "Decisions", "Notable Event"]
    xs = [0.3, 1.6, 3.4, 4.4, 5.4, 6.4, 7.5]
    ax_table.add_patch(mpatches.Rectangle((0, 8.0), 10, 0.8, color=NAVY))
    for x_pos, h in zip(xs, headers):
        ax_table.text(x_pos, 8.4, h, color="white", fontweight="bold", fontsize=9.5, va="center")
    for i, day in enumerate(DAYS):
        y = 7.3 - i * 0.95
        if i % 2 == 0:
            ax_table.add_patch(mpatches.Rectangle((0, y - 0.05), 10, 0.92, color=LIGHT, zorder=0))
        ax_table.text(xs[0], y + 0.4, day, fontsize=9, va="center", color=DARK)
        ax_table.text(xs[1], y + 0.4, HOURS[i], fontsize=9, va="center", color=DARK)
        ax_table.text(xs[2], y + 0.4, str(PATIENTS[i]), fontsize=9, va="center", color=DARK)
        ax_table.text(xs[3], y + 0.4, f"{PSL_START[i]:.1f}", fontsize=9, va="center", color=DARK)
        ax_table.text(xs[4], y + 0.4, f"{PSL_END[i]:.1f}", fontsize=9, va="center", color=GREEN, fontweight="bold")
        ax_table.text(xs[5], y + 0.4, str(DECISIONS[i]), fontsize=9, va="center", color=DARK)
        ax_table.text(xs[6], y + 0.4, EVENTS[i], fontsize=9, va="center", color=DARK)

    fig.savefig(OUTPUT, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

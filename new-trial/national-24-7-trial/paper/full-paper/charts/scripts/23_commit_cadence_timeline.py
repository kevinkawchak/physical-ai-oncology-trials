"""Chart 23: Commit cadence timeline (NEW).

Adds a NEW timeline to Conclusions that contrasts the 168 commits in 7
days from Sim 4 against 6 supervised baseline papers across 5 years.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

OUTPUT = Path(__file__).resolve().parent.parent / "images" / "23_commit_cadence_timeline.png"

NAVY = "#1F4E79"
GREEN = "#2C7A4D"
GRAY = "#7C7C7C"
GRID = "#E6E6E6"
DARK = "#1A1A1A"

BASELINES = [
    (2020.4, "Manz 2020"),
    (2020.7, "SHIELD-RT"),
    (2025.1, "SCORPIO"),
    (2025.3, "PROGPATH"),
    (2025.5, "AIM-LCpro"),
    (2025.8, "Huang 2025"),
]


def main() -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.5), constrained_layout=True, gridspec_kw={"hspace": 0.55})
    fig.patch.set_facecolor("white")
    fig.suptitle(
        "Hourly Commit Cadence vs Per-Cohort Release Cadence",
        fontsize=13.5,
        fontweight="bold",
        color=NAVY,
    )

    ax_top = axes[0]
    ax_top.set_xlim(-3, 170)
    ax_top.set_ylim(-1, 1.5)
    for d in range(0, 169, 24):
        ax_top.axvline(d, color=GRID, linewidth=0.6)
    for h in range(168):
        ax_top.scatter(h, 0, color=GREEN, s=10, zorder=3)
    ax_top.set_yticks([])
    ax_top.set_xticks([0, 24, 48, 72, 96, 120, 144, 167])
    ax_top.set_xticklabels(["H0", "H24", "H48", "H72", "H96", "H120", "H144", "H167"], fontsize=9)
    ax_top.set_xlabel("Hour (Simulation 4 7-day window)", fontsize=10)
    ax_top.set_title(
        "Four LLM Simulations - 1 commit per hour (168 commits in 7 days)", fontsize=11, fontweight="bold", color=GREEN
    )
    for spine in ("top", "right", "left"):
        ax_top.spines[spine].set_visible(False)
    ax_top.text(170, 0.0, "168 commits", color=GREEN, fontsize=9, fontweight="bold", va="center")

    ax_bot = axes[1]
    ax_bot.set_xlim(2019.5, 2026.5)
    ax_bot.set_ylim(-1, 1.5)
    for y in (2020, 2021, 2022, 2023, 2024, 2025, 2026):
        ax_bot.axvline(y, color=GRID, linewidth=0.6)
    for x, label in BASELINES:
        ax_bot.scatter(x, 0, color=GRAY, s=70, zorder=3)
        ax_bot.text(x, 0.45, label, fontsize=8.5, ha="center", color=DARK, rotation=20)
    ax_bot.set_yticks([])
    ax_bot.set_xticks([2020, 2021, 2022, 2023, 2024, 2025, 2026])
    ax_bot.set_xticklabels(["2020", "2021", "2022", "2023", "2024", "2025", "2026"], fontsize=9)
    ax_bot.set_xlabel("Year (supervised baseline release timeline)", fontsize=10)
    ax_bot.set_title(
        "Supervised Baselines - 1 release per cohort or per paper (6 papers in 5 years)",
        fontsize=11,
        fontweight="bold",
        color=GRAY,
    )
    for spine in ("top", "right", "left"):
        ax_bot.spines[spine].set_visible(False)
    ax_bot.text(2026.6, 0.0, "6 papers", color=GRAY, fontsize=9, fontweight="bold", va="center")

    fig.text(
        0.5,
        -0.02,
        "168 commits in 7 days from Sim 4 versus 6 papers in 5 years from the supervised baseline set.",
        ha="center",
        fontsize=9.5,
        style="italic",
        color=DARK,
    )

    rect = mpatches.FancyBboxPatch(
        (0.05, 0.05),
        0.90,
        0.05,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=0,
        facecolor="none",
    )
    fig.patches.append(rect)

    fig.savefig(OUTPUT, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

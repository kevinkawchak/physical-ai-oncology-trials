"""Chart 20: Financial assessment dashboard (full page, NEW).

Adds a NEW financial assessment dashboard to Discussion 4.3 with KPI
cards, per-stage savings driver bars, and per-simulation cost-to-
reproduce horizontal bars.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

OUTPUT = Path(__file__).resolve().parent.parent / "images" / "20_financial_assessment_dashboard.png"

NAVY = "#1F4E79"
GREEN = "#2C7A4D"
GOLD = "#B45424"
LIGHT = "#F5F5F5"
DARK = "#1A1A1A"

PER_STAGE_SAVINGS = [0.02, 0.03, 0.04, 0.05, 0.07, 0.06, 0.08, 0.02, 0.02, 0.01]
PER_SIM_COSTS = {
    "Sim 1 (cloud only, 56h)": 0.020,
    "Sim 2 (cloud + light local)": 0.012,
    "Sim 3 (cloud only, 24h)": 0.016,
    "Sim 4 (cloud + Core i5-6200U)": 0.008,
}


def _kpi_card(ax, x, y, w, h, label, value, color):
    box = mpatches.FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.04", linewidth=1.4, edgecolor=color, facecolor=LIGHT
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h * 0.62, value, fontsize=22, fontweight="bold", ha="center", va="center", color=color)
    ax.text(x + w / 2, y + h * 0.22, label, fontsize=9.5, ha="center", va="center", color=DARK)


def main() -> None:
    fig = plt.figure(figsize=(11, 8.5), constrained_layout=True)
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(3, 2, height_ratios=[0.7, 1.4, 0.4])

    fig.suptitle(
        "Financial Assessment Dashboard - FDA RTCT $390M to $650M Per-Trial Cost Reduction",
        fontsize=14,
        fontweight="bold",
        color=NAVY,
    )

    ax_kpi = fig.add_subplot(gs[0, :])
    ax_kpi.set_xlim(0, 12)
    ax_kpi.set_ylim(0, 1)
    ax_kpi.axis("off")
    _kpi_card(ax_kpi, 0.2, 0.1, 3.6, 0.85, "Baseline Trial Cost", "$1.30B", NAVY)
    _kpi_card(ax_kpi, 4.2, 0.1, 3.6, 0.85, "FDA-cited Reduction Band", "$390M-$650M", GOLD)
    _kpi_card(ax_kpi, 8.2, 0.1, 3.6, 0.85, "Per-Patient Run (Sim 2)", "$0.91M", GREEN)

    ax_stage = fig.add_subplot(gs[1, 0])
    stages = list(range(1, 11))
    bars = ax_stage.bar(stages, PER_STAGE_SAVINGS, color=GREEN, edgecolor=NAVY, linewidth=0.6)
    for bar, val in zip(bars, PER_STAGE_SAVINGS):
        ax_stage.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.002,
            f"{val:.2f}",
            ha="center",
            fontsize=8,
            color=DARK,
        )
    ax_stage.set_xticks(stages)
    ax_stage.set_xticklabels([f"S{s}" for s in stages], fontsize=9)
    ax_stage.set_ylabel("Savings (M$ per patient)", fontsize=10)
    ax_stage.set_title("Per-Stage Savings Drivers (Sim 2)", fontsize=11, fontweight="bold", color=NAVY)
    for spine in ("top", "right"):
        ax_stage.spines[spine].set_visible(False)

    ax_sim = fig.add_subplot(gs[1, 1])
    sims = list(PER_SIM_COSTS.keys())
    costs = list(PER_SIM_COSTS.values())
    y_pos = np.arange(len(sims))
    colors = [NAVY, GREEN, GOLD, "#6A4C8C"]
    barh = ax_sim.barh(y_pos, costs, color=colors, edgecolor=DARK, linewidth=0.6)
    for bar, val in zip(barh, costs):
        ax_sim.text(
            bar.get_width() + 0.0005,
            bar.get_y() + bar.get_height() / 2,
            f"${val * 1000:.0f}K",
            va="center",
            fontsize=8.5,
            color=DARK,
        )
    ax_sim.set_yticks(y_pos)
    ax_sim.set_yticklabels(sims, fontsize=9)
    ax_sim.set_xlabel("Cost to Reproduce (M$, illustrative)", fontsize=10)
    ax_sim.set_title("Per-Simulation Cost to Reproduce", fontsize=11, fontweight="bold", color=NAVY)
    for spine in ("top", "right"):
        ax_sim.spines[spine].set_visible(False)
    ax_sim.invert_yaxis()

    ax_note = fig.add_subplot(gs[2, :])
    ax_note.set_xlim(0, 1)
    ax_note.set_ylim(0, 1)
    ax_note.axis("off")
    ax_note.add_patch(
        mpatches.FancyBboxPatch(
            (0.02, 0.20),
            0.96,
            0.60,
            boxstyle="round,pad=0.02,rounding_size=0.04",
            linewidth=1.2,
            edgecolor=NAVY,
            facecolor="#FFF8E1",
            transform=ax_note.transAxes,
        )
    )
    ax_note.text(
        0.5,
        0.50,
        "Track B (cloud + small local agents) is the architecture that maps the FDA reduction band to per-site hardware floors.",
        ha="center",
        va="center",
        fontsize=10,
        color=DARK,
        fontweight="bold",
        transform=ax_note.transAxes,
    )

    fig.savefig(OUTPUT, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

"""Chart 07: Simulation 3 combined hourly workload (full page).

Combines hour-00, hour-12, hour-23 sponsor workload tables in Results 3.3
into a single full-page comparison figure.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

OUTPUT = Path(__file__).resolve().parent.parent / "images" / "07_sim3_workload_combined.png"

NAVY = "#1F4E79"
GOLD = "#B45424"
TEAL = "#2C7A7A"
RED = "#C9302C"

AGENTS = [
    "portfolio_agent",
    "asset_lead_agent",
    "clinical_accountability_agent",
    "study_orchestrator",
    "clinops_agent",
    "safety_agent",
    "regulatory_agent",
    "quality_agent",
    "supply_agent",
    "data_biostats_agent",
    "site_gateway",
    "robot_execution_gateway",
]

HOUR_00 = [
    (2, 1, 0),
    (3, 2, 0),
    (1, 1, 0),
    (2, 2, 0),
    (3, 2, 0),
    (1, 1, 0),
    (2, 1, 0),
    (3, 2, 0),
    (1, 1, 0),
    (2, 2, 0),
    (3, 2, 0),
    (1, 1, 0),
]
HOUR_12 = [
    (3, 1, 0),
    (1, 1, 0),
    (4, 3, 1),
    (2, 2, 0),
    (5, 5, 0),
    (3, 3, 0),
    (1, 1, 0),
    (4, 4, 0),
    (2, 1, 0),
    (5, 5, 0),
    (3, 2, 0),
    (1, 1, 0),
]
HOUR_23 = [
    (1, 1, 0),
    (2, 2, 0),
    (3, 2, 0),
    (1, 1, 0),
    (2, 1, 0),
    (3, 2, 0),
    (1, 1, 0),
    (2, 2, 0),
    (3, 2, 0),
    (1, 1, 0),
    (2, 1, 0),
    (3, 2, 0),
]

PANELS = [
    ("Hour 00 (overnight, 2 patients in scope)", HOUR_00, NAVY),
    ("Hour 12 (mid-day peak, 5 patients in scope)", HOUR_12, GOLD),
    ("Hour 23 (overnight, day-end audit, 2 patients)", HOUR_23, TEAL),
]


def _draw_panel(ax, title, rows, color):
    y_pos = np.arange(len(AGENTS))[::-1]
    tasks = [r[0] for r in rows]
    decisions = [r[1] for r in rows]
    escalations = [r[2] for r in rows]
    ax.barh(y_pos, tasks, height=0.6, color=color, alpha=0.55, edgecolor=color, linewidth=0.6, label="Tasks")
    ax.barh(y_pos, decisions, height=0.36, color=color, edgecolor=color, label="Decisions")
    for i, esc in enumerate(escalations):
        if esc > 0:
            ax.scatter(rows[i][0] + 0.45, y_pos[i], color=RED, s=80, zorder=5, marker="X", linewidths=0)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(AGENTS, fontsize=8.5)
    ax.set_xlim(0, 6)
    ax.set_xticks(range(0, 7))
    ax.set_xticklabels(range(0, 7), fontsize=8)
    ax.set_xlabel("Count", fontsize=9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#7C7C7C")
    ax.spines["bottom"].set_color("#7C7C7C")
    ax.set_title(title, fontsize=10.5, fontweight="bold", color=color)
    ax.text(
        5.6,
        len(AGENTS) - 0.8,
        f"Tasks {sum(tasks)}   Decisions {sum(decisions)}   Esc {sum(escalations)}",
        fontsize=8.5,
        ha="right",
        va="top",
        color="#1A1A1A",
        bbox={"facecolor": "#F5F5F5", "edgecolor": color, "linewidth": 0.6, "boxstyle": "round,pad=0.3"},
    )


def main() -> None:
    fig, axes = plt.subplots(3, 1, figsize=(8.5, 11), constrained_layout=True)
    fig.patch.set_facecolor("white")
    fig.suptitle(
        "Simulation 3 Sponsor Agent Workload\nHour 00, Hour 12, and Hour 23 of the 24-Hour Run",
        fontsize=14,
        fontweight="bold",
        color="#1F4E79",
    )
    for ax, (title, rows, color) in zip(axes, PANELS):
        _draw_panel(ax, title, rows, color)

    legend_handles = [
        mpatches.Patch(facecolor=NAVY, alpha=0.55, label="Tasks (light fill)"),
        mpatches.Patch(facecolor=NAVY, label="Decisions (solid fill)"),
        mpatches.Patch(facecolor=RED, label="Escalation (X)"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=3,
        frameon=False,
        fontsize=9,
    )

    fig.savefig(OUTPUT, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

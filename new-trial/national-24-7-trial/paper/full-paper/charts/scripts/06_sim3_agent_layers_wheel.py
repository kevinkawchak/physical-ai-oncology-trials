"""Chart 06: Simulation 3 agent layers wheel.

Replaces tab:sim3-layers in Results 3.3 with a donut wheel for the four
agent layers (governance 6, study execution 12, site/robotics 18, trust
17) with example agents and hour-12 load annotations per layer.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

OUTPUT = Path(__file__).resolve().parent.parent / "images" / "06_sim3_agent_layers_wheel.png"

LAYERS = [
    {
        "name": "Governance",
        "count": 6,
        "color": "#1F4E79",
        "examples": "portfolio_agent\nasset_lead_agent\nclinical_accountability_agent",
        "load": "Hr-12: 8 tasks, 1 escalation",
    },
    {
        "name": "Study Execution",
        "count": 12,
        "color": "#2C7A7A",
        "examples": "study_orchestrator\nclinops_agent\nregulatory_agent\nquality_agent",
        "load": "Hr-12: 12 tasks",
    },
    {
        "name": "Site / Robotics",
        "count": 18,
        "color": "#B45424",
        "examples": "site_gateway\nrobot_execution_gateway\nsupply_agent",
        "load": "Hr-12: 6 tasks",
    },
    {
        "name": "Trust",
        "count": 17,
        "color": "#6A4C8C",
        "examples": "data_biostats_agent\nsafety_agent\naudit_trail_manager",
        "load": "Hr-12: 8 tasks",
    },
]


def main() -> None:
    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax.set_aspect("equal")
    ax.axis("off")

    sizes = [layer["count"] for layer in LAYERS]
    colors = [layer["color"] for layer in LAYERS]
    labels = [f"{layer['name']}\n{layer['count']} agents" for layer in LAYERS]

    wedges, texts, autotexts = ax.pie(
        sizes,
        colors=colors,
        labels=labels,
        startangle=90,
        wedgeprops={"width": 0.36, "edgecolor": "white", "linewidth": 2.0},
        autopct=lambda pct: f"{int(round(pct * 53 / 100))}",
        pctdistance=0.82,
        textprops={"fontsize": 10, "color": "#1A1A1A"},
    )
    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_fontweight("bold")
        autotext.set_fontsize(11)

    ax.text(0, 0.06, "53 Core", ha="center", va="center", fontsize=14, fontweight="bold", color="#1F4E79")
    ax.text(0, -0.10, "Sponsor Agents", ha="center", va="center", fontsize=10, color="#1A1A1A")

    ax.set_title(
        "53 Core Sponsor Agents Organized into Four Layers",
        fontsize=13,
        fontweight="bold",
        color="#1F4E79",
        pad=18,
    )

    leg_x = 1.50
    leg_y = 1.0
    for i, layer in enumerate(LAYERS):
        y = leg_y - i * 0.30
        ax.add_patch(mpatches.Rectangle((leg_x, y - 0.04), 0.05, 0.08, color=layer["color"], transform=ax.transAxes))
        ax.text(
            leg_x + 0.07,
            y + 0.01,
            layer["name"],
            transform=ax.transAxes,
            fontsize=10,
            fontweight="bold",
            color="#1A1A1A",
        )
        ax.text(
            leg_x + 0.07,
            y - 0.06,
            layer["examples"],
            transform=ax.transAxes,
            fontsize=7.5,
            color="#1A1A1A",
            verticalalignment="top",
        )
        ax.text(
            leg_x + 0.07,
            y - 0.22,
            layer["load"],
            transform=ax.transAxes,
            fontsize=7.5,
            color=layer["color"],
            fontstyle="italic",
        )

    fig.savefig(OUTPUT, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

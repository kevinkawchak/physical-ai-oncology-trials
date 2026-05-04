"""Chart 22: Multimodal inputs diagram (NEW).

Adds a NEW diagram to Methods 2.6 and Introduction 1.2 visualizing six
input modalities converging on the 1M token context.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

OUTPUT = Path(__file__).resolve().parent.parent / "images" / "22_multimodal_inputs_diagram.png"

NAVY = "#1F4E79"
DARK = "#1A1A1A"
HUB_FILL = "#2C3E50"
LIGHT = "#F0F0F0"

INPUTS = [
    ("Protocol text", "#1F4E79", "50K tokens", (0.5, 5.0)),
    ("EHR records (FHIR)", "#2C7A7A", "200K tokens", (0.5, 3.5)),
    ("Robot telemetry", "#2C7A4D", "80K tokens", (0.5, 2.0)),
    ("Regulatory adaptations", "#B45424", "120K tokens", (0.5, 0.5)),
    ("ASCII facility diagrams", "#6A4C8C", "30K tokens", (3.5, 5.5)),
    ("Python agent code", "#7A1F1F", "150K tokens", (3.5, 0.0)),
]

OUTPUTS = [
    ("Per-hour Markdown", "#1F4E79"),
    ("ASCII diagrams", "#2C7A7A"),
    ("Python agent scripts", "#2C7A4D"),
    ("JSON sponsor logs", "#B45424"),
]


def main() -> None:
    fig, ax = plt.subplots(figsize=(9.5, 6), constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.5)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.text(
        5.0,
        6.20,
        "Six Input Modalities Converge on 1M Token Context, Producing Four Output Artifacts",
        fontsize=12.5,
        fontweight="bold",
        ha="center",
        color=NAVY,
    )

    hub_x, hub_y, hub_r = 7.0, 3.0, 0.95
    hub = mpatches.Circle((hub_x, hub_y), hub_r, facecolor=HUB_FILL, edgecolor=NAVY, linewidth=2.0)
    ax.add_patch(hub)
    ax.text(hub_x, hub_y + 0.20, "1M Token", ha="center", color="white", fontweight="bold", fontsize=11)
    ax.text(hub_x, hub_y, "Repository", ha="center", color="white", fontsize=9.5)
    ax.text(hub_x, hub_y - 0.18, "Context", ha="center", color="white", fontsize=9.5)
    ax.text(hub_x, hub_y - 0.40, "Claude Opus 4.7", ha="center", color="#A4C9A8", fontsize=7.5, style="italic")

    for name, color, tokens, (x, y) in INPUTS:
        box = mpatches.FancyBboxPatch(
            (x, y),
            2.4,
            0.65,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            linewidth=1.2,
            edgecolor=color,
            facecolor=LIGHT,
        )
        ax.add_patch(box)
        ax.text(x + 1.2, y + 0.40, name, ha="center", fontsize=9, fontweight="bold", color=color)
        ax.text(x + 1.2, y + 0.15, tokens, ha="center", fontsize=8, color=DARK)
        ax.annotate(
            "",
            xy=(hub_x - hub_r * 0.85, hub_y),
            xytext=(x + 2.4, y + 0.32),
            arrowprops={"arrowstyle": "->", "color": color, "lw": 1.2, "alpha": 0.85},
        )

    output_y = 0.30
    box_w = 1.85
    for i, (name, color) in enumerate(OUTPUTS):
        x = hub_x - 1.5 + i * 0.30 - box_w / 2 + 1.0
        bx = 6.05 + i * 0.95 - 1.0
        ob = mpatches.FancyBboxPatch(
            (bx, output_y),
            0.90,
            0.42,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            linewidth=1.0,
            edgecolor=color,
            facecolor="white",
        )
        ax.add_patch(ob)
        ax.text(
            bx + 0.45, output_y + 0.21, name, ha="center", va="center", fontsize=7.5, color=color, fontweight="bold"
        )

    ax.annotate("", xy=(7.0, 0.85), xytext=(7.0, 2.10), arrowprops={"arrowstyle": "->", "color": NAVY, "lw": 1.3})
    ax.text(7.0, 1.55, "4 outputs / pass", fontsize=8, ha="center", style="italic", color=NAVY)

    fig.savefig(OUTPUT, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

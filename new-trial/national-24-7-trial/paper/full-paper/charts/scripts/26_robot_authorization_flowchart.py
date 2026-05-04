"""Chart 26: Robot authorization flowchart (NEW)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

OUTPUT = Path(__file__).resolve().parent.parent / "images" / "26_robot_authorization_flowchart.png"

NAVY = "#1F4E79"
GOLD = "#B45424"
GREEN = "#2C7A4D"
PURPLE = "#6A4C8C"
DARK = "#1A1A1A"


def _node(ax, x, y, w, h, text, color):
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.05",
            linewidth=1.4,
            edgecolor=color,
            facecolor="white",
        )
    )
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9, color=DARK)


def _diamond(ax, x, y, w, h, text, color):
    pts = [(x + w / 2, y + h), (x + w, y + h / 2), (x + w / 2, y), (x, y + h / 2)]
    ax.add_patch(mpatches.Polygon(pts, closed=True, facecolor="white", edgecolor=color, linewidth=1.4))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.5, color=DARK)


def _arrow(ax, x1, y1, x2, y2, label=None, color=DARK):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops={"arrowstyle": "->", "color": color, "lw": 1.2})
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.10, label, fontsize=8, ha="center", color=color, style="italic")


def main() -> None:
    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 6)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.text(
        5.5,
        5.65,
        "Robot Authorization Decision Pipeline (Simulation 3 + Simulation 4)",
        fontsize=13,
        fontweight="bold",
        ha="center",
        color=NAVY,
    )

    _node(ax, 0.10, 2.50, 1.50, 1.10, "Site emits\nrobot auth\nrequest", NAVY)
    _node(ax, 1.95, 2.50, 1.65, 1.10, "robot_execution_\ngateway", NAVY)
    _diamond(ax, 4.00, 2.50, 1.40, 1.10, "Borderline?", GOLD)
    _node(ax, 5.85, 4.00, 1.85, 0.95, "clinical_account-\nability_agent", GOLD)
    _diamond(ax, 5.85, 2.50, 1.85, 1.10, "AE-related?", GREEN)
    _node(ax, 7.95, 2.50, 1.65, 1.10, "safety_agent\n-> FDA RTCT", GREEN)
    _node(ax, 9.85, 2.50, 1.10, 1.10, "audit_trail_\nmanager", PURPLE)
    _node(ax, 4.30, 0.40, 4.20, 0.85, "Clearance bus -> site (authorized) or escalation queue (held)", NAVY)

    _arrow(ax, 1.60, 3.05, 1.95, 3.05)
    _arrow(ax, 3.60, 3.05, 4.00, 3.05)
    _arrow(ax, 5.40, 3.05, 5.85, 3.05, label="No")
    _arrow(ax, 4.70, 3.60, 4.70, 4.40)
    _arrow(ax, 4.70, 4.40, 5.85, 4.45, label="Yes (borderline)")
    _arrow(ax, 7.70, 4.45, 6.78, 3.60, label="reviewed")
    _arrow(ax, 7.70, 3.05, 7.95, 3.05, label="Yes")
    _arrow(ax, 9.60, 3.05, 9.85, 3.05)
    _arrow(ax, 6.78, 2.50, 6.40, 1.30, label="No (clearance)")
    _arrow(ax, 9.05, 2.50, 7.50, 1.30, label="cleared")
    _arrow(ax, 10.40, 2.50, 8.50, 1.30, label="audited")

    ax.add_patch(
        mpatches.FancyBboxPatch(
            (0.10, 0.05),
            4.05,
            1.10,
            boxstyle="round,pad=0.02,rounding_size=0.04",
            linewidth=1.0,
            edgecolor=GREEN,
            facecolor="#FFF8E1",
        )
    )
    ax.text(
        2.13,
        0.85,
        "Sim 4 Day 1 Volume",
        fontsize=9,
        ha="center",
        fontweight="bold",
        color=GREEN,
    )
    ax.text(
        2.13,
        0.50,
        "168 routine clearances\n6 borderline forwards reviewed\n0 escalations to safety_agent at H23",
        fontsize=8,
        ha="center",
        color=DARK,
    )

    fig.savefig(OUTPUT, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

"""Chart 13: Code-based vs text-only comparison chart.

Replaces the code-vs-text comparison block in Methods 2.7 and the README.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

OUTPUT = Path(__file__).resolve().parent.parent / "images" / "13_code_text_comparison.png"

NAVY = "#1F4E79"
ORANGE = "#B45424"
GREEN = "#2C7A4D"
DARK = "#1A1A1A"

ROWS = [
    ("Cloud compute use", "Light (text only)", "Moderate to heavy"),
    ("Local compute use", "None", "Yes (Sim 2 light, Sim 4 4GB)"),
    ("Downstream agents", "Cannot consume Python", "Can fan out to local agents"),
    ("Auditability", "Markdown plus ASCII", "Markdown plus ASCII plus .py"),
    ("Re-run determinism", "Re-run varies (LLM)", "Identical (.py is fixed)"),
    ("Verification cost", "Minimal", "i5-6200U / 4 GB tested"),
]


def _pill(ax, x, y, w, h, color, text):
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.10",
            linewidth=1.0,
            edgecolor=color,
            facecolor=color,
            alpha=0.10,
        )
    )
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.10",
            linewidth=1.0,
            edgecolor=color,
            facecolor="none",
        )
    )
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9, color=DARK)


def main() -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.5)
    ax.axis("off")

    ax.text(
        5.0,
        5.20,
        "Text-Only (Sim 1) vs Code-Based (Sims 2, 3, 4) Practical Trade-Offs",
        fontsize=12.5,
        fontweight="bold",
        ha="center",
        color=NAVY,
    )

    ax.add_patch(mpatches.Rectangle((3.05, 4.65), 3.0, 0.40, color=ORANGE))
    ax.add_patch(mpatches.Rectangle((6.20, 4.65), 3.0, 0.40, color=GREEN))
    ax.text(4.55, 4.85, "Text-Only (Sim 1)", ha="center", va="center", color="white", fontweight="bold", fontsize=10)
    ax.text(
        7.70,
        4.85,
        "Code-Based (Sims 2, 3, 4)",
        ha="center",
        va="center",
        color="white",
        fontweight="bold",
        fontsize=10,
    )

    row_h = 0.55
    for i, (prop, text_only, code_based) in enumerate(ROWS):
        y = 4.20 - i * row_h
        ax.text(0.20, y + row_h / 2, prop, fontsize=9.5, fontweight="bold", color=NAVY, va="center")
        _pill(ax, 3.05, y, 3.0, row_h - 0.10, ORANGE, text_only)
        _pill(ax, 6.20, y, 3.0, row_h - 0.10, GREEN, code_based)

    ax.add_patch(
        mpatches.FancyBboxPatch(
            (0.20, 0.20),
            9.60,
            0.55,
            boxstyle="round,pad=0.02,rounding_size=0.04",
            linewidth=1.2,
            edgecolor=NAVY,
            facecolor="#FFF8E1",
        )
    )
    ax.text(
        5.0,
        0.55,
        "Optimal pairing: Sim 1 (text-only narrative) + Sim 4 (code-based scripts).",
        ha="center",
        fontsize=9.5,
        color=DARK,
        fontweight="bold",
    )
    ax.text(
        5.0,
        0.30,
        "Future hybrid: cloud (1M token context) + local (specialist agents) for power and security.",
        ha="center",
        fontsize=9,
        color=DARK,
    )

    fig.savefig(OUTPUT, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

"""Chart 11: AI baseline comparison chart (full page).

Replaces tab:disc-baseline-vs-sim in Discussion 4.2.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

OUTPUT = Path(__file__).resolve().parent.parent / "images" / "11_ai_baseline_comparison.png"

NAVY = "#1F4E79"
SUPER = "#7C7C7C"
FOUND = "#4A6A8A"
HUANG = "#B45424"
SIM = "#2C7A4D"
DARK = "#1A1A1A"

COLUMNS = [
    ("Supervised Models\n(Manz, SHIELD-RT, SCORPIO)", SUPER),
    ("Multimodal Foundation\n(PROGPATH, AIM-LCpro)", FOUND),
    ("Huang 2025 Cox vs ML", HUANG),
    ("Four Simulations Here", SIM),
]

ROWS = [
    (
        "Inputs",
        [
            "Fixed feature set per task",
            "Multimodal images plus tabular",
            "Tabular features only",
            "Repository plus narrative plus code",
        ],
    ),
    (
        "Output",
        [
            "One probability per patient",
            "C-index per cohort",
            "Hazard ratio",
            "Per-hour narrative plus diagrams plus JSON",
        ],
    ),
    ("Cadence", ["Per visit (months)", "Per cohort", "Retrospective", "Per hour or per stage"]),
    ("TRIPOD+AI compliant", ["Some", "Some", "Yes (meta-analytic)", "No (synthetic)"]),
    ("Modalities", ["1", "2 to 4", "1 (tabular)", "Text plus code plus ASCII plus JSON"]),
    ("Real-time stream", ["No", "No", "No", "Yes (hourly commit)"]),
]


def _cell(ax, x, y, w, h, text, color, light=True):
    fill = "white" if light else color
    edge = color
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.04",
            linewidth=1.0,
            edgecolor=edge,
            facecolor=fill,
        )
    )
    text_color = DARK if light else "white"
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.5, color=text_color, wrap=True)


def main() -> None:
    fig, ax = plt.subplots(figsize=(11, 8.5), constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 8.5)
    ax.axis("off")

    ax.text(
        5.5,
        8.10,
        "Computational Signature Comparison: Four LLM Simulations vs Supervised, Multimodal, and Huang 2025",
        fontsize=13.5,
        fontweight="bold",
        ha="center",
        color=NAVY,
    )

    col_x = [0.30, 2.85, 5.40, 7.95]
    col_w = 2.40
    row_h = 1.00
    header_y = 7.20

    for (title, color), x in zip(COLUMNS, col_x):
        ax.add_patch(mpatches.Rectangle((x, header_y), col_w, 0.55, color=color))
        ax.text(
            x + col_w / 2,
            header_y + 0.27,
            title,
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
            fontsize=9.5,
        )

    for i, (prop, cells) in enumerate(ROWS):
        y = 6.40 - i * row_h
        ax.text(0.10, y + row_h / 2 - 0.10, prop, fontsize=10, fontweight="bold", color=NAVY, ha="left", va="center")
        for j, (cell_text, x) in enumerate(zip(cells, col_x)):
            _cell(ax, x, y, col_w, row_h - 0.12, cell_text, COLUMNS[j][1])

    ax.add_patch(
        mpatches.FancyBboxPatch(
            (0.30, 0.10),
            10.40,
            0.50,
            boxstyle="round,pad=0.02,rounding_size=0.04",
            linewidth=1.2,
            edgecolor=NAVY,
            facecolor="#FFF8E1",
        )
    )
    ax.text(
        5.5,
        0.35,
        "Computational signature only - not benchmarked AUROC. Repository scale plus narrative plus code is the qualitative shift.",
        ha="center",
        va="center",
        fontsize=9.5,
        color=DARK,
        fontweight="bold",
    )

    fig.savefig(OUTPUT, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

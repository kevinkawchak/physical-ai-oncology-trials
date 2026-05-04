"""Chart 04: Simulation 2 patient journey timeline (1,120 days).

Replaces the PAT-2026-0042 1,120 day journey ASCII block in Results 3.2.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

OUTPUT = Path(__file__).resolve().parent.parent / "images" / "04_sim2_journey_timeline.png"

PREP = "#2C7A7A"
TREATMENT = "#2C7A4D"
SURVEILLANCE = "#B45424"
CLOSEOUT = "#1F4E79"
COST_LINE = "#7A1F1F"
BASELINE_LINE = "#9CA3AF"
DARK = "#1A1A1A"

STAGES = [
    (1, "Prescreening", -45, -45, PREP, "n/a"),
    (2, "Enrollment", -30, -30, PREP, "n/a"),
    (3, "Digital twin init", -15, -15, PREP, "n/a"),
    (4, "Robot qualification", -7, -1, PREP, "USL 86.0"),
    (5, "Surgery", 0, 0, TREATMENT, "USL 87.5"),
    (6, "Post-op recovery", 1, 30, TREATMENT, "USL 87.0"),
    (7, "Immunotherapy (35 cycles)", 31, 720, TREATMENT, "USL 88.75"),
    (8, "Federated learning", 90, 720, TREATMENT, "n/a"),
    (9, "Surveillance (36 mo)", 1, 1095, SURVEILLANCE, "USL 88.5"),
    (10, "Closeout", 1095, 1095, CLOSEOUT, "n/a"),
]

COST_POINTS = [
    (-45, 0.00),
    (-30, 0.05),
    (-15, 0.08),
    (-7, 0.12),
    (0, 0.30),
    (30, 0.42),
    (180, 0.55),
    (360, 0.68),
    (540, 0.78),
    (720, 0.85),
    (900, 0.89),
    (1095, 0.91),
]


def main() -> None:
    fig, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
    fig.patch.set_facecolor("white")

    ax.set_xlim(-80, 1130)
    ax.set_ylim(-1, len(STAGES))
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(DARK)

    for i, (num, name, start, end, color, usl) in enumerate(STAGES):
        y = len(STAGES) - i - 1
        if start == end:
            width = 12
            x = start - 6
        else:
            width = end - start + 1
            x = start
        rect = mpatches.FancyBboxPatch(
            (x, y - 0.32),
            width,
            0.64,
            boxstyle="round,pad=0.01,rounding_size=4",
            facecolor=color,
            edgecolor=color,
            linewidth=0.6,
            alpha=0.9,
        )
        ax.add_patch(rect)
        label = f"S{num} {name}   {usl}"
        ax.text(-90, y, label, fontsize=9, ha="right", va="center", color=DARK)

    ax.axvline(0, color=DARK, linewidth=0.7, linestyle="--", alpha=0.5)
    ax.text(0, len(STAGES) - 0.4, "Day 0\nSurgery", fontsize=8, ha="center", color=DARK)

    ax.set_xticks([-45, 0, 180, 360, 540, 720, 900, 1095])
    ax.set_xticklabels(["-45", "0", "180", "360", "540", "720", "900", "1095"], fontsize=9)
    ax.set_xlabel("Day (relative to surgery at Day 0)", fontsize=10)
    ax.set_yticks([])

    ax2 = ax.twinx()
    ax2.set_ylim(0, 1.4)
    xs = [p[0] for p in COST_POINTS]
    ys = [p[1] for p in COST_POINTS]
    ax2.plot(xs, ys, color=COST_LINE, linewidth=2.0, marker="o", markersize=4, label="Cumulative cost (M$)")
    ax2.axhline(1.30, color=BASELINE_LINE, linestyle=":", linewidth=1.4)
    ax2.text(1080, 1.32, "Baseline $1.30M", fontsize=8, ha="right", color=BASELINE_LINE)
    ax2.text(1080, 0.95, "Final $0.91M", fontsize=8, ha="right", color=COST_LINE, fontweight="bold")
    ax2.set_ylabel("Cumulative cost (M$)", color=COST_LINE, fontsize=10)
    ax2.tick_params(axis="y", labelcolor=COST_LINE, labelsize=8)
    for spine in ("top", "left"):
        ax2.spines[spine].set_visible(False)
    ax2.spines["right"].set_color(COST_LINE)

    ax.set_title(
        "Patient PAT-2026-0042 - 10 Stage Journey Timeline (1,120 days)",
        fontsize=13,
        fontweight="bold",
        color="#1F4E79",
    )

    legend_items = [
        mpatches.Patch(color=PREP, label="Prep stages"),
        mpatches.Patch(color=TREATMENT, label="Active treatment"),
        mpatches.Patch(color=SURVEILLANCE, label="Surveillance"),
        mpatches.Patch(color=CLOSEOUT, label="Closeout"),
    ]
    ax.legend(
        handles=legend_items,
        loc="lower right",
        bbox_to_anchor=(1.0, -0.30),
        ncol=4,
        frameon=False,
        fontsize=8.5,
    )

    fig.text(
        0.5,
        -0.02,
        "Cumulative cost $0.91M against $1.30M baseline. Savings $0.39M (30 percent).",
        ha="center",
        fontsize=9,
        style="italic",
        color=DARK,
    )

    fig.savefig(OUTPUT, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

"""Chart 17: Cost savings waterfall (NEW).

Adds a NEW waterfall chart to Results 3.2 visualizing the $1.30B
baseline trial cost decomposition through five savings buckets to the
$0.91M per-patient run.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

OUTPUT = Path(__file__).resolve().parent.parent / "images" / "17_cost_savings_waterfall.png"

NAVY = "#1F4E79"
GREEN_LIGHT = "#A4C9A8"
GREEN_MID = "#5FA471"
GREEN_DARK = "#2C7A4D"
GOLD = "#B45424"
DARK = "#1A1A1A"

LABELS = [
    "Baseline\n(industry)",
    "Continuous\ntrial savings",
    "Real-time\nsignal sharing",
    "1M token\ncontext",
    "Autonomous\nagents",
    "Per-patient\namortization",
    "Final per-\npatient run",
]
VALUES = [1300.0, -130.0, -130.0, -130.0, -390.0, -519.09, 0.91]
COLORS = [NAVY, GREEN_LIGHT, GREEN_LIGHT, GREEN_MID, GREEN_MID, GREEN_DARK, GOLD]


def main() -> None:
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    fig.patch.set_facecolor("white")

    cumulative = [VALUES[0]]
    for v in VALUES[1:-1]:
        cumulative.append(cumulative[-1] + v)
    cumulative.append(VALUES[-1])

    x = np.arange(len(LABELS))
    bottoms = [0]
    for i in range(1, len(VALUES) - 1):
        bottoms.append(cumulative[i])
    bottoms.append(0)
    heights = [VALUES[0]]
    for i in range(1, len(VALUES) - 1):
        heights.append(abs(VALUES[i]))
    heights.append(VALUES[-1])

    for i, (xi, h, b) in enumerate(zip(x, heights, bottoms)):
        if i == 0 or i == len(VALUES) - 1:
            ax.bar(xi, h, bottom=b, color=COLORS[i], edgecolor=DARK, linewidth=0.6, width=0.6)
            label_y = b + h / 2
            ax.text(
                xi,
                label_y,
                f"${VALUES[i]:,.2f}M" if i == len(VALUES) - 1 else f"${VALUES[i]:,.0f}M",
                ha="center",
                va="center",
                color="white",
                fontweight="bold",
                fontsize=9,
            )
        else:
            ax.bar(xi, h, bottom=b, color=COLORS[i], edgecolor=DARK, linewidth=0.6, width=0.6)
            ax.text(xi, b + h + 30, f"-${abs(VALUES[i]):,.0f}M", ha="center", va="bottom", color=DARK, fontsize=8.5)

    for i in range(len(VALUES) - 1):
        if i == 0:
            x_start = i + 0.30
            y_start = cumulative[i]
            x_end = i + 1 - 0.30
            y_end = cumulative[i]
        else:
            x_start = i + 0.30
            y_start = cumulative[i]
            x_end = i + 1 - 0.30
            y_end = cumulative[i]
        ax.plot([x_start, x_end], [y_start, y_end], color="#7C7C7C", linewidth=0.8, linestyle="--")

    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, fontsize=9)
    ax.set_ylabel("Million USD", fontsize=10)
    ax.set_ylim(0, 1450)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(DARK)
    ax.spines["bottom"].set_color(DARK)

    ax.set_title(
        "Simulation 2 Cost Savings Waterfall: $1.30B Baseline to $0.91M Per-Patient Run",
        fontsize=13,
        fontweight="bold",
        color=NAVY,
    )

    ax.add_patch(
        mpatches.FancyBboxPatch(
            (4.6, 1100),
            2.2,
            230,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            linewidth=1.0,
            edgecolor=GOLD,
            facecolor="#FFF8E1",
        )
    )
    ax.text(
        5.7,
        1265,
        "FDA cites 30 to 50 percent",
        ha="center",
        fontsize=9,
        fontweight="bold",
        color=GOLD,
    )
    ax.text(
        5.7,
        1175,
        "trial cost reductions\n($390M to $650M band)",
        ha="center",
        fontsize=8.5,
        color=DARK,
    )

    fig.savefig(OUTPUT, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

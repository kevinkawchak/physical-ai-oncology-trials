"""Chart 29: Artifact counts treemap (NEW)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

OUTPUT = Path(__file__).resolve().parent.parent / "images" / "29_artifact_treemap.png"

NAVY = "#1F4E79"
TEAL = "#2C7A7A"
GOLD = "#B45424"
PURPLE = "#6A4C8C"
DARK = "#1A1A1A"

DATA = [
    (
        "Sim 1 - 392 artifacts (RTCT 56h)",
        NAVY,
        [("ASCII diagrams", 168), ("Markdown files", 224)],
    ),
    (
        "Sim 2 - 54 artifacts (single patient)",
        TEAL,
        [("Python modules", 12), ("ASCII progress", 30), ("Deliverables", 6), ("Reg. tables", 6)],
    ),
    (
        "Sim 3 - 176 artifacts (24h sponsor)",
        GOLD,
        [("Python", 24), ("JSON", 24), ("ASCII", 75), ("Agent files", 53)],
    ),
    (
        "Sim 4 - 868 artifacts (168h sponsor)",
        PURPLE,
        [("Python", 168), ("JSON", 168), ("ASCII", 525), ("Daily summ.", 7)],
    ),
]


def _shade(hex_color, factor):
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    r = int(r + (255 - r) * factor)
    g = int(g + (255 - g) * factor)
    b = int(b + (255 - b) * factor)
    return f"#{r:02X}{g:02X}{b:02X}"


def main() -> None:
    fig, ax = plt.subplots(figsize=(11, 6.5), constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 6.5)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.text(
        5.5,
        6.20,
        "Cumulative Artifact Counts Treemap (Total approximately 1,490 artifacts)",
        fontsize=13.5,
        fontweight="bold",
        ha="center",
        color=NAVY,
    )

    totals = [392, 54, 176, 868]
    grand = sum(totals)
    widths = [t / grand * 11 for t in totals]
    x_cursor = 0
    for (title, color, sub), w in zip(DATA, widths):
        ax.add_patch(mpatches.Rectangle((x_cursor, 0.30), w, 5.45, facecolor=color, edgecolor="white", linewidth=2.5))
        ax.text(x_cursor + w / 2, 5.50, title, color="white", fontsize=10.5, fontweight="bold", ha="center")
        sub_total = sum(s[1] for s in sub)
        sub_y_top = 5.20
        for j, (sname, scount) in enumerate(sub):
            sh = (scount / sub_total) * 4.30
            color_sub = _shade(color, 0.05 + j * 0.10)
            ax.add_patch(
                mpatches.Rectangle(
                    (x_cursor + 0.05, sub_y_top - sh),
                    w - 0.10,
                    sh,
                    facecolor=color_sub,
                    edgecolor="white",
                    linewidth=1.0,
                )
            )
            text_color = "white" if scount / sub_total > 0.30 else DARK
            ax.text(
                x_cursor + w / 2,
                sub_y_top - sh / 2 + 0.05,
                sname,
                color=text_color,
                fontsize=9,
                fontweight="bold",
                ha="center",
                va="center",
            )
            ax.text(
                x_cursor + w / 2,
                sub_y_top - sh / 2 - 0.18,
                f"{scount}",
                color=text_color,
                fontsize=10,
                ha="center",
                va="center",
            )
            sub_y_top -= sh
        x_cursor += w

    ax.text(
        5.5,
        0.10,
        "Site simulations (Sims 1, 2) plus sponsor simulations (Sims 3, 4) yield 1,490 cumulative artifacts.",
        ha="center",
        fontsize=9,
        style="italic",
        color=DARK,
    )

    fig.savefig(OUTPUT, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

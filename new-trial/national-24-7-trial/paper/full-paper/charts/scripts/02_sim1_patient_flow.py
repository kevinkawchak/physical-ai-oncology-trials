"""Chart 02: Simulation 1 hour 12 patient flow Gantt.

Replaces the hour-12 patient flow ASCII block in Results 3.1 with a
minute-resolution Gantt chart for the eight named patients.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

OUTPUT = Path(__file__).resolve().parent.parent / "images" / "02_sim1_patient_flow.png"

SITE_COLORS = {
    "SITE-A": "#1F4E79",
    "SITE-B": "#2C7A4D",
    "SITE-C": "#B45424",
    "SITE-D": "#6A4C8C",
}
RECOVERY = "#B6CFE6"
COOL = "#D8E5F0"
DARK_TEXT = "#1A1A1A"
GRID = "#E6E6E6"

PATIENTS = [
    ("PAT-CONT-0058A", "SITE-A", "HCC img", [(50, 60, "IMG")]),
    ("PAT-CONT-0057A", "SITE-D", "Pediatric visit", [(42, 55, "CMP")]),
    ("PAT-CONT-0056A", "SITE-C", "SCLC RT2", [(35, 58, "RT")]),
    ("PAT-CONT-0055A", "SITE-B", "Sarcoma BX", [(20, 33, "BX"), (34, 59, "REC")]),
    ("PAT-CONT-0054A", "SITE-A", "NSCLC RT3", [(8, 34, "RT")]),
    ("PAT-CONT-0053A", "SITE-C", "Mantle BX", [(0, 18, "BX"), (19, 59, "REC")]),
    ("PAT-CONT-0052A", "SITE-A", "GBM RT10 wrap", [(0, 14, "RT")]),
    ("PAT-CONT-0051A", "SITE-B", "HCC ablation", [(0, 25, "ABL"), (26, 35, "COOL"), (36, 59, "REC")]),
]


def main() -> None:
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax.set_xlim(-2, 60)
    ax.set_ylim(-0.5, len(PATIENTS) - 0.5 + 0.4)
    ax.set_facecolor("white")

    for x in range(0, 61, 10):
        ax.axvline(x, color=GRID, linewidth=0.7, zorder=0)

    for idx, (pid, site, label, blocks) in enumerate(PATIENTS):
        site_color = SITE_COLORS[site]
        ax.barh(idx, 60, left=0, height=0.85, color="#FAFAFA", edgecolor="none", zorder=1)
        ax.add_patch(mpatches.Rectangle((-1.7, idx - 0.42), 0.18, 0.84, color=site_color, zorder=2))
        for start, end, kind in blocks:
            if kind == "REC":
                color = RECOVERY
            elif kind == "COOL":
                color = COOL
            else:
                color = site_color
            width = end - start + 1
            rect = mpatches.Rectangle(
                (start, idx - 0.32),
                width,
                0.64,
                facecolor=color,
                edgecolor=site_color,
                linewidth=0.8,
                zorder=3,
            )
            ax.add_patch(rect)
            ax.text(
                start + width / 2,
                idx,
                kind,
                fontsize=8,
                ha="center",
                va="center",
                color="white" if kind not in ("REC", "COOL") else DARK_TEXT,
                fontweight="bold",
                zorder=4,
            )
        ax.text(
            61,
            idx,
            f"{pid}   {site}   {label}",
            fontsize=8.5,
            va="center",
            color=DARK_TEXT,
            zorder=5,
        )

    ax.set_yticks(range(len(PATIENTS)))
    ax.set_yticklabels([])
    ax.set_xticks(range(0, 61, 10))
    ax.set_xticklabels([f"{m:02d}" for m in range(0, 61, 10)], fontsize=9)
    ax.set_xlabel("Minute (12:00 to 12:59 UTC)", fontsize=10)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(DARK_TEXT)

    ax.set_title(
        "Hour 12 Patient Flow - Continuous National 24/7 RTCT (12:00 to 12:59 UTC)",
        fontsize=13,
        fontweight="bold",
        color="#1F4E79",
    )

    legend_handles = [mpatches.Patch(color=c, label=s) for s, c in SITE_COLORS.items()]
    legend_handles.append(mpatches.Patch(color=RECOVERY, label="Recovery"))
    legend_handles.append(mpatches.Patch(color=COOL, label="Cooling"))
    ax.legend(
        handles=legend_handles,
        loc="lower right",
        bbox_to_anchor=(1.02, -0.32),
        ncol=6,
        frameon=False,
        fontsize=8.5,
    )

    ax.text(
        30,
        len(PATIENTS) - 0.10,
        "Network: 5 arrivals, 4 departures (0002, 0007, 0018A, 0030A), AE-002 resolved 12:30",
        fontsize=8.5,
        ha="center",
        style="italic",
        color=DARK_TEXT,
    )

    fig.savefig(OUTPUT, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

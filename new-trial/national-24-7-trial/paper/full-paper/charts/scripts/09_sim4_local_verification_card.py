"""Chart 09: Simulation 4 local verification card.

Replaces the local verification ASCII block in Results 3.4 with a
three-pane verification card.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

OUTPUT = Path(__file__).resolve().parent.parent / "images" / "09_sim4_local_verification_card.png"

NAVY = "#1F4E79"
GRAY_FILL = "#F0F0F0"
GREEN = "#2C7A4D"
GOLD = "#B45424"
DARK = "#1A1A1A"


def _panel(ax, x, y, w, h, title, lines, accent=NAVY):
    box = mpatches.FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.05", linewidth=1.4, edgecolor=accent, facecolor="white"
    )
    ax.add_patch(box)
    ax.add_patch(mpatches.Rectangle((x, y + h - 0.55), w, 0.55, color=accent))
    ax.text(x + 0.18, y + h - 0.28, title, fontsize=11, fontweight="bold", color="white", va="center")
    for i, line in enumerate(lines):
        ax.text(x + 0.18, y + h - 0.85 - 0.30 * i, line, fontsize=9, color=DARK, va="center")


def main() -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.5), constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 9.5)
    ax.set_ylim(0, 5.5)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.text(
        4.75,
        5.20,
        "Simulation 4 Local Verification - Core i5-6200U / 4 GB RAM / Windows 10 Pro",
        fontsize=13,
        fontweight="bold",
        ha="center",
        color=NAVY,
    )

    hardware = [
        "CPU: Intel Core i5-6200U at 2.30 GHz (2 cores, 4 threads, 2015)",
        "RAM: 4 GB DDR4   Storage: 256 GB SSD",
        "OS: Windows 10 Pro 22H2",
        "Python: 3.10.12 CPython, no extensions",
    ]
    mitigations = [
        "Thermal throttling watchdog (poll every 30 s, pause at 85 C)",
        "Windows Update set to manual; deferred during run",
        "Antivirus exclusion: 168_hours/ directory",
        "Pagefile sized to 4096 MB (matches RAM); placed on SSD",
    ]
    results = [
        "168/168 hourly scripts started",
        "167/168 hourly scripts completed end-to-end",
        "Hour 134 paused for thermal throttling, resumed cleanly",
        "sponsor_168h_summary.json mirrors cloud within 1 percent tolerance",
        "PSL trajectory identical to cloud output",
    ]
    _panel(ax, 0.2, 3.20, 9.10, 1.65, "Hardware", hardware, NAVY)
    _panel(ax, 0.2, 1.50, 9.10, 1.55, "Mitigations Applied During the 168h Run", mitigations, GOLD)
    _panel(ax, 0.2, 0.05, 9.10, 1.30, "Results", results, GREEN)

    ax.text(
        4.75,
        4.92,
        "Partial pass: 167/168 scripts completed; one paused for thermal throttling at hour 134, resumed cleanly.",
        fontsize=9,
        ha="center",
        color=GOLD,
        style="italic",
    )

    fig.savefig(OUTPUT, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

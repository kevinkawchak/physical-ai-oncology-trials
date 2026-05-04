"""Chart 16: FDA RTCT capability radar (full page, NEW).

Adds a NEW capability radar chart to Discussion 4.1 visualizing the
seven-axis comparison of FDA RTCT 28 April 2026 proof-of-concept versus
the four author simulations.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUTPUT = Path(__file__).resolve().parent.parent / "images" / "16_fda_rtct_capability_radar.png"

NAVY = "#1F4E79"
FDA_COLOR = "#4A7BAA"
SIM_COLOR = "#2C7A4D"
DARK = "#1A1A1A"

AXES = [
    "Trial count",
    "Site count",
    "Robotic integration",
    "Real-time signal types",
    "Cadence depth",
    "Predictive context size",
    "Local-side compute",
]
FDA_VALUES = [2, 2, 0, 2, 3, 1, 0]
SIM_VALUES = [4, 4, 5, 4, 5, 5, 5]


def main() -> None:
    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True), constrained_layout=True)
    fig.patch.set_facecolor("white")

    n = len(AXES)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]
    fda = FDA_VALUES + FDA_VALUES[:1]
    sim = SIM_VALUES + SIM_VALUES[:1]

    ax.set_facecolor("#FAFBFD")
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.plot(angles, fda, color=FDA_COLOR, linewidth=2.0, label="FDA RTCT 2026 PoC")
    ax.fill(angles, fda, color=FDA_COLOR, alpha=0.30)

    ax.plot(angles, sim, color=SIM_COLOR, linewidth=2.4, label="Four LLM Simulations")
    ax.fill(angles, sim, color=SIM_COLOR, alpha=0.40)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(AXES, fontsize=10, color=DARK)
    ax.set_ylim(0, 5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(["1", "2", "3", "4", "5"], fontsize=8, color="#7C7C7C")
    ax.tick_params(axis="x", pad=18)

    ax.set_title(
        "FDA RTCT 28 April 2026 vs Four LLM Simulations - Capability Radar",
        fontsize=13,
        fontweight="bold",
        color=NAVY,
        pad=24,
    )

    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.16), frameon=False, fontsize=10, ncol=2)

    fig.text(
        0.5,
        0.04,
        "Three biggest deltas: Robotic integration (FDA 0 vs Sim 5), "
        "Predictive context size (FDA 1 vs Sim 5), Local-side compute (FDA 0 vs Sim 5)",
        ha="center",
        fontsize=9,
        style="italic",
        color=DARK,
    )

    fig.savefig(OUTPUT, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

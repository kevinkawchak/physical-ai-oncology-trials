"""Chart 05: Simulation 2 stage table with USL trajectory.

Replaces tab:sim2-stages in Results 3.2 with a hybrid table-plus-trajectory
figure.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

OUTPUT = Path(__file__).resolve().parent.parent / "images" / "05_sim2_stage_usl_table.png"

NAVY = "#1F4E79"
GREEN = "#2C7A4D"
LIGHT_ROW = "#F5F5F5"
DARK = "#1A1A1A"

ROWS = [
    (1, "Prescreening", "EHR scan, eligibility check", "n/a", None),
    (2, "Enrollment", "Consent, regulatory snapshot", "n/a", None),
    (3, "Digital twin init", "Patient digital twin instantiation", "n/a", None),
    (4, "Robot qualification", "da Vinci Xi qualification at site", "da Vinci Xi", 86.0),
    (5, "Robotic surgery", "Lobectomy, R0, neg margins, 168 min", "da Vinci Xi", 87.5),
    (6, "Post-op recovery", "Cardio, respiratory, pain monitor", "monitor cobot", 87.0),
    (7, "Immunotherapy", "35 cycles pembrolizumab over 24 mo", "Franka Emika", 88.75),
    (8, "Federated learning", "Cross-site model update", "n/a", None),
    (9, "Surveillance", "36 mo survival monitoring + DT", "n/a", 88.5),
    (10, "Closeout", "Final report, archive, audit", "n/a", None),
]


def main() -> None:
    fig = plt.figure(figsize=(10, 5), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[2.2, 1])
    fig.patch.set_facecolor("white")

    ax_table = fig.add_subplot(gs[0, 0])
    ax_table.axis("off")
    ax_table.set_xlim(0, 10)
    ax_table.set_ylim(0, len(ROWS) + 2)

    ax_table.add_patch(plt.Rectangle((0, len(ROWS) + 0.7), 10, 0.7, color=NAVY))
    headers = ["St", "Stage Name", "Description", "Robot", "USL"]
    xs = [0.2, 1.0, 4.0, 7.4, 9.1]
    for x, h in zip(xs, headers):
        ax_table.text(x, len(ROWS) + 1.05, h, color="white", fontweight="bold", fontsize=10, va="center")

    for i, (num, name, desc, robot, usl) in enumerate(ROWS):
        y = len(ROWS) - i - 0.5
        if i % 2 == 0:
            ax_table.add_patch(plt.Rectangle((0, y - 0.05), 10, 0.95, color=LIGHT_ROW, zorder=0))
        ax_table.text(0.2, y + 0.4, str(num), color=DARK, fontsize=9, va="center")
        ax_table.text(1.0, y + 0.4, name, color=DARK, fontsize=9, va="center")
        ax_table.text(4.0, y + 0.4, desc, color=DARK, fontsize=8.5, va="center")
        ax_table.text(7.4, y + 0.4, robot, color=DARK, fontsize=8.5, va="center")
        usl_str = "n/a" if usl is None else f"{usl:.2f}"
        ax_table.text(9.1, y + 0.4, usl_str, color=DARK, fontsize=9, va="center")

    ax_table.text(
        5,
        len(ROWS) + 1.85,
        "Simulation 2 Ten-Stage Journey for Patient PAT-2026-0042",
        ha="center",
        fontsize=12,
        fontweight="bold",
        color=NAVY,
    )

    ax_chart = fig.add_subplot(gs[0, 1])
    ax_chart.set_facecolor("white")
    pts = [(num, usl) for (num, _, _, _, usl) in ROWS if usl is not None]
    xs2 = [p[0] for p in pts]
    ys2 = [p[1] for p in pts]
    ax_chart.plot(xs2, ys2, color=GREEN, linewidth=2.2, marker="o", markersize=8)
    for x, y in pts:
        ax_chart.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 8), fontsize=8, ha="center")
    ax_chart.axhline(86.0, color="#9CA3AF", linestyle=":", linewidth=1.0)
    ax_chart.text(4.2, 86.05, "Baseline 86.0", fontsize=7.5, color="#7C7C7C")
    ax_chart.set_xlim(3.5, 9.5)
    ax_chart.set_ylim(85, 90)
    ax_chart.set_xticks([4, 5, 6, 7, 9])
    ax_chart.set_xticklabels(["S4", "S5", "S6", "S7", "S9"], fontsize=9)
    ax_chart.set_ylabel("USL Score", fontsize=9.5, color=DARK)
    ax_chart.set_title("USL Trajectory", fontsize=10, fontweight="bold", color=NAVY)
    for spine in ("top", "right"):
        ax_chart.spines[spine].set_visible(False)
    ax_chart.tick_params(axis="both", labelsize=8)

    fig.text(
        0.5,
        -0.04,
        "USL = Usability Safety Level (0 to 100). Stages 1, 2, 3, 8, and 10 have no robot or USL value.",
        ha="center",
        fontsize=8.5,
        style="italic",
        color=DARK,
    )

    fig.savefig(OUTPUT, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

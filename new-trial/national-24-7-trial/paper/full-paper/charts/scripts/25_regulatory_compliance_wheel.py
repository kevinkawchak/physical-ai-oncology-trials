"""Chart 25: 21 CFR / ICH regulatory compliance wheel (NEW)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

OUTPUT = Path(__file__).resolve().parent.parent / "images" / "25_regulatory_compliance_wheel.png"

NAVY = "#1F4E79"
GREEN_DARK = "#2C7A4D"
GREEN_LIGHT = "#1F4E2C"
BLUE_DARK = "#1F4E79"
BLUE_LIGHT = "#4A7BAA"
DARK = "#1A1A1A"

OUTER = [
    (1, "S1 Prescreen", "21 CFR 312 Subpart B"),
    (2, "S2 Enrollment", "21 CFR 50 Consent"),
    (3, "S3 Digital twin", "21 CFR Part 11"),
    (4, "S4 Robot qual", "ICH E6(R3) GCP"),
    (5, "S5 Surgery", "21 CFR 821"),
    (6, "S6 Recovery", "21 CFR 312 Subpart D"),
    (7, "S7 Immunotherapy", "ICH E2A"),
    (8, "S8 Federation", "21 CFR 11"),
    (9, "S9 Surveillance", "21 CFR 312.32"),
    (10, "S10 Closeout", "ICH E3"),
]

INNER = [
    ("Governance", "21 CFR 312 sponsor"),
    ("Study Exec", "ICH E6(R3) GCP"),
    ("Site/Robotics", "21 CFR 50 + 821"),
    ("Trust", "21 CFR 11 + RTCT"),
    ("Audit", "21 CFR 312.55"),
    ("FDA Liaison", "RTCT Pilot 2026"),
]


def main() -> None:
    fig, ax = plt.subplots(figsize=(9.5, 9.5), constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)

    n_outer = len(OUTER)
    angle_per = 360 / n_outer
    for i, (num, label, cite) in enumerate(OUTER):
        theta1 = 90 - (i + 1) * angle_per
        theta2 = 90 - i * angle_per
        color = GREEN_DARK if i % 2 == 0 else GREEN_LIGHT
        wedge = mpatches.Wedge(
            (0, 0), 1.3, theta1, theta2, width=0.32, facecolor=color, edgecolor="white", linewidth=1.6
        )
        ax.add_patch(wedge)
        mid_angle = np.deg2rad((theta1 + theta2) / 2)
        r1 = 1.21
        ax.text(
            r1 * np.cos(mid_angle),
            r1 * np.sin(mid_angle),
            label,
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
            color="white",
        )
        r2 = 1.05
        ax.text(
            r2 * np.cos(mid_angle), r2 * np.sin(mid_angle), cite, ha="center", va="center", fontsize=6.5, color="white"
        )

    n_inner = len(INNER)
    angle_per_in = 360 / n_inner
    for i, (label, cite) in enumerate(INNER):
        theta1 = 90 - (i + 1) * angle_per_in
        theta2 = 90 - i * angle_per_in
        color = BLUE_DARK if i % 2 == 0 else BLUE_LIGHT
        wedge = mpatches.Wedge(
            (0, 0), 0.92, theta1, theta2, width=0.42, facecolor=color, edgecolor="white", linewidth=1.4
        )
        ax.add_patch(wedge)
        mid_angle = np.deg2rad((theta1 + theta2) / 2)
        r1 = 0.78
        ax.text(
            r1 * np.cos(mid_angle),
            r1 * np.sin(mid_angle),
            label,
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="white",
        )
        r2 = 0.62
        ax.text(
            r2 * np.cos(mid_angle), r2 * np.sin(mid_angle), cite, ha="center", va="center", fontsize=7, color="white"
        )

    hub = mpatches.Circle((0, 0), 0.46, facecolor="white", edgecolor=NAVY, linewidth=2.0)
    ax.add_patch(hub)
    ax.text(0, 0.16, "21 CFR + ICH", ha="center", va="center", fontsize=11, fontweight="bold", color=NAVY)
    ax.text(0, -0.02, "+ FDA RTCT", ha="center", va="center", fontsize=10, color=NAVY)
    ax.text(0, -0.20, "Compliance", ha="center", va="center", fontsize=9, color=DARK)

    ax.text(
        0,
        1.45,
        "Regulatory Compliance Wheel - Sim 2 Stages (outer) and Sim 3 Sponsor Layers (inner)",
        ha="center",
        fontsize=12.5,
        fontweight="bold",
        color=NAVY,
    )

    fig.savefig(OUTPUT, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

"""Chart 18: Patient safety pipeline funnel (full page, NEW).

Adds a NEW funnel chart to Discussion 4.3 visualizing the patient safety
pipeline from prescreening through 36 month surveillance and closeout.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

OUTPUT = Path(__file__).resolve().parent.parent / "images" / "18_patient_safety_funnel.png"

NAVY = "#1F4E79"
DARK = "#1A1A1A"

GREENS = [
    "#E5F0E8",
    "#CDE5D0",
    "#B0D8B6",
    "#94CC9D",
    "#79BF85",
    "#5FA471",
    "#477F58",
    "#356046",
    "#244A36",
    "#1F3F2C",
]

STAGES = [
    (1, "Prescreening", "EHR scan + eligibility filter", 1000),
    (2, "Enrollment", "Consent + regulatory snapshot", 720),
    (3, "Digital twin init", "Twin verification filter", 720),
    (4, "Robot qualification", "Qualification filter at site", 700),
    (5, "Surgery (Day 0)", "Intra-op safety filter", 690),
    (6, "Post-op recovery", "Recovery monitoring filter", 680),
    (7, "Immunotherapy 35 cycles", "Immunotherapy AE filter", 650),
    (8, "Federated learning", "Federation privacy filter", 645),
    (9, "Surveillance 36 mo", "Surveillance filter", 630),
    (10, "Closeout", "Closeout filter", 615),
]


def main() -> None:
    fig, ax = plt.subplots(figsize=(8.5, 11), constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 11)
    ax.axis("off")

    ax.text(
        5.0,
        10.55,
        "Patient Safety Pipeline Funnel - 10 Stages with Per-Stage Filter",
        fontsize=14,
        fontweight="bold",
        ha="center",
        color=NAVY,
    )
    ax.text(
        5.0,
        10.20,
        "Hypothetical 1,000-patient cohort flowing through Simulation 2 stages",
        fontsize=10,
        ha="center",
        color=DARK,
        style="italic",
    )

    max_w = 7.0
    bottom_w = 1.6
    layer_h = 0.85
    top_y = 9.65
    for i, (num, name, filt, cohort) in enumerate(STAGES):
        y_top = top_y - i * layer_h
        y_bot = y_top - layer_h
        frac_top = 1.0 if i == 0 else cohort / 1000
        frac_bot = STAGES[i + 1][3] / 1000 if i + 1 < len(STAGES) else cohort / 1000 * 0.85
        w_top = bottom_w + (max_w - bottom_w) * frac_top
        w_bot = bottom_w + (max_w - bottom_w) * frac_bot
        x_top_left = 5.0 - w_top / 2
        x_top_right = 5.0 + w_top / 2
        x_bot_left = 5.0 - w_bot / 2
        x_bot_right = 5.0 + w_bot / 2
        polygon = mpatches.Polygon(
            [
                (x_top_left, y_top),
                (x_top_right, y_top),
                (x_bot_right, y_bot),
                (x_bot_left, y_bot),
            ],
            closed=True,
            facecolor=GREENS[i],
            edgecolor=NAVY,
            linewidth=0.9,
        )
        ax.add_patch(polygon)
        text_color = "white" if i >= 6 else DARK
        ax.text(
            5.0,
            (y_top + y_bot) / 2 + 0.08,
            f"S{num}  {name}",
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color=text_color,
        )
        ax.text(
            5.0,
            (y_top + y_bot) / 2 - 0.16,
            f"{cohort} patients   |   {filt}",
            ha="center",
            va="center",
            fontsize=8.5,
            color=text_color,
        )
        ax.text(x_top_right + 0.10, (y_top + y_bot) / 2, f"{cohort}", fontsize=9, color=DARK, va="center")

    ax.add_patch(
        mpatches.FancyBboxPatch(
            (0.40, 0.40),
            9.20,
            0.85,
            boxstyle="round,pad=0.02,rounding_size=0.04",
            linewidth=1.2,
            edgecolor=NAVY,
            facecolor="#FFF8E1",
        )
    )
    ax.text(
        5.0,
        1.00,
        "1M token context permits per-stage safety filtering at minute resolution.",
        ha="center",
        fontsize=10,
        fontweight="bold",
        color=NAVY,
    )
    ax.text(
        5.0,
        0.65,
        "Supervised models (Manz, SHIELD-RT) filter at per-visit resolution only.",
        ha="center",
        fontsize=9.5,
        color=DARK,
    )

    fig.savefig(OUTPUT, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

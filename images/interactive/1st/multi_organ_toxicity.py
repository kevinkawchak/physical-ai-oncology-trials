"""Multi-Organ Toxicity Accumulation Over 6 Chemotherapy Cycles.

Shows how the digital twin predicts cumulative organ toxicity across
6 FOLFOX cycles with CTCAE grading thresholds and dose modifications.

Data source:
  - digital-twins/examples-twins/02_multi_organ_toxicity_twin.py
    PBPK compartment model, organ-specific toxicity models, CTCAE thresholds
"""

import os

import plotly.graph_objects as go
from plotly.subplots import make_subplots


CYCLES = [1, 2, 3, 4, 5, 6]

# Simulated values from the multi-organ toxicity twin
CARDIAC_LVEF = [0.58, 0.56, 0.54, 0.52, 0.50, 0.48]
RENAL_GFR = [90, 85, 80, 76, 72, 69]
HEPATIC_BILI = [0.85, 1.0, 1.15, 1.3, 1.4, 1.5]
NEURO_TNSC = [2, 5, 8, 11, 15, 13]  # C6 lower after 25% dose reduction at C5
HEMATO_ANC = [2.1, 1.8, 1.5, 1.2, 1.0, 1.3]  # C6 recovery after dose reduction

PANELS = [
    {
        "title": "Cardiac — LVEF",
        "y_label": "LVEF (fraction)",
        "data": CARDIAC_LVEF,
        "y_range": [0.20, 0.65],
        "thresholds": [
            {"val": 0.40, "label": "CTCAE Grade 3 (<0.40)", "dash": "dash"},
            {"val": 0.20, "label": "CTCAE Grade 4 (<0.20)", "dash": "dot"},
        ],
        "baseline": 0.60,
    },
    {
        "title": "Renal — GFR",
        "y_label": "GFR (mL/min/1.73m²)",
        "data": RENAL_GFR,
        "y_range": [30, 100],
        "thresholds": [
            {"val": 50, "label": "Grade 2 (<50%)", "dash": "dash"},
        ],
        "baseline": 95.0,
    },
    {
        "title": "Hepatic — Bilirubin",
        "y_label": "Bilirubin (mg/dL)",
        "data": HEPATIC_BILI,
        "y_range": [0.5, 3.0],
        "thresholds": [
            {"val": 2.1, "label": "CTCAE Grade 3 (>3× baseline)", "dash": "dash"},
        ],
        "baseline": 0.7,
    },
    {
        "title": "Neurological — TNSc",
        "y_label": "TNSc Score (0–28)",
        "data": NEURO_TNSC,
        "y_range": [0, 20],
        "thresholds": [
            {"val": 8, "label": "Grade 2 (>8)", "dash": "dot"},
            {"val": 14, "label": "Grade 3 (>14)", "dash": "dash"},
        ],
        "baseline": 0,
    },
    {
        "title": "Hematologic — ANC Nadir",
        "y_label": "ANC Nadir (×10⁹/L)",
        "data": HEMATO_ANC,
        "y_range": [0, 5],
        "thresholds": [
            {"val": 1.5, "label": "Grade 2 (<1.5)", "dash": "dot"},
            {"val": 1.0, "label": "Grade 3 (<1.0)", "dash": "dash"},
            {"val": 0.5, "label": "Grade 4 (<0.5)", "dash": "dashdot"},
        ],
        "baseline": 5.2,
    },
]


def create_toxicity_chart(dark_mode=False):
    """Create multi-panel line chart with CTCAE thresholds."""
    if dark_mode:
        template = "plotly_dark"
        line_colors = ["#00e5ff", "#69f0ae", "#ffab40", "#ce93d8", "#ef5350"]
        bg_color = "#1a1a2e"
        title_color = "#e0e0e0"
        thresh_color = "#ef5350"
        dose_line_color = "#ffd740"
    else:
        template = "plotly_white"
        line_colors = ["#1565c0", "#2e7d32", "#e65100", "#6a1b9a", "#c62828"]
        bg_color = "#ffffff"
        title_color = "#1a1a2e"
        thresh_color = "#c62828"
        dose_line_color = "#ff6f00"

    fig = make_subplots(
        rows=5,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=[p["title"] for p in PANELS],
    )

    for i, panel in enumerate(PANELS):
        row = i + 1

        # Main data line
        fig.add_trace(
            go.Scatter(
                x=CYCLES,
                y=panel["data"],
                mode="lines+markers",
                name=panel["title"],
                line=dict(color=line_colors[i], width=3),
                marker=dict(size=8),
                showlegend=False,
            ),
            row=row,
            col=1,
        )

        # CTCAE threshold lines
        for thresh in panel["thresholds"]:
            fig.add_hline(
                y=thresh["val"],
                line_dash=thresh["dash"],
                line_color=thresh_color,
                line_width=1,
                annotation_text=thresh["label"],
                annotation_position="top right",
                annotation_font=dict(size=9, color=thresh_color),
                row=row,
                col=1,
            )

        fig.update_yaxes(
            title_text=panel["y_label"],
            range=panel["y_range"],
            row=row,
            col=1,
            title_font=dict(size=10),
            tickfont=dict(size=9),
        )

    # Dose reduction line between C4 and C5
    for row_idx in range(1, 6):
        fig.add_vline(
            x=4.5,
            line_dash="dash",
            line_color=dose_line_color,
            line_width=2,
            row=row_idx,
            col=1,
        )

    # Dose reduction annotation on top panel
    fig.add_annotation(
        x=4.5,
        y=PANELS[0]["y_range"][1],
        text="Dose Reduction<br>Triggered (25%)",
        showarrow=False,
        font=dict(size=11, color=dose_line_color),
        xanchor="left",
        row=1,
        col=1,
    )

    fig.update_xaxes(
        title_text="Cycle Number (1–6, each 21 days)",
        tickvals=CYCLES,
        row=5,
        col=1,
    )

    fig.update_layout(
        title=dict(
            text=(
                "Multi-Organ Toxicity Twin: Cumulative Toxicity Over 6 FOLFOX Cycles"
                "<br><sup>Source: digital-twins/examples-twins/02_multi_organ_toxicity_twin.py"
                " — Oxaliplatin 85 mg/m², BSA 1.85 m²</sup>"
            ),
            font=dict(size=17, color=title_color),
            x=0.5,
        ),
        template=template,
        paper_bgcolor=bg_color,
        height=1080,
        width=1920,
        margin=dict(t=100, l=80, r=40, b=60),
        showlegend=False,
    )

    return fig


def main():
    """Generate multi-organ toxicity visualizations."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    png_dir = os.path.join(os.path.dirname(os.path.dirname(script_dir)), "png", "1st")
    os.makedirs(png_dir, exist_ok=True)

    for dark_mode in [False, True]:
        suffix = "dark" if dark_mode else "light"
        fig = create_toxicity_chart(dark_mode=dark_mode)

        html_path = os.path.join(script_dir, f"multi_organ_toxicity_{suffix}.html")
        fig.write_html(html_path, include_plotlyjs=True)

        png_path = os.path.join(png_dir, f"multi_organ_toxicity_{suffix}.png")
        fig.write_image(png_path, width=1920, height=1080, scale=2)

    print("multi_organ_toxicity: done")


if __name__ == "__main__":
    main()

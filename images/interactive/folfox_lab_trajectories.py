"""FOLFOX Chemotherapy Cycle: Lab Value Trajectories Over 21 Days.

Shows the expected temporal dynamics of key lab values during a single
FOLFOX cycle as predicted by the digital twin.

Data source:
  - digital-twins/examples-twins/01_realtime_dt_synchronization.py
    FOLFOX_CYCLE simulation parameters (85 mg/m², BSA 1.85 m²)
"""

import os

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Observed data points from the FOLFOX cycle walkthrough
DAYS = [0, 1, 3, 7, 10, 14, 21]
ANC = [4.5, 4.5, 3.8, 1.8, 2.2, 2.9, 4.0]
CREATININE = [0.85, 0.85, 0.90, 0.95, 0.92, 0.90, 0.86]
HEMOGLOBIN = [14.1, 14.1, 13.5, 12.8, 13.0, 13.2, 13.9]
TUMOR_MARKER = [48, 48, 46, 44, 42, 43, 44]

# Dynamics parameters from source
DYNAMICS_NOTE = (
    "ANC recovery 0.15/day · Cr clearance 0.05/day · Hb recovery 0.02/day · Marker sensitivity 3.0 ng/mL per cm³"
)


def interpolate_smooth(days, values, num_points=200):
    """Create smooth interpolation for plotting."""
    x_smooth = np.linspace(days[0], days[-1], num_points)
    y_smooth = np.interp(x_smooth, days, values)
    return x_smooth, y_smooth


def create_folfox_chart(dark_mode=False):
    """Create multi-line chart with markers and nadir shading."""
    if dark_mode:
        template = "plotly_dark"
        anc_color = "#00e5ff"
        cr_color = "#ef9a9a"
        hb_color = "#69f0ae"
        tm_color = "#ce93d8"
        bg_color = "#1a1a2e"
        title_color = "#e0e0e0"
        nadir_color = "rgba(255,255,255,0.08)"
        thresh_color = "#ef5350"
    else:
        template = "plotly_white"
        anc_color = "#1565c0"
        cr_color = "#c62828"
        hb_color = "#2e7d32"
        tm_color = "#6a1b9a"
        bg_color = "#ffffff"
        title_color = "#1a1a2e"
        nadir_color = "rgba(255,0,0,0.06)"
        thresh_color = "#c62828"

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Smooth curves
    d_smooth, anc_smooth = interpolate_smooth(DAYS, ANC)
    _, cr_smooth = interpolate_smooth(DAYS, CREATININE)
    _, hb_smooth = interpolate_smooth(DAYS, HEMOGLOBIN)
    _, tm_smooth = interpolate_smooth(DAYS, TUMOR_MARKER)

    # ANC line + markers
    fig.add_trace(
        go.Scatter(
            x=d_smooth,
            y=anc_smooth,
            mode="lines",
            name="ANC (×10⁹/L)",
            line=dict(color=anc_color, width=3),
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=DAYS,
            y=ANC,
            mode="markers",
            name="ANC observed",
            marker=dict(color=anc_color, size=10, symbol="circle"),
            showlegend=False,
        ),
        secondary_y=False,
    )

    # Creatinine
    fig.add_trace(
        go.Scatter(
            x=d_smooth,
            y=cr_smooth,
            mode="lines",
            name="Creatinine (mg/dL)",
            line=dict(color=cr_color, width=3),
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=DAYS,
            y=CREATININE,
            mode="markers",
            name="Cr observed",
            marker=dict(color=cr_color, size=10, symbol="square"),
            showlegend=False,
        ),
        secondary_y=False,
    )

    # Hemoglobin
    fig.add_trace(
        go.Scatter(
            x=d_smooth,
            y=hb_smooth,
            mode="lines",
            name="Hemoglobin (g/dL)",
            line=dict(color=hb_color, width=3),
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=DAYS,
            y=HEMOGLOBIN,
            mode="markers",
            name="Hb observed",
            marker=dict(color=hb_color, size=10, symbol="diamond"),
            showlegend=False,
        ),
        secondary_y=False,
    )

    # Tumor Marker on secondary y
    fig.add_trace(
        go.Scatter(
            x=d_smooth,
            y=tm_smooth,
            mode="lines",
            name="Tumor Marker (ng/mL)",
            line=dict(color=tm_color, width=3, dash="dot"),
        ),
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(
            x=DAYS,
            y=TUMOR_MARKER,
            mode="markers",
            name="Marker observed",
            marker=dict(color=tm_color, size=10, symbol="triangle-up"),
            showlegend=False,
        ),
        secondary_y=True,
    )

    # Nadir window shading (Day 5-12)
    fig.add_vrect(
        x0=5,
        x1=12,
        fillcolor=nadir_color,
        layer="below",
        line_width=0,
        annotation_text="Nadir Window",
        annotation_position="top left",
        annotation_font=dict(size=12),
    )

    # Grade 4 neutropenia threshold (ANC = 1.0)
    fig.add_hline(
        y=1.0,
        line_dash="dash",
        line_color=thresh_color,
        line_width=1.5,
        annotation_text="Grade 4 Neutropenia (ANC=1.0)",
        annotation_position="bottom right",
        annotation_font=dict(size=10, color=thresh_color),
        secondary_y=False,
    )

    # Grade 3 threshold (ANC = 1.5)
    fig.add_hline(
        y=1.5,
        line_dash="dot",
        line_color=thresh_color,
        line_width=1,
        annotation_text="Grade 3 (ANC=1.5)",
        annotation_position="bottom right",
        annotation_font=dict(size=10, color=thresh_color),
        secondary_y=False,
    )

    fig.update_layout(
        title=dict(
            text=(
                "FOLFOX Cycle Simulation: Lab Value Trajectories (21-Day Cycle)"
                "<br><sup>85 mg/m² oxaliplatin, BSA 1.85 m², Total dose 157.25 mg"
                f" — {DYNAMICS_NOTE}</sup>"
            ),
            font=dict(size=16, color=title_color),
            x=0.5,
        ),
        template=template,
        paper_bgcolor=bg_color,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=12),
        ),
        margin=dict(t=120, l=60, r=60, b=60),
        width=1920,
        height=1080,
    )

    fig.update_xaxes(title_text="Day of Cycle", dtick=1, range=[0, 21])
    fig.update_yaxes(title_text="Lab Value", secondary_y=False)
    fig.update_yaxes(title_text="Tumor Marker (ng/mL)", secondary_y=True)

    return fig


def main():
    """Generate FOLFOX lab trajectory visualizations."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    png_dir = os.path.join(os.path.dirname(script_dir), "png")
    os.makedirs(png_dir, exist_ok=True)

    for dark_mode in [False, True]:
        suffix = "dark" if dark_mode else "light"
        fig = create_folfox_chart(dark_mode=dark_mode)

        html_path = os.path.join(script_dir, f"folfox_lab_trajectories_{suffix}.html")
        fig.write_html(html_path, include_plotlyjs=True)

        png_path = os.path.join(png_dir, f"folfox_lab_trajectories_{suffix}.png")
        fig.write_image(png_path, width=1920, height=1080, scale=2)

    print("folfox_lab_trajectories: done")


if __name__ == "__main__":
    main()

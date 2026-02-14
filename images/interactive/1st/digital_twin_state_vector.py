"""Patient-Specific Digital Twin: 8-Dimensional State Vector Dashboard.

Visualizes the complete patient digital twin state vector that drives
treatment simulation for FOLFOX colorectal cancer treatment.

Data source:
  - digital-twins/examples-twins/01_realtime_dt_synchronization.py
    STATE_CONFIG, OBSERVATION_CONFIG, DYNAMICS_PARAMS
"""

import os

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


STATE_VARIABLES = [
    {"symbol": "V", "name": "Tumor Volume", "baseline": 18.5, "range": [0.01, 100], "unit": "cm³", "noise": 0.5},
    {"symbol": "g", "name": "Growth Rate", "baseline": 0.01, "range": [-0.1, 0.1], "unit": "/day", "noise": 0.001},
    {"symbol": "d", "name": "Drug Effect", "baseline": 0.0, "range": [0, 1], "unit": "dimensionless", "noise": 0.01},
    {"symbol": "ANC", "name": "Neutrophils", "baseline": 4.5, "range": [0.1, 10], "unit": "×10⁹/L", "noise": 0.2},
    {"symbol": "Cr", "name": "Creatinine", "baseline": 0.85, "range": [0.3, 4.0], "unit": "mg/dL", "noise": 0.01},
    {"symbol": "Hb", "name": "Hemoglobin", "baseline": 14.1, "range": [4.0, 18.0], "unit": "g/dL", "noise": 0.05},
    {"symbol": "W", "name": "Weight", "baseline": 78, "range": [40, 150], "unit": "kg", "noise": 0.1},
    {"symbol": "E", "name": "ECOG Status", "baseline": 1.0, "range": [0, 4], "unit": "score", "noise": 0.05},
]


def simulate_tumor_trajectory():
    """Simulate 21-day tumor volume trajectory under FOLFOX."""
    days = np.arange(0, 22, 0.5)
    baseline_growth = 0.01  # /day
    drug_effect_peak = 0.03  # peak drug kill rate
    drug_onset_day = 1
    drug_decay = 0.1  # /day elimination

    volume = np.zeros_like(days)
    volume[0] = 18.5

    for i in range(1, len(days)):
        dt = days[i] - days[i - 1]
        t = days[i]
        # Drug effect: ramps up after day 1, then decays
        if t >= drug_onset_day:
            drug = drug_effect_peak * np.exp(-drug_decay * (t - drug_onset_day))
        else:
            drug = 0.0
        net_growth = baseline_growth - drug
        volume[i] = volume[i - 1] * np.exp(net_growth * dt)

    return days, volume


def create_state_vector_dashboard(dark_mode=False):
    """Create combined indicator cards + line chart dashboard."""
    if dark_mode:
        template = "plotly_dark"
        header_color = "#00bcd4"
        bg_color = "#1a1a2e"
        title_color = "#e0e0e0"
        line_color = "#00e5ff"
        band_color = "rgba(0,229,255,0.15)"
    else:
        template = "plotly_white"
        header_color = "#1565c0"
        bg_color = "#ffffff"
        title_color = "#1a1a2e"
        line_color = "#0d47a1"
        band_color = "rgba(13,71,161,0.1)"

    # Layout: 2 rows of 4 indicators + 1 large row for the chart
    fig = make_subplots(
        rows=3,
        cols=4,
        specs=[
            [{"type": "indicator"}] * 4,
            [{"type": "indicator"}] * 4,
            [{"type": "xy", "colspan": 4}, None, None, None],
        ],
        row_heights=[0.2, 0.2, 0.6],
        vertical_spacing=0.06,
    )

    # Add 8 indicator cards
    for i, var in enumerate(STATE_VARIABLES):
        row = (i // 4) + 1
        col = (i % 4) + 1

        fig.add_trace(
            go.Indicator(
                mode="number",
                value=var["baseline"],
                title=dict(
                    text=f"<b>{var['symbol']}</b> — {var['name']}",
                    font=dict(size=14, color=header_color),
                ),
                number=dict(
                    suffix=f" {var['unit']}",
                    font=dict(size=20),
                ),
                domain=dict(x=[0, 1], y=[0, 1]),
            ),
            row=row,
            col=col,
        )

    # Add tumor volume trajectory
    days, volume = simulate_tumor_trajectory()

    fig.add_trace(
        go.Scatter(
            x=days,
            y=volume,
            mode="lines",
            name="Tumor Volume (cm³)",
            line=dict(color=line_color, width=3),
            fill="tonexty" if not dark_mode else None,
        ),
        row=3,
        col=1,
    )

    # Add uncertainty band (±1σ)
    upper = volume * 1.05
    lower = volume * 0.95
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([days, days[::-1]]),
            y=np.concatenate([upper, lower[::-1]]),
            fill="toself",
            fillcolor=band_color,
            line=dict(width=0),
            name="±1σ uncertainty",
            showlegend=True,
        ),
        row=3,
        col=1,
    )

    # Annotations for key events
    fig.add_annotation(
        x=1,
        y=18.5,
        text="FOLFOX Day 1",
        showarrow=True,
        arrowhead=2,
        row=3,
        col=1,
        font=dict(size=11),
    )
    fig.add_annotation(
        x=10,
        y=volume[20],
        text="Nadir ~Day 10",
        showarrow=True,
        arrowhead=2,
        row=3,
        col=1,
        font=dict(size=11),
    )

    fig.update_xaxes(title_text="Day of Cycle", row=3, col=1)
    fig.update_yaxes(title_text="Tumor Volume (cm³)", row=3, col=1)

    fig.update_layout(
        title=dict(
            text=(
                "Patient-Specific Digital Twin: 8-Dimensional State Vector (FOLFOX Colorectal Cancer)"
                "<br><sup>Source: digital-twins/examples-twins/01_realtime_dt_synchronization.py"
                " — Patient TRIAL-CRC-042</sup>"
            ),
            font=dict(size=17, color=title_color),
            x=0.5,
        ),
        template=template,
        paper_bgcolor=bg_color,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.05,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(t=100, l=60, r=40, b=80),
        width=1920,
        height=1080,
    )

    return fig


def main():
    """Generate digital twin state vector visualizations."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    png_dir = os.path.join(os.path.dirname(os.path.dirname(script_dir)), "png", "1st")
    os.makedirs(png_dir, exist_ok=True)

    for dark_mode in [False, True]:
        suffix = "dark" if dark_mode else "light"
        fig = create_state_vector_dashboard(dark_mode=dark_mode)

        html_path = os.path.join(script_dir, f"digital_twin_state_vector_{suffix}.html")
        fig.write_html(html_path, include_plotlyjs=True)

        png_path = os.path.join(png_dir, f"digital_twin_state_vector_{suffix}.png")
        fig.write_image(png_path, width=1920, height=1080, scale=2)

    print("digital_twin_state_vector: done")


if __name__ == "__main__":
    main()

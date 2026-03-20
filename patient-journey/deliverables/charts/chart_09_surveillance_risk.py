"""
Chart 09 - Surveillance Recurrence Risk (PAT-2026-0042)
========================================================
Area chart showing recurrence risk declining over the surveillance period
with quarterly imaging timepoints marked.

Patient: 58F, Stage IIIB NSCLC
RESEARCH USE ONLY - Not for clinical decision-making.
LICENSE: MIT
"""

try:
    import plotly.graph_objects as go
except ImportError:
    raise ImportError("plotly is required: pip install plotly")

import numpy as np


def generate_surveillance_risk() -> go.Figure:
    """Generate an area chart of declining recurrence risk during surveillance."""

    np.random.seed(42)

    # Surveillance from Day 763 (treatment completion) to Day 1858 (5-year mark)
    start_day = 763
    end_day = 1858
    days = np.linspace(start_day, end_day, 200)

    # Risk declines from 35% to 3%
    t_norm = (days - start_day) / (end_day - start_day)
    risk = 35 * np.exp(-3.5 * t_norm) + 3 * (1 - np.exp(-3.5 * t_norm))
    # Ensure endpoints
    risk[0] = 35.0
    risk[-1] = 3.0

    # Quarterly imaging timepoints (~90 days apart)
    imaging_days = list(range(start_day, end_day + 1, 90))
    imaging_risk = np.interp(imaging_days, days, risk)

    fig = go.Figure()

    # Risk area
    fig.add_trace(
        go.Scatter(
            x=days,
            y=risk,
            mode="lines",
            name="Recurrence Risk (%)",
            line=dict(color="#d62728", width=2),
            fill="tozeroy",
            fillcolor="rgba(214, 39, 40, 0.15)",
            hovertemplate="Day %{x:.0f}<br>Risk: %{y:.1f}%<extra></extra>",
        )
    )

    # Imaging timepoints
    fig.add_trace(
        go.Scatter(
            x=imaging_days,
            y=imaging_risk,
            mode="markers",
            name="Quarterly Imaging",
            marker=dict(size=8, color="#1f77b4", symbol="diamond", line=dict(width=1, color="navy")),
            hovertemplate="Day %{x}<br>Risk at scan: %{y:.1f}%<extra></extra>",
        )
    )

    # Key annotations
    fig.add_annotation(
        x=start_day,
        y=35,
        text="Treatment Complete<br>Risk: 35%",
        showarrow=True,
        arrowhead=2,
        ax=-60,
        ay=-30,
        font=dict(size=10),
    )
    fig.add_annotation(
        x=end_day,
        y=3,
        text="5-Year Mark<br>Risk: 3%",
        showarrow=True,
        arrowhead=2,
        ax=60,
        ay=-30,
        font=dict(size=10),
    )

    fig.update_layout(
        title=dict(
            text=(
                "Surveillance Recurrence Risk - PAT-2026-0042 (58F, Stage IIIB NSCLC)"
                "<br><sup>RESEARCH USE ONLY - Not for clinical decision-making.</sup>"
            ),
            x=0.5,
        ),
        xaxis=dict(
            title="Study Day",
            gridcolor="lightgrey",
            dtick=180,
        ),
        yaxis=dict(
            title="Recurrence Risk (%)",
            range=[0, 40],
            gridcolor="lightgrey",
        ),
        template="plotly_white",
        height=480,
        legend=dict(x=0.65, y=0.95),
        margin=dict(t=80, b=60),
    )

    return fig


if __name__ == "__main__":
    figure = generate_surveillance_risk()
    output_path = "chart_09_surveillance_risk.html"
    figure.write_html(output_path)
    print(f"Saved: {output_path}")

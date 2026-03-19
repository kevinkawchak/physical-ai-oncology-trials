"""
Chart 07 - Immunotherapy Cycles (PAT-2026-0042)
================================================
Line chart of 35 pembrolizumab cycles (200 mg q3w) with adverse events
and declining recurrence risk.

Patient: 58F, Stage IIIB NSCLC
RESEARCH USE ONLY - Not for clinical decision-making.
LICENSE: MIT
"""

try:
    import plotly.graph_objects as go
except ImportError:
    raise ImportError("plotly is required: pip install plotly")

import numpy as np


def generate_immunotherapy_cycles() -> go.Figure:
    """Generate a line chart of pembrolizumab cycles with AEs and recurrence risk."""

    np.random.seed(42)

    cycles = list(range(1, 36))
    # Recurrence risk declines from 35% to 8% roughly exponentially
    risk = 35 * np.exp(-np.linspace(0, 1.47, 35))  # 35 -> ~8
    risk = np.clip(risk, 8, 35)

    fig = go.Figure()

    # Recurrence risk line
    fig.add_trace(
        go.Scatter(
            x=cycles,
            y=risk,
            mode="lines+markers",
            name="Recurrence Risk (%)",
            line=dict(color="#d62728", width=2),
            marker=dict(size=5),
            hovertemplate="Cycle %{x}<br>Risk: %{y:.1f}%<extra></extra>",
        )
    )

    # Adverse event markers
    ae_cycles = [6, 12]
    ae_risks = [float(risk[5]), float(risk[11])]
    ae_labels = ["Hypothyroidism (Gr 1)", "Rash (Gr 1)"]

    fig.add_trace(
        go.Scatter(
            x=ae_cycles,
            y=ae_risks,
            mode="markers+text",
            name="Adverse Events",
            marker=dict(size=14, color="#ff7f0e", symbol="triangle-up", line=dict(width=1, color="red")),
            text=ae_labels,
            textposition=["top right", "top right"],
            textfont=dict(size=10),
            hovertemplate="<b>%{text}</b><br>Cycle %{x}<extra></extra>",
        )
    )

    # Dose annotation
    fig.add_annotation(
        text="Pembrolizumab 200 mg q3w",
        xref="paper",
        yref="paper",
        x=0.02,
        y=0.98,
        showarrow=False,
        font=dict(size=11, color="grey"),
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="grey",
        borderwidth=1,
    )

    fig.update_layout(
        title=dict(
            text=(
                "Immunotherapy Cycles - PAT-2026-0042 (58F, Stage IIIB NSCLC)"
                "<br><sup>RESEARCH USE ONLY - Not for clinical decision-making.</sup>"
            ),
            x=0.5,
        ),
        xaxis=dict(
            title="Cycle Number",
            dtick=5,
            range=[0, 36],
            gridcolor="lightgrey",
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
    figure = generate_immunotherapy_cycles()
    output_path = "chart_07_immunotherapy_cycles.html"
    figure.write_html(output_path)
    print(f"Saved: {output_path}")

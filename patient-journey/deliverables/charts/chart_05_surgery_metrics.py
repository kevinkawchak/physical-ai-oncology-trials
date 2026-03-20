"""
Chart 05 - Surgery Metrics (PAT-2026-0042)
===========================================
Multi-metric bar chart display for surgical procedure outcomes.

Patient: 58F, Stage IIIB NSCLC
RESEARCH USE ONLY - Not for clinical decision-making.
LICENSE: MIT
"""

try:
    import plotly.graph_objects as go
except ImportError:
    raise ImportError("plotly is required: pip install plotly")


def generate_surgery_metrics() -> go.Figure:
    """Generate a bar chart of key surgery metrics."""

    metrics = [
        "Duration (min)",
        "EBL (mL)",
        "LN Sampled",
        "LN Positive",
        "USL Score",
        "Autonomy Level",
    ]
    values = [168, 85, 18, 3, 7.9, 2]
    # Normalize for display: show raw values but color by clinical significance
    colors = [
        "#1f77b4",  # duration - neutral
        "#2ca02c",  # EBL - good (low)
        "#1f77b4",  # LN sampled - neutral
        "#d62728",  # LN positive - concerning
        "#2ca02c",  # USL - good
        "#ff7f0e",  # autonomy level - informational
    ]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=metrics,
            y=values,
            marker=dict(color=colors, line=dict(color="darkgrey", width=1)),
            text=[f"{v}" for v in values],
            textposition="outside",
            textfont=dict(size=14, weight="bold"),
            hovertemplate="<b>%{x}</b><br>Value: %{y}<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(
            text=(
                "Surgery Metrics - PAT-2026-0042 (58F, Stage IIIB NSCLC)"
                "<br><sup>RESEARCH USE ONLY - Not for clinical decision-making.</sup>"
            ),
            x=0.5,
        ),
        xaxis=dict(title=""),
        yaxis=dict(title="Value", showgrid=True, gridcolor="lightgrey"),
        template="plotly_white",
        height=480,
        margin=dict(t=80, b=80),
        showlegend=False,
        annotations=[
            dict(
                text="Margins: NEGATIVE (R0 resection) | Robot: da Vinci Xi",
                xref="paper",
                yref="paper",
                x=0.5,
                y=-0.18,
                showarrow=False,
                font=dict(size=12, color="#2ca02c", weight="bold"),
            )
        ],
    )

    return fig


if __name__ == "__main__":
    figure = generate_surgery_metrics()
    output_path = "chart_05_surgery_metrics.html"
    figure.write_html(output_path)
    print(f"Saved: {output_path}")

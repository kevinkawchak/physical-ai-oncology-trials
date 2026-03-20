"""
Chart 08 - Federated Learning Convergence (PAT-2026-0042)
==========================================================
Line chart of federated learning convergence over 70 rounds with
privacy budget tracking and 5-site aggregation.

Patient: 58F, Stage IIIB NSCLC
RESEARCH USE ONLY - Not for clinical decision-making.
LICENSE: MIT
"""

try:
    import plotly.graph_objects as go
except ImportError:
    raise ImportError("plotly is required: pip install plotly")

import numpy as np


def generate_federation_convergence() -> go.Figure:
    """Generate a line chart of federated learning convergence."""

    np.random.seed(42)

    rounds = np.arange(1, 71)
    # Model loss: starts ~2.5, converges to ~0.15 with noise
    base_loss = 2.5 * np.exp(-0.06 * rounds) + 0.15
    noise = np.random.normal(0, 0.03, len(rounds))
    model_loss = np.clip(base_loss + noise, 0.1, 3.0)

    # Privacy budget consumed (epsilon accumulates from 0 toward 1.0)
    epsilon_consumed = 1.0 * (1 - np.exp(-0.05 * rounds))

    fig = go.Figure()

    # Model loss (primary y-axis)
    fig.add_trace(
        go.Scatter(
            x=rounds,
            y=model_loss,
            mode="lines",
            name="Model Loss",
            line=dict(color="#1f77b4", width=2),
            hovertemplate="Round %{x}<br>Loss: %{y:.3f}<extra></extra>",
        )
    )

    # Privacy budget (secondary y-axis)
    fig.add_trace(
        go.Scatter(
            x=rounds,
            y=epsilon_consumed,
            mode="lines",
            name="Privacy Budget (epsilon)",
            line=dict(color="#d62728", width=2, dash="dash"),
            yaxis="y2",
            hovertemplate="Round %{x}<br>Epsilon: %{y:.3f}<extra></extra>",
        )
    )

    # Privacy limit line on secondary axis
    fig.add_trace(
        go.Scatter(
            x=[1, 70],
            y=[1.0, 1.0],
            mode="lines",
            name="Epsilon Limit (1.0)",
            line=dict(color="red", width=1, dash="dot"),
            yaxis="y2",
            showlegend=True,
        )
    )

    # Site labels
    sites = ["Site A (Academic)", "Site B (Community)", "Site C (VA)", "Site D (Intl.)", "Site E (Pediatric)"]
    fig.add_annotation(
        text="5-Site Aggregation:<br>" + "<br>".join(f"  {s}" for s in sites),
        xref="paper",
        yref="paper",
        x=0.98,
        y=0.98,
        showarrow=False,
        font=dict(size=9, color="grey"),
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="grey",
        borderwidth=1,
        align="left",
    )

    fig.update_layout(
        title=dict(
            text=(
                "Federated Learning Convergence - PAT-2026-0042"
                "<br><sup>RESEARCH USE ONLY - Not for clinical decision-making.</sup>"
            ),
            x=0.5,
        ),
        xaxis=dict(
            title="Aggregation Round",
            dtick=10,
            gridcolor="lightgrey",
        ),
        yaxis=dict(
            title="Model Loss",
            range=[0, 3.0],
            gridcolor="lightgrey",
        ),
        yaxis2=dict(
            title="Privacy Budget (epsilon)",
            overlaying="y",
            side="right",
            range=[0, 1.2],
        ),
        template="plotly_white",
        height=500,
        legend=dict(x=0.02, y=0.98, font=dict(size=10)),
        margin=dict(t=80, b=60, r=80),
    )

    return fig


if __name__ == "__main__":
    figure = generate_federation_convergence()
    output_path = "chart_08_federation_convergence.html"
    figure.write_html(output_path)
    print(f"Saved: {output_path}")

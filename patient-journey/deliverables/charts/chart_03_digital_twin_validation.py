"""
Chart 03 - Digital Twin Validation (PAT-2026-0042)
===================================================
Radar chart of ASME V&V 40 digital twin validation scores.

Patient: 58F, Stage IIIB NSCLC
RESEARCH USE ONLY - Not for clinical decision-making.
LICENSE: MIT
"""

try:
    import plotly.graph_objects as go
except ImportError:
    raise ImportError("plotly is required: pip install plotly")

import numpy as np


def generate_digital_twin_validation() -> go.Figure:
    """Generate a radar chart of ASME V&V 40 validation scores."""

    np.random.seed(42)

    categories = [
        "Geometric Accuracy",
        "Material Properties",
        "Boundary Conditions",
        "Solver Convergence",
        "Experimental Comparison",
    ]
    scores = [0.92, 0.88, 0.91, 0.95, 0.87]
    thresholds = [0.80] * 5

    # Close the radar polygon
    categories_closed = categories + [categories[0]]
    scores_closed = scores + [scores[0]]
    thresholds_closed = thresholds + [thresholds[0]]

    fig = go.Figure()

    # Threshold ring
    fig.add_trace(
        go.Scatterpolar(
            r=thresholds_closed,
            theta=categories_closed,
            fill="toself",
            fillcolor="rgba(255, 0, 0, 0.05)",
            line=dict(color="red", dash="dash", width=1),
            name="Minimum Threshold (0.80)",
        )
    )

    # Actual scores
    fig.add_trace(
        go.Scatterpolar(
            r=scores_closed,
            theta=categories_closed,
            fill="toself",
            fillcolor="rgba(31, 119, 180, 0.25)",
            line=dict(color="#1f77b4", width=2),
            name="PAT-2026-0042 Scores",
            text=[f"{s:.2f}" for s in scores_closed],
            textposition="top center",
            mode="lines+markers+text",
            marker=dict(size=8),
        )
    )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0.5, 1.0],
                dtick=0.1,
            ),
        ),
        title=dict(
            text=(
                "Digital Twin Validation (ASME V&V 40) - PAT-2026-0042"
                "<br><sup>RESEARCH USE ONLY - Not for clinical decision-making.</sup>"
            ),
            x=0.5,
        ),
        template="plotly_white",
        height=550,
        showlegend=True,
        legend=dict(x=0.02, y=-0.1, orientation="h"),
        annotations=[
            dict(
                text=f"Mean Score: {np.mean(scores):.2f}",
                xref="paper",
                yref="paper",
                x=0.98,
                y=0.02,
                showarrow=False,
                font=dict(size=12, color="#1f77b4"),
            )
        ],
    )

    return fig


if __name__ == "__main__":
    figure = generate_digital_twin_validation()
    output_path = "chart_03_digital_twin_validation.html"
    figure.write_html(output_path)
    print(f"Saved: {output_path}")

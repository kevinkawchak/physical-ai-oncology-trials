"""
Chart 04 - Robot USL Scores (PAT-2026-0042)
============================================
Grouped bar chart of Universal Skills Level (USL) scores for surgical robots.

Patient: 58F, Stage IIIB NSCLC
RESEARCH USE ONLY - Not for clinical decision-making.
LICENSE: MIT
"""

try:
    import plotly.graph_objects as go
except ImportError:
    raise ImportError("plotly is required: pip install plotly")


def generate_robot_usl_scores() -> go.Figure:
    """Generate a grouped bar chart comparing USL scores for two robots."""

    metrics = ["Autonomy", "Dexterity", "Safety", "Interop", "Composite"]

    davinci_scores = [6.8, 8.5, 8.2, 8.1, 7.9]
    franka_scores = [6.5, 7.8, 7.5, 7.0, 7.2]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=metrics,
            y=davinci_scores,
            name="da Vinci Xi",
            marker=dict(color="#1f77b4"),
            text=[f"{v:.1f}" for v in davinci_scores],
            textposition="outside",
        )
    )

    fig.add_trace(
        go.Bar(
            x=metrics,
            y=franka_scores,
            name="Franka Emika",
            marker=dict(color="#ff7f0e"),
            text=[f"{v:.1f}" for v in franka_scores],
            textposition="outside",
        )
    )

    # Surgical threshold line
    fig.add_hline(
        y=7.0,
        line=dict(color="red", dash="dash", width=2),
        annotation_text="Surgical Threshold (7.0)",
        annotation_position="top right",
        annotation_font=dict(color="red", size=11),
    )

    # Cobot threshold line
    fig.add_hline(
        y=5.0,
        line=dict(color="orange", dash="dot", width=2),
        annotation_text="Cobot Threshold (5.0)",
        annotation_position="bottom right",
        annotation_font=dict(color="orange", size=11),
    )

    fig.update_layout(
        title=dict(
            text=(
                "Robot USL Scores - PAT-2026-0042 (58F, Stage IIIB NSCLC)"
                "<br><sup>RESEARCH USE ONLY - Not for clinical decision-making.</sup>"
            ),
            x=0.5,
        ),
        xaxis=dict(title="USL Metric"),
        yaxis=dict(title="Score", range=[0, 10], dtick=1),
        barmode="group",
        template="plotly_white",
        height=500,
        legend=dict(x=0.02, y=0.98),
        margin=dict(t=80, b=60),
    )

    return fig


if __name__ == "__main__":
    figure = generate_robot_usl_scores()
    output_path = "chart_04_robot_usl_scores.html"
    figure.write_html(output_path)
    print(f"Saved: {output_path}")

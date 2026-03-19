"""
Chart 01 - Pre-Screening Timeline (PAT-2026-0042)
==================================================
Timeline bar chart of pre-screening steps from Day -30 to Day -14.

Patient: 58F, Stage IIIB NSCLC
RESEARCH USE ONLY - Not for clinical decision-making.
LICENSE: MIT
"""

try:
    import plotly.graph_objects as go
except ImportError:
    raise ImportError("plotly is required: pip install plotly")


def generate_prescreening_timeline() -> go.Figure:
    """Generate a horizontal bar (Gantt-style) chart of pre-screening steps."""

    steps = [
        "PHI Scan",
        "De-identification",
        "Data Harmonization",
        "DICOM Validation",
        "Patient Record Creation",
        "Access Provisioning",
    ]
    start_days = [-30, -27, -24, -21, -18, -16]
    end_days = [-27, -24, -21, -18, -16, -14]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    fig = go.Figure()

    for i, step in enumerate(steps):
        duration = end_days[i] - start_days[i]
        fig.add_trace(
            go.Bar(
                y=[step],
                x=[duration],
                base=[start_days[i]],
                orientation="h",
                marker=dict(color=colors[i]),
                text=f"Day {start_days[i]} to {end_days[i]}",
                textposition="inside",
                hovertemplate=(
                    f"<b>{step}</b><br>"
                    f"Start: Day {start_days[i]}<br>"
                    f"End: Day {end_days[i]}<br>"
                    f"Duration: {duration} days<extra></extra>"
                ),
                showlegend=False,
            )
        )

    fig.update_layout(
        title=dict(
            text=(
                "Pre-Screening Timeline - PAT-2026-0042 (58F, Stage IIIB NSCLC)"
                "<br><sup>RESEARCH USE ONLY - Not for clinical decision-making.</sup>"
            ),
            x=0.5,
        ),
        xaxis=dict(
            title="Study Day",
            dtick=2,
            range=[-32, -12],
            gridcolor="lightgrey",
        ),
        yaxis=dict(
            title="",
            autorange="reversed",
        ),
        barmode="overlay",
        template="plotly_white",
        height=400,
        margin=dict(l=180, r=40, t=80, b=60),
        annotations=[
            dict(
                text="Day 0 = Enrollment",
                xref="paper",
                yref="paper",
                x=1.0,
                y=-0.18,
                showarrow=False,
                font=dict(size=10, color="grey"),
            )
        ],
    )

    return fig


if __name__ == "__main__":
    figure = generate_prescreening_timeline()
    output_path = "chart_01_prescreening_timeline.html"
    figure.write_html(output_path)
    print(f"Saved: {output_path}")

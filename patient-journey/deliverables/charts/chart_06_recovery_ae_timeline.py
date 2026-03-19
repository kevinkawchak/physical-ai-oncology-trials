"""
Chart 06 - Recovery Adverse Event Timeline (PAT-2026-0042)
==========================================================
Timeline/scatter showing adverse events and milestones during recovery.

Patient: 58F, Stage IIIB NSCLC
RESEARCH USE ONLY - Not for clinical decision-making.
LICENSE: MIT
"""

try:
    import plotly.graph_objects as go
except ImportError:
    raise ImportError("plotly is required: pip install plotly")


def generate_recovery_ae_timeline() -> go.Figure:
    """Generate a timeline scatter of adverse events during post-surgical recovery."""

    # Recovery spans roughly Day 15 to Day 25
    ae_days = [16]
    ae_names = ["Atrial Fibrillation"]
    ae_grades = [2]
    ae_colors = ["#ff7f0e"]  # Grade 2 = orange

    milestone_days = [15, 18, 21]
    milestone_names = ["Surgery Complete", "Pathology: pT2aN2M0", "Discharge"]
    milestone_symbols = ["diamond", "star", "triangle-up"]

    fig = go.Figure()

    # Adverse events
    fig.add_trace(
        go.Scatter(
            x=ae_days,
            y=[1] * len(ae_days),
            mode="markers+text",
            marker=dict(size=18, color=ae_colors, symbol="x", line=dict(width=2, color="darkred")),
            text=[f"{n} (Grade {g})" for n, g in zip(ae_names, ae_grades)],
            textposition="top center",
            textfont=dict(size=11),
            name="Adverse Events",
            hovertemplate="<b>%{text}</b><br>Day %{x}<extra></extra>",
        )
    )

    # Milestones
    for i, (day, name, sym) in enumerate(zip(milestone_days, milestone_names, milestone_symbols)):
        fig.add_trace(
            go.Scatter(
                x=[day],
                y=[0.5],
                mode="markers+text",
                marker=dict(size=14, symbol=sym, color="#1f77b4", line=dict(width=1, color="navy")),
                text=[name],
                textposition="bottom center",
                textfont=dict(size=10),
                name=name,
                hovertemplate=f"<b>{name}</b><br>Day {day}<extra></extra>",
            )
        )

    # Vertical reference lines for key days
    for day, label in zip([15, 21], ["Surgery", "Discharge"]):
        fig.add_vline(
            x=day,
            line=dict(color="grey", dash="dot", width=1),
        )

    fig.update_layout(
        title=dict(
            text=(
                "Recovery & Adverse Event Timeline - PAT-2026-0042 (58F, Stage IIIB NSCLC)"
                "<br><sup>RESEARCH USE ONLY - Not for clinical decision-making.</sup>"
            ),
            x=0.5,
        ),
        xaxis=dict(
            title="Study Day",
            dtick=1,
            range=[14, 23],
            gridcolor="lightgrey",
        ),
        yaxis=dict(
            title="",
            range=[0, 1.8],
            showticklabels=False,
            showgrid=False,
        ),
        template="plotly_white",
        height=400,
        showlegend=True,
        legend=dict(x=0.02, y=0.98, font=dict(size=10)),
        margin=dict(t=80, b=60),
    )

    return fig


if __name__ == "__main__":
    figure = generate_recovery_ae_timeline()
    output_path = "chart_06_recovery_ae_timeline.html"
    figure.write_html(output_path)
    print(f"Saved: {output_path}")

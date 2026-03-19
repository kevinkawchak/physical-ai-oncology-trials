"""
Chart 02 - Enrollment Eligibility Criteria (PAT-2026-0042)
==========================================================
Horizontal bar chart showing inclusion/exclusion criteria pass/fail.

Patient: 58F, Stage IIIB NSCLC
RESEARCH USE ONLY - Not for clinical decision-making.
LICENSE: MIT
"""

try:
    import plotly.graph_objects as go
except ImportError:
    raise ImportError("plotly is required: pip install plotly")


def generate_enrollment_eligibility() -> go.Figure:
    """Generate a horizontal bar chart of eligibility criteria with pass/fail status."""

    criteria = [
        "Confirmed NSCLC",
        "Stage IIIB",
        "ECOG 0-1",
        "PD-L1 >= 50%",
        "Adequate Organ Function",
        "Age >= 18",
        "No Prior Immunotherapy",
        "No Autoimmune Disease",
    ]
    statuses = ["PASS"] * 8
    details = [
        "Histologically confirmed",
        "TNM staging verified",
        "ECOG PS = 1",
        "PD-L1 TPS = 65%",
        "All labs within range",
        "Age = 58",
        "Treatment-naive",
        "No history documented",
    ]
    # All pass -> bar length = 1.0 (normalized confidence)
    bar_values = [1.0, 1.0, 1.0, 0.65, 1.0, 1.0, 1.0, 1.0]
    colors = ["#2ca02c"] * 8  # green for pass

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            y=criteria,
            x=bar_values,
            orientation="h",
            marker=dict(color=colors, line=dict(color="#1a7a1a", width=1)),
            text=[f"{s} - {d}" for s, d in zip(statuses, details)],
            textposition="inside",
            textfont=dict(color="white", size=12),
            hovertemplate=("<b>%{y}</b><br>Status: PASS<br>%{text}<extra></extra>"),
        )
    )

    fig.update_layout(
        title=dict(
            text=(
                "Enrollment Eligibility - PAT-2026-0042 (58F, Stage IIIB NSCLC)"
                "<br><sup>RESEARCH USE ONLY - Not for clinical decision-making.</sup>"
            ),
            x=0.5,
        ),
        xaxis=dict(
            title="Criteria Score (1.0 = Pass)",
            range=[0, 1.15],
            showgrid=True,
            gridcolor="lightgrey",
        ),
        yaxis=dict(
            title="",
            autorange="reversed",
        ),
        template="plotly_white",
        height=450,
        margin=dict(l=200, r=40, t=80, b=60),
        showlegend=False,
        annotations=[
            dict(
                text="All 8/8 criteria met - Patient eligible for enrollment",
                xref="paper",
                yref="paper",
                x=0.5,
                y=-0.15,
                showarrow=False,
                font=dict(size=12, color="#1a7a1a", weight="bold"),
            )
        ],
    )

    return fig


if __name__ == "__main__":
    figure = generate_enrollment_eligibility()
    output_path = "chart_02_enrollment_eligibility.html"
    figure.write_html(output_path)
    print(f"Saved: {output_path}")

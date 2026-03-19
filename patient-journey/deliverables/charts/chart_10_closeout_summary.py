"""
Chart 10 - Trial Closeout Summary (PAT-2026-0042)
==================================================
Treemap of trial contribution showing treatment arm, best response,
hazard ratio, event-free months, data quality, and regulatory compliance.

Patient: 58F, Stage IIIB NSCLC
RESEARCH USE ONLY - Not for clinical decision-making.
LICENSE: MIT
"""

try:
    import plotly.graph_objects as go
except ImportError:
    raise ImportError("plotly is required: pip install plotly")


def generate_closeout_summary() -> go.Figure:
    """Generate a treemap summarizing trial closeout data."""

    labels = [
        "PAT-2026-0042<br>Trial Closeout",
        # Level 1 categories
        "Treatment Outcome",
        "Data Quality",
        "Regulatory Compliance",
        # Level 2 - Treatment Outcome
        "Treatment Arm:<br>Pembrolizumab + Surgery",
        "Best Response:<br>CR (Complete Response)",
        "Hazard Ratio:<br>0.62",
        "Event-Free:<br>36 months",
        # Level 2 - Data Quality
        "CRF Completion:<br>100%",
        "Query Resolution:<br>98.5%",
        "Protocol Deviations:<br>0 major",
        "AE Reporting:<br>100% timely",
        # Level 2 - Regulatory Compliance
        "ICF Signed:<br>Day -14",
        "SAE Reports:<br>0 (none required)",
        "Audit Trail:<br>Complete",
        "21 CFR Part 11:<br>Compliant",
    ]

    parents = [
        "",
        # Level 1
        "PAT-2026-0042<br>Trial Closeout",
        "PAT-2026-0042<br>Trial Closeout",
        "PAT-2026-0042<br>Trial Closeout",
        # Level 2 - Treatment Outcome
        "Treatment Outcome",
        "Treatment Outcome",
        "Treatment Outcome",
        "Treatment Outcome",
        # Level 2 - Data Quality
        "Data Quality",
        "Data Quality",
        "Data Quality",
        "Data Quality",
        # Level 2 - Regulatory Compliance
        "Regulatory Compliance",
        "Regulatory Compliance",
        "Regulatory Compliance",
        "Regulatory Compliance",
    ]

    values = [
        0,
        # Level 1 (sum of children)
        0,
        0,
        0,
        # Level 2 - Treatment Outcome
        10,
        10,
        10,
        10,
        # Level 2 - Data Quality
        10,
        10,
        10,
        10,
        # Level 2 - Regulatory Compliance
        10,
        10,
        10,
        10,
    ]

    colors = [
        "#ffffff",
        # Level 1
        "#1f77b4",
        "#2ca02c",
        "#9467bd",
        # Level 2 - Treatment Outcome (blues)
        "#aec7e8",
        "#6baed6",
        "#3182bd",
        "#08519c",
        # Level 2 - Data Quality (greens)
        "#a1d99b",
        "#74c476",
        "#31a354",
        "#006d2c",
        # Level 2 - Regulatory Compliance (purples)
        "#bcbddc",
        "#9e9ac8",
        "#756bb1",
        "#54278f",
    ]

    fig = go.Figure(
        go.Treemap(
            labels=labels,
            parents=parents,
            values=values,
            branchvalues="total",
            marker=dict(colors=colors, line=dict(width=2, color="white")),
            textfont=dict(size=12),
            hovertemplate="<b>%{label}</b><extra></extra>",
            textinfo="label",
        )
    )

    fig.update_layout(
        title=dict(
            text=(
                "Trial Closeout Summary - PAT-2026-0042 (58F, Stage IIIB NSCLC)"
                "<br><sup>RESEARCH USE ONLY - Not for clinical decision-making.</sup>"
            ),
            x=0.5,
        ),
        template="plotly_white",
        height=600,
        margin=dict(t=80, b=30, l=10, r=10),
    )

    return fig


if __name__ == "__main__":
    figure = generate_closeout_summary()
    output_path = "chart_10_closeout_summary.html"
    figure.write_html(output_path)
    print(f"Saved: {output_path}")

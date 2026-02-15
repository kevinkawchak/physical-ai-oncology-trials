"""FDA-Authorized AI/ML Oncology Device Distribution.

Horizontal stacked bar chart with pie chart inset showing the breakdown
of 1,300+ FDA-authorized AI/ML devices by oncology subspecialty.

Data sources:
  - agentic-ai/results.md — 1,300+ FDA AI/ML authorized devices (Dec 2025)
  - regulatory-submit/classification_advisor.py — product codes
  - regulatory/fda-compliance/fda_submission_tracker.py — pathway categories
  - regulatory/regulatory-intelligence/regulatory_tracker.py
"""

import os

import plotly.graph_objects as go
from plotly.subplots import make_subplots

CATEGORIES = [
    "Cancer Radiology (CT/MRI/Mammography)",
    "Pathology (Digital/Computational)",
    "Radiation Oncology",
    "Gastroenterology (Endoscopy AI)",
    "Clinical Oncology (Prognostic/Treatment)",
    "Other",
]

PERCENTAGES = [54.9, 19.7, 8.5, 8.5, 7.0, 1.4]
COUNTS = [714, 256, 111, 111, 91, 18]

RECENT_APPROVALS = [
    "ArteraAI Prostate (Aug 2025, De Novo)",
    "Allix5 Breast Cancer Risk (May 2025)",
    "Serial CTRS NSCLC (Feb 2025, Breakthrough)",
]


def create_chart(dark_mode=False):
    """Create horizontal stacked bar + pie chart."""
    if dark_mode:
        template = "plotly_dark"
        colors = ["#5dade2", "#58d68d", "#f7dc6f", "#eb984e", "#af7ac5", "#85929e"]
        text_color = "#e0e0e0"
        bg_color = "#1a1a2e"
        ann_color = "#b0bec5"
        border_color = "#4a4a5a"
    else:
        template = "plotly_white"
        colors = ["#2980b9", "#27ae60", "#f1c40f", "#e67e22", "#8e44ad", "#95a5a6"]
        text_color = "#1a1a2e"
        bg_color = "#ffffff"
        ann_color = "#616161"
        border_color = "#bdc3c7"

    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[[{"type": "bar", "colspan": 2}, None], [{"type": "pie"}, {"type": "table"}]],
        row_heights=[0.45, 0.55],
        vertical_spacing=0.12,
        subplot_titles=[
            "Category Distribution (Stacked Bar)",
            "Proportion by Subspecialty",
            "Recent Oncology Approvals",
        ],
    )

    # Stacked horizontal bar
    for i, (cat, pct, cnt) in enumerate(zip(CATEGORIES, PERCENTAGES, COUNTS)):
        fig.add_trace(
            go.Bar(
                name=cat,
                y=["FDA AI/ML Oncology Devices"],
                x=[pct],
                orientation="h",
                marker_color=colors[i],
                text=[f"{cat}<br>{pct}% (~{cnt})"],
                textposition="inside",
                textfont=dict(size=11, color="#ffffff"),
                insidetextanchor="middle",
            ),
            row=1,
            col=1,
        )

    # Pie chart
    fig.add_trace(
        go.Pie(
            labels=CATEGORIES,
            values=PERCENTAGES,
            marker=dict(colors=colors, line=dict(color=border_color, width=1)),
            textinfo="percent+label",
            textfont=dict(size=10),
            hole=0.3,
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    # Recent approvals table
    fig.add_trace(
        go.Table(
            header=dict(
                values=["<b>Recent FDA Oncology AI/ML Approvals</b>"],
                fill_color=colors[0],
                font=dict(size=13, color="#ffffff"),
                align="center",
                height=35,
            ),
            cells=dict(
                values=[RECENT_APPROVALS],
                fill_color=bg_color,
                font=dict(size=12, color=text_color),
                align="left",
                height=30,
            ),
        ),
        row=2,
        col=2,
    )

    fig.update_layout(
        title=dict(
            text=(
                "FDA-Authorized AI/ML Devices in Oncology: Category Distribution"
                " (1,300+ as of Dec 2025)"
                "<br><sup>Sources: agentic-ai/results.md;"
                " regulatory-submit/classification_advisor.py</sup>"
            ),
            font=dict(size=18, color=text_color),
            x=0.5,
        ),
        template=template,
        paper_bgcolor=bg_color,
        barmode="stack",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=11),
        ),
        margin=dict(t=140, l=40, r=40, b=60),
        width=1920,
        height=1080,
    )

    return fig


def main():
    """Generate FDA oncology device distribution visualizations."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    png_dir = os.path.join(os.path.dirname(os.path.dirname(script_dir)), "png", "3rd")
    os.makedirs(png_dir, exist_ok=True)

    for dark_mode in [False, True]:
        suffix = "dark" if dark_mode else "light"
        fig = create_chart(dark_mode=dark_mode)

        html_path = os.path.join(script_dir, f"fda_oncology_device_distribution_{suffix}.html")
        fig.write_html(html_path, include_plotlyjs=True)

        png_path = os.path.join(png_dir, f"fda_oncology_device_distribution_{suffix}.png")
        fig.write_image(png_path, width=1920, height=1080, scale=2)

    print("fda_oncology_device_distribution: done")


if __name__ == "__main__":
    main()

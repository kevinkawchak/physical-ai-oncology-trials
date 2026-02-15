"""Regulatory Compliance Checklist Scorecard.

Annotated heatmap showing all compliance items across IEC 62304,
FDA AI/ML PCCP, and ISO 14971 standards with their descriptions.

Data sources:
  - tools/deployment-readiness/deployment_readiness.py — IEC_62304_CHECKLIST
    (8 items), FDA_AIML_CHECKLIST (6 items), ISO_14971_CHECKLIST (5 items)
"""

import os

import plotly.graph_objects as go

IEC_62304 = [
    ("62304-5.1", "Software development planning", "Planning"),
    ("62304-5.2", "Software requirements analysis", "Requirements"),
    ("62304-5.3", "Software architectural design", "Design"),
    ("62304-5.5", "Software integration and testing", "Testing"),
    ("62304-5.7", "Software release", "Release"),
    ("62304-7.1", "Risk analysis for software items", "Risk"),
    ("62304-8", "Software configuration management", "Config"),
    ("62304-9", "Software problem resolution", "Maintenance"),
]

FDA_AIML = [
    ("PCCP-1", "Modification protocol", "Change Control"),
    ("PCCP-2", "Performance monitoring plan", "Monitoring"),
    ("PCCP-3", "Update validation protocol", "Validation"),
    ("PCCP-4", "Data management governance", "Data"),
    ("PCCP-5", "Transparency/labeling documentation", "Transparency"),
    ("PCCP-6", "Bias evaluation across demographics", "Equity"),
]

ISO_14971 = [
    ("14971-4", "Hazard identification & risk estimation", "Analysis"),
    ("14971-5", "Risk acceptability evaluation", "Evaluation"),
    ("14971-6", "Risk control measures", "Control"),
    ("14971-7", "Residual risk & benefit-risk analysis", "Residual"),
    ("14971-9", "Post-production monitoring", "Post-Market"),
]


def create_chart(dark_mode=False):
    """Create regulatory compliance scorecard heatmap."""
    if dark_mode:
        text_color = "#e0e0e0"
        bg_color = "#1a1a2e"
        iec_color = "#1b5e20"
        fda_color = "#0d47a1"
        iso_color = "#e65100"
        header_bg = "#34495e"
        cell_text = "#ffffff"
    else:
        text_color = "#1a1a2e"
        bg_color = "#ffffff"
        iec_color = "#c8e6c9"
        fda_color = "#bbdefb"
        iso_color = "#ffe0b2"
        header_bg = "#2c3e50"
        cell_text = "#1a1a2e"

    # Combine all items with standard labels
    all_items = []
    for item_id, desc, category in IEC_62304:
        all_items.append(("IEC 62304", item_id, desc, category, iec_color))
    for item_id, desc, category in FDA_AIML:
        all_items.append(("FDA AI/ML PCCP", item_id, desc, category, fda_color))
    for item_id, desc, category in ISO_14971:
        all_items.append(("ISO 14971", item_id, desc, category, iso_color))

    standards = [item[0] for item in all_items]
    item_ids = [item[1] for item in all_items]
    descriptions = [item[2] for item in all_items]
    categories = [item[3] for item in all_items]
    colors = [item[4] for item in all_items]

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=["<b>Standard</b>", "<b>Item ID</b>", "<b>Description</b>", "<b>Category</b>"],
                    fill_color=header_bg,
                    font=dict(size=14, color="#ffffff"),
                    align="center",
                    line_color="#4a4a5a" if dark_mode else "#bdc3c7",
                    height=45,
                ),
                cells=dict(
                    values=[standards, item_ids, descriptions, categories],
                    fill_color=[colors, colors, colors, colors],
                    font=dict(size=12, color=cell_text),
                    align=["center", "center", "left", "center"],
                    line_color="#4a4a5a" if dark_mode else "#bdc3c7",
                    height=35,
                ),
                columnwidth=[0.18, 0.15, 0.47, 0.2],
            )
        ]
    )

    # Legend annotation
    fig.add_annotation(
        x=0.5,
        xref="paper",
        y=-0.06,
        yref="paper",
        text=(
            "<b>Color Key:</b>  "
            '<span style="background-color:#27ae60;color:white;padding:2px 8px">IEC 62304 (8 items)</span>  '
            '<span style="background-color:#2980b9;color:white;padding:2px 8px">FDA AI/ML PCCP (6 items)</span>  '
            '<span style="background-color:#e67e22;color:white;padding:2px 8px">ISO 14971 (5 items)</span>  '
            "| Total: 19 checklist items"
        ),
        showarrow=False,
        font=dict(size=11, color=text_color),
    )

    fig.update_layout(
        title=dict(
            text=(
                "Regulatory Compliance Scorecard: IEC 62304 + FDA AI/ML + ISO 14971"
                "<br><sup>Source: tools/deployment-readiness/deployment_readiness.py"
                " — 3 regulatory checklists (19 total items)</sup>"
            ),
            font=dict(size=18, color=text_color),
            x=0.5,
        ),
        paper_bgcolor=bg_color,
        margin=dict(t=120, l=30, r=30, b=100),
        width=1920,
        height=1080,
    )

    return fig


def main():
    """Generate regulatory compliance scorecard visualizations."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    png_dir = os.path.join(os.path.dirname(os.path.dirname(script_dir)), "png", "3rd")
    os.makedirs(png_dir, exist_ok=True)

    for dark_mode in [False, True]:
        suffix = "dark" if dark_mode else "light"
        fig = create_chart(dark_mode=dark_mode)

        html_path = os.path.join(script_dir, f"regulatory_compliance_scorecard_{suffix}.html")
        fig.write_html(html_path, include_plotlyjs=True)

        png_path = os.path.join(png_dir, f"regulatory_compliance_scorecard_{suffix}.png")
        fig.write_image(png_path, width=1920, height=1080, scale=2)

    print("regulatory_compliance_scorecard: done")


if __name__ == "__main__":
    main()

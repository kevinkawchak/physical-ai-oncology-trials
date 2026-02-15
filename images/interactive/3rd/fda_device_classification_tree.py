"""FDA AI/ML Device Classification Decision Flowchart.

Tree diagram showing the regulatory pathway decision logic from device
type to FDA submission pathway for oncology AI/ML devices.

Data sources:
  - regulatory-submit/classification_advisor.py — DEVICE_TYPE_BASELINE dict,
    _check_escalation_factors() method with escalation rules
"""

import os

import plotly.graph_objects as go

# Device type baseline classifications
DEVICE_TYPES = [
    ("CADe\n(Detection)", "II", "510(k)", "B"),
    ("CADx\n(Diagnosis)", "II", "De Novo", "B"),
    ("CADt\n(Triage)", "II", "De Novo", "C"),
    ("Treatment\nPlanning", "II", "De Novo", "C"),
    ("Prognostic", "II", "De Novo", "B"),
    ("Surgical\nGuidance", "II", "De Novo", "C"),
    ("Robotic\nControl", "II", "De Novo", "C"),
    ("Digital\nTwin", "II", "De Novo", "C"),
    ("Dose\nOptimization", "II", "510(k)", "C"),
]

ESCALATION_FACTORS = [
    ("Autonomous operation\n(no clinician)", "Class III → PMA"),
    ("Patient contact", "Add IEC 80601-2-77"),
    ("Life-threatening\ncondition", "Breakthrough Device eligible"),
    ("Novel algorithm\n(no predicate)", "De Novo (blocks 510(k))"),
]

PATHWAY_COLORS_LIGHT = {"510(k)": "#2980b9", "De Novo": "#27ae60", "PMA": "#c0392b", "Breakthrough": "#f39c12"}
PATHWAY_COLORS_DARK = {"510(k)": "#5dade2", "De Novo": "#58d68d", "PMA": "#ec7063", "Breakthrough": "#f9e79f"}


def create_chart(dark_mode=False):
    """Create FDA device classification decision tree."""
    if dark_mode:
        text_color = "#e0e0e0"
        bg_color = "#1a1a2e"
        line_color = "#7f8c8d"
        node_text = "#ffffff"
        root_bg = "#8e44ad"
        escalation_bg = "#7f8c8d"
        pathway_colors = PATHWAY_COLORS_DARK
    else:
        text_color = "#1a1a2e"
        bg_color = "#ffffff"
        line_color = "#95a5a6"
        node_text = "#1a1a2e"
        root_bg = "#9b59b6"
        escalation_bg = "#bdc3c7"
        pathway_colors = PATHWAY_COLORS_LIGHT

    fig = go.Figure()

    # Layout positions
    root_x, root_y = 0.5, 0.92

    # Level 2: 9 device types spread horizontally
    n_devices = len(DEVICE_TYPES)
    x_start, x_end = 0.05, 0.95
    device_xs = [x_start + i * (x_end - x_start) / (n_devices - 1) for i in range(n_devices)]
    device_y = 0.55

    # Level 3: 4 escalation factors at bottom
    n_esc = len(ESCALATION_FACTORS)
    esc_xs = [0.15 + i * 0.7 / (n_esc - 1) for i in range(n_esc)]
    esc_y = 0.12

    # Draw root node
    fig.add_shape(
        type="rect",
        x0=root_x - 0.1,
        y0=root_y - 0.03,
        x1=root_x + 0.1,
        y1=root_y + 0.03,
        fillcolor=root_bg,
        line=dict(color=root_bg, width=2),
        xref="paper",
        yref="paper",
    )
    fig.add_annotation(
        x=root_x,
        y=root_y,
        text="<b>AI/ML Oncology Device</b>",
        showarrow=False,
        font=dict(size=14, color="#ffffff"),
        xref="paper",
        yref="paper",
    )

    # Draw device type nodes (Level 2)
    for i, (name, cls, pathway, sw_risk) in enumerate(DEVICE_TYPES):
        x = device_xs[i]
        color = pathway_colors.get(pathway, "#95a5a6")

        # Connection line from root
        fig.add_shape(
            type="line",
            x0=root_x,
            y0=root_y - 0.03,
            x1=x,
            y1=device_y + 0.06,
            line=dict(color=line_color, width=1.5),
            xref="paper",
            yref="paper",
        )

        # Device node box
        fig.add_shape(
            type="rect",
            x0=x - 0.045,
            y0=device_y - 0.06,
            x1=x + 0.045,
            y1=device_y + 0.06,
            fillcolor=color,
            line=dict(color=color, width=1),
            xref="paper",
            yref="paper",
            opacity=0.85,
        )

        display_name = name.replace("\n", "<br>")
        fig.add_annotation(
            x=x,
            y=device_y + 0.02,
            text=f"<b>{display_name}</b>",
            showarrow=False,
            font=dict(size=9, color="#ffffff"),
            xref="paper",
            yref="paper",
        )
        fig.add_annotation(
            x=x,
            y=device_y - 0.035,
            text=f"Class {cls} | {pathway}<br>SW Risk: {sw_risk}",
            showarrow=False,
            font=dict(size=8, color="#ffffff"),
            xref="paper",
            yref="paper",
        )

    # Escalation header
    fig.add_annotation(
        x=0.5,
        y=0.27,
        text="<b>Escalation Factors (from _check_escalation_factors)</b>",
        showarrow=False,
        font=dict(size=13, color=text_color),
        xref="paper",
        yref="paper",
    )

    # Draw escalation factor nodes (Level 3)
    for i, (factor, result) in enumerate(ESCALATION_FACTORS):
        x = esc_xs[i]

        # Dashed lines from center device area down
        fig.add_shape(
            type="line",
            x0=x,
            y0=0.24,
            x1=x,
            y1=esc_y + 0.05,
            line=dict(color=line_color, width=1, dash="dash"),
            xref="paper",
            yref="paper",
        )

        fig.add_shape(
            type="rect",
            x0=x - 0.08,
            y0=esc_y - 0.05,
            x1=x + 0.08,
            y1=esc_y + 0.05,
            fillcolor=escalation_bg,
            line=dict(color=line_color, width=1),
            xref="paper",
            yref="paper",
            opacity=0.7,
        )

        display_factor = factor.replace("\n", "<br>")
        fig.add_annotation(
            x=x,
            y=esc_y + 0.02,
            text=f"<b>{display_factor}</b>",
            showarrow=False,
            font=dict(size=9, color=node_text),
            xref="paper",
            yref="paper",
        )
        fig.add_annotation(
            x=x,
            y=esc_y - 0.03,
            text=f"→ {result}",
            showarrow=False,
            font=dict(size=9, color=text_color),
            xref="paper",
            yref="paper",
        )

    # Legend for pathway colors
    legend_y = 0.42
    fig.add_annotation(
        x=0.02,
        y=legend_y,
        text="<b>Pathway Legend:</b>",
        showarrow=False,
        font=dict(size=11, color=text_color),
        xref="paper",
        yref="paper",
        xanchor="left",
    )
    for j, (pathway, color) in enumerate(pathway_colors.items()):
        fig.add_shape(
            type="rect",
            x0=0.02 + j * 0.12,
            y0=legend_y - 0.05,
            x1=0.04 + j * 0.12,
            y1=legend_y - 0.035,
            fillcolor=color,
            line=dict(width=0),
            xref="paper",
            yref="paper",
        )
        fig.add_annotation(
            x=0.045 + j * 0.12,
            y=legend_y - 0.043,
            text=pathway,
            showarrow=False,
            font=dict(size=10, color=text_color),
            xref="paper",
            yref="paper",
            xanchor="left",
        )

    fig.update_layout(
        title=dict(
            text=(
                "FDA AI/ML Device Classification Pathway Decision Tree for Oncology"
                "<br><sup>Source: regulatory-submit/classification_advisor.py"
                " — DEVICE_TYPE_BASELINE & escalation rules</sup>"
            ),
            font=dict(size=18, color=text_color),
            x=0.5,
        ),
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        xaxis=dict(visible=False, range=[0, 1]),
        yaxis=dict(visible=False, range=[0, 1]),
        margin=dict(t=120, l=30, r=30, b=60),
        width=1920,
        height=1080,
    )

    return fig


def main():
    """Generate FDA device classification tree visualizations."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    png_dir = os.path.join(os.path.dirname(os.path.dirname(script_dir)), "png", "3rd")
    os.makedirs(png_dir, exist_ok=True)

    for dark_mode in [False, True]:
        suffix = "dark" if dark_mode else "light"
        fig = create_chart(dark_mode=dark_mode)

        html_path = os.path.join(script_dir, f"fda_device_classification_tree_{suffix}.html")
        fig.write_html(html_path, include_plotlyjs=True)

        png_path = os.path.join(png_dir, f"fda_device_classification_tree_{suffix}.png")
        fig.write_image(png_path, width=1920, height=1080, scale=2)

    print("fda_device_classification_tree: done")


if __name__ == "__main__":
    main()

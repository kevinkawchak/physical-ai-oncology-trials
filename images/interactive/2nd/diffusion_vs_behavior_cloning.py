"""Diffusion Policy vs Behavior Cloning: ORBIT-Surgical Task Performance (dVRK).

Grouped bar chart with delta annotations showing diffusion policy superiority
over standard behavior cloning for surgical manipulation tasks.

Data source: generative-ai/results.md — ORBIT-Surgical Task Suite (dVRK Platform)
"""

import os

import plotly.graph_objects as go


TASKS = [
    "Peg Transfer",
    "Needle Pickup",
    "Needle Handover",
    "Tissue Retraction",
    "Suture Throw",
]
DIFFUSION = [94.1, 89.3, 85.7, 82.4, 78.2]
BC = [87.3, 72.1, 68.4, 69.8, 61.5]
IMPROVEMENT = [6.8, 17.2, 17.3, 12.6, 16.7]

DIFF_CONFIG = {
    "Denoising steps": "15",
    "Action chunk size": "16",
    "Observation horizon": "2",
    "Visual encoder": "ResNet-18",
    "Demonstrations": "200/task",
    "Training epochs": "500",
    "GPU": "RTX 4090",
}


def create_diffusion_chart(dark_mode=False):
    """Create grouped bar chart with delta annotations."""
    if dark_mode:
        template = "plotly_dark"
        diff_color = "#448aff"
        bc_color = "#616161"
        delta_color = "#76ff03"
        thresh_color = "#ef5350"
        text_color = "#e0e0e0"
        bg_color = "#1a1a2e"
        ann_color = "#ffd740"
    else:
        template = "plotly_white"
        diff_color = "#0d47a1"
        bc_color = "#bdbdbd"
        delta_color = "#1b5e20"
        thresh_color = "#d32f2f"
        text_color = "#1a1a2e"
        bg_color = "#ffffff"
        ann_color = "#e65100"

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            name="Diffusion Policy",
            x=TASKS,
            y=DIFFUSION,
            marker_color=diff_color,
            text=[f"{v}%" for v in DIFFUSION],
            textposition="outside",
            textfont=dict(size=12),
        )
    )

    fig.add_trace(
        go.Bar(
            name="Behavior Cloning",
            x=TASKS,
            y=BC,
            marker_color=bc_color,
            text=[f"{v}%" for v in BC],
            textposition="outside",
            textfont=dict(size=12),
        )
    )

    # Delta annotations above each pair
    for i in range(len(TASKS)):
        higher = max(DIFFUSION[i], BC[i])
        fig.add_annotation(
            x=TASKS[i],
            y=higher + 4,
            text=f"<b>+{IMPROVEMENT[i]}%</b>",
            showarrow=False,
            font=dict(size=14, color=delta_color),
        )

    # Clinical feasibility threshold at 80%
    fig.add_hline(
        y=80,
        line_dash="dash",
        line_color=thresh_color,
        line_width=2,
        annotation_text="Clinical Feasibility (80%)",
        annotation_position="top left",
        annotation_font=dict(size=12, color=thresh_color),
    )

    # Diffusion config annotation
    config_text = "<br>".join([f"{k}: {v}" for k, v in DIFF_CONFIG.items()])
    fig.add_annotation(
        x=0.98,
        xref="paper",
        y=0.95,
        yref="paper",
        text=f"<b>Diffusion Policy Config:</b><br>{config_text}",
        showarrow=False,
        font=dict(size=10, color=ann_color),
        bgcolor="rgba(0,0,0,0.7)" if dark_mode else "rgba(255,255,255,0.9)",
        bordercolor=ann_color,
        borderwidth=1,
        align="left",
        xanchor="right",
    )

    fig.update_layout(
        title=dict(
            text=(
                "Diffusion Policy vs Behavior Cloning:"
                " ORBIT-Surgical Task Performance (dVRK)"
                "<br><sup>Source: generative-ai/results.md —"
                " ORBIT-Surgical Task Suite</sup>"
            ),
            font=dict(size=18, color=text_color),
            x=0.5,
        ),
        template=template,
        paper_bgcolor=bg_color,
        barmode="group",
        xaxis=dict(title="Surgical Task", tickfont=dict(size=12)),
        yaxis=dict(
            title="Success Rate (%)",
            range=[50, 105],
            tickfont=dict(size=12),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=13),
        ),
        margin=dict(t=120, l=60, r=200, b=80),
        width=1920,
        height=1080,
    )

    return fig


def main():
    """Generate diffusion vs behavior cloning visualizations."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    png_dir = os.path.join(os.path.dirname(os.path.dirname(script_dir)), "png", "2nd")
    os.makedirs(png_dir, exist_ok=True)

    for dark_mode in [False, True]:
        suffix = "dark" if dark_mode else "light"
        fig = create_diffusion_chart(dark_mode=dark_mode)

        html_path = os.path.join(script_dir, f"diffusion_vs_behavior_cloning_{suffix}.html")
        fig.write_html(html_path, include_plotlyjs=True)

        png_path = os.path.join(png_dir, f"diffusion_vs_behavior_cloning_{suffix}.png")
        fig.write_image(png_path, width=1920, height=1080, scale=2)

    print("diffusion_vs_behavior_cloning: done")


if __name__ == "__main__":
    main()

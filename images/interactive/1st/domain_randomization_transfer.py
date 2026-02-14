"""Domain Randomization Impact on Sim-to-Real Transfer.

Quantifies how each type of domain randomization affects sim-to-real
transfer for needle insertion tasks in oncology surgical robotics.

Data sources:
  - reinforcement-learning/results.md — Domain Randomization Ablation
  - configs/training_config.yaml — domain_randomization parameters
  - generative-ai/results.md — Oncology-Specific Domain Randomization
"""

import os

import plotly.graph_objects as go
from plotly.subplots import make_subplots


LEVELS = ["None", "Visual Only", "Physics Only", "Visual +\nPhysics", "Full (+ action\ndelay)"]
SIM_SUCCESS = [95, 94, 93, 92, 91]
REAL_SUCCESS = [52, 68, 65, 78, 82]
TRANSFER_RATIO = [55, 72, 70, 85, 90]

MIN_TRANSFER_RATE = 80  # From configs/training_config.yaml


def create_domain_rand_chart(dark_mode=False):
    """Create grouped bar + line chart for domain randomization ablation."""
    if dark_mode:
        template = "plotly_dark"
        sim_color = "#4dd0e1"
        real_color = "#039be5"
        line_color = "#ff9800"
        thresh_color = "#ef5350"
        text_color = "#e0e0e0"
        bg_color = "#1a1a2e"
    else:
        template = "plotly_white"
        sim_color = "#90caf9"
        real_color = "#0d47a1"
        line_color = "#d32f2f"
        thresh_color = "#d32f2f"
        text_color = "#1a1a2e"
        bg_color = "#ffffff"

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Sim success bars
    fig.add_trace(
        go.Bar(
            name="Sim Success (%)",
            x=LEVELS,
            y=SIM_SUCCESS,
            marker_color=sim_color,
            offsetgroup=0,
        ),
        secondary_y=False,
    )

    # Real success bars
    fig.add_trace(
        go.Bar(
            name="Real Success (%)",
            x=LEVELS,
            y=REAL_SUCCESS,
            marker_color=real_color,
            offsetgroup=1,
        ),
        secondary_y=False,
    )

    # Transfer ratio line on secondary y-axis
    fig.add_trace(
        go.Scatter(
            name="Transfer Ratio (%)",
            x=LEVELS,
            y=TRANSFER_RATIO,
            mode="lines+markers",
            line=dict(color=line_color, width=3),
            marker=dict(size=10, symbol="diamond"),
        ),
        secondary_y=True,
    )

    # Min transfer rate threshold line
    fig.add_hline(
        y=MIN_TRANSFER_RATE,
        line_dash="dash",
        line_color=thresh_color,
        line_width=2,
        annotation_text="min_transfer_rate = 80%",
        annotation_position="top left",
        annotation_font=dict(size=12, color=thresh_color),
        secondary_y=True,
    )

    fig.update_layout(
        title=dict(
            text=(
                "Domain Randomization Ablation: Needle Insertion Sim-to-Real Transfer"
                "<br><sup>Source: reinforcement-learning/results.md, "
                "configs/training_config.yaml</sup>"
            ),
            font=dict(size=18, color=text_color),
            x=0.5,
        ),
        template=template,
        paper_bgcolor=bg_color,
        barmode="group",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=13),
        ),
        margin=dict(t=120, l=60, r=60, b=80),
        width=1920,
        height=1080,
    )

    fig.update_yaxes(
        title_text="Success Rate (%)",
        range=[0, 100],
        secondary_y=False,
    )
    fig.update_yaxes(
        title_text="Transfer Ratio (%)",
        range=[0, 100],
        secondary_y=True,
    )
    fig.update_xaxes(title_text="Randomization Level")

    return fig


def main():
    """Generate domain randomization visualizations."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    png_dir = os.path.join(os.path.dirname(os.path.dirname(script_dir)), "png", "1st")
    os.makedirs(png_dir, exist_ok=True)

    for dark_mode in [False, True]:
        suffix = "dark" if dark_mode else "light"
        fig = create_domain_rand_chart(dark_mode=dark_mode)

        html_path = os.path.join(script_dir, f"domain_randomization_transfer_{suffix}.html")
        fig.write_html(html_path, include_plotlyjs=True)

        png_path = os.path.join(png_dir, f"domain_randomization_transfer_{suffix}.png")
        fig.write_image(png_path, width=1920, height=1080, scale=2)

    print("domain_randomization_transfer: done")


if __name__ == "__main__":
    main()

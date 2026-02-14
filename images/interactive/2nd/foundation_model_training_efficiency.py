"""Foundation Model Training Efficiency: Synthetic vs Human Data Collection.

Bubble chart visualizing data efficiency gains from generative AI synthetic
data pipelines compared to manual human data collection.

Data sources:
  - generative-ai/results.md — Training Efficiency, FF-SRL, Surgical Video Synthesis
  - generative-ai/strengths.md — Foundation model overview
"""

import os
import math

import plotly.graph_objects as go


METHODS = ["Human Demonstration", "Isaac Lab Synthetic (GPU)", "FF-SRL (GPU, tissue)"]
TIME_HOURS = [6500, 11, 0.2]
TRAJECTORIES_K = [780, 780, 50]
EFFICIENCY = ["1x (baseline)", "35x", "40x speedup"]
BUBBLE_SIZE = [20, 60, 40]


def create_efficiency_chart(dark_mode=False):
    """Create bubble chart for training efficiency comparison."""
    if dark_mode:
        template = "plotly_dark"
        colors = ["#616161", "#00e5ff", "#76ff03"]
        text_color = "#e0e0e0"
        bg_color = "#1a1a2e"
        ann_color = "#ffd740"
        arrow_color = "#b0bec5"
    else:
        template = "plotly_white"
        colors = ["#9e9e9e", "#1565c0", "#2e7d32"]
        text_color = "#1a1a2e"
        bg_color = "#ffffff"
        ann_color = "#e65100"
        arrow_color = "#616161"

    fig = go.Figure()

    for i, (method, time, traj, eff, size) in enumerate(
        zip(METHODS, TIME_HOURS, TRAJECTORIES_K, EFFICIENCY, BUBBLE_SIZE)
    ):
        fig.add_trace(
            go.Scatter(
                name=f"{method} ({eff})",
                x=[time],
                y=[traj],
                mode="markers+text",
                marker=dict(
                    size=size,
                    color=colors[i],
                    opacity=0.8,
                    line=dict(width=2, color="white"),
                ),
                text=[method],
                textposition="top center",
                textfont=dict(size=12, color=text_color),
            )
        )

    # Arrow from Human to Isaac Lab showing 35x efficiency
    fig.add_annotation(
        x=math.log10(11),
        y=780,
        ax=math.log10(6500),
        ay=780,
        xref="x",
        yref="y",
        axref="x",
        ayref="y",
        showarrow=True,
        arrowhead=3,
        arrowsize=1.5,
        arrowwidth=2,
        arrowcolor=arrow_color,
    )
    fig.add_annotation(
        x=math.log10(100),
        y=810,
        text="<b>35x faster, same output</b>",
        showarrow=False,
        font=dict(size=13, color=ann_color),
        xref="x",
        yref="y",
    )

    # Model annotations
    model_notes = [
        "NVIDIA GR00T N1.6: 94.2% novel object grasping",
        "Cosmos Predict 2.5: FVD 142 (24% better than prior)",
    ]
    for j, note in enumerate(model_notes):
        fig.add_annotation(
            x=0.02,
            xref="paper",
            y=0.15 - j * 0.08,
            yref="paper",
            text=f"<b>{note}</b>",
            showarrow=False,
            font=dict(size=12, color=ann_color),
            bgcolor="rgba(0,0,0,0.6)" if dark_mode else "rgba(255,255,255,0.9)",
            bordercolor=ann_color,
            borderwidth=1,
            xanchor="left",
        )

    # Surgical video quality annotation
    fig.add_annotation(
        x=0.98,
        xref="paper",
        y=0.95,
        yref="paper",
        text=(
            "<b>Surgical Video Synthesis Quality:</b><br>"
            "Anatomical accuracy: 89%<br>"
            "Instrument realism: 92%<br>"
            "Tissue deformation plausibility: 84%"
        ),
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
                "Foundation Model Training Efficiency:"
                " Synthetic vs Human Data Collection"
                "<br><sup>Sources: generative-ai/results.md — Training Efficiency,"
                " generative-ai/strengths.md</sup>"
            ),
            font=dict(size=18, color=text_color),
            x=0.5,
        ),
        template=template,
        paper_bgcolor=bg_color,
        xaxis=dict(
            title="Data Collection Time (hours, log scale)",
            type="log",
            tickfont=dict(size=12),
            range=[-1, 4],
        ),
        yaxis=dict(
            title="Trajectories Generated (thousands)",
            tickfont=dict(size=12),
            range=[0, 900],
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=12),
        ),
        margin=dict(t=120, l=60, r=200, b=80),
        width=1920,
        height=1080,
    )

    return fig


def main():
    """Generate foundation model training efficiency visualizations."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    png_dir = os.path.join(os.path.dirname(os.path.dirname(script_dir)), "png", "2nd")
    os.makedirs(png_dir, exist_ok=True)

    for dark_mode in [False, True]:
        suffix = "dark" if dark_mode else "light"
        fig = create_efficiency_chart(dark_mode=dark_mode)

        html_path = os.path.join(script_dir, f"foundation_model_training_efficiency_{suffix}.html")
        fig.write_html(html_path, include_plotlyjs=True)

        png_path = os.path.join(png_dir, f"foundation_model_training_efficiency_{suffix}.png")
        fig.write_image(png_path, width=1920, height=1080, scale=2)

    print("foundation_model_training_efficiency: done")


if __name__ == "__main__":
    main()

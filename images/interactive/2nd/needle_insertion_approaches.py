"""Needle Insertion: Success Rate by Control Approach (Hand-Coded to VLA).

Horizontal bar chart with adaptability color coding tracing the evolution
of control approaches for the needle insertion task.

Data source: reinforcement-learning/results.md — Needle Insertion Approach Comparison
"""

import os

import plotly.graph_objects as go


APPROACHES = [
    "Hand-Coded",
    "Model Predictive\nControl (MPC)",
    "Behavior Cloning",
    "RL (PPO)",
    "Diffusion Policy",
    "Vision-Language-\nAction (VLA)",
]
SUCCESS = [65, 72, 78, 86, 89, 92]
ADAPTABILITY = ["None", "Low", "Low", "High", "Medium", "Very High"]


def _adapt_color(adapt, dark_mode=False):
    """Return color based on adaptability level."""
    mapping_light = {
        "None": "#9e9e9e",
        "Low": "#f9a825",
        "Medium": "#e65100",
        "High": "#1565c0",
        "Very High": "#2e7d32",
    }
    mapping_dark = {
        "None": "#757575",
        "Low": "#fdd835",
        "Medium": "#ff9800",
        "High": "#42a5f5",
        "Very High": "#66bb6a",
    }
    return mapping_dark[adapt] if dark_mode else mapping_light[adapt]


def create_needle_chart(dark_mode=False):
    """Create horizontal bar chart with adaptability color coding."""
    if dark_mode:
        template = "plotly_dark"
        text_color = "#e0e0e0"
        bg_color = "#1a1a2e"
        ann_color = "#ffd740"
    else:
        template = "plotly_white"
        text_color = "#1a1a2e"
        bg_color = "#ffffff"
        ann_color = "#e65100"

    colors = [_adapt_color(a, dark_mode) for a in ADAPTABILITY]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            y=APPROACHES,
            x=SUCCESS,
            orientation="h",
            marker_color=colors,
            text=[f"{s}% ({a})" for s, a in zip(SUCCESS, ADAPTABILITY)],
            textposition="outside",
            textfont=dict(size=12),
        )
    )

    # Annotations for key references
    annotations_data = [
        ("RL (PPO)", "PPO: 2M samples, 48 GPU hrs"),
        ("Vision-Language-\nAction (VLA)", "VLA: language-conditioned, zero-shot transfer"),
    ]
    for approach, note in annotations_data:
        idx = APPROACHES.index(approach)
        fig.add_annotation(
            x=SUCCESS[idx] - 5,
            y=approach,
            text=f"<i>{note}</i>",
            showarrow=False,
            font=dict(size=10, color="white" if dark_mode else "#424242"),
            xanchor="right",
        )

    # Adaptability legend
    adapt_levels = ["None", "Low", "Medium", "High", "Very High"]
    legend_text = "<br>".join([f"<span style='color:{_adapt_color(a, dark_mode)}'>■</span> {a}" for a in adapt_levels])
    fig.add_annotation(
        x=0.98,
        xref="paper",
        y=0.15,
        yref="paper",
        text=f"<b>Adaptability:</b><br>{legend_text}",
        showarrow=False,
        font=dict(size=11, color=text_color),
        bgcolor="rgba(0,0,0,0.7)" if dark_mode else "rgba(255,255,255,0.9)",
        bordercolor=ann_color,
        borderwidth=1,
        align="left",
        xanchor="right",
    )

    fig.update_layout(
        title=dict(
            text=(
                "Needle Insertion: Success Rate by Control Approach"
                " (Hand-Coded → VLA)"
                "<br><sup>Source: reinforcement-learning/results.md —"
                " Needle Insertion Approach Comparison</sup>"
            ),
            font=dict(size=18, color=text_color),
            x=0.5,
        ),
        template=template,
        paper_bgcolor=bg_color,
        xaxis=dict(
            title="Success Rate (%)",
            range=[0, 105],
            tickfont=dict(size=12),
        ),
        yaxis=dict(
            title="Control Approach",
            tickfont=dict(size=12),
        ),
        showlegend=False,
        margin=dict(t=120, l=160, r=180, b=80),
        width=1920,
        height=1080,
    )

    return fig


def main():
    """Generate needle insertion approach comparison visualizations."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    png_dir = os.path.join(os.path.dirname(os.path.dirname(script_dir)), "png", "2nd")
    os.makedirs(png_dir, exist_ok=True)

    for dark_mode in [False, True]:
        suffix = "dark" if dark_mode else "light"
        fig = create_needle_chart(dark_mode=dark_mode)

        html_path = os.path.join(script_dir, f"needle_insertion_approaches_{suffix}.html")
        fig.write_html(html_path, include_plotlyjs=True)

        png_path = os.path.join(png_dir, f"needle_insertion_approaches_{suffix}.png")
        fig.write_image(png_path, width=1920, height=1080, scale=2)

    print("needle_insertion_approaches: done")


if __name__ == "__main__":
    main()

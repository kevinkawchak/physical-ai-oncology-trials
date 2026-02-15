"""Production Readiness by Surgical Task Category.

Horizontal bar chart with deployment threshold lines showing maturity
level of each surgical task category across RL, SL, and SSL approaches.

Data sources:
  - reinforcement-learning/results.md — Production Readiness Assessment
  - supervised-learning/results.md — Production Readiness
  - self-supervised-learning/results.md — Production Readiness
"""

import os

import plotly.graph_objects as go

TASKS = [
    ("Basic Manipulation (RL)", 98, "Deploy Ready", "reinforcement-learning"),
    ("Anatomical Segmentation (SL)", 97.2, "Deploy Ready", "supervised-learning"),
    ("Depth Estimation (SL)", 96.8, "Deploy Ready", "supervised-learning"),
    ("Instrument Detection (SSL)", 96, "Deploy Ready", "self-supervised-learning"),
    ("Instrument Detection (SL)", 94.8, "Deploy Ready", "supervised-learning"),
    ("Phase Recognition (SL)", 94.2, "Deploy Ready", "supervised-learning"),
    ("Pose Estimation (SL)", 94, "Deploy Ready", "supervised-learning"),
    ("Precision Grasping (RL)", 91, "Supervised Use", "reinforcement-learning"),
    ("Skill Assessment (SL)", 91.2, "Supervised Use", "supervised-learning"),
    ("Tissue Segmentation (SSL)", 89, "Supervised Use", "self-supervised-learning"),
    ("Action Recognition (SSL)", 85, "Limited/Research", "self-supervised-learning"),
    ("Tissue Interaction (RL)", 84, "Limited/Research", "reinforcement-learning"),
    ("Suturing (RL)", 78, "Research Only", "reinforcement-learning"),
    ("Autonomous Control (SSL)", 78, "Research Only", "self-supervised-learning"),
    ("Complex Procedures (RL)", 65, "Research Only", "reinforcement-learning"),
]

# Sorted by readiness descending for display
TASKS_SORTED = sorted(TASKS, key=lambda x: x[1])


def create_chart(dark_mode=False):
    """Create horizontal bar chart with threshold lines."""
    if dark_mode:
        template = "plotly_dark"
        deploy_color = "#27ae60"
        supervised_color = "#f39c12"
        research_color = "#e74c3c"
        text_color = "#e0e0e0"
        bg_color = "#1a1a2e"
        threshold_color = "#ffffff"
        ann_bg = "rgba(0,0,0,0.7)"
    else:
        template = "plotly_white"
        deploy_color = "#2ecc71"
        supervised_color = "#f1c40f"
        research_color = "#e74c3c"
        text_color = "#1a1a2e"
        bg_color = "#ffffff"
        threshold_color = "#000000"
        ann_bg = "rgba(255,255,255,0.9)"

    fig = go.Figure()

    names = [t[0] for t in TASKS_SORTED]
    scores = [t[1] for t in TASKS_SORTED]
    statuses = [t[2] for t in TASKS_SORTED]

    bar_colors = []
    for score in scores:
        if score >= 90:
            bar_colors.append(deploy_color)
        elif score >= 80:
            bar_colors.append(supervised_color)
        else:
            bar_colors.append(research_color)

    fig.add_trace(
        go.Bar(
            name="Readiness Score",
            y=names,
            x=scores,
            orientation="h",
            marker_color=bar_colors,
            text=[f"{s}%" for s in scores],
            textposition="outside",
            textfont=dict(size=11, color=text_color),
        )
    )

    # Deployment threshold line at 90%
    fig.add_vline(
        x=90,
        line_dash="dash",
        line_color=deploy_color,
        line_width=2,
        annotation_text="Deployment Threshold (90%)",
        annotation_position="top",
        annotation_font=dict(size=12, color=deploy_color),
    )

    # Supervised use threshold at 80%
    fig.add_vline(
        x=80,
        line_dash="dash",
        line_color=supervised_color,
        line_width=2,
        annotation_text="Supervised Use Threshold (80%)",
        annotation_position="bottom left",
        annotation_font=dict(size=12, color=supervised_color),
    )

    # Legend annotation for status categories
    fig.add_annotation(
        x=0.98,
        xref="paper",
        y=0.02,
        yref="paper",
        text=(
            f"<b>Status Key:</b><br>"
            f'<span style="color:{deploy_color}">■ Deploy Ready (≥90%)</span><br>'
            f'<span style="color:{supervised_color}">■ Supervised Use (80–89%)</span><br>'
            f'<span style="color:{research_color}">■ Research Only (&lt;80%)</span>'
        ),
        showarrow=False,
        font=dict(size=11),
        bgcolor=ann_bg,
        bordercolor=text_color,
        borderwidth=1,
        align="left",
        xanchor="right",
        yanchor="bottom",
    )

    fig.update_layout(
        title=dict(
            text=(
                "Surgical Task Production Readiness: From Research to Deployment"
                "<br><sup>Sources: reinforcement-learning/results.md;"
                " supervised-learning/results.md;"
                " self-supervised-learning/results.md</sup>"
            ),
            font=dict(size=18, color=text_color),
            x=0.5,
        ),
        template=template,
        paper_bgcolor=bg_color,
        xaxis=dict(
            title="Success Rate / Readiness Score (%)",
            range=[55, 105],
            tickfont=dict(size=12),
        ),
        yaxis=dict(
            title="Task Category",
            tickfont=dict(size=11),
        ),
        margin=dict(t=120, l=250, r=60, b=80),
        width=1920,
        height=1080,
        showlegend=False,
    )

    return fig


def main():
    """Generate production readiness task visualizations."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    png_dir = os.path.join(os.path.dirname(os.path.dirname(script_dir)), "png", "3rd")
    os.makedirs(png_dir, exist_ok=True)

    for dark_mode in [False, True]:
        suffix = "dark" if dark_mode else "light"
        fig = create_chart(dark_mode=dark_mode)

        html_path = os.path.join(script_dir, f"production_readiness_tasks_{suffix}.html")
        fig.write_html(html_path, include_plotlyjs=True)

        png_path = os.path.join(png_dir, f"production_readiness_tasks_{suffix}.png")
        fig.write_image(png_path, width=1920, height=1080, scale=2)

    print("production_readiness_tasks: done")


if __name__ == "__main__":
    main()

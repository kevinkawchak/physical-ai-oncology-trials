"""Safety-Constrained RL: Task Success vs Safety Violation Tradeoff.

Scatter plot with Pareto frontier showing the tradeoff between task
performance and safety violations for clinical deployment.

Data sources:
  - reinforcement-learning/results.md — Constrained Policy Optimization,
    Performance vs Safety Tradeoff, Error Recovery
  - configs/training_config.yaml — safety parameters
"""

import os

import plotly.graph_objects as go


ALGORITHMS = ["PPO (Unconstrained)", "PPO + Penalty", "CPO", "Safe Layer"]
SUCCESS = [89, 84, 82, 78]
VIOLATIONS = [12.0, 5.0, 1.5, 0.2]

ERROR_RECOVERY = {
    "Excessive force": "98% detect / 92% recover / 0.8s",
    "Workspace violation": "99% detect / 95% recover / 0.5s",
    "Collision prediction": "94% detect / 88% recover / 0.3s",
}

SAFETY_CONFIG = {
    "max_force": "5.0 N",
    "max_velocity": "0.1 m/s",
    "critical_margin": "5.0 mm",
}


def create_safety_chart(dark_mode=False):
    """Create scatter plot with Pareto frontier and deployment zones."""
    if dark_mode:
        template = "plotly_dark"
        point_color = "#00e5ff"
        pareto_color = "#b0bec5"
        danger_color = "rgba(244,67,54,0.15)"
        safe_color = "rgba(76,175,80,0.15)"
        thresh_color = "#ef5350"
        text_color = "#e0e0e0"
        bg_color = "#1a1a2e"
        ann_color = "#ffd740"
        label_color = "#ffffff"
    else:
        template = "plotly_white"
        point_color = "#1565c0"
        pareto_color = "#616161"
        danger_color = "rgba(244,67,54,0.1)"
        safe_color = "rgba(76,175,80,0.1)"
        thresh_color = "#d32f2f"
        text_color = "#1a1a2e"
        bg_color = "#ffffff"
        ann_color = "#e65100"
        label_color = "#1a1a2e"

    fig = go.Figure()

    # Deployment zone (green shaded region: low violations, high success)
    fig.add_shape(
        type="rect",
        x0=0,
        x1=1.0,
        y0=75,
        y1=100,
        fillcolor=safe_color,
        line=dict(width=0),
        layer="below",
    )
    fig.add_annotation(
        x=0.5,
        y=98,
        text="<b>Deployment Zone</b>",
        showarrow=False,
        font=dict(size=14, color="#4caf50"),
    )

    # Pareto frontier line
    fig.add_trace(
        go.Scatter(
            name="Pareto Frontier",
            x=VIOLATIONS,
            y=SUCCESS,
            mode="lines",
            line=dict(color=pareto_color, width=2, dash="dash"),
            showlegend=True,
        )
    )

    # Algorithm points
    fig.add_trace(
        go.Scatter(
            name="Algorithms",
            x=VIOLATIONS,
            y=SUCCESS,
            mode="markers+text",
            marker=dict(size=20, color=point_color, line=dict(width=2, color="white")),
            text=ALGORITHMS,
            textposition=["top right", "top right", "bottom left", "top right"],
            textfont=dict(size=12, color=label_color),
            showlegend=False,
        )
    )

    # Clinical safety threshold at 1% violations
    fig.add_vline(
        x=1.0,
        line_dash="dash",
        line_color=thresh_color,
        line_width=2,
        annotation_text="Clinical Safety Threshold (1%)",
        annotation_position="top",
        annotation_font=dict(size=12, color=thresh_color),
    )

    # Error recovery annotation
    recovery_text = "<br>".join([f"{k}: {v}" for k, v in ERROR_RECOVERY.items()])
    fig.add_annotation(
        x=0.98,
        xref="paper",
        y=0.25,
        yref="paper",
        text=f"<b>Error Recovery Rates:</b><br>{recovery_text}",
        showarrow=False,
        font=dict(size=10, color=ann_color),
        bgcolor="rgba(0,0,0,0.7)" if dark_mode else "rgba(255,255,255,0.9)",
        bordercolor=ann_color,
        borderwidth=1,
        align="left",
        xanchor="right",
    )

    # Safety config annotation
    config_text = "<br>".join([f"{k}: {v}" for k, v in SAFETY_CONFIG.items()])
    fig.add_annotation(
        x=0.02,
        xref="paper",
        y=0.05,
        yref="paper",
        text=f"<b>Safety Config:</b><br>{config_text}",
        showarrow=False,
        font=dict(size=10, color=ann_color),
        bgcolor="rgba(0,0,0,0.7)" if dark_mode else "rgba(255,255,255,0.9)",
        bordercolor=ann_color,
        borderwidth=1,
        align="left",
        xanchor="left",
    )

    fig.update_layout(
        title=dict(
            text=(
                "Safety-Constrained RL: Task Success vs Safety Violation Tradeoff"
                "<br><sup>Sources: reinforcement-learning/results.md,"
                " configs/training_config.yaml</sup>"
            ),
            font=dict(size=18, color=text_color),
            x=0.5,
        ),
        template=template,
        paper_bgcolor=bg_color,
        xaxis=dict(
            title="Safety Violation Rate (%)",
            range=[-0.5, 14],
            tickfont=dict(size=12),
        ),
        yaxis=dict(
            title="Task Success Rate (%)",
            range=[74, 100],
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
    """Generate safety-constrained RL visualizations."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    png_dir = os.path.join(os.path.dirname(os.path.dirname(script_dir)), "png", "2nd")
    os.makedirs(png_dir, exist_ok=True)

    for dark_mode in [False, True]:
        suffix = "dark" if dark_mode else "light"
        fig = create_safety_chart(dark_mode=dark_mode)

        html_path = os.path.join(script_dir, f"safety_constrained_rl_{suffix}.html")
        fig.write_html(html_path, include_plotlyjs=True)

        png_path = os.path.join(png_dir, f"safety_constrained_rl_{suffix}.png")
        fig.write_image(png_path, width=1920, height=1080, scale=2)

    print("safety_constrained_rl: done")


if __name__ == "__main__":
    main()

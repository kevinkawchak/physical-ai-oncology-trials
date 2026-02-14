"""ORBIT-Surgical Benchmark: PPO vs SAC Success Rates Across 14 Surgical Tasks.

Horizontal grouped bar chart showing complete benchmark landscape of RL
algorithm performance on the dVRK surgical robot platform.

Data source: reinforcement-learning/results.md — ORBIT-Surgical Benchmark Suite
"""

import os

import plotly.graph_objects as go


# Ordered by PPO success rate descending
TASKS = [
    "Reach",
    "Lift",
    "Peg Transfer",
    "Needle Pickup",
    "Needle Handover",
    "Gauze Cutting",
    "Tissue Retraction",
    "Tissue Manipulation",
    "Thread Through Rings",
    "Suture Throw",
    "Tissue Cutting",
    "Needle Driving",
    "Debridement",
    "Resection",
]
PPO = [98.2, 95.7, 94.1, 89.3, 85.7, 85.3, 82.4, 79.8, 78.4, 78.2, 76.2, 72.5, 68.4, 62.1]
SAC = [97.8, 96.3, 93.8, 91.2, 87.4, 86.7, 84.1, 82.3, 81.2, 76.9, 78.4, 75.8, 71.2, 65.8]


def create_orbit_chart(dark_mode=False):
    """Create horizontal grouped bar chart for ORBIT-Surgical benchmark."""
    if dark_mode:
        template = "plotly_dark"
        ppo_color = "#00e5ff"
        sac_color = "#ffab40"
        thresh80_color = "#ef5350"
        thresh90_color = "#66bb6a"
        text_color = "#e0e0e0"
        bg_color = "#1a1a2e"
    else:
        template = "plotly_white"
        ppo_color = "#1565c0"
        sac_color = "#e65100"
        thresh80_color = "#d32f2f"
        thresh90_color = "#2e7d32"
        text_color = "#1a1a2e"
        bg_color = "#ffffff"

    # Reverse for horizontal display (bottom to top)
    tasks_rev = list(reversed(TASKS))
    ppo_rev = list(reversed(PPO))
    sac_rev = list(reversed(SAC))

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            name="PPO",
            y=tasks_rev,
            x=ppo_rev,
            orientation="h",
            marker_color=ppo_color,
            text=[f"{v}%" for v in ppo_rev],
            textposition="outside",
            textfont=dict(size=10),
        )
    )

    fig.add_trace(
        go.Bar(
            name="SAC",
            y=tasks_rev,
            x=sac_rev,
            orientation="h",
            marker_color=sac_color,
            text=[f"{v}%" for v in sac_rev],
            textposition="outside",
            textfont=dict(size=10),
        )
    )

    # Clinical feasibility threshold at 80%
    fig.add_vline(
        x=80,
        line_dash="dash",
        line_color=thresh80_color,
        line_width=2,
        annotation_text="Clinical Feasibility (80%)",
        annotation_position="top",
        annotation_font=dict(size=11, color=thresh80_color),
    )

    # Deployment ready threshold at 90%
    fig.add_vline(
        x=90,
        line_dash="dash",
        line_color=thresh90_color,
        line_width=2,
        annotation_text="Deployment Ready (90%)",
        annotation_position="top",
        annotation_font=dict(size=11, color=thresh90_color),
    )

    fig.update_layout(
        title=dict(
            text=(
                "ORBIT-Surgical Benchmark: PPO vs SAC Success Rates"
                " Across 14 Surgical Tasks"
                "<br><sup>Source: reinforcement-learning/results.md —"
                " dVRK Platform</sup>"
            ),
            font=dict(size=18, color=text_color),
            x=0.5,
        ),
        template=template,
        paper_bgcolor=bg_color,
        barmode="group",
        xaxis=dict(
            title="Success Rate (%)",
            range=[0, 108],
            tickfont=dict(size=12),
        ),
        yaxis=dict(
            title="Surgical Task",
            tickfont=dict(size=11),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=13),
        ),
        margin=dict(t=120, l=160, r=60, b=80),
        width=1920,
        height=1080,
    )

    return fig


def main():
    """Generate ORBIT-Surgical benchmark visualizations."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    png_dir = os.path.join(os.path.dirname(os.path.dirname(script_dir)), "png", "2nd")
    os.makedirs(png_dir, exist_ok=True)

    for dark_mode in [False, True]:
        suffix = "dark" if dark_mode else "light"
        fig = create_orbit_chart(dark_mode=dark_mode)

        html_path = os.path.join(script_dir, f"orbit_surgical_benchmark_{suffix}.html")
        fig.write_html(html_path, include_plotlyjs=True)

        png_path = os.path.join(png_dir, f"orbit_surgical_benchmark_{suffix}.png")
        fig.write_image(png_path, width=1920, height=1080, scale=2)

    print("orbit_surgical_benchmark: done")


if __name__ == "__main__":
    main()

"""Training Efficiency: GPU Hours to Reach 80% Success Rate by Algorithm.

Grouped bar chart with log-scale y-axis comparing compute requirements
for PPO, SAC, and DreamerV3 across surgical tasks.

Data source: reinforcement-learning/results.md — Training Efficiency Comparison
"""

import os

import plotly.graph_objects as go


TASKS = ["Reach", "Needle\nPickup", "Tissue\nRetraction", "Suture\nThrow", "Debridement"]
PPO_HOURS = [0.5, 8, 24, 48, 96]
SAC_HOURS = [0.3, 4, 12, 32, 64]
DREAMER_HOURS = [0.2, 1.5, 4, 12, 24]

DREAMER_ABLATION = {
    "Full DreamerV3": "50K samples (baseline)",
    "Without imagination": "150K samples (-67%)",
    "Without representation": "200K samples (-75%)",
}


def create_efficiency_chart(dark_mode=False):
    """Create grouped bar chart with log-scale y-axis."""
    if dark_mode:
        template = "plotly_dark"
        ppo_color = "#6495ed"
        sac_color = "#fa8072"
        dreamer_color = "#00ff7f"
        text_color = "#e0e0e0"
        bg_color = "#1a1a2e"
        ann_color = "#ffd740"
        speedup_color = "#76ff03"
    else:
        template = "plotly_white"
        ppo_color = "#6a5acd"
        sac_color = "#ff6347"
        dreamer_color = "#228b22"
        text_color = "#1a1a2e"
        bg_color = "#ffffff"
        ann_color = "#e65100"
        speedup_color = "#1b5e20"

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            name="PPO",
            x=TASKS,
            y=PPO_HOURS,
            marker_color=ppo_color,
            text=[f"{v}h" for v in PPO_HOURS],
            textposition="outside",
            textfont=dict(size=11),
        )
    )

    fig.add_trace(
        go.Bar(
            name="SAC",
            x=TASKS,
            y=SAC_HOURS,
            marker_color=sac_color,
            text=[f"{v}h" for v in SAC_HOURS],
            textposition="outside",
            textfont=dict(size=11),
        )
    )

    fig.add_trace(
        go.Bar(
            name="DreamerV3",
            x=TASKS,
            y=DREAMER_HOURS,
            marker_color=dreamer_color,
            text=[f"{v}h" for v in DREAMER_HOURS],
            textposition="outside",
            textfont=dict(size=11),
        )
    )

    # Speedup annotations for DreamerV3 vs PPO
    speedups = [PPO_HOURS[i] / DREAMER_HOURS[i] for i in range(len(TASKS))]
    for i, speedup in enumerate(speedups):
        fig.add_annotation(
            x=TASKS[i],
            y=DREAMER_HOURS[i],
            text=f"<b>{speedup:.0f}x faster</b>",
            showarrow=True,
            arrowhead=2,
            ay=-40,
            font=dict(size=10, color=speedup_color),
        )

    # DreamerV3 ablation annotation
    ablation_text = "<br>".join([f"{k}: {v}" for k, v in DREAMER_ABLATION.items()])
    fig.add_annotation(
        x=0.98,
        xref="paper",
        y=0.95,
        yref="paper",
        text=f"<b>DreamerV3 Ablation:</b><br>{ablation_text}",
        showarrow=False,
        font=dict(size=11, color=ann_color),
        bgcolor="rgba(0,0,0,0.7)" if dark_mode else "rgba(255,255,255,0.9)",
        bordercolor=ann_color,
        borderwidth=1,
        align="left",
        xanchor="right",
    )

    fig.update_layout(
        title=dict(
            text=(
                "Training Efficiency: GPU Hours to Reach 80%"
                " Success Rate by Algorithm"
                "<br><sup>Source: reinforcement-learning/results.md —"
                " Training Efficiency Comparison</sup>"
            ),
            font=dict(size=18, color=text_color),
            x=0.5,
        ),
        template=template,
        paper_bgcolor=bg_color,
        barmode="group",
        xaxis=dict(title="Surgical Task", tickfont=dict(size=12)),
        yaxis=dict(
            title="GPU Hours to 80% Success (log scale)",
            type="log",
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
    """Generate GPU training efficiency visualizations."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    png_dir = os.path.join(os.path.dirname(os.path.dirname(script_dir)), "png", "2nd")
    os.makedirs(png_dir, exist_ok=True)

    for dark_mode in [False, True]:
        suffix = "dark" if dark_mode else "light"
        fig = create_efficiency_chart(dark_mode=dark_mode)

        html_path = os.path.join(script_dir, f"gpu_training_efficiency_{suffix}.html")
        fig.write_html(html_path, include_plotlyjs=True)

        png_path = os.path.join(png_dir, f"gpu_training_efficiency_{suffix}.png")
        fig.write_image(png_path, width=1920, height=1080, scale=2)

    print("gpu_training_efficiency: done")


if __name__ == "__main__":
    main()

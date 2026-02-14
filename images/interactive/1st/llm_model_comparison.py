"""LLM Model Comparison for Surgical Robot Control.

Helps engineers select the optimal LLM backbone for their agentic
surgical system by comparing accuracy across task complexities.

Data source:
  - agentic-ai/results.md — LLM Model Comparison, Latency vs Quality Tradeoff
"""

import os

import plotly.graph_objects as go


MODELS = ["Claude Opus 4", "Claude Sonnet 4", "GPT-4o", "Claude Haiku 4", "Llama 70B"]
NAVIGATION = [97, 95, 94, 91, 88]
MANIPULATION = [89, 86, 84, 78, 75]
MULTI_STEP = [92, 89, 87, 81, 77]
AVERAGE = [93, 90, 88, 83, 80]

LATENCY_NOTE = "Sonnet 4: Best latency–quality tradeoff at 200 ms"


def create_llm_chart(dark_mode=False):
    """Create grouped bar chart with average line overlay."""
    if dark_mode:
        template = "plotly_dark"
        nav_color = "#00e5ff"
        manip_color = "#69f0ae"
        multi_color = "#ffab40"
        avg_color = "white"
        bg_color = "#1a1a2e"
        title_color = "#e0e0e0"
        ann_color = "#ffd740"
    else:
        template = "plotly_white"
        nav_color = "#1565c0"
        manip_color = "#2e7d32"
        multi_color = "#e65100"
        avg_color = "#c62828"
        bg_color = "#ffffff"
        title_color = "#1a1a2e"
        ann_color = "#ff6f00"

    fig = go.Figure()

    # Navigation bars
    fig.add_trace(
        go.Bar(
            name="Navigation",
            x=MODELS,
            y=NAVIGATION,
            marker_color=nav_color,
        )
    )

    # Manipulation bars
    fig.add_trace(
        go.Bar(
            name="Manipulation",
            x=MODELS,
            y=MANIPULATION,
            marker_color=manip_color,
        )
    )

    # Multi-Step bars
    fig.add_trace(
        go.Bar(
            name="Multi-Step",
            x=MODELS,
            y=MULTI_STEP,
            marker_color=multi_color,
        )
    )

    # Average line with markers
    fig.add_trace(
        go.Scatter(
            name="Average",
            x=MODELS,
            y=AVERAGE,
            mode="lines+markers",
            line=dict(color=avg_color, width=3, dash="solid"),
            marker=dict(size=12, symbol="star", color=avg_color),
        )
    )

    # Sonnet annotation
    fig.add_annotation(
        x="Claude Sonnet 4",
        y=AVERAGE[1] + 3,
        text=f"<b>{LATENCY_NOTE}</b>",
        showarrow=True,
        arrowhead=2,
        font=dict(size=12, color=ann_color),
        bgcolor="rgba(0,0,0,0.6)" if dark_mode else "rgba(255,255,255,0.8)",
        bordercolor=ann_color,
        borderwidth=1,
    )

    # Latency annotations under each model
    latencies = {"Claude Opus 4": "350ms", "Claude Sonnet 4": "200ms", "Claude Haiku 4": "80ms"}
    for model, latency in latencies.items():
        fig.add_annotation(
            x=model,
            y=72,
            text=f"<i>{latency}</i>",
            showarrow=False,
            font=dict(size=11, color=ann_color),
        )

    fig.update_layout(
        title=dict(
            text=(
                "LLM Performance Comparison: Natural Language → Surgical Robot Actions"
                "<br><sup>Source: agentic-ai/results.md — ROS 2 Agentic Action Mapping</sup>"
            ),
            font=dict(size=18, color=title_color),
            x=0.5,
        ),
        template=template,
        paper_bgcolor=bg_color,
        barmode="group",
        xaxis=dict(title="LLM Model", tickfont=dict(size=13)),
        yaxis=dict(title="Task Accuracy (%)", range=[70, 100], tickfont=dict(size=12)),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=13),
        ),
        margin=dict(t=120, l=60, r=40, b=80),
        width=1920,
        height=1080,
    )

    return fig


def main():
    """Generate LLM model comparison visualizations."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    png_dir = os.path.join(os.path.dirname(os.path.dirname(script_dir)), "png", "1st")
    os.makedirs(png_dir, exist_ok=True)

    for dark_mode in [False, True]:
        suffix = "dark" if dark_mode else "light"
        fig = create_llm_chart(dark_mode=dark_mode)

        html_path = os.path.join(script_dir, f"llm_model_comparison_{suffix}.html")
        fig.write_html(html_path, include_plotlyjs=True)

        png_path = os.path.join(png_dir, f"llm_model_comparison_{suffix}.png")
        fig.write_image(png_path, width=1920, height=1080, scale=2)

    print("llm_model_comparison: done")


if __name__ == "__main__":
    main()

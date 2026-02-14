"""Agentic AI Pipeline: End-to-End Latency Breakdown (Command to Robot Action).

Stacked bar chart with P95 scatter overlay showing latency budget for each
stage of the agentic pipeline from voice/text command to physical robot execution.

Data sources:
  - agentic-ai/results.md — Latency Distribution
  - generative-ai/results.md — Real-Time Latency Benchmarks
"""

import os

import plotly.graph_objects as go


STAGES = ["Command →\nRecognition", "Recognition →\nPlanning", "Planning →\nExecution"]
MEDIAN = [200, 150, 300]
P95 = [300, 280, 500]

EDGE_BENCHMARKS = [
    ("GR00T N1.6 edge", 5, "200 Hz"),
    ("Diffusion distilled", 25, "40 Hz"),
    ("VLM+Policy async", 155, "200 Hz ctrl"),
]


def create_latency_chart(dark_mode=False):
    """Create stacked bar chart with P95 scatter overlay."""
    if dark_mode:
        template = "plotly_dark"
        colors = ["#00e5ff", "#009688", "#76ff03"]
        p95_color = "#ff9800"
        total_color = "#ce93d8"
        text_color = "#e0e0e0"
        bg_color = "#1a1a2e"
        line_color = "#b0bec5"
        bench_color = "#81d4fa"
    else:
        template = "plotly_white"
        colors = ["#1565c0", "#00796b", "#2e7d32"]
        p95_color = "#d32f2f"
        total_color = "#7b1fa2"
        text_color = "#1a1a2e"
        bg_color = "#ffffff"
        line_color = "#616161"
        bench_color = "#0277bd"

    fig = go.Figure()

    # Stacked bar segments for median latency
    cumulative = 0
    for i, (stage, med) in enumerate(zip(STAGES, MEDIAN)):
        fig.add_trace(
            go.Bar(
                name=f"{stage.replace(chr(10), ' ')} ({med}ms)",
                x=["Agentic Pipeline"],
                y=[med],
                marker_color=colors[i],
                text=[f"{med}ms"],
                textposition="inside",
                textfont=dict(size=14, color="white"),
                width=0.5,
            )
        )
        cumulative += med

    # P95 scatter points at cumulative positions
    p95_cumulative = []
    running = 0
    for p in P95:
        running += p
        p95_cumulative.append(running)

    # Individual stage P95 markers
    for i, (stage, p95_val) in enumerate(zip(STAGES, P95)):
        fig.add_trace(
            go.Scatter(
                name=f"P95: {stage.replace(chr(10), ' ')}",
                x=["Agentic Pipeline"],
                y=[sum(P95[: i + 1])],
                mode="markers+text",
                marker=dict(size=14, color=p95_color, symbol="diamond"),
                text=[f"P95: {p95_val}ms"],
                textposition="middle right",
                textfont=dict(size=11, color=p95_color),
                showlegend=False,
            )
        )

    # Total end-to-end annotations
    fig.add_annotation(
        x="Agentic Pipeline",
        y=cumulative + 40,
        text=f"<b>Median Total: {cumulative}ms</b>",
        showarrow=False,
        font=dict(size=15, color=total_color),
    )
    fig.add_annotation(
        x="Agentic Pipeline",
        y=cumulative + 80,
        text=f"<b>P95 Total: {sum(P95)}ms</b>",
        showarrow=False,
        font=dict(size=15, color=p95_color),
    )

    # Edge benchmark reference bars
    for j, (name, latency, freq) in enumerate(EDGE_BENCHMARKS):
        fig.add_trace(
            go.Bar(
                name=f"{name} ({latency}ms, {freq})",
                x=[name],
                y=[latency],
                marker_color=bench_color,
                text=[f"{latency}ms"],
                textposition="outside",
                textfont=dict(size=12),
                width=0.4,
            )
        )

    fig.update_layout(
        title=dict(
            text=(
                "Agentic AI Pipeline: End-to-End Latency Breakdown"
                " (Command → Robot Action)"
                "<br><sup>Source: agentic-ai/results.md — Latency Distribution; "
                "generative-ai/results.md — Real-Time Benchmarks</sup>"
            ),
            font=dict(size=18, color=text_color),
            x=0.5,
        ),
        template=template,
        paper_bgcolor=bg_color,
        barmode="stack",
        xaxis=dict(title="Latency Component", tickfont=dict(size=12)),
        yaxis=dict(title="Latency (milliseconds)", tickfont=dict(size=12)),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=11),
        ),
        margin=dict(t=140, l=60, r=40, b=80),
        width=1920,
        height=1080,
    )

    return fig


def main():
    """Generate latency breakdown visualizations."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    png_dir = os.path.join(os.path.dirname(os.path.dirname(script_dir)), "png", "2nd")
    os.makedirs(png_dir, exist_ok=True)

    for dark_mode in [False, True]:
        suffix = "dark" if dark_mode else "light"
        fig = create_latency_chart(dark_mode=dark_mode)

        html_path = os.path.join(script_dir, f"agentic_latency_breakdown_{suffix}.html")
        fig.write_html(html_path, include_plotlyjs=True)

        png_path = os.path.join(png_dir, f"agentic_latency_breakdown_{suffix}.png")
        fig.write_image(png_path, width=1920, height=1080, scale=2)

    print("agentic_latency_breakdown: done")


if __name__ == "__main__":
    main()

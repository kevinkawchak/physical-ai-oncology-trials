"""Deployment Readiness Assessment: Multi-Dimensional Radar + Benchmarks.

Radar chart with table inset showing ONNX model validation, safety
constraint verification, and regulatory checklist status.

Data sources:
  - tools/deployment-readiness/deployment_readiness.py — run_benchmark(),
    check_safety_constraints(), run_regulatory_checklists()
  - reinforcement-learning/results.md — force/velocity safety metrics
  - configs/training_config.yaml — workspace bounds
"""

import os

import plotly.graph_objects as go
from plotly.subplots import make_subplots

RADAR_DIMENSIONS = [
    "ONNX Compatibility",
    "Latency Budget",
    "Force Safety",
    "Velocity Safety",
    "Workspace Bounds",
    "Regulatory Readiness",
]

RADAR_SCORES = [100, 77, 96, 100, 100, 68]

RADAR_DETAILS = [
    "Valid ONNX, opset 17, CPU+CUDA",
    "23.5ms / 100ms budget = 77% margin",
    "96.5% within 5N limit",
    "99.8% within 0.1 m/s limit",
    "99.9% within bounds",
    "13/19 checklist items",
]

BENCHMARK_DATA = [
    ("Mean Latency", "23.456 ms", "<100 ms", "PASS"),
    ("P95 Latency", "34.127 ms", "<100 ms", "PASS"),
    ("P99 Latency", "42.890 ms", "<100 ms", "PASS"),
    ("Throughput", "42.6 FPS", ">10 FPS", "PASS"),
    ("Model Size", "142.5 MB", "N/A", "—"),
    ("Graph Nodes", "847", "N/A", "—"),
    ("Validation", "40/42", "100%", "REVIEW"),
]


def create_chart(dark_mode=False):
    """Create radar chart with benchmark table inset."""
    if dark_mode:
        template = "plotly_dark"
        radar_color = "#00e5ff"
        fill_opacity = 0.2
        text_color = "#e0e0e0"
        bg_color = "#1a1a2e"
        header_bg = "#34495e"
        table_bg = "#2c2c3e"
        pass_color = "#27ae60"
        review_color = "#e67e22"
        line_color = "#4a4a5a"
    else:
        template = "plotly_white"
        radar_color = "#2980b9"
        fill_opacity = 0.3
        text_color = "#1a1a2e"
        bg_color = "#ffffff"
        header_bg = "#2c3e50"
        table_bg = "#f8f9fa"
        pass_color = "#27ae60"
        review_color = "#e67e22"
        line_color = "#bdc3c7"

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "polar"}, {"type": "table"}]],
        column_widths=[0.55, 0.45],
        subplot_titles=["Readiness Radar (0–100)", "Benchmark Results"],
    )

    # Radar chart
    theta = RADAR_DIMENSIONS + [RADAR_DIMENSIONS[0]]
    r = RADAR_SCORES + [RADAR_SCORES[0]]

    fig.add_trace(
        go.Scatterpolar(
            name="Readiness Score",
            r=r,
            theta=theta,
            fill="toself",
            fillcolor=f"rgba({int(radar_color[1:3], 16)},{int(radar_color[3:5], 16)},"
            f"{int(radar_color[5:7], 16)},{fill_opacity})",
            line=dict(color=radar_color, width=2),
            marker=dict(size=8),
            text=[
                f"{dim}: {score}%<br>{detail}"
                for dim, score, detail in zip(RADAR_DIMENSIONS, RADAR_SCORES, RADAR_DETAILS)
            ]
            + [f"{RADAR_DIMENSIONS[0]}: {RADAR_SCORES[0]}%"],
            hoverinfo="text",
        ),
        row=1,
        col=1,
    )

    # Score labels on radar
    for dim, score in zip(RADAR_DIMENSIONS, RADAR_SCORES):
        fig.add_annotation(
            x=0.27,
            y=0.5,
            text="",
            showarrow=False,
            xref="paper",
            yref="paper",
        )

    # Benchmark table
    metrics = [row[0] for row in BENCHMARK_DATA]
    values = [row[1] for row in BENCHMARK_DATA]
    thresholds = [row[2] for row in BENCHMARK_DATA]
    statuses = [row[3] for row in BENCHMARK_DATA]

    status_colors = []
    for s in statuses:
        if s == "PASS":
            status_colors.append(pass_color)
        elif s == "REVIEW":
            status_colors.append(review_color)
        else:
            status_colors.append(table_bg)

    fig.add_trace(
        go.Table(
            header=dict(
                values=["<b>Metric</b>", "<b>Value</b>", "<b>Threshold</b>", "<b>Status</b>"],
                fill_color=header_bg,
                font=dict(size=13, color="#ffffff"),
                align="center",
                line_color=line_color,
                height=40,
            ),
            cells=dict(
                values=[metrics, values, thresholds, statuses],
                fill_color=[
                    [table_bg] * len(metrics),
                    [table_bg] * len(metrics),
                    [table_bg] * len(metrics),
                    status_colors,
                ],
                font=dict(size=12, color=text_color),
                align="center",
                line_color=line_color,
                height=35,
            ),
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        title=dict(
            text=(
                "Deployment Readiness Assessment: ONNX Model Validation & Safety Compliance"
                "<br><sup>Source: tools/deployment-readiness/deployment_readiness.py;"
                " reinforcement-learning/results.md</sup>"
            ),
            font=dict(size=18, color=text_color),
            x=0.5,
        ),
        template=template,
        paper_bgcolor=bg_color,
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickvals=[20, 40, 60, 80, 100],
                tickfont=dict(size=10),
            ),
            angularaxis=dict(tickfont=dict(size=11)),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.3,
            font=dict(size=12),
        ),
        margin=dict(t=140, l=40, r=40, b=60),
        width=1920,
        height=1080,
    )

    return fig


def main():
    """Generate deployment readiness radar visualizations."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    png_dir = os.path.join(os.path.dirname(os.path.dirname(script_dir)), "png", "3rd")
    os.makedirs(png_dir, exist_ok=True)

    for dark_mode in [False, True]:
        suffix = "dark" if dark_mode else "light"
        fig = create_chart(dark_mode=dark_mode)

        html_path = os.path.join(script_dir, f"deployment_readiness_radar_{suffix}.html")
        fig.write_html(html_path, include_plotlyjs=True)

        png_path = os.path.join(png_dir, f"deployment_readiness_radar_{suffix}.png")
        fig.write_image(png_path, width=1920, height=1080, scale=2)

    print("deployment_readiness_radar: done")


if __name__ == "__main__":
    main()

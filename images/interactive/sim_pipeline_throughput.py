"""Sim-to-Real Pipeline: Framework Throughput & Hardware Requirements.

Visualizes the four-stage simulation-to-deployment pipeline with
throughput in steps/second and GPU/hardware requirements.
"""

import os

import plotly.graph_objects as go


STAGES = [
    "Stage 4: Physical Hardware<br>(Deployment)",
    "Stage 3: Gazebo Ionic 10.0.0<br>(ROS 2 Integration)",
    "Stage 2: MuJoCo 3.4.0<br>(Physics Validation)",
    "Stage 1: Isaac Lab 2.3.1<br>(GPU RL Training)",
]

THROUGHPUT = [1, 1000, 50000, 40000]  # Stage 4 = 1 (real-time placeholder)

GPU_REQ = [
    "Jetson AGX / edge",
    "CPU (real-time)",
    "CUDA 12.x (MJX) or CPU",
    "RTX 4090 / L40 / Blackwell",
]

DETAIL = [
    "ONNX opset 17, FP16",
    "Native ROS 2 Jazzy",
    "<1% dynamics deviation target",
    "4096 parallel envs",
]

BRIDGES = [
    dict(
        ax=0,
        ay=-40,
        x=0.5,
        y=2.5,
        text="ros2_surgical_deployment.py",
    ),
    dict(
        ax=0,
        ay=-40,
        x=0.5,
        y=1.5,
        text="urdf_sdf_mjcf_converter.py",
    ),
    dict(
        ax=0,
        ay=-40,
        x=0.5,
        y=0.5,
        text="isaac_mujoco_bridge.py",
    ),
]


def create_pipeline_chart(dark_mode=False):
    """Create horizontal bar chart for sim-to-real pipeline."""
    if dark_mode:
        template = "plotly_dark"
        bar_colors = ["#00e5ff", "#00bcd4", "#0097a7", "#006064"]
        text_color = "#e0e0e0"
        bg_color = "#1a1a2e"
        annotation_color = "#ffab40"
        arrow_color = "#ffab40"
    else:
        template = "plotly_white"
        bar_colors = ["#bbdefb", "#64b5f6", "#1e88e5", "#0d47a1"]
        text_color = "#1a1a2e"
        bg_color = "#ffffff"
        annotation_color = "#d32f2f"
        arrow_color = "#d32f2f"

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            y=STAGES,
            x=THROUGHPUT,
            orientation="h",
            marker=dict(color=bar_colors),
            text=[f"{t:,} steps/s" if t > 1 else "Real-time" for t in THROUGHPUT],
            textposition="inside",
            textfont=dict(size=14, color="white"),
            hovertext=[f"GPU: {gpu}<br>{det}" for gpu, det in zip(GPU_REQ, DETAIL)],
            hoverinfo="text",
        )
    )

    # Add GPU requirement annotations on the right side
    for i, (gpu, det) in enumerate(zip(GPU_REQ, DETAIL)):
        fig.add_annotation(
            x=THROUGHPUT[i],
            y=i,
            text=f"  {gpu} — {det}",
            showarrow=False,
            xanchor="left",
            font=dict(size=11, color=text_color),
        )

    # Add bridge arrows between stages
    annotations = []
    for bridge in BRIDGES:
        annotations.append(
            dict(
                x=25000,
                y=bridge["y"],
                ax=0,
                ay=-30,
                text=f"↕ {bridge['text']}",
                showarrow=False,
                font=dict(size=10, color=annotation_color),
            )
        )

    fig.update_layout(
        title=dict(
            text=(
                "Sim-to-Real Pipeline: Framework Throughput & Hardware Requirements"
                "<br><sup>Isaac Lab 2.3.1 → MuJoCo 3.4.0 → Gazebo Ionic 10.0.0"
                " → Physical Hardware</sup>"
            ),
            font=dict(size=18, color=text_color),
            x=0.5,
        ),
        template=template,
        paper_bgcolor=bg_color,
        xaxis=dict(
            title="Throughput (steps/second)",
            type="log",
            tickfont=dict(size=12),
        ),
        yaxis=dict(
            title="Pipeline Stage",
            tickfont=dict(size=12),
        ),
        annotations=annotations,
        margin=dict(t=100, l=250, r=250, b=60),
        width=1920,
        height=1080,
        showlegend=False,
    )

    return fig


def main():
    """Generate pipeline throughput visualizations."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    png_dir = os.path.join(os.path.dirname(script_dir), "png")
    os.makedirs(png_dir, exist_ok=True)

    for dark_mode in [False, True]:
        suffix = "dark" if dark_mode else "light"
        fig = create_pipeline_chart(dark_mode=dark_mode)

        html_path = os.path.join(script_dir, f"sim_pipeline_throughput_{suffix}.html")
        fig.write_html(html_path, include_plotlyjs=True)

        png_path = os.path.join(png_dir, f"sim_pipeline_throughput_{suffix}.png")
        fig.write_image(png_path, width=1920, height=1080, scale=2)

    print("sim_pipeline_throughput: done")


if __name__ == "__main__":
    main()

"""Physics Simulation Framework Comparison Radar Chart.

Gives engineers an at-a-glance comparison of the four simulation frameworks
across key engineering dimensions for surgical robotics.

Data sources:
  - frameworks/nvidia-isaac/INTEGRATION.md
  - frameworks/mujoco/INTEGRATION.md
  - frameworks/gazebo/INTEGRATION.md
  - frameworks/pybullet/INTEGRATION.md
"""

import os

import plotly.graph_objects as go


DIMENSIONS = [
    "GPU Throughput",
    "Physics Fidelity",
    "Parallel Environments",
    "ROS 2 Integration",
    "Ease of Setup",
    "Domain Randomization",
]

FRAMEWORKS = {
    "Isaac Lab 2.3.1": [9, 8, 10, 7, 5, 10],
    "MuJoCo 3.4.0 (MJX)": [9, 10, 9, 4, 8, 7],
    "Gazebo Ionic 10.0.0": [3, 7, 2, 10, 6, 5],
    "PyBullet 3.2.5": [4, 6, 3, 3, 9, 4],
}


def create_radar_chart(dark_mode=False):
    """Create radar/polar chart comparing simulation frameworks."""
    if dark_mode:
        template = "plotly_dark"
        colors = {
            "Isaac Lab 2.3.1": "cyan",
            "MuJoCo 3.4.0 (MJX)": "magenta",
            "Gazebo Ionic 10.0.0": "lime",
            "PyBullet 3.2.5": "gold",
        }
        fill_opacity = 0.15
        text_color = "#e0e0e0"
        bg_color = "#1a1a2e"
    else:
        template = "plotly_white"
        colors = {
            "Isaac Lab 2.3.1": "#1565c0",
            "MuJoCo 3.4.0 (MJX)": "#c62828",
            "Gazebo Ionic 10.0.0": "#2e7d32",
            "PyBullet 3.2.5": "#e65100",
        }
        fill_opacity = 0.2
        text_color = "#1a1a2e"
        bg_color = "#ffffff"

    fig = go.Figure()

    for name, scores in FRAMEWORKS.items():
        # Close the polygon by repeating the first value
        values = scores + [scores[0]]
        theta = DIMENSIONS + [DIMENSIONS[0]]

        fig.add_trace(
            go.Scatterpolar(
                r=values,
                theta=theta,
                fill="toself",
                fillcolor=f"rgba({_hex_to_rgb(colors[name])},{fill_opacity})",
                line=dict(color=colors[name], width=2),
                name=name,
                marker=dict(size=6),
            )
        )

    fig.update_layout(
        title=dict(
            text="Physics Simulation Framework Comparison for Surgical Robotics",
            font=dict(size=20, color=text_color),
            x=0.5,
        ),
        template=template,
        paper_bgcolor=bg_color,
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 10],
                tickvals=[2, 4, 6, 8, 10],
                tickfont=dict(size=10),
            ),
            angularaxis=dict(tickfont=dict(size=13)),
        ),
        legend=dict(
            font=dict(size=14),
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(t=100, l=80, r=80, b=120),
        width=1920,
        height=1080,
    )

    return fig


def _hex_to_rgb(color_str):
    """Convert hex color or named color to 'r,g,b' string."""
    named = {
        "cyan": "0,255,255",
        "magenta": "255,0,255",
        "lime": "0,255,0",
        "gold": "255,215,0",
    }
    if color_str in named:
        return named[color_str]
    c = color_str.lstrip("#")
    return f"{int(c[0:2], 16)},{int(c[2:4], 16)},{int(c[4:6], 16)}"


def main():
    """Generate radar chart visualizations."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    png_dir = os.path.join(os.path.dirname(script_dir), "png")
    os.makedirs(png_dir, exist_ok=True)

    for dark_mode in [False, True]:
        suffix = "dark" if dark_mode else "light"
        fig = create_radar_chart(dark_mode=dark_mode)

        html_path = os.path.join(script_dir, f"framework_comparison_radar_{suffix}.html")
        fig.write_html(html_path, include_plotlyjs=True)

        png_path = os.path.join(png_dir, f"framework_comparison_radar_{suffix}.png")
        fig.write_image(png_path, width=1920, height=1080, scale=2)

    print("framework_comparison_radar: done")


if __name__ == "__main__":
    main()

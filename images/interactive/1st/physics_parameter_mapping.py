"""Isaac Lab / MuJoCo Physics Parameter Mapping Heatmap.

Shows the exact parameter equivalences between Isaac Lab and MuJoCo
so engineers understand the bridge conversion accuracy.

Data sources:
  - q1-2026-standards/objective-1-bidirectional-conversion/format_mappings.yaml
  - unification/simulation_physics/physics_parameter_mapping.yaml
  - q1-2026-standards/objective-1-bidirectional-conversion/physics_equivalence_tests.py
"""

import os

import plotly.graph_objects as go


# Parameter mapping data — verified against format_mappings.yaml
ROWS = [
    ("Contact", "Stiffness", "contact_stiffness", "solref[0]", "solref = -stiff/damp", 2),
    ("Contact", "Damping", "contact_damping", "solref[1]", "solref[1] = -2√(k·m)", 2),
    ("Contact", "Friction", "static/dynamic_friction", "friction[0:2]", "Direct + cone model", 2),
    ("Joint", "Damping", "joint_damping", "damping", "Direct mapping", 1),
    ("Joint", "Stiffness", "joint_stiffness", "stiffness", "Direct mapping", 1),
    ("Joint", "Limits", "lower/upper_limit", "range[0:2]", "Direct mapping", 1),
    ("Solver", "Iterations", "solver_iterations", "iterations", "Direct mapping", 1),
    ("Solver", "Timestep", "physics_dt: 0.002", "timestep: 0.002", "Direct mapping", 1),
    ("Environment", "Gravity", "gravity: [0,0,-9.81]", "gravity: [0,0,-9.81]", "Direct mapping", 1),
    ("Actuator", "Gains", "stiffness, damping", "kp, kv", "Scale transform", 2),
    ("Actuator", "Force limits", "effort_limit", "forcerange", "Direct mapping", 1),
]


def create_heatmap(dark_mode=False):
    """Create annotated heatmap for parameter mapping."""
    if dark_mode:
        template = "plotly_dark"
        colorscale = [[0, "#006064"], [0.5, "#ff8f00"], [1.0, "#b71c1c"]]
        text_color = "white"
        bg_color = "#1a1a2e"
        title_color = "#e0e0e0"
    else:
        template = "plotly_white"
        colorscale = [[0, "#4caf50"], [0.5, "#ffc107"], [1.0, "#f44336"]]
        text_color = "#1a1a2e"
        bg_color = "#ffffff"
        title_color = "#1a1a2e"

    categories = [f"{r[0]}: {r[1]}" for r in ROWS]
    isaac_names = [r[2] for r in ROWS]
    mujoco_names = [r[3] for r in ROWS]
    conversions = [r[4] for r in ROWS]
    complexities = [r[5] for r in ROWS]

    # Build annotation text for each cell
    cell_text = [
        f"Isaac: {isaac}<br>MuJoCo: {mj}<br>Conversion: {conv}"
        for isaac, mj, conv in zip(isaac_names, mujoco_names, conversions)
    ]

    # Reshape into a column vector for heatmap display
    z = [[c] for c in complexities]
    annotations_text = [[t] for t in cell_text]

    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=["Conversion Complexity"],
            y=categories,
            colorscale=colorscale,
            zmin=1,
            zmax=3,
            text=annotations_text,
            texttemplate="%{text}",
            textfont=dict(size=10, color=text_color),
            hoverinfo="text",
            colorbar=dict(
                title="Complexity",
                tickvals=[1, 2, 3],
                ticktext=["1: Direct", "2: Formula", "3: Approx"],
                len=0.5,
            ),
            showscale=True,
        )
    )

    # Add a second column with the conversion detail
    fig.add_trace(
        go.Heatmap(
            z=z,
            x=["Mapping Detail"],
            y=categories,
            colorscale=colorscale,
            zmin=1,
            zmax=3,
            text=[[conv] for conv in conversions],
            texttemplate="%{text}",
            textfont=dict(size=12, color=text_color),
            hoverinfo="skip",
            showscale=False,
        )
    )

    # Re-do as a wider layout: 4 columns
    columns = ["Isaac Lab Parameter", "MuJoCo Parameter", "Conversion Method", "Complexity"]

    z_matrix = []
    text_matrix = []
    for row in ROWS:
        z_matrix.append([row[5], row[5], row[5], row[5]])
        text_matrix.append([row[2], row[3], row[4], str(row[5])])

    fig = go.Figure(
        go.Heatmap(
            z=z_matrix,
            x=columns,
            y=categories,
            colorscale=colorscale,
            zmin=1,
            zmax=3,
            text=text_matrix,
            texttemplate="%{text}",
            textfont=dict(size=12, color=text_color),
            hovertemplate="Category: %{y}<br>Column: %{x}<br>Value: %{text}<extra></extra>",
            colorbar=dict(
                title="Complexity",
                tickvals=[1, 2, 3],
                ticktext=["1: Direct Mapping", "2: Formula Transform", "3: Approximation"],
                len=0.6,
            ),
        )
    )

    fig.update_layout(
        title=dict(
            text=(
                "Isaac Lab ↔ MuJoCo Physics Parameter Mapping (Bidirectional Conversion)"
                "<br><sup>Source: format_mappings.yaml, physics_parameter_mapping.yaml"
                " — <1% deviation target</sup>"
            ),
            font=dict(size=18, color=title_color),
            x=0.5,
        ),
        template=template,
        paper_bgcolor=bg_color,
        xaxis=dict(side="top", tickfont=dict(size=13)),
        yaxis=dict(tickfont=dict(size=11), autorange="reversed"),
        margin=dict(t=120, l=180, r=120, b=40),
        width=1920,
        height=1080,
    )

    return fig


def main():
    """Generate physics parameter mapping visualizations."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    png_dir = os.path.join(os.path.dirname(os.path.dirname(script_dir)), "png", "1st")
    os.makedirs(png_dir, exist_ok=True)

    for dark_mode in [False, True]:
        suffix = "dark" if dark_mode else "light"
        fig = create_heatmap(dark_mode=dark_mode)

        html_path = os.path.join(script_dir, f"physics_parameter_mapping_{suffix}.html")
        fig.write_html(html_path, include_plotlyjs=True)

        png_path = os.path.join(png_dir, f"physics_parameter_mapping_{suffix}.png")
        fig.write_image(png_path, width=1920, height=1080, scale=2)

    print("physics_parameter_mapping: done")


if __name__ == "__main__":
    main()

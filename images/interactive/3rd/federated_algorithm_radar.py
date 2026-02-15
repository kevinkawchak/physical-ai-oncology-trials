"""Federated Learning Algorithm Comparison Radar.

Radar chart comparing FedAvg, FedProx, and SCAFFOLD across five
operational dimensions relevant to multi-site oncology trials.

Data sources:
  - federation/federated_coordinator.py — FedAvg (weighted average),
    FedProx (proximal term mu=0.01), SCAFFOLD (control variates)
  - federation/differential_privacy.py — epsilon-delta budgets
  - federation/secure_aggregation.py — secure multi-party computation
"""

import os

import plotly.graph_objects as go

DIMENSIONS = [
    "Convergence Speed",
    "Communication Efficiency",
    "Heterogeneity Handling",
    "Implementation Simplicity",
    "Privacy Compatibility",
]

FEDAVG = [7, 8, 4, 9, 8]
FEDPROX = [7, 7, 7, 7, 8]
SCAFFOLD = [9, 6, 9, 5, 7]

ALGO_NOTES = {
    "FedAvg": "Simple weighted average by patient count; fast but struggles with non-IID data",
    "FedProx": "Adds proximal regularization (mu=0.01); handles stragglers",
    "SCAFFOLD": "Uses control variates tracking client drift; best for heterogeneous sites but complex",
}


def create_chart(dark_mode=False):
    """Create radar chart comparing federated aggregation algorithms."""
    if dark_mode:
        template = "plotly_dark"
        colors = ["#00e5ff", "#76ff03", "#ff00ff"]
        fill_opacity = 0.15
        text_color = "#e0e0e0"
        bg_color = "#1a1a2e"
        ann_color = "#b0bec5"
    else:
        template = "plotly_white"
        colors = ["#1565c0", "#2e7d32", "#c62828"]
        fill_opacity = 0.2
        text_color = "#1a1a2e"
        bg_color = "#ffffff"
        ann_color = "#616161"

    fig = go.Figure()

    # Close the radar polygon by repeating first value
    theta = DIMENSIONS + [DIMENSIONS[0]]
    for name, values, color in [
        ("FedAvg", FEDAVG, colors[0]),
        ("FedProx", FEDPROX, colors[1]),
        ("SCAFFOLD", SCAFFOLD, colors[2]),
    ]:
        r = values + [values[0]]
        fig.add_trace(
            go.Scatterpolar(
                name=name,
                r=r,
                theta=theta,
                fill="toself",
                fillcolor=color.replace(")", f",{fill_opacity})").replace("#", "rgba(")
                if "rgba" in color
                else f"rgba({int(color[1:3], 16)},{int(color[3:5], 16)},{int(color[5:7], 16)},{fill_opacity})",
                line=dict(color=color, width=2),
                marker=dict(size=6),
            )
        )

    # Algorithm notes annotation
    notes_text = "<br>".join([f"<b>{k}:</b> {v}" for k, v in ALGO_NOTES.items()])
    fig.add_annotation(
        x=0.98,
        xref="paper",
        y=0.02,
        yref="paper",
        text=notes_text,
        showarrow=False,
        font=dict(size=10, color=ann_color),
        bgcolor="rgba(0,0,0,0.7)" if dark_mode else "rgba(255,255,255,0.9)",
        bordercolor=ann_color,
        borderwidth=1,
        align="left",
        xanchor="right",
        yanchor="bottom",
    )

    fig.update_layout(
        title=dict(
            text=(
                "Federated Aggregation Algorithms: FedAvg vs FedProx vs SCAFFOLD"
                "<br><sup>Source: federation/federated_coordinator.py — 3 aggregation strategies;"
                " federation/differential_privacy.py</sup>"
            ),
            font=dict(size=18, color=text_color),
            x=0.5,
        ),
        template=template,
        paper_bgcolor=bg_color,
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 10],
                tickvals=[2, 4, 6, 8, 10],
                tickfont=dict(size=11),
            ),
            angularaxis=dict(tickfont=dict(size=13)),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=13),
        ),
        margin=dict(t=140, l=80, r=200, b=80),
        width=1920,
        height=1080,
    )

    return fig


def main():
    """Generate federated algorithm radar visualizations."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    png_dir = os.path.join(os.path.dirname(os.path.dirname(script_dir)), "png", "3rd")
    os.makedirs(png_dir, exist_ok=True)

    for dark_mode in [False, True]:
        suffix = "dark" if dark_mode else "light"
        fig = create_chart(dark_mode=dark_mode)

        html_path = os.path.join(script_dir, f"federated_algorithm_radar_{suffix}.html")
        fig.write_html(html_path, include_plotlyjs=True)

        png_path = os.path.join(png_dir, f"federated_algorithm_radar_{suffix}.png")
        fig.write_image(png_path, width=1920, height=1080, scale=2)

    print("federated_algorithm_radar: done")


if __name__ == "__main__":
    main()

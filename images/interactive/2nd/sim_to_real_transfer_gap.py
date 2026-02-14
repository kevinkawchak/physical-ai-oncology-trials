"""Sim-to-Real Transfer Gap: dVRK Physical Platform Validation (5 Core Tasks).

Dumbbell/lollipop chart showing performance gap between simulation and
physical robot execution for each surgical task.

Data source: reinforcement-learning/results.md — Physical Robot Validation
"""

import os

import plotly.graph_objects as go


TASKS = ["Peg Transfer", "Needle Pickup", "Tissue Retraction", "Cutting", "Suture Throw"]
SIM_SUCCESS = [94, 91, 84, 86, 79]
PHYS_SUCCESS = [89, 82, 71, 72, 64]
GAPS = [5, 9, 13, 14, 15]

FAILURE_MODES = {
    "Grasp slip": 28,
    "Position error": 22,
    "Timing error": 18,
    "Visual confusion": 15,
    "Unexpected obstacle": 12,
    "Hardware fault": 5,
}


def _gap_color(gap, dark_mode=False):
    """Return color based on gap severity."""
    if gap < 10:
        return "#66bb6a" if dark_mode else "#2e7d32"
    elif gap <= 13:
        return "#fdd835" if dark_mode else "#f9a825"
    else:
        return "#ef5350" if dark_mode else "#c62828"


def create_transfer_chart(dark_mode=False):
    """Create dumbbell chart with gap annotations and failure mode inset."""
    if dark_mode:
        template = "plotly_dark"
        sim_color = "#00e5ff"
        phys_color = "#ffd700"
        text_color = "#e0e0e0"
        bg_color = "#1a1a2e"
        ann_color = "#ffd740"
    else:
        template = "plotly_white"
        sim_color = "#1565c0"
        phys_color = "#e65100"
        text_color = "#1a1a2e"
        bg_color = "#ffffff"
        ann_color = "#e65100"

    fig = go.Figure()

    # Connecting lines colored by gap severity
    for i in range(len(TASKS)):
        color = _gap_color(GAPS[i], dark_mode)
        fig.add_trace(
            go.Scatter(
                x=[PHYS_SUCCESS[i], SIM_SUCCESS[i]],
                y=[TASKS[i], TASKS[i]],
                mode="lines",
                line=dict(color=color, width=4),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        # Gap annotation on the line
        mid_x = (PHYS_SUCCESS[i] + SIM_SUCCESS[i]) / 2
        fig.add_annotation(
            x=mid_x,
            y=TASKS[i],
            text=f"<b>-{GAPS[i]}%</b>",
            showarrow=False,
            font=dict(size=13, color=color),
            yshift=18,
        )

    # Physical success dots
    fig.add_trace(
        go.Scatter(
            name="Physical (dVRK)",
            x=PHYS_SUCCESS,
            y=TASKS,
            mode="markers",
            marker=dict(size=16, color=phys_color, symbol="circle", line=dict(width=2, color="white")),
        )
    )

    # Simulation success dots
    fig.add_trace(
        go.Scatter(
            name="Simulation",
            x=SIM_SUCCESS,
            y=TASKS,
            mode="markers",
            marker=dict(size=16, color=sim_color, symbol="diamond", line=dict(width=2, color="white")),
        )
    )

    # Color legend for gap severity
    fig.add_annotation(
        x=0.98,
        xref="paper",
        y=0.95,
        yref="paper",
        text=(
            "<b>Gap Severity:</b><br>"
            "<span style='color:#2e7d32'>■</span> &lt;10% (Good)<br>"
            "<span style='color:#f9a825'>■</span> 10–13% (Moderate)<br>"
            "<span style='color:#c62828'>■</span> &gt;13% (Critical)"
        ),
        showarrow=False,
        font=dict(size=11, color=text_color),
        bgcolor="rgba(0,0,0,0.7)" if dark_mode else "rgba(255,255,255,0.9)",
        bordercolor=ann_color,
        borderwidth=1,
        align="left",
        xanchor="right",
    )

    # Failure mode analysis annotation
    failure_text = "<br>".join([f"{k}: {v}%" for k, v in FAILURE_MODES.items()])
    fig.add_annotation(
        x=0.98,
        xref="paper",
        y=0.45,
        yref="paper",
        text=f"<b>Failure Modes (n=200):</b><br>{failure_text}",
        showarrow=False,
        font=dict(size=10, color=ann_color),
        bgcolor="rgba(0,0,0,0.7)" if dark_mode else "rgba(255,255,255,0.9)",
        bordercolor=ann_color,
        borderwidth=1,
        align="left",
        xanchor="right",
    )

    fig.update_layout(
        title=dict(
            text=(
                "Sim-to-Real Transfer Gap: dVRK Physical Platform"
                " Validation (5 Core Tasks)"
                "<br><sup>Source: reinforcement-learning/results.md —"
                " Physical Robot Validation</sup>"
            ),
            font=dict(size=18, color=text_color),
            x=0.5,
        ),
        template=template,
        paper_bgcolor=bg_color,
        xaxis=dict(
            title="Success Rate (%)",
            range=[55, 100],
            tickfont=dict(size=12),
        ),
        yaxis=dict(
            title="Surgical Task",
            tickfont=dict(size=13),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=13),
        ),
        margin=dict(t=120, l=140, r=200, b=80),
        width=1920,
        height=1080,
    )

    return fig


def main():
    """Generate sim-to-real transfer gap visualizations."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    png_dir = os.path.join(os.path.dirname(os.path.dirname(script_dir)), "png", "2nd")
    os.makedirs(png_dir, exist_ok=True)

    for dark_mode in [False, True]:
        suffix = "dark" if dark_mode else "light"
        fig = create_transfer_chart(dark_mode=dark_mode)

        html_path = os.path.join(script_dir, f"sim_to_real_transfer_gap_{suffix}.html")
        fig.write_html(html_path, include_plotlyjs=True)

        png_path = os.path.join(png_dir, f"sim_to_real_transfer_gap_{suffix}.png")
        fig.write_image(png_path, width=1920, height=1080, scale=2)

    print("sim_to_real_transfer_gap: done")


if __name__ == "__main__":
    main()

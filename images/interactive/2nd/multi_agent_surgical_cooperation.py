"""Multi-Agent Surgical Cooperation: Procedure Time, Collisions, and Success Rate.

Grouped bar chart with 3 metric clusters showing benefits of adding AI agents
to surgical teams.

Data sources:
  - agentic-ai/results.md — Multi-Agent Surgical Cooperation
  - generative-ai/results.md — Human-Robot Team Performance
  - reinforcement-learning/results.md — Two-Agent Camera + Retractor, Surgeon Satisfaction
"""

import os

import plotly.graph_objects as go


CONFIGS = ["2 Humans\n(Baseline)", "1 Human +\n1 Agent", "2 Agents\n(Cooperative)"]
PROC_TIME = [100, 55.6, 28.8]
COLLISION = [100, 55.3, 2.0]
SUCCESS = [94, 96, 92]

TASK_REDUCTIONS = {
    "Tissue retraction": "-52%",
    "Camera positioning": "-63%",
    "Instrument exchange": "-41%",
    "Suture assistance": "-38%",
}

SATISFACTION = {
    "Independent": "3.2/5",
    "Centralized": "4.1/5",
    "Communication": "4.5/5",
}


def create_cooperation_chart(dark_mode=False):
    """Create grouped bar chart with 3 metric clusters."""
    if dark_mode:
        template = "plotly_dark"
        time_color = "#00e5ff"
        coll_color = "#ef9a9a"
        succ_color = "#76ff03"
        text_color = "#e0e0e0"
        bg_color = "#1a1a2e"
        ann_color = "#ffd740"
    else:
        template = "plotly_white"
        time_color = "#1565c0"
        coll_color = "#c62828"
        succ_color = "#2e7d32"
        text_color = "#1a1a2e"
        bg_color = "#ffffff"
        ann_color = "#e65100"

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            name="Procedure Time (% of baseline)",
            x=CONFIGS,
            y=PROC_TIME,
            marker_color=time_color,
            text=[f"{v}%" for v in PROC_TIME],
            textposition="outside",
            textfont=dict(size=12),
        )
    )

    fig.add_trace(
        go.Bar(
            name="Collision Rate (% of baseline)",
            x=CONFIGS,
            y=COLLISION,
            marker_color=coll_color,
            text=[f"{v}%" for v in COLLISION],
            textposition="outside",
            textfont=dict(size=12),
        )
    )

    fig.add_trace(
        go.Bar(
            name="Success Rate (%)",
            x=CONFIGS,
            y=SUCCESS,
            marker_color=succ_color,
            text=[f"{v}%" for v in SUCCESS],
            textposition="outside",
            textfont=dict(size=12),
        )
    )

    # Task-specific reduction annotations
    task_text = "<br>".join([f"{k}: {v}" for k, v in TASK_REDUCTIONS.items()])
    fig.add_annotation(
        x=0.98,
        xref="paper",
        y=0.95,
        yref="paper",
        text=f"<b>Task-Specific Reductions (1H+1A):</b><br>{task_text}",
        showarrow=False,
        font=dict(size=11, color=ann_color),
        bgcolor="rgba(0,0,0,0.7)" if dark_mode else "rgba(255,255,255,0.9)",
        bordercolor=ann_color,
        borderwidth=1,
        align="left",
        xanchor="right",
    )

    # Surgeon satisfaction annotation
    sat_text = "<br>".join([f"{k}: {v}" for k, v in SATISFACTION.items()])
    fig.add_annotation(
        x=0.98,
        xref="paper",
        y=0.65,
        yref="paper",
        text=f"<b>Surgeon Satisfaction:</b><br>{sat_text}",
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
                "Multi-Agent Surgical Cooperation: Procedure Time,"
                " Collisions, and Success Rate"
                "<br><sup>Sources: agentic-ai/results.md,"
                " reinforcement-learning/results.md</sup>"
            ),
            font=dict(size=18, color=text_color),
            x=0.5,
        ),
        template=template,
        paper_bgcolor=bg_color,
        barmode="group",
        xaxis=dict(title="Team Configuration", tickfont=dict(size=13)),
        yaxis=dict(title="Metric Value (%)", range=[0, 120], tickfont=dict(size=12)),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=13),
        ),
        margin=dict(t=120, l=60, r=200, b=100),
        width=1920,
        height=1080,
    )

    return fig


def main():
    """Generate multi-agent cooperation visualizations."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    png_dir = os.path.join(os.path.dirname(os.path.dirname(script_dir)), "png", "2nd")
    os.makedirs(png_dir, exist_ok=True)

    for dark_mode in [False, True]:
        suffix = "dark" if dark_mode else "light"
        fig = create_cooperation_chart(dark_mode=dark_mode)

        html_path = os.path.join(script_dir, f"multi_agent_surgical_cooperation_{suffix}.html")
        fig.write_html(html_path, include_plotlyjs=True)

        png_path = os.path.join(png_dir, f"multi_agent_surgical_cooperation_{suffix}.png")
        fig.write_image(png_path, width=1920, height=1080, scale=2)

    print("multi_agent_surgical_cooperation: done")


if __name__ == "__main__":
    main()

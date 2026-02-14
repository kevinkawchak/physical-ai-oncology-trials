"""Clinical Trial Workflow Time Comparison: Traditional vs Rule-Based vs Agentic.

Shows the dramatic time savings when moving from traditional manual workflows
to rule-based automation to full agentic AI.

Data source: agentic-ai/results.md — Clinical Trial Workflow Comparison
"""

import os

import plotly.graph_objects as go


TASKS = [
    "Patient Screening",
    "Scheduling",
    "Documentation",
    "Adverse Event<br>Reporting",
    "Protocol Compliance<br>Check",
]

TRADITIONAL = [45, 30, 60, 40, 20]
RULE_BASED = [15, 12, 25, 20, 8]
AGENTIC = [5, 3, 8, 7, 2]
IMPROVEMENT = ["89%", "90%", "87%", "83%", "90%"]


def create_workflow_chart(dark_mode=False):
    """Create grouped bar chart comparing workflow automation approaches."""
    if dark_mode:
        template = "plotly_dark"
        trad_color = "#616161"
        rule_color = "#1e90ff"
        agent_color = "#32cd32"
        text_color = "#e0e0e0"
        bg_color = "#1a1a2e"
        ann_color = "#32cd32"
    else:
        template = "plotly_white"
        trad_color = "#bdbdbd"
        rule_color = "#4682b4"
        agent_color = "#2e7d32"
        text_color = "#1a1a2e"
        bg_color = "#ffffff"
        ann_color = "#2e7d32"

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            name="Traditional (Manual)",
            x=TASKS,
            y=TRADITIONAL,
            marker_color=trad_color,
            textposition="outside",
        )
    )

    fig.add_trace(
        go.Bar(
            name="Rule-Based Automation",
            x=TASKS,
            y=RULE_BASED,
            marker_color=rule_color,
            textposition="outside",
        )
    )

    fig.add_trace(
        go.Bar(
            name="Agentic AI (Claude/GPT-4)",
            x=TASKS,
            y=AGENTIC,
            marker_color=agent_color,
            text=[f"{v} min" for v in AGENTIC],
            textposition="outside",
        )
    )

    # Improvement annotations above agentic bars
    annotations = []
    for i, imp in enumerate(IMPROVEMENT):
        annotations.append(
            dict(
                x=TASKS[i],
                y=AGENTIC[i] + 4,
                text=f"<b>{imp}↓</b>",
                showarrow=False,
                font=dict(size=13, color=ann_color),
                xanchor="center",
            )
        )

    fig.update_layout(
        title=dict(
            text="Clinical Trial Workflow Automation: Time per Task (minutes)",
            font=dict(size=20, color=text_color),
            x=0.5,
        ),
        template=template,
        paper_bgcolor=bg_color,
        barmode="group",
        xaxis=dict(title="Workflow Task", tickfont=dict(size=12)),
        yaxis=dict(title="Time (minutes)", tickfont=dict(size=12)),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=13),
        ),
        annotations=annotations,
        margin=dict(t=120, l=60, r=40, b=80),
        width=1920,
        height=1080,
    )

    return fig


def main():
    """Generate workflow comparison visualizations."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    png_dir = os.path.join(os.path.dirname(script_dir), "png")
    os.makedirs(png_dir, exist_ok=True)

    for dark_mode in [False, True]:
        suffix = "dark" if dark_mode else "light"
        fig = create_workflow_chart(dark_mode=dark_mode)

        html_path = os.path.join(script_dir, f"clinical_trial_workflow_{suffix}.html")
        fig.write_html(html_path, include_plotlyjs=True)

        png_path = os.path.join(png_dir, f"clinical_trial_workflow_{suffix}.png")
        fig.write_image(png_path, width=1920, height=1080, scale=2)

    print("clinical_trial_workflow: done")


if __name__ == "__main__":
    main()

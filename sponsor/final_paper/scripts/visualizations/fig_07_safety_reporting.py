# =======================================================================
# Figure 7: Safety Reporting Timeline Requirements (Horizontal Bar Timeline)
# Target: Section 5 (safety_pharmacovigilance.tex), Table tab:safety-timeline
# Visualization: Horizontal bar chart as a timeline
# Shows seven safety reporting types and their required timelines.
# =======================================================================

import plotly.graph_objects as go
import plotly.io as pio
import os

# --- Hardcoded data from Table tab:safety-timeline ---
# Note: For sorting, we use a numeric day value. Annual reports (60 days)
# and DSUR (60 days) represent the window after their trigger date.
# Robotic Safety: Cat C = 24 hrs = 1 day, Cat D = immediate = 0.04 days (1 hour proxy)

report_types = [
    "Robotic Safety Event\n(Category D: immediate)*",
    "Robotic Safety Event\n(Category C: 24 hours)*",
    "IND Safety Report\n(fatal/life-threatening)",
    "IND Safety Report\n(serious, unexpected)",
    "Follow-up IND\nSafety Report",
    "E2B(R3) ICSR",
    "DSUR",
    "IND Annual Report",
]

# Timeline in days (for bar lengths)
timeline_days = [0.25, 1, 7, 15, 15, 15, 60, 60]

# Regulatory basis
reg_basis = [
    "Adapted 21 CFR 312",
    "Adapted 21 CFR 312",
    "21 CFR \u00a7312.32(c)(1)",
    "21 CFR \u00a7312.32(c)(1)",
    "21 CFR \u00a7312.32(d)",
    "ICH E2B(R3)",
    "ICH E2F",
    "21 CFR \u00a7312.33",
]

# Agent responsibility
agent_resp = [
    "robot_execution_gateway",
    "robot_execution_gateway",
    "safety_agent",
    "safety_agent",
    "safety_agent",
    "safety_agent",
    "safety_agent",
    "regulatory_agent",
]

# Timeline labels
timeline_labels = [
    "Immediate",
    "24 hours",
    "7 calendar days",
    "15 calendar days",
    "15 calendar days",
    "15 days (typical)",
    "Annual, within 60 days of DIBD",
    "Within 60 days of anniversary",
]

# Color scale: red (shortest/most urgent) to blue (longest)
n = len(report_types)
colors = []
for i in range(n):
    frac = i / (n - 1) if n > 1 else 0
    r = int(220 * (1 - frac) + 66 * frac)
    g = int(50 * (1 - frac) + 133 * frac)
    b = int(50 * (1 - frac) + 200 * frac)
    colors.append(f"rgb({r},{g},{b})")

fig = go.Figure()

# Use log scale for better visibility of small values
import math
bar_values = [max(0.2, d) for d in timeline_days]

for i in range(n):
    # Robotic Safety Events get dashed border (unique to Physical AI)
    is_robotic = i < 2
    fig.add_trace(go.Bar(
        y=[report_types[i]],
        x=[bar_values[i]],
        orientation="h",
        marker_color=colors[i],
        marker_line=dict(
            color="black" if is_robotic else colors[i],
            width=2.5 if is_robotic else 0.5,
        ),
        marker_pattern_shape="x" if is_robotic else "",
        showlegend=False,
    ))

# Annotations for regulatory basis and agent responsibility
for i in range(n):
    fig.add_annotation(
        x=bar_values[i],
        y=report_types[i],
        text=f"  {timeline_labels[i]} | {reg_basis[i]} | {agent_resp[i]}",
        showarrow=False,
        font=dict(family="Times New Roman, serif", size=9, color="black"),
        xanchor="left",
    )

fig.update_layout(
    title=dict(
        text="Safety Reporting Timeline Requirements: Regulatory Deadlines and Agent Responsibility",
        font=dict(family="Times New Roman, serif", size=15, color="black"),
        x=0.5,
    ),
    xaxis=dict(
        title="Deadline (Days, log scale)",
        title_font=dict(family="Times New Roman, serif", size=14, color="black"),
        tickfont=dict(family="Times New Roman, serif", size=12, color="black"),
        type="log",
        gridcolor="lightgray",
        range=[-1, 2.2],
    ),
    yaxis=dict(
        tickfont=dict(family="Times New Roman, serif", size=10, color="black"),
    ),
    font=dict(family="Times New Roman, serif", size=14, color="black"),
    plot_bgcolor="white",
    paper_bgcolor="white",
    margin=dict(l=200, r=350, t=100, b=100),
    annotations=[
        dict(
            text="* Robotic Safety Event Reports are unique to Physical AI trials \u2014 no precedent in conventional clinical trials.",
            xref="paper", yref="paper",
            x=0.5, y=-0.14,
            showarrow=False,
            font=dict(family="Times New Roman, serif", size=10, color="#D62728"),
            xanchor="center",
        ),
    ] + [
        dict(
            x=bar_values[i],
            y=report_types[i],
            text=f"  {timeline_labels[i]} | {reg_basis[i]} | {agent_resp[i]}",
            showarrow=False,
            font=dict(family="Times New Roman, serif", size=9, color="black"),
            xanchor="left",
        )
        for i in range(n)
    ],
)

# --- Output ---
output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "figures")
os.makedirs(output_dir, exist_ok=True)

html_path = os.path.join(output_dir, "fig_07_safety_reporting.html")
png_path = os.path.join(output_dir, "fig_07_safety_reporting.png")

pio.write_html(fig, html_path, config={"displaylogo": False})
pio.write_image(fig, png_path, width=1200, height=800, scale=2)

print(f"HTML: {html_path}")
print(f"PNG:  {png_path}")

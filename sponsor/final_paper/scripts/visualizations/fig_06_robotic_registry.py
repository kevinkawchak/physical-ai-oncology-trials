# =======================================================================
# Figure 6: Robotic System Registry (Bubble Chart)
# Target: Section 10 (robotic_execution.tex), Table tab:robot-registry
# Visualization: Bubble chart / horizontal dot plot with sized markers
# Shows Robot Category, USL Score, Count, and Safety Tier.
# Historical evaluation results from the national platform paper.
# =======================================================================

import plotly.graph_objects as go
import plotly.io as pio
import os

# --- Hardcoded data from Table tab:robot-registry ---
categories = [
    "Surgical Robots",
    "Cobots",
    "RT Positioning",
    "Needle Placement",
    "Social Companions",
    "Humanoids",
    "RT Motion Tracking",
    "Imaging Assistants",
    "Steerable Needles",
    "Rehab Exoskeletons",
]

counts = [3, 4, 3, 2, 5, 3, 3, 4, 2, 3]
usl_scores = [87.5, 88.75, 82.5, 85.0, 90.0, 75.0, 86.25, 83.75, 85.0, 81.25]

# Safety tiers
# Tier 1 (green): Social Companions, Imaging Assistants
# Tier 2 (amber): Cobots, RT Positioning, Humanoids, RT Motion Tracking, Rehab Exoskeletons
# Tier 3 (red): Surgical Robots, Needle Placement, Steerable Needles
tiers = [3, 2, 2, 3, 1, 2, 2, 1, 3, 2]

tier_colors = {
    1: "#2CA02C",  # Green
    2: "#DDAA00",  # Amber/yellow (darkened for visibility)
    3: "#D62728",  # Red
}
tier_labels = {
    1: "Tier 1: Autonomous operation",
    2: "Tier 2: Clinician confirmation",
    3: "Tier 3: Mandatory human gate",
}

# Bubble size scaling
size_scale = 15
marker_sizes = [c * size_scale for c in counts]

fig = go.Figure()

# Plot each tier as a separate trace for legend
for tier in [1, 2, 3]:
    indices = [i for i, t in enumerate(tiers) if t == tier]
    fig.add_trace(go.Scatter(
        x=[usl_scores[i] for i in indices],
        y=[categories[i] for i in indices],
        mode="markers+text",
        name=tier_labels[tier],
        marker=dict(
            size=[marker_sizes[i] for i in indices],
            color=tier_colors[tier],
            line=dict(color="black", width=1),
            opacity=0.8,
        ),
        text=[f"n={counts[i]}" for i in indices],
        textposition="middle right",
        textfont=dict(family="Times New Roman, serif", size=10, color="black"),
    ))

# Size legend annotation
fig.add_annotation(
    text="Bubble size = robot count (n)",
    xref="paper", yref="paper",
    x=1.0, y=-0.12,
    showarrow=False,
    font=dict(family="Times New Roman, serif", size=10, color="gray"),
    xanchor="right",
)

fig.update_layout(
    title=dict(
        text="Physical AI Robotic System Registry: USL Scores, Count, and Safety Tier",
        font=dict(family="Times New Roman, serif", size=16, color="black"),
        x=0.5,
    ),
    xaxis=dict(
        title="Unification Standard Level (USL) Score",
        title_font=dict(family="Times New Roman, serif", size=14, color="black"),
        tickfont=dict(family="Times New Roman, serif", size=12, color="black"),
        gridcolor="lightgray",
        range=[70, 95],
    ),
    yaxis=dict(
        tickfont=dict(family="Times New Roman, serif", size=11, color="black"),
    ),
    legend=dict(
        font=dict(family="Times New Roman, serif", size=11, color="black"),
        title=dict(
            text="Safety Classification",
            font=dict(family="Times New Roman, serif", size=12, color="black"),
        ),
    ),
    font=dict(family="Times New Roman, serif", size=14, color="black"),
    plot_bgcolor="white",
    paper_bgcolor="white",
    margin=dict(l=150, r=40, t=100, b=80),
    annotations=[
        dict(
            text="Historical evaluation results from the 24-hour simulation (national platform paper).",
            xref="paper", yref="paper",
            x=0.5, y=-0.12,
            showarrow=False,
            font=dict(family="Times New Roman, serif", size=10, color="gray"),
            xanchor="center",
        ),
    ],
)

# --- Output ---
output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "figures")
os.makedirs(output_dir, exist_ok=True)

html_path = os.path.join(output_dir, "fig_06_robotic_registry.html")
png_path = os.path.join(output_dir, "fig_06_robotic_registry.png")

pio.write_html(fig, html_path, config={"displaylogo": False})
pio.write_image(fig, png_path, width=1200, height=800, scale=2)

print(f"HTML: {html_path}")
print(f"PNG:  {png_path}")

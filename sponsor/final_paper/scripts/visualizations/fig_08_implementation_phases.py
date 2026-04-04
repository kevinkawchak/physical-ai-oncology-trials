# =======================================================================
# Figure 8: Implementation Phase Timeline (Phased Roadmap with Milestones)
# Target: Section 16 (implementation_strategy.tex), Table tab:implementation-phases
# Visualization: Horizontal phased roadmap/Gantt with milestones
# Shows three implementation phases with key milestones.
# All data is projected/forecasted.
# =======================================================================

import plotly.graph_objects as go
import plotly.io as pio
import os

# --- Hardcoded data from Table tab:implementation-phases ---
# Phase 1: 12 months (Month 0-12), 1 site
# Phase 2: 18 months (Month 12-30), 3-5 sites
# Phase 3: 24 months (Month 30-54), 20+ sites

phases = ["Phase 1: Single-Site Pilot", "Phase 2: Multi-Site Expansion", "Phase 3: National Network"]
starts = [0, 12, 30]
durations = [12, 18, 24]
sites = ["1 site", "3\u20135 sites", "20+ sites"]

# Colors per phase
phase_colors = ["#4472C4", "#ED7D31", "#548235"]

# Milestones: (label, overall_month, phase_index)
milestones = [
    # Phase 1
    ("Shadow mode\nvalidation", 1.5, 0),       # months 1-3 midpoint
    ("Autonomous mode\nactivation", 4, 0),
    ("First patient\nenrolled", 6, 0),
    # Phase 2
    ("Site 2\nactivation", 15, 1),              # month 3 of Phase 2 = 15
    ("Federated learning\noperational", 18, 1), # month 6 = 18
    ("Multi-site\nenrollment", 21, 1),          # month 9 = 21
    # Phase 3
    ("10 sites\nactive", 36, 2),                # month 6 = 36
    ("Industry\npartnerships", 42, 2),          # month 12 = 42
    ("20+ sites\nactive", 48, 2),               # month 18 = 48
]

fig = go.Figure()

# Phase bars (horizontal)
y_positions = [2, 1, 0]
for i in range(3):
    fig.add_trace(go.Bar(
        x=[durations[i]],
        y=[y_positions[i]],
        base=[starts[i]],
        orientation="h",
        marker_color=phase_colors[i],
        marker_pattern_shape="/",  # Hatched for projected data
        marker_line=dict(color="black", width=1),
        name=phases[i],
        width=0.5,
        text=[f"{phases[i]} ({durations[i]} mo, {sites[i]})"],
        textposition="inside",
        textfont=dict(family="Times New Roman, serif", size=11, color="white"),
    ))

# Milestone diamond markers
for label, month, phase_idx in milestones:
    y = y_positions[phase_idx]
    fig.add_trace(go.Scatter(
        x=[month],
        y=[y + 0.35],
        mode="markers",
        marker=dict(
            symbol="diamond",
            size=10,
            color=phase_colors[phase_idx],
            line=dict(color="black", width=1),
        ),
        showlegend=False,
    ))
    fig.add_annotation(
        x=month,
        y=y + 0.45,
        text=label,
        showarrow=False,
        font=dict(family="Times New Roman, serif", size=8, color="black"),
        yanchor="bottom",
        xanchor="center",
    )

# Site count annotation track (below bars)
site_annotations = [
    (0, "1 site"),
    (12, "1 \u2192 3\u20135 sites"),
    (30, "3\u20135 \u2192 20+ sites"),
    (54, "20+ sites"),
]
for month, label in site_annotations:
    fig.add_annotation(
        x=month,
        y=-0.5,
        text=f"<b>{label}</b>",
        showarrow=False,
        font=dict(family="Times New Roman, serif", size=9, color="black"),
        xanchor="center",
    )

fig.update_layout(
    title=dict(
        text="National Implementation Strategy: Three-Phase Deployment Roadmap",
        font=dict(family="Times New Roman, serif", size=17, color="black"),
        x=0.5,
    ),
    xaxis=dict(
        title="Month",
        title_font=dict(family="Times New Roman, serif", size=14, color="black"),
        tickfont=dict(family="Times New Roman, serif", size=12, color="black"),
        tickvals=[0, 6, 12, 18, 24, 30, 36, 42, 48, 54],
        gridcolor="lightgray",
        range=[-1, 56],
    ),
    yaxis=dict(
        showticklabels=False,
        showgrid=False,
        range=[-1, 3.2],
    ),
    legend=dict(
        font=dict(family="Times New Roman, serif", size=12, color="black"),
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="center",
        x=0.5,
    ),
    font=dict(family="Times New Roman, serif", size=14, color="black"),
    plot_bgcolor="white",
    paper_bgcolor="white",
    margin=dict(l=80, r=40, t=120, b=100),
    annotations=[
        dict(
            text="Timeline represents projected deployment plan; actual timelines subject to "
                 "regulatory engagement and operational validation.",
            xref="paper", yref="paper",
            x=0.5, y=-0.16,
            showarrow=False,
            font=dict(family="Times New Roman, serif", size=10, color="gray"),
            xanchor="center",
        ),
    ] + [
        dict(
            x=month,
            y=-0.5,
            text=f"<b>{label}</b>",
            showarrow=False,
            font=dict(family="Times New Roman, serif", size=9, color="black"),
            xanchor="center",
        )
        for month, label in site_annotations
    ] + [
        dict(
            x=month,
            y=y_positions[phase_idx] + 0.45,
            text=label,
            showarrow=False,
            font=dict(family="Times New Roman, serif", size=8, color="black"),
            yanchor="bottom",
            xanchor="center",
        )
        for label, month, phase_idx in milestones
    ],
)

# --- Output ---
output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "figures")
os.makedirs(output_dir, exist_ok=True)

html_path = os.path.join(output_dir, "fig_08_implementation_phases.html")
png_path = os.path.join(output_dir, "fig_08_implementation_phases.png")

pio.write_html(fig, html_path, config={"displaylogo": False})
pio.write_image(fig, png_path, width=1200, height=800, scale=2)

print(f"HTML: {html_path}")
print(f"PNG:  {png_path}")

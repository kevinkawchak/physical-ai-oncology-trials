# =======================================================================
# Figure 5: Study Startup Traditional vs. Autonomous Comparison
# Target: Section 4 (clinical_operations.tex), Table tab:study-startup
# Visualization: Paired horizontal bar chart
# Compares Traditional and Autonomous timelines for seven startup steps.
# Note: Uses MIDPOINT of each range. Bars are independent comparisons,
# NOT sequential Gantt bars (the paper states they run in parallel).
# =======================================================================

import plotly.graph_objects as go
import plotly.io as pio
import os

# --- Hardcoded data from Table tab:study-startup ---
steps = [
    "Site identification\n& feasibility",
    "Regulatory/ethics\nsubmission",
    "Contract/budget\nnegotiation",
    "Site training &\nsystem setup",
    "Supply & IRT\nreadiness",
    "Site activation",
    "Total",
]

# Ranges: (low, high) in days
trad_ranges = [(30, 45), (30, 60), (30, 60), (14, 30), (14, 21), (7, 14), (90, 180)]
auto_ranges = [(5, 10), (10, 20), (7, 14), (5, 10), (3, 7), (1, 3), (25, 50)]

# Midpoints
trad_mid = [(lo + hi) / 2 for lo, hi in trad_ranges]
auto_mid = [(lo + hi) / 2 for lo, hi in auto_ranges]

# Range labels
trad_labels = [f"{lo}\u2013{hi} days" for lo, hi in trad_ranges]
auto_labels = [f"{lo}\u2013{hi} days" for lo, hi in auto_ranges]

# Colors
color_traditional = "#E07050"  # Coral/red
color_autonomous = "#2CA58D"   # Teal/green

fig = go.Figure()

# Traditional bars
fig.add_trace(go.Bar(
    name="Traditional",
    y=steps,
    x=trad_mid,
    orientation="h",
    marker_color=color_traditional,
    text=trad_labels,
    textposition="outside",
    textfont=dict(family="Times New Roman, serif", size=10, color="black"),
))

# Autonomous bars
fig.add_trace(go.Bar(
    name="Autonomous",
    y=steps,
    x=auto_mid,
    orientation="h",
    marker_color=color_autonomous,
    text=auto_labels,
    textposition="outside",
    textfont=dict(family="Times New Roman, serif", size=10, color="black"),
))

# Visually separate the Total row with a shape
fig.add_shape(
    type="line",
    x0=0, x1=1, xref="paper",
    y0=5.5, y1=5.5, yref="y",
    line=dict(color="black", width=1.5, dash="dot"),
)

fig.update_layout(
    title=dict(
        text="Study Startup Acceleration: Traditional vs. Autonomous Timelines",
        font=dict(family="Times New Roman, serif", size=17, color="black"),
        x=0.5,
    ),
    xaxis=dict(
        title="Duration (Days, midpoint of range)",
        title_font=dict(family="Times New Roman, serif", size=14, color="black"),
        tickfont=dict(family="Times New Roman, serif", size=12, color="black"),
        gridcolor="lightgray",
    ),
    yaxis=dict(
        tickfont=dict(family="Times New Roman, serif", size=11, color="black"),
        autorange="reversed",
    ),
    barmode="group",
    bargroupgap=0.2,
    legend=dict(
        font=dict(family="Times New Roman, serif", size=13, color="black"),
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="center",
        x=0.5,
    ),
    font=dict(family="Times New Roman, serif", size=14, color="black"),
    plot_bgcolor="white",
    paper_bgcolor="white",
    margin=dict(l=150, r=100, t=100, b=80),
    annotations=[
        dict(
            text="Bars show midpoint of range. Autonomous steps run in parallel, not sequentially.",
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

html_path = os.path.join(output_dir, "fig_05_study_startup.html")
png_path = os.path.join(output_dir, "fig_05_study_startup.png")

pio.write_html(fig, html_path, config={"displaylogo": False})
pio.write_image(fig, png_path, width=1200, height=800, scale=2)

print(f"HTML: {html_path}")
print(f"PNG:  {png_path}")

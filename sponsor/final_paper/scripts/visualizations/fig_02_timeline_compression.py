# =======================================================================
# Figure 2: Timeline Compression by Phase (Butterfly Chart)
# Target: Section 15 (financial_analysis.tex), Table tab:timeline-compression
# Visualization: Horizontal paired bar chart (butterfly/diverging)
# Shows Traditional vs. Autonomous durations for each development phase.
# =======================================================================

import plotly.graph_objects as go
import plotly.io as pio
import os

# --- Hardcoded data from Table tab:timeline-compression ---
phases = ["Phase I", "Phase II", "Phase III", "Regulatory submission"]
traditional_months = [18, 24, 36, 9]
autonomous_months = [12, 16, 24, 3]
days_saved = [180, 240, 360, 180]
total_savings = ["$1.4M", "$5.7M", "$20.1M", "$10.0M"]

# --- Colors ---
color_traditional = "#C05050"  # Muted red
color_autonomous = "#5078A8"   # Muted blue

fig = go.Figure()

# Traditional bars (extend left, negative values)
fig.add_trace(go.Bar(
    name="Traditional Duration",
    y=phases,
    x=[-m for m in traditional_months],
    orientation="h",
    marker_color=color_traditional,
    marker_pattern_shape="/",  # Hatched for projected data
    text=[f"{m} mo" for m in traditional_months],
    textposition="inside",
    textfont=dict(family="Times New Roman, serif", size=12, color="white"),
))

# Autonomous bars (extend right, positive values)
fig.add_trace(go.Bar(
    name="Autonomous Duration",
    y=phases,
    x=autonomous_months,
    orientation="h",
    marker_color=color_autonomous,
    marker_pattern_shape="/",  # Hatched for projected data
    text=[f"{m} mo" for m in autonomous_months],
    textposition="inside",
    textfont=dict(family="Times New Roman, serif", size=12, color="white"),
))

# Annotations for days saved and total savings (right side)
for i, phase in enumerate(phases):
    fig.add_annotation(
        x=max(autonomous_months) + 5,
        y=phase,
        text=f"{days_saved[i]} days saved | {total_savings[i]}",
        showarrow=False,
        font=dict(family="Times New Roman, serif", size=11, color="black"),
        xanchor="left",
    )

fig.update_layout(
    title=dict(
        text="Development Timeline Compression: Traditional vs. Autonomous",
        font=dict(family="Times New Roman, serif", size=18, color="black"),
        x=0.5,
    ),
    xaxis=dict(
        title="Duration (Months)",
        title_font=dict(family="Times New Roman, serif", size=14, color="black"),
        tickfont=dict(family="Times New Roman, serif", size=12, color="black"),
        tickvals=[-36, -24, -18, -12, -9, 0, 3, 12, 16, 24],
        ticktext=["36", "24", "18", "12", "9", "0", "3", "12", "16", "24"],
        gridcolor="lightgray",
        zeroline=True,
        zerolinecolor="black",
        zerolinewidth=2,
    ),
    yaxis=dict(
        tickfont=dict(family="Times New Roman, serif", size=13, color="black"),
        autorange="reversed",
    ),
    barmode="overlay",
    legend=dict(
        font=dict(family="Times New Roman, serif", size=13, color="black"),
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="center",
        x=0.4,
    ),
    font=dict(family="Times New Roman, serif", size=14, color="black"),
    plot_bgcolor="white",
    paper_bgcolor="white",
    margin=dict(l=160, r=200, t=120, b=100),
    annotations=[
        dict(
            text="<b>Total: 87 months \u2192 55 months | 960 days saved | $37.2M total savings</b>",
            xref="paper", yref="paper",
            x=0.5, y=-0.15,
            showarrow=False,
            font=dict(family="Times New Roman, serif", size=12, color="black"),
            xanchor="center",
        ),
        dict(
            text="All durations represent projected estimates based on Tufts CSDD benchmarks and conservative assumptions.",
            xref="paper", yref="paper",
            x=0.5, y=-0.22,
            showarrow=False,
            font=dict(family="Times New Roman, serif", size=10, color="gray"),
            xanchor="center",
        ),
    ] + [
        dict(
            x=max(autonomous_months) + 5,
            y=phase,
            text=f"{days_saved[i]} days saved | {total_savings[i]}",
            showarrow=False,
            font=dict(family="Times New Roman, serif", size=11, color="black"),
            xanchor="left",
        )
        for i, phase in enumerate(phases)
    ],
)

# Remove the duplicate trace-level annotations since we put them in layout
fig.data[0].text = [f"{m} mo" for m in traditional_months]
fig.data[1].text = [f"{m} mo" for m in autonomous_months]

# --- Output ---
output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "figures")
os.makedirs(output_dir, exist_ok=True)

html_path = os.path.join(output_dir, "fig_02_timeline_compression.html")
png_path = os.path.join(output_dir, "fig_02_timeline_compression.png")

pio.write_html(fig, html_path, config={"displaylogo": False})
pio.write_image(fig, png_path, width=1200, height=800, scale=2)

print(f"HTML: {html_path}")
print(f"PNG:  {png_path}")

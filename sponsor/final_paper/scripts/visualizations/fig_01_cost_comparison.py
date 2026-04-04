# =======================================================================
# Figure 1: Traditional vs. Autonomous Sponsor Cost Comparison
# Target: Section 15 (financial_analysis.tex), Table tab:cost-comparison
# Visualization: Grouped bar chart
# Compares Traditional vs. Autonomous sponsor costs across five categories.
# =======================================================================

import plotly.graph_objects as go
import plotly.io as pio
import os

# --- Hardcoded data from Table tab:cost-comparison ---
categories = [
    "Direct trial costs",
    "Sponsor staffing",
    "Infrastructure/compute",
    "CRO/vendor fees",
    "Delay opportunity cost",
]

traditional_totals = [105.5, 23.0, 3.5, 32.0, 26.1]  # Millions USD
autonomous_totals = [63.3, 3.5, 3.5, 25.6, 10.4]      # Millions USD

# --- Colors (colorblind-friendly) ---
color_traditional = "#4472C4"  # Blue
color_autonomous = "#2CA58D"   # Teal/green

fig = go.Figure()

# Traditional bars (projected/forecasted: use solid fill as these are benchmark-based)
fig.add_trace(go.Bar(
    name="Traditional Total",
    x=categories,
    y=traditional_totals,
    marker_color=color_traditional,
    text=[f"${v:.1f}M" for v in traditional_totals],
    textposition="outside",
    textfont=dict(family="Times New Roman, serif", size=12, color="black"),
))

# Autonomous bars (projected/forecasted)
fig.add_trace(go.Bar(
    name="Autonomous Total",
    x=categories,
    y=autonomous_totals,
    marker_color=color_autonomous,
    text=[f"${v:.1f}M" for v in autonomous_totals],
    textposition="outside",
    textfont=dict(family="Times New Roman, serif", size=12, color="black"),
))

fig.update_layout(
    title=dict(
        text="Traditional vs. Autonomous Sponsor Cost Comparison",
        font=dict(family="Times New Roman, serif", size=18, color="black"),
        x=0.5,
    ),
    yaxis=dict(
        title="Cost (Millions USD)",
        title_font=dict(family="Times New Roman, serif", size=14, color="black"),
        tickfont=dict(family="Times New Roman, serif", size=12, color="black"),
        gridcolor="lightgray",
        range=[0, 130],
    ),
    xaxis=dict(
        tickfont=dict(family="Times New Roman, serif", size=11, color="black"),
    ),
    barmode="group",
    bargroupgap=0.25,
    bargap=0.15,
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
    margin=dict(l=80, r=40, t=120, b=80),
    annotations=[
        dict(
            text="<b>Total: Traditional $190.1M | Autonomous $106.3M (44% reduction)</b>",
            xref="paper", yref="paper",
            x=0.5, y=-0.18,
            showarrow=False,
            font=dict(family="Times New Roman, serif", size=13, color="black"),
            xanchor="center",
        ),
        dict(
            text="All cost figures are projected estimates based on Tufts CSDD benchmarks and conservative assumptions.",
            xref="paper", yref="paper",
            x=0.5, y=-0.25,
            showarrow=False,
            font=dict(family="Times New Roman, serif", size=10, color="gray"),
            xanchor="center",
        ),
    ],
)

fig.update_xaxes(showgrid=False)

# --- Output ---
output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "figures")
os.makedirs(output_dir, exist_ok=True)

html_path = os.path.join(output_dir, "fig_01_cost_comparison.html")
png_path = os.path.join(output_dir, "fig_01_cost_comparison.png")

pio.write_html(fig, html_path, config={"displaylogo": False})
pio.write_image(fig, png_path, width=1200, height=800, scale=2)

print(f"HTML: {html_path}")
print(f"PNG:  {png_path}")

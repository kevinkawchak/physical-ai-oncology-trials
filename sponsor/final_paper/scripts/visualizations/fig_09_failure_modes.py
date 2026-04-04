# =======================================================================
# Figure 9: Failure Mode Mitigation Matrix (Risk Heatmap)
# Target: Section 4 (clinical_operations.tex), Table tab:failure-modes
#         Section 16 (implementation_strategy.tex), Table tab:implementation-risks
# Visualization: Risk heatmap matrix
# Combines five clinical operations failure modes and seven implementation risks.
# =======================================================================

import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import os

# --- Hardcoded data ---

# Failure modes (from tab:failure-modes) - likelihood/impact INFERRED as Medium/High
failure_modes = [
    ("Study startup delay", "Medium", "High", "clinops_agent"),
    ("High screen failure", "Medium", "High", "clinops_agent"),
    ("Protocol amendments", "Medium", "High", "clinical_accountability_agent"),
    ("Patient dropout", "Medium", "Medium", "clinops_agent"),
    ("Supply interruption", "Medium", "High", "supply_agent"),
]

# Implementation risks (from tab:implementation-risks) - EXPLICIT values
impl_risks = [
    ("Agent system reliability\nbelow 99.9%", "Medium", "High", "quality_agent"),
    ("Integration failures with\ndiverse site systems", "High", "Medium", "site_gateway"),
    ("FDA non-acceptance of\nautonomous sponsor", "Medium", "High", "regulatory_agent"),
    ("International regulatory\ndivergence on AI sponsors", "High", "Medium", "regulatory_agent"),
    ("Site staff resistance to\nautonomous oversight", "Medium", "Medium", "site_gateway"),
    ("Insufficient qualified sites\nfor national scale", "Low", "High", "clinops_agent"),
    ("Robotic procedure safety\nevent during pilot", "Low", "High", "robot_execution_gateway"),
]

# Combine
all_risks = failure_modes + impl_risks
labels = [r[0] for r in all_risks]
likelihoods = [r[1] for r in all_risks]
impacts = [r[2] for r in all_risks]
owners = [r[3] for r in all_risks]

# Numeric mapping: Low=1, Medium=2, High=3
val_map = {"Low": 1, "Medium": 2, "High": 3}
composite_scores = [val_map[l] * val_map[i] for l, i in zip(likelihoods, impacts)]

# Color mapping: Green (1-2), Yellow (3-4), Red (6+)
def risk_color(score):
    if score <= 2:
        return "#4CAF50"  # Green
    elif score <= 4:
        return "#FFC107"  # Yellow/amber
    else:
        return "#F44336"  # Red

colors = [risk_color(s) for s in composite_scores]

fig = go.Figure()

# Create horizontal bars colored by composite risk score
fig.add_trace(go.Bar(
    y=labels,
    x=composite_scores,
    orientation="h",
    marker_color=colors,
    marker_line=dict(color="black", width=0.5),
    text=[f"L: {l} | I: {i} | Score: {s} | {o}"
          for l, i, s, o in zip(likelihoods, impacts, composite_scores, owners)],
    textposition="inside",
    textfont=dict(family="Times New Roman, serif", size=9, color="black"),
    showlegend=False,
))

# Visual separator between failure modes and implementation risks
fig.add_shape(
    type="line",
    x0=0, x1=1, xref="paper",
    y0=4.5, y1=4.5, yref="y",
    line=dict(color="black", width=2, dash="dash"),
)

# Group labels
fig.add_annotation(
    x=-0.02, y=2,
    xref="paper", yref="y",
    text="<b>Clinical Operations<br>Failure Modes</b>",
    showarrow=False,
    font=dict(family="Times New Roman, serif", size=10, color="black"),
    xanchor="right",
    textangle=-90,
)
fig.add_annotation(
    x=-0.02, y=8,
    xref="paper", yref="y",
    text="<b>Implementation<br>Risks</b>",
    showarrow=False,
    font=dict(family="Times New Roman, serif", size=10, color="black"),
    xanchor="right",
    textangle=-90,
)

# Legend for risk levels
for label_text, color, y_pos in [
    ("Low risk (1\u20132)", "#4CAF50", 1.08),
    ("Moderate risk (3\u20134)", "#FFC107", 1.03),
    ("High risk (6+)", "#F44336", 0.98),
]:
    fig.add_annotation(
        x=0.85, y=y_pos,
        xref="paper", yref="paper",
        text=f'<span style="color:{color}">\u25A0</span> {label_text}',
        showarrow=False,
        font=dict(family="Times New Roman, serif", size=11, color="black"),
        xanchor="left",
    )

fig.update_layout(
    title=dict(
        text="Risk Assessment Matrix: Clinical Operations Failure Modes and Implementation Risks",
        font=dict(family="Times New Roman, serif", size=15, color="black"),
        x=0.5,
    ),
    xaxis=dict(
        title="Composite Risk Score (Likelihood \u00d7 Impact)",
        title_font=dict(family="Times New Roman, serif", size=14, color="black"),
        tickfont=dict(family="Times New Roman, serif", size=12, color="black"),
        tickvals=[1, 2, 3, 4, 6, 9],
        gridcolor="lightgray",
        range=[0, 10],
    ),
    yaxis=dict(
        tickfont=dict(family="Times New Roman, serif", size=10, color="black"),
        autorange="reversed",
    ),
    font=dict(family="Times New Roman, serif", size=14, color="black"),
    plot_bgcolor="white",
    paper_bgcolor="white",
    margin=dict(l=200, r=40, t=120, b=100),
    annotations=[
        dict(
            text="Failure mode likelihood/impact values are inferred (Medium/High) as explicit values are not given in the table. "
                 "Implementation risk values are from Table tab:implementation-risks.",
            xref="paper", yref="paper",
            x=0.5, y=-0.14,
            showarrow=False,
            font=dict(family="Times New Roman, serif", size=9, color="gray"),
            xanchor="center",
        ),
    ],
)

# --- Output ---
output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "figures")
os.makedirs(output_dir, exist_ok=True)

html_path = os.path.join(output_dir, "fig_09_failure_modes.html")
png_path = os.path.join(output_dir, "fig_09_failure_modes.png")

pio.write_html(fig, html_path, config={"displaylogo": False})
pio.write_image(fig, png_path, width=1200, height=800, scale=2)

print(f"HTML: {html_path}")
print(f"PNG:  {png_path}")

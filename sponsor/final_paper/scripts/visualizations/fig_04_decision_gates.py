# =======================================================================
# Figure 4: Seven Decision Gate Lifecycle (Stepped Stage Diagram)
# Target: Section 2 (governance.tex), Table tab:decision-gates
# Visualization: Horizontal stepped bar chart
# Shows the seven lifecycle decision gates in chronological order.
# =======================================================================

import plotly.graph_objects as go
import plotly.io as pio
import os

# --- Hardcoded data from Table tab:decision-gates ---
gates = [
    "1. Candidate\nSelection",
    "2. IND/CTA\nGreen Light",
    "3. Dose Optimization\nReadiness",
    "4. Pivotal\nDesign Lock",
    "5. Pivotal Execution\nControl",
    "6. Database Lock\n& CSR",
    "7. Launch &\nPost-marketing",
]

agents = [
    "portfolio_agent",
    "regulatory_agent",
    "clinical_accountability_agent",
    "clinical_accountability_agent",
    "study_orchestrator",
    "data_biostats_agent",
    "regulatory_agent",
]

kpi_thresholds = [
    "P(technical success) >= 40%",
    "Dossier completeness >= 95%\nAssay validated",
    "Therapeutic index\nwithin target range",
    "Regulatory alignment\nconfirmed",
    "Enrollment >= 80% target\nQTLs within limits",
    "Query resolution >= 99%\nNo open critical queries",
    "Submission package validated\nDisclosure timeline confirmed",
]

# Color gradient from light to dark (progression)
n = len(gates)
base_r, base_g, base_b = 66, 133, 180  # Steel blue base
colors = []
for i in range(n):
    factor = 0.4 + 0.6 * (i / (n - 1))
    r = int(base_r * factor)
    g = int(base_g * factor)
    b = int(base_b * factor)
    colors.append(f"rgb({r},{g},{b})")

fig = go.Figure()

# Equal-width stepped bars
bar_width = 0.7
for i in range(n):
    fig.add_trace(go.Bar(
        x=[1],
        y=[gates[i]],
        orientation="h",
        marker_color=colors[i],
        showlegend=False,
        width=bar_width,
    ))

    # Agent owner annotation (inside bar)
    fig.add_annotation(
        x=0.5,
        y=gates[i],
        text=f"<b>{agents[i]}</b>",
        showarrow=False,
        font=dict(family="Times New Roman, serif", size=10, color="white"),
        xanchor="center",
    )

    # KPI threshold annotation (to the right)
    fig.add_annotation(
        x=1.05,
        y=gates[i],
        text=kpi_thresholds[i],
        showarrow=False,
        font=dict(family="Times New Roman, serif", size=9, color="black"),
        xanchor="left",
        align="left",
    )

# Add flow arrow annotation
fig.add_annotation(
    x=0.5, y=-0.08,
    xref="paper", yref="paper",
    text="\u2190 Early Development \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500 Lifecycle Progression \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500 Market Access \u2192",
    showarrow=False,
    font=dict(family="Times New Roman, serif", size=12, color="black"),
    xanchor="center",
)

fig.update_layout(
    title=dict(
        text="Oncology Drug Development Lifecycle: Seven Autonomous Decision Gates",
        font=dict(family="Times New Roman, serif", size=17, color="black"),
        x=0.5,
    ),
    xaxis=dict(
        showticklabels=False,
        showgrid=False,
        zeroline=False,
        range=[0, 2.5],
    ),
    yaxis=dict(
        tickfont=dict(family="Times New Roman, serif", size=11, color="black"),
        autorange="reversed",
        showgrid=False,
    ),
    font=dict(family="Times New Roman, serif", size=14, color="black"),
    plot_bgcolor="white",
    paper_bgcolor="white",
    margin=dict(l=160, r=250, t=100, b=80),
    barmode="stack",
)

# --- Output ---
output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "figures")
os.makedirs(output_dir, exist_ok=True)

html_path = os.path.join(output_dir, "fig_04_decision_gates.html")
png_path = os.path.join(output_dir, "fig_04_decision_gates.png")

pio.write_html(fig, html_path, config={"displaylogo": False})
pio.write_image(fig, png_path, width=1200, height=800, scale=2)

print(f"HTML: {html_path}")
print(f"PNG:  {png_path}")

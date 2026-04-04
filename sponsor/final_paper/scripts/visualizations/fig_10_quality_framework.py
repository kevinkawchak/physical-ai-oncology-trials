# =======================================================================
# Figure 10: QTL/KRI Quality Framework Dashboard (Multi-Panel Bullet Charts)
# Target: Section 7 (quality_compliance.tex), Table tab:qtl-kri
# Visualization: Multi-panel indicator chart (bullet-chart-style)
# Seven sub-panels showing threshold zones for each quality domain.
# These are threshold DEFINITIONS only, not measured performance data.
# =======================================================================

import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import os

# --- Hardcoded data from Table tab:qtl-kri ---
# Each domain: (name, kri_metric, thresholds_description, zones)
# zones: list of (start, end, color, label) for the bullet chart

domains = [
    {
        "name": "1. Informed Consent",
        "kri": "Consent compliance rate by site",
        "zones": [
            (0, 99.9, "#F44336", "Investigation (<100%)"),
            (99.9, 100, "#4CAF50", "Compliant (100%)"),
        ],
        "ticks": [0, 50, 99.9, 100],
        "tick_labels": ["0%", "50%", "<100%\ntrigger", "100%"],
        "range": [0, 100],
        "unit": "%",
    },
    {
        "name": "2. SAE Reporting Timeliness",
        "kri": "Median hours from awareness to initial report",
        "zones": [
            (0, 12, "#4CAF50", "Normal (\u226412h)"),
            (12, 24, "#FFC107", "Warning (>12h)"),
            (24, 36, "#F44336", "Action (>24h)"),
        ],
        "ticks": [0, 12, 24, 36],
        "tick_labels": ["0h", "12h\nwarn", "24h\naction", "36h"],
        "range": [0, 36],
        "unit": "hours",
    },
    {
        "name": "3. Protocol Deviation Rate",
        "kri": "Major deviation rate per 100 subject-visits",
        "zones": [
            (0, 3, "#4CAF50", "Normal (\u22643%)"),
            (3, 5, "#FFC107", "Warning (>3%)"),
            (5, 8, "#F44336", "Action (>5%)"),
        ],
        "ticks": [0, 3, 5, 8],
        "tick_labels": ["0%", "3%\nwarn", "5%\naction", "8%"],
        "range": [0, 8],
        "unit": "%",
    },
    {
        "name": "4. Data Entry Timeliness",
        "kri": "Median days from visit to CRF completion",
        "zones": [
            (0, 5, "#4CAF50", "Normal (\u22645d)"),
            (5, 10, "#FFC107", "Warning (>5d)"),
            (10, 15, "#F44336", "Action (>10d)"),
        ],
        "ticks": [0, 5, 10, 15],
        "tick_labels": ["0d", "5d\nwarn", "10d\naction", "15d"],
        "range": [0, 15],
        "unit": "days",
    },
    {
        "name": "5. Query Resolution Cycle",
        "kri": "Median days from query opening to resolution",
        "zones": [
            (0, 7, "#4CAF50", "Normal (\u22647d)"),
            (7, 14, "#FFC107", "Warning (>7d)"),
            (14, 21, "#F44336", "Action (>14d)"),
        ],
        "ticks": [0, 7, 14, 21],
        "tick_labels": ["0d", "7d\nwarn", "14d\naction", "21d"],
        "range": [0, 21],
        "unit": "days",
    },
    {
        "name": "6. Monitoring Visit Completion",
        "kri": "% of planned visits completed on schedule",
        "zones": [
            (0, 80, "#F44336", "Action (<80%)"),
            (80, 90, "#FFC107", "Warning (<90%)"),
            (90, 100, "#4CAF50", "Normal (\u226590%)"),
        ],
        "ticks": [0, 80, 90, 100],
        "tick_labels": ["0%", "80%\naction", "90%\nwarn", "100%"],
        "range": [0, 100],
        "unit": "%",
    },
    {
        "name": "7. IMP Accountability",
        "kri": "% of IMP with complete chain of custody",
        "zones": [
            (0, 99.9, "#F44336", "Investigation (<100%)"),
            (99.9, 100, "#4CAF50", "Compliant (100%)"),
        ],
        "ticks": [0, 50, 99.9, 100],
        "tick_labels": ["0%", "50%", "<100%\ntrigger", "100%"],
        "range": [0, 100],
        "unit": "%",
    },
]

# 4 rows x 2 cols grid (last cell empty)
fig = make_subplots(
    rows=4, cols=2,
    subplot_titles=[d["name"] for d in domains] + [""],
    horizontal_spacing=0.12,
    vertical_spacing=0.10,
)

for idx, domain in enumerate(domains):
    row = idx // 2 + 1
    col = idx % 2 + 1

    # Draw zone bars (stacked horizontal)
    for zone_start, zone_end, color, label in domain["zones"]:
        fig.add_trace(
            go.Bar(
                x=[zone_end - zone_start],
                y=[domain["name"]],
                base=[zone_start],
                orientation="h",
                marker_color=color,
                marker_line=dict(color="white", width=0.5),
                showlegend=False,
                hovertext=label,
                hoverinfo="text",
                width=0.6,
            ),
            row=row, col=col,
        )

    # Update axes for this subplot
    fig.update_xaxes(
        range=domain["range"],
        tickvals=domain["ticks"],
        ticktext=domain["tick_labels"],
        tickfont=dict(family="Times New Roman, serif", size=9, color="black"),
        gridcolor="lightgray",
        row=row, col=col,
    )
    fig.update_yaxes(
        showticklabels=False,
        row=row, col=col,
    )

    # KRI metric annotation below each subplot
    fig.add_annotation(
        text=f"KRI: {domain['kri']}",
        xref=f"x{idx + 1 if idx > 0 else ''} domain",
        yref=f"y{idx + 1 if idx > 0 else ''} domain",
        x=0.5, y=-0.5,
        showarrow=False,
        font=dict(family="Times New Roman, serif", size=8, color="gray"),
        xanchor="center",
    )

# Update subplot titles font
for annotation in fig.layout.annotations:
    annotation.font = dict(family="Times New Roman, serif", size=12, color="black")

fig.update_layout(
    title=dict(
        text="Risk-Based Quality Management: QTL/KRI Threshold Framework",
        font=dict(family="Times New Roman, serif", size=17, color="black"),
        x=0.5,
        y=0.98,
    ),
    font=dict(family="Times New Roman, serif", size=14, color="black"),
    plot_bgcolor="white",
    paper_bgcolor="white",
    margin=dict(l=80, r=40, t=100, b=100),
    height=900,
    width=1200,
    barmode="stack",
)

# Legend for zones (manual)
for label_text, color, x_pos in [
    ("Green: Within tolerance", "#4CAF50", 0.15),
    ("Yellow: Warning zone", "#FFC107", 0.45),
    ("Red: Action zone", "#F44336", 0.72),
]:
    fig.add_annotation(
        x=x_pos, y=-0.06,
        xref="paper", yref="paper",
        text=f'<span style="color:{color}">\u25A0</span> {label_text}',
        showarrow=False,
        font=dict(family="Times New Roman, serif", size=11, color="black"),
        xanchor="center",
    )

fig.add_annotation(
    text="Threshold definitions only \u2014 no actual performance data plotted. "
         "From Table tab:qtl-kri (quality_compliance.tex).",
    xref="paper", yref="paper",
    x=0.5, y=-0.10,
    showarrow=False,
    font=dict(family="Times New Roman, serif", size=9, color="gray"),
    xanchor="center",
)

# --- Output ---
output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "figures")
os.makedirs(output_dir, exist_ok=True)

html_path = os.path.join(output_dir, "fig_10_quality_framework.html")
png_path = os.path.join(output_dir, "fig_10_quality_framework.png")

pio.write_html(fig, html_path, config={"displaylogo": False})
pio.write_image(fig, png_path, width=1200, height=900, scale=2)

print(f"HTML: {html_path}")
print(f"PNG:  {png_path}")

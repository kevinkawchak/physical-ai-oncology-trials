# =======================================================================
# Figure 3: Twelve-Agent Architecture Layer Diagram (Sunburst)
# Target: Section 2 (governance.tex), Table tab:agent-mapping
# Visualization: Sunburst chart
# Shows the four-layer autonomous sponsor architecture with twelve agents.
# =======================================================================

import plotly.graph_objects as go
import plotly.io as pio
import os

# --- Hardcoded data from Table tab:agent-mapping ---
# Structure: Center -> Layer -> Agent (with traditional role)
# Using "remainder" branchvalues so parent values are computed from children

labels = [
    "Autonomous Sponsor",
    # Layers (ring 1)
    "Governance Layer",
    "Execution Layer",
    "Site/Robotics Layer",
    "Trust Layer",
    # Agents (ring 2) - Governance
    "portfolio_agent — VP R&D",
    "asset_lead_agent — Clinical Dev Head",
    # Agents - Execution
    "clinical_accountability_agent — Study Physician",
    "study_orchestrator — Cross-functional Team",
    "clinops_agent — Clinical Ops Director",
    "safety_agent — Safety/PV Team",
    "regulatory_agent — Regulatory Affairs Dir",
    "quality_agent — QA Director",
    "supply_agent — CMC/Supply Lead",
    "data_biostats_agent — Data Mgmt/Biostats",
    # Agents - Site/Robotics
    "site_gateway — Site Relationship Mgr",
    "robot_execution_gateway — (No traditional equiv.)",
    # Trust Layer infrastructure
    "Infrastructure (Identity, Audit, Policy)",
]

parents = [
    "",
    # Layers -> Center
    "Autonomous Sponsor",
    "Autonomous Sponsor",
    "Autonomous Sponsor",
    "Autonomous Sponsor",
    # Governance agents -> Governance Layer
    "Governance Layer",
    "Governance Layer",
    # Execution agents -> Execution Layer
    "Execution Layer",
    "Execution Layer",
    "Execution Layer",
    "Execution Layer",
    "Execution Layer",
    "Execution Layer",
    "Execution Layer",
    "Execution Layer",
    # Site/Robotics agents -> Site/Robotics Layer
    "Site/Robotics Layer",
    "Site/Robotics Layer",
    # Trust Layer infrastructure -> Trust Layer
    "Trust Layer",
]

# All leaf nodes get value=1; parent nodes get value=0 with remainder branchvalues
values = [
    0,     # Center (auto-summed)
    0, 0, 0, 0,  # Layers (auto-summed)
    1, 1,              # Governance agents (2)
    1, 1, 1, 1, 1, 1, 1, 1,  # Execution agents (8)
    1, 1,              # Site/Robotics agents (2)
    1,                 # Trust infrastructure (1)
]

# Color families per layer
colors = [
    "#E8E8E8",          # Center - light gray
    # Layers
    "#4472C4",          # Governance - blue
    "#548235",          # Execution - green
    "#ED7D31",          # Site/Robotics - orange
    "#7B68A0",          # Trust - purple
    # Governance agents - lighter blues
    "#6B9BD2", "#8BB8E8",
    # Execution agents - greens
    "#6BA34A", "#7FB860", "#93C97A", "#A7D490",
    "#78AB5A", "#8CC070", "#68A040", "#84B668",
    # Site/Robotics agents - oranges
    "#F4A460", "#F5B77A",
    # Trust infrastructure - lighter purple
    "#9B88B8",
]

fig = go.Figure(go.Sunburst(
    labels=labels,
    parents=parents,
    values=values,
    branchvalues="remainder",
    marker=dict(colors=colors),
    textfont=dict(family="Times New Roman, serif", size=10, color="black"),
    insidetextorientation="radial",
    hoverinfo="label",
))

fig.update_layout(
    title=dict(
        text="Autonomous Sponsor: Four-Layer, Twelve-Agent Architecture",
        font=dict(family="Times New Roman, serif", size=18, color="black"),
        x=0.5,
    ),
    font=dict(family="Times New Roman, serif", size=14, color="black"),
    plot_bgcolor="white",
    paper_bgcolor="white",
    margin=dict(l=40, r=40, t=100, b=40),
    width=1200,
    height=800,
)

# --- Output ---
output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "figures")
os.makedirs(output_dir, exist_ok=True)

html_path = os.path.join(output_dir, "fig_03_agent_architecture.html")
png_path = os.path.join(output_dir, "fig_03_agent_architecture.png")

pio.write_html(fig, html_path, config={"displaylogo": False})
pio.write_image(fig, png_path, width=1200, height=800, scale=2)

print(f"HTML: {html_path}")
print(f"PNG:  {png_path}")

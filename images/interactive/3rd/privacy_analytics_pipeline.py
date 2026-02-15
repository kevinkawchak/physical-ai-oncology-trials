"""Privacy-Preserving Analytics Pipeline: Step-by-Step Process.

Horizontal pipeline diagram showing the complete data privacy workflow
from raw clinical data to publishable aggregate results.

Data sources:
  - privacy/phi-pii-management/phi_detector.py — PHI detection (18 identifiers)
  - privacy/de-identification/deidentification_pipeline.py — Safe Harbor, Expert Determination
  - federation/differential_privacy.py — epsilon-delta noise mechanisms
  - federation/secure_aggregation.py — simulated secure MPC
  - federation/privacy_analytics.py — KM survival, Cox PH, response rates
  - privacy/access-control/access_control_manager.py — RBAC + 21 CFR Part 11
  - privacy/breach-response/breach_response_protocol.py — incident detection
"""

import os

import plotly.graph_objects as go

STAGES = [
    {
        "label": "Raw Clinical\nData",
        "detail": "DICOM, FHIR R4,\nHL7 v2",
        "guarantee": "None\n(PHI present)",
    },
    {
        "label": "PHI\nDetection",
        "detail": "18 HIPAA\nidentifiers scanned",
        "guarantee": "Confidence:\n0.65–0.98",
    },
    {
        "label": "De-\nIdentification",
        "detail": "Safe Harbor or\nExpert Determination",
        "guarantee": "Re-ID risk\n< threshold",
    },
    {
        "label": "Differential\nPrivacy",
        "detail": "ε-δ noise\ninjection",
        "guarantee": "Formal ε-δ\nguarantee",
    },
    {
        "label": "Secure\nAggregation",
        "detail": "Multi-party\ncomputation",
        "guarantee": "No single party\nsees raw data",
    },
    {
        "label": "Privacy\nAnalytics",
        "detail": "KM, Cox PH,\nResponse Rates",
        "guarantee": "Publishable\naggregate results",
    },
]


def create_chart(dark_mode=False):
    """Create privacy-preserving analytics pipeline diagram."""
    if dark_mode:
        text_color = "#e0e0e0"
        bg_color = "#1a1a2e"
        arrow_color = "#ffffff"
        audit_color = "#f39c12"
        shield_color = "#76ff03"
        stage_colors = ["#1a3a5c", "#1a4a6c", "#1a5a7c", "#1a6a8c", "#1a7a9c", "#1a8aac"]
        box_border = "#5dade2"
    else:
        text_color = "#1a1a2e"
        bg_color = "#ffffff"
        arrow_color = "#95a5a6"
        audit_color = "#d35400"
        shield_color = "#27ae60"
        stage_colors = ["#d6eaf8", "#aed6f1", "#85c1e9", "#5dade2", "#3498db", "#2980b9"]
        box_border = "#2c3e50"

    fig = go.Figure()

    n = len(STAGES)
    box_w = 0.12
    gap = 0.02
    total_w = n * box_w + (n - 1) * gap
    start_x = (1.0 - total_w) / 2

    for i, stage in enumerate(STAGES):
        x_center = start_x + i * (box_w + gap) + box_w / 2
        y_center = 0.55
        box_h = 0.30

        # Stage box
        fig.add_shape(
            type="rect",
            x0=x_center - box_w / 2,
            y0=y_center - box_h / 2,
            x1=x_center + box_w / 2,
            y1=y_center + box_h / 2,
            fillcolor=stage_colors[i],
            line=dict(color=box_border, width=2),
            xref="paper",
            yref="paper",
        )

        # Stage label (top of box)
        label = stage["label"].replace("\n", "<br>")
        fig.add_annotation(
            x=x_center,
            y=y_center + 0.09,
            text=f"<b>{label}</b>",
            showarrow=False,
            font=dict(size=11, color="#ffffff" if dark_mode else text_color),
            xref="paper",
            yref="paper",
        )

        # Detail (middle)
        detail = stage["detail"].replace("\n", "<br>")
        fig.add_annotation(
            x=x_center,
            y=y_center - 0.01,
            text=detail,
            showarrow=False,
            font=dict(size=9, color="#b0bec5" if dark_mode else "#555555"),
            xref="paper",
            yref="paper",
        )

        # Privacy guarantee (below box)
        guarantee = stage["guarantee"].replace("\n", "<br>")
        fig.add_annotation(
            x=x_center,
            y=y_center - box_h / 2 - 0.06,
            text=f"🛡 {guarantee}",
            showarrow=False,
            font=dict(size=9, color=shield_color),
            xref="paper",
            yref="paper",
        )

        # Stage number
        fig.add_annotation(
            x=x_center,
            y=y_center + box_h / 2 + 0.04,
            text=f"<b>Stage {i + 1}</b>",
            showarrow=False,
            font=dict(size=10, color=text_color),
            xref="paper",
            yref="paper",
        )

        # Arrow between stages (line shape + triangle marker)
        if i < n - 1:
            x_arrow_start = x_center + box_w / 2 + 0.002
            x_arrow_end = x_center + box_w / 2 + gap - 0.002
            fig.add_shape(
                type="line",
                x0=x_arrow_start,
                y0=y_center,
                x1=x_arrow_end,
                y1=y_center,
                line=dict(color=arrow_color, width=2),
                xref="paper",
                yref="paper",
            )
            fig.add_annotation(
                x=x_arrow_end,
                y=y_center,
                text="▶",
                showarrow=False,
                font=dict(size=12, color=arrow_color),
                xref="paper",
                yref="paper",
            )

    # 21 CFR Part 11 Audit Trail bar at bottom
    audit_y = 0.15
    fig.add_shape(
        type="rect",
        x0=start_x,
        y0=audit_y - 0.02,
        x1=start_x + total_w,
        y1=audit_y + 0.02,
        fillcolor=audit_color,
        line=dict(color=audit_color, width=1),
        xref="paper",
        yref="paper",
        opacity=0.8,
    )
    fig.add_annotation(
        x=0.5,
        y=audit_y,
        text="<b>21 CFR Part 11 Audit Trail — RBAC Access Control — Breach Response Protocol</b>",
        showarrow=False,
        font=dict(size=12, color="#ffffff"),
        xref="paper",
        yref="paper",
    )

    # Source modules listed at bottom
    sources = [
        "phi_detector.py",
        "deidentification_pipeline.py",
        "differential_privacy.py",
        "secure_aggregation.py",
        "privacy_analytics.py",
        "access_control_manager.py",
    ]
    for i, src in enumerate(sources):
        x_center = start_x + i * (box_w + gap) + box_w / 2
        fig.add_annotation(
            x=x_center,
            y=0.05,
            text=f"<i>{src}</i>",
            showarrow=False,
            font=dict(size=8, color="#7f8c8d"),
            xref="paper",
            yref="paper",
        )

    fig.update_layout(
        title=dict(
            text=(
                "Privacy-Preserving Analytics Pipeline:"
                " From Raw Clinical Data to Publishable Results"
                "<br><sup>Sources: privacy/, federation/ modules"
                " — 45 CFR 164.514, 21 CFR Part 11</sup>"
            ),
            font=dict(size=18, color=text_color),
            x=0.5,
        ),
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        xaxis=dict(visible=False, range=[0, 1]),
        yaxis=dict(visible=False, range=[0, 1]),
        margin=dict(t=120, l=30, r=30, b=60),
        width=1920,
        height=1080,
    )

    return fig


def main():
    """Generate privacy analytics pipeline visualizations."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    png_dir = os.path.join(os.path.dirname(os.path.dirname(script_dir)), "png", "3rd")
    os.makedirs(png_dir, exist_ok=True)

    for dark_mode in [False, True]:
        suffix = "dark" if dark_mode else "light"
        fig = create_chart(dark_mode=dark_mode)

        html_path = os.path.join(script_dir, f"privacy_analytics_pipeline_{suffix}.html")
        fig.write_html(html_path, include_plotlyjs=True)

        png_path = os.path.join(png_dir, f"privacy_analytics_pipeline_{suffix}.png")
        fig.write_image(png_path, width=1920, height=1080, scale=2)

    print("privacy_analytics_pipeline: done")


if __name__ == "__main__":
    main()

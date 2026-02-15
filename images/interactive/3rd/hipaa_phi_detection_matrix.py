"""HIPAA PHI Detection Confidence & Risk Matrix.

Annotated heatmap showing detection confidence and risk stratification
for all 18 HIPAA identifier types.

Data sources:
  - privacy/phi-pii-management/phi_detector.py — HIPAA_IDENTIFIERS dict,
    _calculate_confidence(), _stratify_risk() with Critical/High/Medium/Low
"""

import os

import plotly.graph_objects as go

IDENTIFIERS = [
    ("NAME", 0.92, "Critical"),
    ("GEOGRAPHIC_DATA", 0.85, "High"),
    ("DATES", 0.80, "High"),
    ("PHONE_NUMBERS", 0.85, "High"),
    ("FAX_NUMBERS", 0.82, "Medium"),
    ("EMAIL_ADDRESSES", 0.95, "High"),
    ("SOCIAL_SECURITY_NUMBERS", 0.98, "Critical"),
    ("MEDICAL_RECORD_NUMBERS", 0.95, "Critical"),
    ("HEALTH_PLAN_BENEFICIARY", 0.90, "High"),
    ("ACCOUNT_NUMBERS", 0.90, "Medium"),
    ("CERTIFICATE_LICENSE", 0.78, "Medium"),
    ("VEHICLE_IDENTIFIERS", 0.75, "Low"),
    ("DEVICE_IDENTIFIERS", 0.80, "Medium"),
    ("WEB_URLS", 0.88, "Medium"),
    ("IP_ADDRESSES", 0.85, "Medium"),
    ("BIOMETRIC_IDENTIFIERS", 0.70, "High"),
    ("FULL_FACE_PHOTOGRAPHS", 0.65, "High"),
    ("UNIQUE_IDENTIFYING_CODES", 0.72, "Medium"),
]

RISK_LEVELS = {"Critical": 4, "High": 3, "Medium": 2, "Low": 1}


def _confidence_color(val, dark_mode=False):
    """Map confidence value to a color on a green-yellow-red scale."""
    if dark_mode:
        if val >= 0.90:
            return "#00897b"
        elif val >= 0.80:
            return "#ffa000"
        elif val >= 0.70:
            return "#e65100"
        else:
            return "#b71c1c"
    else:
        if val >= 0.90:
            return "#a5d6a7"
        elif val >= 0.80:
            return "#fff59d"
        elif val >= 0.70:
            return "#ffcc80"
        else:
            return "#ef9a9a"


def create_chart(dark_mode=False):
    """Create HIPAA PHI detection confidence and risk heatmap."""
    if dark_mode:
        text_color = "#e0e0e0"
        bg_color = "#1a1a2e"
        header_bg = "#34495e"
        cell_text = "#ffffff"
        line_color = "#4a4a5a"
        critical_color = "#b71c1c"
        high_color = "#e65100"
        medium_color = "#f57f17"
        low_color = "#1b5e20"
    else:
        text_color = "#1a1a2e"
        bg_color = "#ffffff"
        header_bg = "#2c3e50"
        cell_text = "#1a1a2e"
        line_color = "#bdc3c7"
        critical_color = "#e74c3c"
        high_color = "#e67e22"
        medium_color = "#f1c40f"
        low_color = "#27ae60"

    risk_colors = {"Critical": critical_color, "High": high_color, "Medium": medium_color, "Low": low_color}

    names = [ident[0] for ident in IDENTIFIERS]
    confidences = [f"{ident[1]:.2f}" for ident in IDENTIFIERS]
    risk_cats = [ident[2] for ident in IDENTIFIERS]
    risk_nums = [str(RISK_LEVELS[ident[2]]) for ident in IDENTIFIERS]

    # Color coding per cell
    conf_colors = [_confidence_color(ident[1], dark_mode) for ident in IDENTIFIERS]
    risk_cell_colors = [risk_colors[ident[2]] for ident in IDENTIFIERS]
    default_bg = "#2c2c3e" if dark_mode else "#f8f9fa"

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=[
                        "<b>HIPAA Identifier</b>",
                        "<b>Detection Confidence</b>",
                        "<b>Risk Level (1-4)</b>",
                        "<b>Risk Category</b>",
                    ],
                    fill_color=header_bg,
                    font=dict(size=14, color="#ffffff"),
                    align="center",
                    line_color=line_color,
                    height=45,
                ),
                cells=dict(
                    values=[names, confidences, risk_nums, risk_cats],
                    fill_color=[
                        [default_bg] * len(names),
                        conf_colors,
                        risk_cell_colors,
                        risk_cell_colors,
                    ],
                    font=dict(size=12, color=cell_text),
                    align="center",
                    line_color=line_color,
                    height=32,
                ),
                columnwidth=[0.35, 0.22, 0.18, 0.25],
            )
        ]
    )

    # Escalation annotation
    fig.add_annotation(
        x=0.5,
        xref="paper",
        y=-0.06,
        yref="paper",
        text=(
            "<b>Critical Risk = Immediate escalation</b>"
            " (Name+MRN combo, SSN detected)"
            " | Confidence range: 0.65–0.98"
            " | Risk levels: Critical (4), High (3), Medium (2), Low (1)"
        ),
        showarrow=False,
        font=dict(size=11, color=text_color),
    )

    fig.update_layout(
        title=dict(
            text=(
                "HIPAA PHI Detection: 18 Identifier Types — Confidence & Risk Stratification"
                "<br><sup>Source: privacy/phi-pii-management/phi_detector.py"
                " — HIPAA_IDENTIFIERS, _calculate_confidence(), _stratify_risk()</sup>"
            ),
            font=dict(size=18, color=text_color),
            x=0.5,
        ),
        paper_bgcolor=bg_color,
        margin=dict(t=120, l=30, r=30, b=100),
        width=1920,
        height=1080,
    )

    return fig


def main():
    """Generate HIPAA PHI detection matrix visualizations."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    png_dir = os.path.join(os.path.dirname(os.path.dirname(script_dir)), "png", "3rd")
    os.makedirs(png_dir, exist_ok=True)

    for dark_mode in [False, True]:
        suffix = "dark" if dark_mode else "light"
        fig = create_chart(dark_mode=dark_mode)

        html_path = os.path.join(script_dir, f"hipaa_phi_detection_matrix_{suffix}.html")
        fig.write_html(html_path, include_plotlyjs=True)

        png_path = os.path.join(png_dir, f"hipaa_phi_detection_matrix_{suffix}.png")
        fig.write_image(png_path, width=1920, height=1080, scale=2)

    print("hipaa_phi_detection_matrix: done")


if __name__ == "__main__":
    main()

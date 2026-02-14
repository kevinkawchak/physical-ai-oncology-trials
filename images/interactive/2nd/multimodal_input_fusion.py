"""Multi-Modal Input Fusion: Accuracy and Latency by Modality Combination.

Bar chart with secondary axis scatter for latency, showing how combining
input modalities improves command accuracy while increasing latency.

Data source: agentic-ai/results.md — Multi-Modal Input Fusion
"""

import os

import plotly.graph_objects as go
from plotly.subplots import make_subplots


MODALITIES = ["Speech Only", "Vision Only", "Speech +\nVision", "Speech + Vision\n+ Context"]
ACCURACY = [92.3, 89.7, 96.5, 98.1]
LATENCY = [200, 150, 280, 350]

CLINICAL_THRESHOLD = 95.0


def create_fusion_chart(dark_mode=False):
    """Create bar chart with secondary axis scatter for latency."""
    if dark_mode:
        template = "plotly_dark"
        acc_color = "#1e90ff"
        lat_color = "#ffd700"
        thresh_color = "#ef5350"
        text_color = "#e0e0e0"
        bg_color = "#1a1a2e"
        best_color = "#76ff03"
    else:
        template = "plotly_white"
        acc_color = "#4682b4"
        lat_color = "#ff6347"
        thresh_color = "#d32f2f"
        text_color = "#1a1a2e"
        bg_color = "#ffffff"
        best_color = "#2e7d32"

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Accuracy bars
    fig.add_trace(
        go.Bar(
            name="Accuracy (%)",
            x=MODALITIES,
            y=ACCURACY,
            marker_color=acc_color,
            text=[f"{a}%" for a in ACCURACY],
            textposition="outside",
            textfont=dict(size=13),
        ),
        secondary_y=False,
    )

    # Latency scatter on secondary y-axis
    fig.add_trace(
        go.Scatter(
            name="Latency (ms)",
            x=MODALITIES,
            y=LATENCY,
            mode="lines+markers",
            line=dict(color=lat_color, width=3),
            marker=dict(size=12, symbol="circle", color=lat_color),
            text=[f"{l}ms" for l in LATENCY],
            textposition="top center",
            textfont=dict(size=11, color=lat_color),
        ),
        secondary_y=True,
    )

    # Clinical acceptance threshold
    fig.add_hline(
        y=CLINICAL_THRESHOLD,
        line_dash="dash",
        line_color=thresh_color,
        line_width=2,
        annotation_text="Clinical Acceptance Threshold (95%)",
        annotation_position="top left",
        annotation_font=dict(size=12, color=thresh_color),
        secondary_y=False,
    )

    # Best accuracy annotation
    fig.add_annotation(
        x="Speech + Vision\n+ Context",
        y=98.1 + 1.5,
        text="<b>Best: 98.1% accuracy</b>",
        showarrow=True,
        arrowhead=2,
        font=dict(size=13, color=best_color),
        bgcolor="rgba(0,0,0,0.6)" if dark_mode else "rgba(255,255,255,0.9)",
        bordercolor=best_color,
        borderwidth=1,
    )

    fig.update_layout(
        title=dict(
            text=(
                "Multi-Modal Input Fusion: Accuracy and Latency by Modality Combination"
                "<br><sup>Source: agentic-ai/results.md — Multi-Modal Input Fusion</sup>"
            ),
            font=dict(size=18, color=text_color),
            x=0.5,
        ),
        template=template,
        paper_bgcolor=bg_color,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=13),
        ),
        margin=dict(t=120, l=60, r=60, b=100),
        width=1920,
        height=1080,
    )

    fig.update_yaxes(
        title_text="Accuracy (%)",
        range=[85, 102],
        secondary_y=False,
    )
    fig.update_yaxes(
        title_text="Latency (ms)",
        range=[100, 420],
        secondary_y=True,
    )
    fig.update_xaxes(title_text="Input Modality")

    return fig


def main():
    """Generate multi-modal fusion visualizations."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    png_dir = os.path.join(os.path.dirname(os.path.dirname(script_dir)), "png", "2nd")
    os.makedirs(png_dir, exist_ok=True)

    for dark_mode in [False, True]:
        suffix = "dark" if dark_mode else "light"
        fig = create_fusion_chart(dark_mode=dark_mode)

        html_path = os.path.join(script_dir, f"multimodal_input_fusion_{suffix}.html")
        fig.write_html(html_path, include_plotlyjs=True)

        png_path = os.path.join(png_dir, f"multimodal_input_fusion_{suffix}.png")
        fig.write_image(png_path, width=1920, height=1080, scale=2)

    print("multimodal_input_fusion: done")


if __name__ == "__main__":
    main()

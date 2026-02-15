"""Multi-Site Clinical Trial Enrollment & Quality Dashboard.

Heatmap table with color-coded status cells showing enrollment and data
quality metrics for each trial site with risk-based monitoring thresholds.

Data sources:
  - tools/trial-site-monitor/trial_site_monitor.py — DEFAULT_THRESHOLDS,
    SiteMetrics dataclass, status classification (Green/Yellow/Red)
"""

import os

import plotly.graph_objects as go

HEADERS = [
    "Site ID",
    "Institution",
    "Screened",
    "Enrolled",
    "Failure %",
    "Monthly Enroll",
    "Data Complete %",
    "Query Rate",
    "Deviations",
    "Dev Rate",
    "AE Delay (d)",
    "Days Gap",
    "Flags",
    "Status",
]

SITES = [
    ["SITE-001", "Hospital Alpha", "45", "15", "67%", "2.1", "92%", "3.2", "1", "0.07", "2.1", "8", "0", "GREEN"],
    ["SITE-002", "Hospital Beta", "28", "5", "82%", "0.7", "87%", "5.4", "2", "0.40", "3.5", "35", "5", "RED"],
    ["SITE-003", "Hospital Gamma", "38", "12", "68%", "1.5", "94%", "2.8", "1", "0.08", "1.5", "12", "0", "GREEN"],
    ["SITE-004", "Hospital Delta", "32", "8", "75%", "0.9", "89%", "4.8", "1", "0.13", "2.8", "22", "2", "YELLOW"],
]

THRESHOLDS = {
    "min_screening_ratio": 0.30,
    "max_screen_failure_rate": 0.70,
    "max_protocol_deviation_rate": 0.10,
    "min_data_completeness": 90.0,
    "max_query_rate": 5.0,
    "max_enrollment_gap_days": 30,
    "min_monthly_enrollment": 1.0,
    "max_ae_reporting_delay_days": 3,
}

STATUS_RULES = "Green = 0 flags | Yellow = 1–2 flags | Red = ≥3 flags"


def _cell_colors(sites, dark_mode=False):
    """Generate per-cell background colors based on threshold violations."""
    green = "#27ae60" if dark_mode else "#d5f5e3"
    yellow = "#e67e22" if dark_mode else "#fdebd0"
    red = "#c0392b" if dark_mode else "#fadbd8"
    default = "#2c2c3e" if dark_mode else "#ffffff"
    status_map = {"GREEN": green, "YELLOW": yellow, "RED": red}

    # Column indices that have thresholds (0-indexed): 4=Fail%, 5=Monthly, 6=Complete%,
    # 7=Query, 9=DevRate, 10=AEDelay, 11=Gap
    all_colors = []
    for col_idx in range(len(HEADERS)):
        col_colors = []
        for row in sites:
            if col_idx == 13:  # Status column
                col_colors.append(status_map.get(row[13], default))
            elif col_idx == 4:  # Failure %
                val = float(row[4].replace("%", "")) / 100
                col_colors.append(red if val > 0.70 else (yellow if val > 0.65 else default))
            elif col_idx == 5:  # Monthly enrollment
                val = float(row[5])
                col_colors.append(red if val < 1.0 else default)
            elif col_idx == 6:  # Data completeness
                val = float(row[6].replace("%", ""))
                col_colors.append(red if val < 90.0 else default)
            elif col_idx == 7:  # Query rate
                val = float(row[7])
                col_colors.append(red if val > 5.0 else (yellow if val > 4.0 else default))
            elif col_idx == 9:  # Deviation rate
                val = float(row[9])
                col_colors.append(red if val > 0.10 else default)
            elif col_idx == 10:  # AE delay
                val = float(row[10])
                col_colors.append(red if val > 3.0 else (yellow if val > 2.5 else default))
            elif col_idx == 11:  # Days gap
                val = int(row[11])
                col_colors.append(red if val > 30 else (yellow if val > 20 else default))
            else:
                col_colors.append(default)
        all_colors.append(col_colors)
    return all_colors


def create_chart(dark_mode=False):
    """Create trial monitoring dashboard table."""
    if dark_mode:
        text_color = "#e0e0e0"
        bg_color = "#1a1a2e"
        header_bg = "#34495e"
        header_font_color = "#ffffff"
        line_color = "#4a4a5a"
    else:
        text_color = "#1a1a2e"
        bg_color = "#ffffff"
        header_bg = "#2c3e50"
        header_font_color = "#ffffff"
        line_color = "#bdc3c7"

    cell_colors = _cell_colors(SITES, dark_mode)

    col_values = [[row[i] for row in SITES] for i in range(len(HEADERS))]

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=[f"<b>{h}</b>" for h in HEADERS],
                    fill_color=header_bg,
                    font=dict(size=13, color=header_font_color),
                    align="center",
                    line_color=line_color,
                    height=40,
                ),
                cells=dict(
                    values=col_values,
                    fill_color=cell_colors,
                    font=dict(size=12, color=text_color),
                    align="center",
                    line_color=line_color,
                    height=35,
                ),
            )
        ]
    )

    # Threshold reference annotation
    threshold_text = " | ".join([f"{k}: {v}" for k, v in THRESHOLDS.items()])
    fig.add_annotation(
        x=0.5,
        xref="paper",
        y=-0.08,
        yref="paper",
        text=f"<b>Thresholds:</b> {threshold_text}",
        showarrow=False,
        font=dict(size=10, color=text_color),
        align="center",
    )

    fig.add_annotation(
        x=0.5,
        xref="paper",
        y=-0.13,
        yref="paper",
        text=f"<b>Status Rules:</b> {STATUS_RULES}",
        showarrow=False,
        font=dict(size=11, color=text_color),
    )

    fig.update_layout(
        title=dict(
            text=(
                "Multi-Site Trial Monitoring Dashboard: Enrollment & Data Quality Metrics"
                "<br><sup>Source: tools/trial-site-monitor/trial_site_monitor.py"
                " — DEFAULT_THRESHOLDS & SiteMetrics</sup>"
            ),
            font=dict(size=18, color=text_color),
            x=0.5,
        ),
        paper_bgcolor=bg_color,
        margin=dict(t=120, l=30, r=30, b=120),
        width=1920,
        height=1080,
    )

    return fig


def main():
    """Generate multi-site trial dashboard visualizations."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    png_dir = os.path.join(os.path.dirname(os.path.dirname(script_dir)), "png", "3rd")
    os.makedirs(png_dir, exist_ok=True)

    for dark_mode in [False, True]:
        suffix = "dark" if dark_mode else "light"
        fig = create_chart(dark_mode=dark_mode)

        html_path = os.path.join(script_dir, f"multi_site_trial_dashboard_{suffix}.html")
        fig.write_html(html_path, include_plotlyjs=True)

        png_path = os.path.join(png_dir, f"multi_site_trial_dashboard_{suffix}.png")
        fig.write_image(png_path, width=1920, height=1080, scale=2)

    print("multi_site_trial_dashboard: done")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Trial Site Monitor: CLI for multi-site oncology trial enrollment tracking,
data quality monitoring, and protocol deviation detection.

Reads trial data from a JSON manifest and generates site performance reports.
Designed for CRO and sponsor-level oversight of physical AI oncology trials
across multiple institutions.

Usage:
    python trial_site_monitor.py enrollment <manifest.json>
    python trial_site_monitor.py quality <manifest.json>
    python trial_site_monitor.py deviations <manifest.json>
    python trial_site_monitor.py report <manifest.json> [--output report.json]
    python trial_site_monitor.py init-manifest --sites 5 [--output manifest.json]

Manifest format:
    See 'init-manifest' command for the expected JSON schema.

Note: All illustrative parameters should be validated against your
institution's trial protocols before clinical use.
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime

# ---------------------------------------------------------------------------
# Trial quality thresholds (configurable per protocol)
# ---------------------------------------------------------------------------
DEFAULT_THRESHOLDS = {
    "min_screening_to_enrollment_ratio": 0.30,
    "max_screen_failure_rate": 0.70,
    "max_protocol_deviation_rate_per_subject": 0.10,
    "min_data_completeness_pct": 90.0,
    "max_query_rate_per_subject": 5.0,
    "max_days_enrollment_gap": 30,
    "min_monthly_enrollment_rate": 1.0,
    "max_ae_reporting_delay_days": 3,
}

# Site status classification
SITE_STATUS = {
    "green": "On track",
    "yellow": "Needs attention",
    "red": "Intervention required",
}


@dataclass
class SiteMetrics:
    """Computed metrics for a single trial site."""

    site_id: str
    site_name: str
    status: str = "green"
    total_screened: int = 0
    total_enrolled: int = 0
    total_completed: int = 0
    total_withdrawn: int = 0
    screen_failure_rate: float = 0.0
    enrollment_rate_monthly: float = 0.0
    data_completeness_pct: float = 100.0
    query_rate_per_subject: float = 0.0
    protocol_deviations: int = 0
    deviation_rate_per_subject: float = 0.0
    mean_ae_reporting_delay_days: float = 0.0
    days_since_last_enrollment: int = 0
    flags: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "site_id": self.site_id,
            "site_name": self.site_name,
            "status": self.status,
            "total_screened": self.total_screened,
            "total_enrolled": self.total_enrolled,
            "total_completed": self.total_completed,
            "total_withdrawn": self.total_withdrawn,
            "screen_failure_rate": round(self.screen_failure_rate, 3),
            "enrollment_rate_monthly": round(self.enrollment_rate_monthly, 2),
            "data_completeness_pct": round(self.data_completeness_pct, 1),
            "query_rate_per_subject": round(self.query_rate_per_subject, 2),
            "protocol_deviations": self.protocol_deviations,
            "deviation_rate_per_subject": round(self.deviation_rate_per_subject, 3),
            "mean_ae_reporting_delay_days": round(self.mean_ae_reporting_delay_days, 1),
            "days_since_last_enrollment": self.days_since_last_enrollment,
            "flags": self.flags,
        }


def load_manifest(filepath: str) -> dict:
    """Load and validate a trial manifest JSON file."""
    if not os.path.isfile(filepath):
        print(f"ERROR: Manifest not found: {filepath}", file=sys.stderr)
        sys.exit(1)

    with open(filepath) as f:
        data = json.load(f)

    required_keys = ["trial_id", "trial_name", "sites"]
    for key in required_keys:
        if key not in data:
            print(f"ERROR: Manifest missing required key '{key}'", file=sys.stderr)
            sys.exit(1)

    return data


def compute_site_metrics(site: dict, thresholds: dict) -> SiteMetrics:
    """Compute quality metrics for a single site from manifest data."""
    metrics = SiteMetrics(
        site_id=site.get("site_id", "UNKNOWN"),
        site_name=site.get("site_name", "Unknown Site"),
    )

    enrollment = site.get("enrollment", {})
    metrics.total_screened = enrollment.get("screened", 0)
    metrics.total_enrolled = enrollment.get("enrolled", 0)
    metrics.total_completed = enrollment.get("completed", 0)
    metrics.total_withdrawn = enrollment.get("withdrawn", 0)

    # Screen failure rate
    if metrics.total_screened > 0:
        metrics.screen_failure_rate = 1.0 - (metrics.total_enrolled / metrics.total_screened)
    if metrics.screen_failure_rate > thresholds["max_screen_failure_rate"]:
        metrics.flags.append(f"High screen failure rate: {metrics.screen_failure_rate:.1%}")

    # Monthly enrollment rate
    activation_date_str = site.get("activation_date", "")
    if activation_date_str and metrics.total_enrolled > 0:
        try:
            activation = datetime.strptime(activation_date_str, "%Y-%m-%d")
            months_active = max((datetime.now() - activation).days / 30.0, 1.0)
            metrics.enrollment_rate_monthly = metrics.total_enrolled / months_active
        except ValueError:
            pass
    if metrics.enrollment_rate_monthly < thresholds["min_monthly_enrollment_rate"] and metrics.total_screened > 0:
        metrics.flags.append(f"Low enrollment rate: {metrics.enrollment_rate_monthly:.1f}/month")

    # Days since last enrollment
    last_enrollment_str = enrollment.get("last_enrollment_date", "")
    if last_enrollment_str:
        try:
            last_enrollment = datetime.strptime(last_enrollment_str, "%Y-%m-%d")
            metrics.days_since_last_enrollment = (datetime.now() - last_enrollment).days
        except ValueError:
            pass
    if metrics.days_since_last_enrollment > thresholds["max_days_enrollment_gap"]:
        metrics.flags.append(f"Enrollment gap: {metrics.days_since_last_enrollment} days")

    # Data quality
    data_quality = site.get("data_quality", {})
    metrics.data_completeness_pct = data_quality.get("completeness_pct", 100.0)
    if metrics.data_completeness_pct < thresholds["min_data_completeness_pct"]:
        metrics.flags.append(f"Low data completeness: {metrics.data_completeness_pct:.1f}%")

    total_queries = data_quality.get("open_queries", 0)
    if metrics.total_enrolled > 0:
        metrics.query_rate_per_subject = total_queries / metrics.total_enrolled
    if metrics.query_rate_per_subject > thresholds["max_query_rate_per_subject"]:
        metrics.flags.append(f"High query rate: {metrics.query_rate_per_subject:.1f}/subject")

    # Protocol deviations
    deviations = site.get("protocol_deviations", [])
    metrics.protocol_deviations = len(deviations) if isinstance(deviations, list) else int(deviations)
    if metrics.total_enrolled > 0:
        metrics.deviation_rate_per_subject = metrics.protocol_deviations / metrics.total_enrolled
    if metrics.deviation_rate_per_subject > thresholds["max_protocol_deviation_rate_per_subject"]:
        metrics.flags.append(f"High deviation rate: {metrics.deviation_rate_per_subject:.2f}/subject")

    # AE reporting delay
    ae_data = site.get("adverse_events", {})
    metrics.mean_ae_reporting_delay_days = ae_data.get("mean_reporting_delay_days", 0.0)
    if metrics.mean_ae_reporting_delay_days > thresholds["max_ae_reporting_delay_days"]:
        metrics.flags.append(f"AE reporting delay: {metrics.mean_ae_reporting_delay_days:.1f} days")

    # Determine overall status
    if len(metrics.flags) >= 3:
        metrics.status = "red"
    elif len(metrics.flags) >= 1:
        metrics.status = "yellow"
    else:
        metrics.status = "green"

    return metrics


def cmd_enrollment(args):
    """Display enrollment dashboard across all sites."""
    manifest = load_manifest(args.manifest)
    thresholds = {**DEFAULT_THRESHOLDS, **manifest.get("thresholds", {})}
    sites = manifest["sites"]

    all_metrics = [compute_site_metrics(s, thresholds) for s in sites]

    total_screened = sum(m.total_screened for m in all_metrics)
    total_enrolled = sum(m.total_enrolled for m in all_metrics)
    total_completed = sum(m.total_completed for m in all_metrics)
    total_withdrawn = sum(m.total_withdrawn for m in all_metrics)
    target = manifest.get("enrollment_target", "N/A")

    print("=" * 75)
    print(f"ENROLLMENT DASHBOARD: {manifest['trial_name']}")
    print(f"Trial ID: {manifest['trial_id']}")
    print("=" * 75)
    print(f"  Target enrollment:   {target}")
    print(f"  Total screened:      {total_screened}")
    print(f"  Total enrolled:      {total_enrolled}")
    print(f"  Total completed:     {total_completed}")
    print(f"  Total withdrawn:     {total_withdrawn}")
    if isinstance(target, (int, float)) and target > 0:
        pct = (total_enrolled / target) * 100
        print(f"  Enrollment progress: {pct:.1f}%")
    print()

    print(f"  {'Site ID':<12} {'Site Name':<25} {'Scrn':<6} {'Enrl':<6} {'Comp':<6} {'Rate/mo':<8} {'Status'}")
    print("-" * 75)
    for m in sorted(all_metrics, key=lambda x: x.total_enrolled, reverse=True):
        status_marker = {"green": "[OK]", "yellow": "[!!]", "red": "[XX]"}.get(m.status, "")
        print(
            f"  {m.site_id:<12} {m.site_name:<25} "
            f"{m.total_screened:<6} {m.total_enrolled:<6} {m.total_completed:<6} "
            f"{m.enrollment_rate_monthly:<8.1f} {status_marker}"
        )

    if args.output:
        report = {
            "report_type": "enrollment_dashboard",
            "trial_id": manifest["trial_id"],
            "timestamp": datetime.now().isoformat(),
            "totals": {
                "screened": total_screened,
                "enrolled": total_enrolled,
                "completed": total_completed,
                "withdrawn": total_withdrawn,
                "target": target,
            },
            "sites": [m.to_dict() for m in all_metrics],
        }
        _write_json(args.output, report)
        print(f"\nReport written to {args.output}")


def cmd_quality(args):
    """Run data quality checks across all sites."""
    manifest = load_manifest(args.manifest)
    thresholds = {**DEFAULT_THRESHOLDS, **manifest.get("thresholds", {})}
    sites = manifest["sites"]

    all_metrics = [compute_site_metrics(s, thresholds) for s in sites]

    print("=" * 75)
    print(f"DATA QUALITY REPORT: {manifest['trial_name']}")
    print(f"Trial ID: {manifest['trial_id']}")
    print("=" * 75)
    print()

    print(f"  {'Site ID':<12} {'Completeness':<14} {'Queries/Subj':<14} {'Dev Rate':<10} {'AE Delay':<10} {'Status'}")
    print("-" * 75)
    for m in sorted(all_metrics, key=lambda x: x.data_completeness_pct):
        status_marker = {"green": "[OK]", "yellow": "[!!]", "red": "[XX]"}.get(m.status, "")
        print(
            f"  {m.site_id:<12} {m.data_completeness_pct:<14.1f} "
            f"{m.query_rate_per_subject:<14.1f} {m.deviation_rate_per_subject:<10.3f} "
            f"{m.mean_ae_reporting_delay_days:<10.1f} {status_marker}"
        )

    # Aggregate quality metrics
    if all_metrics:
        avg_completeness = sum(m.data_completeness_pct for m in all_metrics) / len(all_metrics)
        total_deviations = sum(m.protocol_deviations for m in all_metrics)
        sites_needing_attention = sum(1 for m in all_metrics if m.status != "green")

        print()
        print(f"  Average data completeness: {avg_completeness:.1f}%")
        print(f"  Total protocol deviations: {total_deviations}")
        print(f"  Sites needing attention:   {sites_needing_attention}/{len(all_metrics)}")

    if args.output:
        report = {
            "report_type": "data_quality",
            "trial_id": manifest["trial_id"],
            "timestamp": datetime.now().isoformat(),
            "sites": [m.to_dict() for m in all_metrics],
        }
        _write_json(args.output, report)
        print(f"\nReport written to {args.output}")


def cmd_deviations(args):
    """List and classify protocol deviations across sites."""
    manifest = load_manifest(args.manifest)
    sites = manifest["sites"]

    print("=" * 75)
    print(f"PROTOCOL DEVIATIONS: {manifest['trial_name']}")
    print(f"Trial ID: {manifest['trial_id']}")
    print("=" * 75)
    print()

    total_deviations = 0
    deviation_categories: dict[str, int] = {}

    for site in sites:
        site_id = site.get("site_id", "UNKNOWN")
        deviations = site.get("protocol_deviations", [])

        if isinstance(deviations, list) and deviations:
            print(f"  Site {site_id} ({site.get('site_name', '')}):")
            for dev in deviations:
                if isinstance(dev, dict):
                    cat = dev.get("category", "uncategorized")
                    severity = dev.get("severity", "unknown")
                    desc = dev.get("description", "No description")
                    date = dev.get("date", "")
                    print(f"    [{severity.upper():>8}] {date} - {cat}: {desc}")
                    deviation_categories[cat] = deviation_categories.get(cat, 0) + 1
                    total_deviations += 1
                else:
                    total_deviations += 1
            print()
        elif isinstance(deviations, (int, float)) and deviations > 0:
            print(f"  Site {site_id}: {int(deviations)} deviation(s) (no detail provided)")
            total_deviations += int(deviations)

    print("-" * 75)
    print(f"  Total deviations: {total_deviations}")
    if deviation_categories:
        print("  By category:")
        for cat, count in sorted(deviation_categories.items(), key=lambda x: -x[1]):
            print(f"    {cat}: {count}")

    if args.output:
        report = {
            "report_type": "protocol_deviations",
            "trial_id": manifest["trial_id"],
            "timestamp": datetime.now().isoformat(),
            "total_deviations": total_deviations,
            "categories": deviation_categories,
        }
        _write_json(args.output, report)
        print(f"\nReport written to {args.output}")


def cmd_report(args):
    """Generate comprehensive site performance report."""
    manifest = load_manifest(args.manifest)
    thresholds = {**DEFAULT_THRESHOLDS, **manifest.get("thresholds", {})}
    sites = manifest["sites"]

    all_metrics = [compute_site_metrics(s, thresholds) for s in sites]

    print("=" * 75)
    print(f"COMPREHENSIVE SITE REPORT: {manifest['trial_name']}")
    print(f"Trial ID: {manifest['trial_id']}")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 75)

    for m in all_metrics:
        status_label = SITE_STATUS.get(m.status, "Unknown")
        print(f"\n  --- {m.site_id}: {m.site_name} [{status_label}] ---")
        print(f"  Enrollment:   {m.total_enrolled} enrolled / {m.total_screened} screened")
        print(f"  Screen fail:  {m.screen_failure_rate:.1%}")
        print(f"  Rate:         {m.enrollment_rate_monthly:.1f}/month")
        print(f"  Completeness: {m.data_completeness_pct:.1f}%")
        print(f"  Queries:      {m.query_rate_per_subject:.1f}/subject")
        print(f"  Deviations:   {m.protocol_deviations} ({m.deviation_rate_per_subject:.3f}/subject)")
        print(f"  AE delay:     {m.mean_ae_reporting_delay_days:.1f} days")
        if m.flags:
            print("  FLAGS:")
            for flag in m.flags:
                print(f"    - {flag}")

    # Summary
    red_sites = [m for m in all_metrics if m.status == "red"]
    yellow_sites = [m for m in all_metrics if m.status == "yellow"]
    green_sites = [m for m in all_metrics if m.status == "green"]

    print()
    print("=" * 75)
    print("SUMMARY")
    print(f"  Sites on track (green):       {len(green_sites)}")
    print(f"  Sites need attention (yellow): {len(yellow_sites)}")
    print(f"  Sites need intervention (red): {len(red_sites)}")

    if red_sites:
        print("\n  SITES REQUIRING INTERVENTION:")
        for m in red_sites:
            print(f"    {m.site_id} ({m.site_name}): {', '.join(m.flags)}")

    output_path = args.output or None
    if output_path:
        report = {
            "report_type": "comprehensive",
            "trial_id": manifest["trial_id"],
            "trial_name": manifest["trial_name"],
            "timestamp": datetime.now().isoformat(),
            "thresholds": thresholds,
            "summary": {
                "total_sites": len(all_metrics),
                "green": len(green_sites),
                "yellow": len(yellow_sites),
                "red": len(red_sites),
            },
            "sites": [m.to_dict() for m in all_metrics],
        }
        _write_json(output_path, report)
        print(f"\nReport written to {output_path}")


def cmd_init_manifest(args):
    """Generate a sample trial manifest template."""
    num_sites = args.sites
    today = datetime.now().strftime("%Y-%m-%d")

    manifest = {
        "trial_id": "ONCO-PAI-2026-001",
        "trial_name": "Physical AI-Guided Surgical Oncology Phase II",
        "sponsor": "Physical AI Oncology Consortium",
        "enrollment_target": num_sites * 20,
        "start_date": today,
        "thresholds": DEFAULT_THRESHOLDS,
        "sites": [],
    }

    for i in range(1, num_sites + 1):
        site = {
            "site_id": f"SITE-{i:03d}",
            "site_name": f"Institution {i} Cancer Center",
            "activation_date": today,
            "principal_investigator": f"PI {i}",
            "enrollment": {
                "screened": 0,
                "enrolled": 0,
                "completed": 0,
                "withdrawn": 0,
                "last_enrollment_date": "",
            },
            "data_quality": {
                "completeness_pct": 100.0,
                "open_queries": 0,
            },
            "protocol_deviations": [],
            "adverse_events": {
                "total_reported": 0,
                "serious": 0,
                "mean_reporting_delay_days": 0.0,
            },
        }
        manifest["sites"].append(site)

    output_path = args.output or "trial_manifest.json"
    _write_json(output_path, manifest)
    print(f"Trial manifest template written to {output_path}")
    print(f"  Sites: {num_sites}")
    print("  Edit the manifest to add enrollment and quality data.")


def _write_json(filepath: str, data: dict):
    """Write data to a JSON file."""
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, default=str)


def main():
    parser = argparse.ArgumentParser(
        prog="trial_site_monitor",
        description="Trial Site Monitor: Multi-site enrollment tracking and data quality monitoring for oncology trials.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # enrollment
    p_enroll = subparsers.add_parser("enrollment", help="Display enrollment dashboard")
    p_enroll.add_argument("manifest", help="Path to trial manifest JSON file")
    p_enroll.add_argument("--output", "-o", help="Write JSON report to file")

    # quality
    p_quality = subparsers.add_parser("quality", help="Run data quality checks")
    p_quality.add_argument("manifest", help="Path to trial manifest JSON file")
    p_quality.add_argument("--output", "-o", help="Write JSON report to file")

    # deviations
    p_dev = subparsers.add_parser("deviations", help="List protocol deviations")
    p_dev.add_argument("manifest", help="Path to trial manifest JSON file")
    p_dev.add_argument("--output", "-o", help="Write JSON report to file")

    # report
    p_report = subparsers.add_parser("report", help="Generate comprehensive site report")
    p_report.add_argument("manifest", help="Path to trial manifest JSON file")
    p_report.add_argument("--output", "-o", help="Write JSON report to file")

    # init-manifest
    p_init = subparsers.add_parser("init-manifest", help="Generate sample trial manifest template")
    p_init.add_argument("--sites", type=int, default=5, help="Number of sites (default: 5)")
    p_init.add_argument("--output", "-o", help="Output file path (default: trial_manifest.json)")

    args = parser.parse_args()

    commands = {
        "enrollment": cmd_enrollment,
        "quality": cmd_quality,
        "deviations": cmd_deviations,
        "report": cmd_report,
        "init-manifest": cmd_init_manifest,
    }

    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

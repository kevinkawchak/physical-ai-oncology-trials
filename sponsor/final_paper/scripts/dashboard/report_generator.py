from __future__ import annotations

"""Markdown report generation for the autonomous sponsor simulation.

Produces six report types written to the configured output directory:

1. Hourly summary       - Key metrics and events from the last hour.
2. Daily summary        - Aggregated daily statistics.
3. Agent performance    - Per-agent decision quality and throughput.
4. Safety log           - Robotic safety events and dispositions.
5. Escalation audit     - Full escalation trace for compliance review.
6. Robot utilization    - Per-robot uptime, procedures, and alerts.
"""

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output" / "reports"


class ReportType(str, Enum):
    """Supported report types."""

    HOURLY_SUMMARY = "hourly_summary"
    DAILY_SUMMARY = "daily_summary"
    AGENT_PERFORMANCE = "agent_performance"
    SAFETY_LOG = "safety_log"
    ESCALATION_AUDIT = "escalation_audit"
    ROBOT_UTILIZATION = "robot_utilization"


# ---------------------------------------------------------------------------
# Data structures consumed by the generator
# ---------------------------------------------------------------------------


@dataclass
class HourlySummaryData:
    """Input data for the hourly summary report."""

    hour: int
    trial_id: str
    events_processed: int
    decisions_made: int
    escalations_opened: int
    escalations_resolved: int
    safety_events: int
    safety_events_by_category: dict[str, int]
    enrollment_delta: int
    current_enrollment: int
    target_enrollment: int
    psl_value: float
    psl_delta: float
    key_events: list[str]
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class DailySummaryData:
    """Input data for the daily summary report."""

    day: int
    trial_id: str
    hours_completed: int
    total_events: int
    total_decisions: int
    gate_transitions: dict[str, int]
    enrollment_start: int
    enrollment_end: int
    enrollment_target: int
    safety_summary: dict[str, int]
    escalation_summary: dict[str, int]
    robot_procedures: int
    robot_halts: int
    psl_open: float
    psl_close: float
    highlights: list[str]
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class AgentPerformanceData:
    """Per-agent performance metrics."""

    agent_id: str
    agent_type: str
    decisions_made: int
    avg_confidence: float
    escalation_rate: float
    error_count: int
    avg_response_time_ms: float
    uptime_pct: float


@dataclass
class SafetyLogEntry:
    """A single safety event for the safety log report."""

    event_id: str
    timestamp: str
    robot_id: str
    robot_category: str
    procedure_id: str
    category: str
    disposition: str
    force_deviation: float
    position_deviation_mm: float
    requires_human: bool
    description: str


@dataclass
class EscalationAuditEntry:
    """A single escalation record for the audit report."""

    decision_id: str
    timestamp: str
    level: int
    level_label: str
    decision_type: str
    source_agent: str
    confidence: float
    required_approvals: list[str]
    timeout_seconds: int | None
    outcome: str
    rationale: str


@dataclass
class RobotUtilizationEntry:
    """Per-robot utilization metrics."""

    robot_id: str
    robot_category: str
    procedures_completed: int
    procedures_aborted: int
    total_uptime_hours: float
    total_downtime_hours: float
    utilization_pct: float
    safety_events: int
    halts: int
    current_status: str


# ---------------------------------------------------------------------------
# Report Generator
# ---------------------------------------------------------------------------


class ReportGenerator:
    """Generate Markdown reports and write them to the output directory.

    Usage::

        gen = ReportGenerator()
        gen.generate_hourly_summary(data)
        gen.generate_safety_log(trial_id, entries)
    """

    def __init__(self, output_dir: str | Path | None = None) -> None:
        self._output_dir = Path(output_dir) if output_dir else _DEFAULT_OUTPUT_DIR
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._generated: list[tuple[ReportType, str]] = []

    @property
    def output_dir(self) -> Path:
        return self._output_dir

    def generated_reports(self) -> list[tuple[str, str]]:
        """Return list of (report_type, file_path) for all generated reports."""
        return [(rt.value, fp) for rt, fp in self._generated]

    # ------------------------------------------------------------------
    # 1. Hourly Summary
    # ------------------------------------------------------------------

    def generate_hourly_summary(self, data: HourlySummaryData) -> str:
        """Generate and write the hourly summary report. Returns file path."""
        lines = [
            f"# Hourly Summary - Hour {data.hour}",
            "",
            f"**Trial:** {data.trial_id}  ",
            f"**Generated:** {data.generated_at.strftime('%Y-%m-%d %H:%M:%S UTC')}",
            "",
            "## Key Metrics",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Events processed | {data.events_processed} |",
            f"| Decisions made | {data.decisions_made} |",
            f"| Escalations opened | {data.escalations_opened} |",
            f"| Escalations resolved | {data.escalations_resolved} |",
            f"| Safety events | {data.safety_events} |",
            f"| Enrollment delta | {data.enrollment_delta:+d} |",
            f"| Current enrollment | {data.current_enrollment}/{data.target_enrollment} |",
            f"| PSL | {data.psl_value:.1%} ({data.psl_delta:+.1%}) |",
            "",
            "## Safety Events by Category",
            "",
            "| Category | Count |",
            "|----------|-------|",
        ]
        for cat, count in sorted(data.safety_events_by_category.items()):
            lines.append(f"| {cat} | {count} |")

        lines.extend([
            "",
            "## Key Events",
            "",
        ])
        for event in data.key_events:
            lines.append(f"- {event}")

        lines.append("")
        return self._write_report(
            ReportType.HOURLY_SUMMARY,
            f"hourly_summary_h{data.hour:04d}.md",
            "\n".join(lines),
        )

    # ------------------------------------------------------------------
    # 2. Daily Summary
    # ------------------------------------------------------------------

    def generate_daily_summary(self, data: DailySummaryData) -> str:
        """Generate and write the daily summary report. Returns file path."""
        psl_delta = data.psl_close - data.psl_open
        lines = [
            f"# Daily Summary - Day {data.day}",
            "",
            f"**Trial:** {data.trial_id}  ",
            f"**Generated:** {data.generated_at.strftime('%Y-%m-%d %H:%M:%S UTC')}",
            "",
            "## Overview",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Hours completed | {data.hours_completed} |",
            f"| Total events | {data.total_events} |",
            f"| Total decisions | {data.total_decisions} |",
            f"| Enrollment | {data.enrollment_start} -> {data.enrollment_end} / {data.enrollment_target} |",
            f"| Robot procedures | {data.robot_procedures} |",
            f"| Robot halts | {data.robot_halts} |",
            f"| PSL | {data.psl_open:.1%} -> {data.psl_close:.1%} ({psl_delta:+.1%}) |",
            "",
            "## Gate Transitions",
            "",
            "| Outcome | Count |",
            "|---------|-------|",
        ]
        for outcome, count in sorted(data.gate_transitions.items()):
            lines.append(f"| {outcome} | {count} |")

        lines.extend([
            "",
            "## Safety Summary",
            "",
            "| Category | Count |",
            "|----------|-------|",
        ])
        for cat, count in sorted(data.safety_summary.items()):
            lines.append(f"| {cat} | {count} |")

        lines.extend([
            "",
            "## Escalation Summary",
            "",
            "| Level | Count |",
            "|-------|-------|",
        ])
        for level, count in sorted(data.escalation_summary.items()):
            lines.append(f"| {level} | {count} |")

        lines.extend([
            "",
            "## Highlights",
            "",
        ])
        for h in data.highlights:
            lines.append(f"- {h}")

        lines.append("")
        return self._write_report(
            ReportType.DAILY_SUMMARY,
            f"daily_summary_d{data.day:03d}.md",
            "\n".join(lines),
        )

    # ------------------------------------------------------------------
    # 3. Agent Performance
    # ------------------------------------------------------------------

    def generate_agent_performance(
        self,
        trial_id: str,
        agents: list[AgentPerformanceData],
        period_label: str = "current",
    ) -> str:
        """Generate and write the agent performance report."""
        now = datetime.now(timezone.utc)
        lines = [
            "# Agent Performance Report",
            "",
            f"**Trial:** {trial_id}  ",
            f"**Period:** {period_label}  ",
            f"**Generated:** {now.strftime('%Y-%m-%d %H:%M:%S UTC')}",
            "",
            "## Agent Metrics",
            "",
            (
                "| Agent | Type | Decisions | Avg Conf | Esc Rate "
                "| Errors | Avg RT (ms) | Uptime |"
            ),
            "|-------|------|-----------|----------|----------|--------|-------------|--------|",
        ]
        for a in agents:
            lines.append(
                f"| {a.agent_id} | {a.agent_type} | {a.decisions_made} "
                f"| {a.avg_confidence:.2f} | {a.escalation_rate:.1%} "
                f"| {a.error_count} | {a.avg_response_time_ms:.0f} "
                f"| {a.uptime_pct:.1%} |"
            )

        # Summary statistics
        if agents:
            total_decisions = sum(a.decisions_made for a in agents)
            avg_confidence = sum(a.avg_confidence for a in agents) / len(agents)
            total_errors = sum(a.error_count for a in agents)
            lines.extend([
                "",
                "## Summary",
                "",
                f"- **Total decisions:** {total_decisions}",
                f"- **Average confidence:** {avg_confidence:.2f}",
                f"- **Total errors:** {total_errors}",
                f"- **Agent count:** {len(agents)}",
            ])

        lines.append("")
        filename = f"agent_performance_{period_label}.md"
        return self._write_report(ReportType.AGENT_PERFORMANCE, filename, "\n".join(lines))

    # ------------------------------------------------------------------
    # 4. Safety Log
    # ------------------------------------------------------------------

    def generate_safety_log(
        self,
        trial_id: str,
        entries: list[SafetyLogEntry],
        period_label: str = "current",
    ) -> str:
        """Generate and write the safety log report."""
        now = datetime.now(timezone.utc)
        lines = [
            "# Safety Log",
            "",
            f"**Trial:** {trial_id}  ",
            f"**Period:** {period_label}  ",
            f"**Generated:** {now.strftime('%Y-%m-%d %H:%M:%S UTC')}  ",
            f"**Total events:** {len(entries)}",
            "",
        ]

        # Category breakdown
        cat_counts: dict[str, int] = {}
        for e in entries:
            cat_counts[e.category] = cat_counts.get(e.category, 0) + 1

        lines.extend([
            "## Category Breakdown",
            "",
            "| Category | Count | Pct |",
            "|----------|-------|-----|",
        ])
        for cat in sorted(cat_counts):
            count = cat_counts[cat]
            pct = count / len(entries) * 100 if entries else 0
            lines.append(f"| {cat} | {count} | {pct:.1f}% |")

        # Event details
        lines.extend([
            "",
            "## Event Details",
            "",
        ])

        for e in entries:
            human_flag = " **[HUMAN REQUIRED]**" if e.requires_human else ""
            lines.extend([
                f"### Event {e.event_id}{human_flag}",
                "",
                f"- **Timestamp:** {e.timestamp}",
                f"- **Robot:** {e.robot_id} ({e.robot_category})",
                f"- **Procedure:** {e.procedure_id}",
                f"- **Category:** {e.category}",
                f"- **Disposition:** {e.disposition}",
                f"- **Force deviation:** {e.force_deviation:.4f}",
                f"- **Position deviation:** {e.position_deviation_mm:.2f} mm",
                f"- **Description:** {e.description}",
                "",
            ])

        return self._write_report(
            ReportType.SAFETY_LOG,
            f"safety_log_{period_label}.md",
            "\n".join(lines),
        )

    # ------------------------------------------------------------------
    # 5. Escalation Audit
    # ------------------------------------------------------------------

    def generate_escalation_audit(
        self,
        trial_id: str,
        entries: list[EscalationAuditEntry],
        period_label: str = "current",
    ) -> str:
        """Generate and write the escalation audit report."""
        now = datetime.now(timezone.utc)
        lines = [
            "# Escalation Audit Report",
            "",
            f"**Trial:** {trial_id}  ",
            f"**Period:** {period_label}  ",
            f"**Generated:** {now.strftime('%Y-%m-%d %H:%M:%S UTC')}  ",
            f"**Total escalations:** {len(entries)}",
            "",
        ]

        # Level breakdown
        level_counts: dict[int, int] = {}
        for e in entries:
            level_counts[e.level] = level_counts.get(e.level, 0) + 1

        lines.extend([
            "## Level Distribution",
            "",
            "| Level | Label | Count |",
            "|-------|-------|-------|",
        ])
        for lvl in sorted(level_counts):
            label = entries[0].level_label if entries else ""
            for e in entries:
                if e.level == lvl:
                    label = e.level_label
                    break
            lines.append(f"| L{lvl} | {label} | {level_counts[lvl]} |")

        # Full audit trail
        lines.extend([
            "",
            "## Audit Trail",
            "",
            (
                "| Decision ID | Timestamp | Level | Type | Source "
                "| Confidence | Outcome |"
            ),
            "|-------------|-----------|-------|------|--------|------------|---------|",
        ])
        for e in entries:
            lines.append(
                f"| {e.decision_id} | {e.timestamp} | L{e.level} "
                f"| {e.decision_type} | {e.source_agent} "
                f"| {e.confidence:.2f} | {e.outcome} |"
            )

        # Detailed rationale section
        lines.extend([
            "",
            "## Detailed Rationale",
            "",
        ])
        for e in entries:
            timeout_str = f"{e.timeout_seconds}s" if e.timeout_seconds else "none"
            lines.extend([
                f"### {e.decision_id} (L{e.level})",
                "",
                f"- **Type:** {e.decision_type}",
                f"- **Source:** {e.source_agent}",
                f"- **Approvals required:** {', '.join(e.required_approvals)}",
                f"- **Timeout:** {timeout_str}",
                f"- **Rationale:** {e.rationale}",
                "",
            ])

        return self._write_report(
            ReportType.ESCALATION_AUDIT,
            f"escalation_audit_{period_label}.md",
            "\n".join(lines),
        )

    # ------------------------------------------------------------------
    # 6. Robot Utilization
    # ------------------------------------------------------------------

    def generate_robot_utilization(
        self,
        trial_id: str,
        robots: list[RobotUtilizationEntry],
        period_label: str = "current",
    ) -> str:
        """Generate and write the robot utilization report."""
        now = datetime.now(timezone.utc)
        lines = [
            "# Robot Utilization Report",
            "",
            f"**Trial:** {trial_id}  ",
            f"**Period:** {period_label}  ",
            f"**Generated:** {now.strftime('%Y-%m-%d %H:%M:%S UTC')}  ",
            f"**Robots tracked:** {len(robots)}",
            "",
            "## Fleet Overview",
            "",
            (
                "| Robot | Category | Completed | Aborted | Uptime (h) "
                "| Util% | Safety | Halts | Status |"
            ),
            (
                "|-------|----------|-----------|---------|------------ "
                "|-------|--------|-------|--------|"
            ),
        ]
        for r in robots:
            lines.append(
                f"| {r.robot_id} | {r.robot_category} "
                f"| {r.procedures_completed} | {r.procedures_aborted} "
                f"| {r.total_uptime_hours:.1f} "
                f"| {r.utilization_pct:.1f}% | {r.safety_events} "
                f"| {r.halts} | {r.current_status} |"
            )

        # Summary
        if robots:
            total_completed = sum(r.procedures_completed for r in robots)
            total_aborted = sum(r.procedures_aborted for r in robots)
            avg_util = sum(r.utilization_pct for r in robots) / len(robots)
            total_safety = sum(r.safety_events for r in robots)
            total_halts = sum(r.halts for r in robots)
            lines.extend([
                "",
                "## Fleet Summary",
                "",
                f"- **Total procedures completed:** {total_completed}",
                f"- **Total procedures aborted:** {total_aborted}",
                f"- **Average utilization:** {avg_util:.1f}%",
                f"- **Total safety events:** {total_safety}",
                f"- **Total halts:** {total_halts}",
                f"- **Abort rate:** {total_aborted / max(total_completed + total_aborted, 1) * 100:.1f}%",
            ])

            # Per-category breakdown
            cat_stats: dict[str, list[RobotUtilizationEntry]] = {}
            for r in robots:
                cat_stats.setdefault(r.robot_category, []).append(r)

            lines.extend([
                "",
                "## Category Breakdown",
                "",
                "| Category | Robots | Avg Util% | Total Procedures | Total Halts |",
                "|----------|--------|-----------|------------------|-------------|",
            ])
            for cat in sorted(cat_stats):
                group = cat_stats[cat]
                avg_u = sum(r.utilization_pct for r in group) / len(group)
                total_p = sum(r.procedures_completed for r in group)
                total_h = sum(r.halts for r in group)
                lines.append(f"| {cat} | {len(group)} | {avg_u:.1f}% | {total_p} | {total_h} |")

        lines.append("")
        return self._write_report(
            ReportType.ROBOT_UTILIZATION,
            f"robot_utilization_{period_label}.md",
            "\n".join(lines),
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _write_report(self, report_type: ReportType, filename: str, content: str) -> str:
        """Write content to a file and return the absolute path."""
        filepath = self._output_dir / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text(content, encoding="utf-8")
        abs_path = str(filepath.resolve())
        self._generated.append((report_type, abs_path))
        logger.info("Report written: %s -> %s", report_type.value, abs_path)
        return abs_path

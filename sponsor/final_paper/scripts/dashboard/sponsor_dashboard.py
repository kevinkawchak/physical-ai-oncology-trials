from __future__ import annotations

"""Terminal-based analytics dashboard for the autonomous sponsor simulation.

Renders a multi-section status view to the terminal using only standard
library facilities (no curses, no GUI).  Designed for periodic refresh
in a headless CI/server environment.

Sections
--------
1. Active Agents       - Current agent states and health.
2. Decision Queue      - Pending decisions awaiting evaluation.
3. Escalation Status   - Open escalations by level.
4. Robot Auth Pipeline  - Procedure authorization gate status.
5. Patient Flow        - Enrollment and treatment pipeline.
6. PSL Trend           - Program Success Likelihood over recent windows.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from io import StringIO

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Terminal formatting helpers
# ---------------------------------------------------------------------------

_WIDTH = 100
_SEPARATOR = "-" * _WIDTH
_DOUBLE_SEP = "=" * _WIDTH


# ---------------------------------------------------------------------------
# Domain types for dashboard data
# ---------------------------------------------------------------------------


class AgentState(str, Enum):
    ACTIVE = "active"
    IDLE = "idle"
    WAITING = "waiting"
    ERROR = "error"
    OFFLINE = "offline"


@dataclass
class AgentStatus:
    """Snapshot of a single agent's state."""

    agent_id: str
    agent_type: str
    state: AgentState
    current_task: str
    decisions_made: int
    errors: int
    last_heartbeat: datetime


@dataclass
class DecisionQueueItem:
    """A pending decision awaiting processing."""

    decision_id: str
    decision_type: str
    source_agent: str
    priority: int  # 1 (highest) to 5 (lowest)
    queued_at: datetime
    description: str


@dataclass
class EscalationStatusEntry:
    """An open escalation record."""

    escalation_id: str
    level: int
    level_label: str
    decision_type: str
    source_agent: str
    opened_at: datetime
    timeout_seconds: int | None
    status: str  # "open", "approved", "timed_out"


@dataclass
class RobotAuthEntry:
    """Status of a procedure in the authorization pipeline."""

    procedure_id: str
    robot_id: str
    patient_id: str
    current_gate: int
    gates_passed: int
    total_gates: int
    status: str  # "in_progress", "authorized", "blocked"


@dataclass
class PatientFlowEntry:
    """Patient flow through the trial pipeline."""

    stage: str
    count: int
    target: int
    pct_of_target: float


@dataclass
class PSLDataPoint:
    """Program Success Likelihood data point."""

    timestamp: datetime
    psl_value: float  # 0.0 - 1.0
    confidence_interval_low: float
    confidence_interval_high: float


@dataclass
class DashboardData:
    """Complete data payload for a dashboard render cycle."""

    agents: list[AgentStatus] = field(default_factory=list)
    decision_queue: list[DecisionQueueItem] = field(default_factory=list)
    escalations: list[EscalationStatusEntry] = field(default_factory=list)
    robot_auth_pipeline: list[RobotAuthEntry] = field(default_factory=list)
    patient_flow: list[PatientFlowEntry] = field(default_factory=list)
    psl_trend: list[PSLDataPoint] = field(default_factory=list)
    simulation_hour: int = 0
    trial_id: str = "ONCO-TRIAL-001"
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# Dashboard renderer
# ---------------------------------------------------------------------------


class SponsorDashboard:
    """Render a terminal-based analytics dashboard.

    Usage::

        dashboard = SponsorDashboard()
        data = DashboardData(agents=[...], ...)
        output = dashboard.render(data)
        print(output)
    """

    def __init__(self, width: int = _WIDTH) -> None:
        self._width = width
        self._sep = "-" * width
        self._dsep = "=" * width

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render(self, data: DashboardData) -> str:
        """Render all dashboard sections and return the complete string."""
        buf = StringIO()
        self._header(buf, data)
        self._section_agents(buf, data.agents)
        self._section_decision_queue(buf, data.decision_queue)
        self._section_escalations(buf, data.escalations)
        self._section_robot_auth(buf, data.robot_auth_pipeline)
        self._section_patient_flow(buf, data.patient_flow)
        self._section_psl_trend(buf, data.psl_trend)
        self._footer(buf, data)
        return buf.getvalue()

    # ------------------------------------------------------------------
    # Header / Footer
    # ------------------------------------------------------------------

    def _header(self, buf: StringIO, data: DashboardData) -> None:
        buf.write(f"\n{self._dsep}\n")
        title = f"AUTONOMOUS SPONSOR DASHBOARD  |  {data.trial_id}  |  Hour {data.simulation_hour}"
        buf.write(f"  {title}\n")
        buf.write(f"  Generated: {data.generated_at.strftime('%Y-%m-%d %H:%M:%S UTC')}\n")
        buf.write(f"{self._dsep}\n")

    def _footer(self, buf: StringIO, data: DashboardData) -> None:
        buf.write(f"\n{self._dsep}\n")
        agent_active = sum(1 for a in data.agents if a.state == AgentState.ACTIVE)
        buf.write(f"  Agents: {agent_active}/{len(data.agents)} active")
        buf.write(f"  |  Queue: {len(data.decision_queue)} pending")
        open_esc = sum(1 for e in data.escalations if e.status == "open")
        buf.write(f"  |  Escalations: {open_esc} open")
        if data.psl_trend:
            buf.write(f"  |  PSL: {data.psl_trend[-1].psl_value:.1%}")
        buf.write(f"\n{self._dsep}\n")

    # ------------------------------------------------------------------
    # Section: Active Agents
    # ------------------------------------------------------------------

    def _section_agents(self, buf: StringIO, agents: list[AgentStatus]) -> None:
        buf.write(f"\n  [1] ACTIVE AGENTS ({len(agents)})\n")
        buf.write(f"  {self._sep}\n")
        if not agents:
            buf.write("  (no agents registered)\n")
            return
        header = f"  {'Agent':<20} {'Type':<18} {'State':<10} {'Decisions':>9} {'Errors':>7} {'Task':<30}"
        buf.write(f"{header}\n")
        buf.write(f"  {self._sep}\n")
        for a in agents:
            state_indicator = _state_symbol(a.state)
            task_display = (a.current_task[:28] + "..") if len(a.current_task) > 30 else a.current_task
            line = (
                f"  {a.agent_id:<20} {a.agent_type:<18} "
                f"{state_indicator} {a.state.value:<7} "
                f"{a.decisions_made:>9} {a.errors:>7} {task_display:<30}"
            )
            buf.write(f"{line}\n")

    # ------------------------------------------------------------------
    # Section: Decision Queue
    # ------------------------------------------------------------------

    def _section_decision_queue(self, buf: StringIO, queue: list[DecisionQueueItem]) -> None:
        buf.write(f"\n  [2] DECISION QUEUE ({len(queue)} pending)\n")
        buf.write(f"  {self._sep}\n")
        if not queue:
            buf.write("  (queue empty)\n")
            return
        header = f"  {'ID':<14} {'Type':<22} {'Source':<18} {'Pri':>3} {'Queued':<20} {'Description':<20}"
        buf.write(f"{header}\n")
        buf.write(f"  {self._sep}\n")
        sorted_q = sorted(queue, key=lambda d: d.priority)
        for item in sorted_q[:15]:
            desc = (item.description[:18] + "..") if len(item.description) > 20 else item.description
            line = (
                f"  {item.decision_id:<14} {item.decision_type:<22} "
                f"{item.source_agent:<18} {item.priority:>3} "
                f"{item.queued_at.strftime('%H:%M:%S'):>20} {desc:<20}"
            )
            buf.write(f"{line}\n")
        if len(queue) > 15:
            buf.write(f"  ... and {len(queue) - 15} more\n")

    # ------------------------------------------------------------------
    # Section: Escalation Status
    # ------------------------------------------------------------------

    def _section_escalations(self, buf: StringIO, escalations: list[EscalationStatusEntry]) -> None:
        open_esc = [e for e in escalations if e.status == "open"]
        buf.write(f"\n  [3] ESCALATION STATUS ({len(open_esc)} open / {len(escalations)} total)\n")
        buf.write(f"  {self._sep}\n")

        # Level summary
        level_counts: dict[int, int] = {i: 0 for i in range(1, 6)}
        for e in open_esc:
            if 1 <= e.level <= 5:
                level_counts[e.level] += 1

        level_labels = {
            1: "Informational",
            2: "Confirmation",
            3: "Clinical",
            4: "Regulatory",
            5: "Safety-critical",
        }
        summary_parts: list[str] = []
        for lvl in range(1, 6):
            count = level_counts[lvl]
            bar = _bar(count, max(level_counts.values()) if level_counts.values() else 1, width=20)
            summary_parts.append(f"  L{lvl} {level_labels[lvl]:<18} {bar} {count}")
        for part in summary_parts:
            buf.write(f"{part}\n")

        if open_esc:
            buf.write(f"\n  {'ID':<14} {'Level':<20} {'Type':<22} {'Source':<18} {'Timeout':<10}\n")
            buf.write(f"  {self._sep}\n")
            for e in open_esc[:10]:
                timeout_str = f"{e.timeout_seconds}s" if e.timeout_seconds else "none"
                line = (
                    f"  {e.escalation_id:<14} L{e.level} {e.level_label:<17} "
                    f"{e.decision_type:<22} {e.source_agent:<18} {timeout_str:<10}"
                )
                buf.write(f"{line}\n")

    # ------------------------------------------------------------------
    # Section: Robot Authorization Pipeline
    # ------------------------------------------------------------------

    def _section_robot_auth(self, buf: StringIO, pipeline: list[RobotAuthEntry]) -> None:
        buf.write(f"\n  [4] ROBOT AUTHORIZATION PIPELINE ({len(pipeline)} procedures)\n")
        buf.write(f"  {self._sep}\n")
        if not pipeline:
            buf.write("  (no procedures in pipeline)\n")
            return

        header = f"  {'Procedure':<14} {'Robot':<12} {'Patient':<12} {'Gate':>4} {'Progress':<22} {'Status':<12}"
        buf.write(f"{header}\n")
        buf.write(f"  {self._sep}\n")
        for entry in pipeline[:12]:
            progress = _gate_progress(entry.gates_passed, entry.total_gates)
            line = (
                f"  {entry.procedure_id:<14} {entry.robot_id:<12} "
                f"{entry.patient_id:<12} {entry.current_gate:>4} "
                f"{progress:<22} {entry.status:<12}"
            )
            buf.write(f"{line}\n")

    # ------------------------------------------------------------------
    # Section: Patient Flow
    # ------------------------------------------------------------------

    def _section_patient_flow(self, buf: StringIO, flow: list[PatientFlowEntry]) -> None:
        buf.write("\n  [5] PATIENT FLOW\n")
        buf.write(f"  {self._sep}\n")
        if not flow:
            buf.write("  (no patient flow data)\n")
            return
        max_target = max((s.target for s in flow), default=1)
        for entry in flow:
            count_bar = _bar(entry.count, max_target, width=30)
            line = f"  {entry.stage:<22} {count_bar} {entry.count:>5}/{entry.target:<5} ({entry.pct_of_target:>5.1f}%)"
            buf.write(f"{line}\n")

    # ------------------------------------------------------------------
    # Section: PSL Trend
    # ------------------------------------------------------------------

    def _section_psl_trend(self, buf: StringIO, trend: list[PSLDataPoint]) -> None:
        buf.write("\n  [6] PROGRAM SUCCESS LIKELIHOOD (PSL) TREND\n")
        buf.write(f"  {self._sep}\n")
        if not trend:
            buf.write("  (no PSL data available)\n")
            return

        # Show sparkline-style text chart
        recent = trend[-20:]
        buf.write("  1.0 |")
        for dp in recent:
            buf.write(_spark_char(dp.psl_value))
        buf.write("\n")
        buf.write("  0.5 |")
        for _ in recent:
            buf.write("-")
        buf.write("\n")
        buf.write("  0.0 |")
        for _ in recent:
            buf.write(" ")
        buf.write("\n")

        if recent:
            latest = recent[-1]
            buf.write(
                f"  Latest PSL: {latest.psl_value:.1%} "
                f"(CI: {latest.confidence_interval_low:.1%} - {latest.confidence_interval_high:.1%})\n"
            )

            # Trend direction
            if len(recent) >= 3:
                prev_avg = sum(d.psl_value for d in recent[-4:-1]) / min(3, len(recent) - 1)
                delta = latest.psl_value - prev_avg
                direction = "UP" if delta > 0.01 else ("DOWN" if delta < -0.01 else "STABLE")
                buf.write(f"  Trend: {direction} ({delta:+.2%} vs prior 3 windows)\n")


# ---------------------------------------------------------------------------
# Formatting utilities
# ---------------------------------------------------------------------------

_STATE_SYMBOLS: dict[AgentState, str] = {
    AgentState.ACTIVE: "[*]",
    AgentState.IDLE: "[ ]",
    AgentState.WAITING: "[~]",
    AgentState.ERROR: "[!]",
    AgentState.OFFLINE: "[x]",
}


def _state_symbol(state: AgentState) -> str:
    return _STATE_SYMBOLS.get(state, "[?]")


def _bar(value: int, max_value: int, width: int = 20) -> str:
    """Render a simple text bar chart segment."""
    if max_value <= 0:
        return " " * width
    filled = int((value / max_value) * width)
    filled = min(filled, width)
    return "#" * filled + "." * (width - filled)


def _gate_progress(passed: int, total: int) -> str:
    """Render gate progress as a visual indicator."""
    parts: list[str] = []
    for i in range(1, total + 1):
        if i <= passed:
            parts.append(f"[G{i}:OK]")
        elif i == passed + 1:
            parts.append(f"[G{i}:>>]")
        else:
            parts.append(f"[G{i}:--]")
    return " ".join(parts)


def _spark_char(value: float) -> str:
    """Return a single character representing a 0-1 value."""
    if value >= 0.875:
        return "^"
    if value >= 0.75:
        return "'"
    if value >= 0.625:
        return "-"
    if value >= 0.50:
        return ","
    if value >= 0.375:
        return "."
    if value >= 0.25:
        return "_"
    return " "

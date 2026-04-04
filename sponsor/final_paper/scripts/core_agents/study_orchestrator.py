"""Study orchestrator: master orchestrator, dependencies, work queues."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class TaskStatus(Enum):
    PENDING = "pending"
    READY = "ready"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskPriority(Enum):
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4


@dataclass
class WorkItem:
    task_id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    name: str = ""
    agent: str = ""
    status: TaskStatus = TaskStatus.PENDING
    priority: TaskPriority = TaskPriority.MEDIUM
    dependencies: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None
    output: dict[str, object] = field(default_factory=dict)


@dataclass
class StudyPhase:
    phase_id: str
    name: str
    tasks: list[WorkItem] = field(default_factory=list)


@dataclass
class StudyPlan:
    study_id: str
    title: str
    phases: list[StudyPhase] = field(default_factory=list)


class WorkQueue:
    """Priority-based work queue with dependency resolution."""

    def __init__(self) -> None:
        self._items: list[WorkItem] = []

    def enqueue(self, item: WorkItem) -> None:
        self._items.append(item)

    def dequeue(self) -> WorkItem | None:
        ready = [i for i in self._items if i.status == TaskStatus.READY]
        if not ready:
            return None
        ready.sort(key=lambda x: x.priority.value)
        item = ready[0]
        item.status = TaskStatus.IN_PROGRESS
        return item

    @property
    def pending_count(self) -> int:
        return sum(1 for i in self._items if i.status in (TaskStatus.PENDING, TaskStatus.READY))

    @property
    def items(self) -> list[WorkItem]:
        return list(self._items)


class StudyOrchestrator:
    """Master orchestrator managing study lifecycle, dependencies, and work dispatch."""

    def __init__(self) -> None:
        self.studies: dict[str, StudyPlan] = {}
        self.queue = WorkQueue()
        self._completed: set[str] = set()

    def create_study(self, study_id: str, title: str) -> StudyPlan:
        plan = StudyPlan(study_id=study_id, title=title)
        self.studies[study_id] = plan
        return plan

    def add_task(self, study_id: str, task: WorkItem) -> str:
        plan = self.studies[study_id]
        if not plan.phases:
            plan.phases.append(StudyPhase("default", "Default Phase"))
        plan.phases[-1].tasks.append(task)
        self.queue.enqueue(task)
        return task.task_id

    def resolve_dependencies(self) -> int:
        resolved = 0
        for item in self.queue.items:
            if item.status != TaskStatus.PENDING:
                continue
            if all(dep in self._completed for dep in item.dependencies):
                item.status = TaskStatus.READY
                resolved += 1
            elif any(dep in self._failed_ids() for dep in item.dependencies):
                item.status = TaskStatus.BLOCKED
        return resolved

    def _failed_ids(self) -> set[str]:
        return {i.task_id for i in self.queue.items if i.status == TaskStatus.FAILED}

    def complete_task(self, task_id: str, output: dict[str, object] | None = None) -> None:
        for item in self.queue.items:
            if item.task_id == task_id:
                item.status = TaskStatus.COMPLETED
                item.completed_at = datetime.utcnow()
                item.output = output or {}
                self._completed.add(task_id)
                break
        self.resolve_dependencies()

    def dispatch_next(self) -> WorkItem | None:
        self.resolve_dependencies()
        return self.queue.dequeue()

    def study_progress(self, study_id: str) -> dict[str, int]:
        plan = self.studies[study_id]
        all_tasks = [t for p in plan.phases for t in p.tasks]
        return {s.value: sum(1 for t in all_tasks if t.status == s) for s in TaskStatus}


def main() -> None:
    orch = StudyOrchestrator()
    orch.create_study("STU-001", "Phase II NSCLC Robotic Biopsy")
    t1 = WorkItem(task_id="T1", name="Protocol Finalization", agent="protocol_generator", priority=TaskPriority.CRITICAL)
    t2 = WorkItem(task_id="T2", name="Site Selection", agent="site_gateway", dependencies=["T1"])
    t3 = WorkItem(task_id="T3", name="IND Submission", agent="ind_filing_engine", dependencies=["T1"])
    t1.status = TaskStatus.READY
    for t in [t1, t2, t3]:
        orch.add_task("STU-001", t)
    dispatched = orch.dispatch_next()
    if dispatched:
        print(f"Dispatched: {dispatched.name}")
        orch.complete_task(dispatched.task_id, {"version": "1.0"})
    next_task = orch.dispatch_next()
    if next_task:
        print(f"Next: {next_task.name}")
    print("Progress:", orch.study_progress("STU-001"))


if __name__ == "__main__":
    main()

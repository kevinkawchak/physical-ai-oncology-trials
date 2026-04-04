"""Trust layer core: agent PKI identity, RBAC, audit trail."""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum


class AgentRole(Enum):
    ORCHESTRATOR = "orchestrator"
    SAFETY = "safety"
    REGULATORY = "regulatory"
    CLINICAL = "clinical"
    ROBOTIC = "robotic"
    DATA = "data"
    QUALITY = "quality"
    SUPPLY = "supply"
    READONLY = "readonly"


class Permission(Enum):
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    APPROVE = "approve"
    ADMIN = "admin"


@dataclass
class PKIIdentity:
    agent_id: str = field(default_factory=lambda: f"AGT-{uuid.uuid4().hex[:8]}")
    common_name: str = ""
    public_key_hash: str = ""
    issued_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: datetime = field(default_factory=lambda: datetime.utcnow() + timedelta(days=365))
    revoked: bool = False

    @property
    def is_valid(self) -> bool:
        return not self.revoked and datetime.utcnow() < self.expires_at


@dataclass
class RBACPolicy:
    role: AgentRole
    permissions: set[Permission] = field(default_factory=set)
    resource_patterns: list[str] = field(default_factory=list)


@dataclass
class AuditEntry:
    entry_id: str = field(default_factory=lambda: f"AUD-{uuid.uuid4().hex[:8]}")
    agent_id: str = ""
    action: str = ""
    resource: str = ""
    outcome: str = "success"
    timestamp: datetime = field(default_factory=datetime.utcnow)
    detail: str = ""


ROLE_PERMISSIONS: dict[AgentRole, set[Permission]] = {
    AgentRole.ORCHESTRATOR: {Permission.READ, Permission.WRITE, Permission.EXECUTE, Permission.APPROVE},
    AgentRole.SAFETY: {Permission.READ, Permission.WRITE, Permission.EXECUTE},
    AgentRole.REGULATORY: {Permission.READ, Permission.WRITE},
    AgentRole.CLINICAL: {Permission.READ, Permission.WRITE, Permission.EXECUTE},
    AgentRole.ROBOTIC: {Permission.READ, Permission.EXECUTE},
    AgentRole.DATA: {Permission.READ, Permission.WRITE},
    AgentRole.QUALITY: {Permission.READ, Permission.WRITE, Permission.APPROVE},
    AgentRole.SUPPLY: {Permission.READ, Permission.WRITE},
    AgentRole.READONLY: {Permission.READ},
}


class TrustLayerCore:
    """Manages agent PKI identities, RBAC policies, and audit trails."""

    def __init__(self) -> None:
        self.identities: dict[str, PKIIdentity] = {}
        self.role_assignments: dict[str, AgentRole] = {}
        self.audit_log: list[AuditEntry] = []

    def issue_identity(self, common_name: str, role: AgentRole) -> PKIIdentity:
        key_material = f"{common_name}|{uuid.uuid4().hex}"
        identity = PKIIdentity(
            common_name=common_name,
            public_key_hash=hashlib.sha256(key_material.encode()).hexdigest()[:32],
        )
        self.identities[identity.agent_id] = identity
        self.role_assignments[identity.agent_id] = role
        self._log(identity.agent_id, "identity_issued", f"role={role.value}")
        return identity

    def revoke_identity(self, agent_id: str) -> bool:
        identity = self.identities.get(agent_id)
        if identity:
            identity.revoked = True
            self._log(agent_id, "identity_revoked", "")
            return True
        return False

    def check_permission(self, agent_id: str, permission: Permission, resource: str = "") -> bool:
        identity = self.identities.get(agent_id)
        if not identity or not identity.is_valid:
            self._log(agent_id, "permission_check", resource, "denied_invalid_identity")
            return False
        role = self.role_assignments.get(agent_id)
        if role is None:
            return False
        allowed = ROLE_PERMISSIONS.get(role, set())
        granted = permission in allowed
        outcome = "granted" if granted else "denied"
        self._log(agent_id, f"permission_check:{permission.value}", resource, outcome)
        return granted

    def _log(self, agent_id: str, action: str, resource: str, outcome: str = "success") -> None:
        entry = AuditEntry(agent_id=agent_id, action=action, resource=resource, outcome=outcome)
        self.audit_log.append(entry)

    def audit_trail(self, agent_id: str | None = None, limit: int = 50) -> list[dict[str, str]]:
        entries = self.audit_log if agent_id is None else [e for e in self.audit_log if e.agent_id == agent_id]
        return [
            {"agent": e.agent_id, "action": e.action, "outcome": e.outcome, "time": e.timestamp.isoformat()}
            for e in entries[-limit:]
        ]

    def summary(self) -> dict[str, object]:
        valid = sum(1 for i in self.identities.values() if i.is_valid)
        return {
            "total_agents": len(self.identities),
            "valid": valid,
            "revoked": len(self.identities) - valid,
            "audit_entries": len(self.audit_log),
        }


def main() -> None:
    trust = TrustLayerCore()
    orch = trust.issue_identity("StudyOrchestrator", AgentRole.ORCHESTRATOR)
    safety = trust.issue_identity("SafetyAgent", AgentRole.SAFETY)
    readonly = trust.issue_identity("DashboardViewer", AgentRole.READONLY)
    print(f"Orchestrator WRITE: {trust.check_permission(orch.agent_id, Permission.WRITE)}")
    print(f"Safety APPROVE: {trust.check_permission(safety.agent_id, Permission.APPROVE)}")
    print(f"Readonly WRITE: {trust.check_permission(readonly.agent_id, Permission.WRITE)}")
    print(f"Summary: {trust.summary()}")


if __name__ == "__main__":
    main()

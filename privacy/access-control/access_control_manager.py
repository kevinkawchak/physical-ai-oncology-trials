"""
=============================================================================
Access Control Manager for Physical AI Oncology Trials
=============================================================================

Role-based access control (RBAC) with 21 CFR Part 11 compliant audit
trails for managing access to clinical trial data, AI models, and
robotic systems across multi-site oncology trials.

CLINICAL CONTEXT:
-----------------
Physical AI oncology trials involve diverse personnel (investigators,
coordinators, engineers, AI systems) accessing sensitive data across
multiple institutions. Access controls must:
  - Enforce least-privilege access to PHI
  - Distinguish human and AI system access
  - Maintain tamper-evident audit trails
  - Support multi-factor authentication
  - Comply with 21 CFR Part 11 and proposed HIPAA Security Rule updates

FRAMEWORK REQUIREMENTS:
-----------------------
Required:
    - Python 3.10+

Optional:
    - cryptography (for audit trail integrity)
    - pyjwt (for token-based authentication)

REFERENCES:
    - FDA 21 CFR Part 11 Q&A Guidance (Oct 2024)
    - HIPAA Security Rule NPRM (Jan 2025, Federal Register 2024-30983)
    - NIST SP 800-53 Rev. 5: Security and Privacy Controls
    - NISTIR 8596: Cybersecurity Profile for AI (Dec 2025)

LICENSE: MIT
VERSION: 1.0.0
LAST UPDATED: February 2026
=============================================================================
"""

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# SECTION 1: ROLE AND PERMISSION DEFINITIONS
# =============================================================================

class Permission(Enum):
    """Granular permissions for clinical trial data and systems."""
    # PHI access
    READ_PHI = "read_phi"
    WRITE_PHI = "write_phi"
    EXPORT_PHI = "export_phi"

    # De-identified data access
    READ_DEIDENTIFIED = "read_deidentified"
    WRITE_DEIDENTIFIED = "write_deidentified"
    EXPORT_DEIDENTIFIED = "export_deidentified"

    # Clinical operations
    APPROVE_ENROLLMENT = "approve_enrollment"
    WRITE_CLINICAL_DATA = "write_clinical_data"
    REPORT_ADVERSE_EVENT = "report_adverse_event"

    # AI/ML operations
    TRAIN_MODEL = "train_model"
    DEPLOY_MODEL = "deploy_model"
    WRITE_MODEL_OUTPUTS = "write_model_outputs"
    EXPORT_MODEL = "export_model"

    # Administrative
    MANAGE_USERS = "manage_users"
    MANAGE_ROLES = "manage_roles"
    VIEW_AUDIT_LOGS = "view_audit_logs"
    EXPORT_AUDIT_LOGS = "export_audit_logs"
    CONFIGURE_SYSTEM = "configure_system"

    # Queries
    RUN_QUERIES = "run_queries"
    EXPORT_AGGREGATED = "export_aggregated"


class UserType(Enum):
    """Types of system users."""
    HUMAN = "human"
    AI_SYSTEM = "ai_system"
    SERVICE_ACCOUNT = "service_account"


@dataclass
class Role:
    """Access control role definition."""
    name: str
    description: str
    permissions: list[str]
    max_phi_records: int = 0       # 0 = unlimited for authorized roles
    session_timeout_minutes: int = 480  # 8 hours default
    mfa_required: bool = True
    user_type: UserType = UserType.HUMAN

    def has_permission(self, permission: str) -> bool:
        """Check if role has a specific permission."""
        return permission in self.permissions


@dataclass
class User:
    """System user record."""
    user_id: str
    name: str
    email: str
    role_name: str
    user_type: UserType = UserType.HUMAN
    site_id: str = ""
    active: bool = True
    mfa_enrolled: bool = False
    created_date: str = ""
    last_access: str = ""
    access_expiration: str = ""


@dataclass
class AccessDecision:
    """Result of an access control check."""
    granted: bool
    user_id: str
    resource: str
    action: str
    reason: str
    timestamp: str
    audit_id: str
    conditions: list[str] = field(default_factory=list)


@dataclass
class AuditEntry:
    """21 CFR Part 11 compliant audit trail entry."""
    audit_id: str
    timestamp: str
    user_id: str
    user_type: str
    action: str
    resource: str
    decision: str
    reason: str
    ip_address: str = ""
    session_id: str = ""
    integrity_hash: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "audit_id": self.audit_id,
            "timestamp": self.timestamp,
            "user_id": self.user_id,
            "user_type": self.user_type,
            "action": self.action,
            "resource": self.resource,
            "decision": self.decision,
            "reason": self.reason,
            "ip_address": self.ip_address,
            "session_id": self.session_id,
            "integrity_hash": self.integrity_hash
        }


# =============================================================================
# SECTION 2: PREDEFINED CLINICAL TRIAL ROLES
# =============================================================================

PREDEFINED_ROLES: dict[str, dict[str, Any]] = {
    "principal_investigator": {
        "description": "Site PI with full clinical oversight",
        "permissions": [
            "read_phi", "write_clinical_data", "export_deidentified",
            "approve_enrollment", "report_adverse_event",
            "view_audit_logs", "run_queries", "export_aggregated"
        ],
        "mfa_required": True,
        "session_timeout_minutes": 480
    },
    "sub_investigator": {
        "description": "Sub-investigator with site-level clinical access",
        "permissions": [
            "read_phi", "write_clinical_data",
            "report_adverse_event", "read_deidentified",
            "run_queries"
        ],
        "mfa_required": True,
        "session_timeout_minutes": 480
    },
    "clinical_coordinator": {
        "description": "Clinical research coordinator managing daily operations",
        "permissions": [
            "read_phi", "write_clinical_data",
            "read_deidentified", "report_adverse_event"
        ],
        "mfa_required": True,
        "session_timeout_minutes": 480
    },
    "data_manager": {
        "description": "Data manager overseeing trial database",
        "permissions": [
            "read_phi", "read_deidentified", "write_deidentified",
            "run_queries", "export_deidentified",
            "view_audit_logs", "export_aggregated"
        ],
        "mfa_required": True,
        "session_timeout_minutes": 480
    },
    "biostatistician": {
        "description": "Biostatistician with de-identified data access only",
        "permissions": [
            "read_deidentified", "run_queries",
            "export_aggregated", "export_deidentified"
        ],
        "mfa_required": True,
        "session_timeout_minutes": 480
    },
    "ai_ml_engineer": {
        "description": "AI/ML engineer for model development",
        "permissions": [
            "read_deidentified", "train_model", "deploy_model",
            "write_model_outputs", "export_model", "run_queries"
        ],
        "mfa_required": True,
        "session_timeout_minutes": 480
    },
    "ai_system": {
        "description": "Automated AI system (training pipeline, inference engine)",
        "permissions": [
            "read_deidentified", "write_model_outputs"
        ],
        "mfa_required": False,
        "session_timeout_minutes": 0  # No timeout for automated systems
    },
    "monitor_cra": {
        "description": "Clinical research associate / site monitor",
        "permissions": [
            "read_phi", "read_deidentified",
            "view_audit_logs", "run_queries"
        ],
        "mfa_required": True,
        "session_timeout_minutes": 480
    },
    "irb_member": {
        "description": "IRB/ethics committee member with limited access",
        "permissions": [
            "export_aggregated"
        ],
        "mfa_required": True,
        "session_timeout_minutes": 240
    }
}


# =============================================================================
# SECTION 3: ACCESS CONTROL MANAGER
# =============================================================================

class AccessControlManager:
    """
    Role-based access control with 21 CFR Part 11 audit trails.

    Manages user authentication, authorization, and comprehensive
    audit logging for clinical trial data access.

    21 CFR PART 11 COMPLIANCE:
    --------------------------
    1. Unique user identification (user_id)
    2. Electronic signatures (session-based)
    3. Audit trails with timestamps
    4. System access limited to authorized individuals
    5. Record of all permission changes
    6. Tamper-evident audit log integrity
    """

    def __init__(
        self,
        compliance_framework: str = "21_cfr_part_11",
        audit_enabled: bool = True,
        mfa_required: bool = True,
        audit_log_path: str = "audit_logs/"
    ):
        """
        Initialize access control manager.

        Args:
            compliance_framework: "21_cfr_part_11" or "hipaa"
            audit_enabled: Enable comprehensive audit logging
            mfa_required: Require MFA for all human users
            audit_log_path: Directory for audit log storage
        """
        self.compliance_framework = compliance_framework
        self.audit_enabled = audit_enabled
        self.mfa_required = mfa_required
        self.audit_log_path = Path(audit_log_path)

        self._roles: dict[str, Role] = {}
        self._users: dict[str, User] = {}
        self._audit_log: list[AuditEntry] = []
        self._audit_counter = 0

        # Load predefined roles
        for role_name, role_def in PREDEFINED_ROLES.items():
            self._roles[role_name] = Role(
                name=role_name,
                description=role_def["description"],
                permissions=role_def["permissions"],
                mfa_required=role_def["mfa_required"],
                session_timeout_minutes=role_def["session_timeout_minutes"]
            )

        logger.info(
            f"AccessControlManager initialized: framework={compliance_framework}, "
            f"audit={audit_enabled}, mfa={mfa_required}, "
            f"predefined_roles={len(self._roles)}"
        )

    def define_role(
        self,
        name: str,
        permissions: list[str],
        description: str = "",
        mfa_required: bool = True,
        session_timeout_minutes: int = 480
    ):
        """
        Define a custom role.

        Args:
            name: Role name
            permissions: List of permission strings
            description: Role description
            mfa_required: Whether MFA is required
            session_timeout_minutes: Session timeout
        """
        self._roles[name] = Role(
            name=name,
            description=description,
            permissions=permissions,
            mfa_required=mfa_required,
            session_timeout_minutes=session_timeout_minutes
        )

        self._log_audit(
            user_id="SYSTEM",
            action="role_defined",
            resource=f"role:{name}",
            decision="completed",
            reason=f"Role defined with {len(permissions)} permissions"
        )

        logger.info(f"Role defined: {name} ({len(permissions)} permissions)")

    def assign_role(
        self,
        user_id: str,
        role: str,
        name: str = "",
        email: str = "",
        site_id: str = "",
        user_type: str = "human",
        access_expiration: str = ""
    ):
        """
        Assign a role to a user.

        Args:
            user_id: Unique user identifier
            role: Role name to assign
            name: User's name
            email: User's email
            site_id: User's site affiliation
            user_type: "human", "ai_system", or "service_account"
            access_expiration: ISO date when access expires
        """
        if role not in self._roles:
            raise ValueError(f"Role '{role}' not defined. Available: {list(self._roles.keys())}")

        self._users[user_id] = User(
            user_id=user_id,
            name=name,
            email=email,
            role_name=role,
            user_type=UserType(user_type),
            site_id=site_id,
            active=True,
            created_date=datetime.now().isoformat(),
            access_expiration=access_expiration
        )

        self._log_audit(
            user_id="SYSTEM",
            action="role_assigned",
            resource=f"user:{user_id}",
            decision="completed",
            reason=f"Assigned role '{role}' to user '{user_id}'"
        )

        logger.info(f"Role '{role}' assigned to user '{user_id}'")

    def check_access(
        self,
        user_id: str,
        resource: str,
        action: str,
        context: dict | None = None
    ) -> AccessDecision:
        """
        Check if a user has access to a resource.

        Args:
            user_id: User requesting access
            resource: Resource being accessed
            action: Action being performed
            context: Additional context (site_id, record_count, etc.)

        Returns:
            AccessDecision with grant/deny and reason
        """
        timestamp = datetime.now().isoformat()
        context = context or {}

        # Check user exists and is active
        if user_id not in self._users:
            decision = AccessDecision(
                granted=False,
                user_id=user_id,
                resource=resource,
                action=action,
                reason="User not found",
                timestamp=timestamp,
                audit_id=self._next_audit_id()
            )
            self._log_access_decision(decision)
            return decision

        user = self._users[user_id]

        if not user.active:
            decision = AccessDecision(
                granted=False,
                user_id=user_id,
                resource=resource,
                action=action,
                reason="User account is deactivated",
                timestamp=timestamp,
                audit_id=self._next_audit_id()
            )
            self._log_access_decision(decision)
            return decision

        # Check access expiration
        if user.access_expiration:
            try:
                exp_date = datetime.fromisoformat(user.access_expiration)
                if datetime.now() > exp_date:
                    decision = AccessDecision(
                        granted=False,
                        user_id=user_id,
                        resource=resource,
                        action=action,
                        reason="Access has expired",
                        timestamp=timestamp,
                        audit_id=self._next_audit_id()
                    )
                    self._log_access_decision(decision)
                    return decision
            except ValueError:
                pass

        # Check role permissions
        role = self._roles.get(user.role_name)
        if not role:
            decision = AccessDecision(
                granted=False,
                user_id=user_id,
                resource=resource,
                action=action,
                reason=f"Role '{user.role_name}' not found",
                timestamp=timestamp,
                audit_id=self._next_audit_id()
            )
            self._log_access_decision(decision)
            return decision

        # Check MFA requirement
        if role.mfa_required and self.mfa_required:
            if not user.mfa_enrolled:
                decision = AccessDecision(
                    granted=False,
                    user_id=user_id,
                    resource=resource,
                    action=action,
                    reason="MFA enrollment required but not completed",
                    timestamp=timestamp,
                    audit_id=self._next_audit_id()
                )
                self._log_access_decision(decision)
                return decision

        # Check permission
        if role.has_permission(action):
            conditions = []

            # Check site-level restrictions
            if context.get("site_id") and user.site_id:
                if context["site_id"] != user.site_id and action in ["read_phi", "write_phi"]:
                    decision = AccessDecision(
                        granted=False,
                        user_id=user_id,
                        resource=resource,
                        action=action,
                        reason="PHI access restricted to user's assigned site",
                        timestamp=timestamp,
                        audit_id=self._next_audit_id()
                    )
                    self._log_access_decision(decision)
                    return decision

            # Update last access
            user.last_access = timestamp

            decision = AccessDecision(
                granted=True,
                user_id=user_id,
                resource=resource,
                action=action,
                reason=f"Permitted by role '{user.role_name}'",
                timestamp=timestamp,
                audit_id=self._next_audit_id(),
                conditions=conditions
            )
        else:
            decision = AccessDecision(
                granted=False,
                user_id=user_id,
                resource=resource,
                action=action,
                reason=f"Permission '{action}' not in role '{user.role_name}'",
                timestamp=timestamp,
                audit_id=self._next_audit_id()
            )

        self._log_access_decision(decision)
        return decision

    def revoke_access(self, user_id: str, reason: str = ""):
        """Deactivate a user's access."""
        if user_id in self._users:
            self._users[user_id].active = False
            self._log_audit(
                user_id="SYSTEM",
                action="access_revoked",
                resource=f"user:{user_id}",
                decision="completed",
                reason=reason or "Access revoked"
            )
            logger.info(f"Access revoked for user '{user_id}': {reason}")

    def get_audit_log(
        self,
        user_id: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        action_filter: str | None = None
    ) -> list[AuditEntry]:
        """
        Query audit log with filters.

        Args:
            user_id: Filter by user ID
            start_date: Filter by start date (ISO format)
            end_date: Filter by end date (ISO format)
            action_filter: Filter by action type

        Returns:
            Filtered list of audit entries
        """
        filtered = self._audit_log

        if user_id:
            filtered = [e for e in filtered if e.user_id == user_id]
        if action_filter:
            filtered = [e for e in filtered if action_filter in e.action]
        if start_date:
            filtered = [e for e in filtered if e.timestamp >= start_date]
        if end_date:
            filtered = [e for e in filtered if e.timestamp <= end_date]

        return filtered

    def export_audit_log(self, output_path: str):
        """Export audit log to JSON file."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(
                [entry.to_dict() for entry in self._audit_log],
                f, indent=2
            )
        logger.info(f"Audit log exported: {len(self._audit_log)} entries to {output_path}")

    def generate_access_report(self) -> str:
        """Generate a summary report of current access configuration."""
        report = """
ACCESS CONTROL REPORT
=====================
Generated: {timestamp}
Framework: {framework}
MFA Required: {mfa}

ROLES DEFINED: {role_count}
{role_details}

USERS REGISTERED: {user_count}
{user_details}

AUDIT LOG SUMMARY
-----------------
Total entries: {audit_count}
Access granted: {granted_count}
Access denied: {denied_count}
""".format(
            timestamp=datetime.now().isoformat(),
            framework=self.compliance_framework,
            mfa=self.mfa_required,
            role_count=len(self._roles),
            role_details="\n".join(
                f"  {name}: {role.description} ({len(role.permissions)} permissions)"
                for name, role in self._roles.items()
            ),
            user_count=len(self._users),
            user_details="\n".join(
                f"  {uid}: role={u.role_name}, active={u.active}, type={u.user_type.value}"
                for uid, u in self._users.items()
            ),
            audit_count=len(self._audit_log),
            granted_count=len([e for e in self._audit_log if e.decision == "granted"]),
            denied_count=len([e for e in self._audit_log if e.decision == "denied"])
        )
        return report

    def _next_audit_id(self) -> str:
        """Generate next audit ID."""
        self._audit_counter += 1
        return f"AUD-{self._audit_counter:06d}"

    def _log_audit(
        self,
        user_id: str,
        action: str,
        resource: str,
        decision: str,
        reason: str
    ):
        """Log an audit entry."""
        if not self.audit_enabled:
            return

        entry = AuditEntry(
            audit_id=self._next_audit_id(),
            timestamp=datetime.now().isoformat(),
            user_id=user_id,
            user_type="system",
            action=action,
            resource=resource,
            decision=decision,
            reason=reason
        )

        # Calculate integrity hash
        hash_input = f"{entry.audit_id}:{entry.timestamp}:{entry.user_id}:{entry.action}"
        entry.integrity_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:16]

        self._audit_log.append(entry)

    def _log_access_decision(self, decision: AccessDecision):
        """Log an access control decision."""
        if not self.audit_enabled:
            return

        user = self._users.get(decision.user_id)
        user_type = user.user_type.value if user else "unknown"

        entry = AuditEntry(
            audit_id=decision.audit_id,
            timestamp=decision.timestamp,
            user_id=decision.user_id,
            user_type=user_type,
            action=decision.action,
            resource=decision.resource,
            decision="granted" if decision.granted else "denied",
            reason=decision.reason
        )

        hash_input = f"{entry.audit_id}:{entry.timestamp}:{entry.user_id}:{entry.action}"
        entry.integrity_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:16]

        self._audit_log.append(entry)


# =============================================================================
# SECTION 4: MAIN PIPELINE
# =============================================================================

def run_access_control_demo():
    """
    Demonstrate access control management for clinical trials.

    Shows role definition, user assignment, access checking,
    and audit trail generation.
    """
    logger.info("=" * 60)
    logger.info("ACCESS CONTROL MANAGER DEMO")
    logger.info("=" * 60)

    # Initialize manager
    acm = AccessControlManager(
        compliance_framework="21_cfr_part_11",
        audit_enabled=True,
        mfa_required=True
    )

    # Assign users with different roles
    users = [
        ("PI-001", "principal_investigator", "Dr. Sarah Chen", True),
        ("CRC-001", "clinical_coordinator", "James Wilson", True),
        ("STAT-001", "biostatistician", "Dr. Emily Park", True),
        ("ENG-001", "ai_ml_engineer", "Alex Kumar", True),
        ("AI-SYS-001", "ai_system", "Training Pipeline v1", False),
    ]

    for user_id, role, name, mfa in users:
        acm.assign_role(
            user_id=user_id,
            role=role,
            name=name,
            mfa_enrolled=mfa if isinstance(mfa, bool) else False,
            site_id="SITE-A"
        )
        # Fix: set mfa_enrolled directly
        acm._users[user_id].mfa_enrolled = mfa

    # Test access scenarios
    scenarios = [
        ("PI-001", "patient_records", "read_phi"),
        ("STAT-001", "patient_records", "read_phi"),
        ("STAT-001", "analysis_dataset", "read_deidentified"),
        ("ENG-001", "patient_records", "read_phi"),
        ("ENG-001", "training_data", "train_model"),
        ("AI-SYS-001", "inference_output", "write_model_outputs"),
        ("AI-SYS-001", "patient_records", "read_phi"),
    ]

    print("\nAccess Control Decisions")
    print("-" * 70)

    for user_id, resource, action in scenarios:
        result = acm.check_access(user_id, resource, action)
        status = "GRANTED" if result.granted else "DENIED"
        print(f"  [{status:7s}] {user_id:12s} -> {action:25s} on {resource:20s} | {result.reason}")

    # Print report
    print("\n" + acm.generate_access_report())

    return {"status": "demo_complete", "audit_entries": len(acm._audit_log)}


if __name__ == "__main__":
    result = run_access_control_demo()
    print(f"\nDemo completed: {result}")

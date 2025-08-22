"""
FlashMM Comprehensive Audit Trail System

Tamper-proof audit logging with compliance reporting, forensic analysis,
and comprehensive audit trail management for all system operations.
"""

import asyncio
import base64
import hashlib
import json
import secrets
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

import aiosqlite
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from flashmm.config.settings import get_config
from flashmm.security.encryption import DataEncryption
from flashmm.utils.exceptions import SecurityError
from flashmm.utils.logging import SecurityLogger

logger = SecurityLogger()


class AuditEventType(Enum):
    """Types of audit events."""
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    DATA_ACCESS = "data_access"
    DATA_MODIFICATION = "data_modification"
    SYSTEM_CONFIGURATION = "system_configuration"
    KEY_MANAGEMENT = "key_management"
    SECURITY_EVENT = "security_event"
    TRADING_OPERATION = "trading_operation"
    API_ACCESS = "api_access"
    ADMINISTRATIVE = "administrative"
    COMPLIANCE = "compliance"
    EMERGENCY = "emergency"


class AuditLevel(Enum):
    """Audit importance levels."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ComplianceFramework(Enum):
    """Supported compliance frameworks."""
    SOC2 = "soc2"
    ISO27001 = "iso27001"
    PCI_DSS = "pci_dss"
    GDPR = "gdpr"
    FFIEC = "ffiec"
    NIST = "nist"


@dataclass
class AuditEvent:
    """Comprehensive audit event structure."""
    event_id: str
    timestamp: datetime
    event_type: AuditEventType
    level: AuditLevel
    actor_id: str  # User, system, or service performing the action
    actor_type: str  # user, system, api_client, etc.
    source_ip: str | None
    user_agent: str | None
    session_id: str | None
    component: str  # System component that generated the event
    action: str  # Specific action performed
    resource: str | None  # Resource affected
    resource_type: str | None  # Type of resource
    outcome: str  # success, failure, partial
    risk_level: str  # low, medium, high, critical
    compliance_tags: list[str]  # Compliance framework tags
    metadata: dict[str, Any]  # Additional event-specific data
    hash_chain_prev: str | None = None  # Previous event hash for chain integrity
    hash_signature: str | None = None  # Cryptographic signature of this event


@dataclass
class ComplianceRule:
    """Compliance rule definition."""
    rule_id: str
    framework: ComplianceFramework
    rule_name: str
    description: str
    event_patterns: list[dict[str, Any]]  # Patterns to match events
    retention_period: timedelta
    alert_conditions: list[dict[str, Any]]
    required_fields: list[str]


class AuditIntegrityManager:
    """Manages audit trail integrity using cryptographic methods."""

    def __init__(self, encryption_key: str):
        self.data_encryption = DataEncryption(encryption_key)
        self.hash_chain: list[str] = []

        # Generate signing key pair for integrity
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        self.public_key = self.private_key.public_key()

    def sign_event(self, event: AuditEvent) -> str:
        """Create cryptographic signature for audit event."""
        # Create canonical representation
        event_data = {
            "event_id": event.event_id,
            "timestamp": event.timestamp.isoformat(),
            "event_type": event.event_type.value,
            "actor_id": event.actor_id,
            "component": event.component,
            "action": event.action,
            "outcome": event.outcome,
            "metadata": json.dumps(event.metadata, sort_keys=True)
        }

        canonical_json = json.dumps(event_data, sort_keys=True)
        message = canonical_json.encode('utf-8')

        # Sign the message
        signature = self.private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )

        return base64.b64encode(signature).decode('utf-8')

    def verify_event_signature(self, event: AuditEvent, signature: str) -> bool:
        """Verify cryptographic signature of audit event."""
        try:
            # Recreate canonical representation
            event_data = {
                "event_id": event.event_id,
                "timestamp": event.timestamp.isoformat(),
                "event_type": event.event_type.value,
                "actor_id": event.actor_id,
                "component": event.component,
                "action": event.action,
                "outcome": event.outcome,
                "metadata": json.dumps(event.metadata, sort_keys=True)
            }

            canonical_json = json.dumps(event_data, sort_keys=True)
            message = canonical_json.encode('utf-8')
            signature_bytes = base64.b64decode(signature)

            # Verify signature
            self.public_key.verify(
                signature_bytes,
                message,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )

            return True

        except Exception:
            return False

    def compute_event_hash(self, event: AuditEvent) -> str:
        """Compute hash for event in chain."""
        event_data = f"{event.event_id}|{event.timestamp.isoformat()}|{event.actor_id}|{event.action}"

        # Include previous hash in chain
        if self.hash_chain:
            event_data = f"{self.hash_chain[-1]}|{event_data}"

        event_hash = hashlib.sha256(event_data.encode()).hexdigest()
        self.hash_chain.append(event_hash)

        return event_hash

    def verify_hash_chain(self, events: list[AuditEvent]) -> bool:
        """Verify integrity of audit event hash chain."""
        if not events:
            return True

        # Rebuild hash chain
        rebuilt_chain = []

        for i, event in enumerate(events):
            event_data = f"{event.event_id}|{event.timestamp.isoformat()}|{event.actor_id}|{event.action}"

            if i > 0:
                event_data = f"{rebuilt_chain[i-1]}|{event_data}"

            computed_hash = hashlib.sha256(event_data.encode()).hexdigest()
            rebuilt_chain.append(computed_hash)

            # Verify against stored hash
            if event.hash_chain_prev and i > 0:
                if event.hash_chain_prev != rebuilt_chain[i-1]:
                    return False

        return True


class AuditStorage:
    """Secure, tamper-resistant audit log storage."""

    def __init__(self, db_path: str, encryption_key: str):
        self.db_path = db_path
        self.integrity_manager = AuditIntegrityManager(encryption_key)
        self.data_encryption = DataEncryption(encryption_key)

        # Ensure database directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    async def initialize(self) -> None:
        """Initialize audit storage database."""
        async with aiosqlite.connect(self.db_path) as db:
            # Create audit events table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS audit_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT UNIQUE NOT NULL,
                    timestamp REAL NOT NULL,
                    event_type TEXT NOT NULL,
                    level TEXT NOT NULL,
                    actor_id TEXT NOT NULL,
                    actor_type TEXT NOT NULL,
                    source_ip TEXT,
                    user_agent TEXT,
                    session_id TEXT,
                    component TEXT NOT NULL,
                    action TEXT NOT NULL,
                    resource TEXT,
                    resource_type TEXT,
                    outcome TEXT NOT NULL,
                    risk_level TEXT NOT NULL,
                    compliance_tags TEXT NOT NULL,
                    metadata_encrypted TEXT NOT NULL,
                    hash_chain_prev TEXT,
                    hash_signature TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    INDEX(timestamp),
                    INDEX(event_type),
                    INDEX(actor_id),
                    INDEX(component),
                    INDEX(compliance_tags)
                )
            """)

            # Create audit integrity log
            await db.execute("""
                CREATE TABLE IF NOT EXISTS audit_integrity (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    check_timestamp REAL NOT NULL,
                    events_checked INTEGER NOT NULL,
                    integrity_status TEXT NOT NULL,
                    hash_chain_valid BOOLEAN NOT NULL,
                    signature_failures INTEGER NOT NULL,
                    details TEXT
                )
            """)

            # Create compliance reports table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS compliance_reports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    report_id TEXT UNIQUE NOT NULL,
                    framework TEXT NOT NULL,
                    period_start REAL NOT NULL,
                    period_end REAL NOT NULL,
                    generated_at REAL NOT NULL,
                    generated_by TEXT NOT NULL,
                    report_data_encrypted TEXT NOT NULL,
                    report_hash TEXT NOT NULL
                )
            """)

            await db.commit()

    async def store_event(self, event: AuditEvent) -> None:
        """Store audit event with integrity protection."""
        # Set hash chain reference
        if self.integrity_manager.hash_chain:
            event.hash_chain_prev = self.integrity_manager.hash_chain[-1]

        # Compute and set event hash
        _event_hash = self.integrity_manager.compute_event_hash(event)

        # Generate cryptographic signature
        event.hash_signature = self.integrity_manager.sign_event(event)

        # Encrypt sensitive metadata
        metadata_encrypted = self.data_encryption.encrypt_dict(event.metadata)

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT INTO audit_events (
                    event_id, timestamp, event_type, level, actor_id, actor_type,
                    source_ip, user_agent, session_id, component, action,
                    resource, resource_type, outcome, risk_level, compliance_tags,
                    metadata_encrypted, hash_chain_prev, hash_signature, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                event.event_id,
                event.timestamp.timestamp(),
                event.event_type.value,
                event.level.value,
                event.actor_id,
                event.actor_type,
                event.source_ip,
                event.user_agent,
                event.session_id,
                event.component,
                event.action,
                event.resource,
                event.resource_type,
                event.outcome,
                event.risk_level,
                json.dumps(event.compliance_tags),
                metadata_encrypted,
                event.hash_chain_prev,
                event.hash_signature,
                time.time()
            ))

            await db.commit()

    async def get_events(self,
                        start_time: datetime | None = None,
                        end_time: datetime | None = None,
                        event_types: list[AuditEventType] | None = None,
                        actor_id: str | None = None,
                        component: str | None = None,
                        limit: int = 1000) -> list[AuditEvent]:
        """Retrieve audit events with filtering."""
        query = "SELECT * FROM audit_events WHERE 1=1"
        params = []

        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time.timestamp())

        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time.timestamp())

        if event_types:
            placeholders = ",".join("?" * len(event_types))
            query += f" AND event_type IN ({placeholders})"
            params.extend([et.value for et in event_types])

        if actor_id:
            query += " AND actor_id = ?"
            params.append(actor_id)

        if component:
            query += " AND component = ?"
            params.append(component)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(query, params)
            rows = await cursor.fetchall()

            events = []
            for row in rows:
                # Decrypt metadata
                metadata = self.data_encryption.decrypt_dict(row[16])

                event = AuditEvent(
                    event_id=row[1],
                    timestamp=datetime.fromtimestamp(row[2]),
                    event_type=AuditEventType(row[3]),
                    level=AuditLevel(row[4]),
                    actor_id=row[5],
                    actor_type=row[6],
                    source_ip=row[7],
                    user_agent=row[8],
                    session_id=row[9],
                    component=row[10],
                    action=row[11],
                    resource=row[12],
                    resource_type=row[13],
                    outcome=row[14],
                    risk_level=row[15],
                    compliance_tags=json.loads(row[17]),
                    metadata=metadata,
                    hash_chain_prev=row[18],
                    hash_signature=row[19]
                )

                events.append(event)

            return events

    async def verify_integrity(self) -> dict[str, Any]:
        """Verify integrity of audit trail."""
        async with aiosqlite.connect(self.db_path) as db:
            # Get all events ordered by timestamp
            cursor = await db.execute("""
                SELECT * FROM audit_events ORDER BY timestamp ASC
            """)
            rows = await cursor.fetchall()

            events = []
            signature_failures = 0

            for row in rows:
                # Decrypt metadata for verification
                metadata = self.data_encryption.decrypt_dict(row[16])

                event = AuditEvent(
                    event_id=row[1],
                    timestamp=datetime.fromtimestamp(row[2]),
                    event_type=AuditEventType(row[3]),
                    level=AuditLevel(row[4]),
                    actor_id=row[5],
                    actor_type=row[6],
                    source_ip=row[7],
                    user_agent=row[8],
                    session_id=row[9],
                    component=row[10],
                    action=row[11],
                    resource=row[12],
                    resource_type=row[13],
                    outcome=row[14],
                    risk_level=row[15],
                    compliance_tags=json.loads(row[17]),
                    metadata=metadata,
                    hash_chain_prev=row[18],
                    hash_signature=row[19]
                )

                # Verify signature
                if event.hash_signature and not self.integrity_manager.verify_event_signature(event, event.hash_signature):
                    signature_failures += 1

                events.append(event)

            # Verify hash chain
            hash_chain_valid = self.integrity_manager.verify_hash_chain(events)

            integrity_result = {
                "events_checked": len(events),
                "hash_chain_valid": hash_chain_valid,
                "signature_failures": signature_failures,
                "integrity_status": "valid" if hash_chain_valid and signature_failures == 0 else "compromised",
                "check_timestamp": datetime.utcnow().isoformat()
            }

            # Store integrity check result
            await db.execute("""
                INSERT INTO audit_integrity (
                    check_timestamp, events_checked, integrity_status,
                    hash_chain_valid, signature_failures, details
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                time.time(),
                len(events),
                integrity_result["integrity_status"],
                hash_chain_valid,
                signature_failures,
                json.dumps(integrity_result)
            ))

            await db.commit()

            return integrity_result


class ComplianceReporter:
    """Generates compliance reports for various frameworks."""

    def __init__(self, audit_storage: AuditStorage):
        self.audit_storage = audit_storage

        # Define compliance rules for different frameworks
        self.compliance_rules = {
            ComplianceFramework.SOC2: [
                ComplianceRule(
                    rule_id="soc2_cc6_1",
                    framework=ComplianceFramework.SOC2,
                    rule_name="Logical and Physical Access Controls",
                    description="System access attempts and authorization changes",
                    event_patterns=[
                        {"event_type": "authentication"},
                        {"event_type": "authorization"},
                        {"action": "login"},
                        {"action": "privilege_change"}
                    ],
                    retention_period=timedelta(days=365),
                    alert_conditions=[
                        {"outcome": "failure", "count_threshold": 5}
                    ],
                    required_fields=["actor_id", "source_ip", "timestamp", "outcome"]
                ),
                ComplianceRule(
                    rule_id="soc2_cc7_1",
                    framework=ComplianceFramework.SOC2,
                    rule_name="System Operations",
                    description="System configuration and operational changes",
                    event_patterns=[
                        {"event_type": "system_configuration"},
                        {"event_type": "administrative"}
                    ],
                    retention_period=timedelta(days=365),
                    alert_conditions=[],
                    required_fields=["actor_id", "action", "resource", "timestamp"]
                )
            ],
            ComplianceFramework.ISO27001: [
                ComplianceRule(
                    rule_id="iso27001_a9_2",
                    framework=ComplianceFramework.ISO27001,
                    rule_name="User Access Management",
                    description="User access provisioning and review",
                    event_patterns=[
                        {"event_type": "authentication"},
                        {"event_type": "authorization"},
                        {"action": "user_create"},
                        {"action": "user_delete"},
                        {"action": "permission_change"}
                    ],
                    retention_period=timedelta(days=1095),  # 3 years
                    alert_conditions=[
                        {"action": "admin_access", "risk_level": "high"}
                    ],
                    required_fields=["actor_id", "action", "outcome", "timestamp"]
                )
            ]
        }

    async def generate_compliance_report(self,
                                       framework: ComplianceFramework,
                                       start_date: datetime,
                                       end_date: datetime,
                                       generated_by: str) -> dict[str, Any]:
        """Generate comprehensive compliance report."""

        rules = self.compliance_rules.get(framework, [])
        if not rules:
            raise SecurityError(f"No compliance rules defined for {framework.value}")

        report_data = {
            "framework": framework.value,
            "period_start": start_date.isoformat(),
            "period_end": end_date.isoformat(),
            "generated_at": datetime.utcnow().isoformat(),
            "generated_by": generated_by,
            "rule_compliance": {},
            "violations": [],
            "statistics": {},
            "recommendations": []
        }

        for rule in rules:
            rule_events = await self._get_events_for_rule(rule, start_date, end_date)
            compliance_result = await self._evaluate_rule_compliance(rule, rule_events)

            report_data["rule_compliance"][rule.rule_id] = {
                "rule_name": rule.rule_name,
                "description": rule.description,
                "events_found": len(rule_events),
                "compliant": compliance_result["compliant"],
                "violations": compliance_result["violations"],
                "details": compliance_result["details"]
            }

            report_data["violations"].extend(compliance_result["violations"])

        # Generate statistics
        all_events = await self.audit_storage.get_events(
            start_time=start_date,
            end_time=end_date,
            limit=10000
        )

        report_data["statistics"] = {
            "total_events": len(all_events),
            "authentication_events": len([e for e in all_events if e.event_type == AuditEventType.AUTHENTICATION]),
            "failed_events": len([e for e in all_events if e.outcome == "failure"]),
            "high_risk_events": len([e for e in all_events if e.risk_level == "high"]),
            "unique_actors": len({e.actor_id for e in all_events}),
            "components_accessed": len({e.component for e in all_events})
        }

        # Generate recommendations
        report_data["recommendations"] = await self._generate_recommendations(
            framework, all_events, report_data["violations"]
        )

        # Store report
        report_id = f"{framework.value}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}_{secrets.token_hex(8)}"

        # Encrypt report data
        report_encrypted = self.audit_storage.data_encryption.encrypt_dict(report_data)
        report_hash = hashlib.sha256(json.dumps(report_data, sort_keys=True).encode()).hexdigest()

        async with aiosqlite.connect(self.audit_storage.db_path) as db:
            await db.execute("""
                INSERT INTO compliance_reports (
                    report_id, framework, period_start, period_end,
                    generated_at, generated_by, report_data_encrypted, report_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                report_id,
                framework.value,
                start_date.timestamp(),
                end_date.timestamp(),
                time.time(),
                generated_by,
                report_encrypted,
                report_hash
            ))

            await db.commit()

        report_data["report_id"] = report_id
        return report_data

    async def _get_events_for_rule(self, rule: ComplianceRule,
                                  start_date: datetime,
                                  end_date: datetime) -> list[AuditEvent]:
        """Get events matching compliance rule patterns."""
        all_events = await self.audit_storage.get_events(
            start_time=start_date,
            end_time=end_date,
            limit=10000
        )

        matching_events = []

        for event in all_events:
            for pattern in rule.event_patterns:
                if self._event_matches_pattern(event, pattern):
                    matching_events.append(event)
                    break

        return matching_events

    def _event_matches_pattern(self, event: AuditEvent, pattern: dict[str, Any]) -> bool:
        """Check if event matches compliance rule pattern."""
        for key, value in pattern.items():
            if key == "event_type":
                if event.event_type.value != value:
                    return False
            elif key == "action":
                if event.action != value:
                    return False
            elif key == "outcome":
                if event.outcome != value:
                    return False
            elif key == "risk_level":
                if event.risk_level != value:
                    return False
            # Add more pattern matching as needed

        return True

    async def _evaluate_rule_compliance(self, rule: ComplianceRule,
                                      events: list[AuditEvent]) -> dict[str, Any]:
        """Evaluate compliance for a specific rule."""
        violations = []
        compliant = True

        # Check for missing required fields
        for event in events:
            missing_fields = []
            for field in rule.required_fields:
                if not hasattr(event, field) or getattr(event, field) is None:
                    missing_fields.append(field)

            if missing_fields:
                violations.append({
                    "event_id": event.event_id,
                    "violation_type": "missing_required_fields",
                    "details": f"Missing fields: {missing_fields}",
                    "timestamp": event.timestamp.isoformat()
                })
                compliant = False

        # Check alert conditions
        for condition in rule.alert_conditions:
            condition_violations = self._check_alert_condition(events, condition)
            violations.extend(condition_violations)
            if condition_violations:
                compliant = False

        return {
            "compliant": compliant,
            "violations": violations,
            "details": {
                "events_evaluated": len(events),
                "violation_count": len(violations)
            }
        }

    def _check_alert_condition(self, events: list[AuditEvent],
                              condition: dict[str, Any]) -> list[dict[str, Any]]:
        """Check specific alert condition against events."""
        violations = []

        if "count_threshold" in condition:
            matching_events = [
                e for e in events
                if all(getattr(e, k, None) == v for k, v in condition.items() if k != "count_threshold")
            ]

            if len(matching_events) >= condition["count_threshold"]:
                violations.append({
                    "violation_type": "threshold_exceeded",
                    "condition": condition,
                    "actual_count": len(matching_events),
                    "threshold": condition["count_threshold"],
                    "events": [e.event_id for e in matching_events]
                })

        return violations

    async def _generate_recommendations(self, framework: ComplianceFramework,
                                      events: list[AuditEvent],
                                      violations: list[dict[str, Any]]) -> list[str]:
        """Generate compliance recommendations based on analysis."""
        recommendations = []

        # Analyze patterns and generate recommendations
        failed_logins = [e for e in events if e.action == "login" and e.outcome == "failure"]
        if len(failed_logins) > 100:
            recommendations.append(
                "Consider implementing account lockout policies due to high number of failed login attempts"
            )

        high_risk_events = [e for e in events if e.risk_level == "high"]
        if len(high_risk_events) > len(events) * 0.1:  # More than 10% high risk
            recommendations.append(
                "Review security controls as high-risk events exceed 10% of total activity"
            )

        if violations:
            recommendations.append(
                f"Address {len(violations)} compliance violations identified in this report"
            )

        unique_actors = {e.actor_id for e in events}
        if len(unique_actors) > 1000:
            recommendations.append(
                "Consider implementing user access reviews due to large number of active users"
            )

        return recommendations


class ForensicAnalyzer:
    """Forensic analysis capabilities for audit trails."""

    def __init__(self, audit_storage: AuditStorage):
        self.audit_storage = audit_storage

    async def investigate_incident(self,
                                 incident_id: str,
                                 start_time: datetime,
                                 end_time: datetime,
                                 related_actors: list[str] | None = None,
                                 related_resources: list[str] | None = None) -> dict[str, Any]:
        """Conduct forensic investigation of security incident."""

        # Get all events in the time window
        all_events = await self.audit_storage.get_events(
            start_time=start_time,
            end_time=end_time,
            limit=50000
        )

        # Filter for relevant events
        relevant_events = []
        for event in all_events:
            if related_actors and event.actor_id in related_actors:
                relevant_events.append(event)
            elif related_resources and event.resource and event.resource in related_resources:
                relevant_events.append(event)
            elif event.risk_level in ["high", "critical"]:
                relevant_events.append(event)

        # Analyze event timeline
        timeline = self._build_event_timeline(relevant_events)

        # Identify suspicious patterns
        suspicious_patterns = await self._identify_suspicious_patterns(relevant_events)

        # Analyze actor behavior
        actor_analysis = self._analyze_actor_behavior(relevant_events)

        # Generate investigation report
        investigation_report = {
            "incident_id": incident_id,
            "investigation_period": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat()
            },
            "total_events_analyzed": len(all_events),
            "relevant_events": len(relevant_events),
            "timeline": timeline,
            "suspicious_patterns": suspicious_patterns,
            "actor_analysis": actor_analysis,
            "key_findings": self._generate_key_findings(relevant_events, suspicious_patterns),
            "recommendations": self._generate_forensic_recommendations(suspicious_patterns, actor_analysis)
        }

        return investigation_report

    def _build_event_timeline(self, events: list[AuditEvent]) -> list[dict[str, Any]]:
        """Build chronological timeline of events."""
        sorted_events = sorted(events, key=lambda e: e.timestamp)

        timeline = []
        for event in sorted_events:
            timeline.append({
                "timestamp": event.timestamp.isoformat(),
                "event_id": event.event_id,
                "actor_id": event.actor_id,
                "action": event.action,
                "component": event.component,
                "outcome": event.outcome,
                "risk_level": event.risk_level,
                "resource": event.resource
            })

        return timeline

    async def _identify_suspicious_patterns(self, events: list[AuditEvent]) -> list[dict[str, Any]]:
        """Identify suspicious patterns in event data."""
        patterns = []

        # Group events by actor
        actor_events = {}
        for event in events:
            if event.actor_id not in actor_events:
                actor_events[event.actor_id] = []
            actor_events[event.actor_id].append(event)

        # Look for suspicious patterns
        for actor_id, actor_event_list in actor_events.items():
            # Rapid succession of failed attempts
            failed_events = [e for e in actor_event_list if e.outcome == "failure"]
            if len(failed_events) >= 5:
                failed_times = [e.timestamp for e in failed_events]
                failed_times.sort()

                # Check if failures occurred within 5 minutes
                time_window = timedelta(minutes=5)
                rapid_failures = []

                for i in range(len(failed_times) - 4):
                    if failed_times[i + 4] - failed_times[i] <= time_window:
                        rapid_failures = failed_times[i:i+5]
                        break

                if rapid_failures:
                    patterns.append({
                        "pattern_type": "rapid_failed_attempts",
                        "actor_id": actor_id,
                        "description": "5 failed attempts within 5 minutes",
                        "event_count": len(failed_events),
                        "time_window": f"{rapid_failures[0].isoformat()} - {rapid_failures[-1].isoformat()}"
                    })

            # Privilege escalation patterns
            privilege_events = [e for e in actor_event_list if "privilege" in e.action.lower() or "admin" in e.action.lower()]
            if privilege_events:
                patterns.append({
                    "pattern_type": "privilege_operations",
                    "actor_id": actor_id,
                    "description": "Privilege-related operations detected",
                    "event_count": len(privilege_events),
                    "actions": list({e.action for e in privilege_events})
                })

            # Unusual time patterns
            off_hours_events = [
                e for e in actor_event_list
                if e.timestamp.hour < 6 or e.timestamp.hour > 22
            ]

            if len(off_hours_events) > len(actor_event_list) * 0.8:  # 80% off-hours activity
                patterns.append({
                    "pattern_type": "off_hours_activity",
                    "actor_id": actor_id,
                    "description": "High percentage of off-hours activity",
                    "off_hours_percentage": len(off_hours_events) / len(actor_event_list) * 100,
                    "event_count": len(off_hours_events)
                })

        return patterns

    def _analyze_actor_behavior(self, events: list[AuditEvent]) -> dict[str, Any]:
        """Analyze behavior patterns for each actor."""
        actor_analysis = {}

        # Group events by actor
        actor_events = {}
        for event in events:
            if event.actor_id not in actor_events:
                actor_events[event.actor_id] = []
            actor_events[event.actor_id].append(event)

        for actor_id, actor_event_list in actor_events.items():
            # Calculate behavior metrics
            total_events = len(actor_event_list)
            failed_events = len([e for e in actor_event_list if e.outcome == "failure"])
            high_risk_events = len([e for e in actor_event_list if e.risk_level == "high"])

            unique_components = len({e.component for e in actor_event_list})
            unique_actions = len({e.action for e in actor_event_list})
            unique_resources = len({e.resource for e in actor_event_list if e.resource})

            # Time analysis
            timestamps = [e.timestamp for e in actor_event_list]
            timestamps.sort()

            if len(timestamps) > 1:
                time_span = timestamps[-1] - timestamps[0]
                activity_rate = total_events / max(time_span.total_seconds() / 3600, 1)  # Events per hour
            else:
                time_span = timedelta(0)
                activity_rate = 0

            actor_analysis[actor_id] = {
                "total_events": total_events,
                "failed_events": failed_events,
                "failure_rate": failed_events / total_events if total_events > 0 else 0,
                "high_risk_events": high_risk_events,
                "risk_rate": high_risk_events / total_events if total_events > 0 else 0,
                "unique_components": unique_components,
                "unique_actions": unique_actions,
                "unique_resources": unique_resources,
                "time_span_hours": time_span.total_seconds() / 3600,
                "activity_rate_per_hour": activity_rate,
                "first_activity": timestamps[0].isoformat() if timestamps else None,
                "last_activity": timestamps[-1].isoformat() if timestamps else None
            }

        return actor_analysis

    def _generate_key_findings(self, events: list[AuditEvent],
                              suspicious_patterns: list[dict[str, Any]]) -> list[str]:
        """Generate key findings from forensic analysis."""
        findings = []

        if suspicious_patterns:
            findings.append(f"Identified {len(suspicious_patterns)} suspicious behavior patterns")

            for pattern in suspicious_patterns:
                if pattern["pattern_type"] == "rapid_failed_attempts":
                    findings.append(f"Actor {pattern['actor_id']} showed brute force attack behavior")
                elif pattern["pattern_type"] == "privilege_operations":
                    findings.append(f"Actor {pattern['actor_id']} performed privilege escalation operations")
                elif pattern["pattern_type"] == "off_hours_activity":
                    findings.append(f"Actor {pattern['actor_id']} has {pattern['off_hours_percentage']:.1f}% off-hours activity")

        high_risk_events = [e for e in events if e.risk_level == "high"]
        if high_risk_events:
            findings.append(f"Found {len(high_risk_events)} high-risk security events")

        failed_events = [e for e in events if e.outcome == "failure"]
        if len(failed_events) > len(events) * 0.2:  # More than 20% failures
            findings.append(f"High failure rate detected: {len(failed_events)}/{len(events)} events failed")

        unique_actors = {e.actor_id for e in events}
        if len(unique_actors) == 1:
            findings.append(f"All suspicious activity traced to single actor: {list(unique_actors)[0]}")

        return findings

    def _generate_forensic_recommendations(self, suspicious_patterns: list[dict[str, Any]],
                                         actor_analysis: dict[str, Any]) -> list[str]:
        """Generate forensic investigation recommendations."""
        recommendations = []

        for pattern in suspicious_patterns:
            if pattern["pattern_type"] == "rapid_failed_attempts":
                recommendations.append(f"Implement account lockout for actor {pattern['actor_id']}")
                recommendations.append("Review and strengthen password policies")

            elif pattern["pattern_type"] == "privilege_operations":
                recommendations.append(f"Review all privilege changes for actor {pattern['actor_id']}")
                recommendations.append("Implement additional approval workflow for privilege escalation")

            elif pattern["pattern_type"] == "off_hours_activity":
                recommendations.append(f"Verify legitimacy of off-hours access for actor {pattern['actor_id']}")
                recommendations.append("Consider implementing time-based access restrictions")

        # Analyze actor behavior for recommendations
        for actor_id, analysis in actor_analysis.items():
            if analysis["failure_rate"] > 0.3:  # More than 30% failures
                recommendations.append(f"Investigate high failure rate for actor {actor_id}")

            if analysis["risk_rate"] > 0.1:  # More than 10% high-risk events
                recommendations.append(f"Review security controls for high-risk actor {actor_id}")

            if analysis["activity_rate_per_hour"] > 100:  # Very high activity
                recommendations.append(f"Verify automated vs manual activity for actor {actor_id}")

        return list(set(recommendations))  # Remove duplicates


class AuditLogger:
    """Main audit logging system with comprehensive audit trail management."""

    def __init__(self):
        self.config = get_config()

        # Initialize storage
        db_path = self.config.get("security.audit_db_path", "data/audit.db")
        encryption_key = self.config.get("security.audit_encryption_key", "default_audit_key")

        self.storage = AuditStorage(db_path, encryption_key)
        self.compliance_reporter = ComplianceReporter(self.storage)
        self.forensic_analyzer = ForensicAnalyzer(self.storage)

        # Background tasks
        self._background_tasks: list[asyncio.Task] = []
        self._shutdown_event = asyncio.Event()

        # Event buffer for batch processing
        self.event_buffer: list[AuditEvent] = []
        self.buffer_lock = asyncio.Lock()
        self.batch_size = 100

        # Performance metrics
        self.metrics = {
            "events_logged": 0,
            "events_failed": 0,
            "integrity_checks": 0,
            "compliance_reports": 0
        }

    async def initialize(self) -> None:
        """Initialize the audit logging system."""
        await self.storage.initialize()

        # Start background tasks
        self._background_tasks = [
            asyncio.create_task(self._batch_processor()),
            asyncio.create_task(self._integrity_monitor()),
            asyncio.create_task(self._cleanup_old_data())
        ]

        await logger.log_critical_event(
            "audit_logger_initialized",
            "audit_logger",
            {"timestamp": datetime.utcnow().isoformat()}
        )

    async def shutdown(self) -> None:
        """Shutdown audit logging system."""
        self._shutdown_event.set()

        # Process remaining events in buffer
        await self._process_buffer()

        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()

        await asyncio.gather(*self._background_tasks, return_exceptions=True)

        await logger.log_critical_event(
            "audit_logger_shutdown",
            "audit_logger",
            {"timestamp": datetime.utcnow().isoformat()}
        )

    async def log_event(self,
                       event_type: AuditEventType,
                       actor_id: str,
                       action: str,
                       component: str,
                       outcome: str = "success",
                       actor_type: str = "user",
                       source_ip: str | None = None,
                       user_agent: str | None = None,
                       session_id: str | None = None,
                       resource: str | None = None,
                       resource_type: str | None = None,
                       risk_level: str = "low",
                       compliance_tags: list[str] | None = None,
                       metadata: dict[str, Any] | None = None,
                       level: AuditLevel = AuditLevel.INFO) -> str:
        """Log an audit event."""

        event_id = f"{event_type.value}_{secrets.token_hex(8)}_{int(time.time())}"

        event = AuditEvent(
            event_id=event_id,
            timestamp=datetime.utcnow(),
            event_type=event_type,
            level=level,
            actor_id=actor_id,
            actor_type=actor_type,
            source_ip=source_ip,
            user_agent=user_agent,
            session_id=session_id,
            component=component,
            action=action,
            resource=resource,
            resource_type=resource_type,
            outcome=outcome,
            risk_level=risk_level,
            compliance_tags=compliance_tags or [],
            metadata=metadata or {}
        )

        # Add to buffer for batch processing
        async with self.buffer_lock:
            self.event_buffer.append(event)

            # Process buffer if it's full
            if len(self.event_buffer) >= self.batch_size:
                await self._process_buffer()

        return event_id

    async def _batch_processor(self) -> None:
        """Background task to process event buffer."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(5)  # Process every 5 seconds

                async with self.buffer_lock:
                    if self.event_buffer:
                        await self._process_buffer()

            except Exception as e:
                await logger.log_critical_event(
                    "audit_batch_processor_error",
                    "audit_logger",
                    {"error": str(e)}
                )

    async def _process_buffer(self) -> None:
        """Process events in buffer."""
        if not self.event_buffer:
            return

        events_to_process = self.event_buffer.copy()
        self.event_buffer.clear()

        for event in events_to_process:
            try:
                await self.storage.store_event(event)
                self.metrics["events_logged"] += 1

            except Exception as e:
                self.metrics["events_failed"] += 1

                # Log the failure (but avoid infinite recursion)
                await logger.log_critical_event(
                    "audit_event_storage_failed",
                    "audit_logger",
                    {
                        "event_id": event.event_id,
                        "error": str(e)
                    }
                )

    async def _integrity_monitor(self) -> None:
        """Background task to monitor audit trail integrity."""
        while not self._shutdown_event.is_set():
            try:
                # Check integrity every hour
                await asyncio.sleep(3600)

                integrity_result = await self.storage.verify_integrity()
                self.metrics["integrity_checks"] += 1

                if integrity_result["integrity_status"] != "valid":
                    await logger.log_critical_event(
                        "audit_integrity_violation",
                        "audit_logger",
                        integrity_result
                    )

            except Exception as e:
                await logger.log_critical_event(
                    "audit_integrity_monitor_error",
                    "audit_logger",
                    {"error": str(e)}
                )

    async def _cleanup_old_data(self) -> None:
        """Background task to cleanup old audit data."""
        while not self._shutdown_event.is_set():
            try:
                # Cleanup every 24 hours
                await asyncio.sleep(86400)

                # Define retention periods by event type (for future use)
                _retention_periods = {
                    AuditEventType.AUTHENTICATION: timedelta(days=365),
                    AuditEventType.AUTHORIZATION: timedelta(days=365),
                    AuditEventType.SECURITY_EVENT: timedelta(days=1095),  # 3 years
                    AuditEventType.TRADING_OPERATION: timedelta(days=2555),  # 7 years
                    AuditEventType.KEY_MANAGEMENT: timedelta(days=1095),
                    AuditEventType.COMPLIANCE: timedelta(days=2555)
                }

                # Archive old events (implementation would depend on archival system)
                cutoff_date = datetime.utcnow() - timedelta(days=30)

                await logger.log_critical_event(
                    "audit_data_cleanup",
                    "audit_logger",
                    {"cutoff_date": cutoff_date.isoformat()}
                )

            except Exception as e:
                await logger.log_critical_event(
                    "audit_cleanup_error",
                    "audit_logger",
                    {"error": str(e)}
                )

    async def search_events(self,
                           start_time: datetime | None = None,
                           end_time: datetime | None = None,
                           event_types: list[AuditEventType] | None = None,
                           actor_id: str | None = None,
                           component: str | None = None,
                           action: str | None = None,
                           outcome: str | None = None,
                           risk_level: str | None = None,
                           limit: int = 1000) -> list[AuditEvent]:
        """Search audit events with comprehensive filtering."""

        # Get events from storage
        events = await self.storage.get_events(
            start_time=start_time,
            end_time=end_time,
            event_types=event_types,
            actor_id=actor_id,
            component=component,
            limit=limit * 2  # Get more to allow for additional filtering
        )

        # Apply additional filters
        filtered_events = []
        for event in events:
            if action and event.action != action:
                continue
            if outcome and event.outcome != outcome:
                continue
            if risk_level and event.risk_level != risk_level:
                continue

            filtered_events.append(event)

            if len(filtered_events) >= limit:
                break

        return filtered_events

    async def generate_compliance_report(self,
                                       framework: ComplianceFramework,
                                       start_date: datetime,
                                       end_date: datetime,
                                       generated_by: str) -> dict[str, Any]:
        """Generate compliance report."""
        report = await self.compliance_reporter.generate_compliance_report(
            framework, start_date, end_date, generated_by
        )

        self.metrics["compliance_reports"] += 1

        # Log report generation
        await self.log_event(
            event_type=AuditEventType.COMPLIANCE,
            actor_id=generated_by,
            action="generate_compliance_report",
            component="audit_logger",
            outcome="success",
            metadata={
                "framework": framework.value,
                "report_id": report["report_id"],
                "period_start": start_date.isoformat(),
                "period_end": end_date.isoformat()
            },
            compliance_tags=[framework.value]
        )

        return report

    async def conduct_forensic_investigation(self,
                                           incident_id: str,
                                           start_time: datetime,
                                           end_time: datetime,
                                           investigator_id: str,
                                           related_actors: list[str] | None = None,
                                           related_resources: list[str] | None = None) -> dict[str, Any]:
        """Conduct forensic investigation."""
        investigation = await self.forensic_analyzer.investigate_incident(
            incident_id, start_time, end_time, related_actors, related_resources
        )

        # Log investigation
        await self.log_event(
            event_type=AuditEventType.SECURITY_EVENT,
            actor_id=investigator_id,
            action="forensic_investigation",
            component="audit_logger",
            outcome="success",
            risk_level="high",
            metadata={
                "incident_id": incident_id,
                "investigation_period_start": start_time.isoformat(),
                "investigation_period_end": end_time.isoformat(),
                "related_actors": related_actors or [],
                "related_resources": related_resources or []
            },
            level=AuditLevel.CRITICAL
        )

        return investigation

    def get_audit_statistics(self) -> dict[str, Any]:
        """Get audit system statistics."""
        return {
            "metrics": self.metrics.copy(),
            "buffer_size": len(self.event_buffer),
            "background_tasks_active": len([t for t in self._background_tasks if not t.done()]),
            "storage_path": self.storage.db_path
        }

"""
FlashMM Comprehensive Security Module

Provides complete security infrastructure including:
- Advanced authentication with MFA and RBAC
- Enhanced key management with HSM integration
- Security orchestration and threat detection
- Security monitoring and intrusion detection
- Comprehensive audit logging and compliance
- Emergency procedures and incident response
- Security policy management and enforcement
- Penetration testing and vulnerability assessment
"""

from flashmm.security.audit_logger import (
    AuditEvent,
    AuditEventType,
    AuditLevel,
    AuditLogger,
    ComplianceFramework,
    ComplianceReporter,
    ForensicAnalyzer,
)
from flashmm.security.auth import (
    AuthenticationManager,
    AuthenticationMethod,
    AuthorizationManager,
    EnhancedSession,
    SessionState,
    UserRole,
)
from flashmm.security.emergency_manager import (
    DataProtectionManager,
    EmergencyCommunicator,
    EmergencyIncident,
    EmergencyLevel,
    EmergencyManager,
    EmergencyType,
    RecoveryManager,
    SystemShutdownManager,
    SystemState,
)
from flashmm.security.encryption import DataEncryption
from flashmm.security.key_manager import (
    ColdKeyManager,
    EnhancedKeyManager,
    HotKeyManager,
    HSMInterface,
    KeyEscrowManager,
    KeyRotationManager,
    KeySecurityLevel,
    KeyStatus,
    KeyType,
    WarmKeyManager,
)
from flashmm.security.pentest_framework import (
    FuzzingEngine,
    NetworkScanner,
    PenetrationTestFramework,
    SecurityTest,
    TestCategory,
    TestFinding,
    TestResult,
    TestSeverity,
    TestSuite,
    TestType,
    VulnerabilityScanner,
)
from flashmm.security.policy_engine import (
    EnforcementMode,
    PolicyConditionEvaluator,
    PolicyEnforcer,
    PolicyEngine,
    PolicyRule,
    PolicySeverity,
    PolicyStatus,
    PolicyType,
    PolicyViolation,
    SecurityPolicy,
)
from flashmm.security.security_monitor import (
    BehavioralAnalyzer,
    IntrusionDetector,
    MonitoringEvent,
    SecurityEvent,
    SecurityMonitor,
    ThreatLevel,
)
from flashmm.security.security_orchestrator import (
    SecurityLevel,
    SecurityMetrics,
    SecurityOrchestrator,
    SecurityState,
    SecurityThreat,
    ThreatType,
)

__all__ = [
    # Legacy key management
    "HotKeyManager",
    "WarmKeyManager",
    "ColdKeyManager",
    "KeyRotationManager",

    # Enhanced key management
    "EnhancedKeyManager",
    "KeyType",
    "KeySecurityLevel",
    "KeyStatus",
    "HSMInterface",
    "KeyEscrowManager",

    # Encryption
    "DataEncryption",

    # Authentication and Authorization
    "AuthenticationManager",
    "AuthorizationManager",
    "EnhancedSession",
    "AuthenticationMethod",
    "SessionState",
    "UserRole",

    # Security Orchestration
    "SecurityOrchestrator",
    "SecurityLevel",
    "ThreatType",
    "SecurityThreat",
    "SecurityState",
    "SecurityMetrics",

    # Security Monitoring
    "SecurityMonitor",
    "SecurityEvent",
    "MonitoringEvent",
    "ThreatLevel",
    "IntrusionDetector",
    "BehavioralAnalyzer",

    # Audit Logging
    "AuditLogger",
    "AuditEvent",
    "AuditEventType",
    "AuditLevel",
    "ComplianceFramework",
    "ComplianceReporter",
    "ForensicAnalyzer",

    # Emergency Management
    "EmergencyManager",
    "EmergencyIncident",
    "EmergencyLevel",
    "EmergencyType",
    "SystemState",
    "EmergencyCommunicator",
    "SystemShutdownManager",
    "DataProtectionManager",
    "RecoveryManager",

    # Policy Engine
    "PolicyEngine",
    "SecurityPolicy",
    "PolicyRule",
    "PolicyViolation",
    "PolicyType",
    "PolicySeverity",
    "PolicyStatus",
    "EnforcementMode",
    "PolicyConditionEvaluator",
    "PolicyEnforcer",

    # Penetration Testing
    "PenetrationTestFramework",
    "TestFinding",
    "SecurityTest",
    "TestSuite",
    "TestResult",
    "TestSeverity",
    "TestCategory",
    "TestType",
    "VulnerabilityScanner",
    "NetworkScanner",
    "FuzzingEngine",
]

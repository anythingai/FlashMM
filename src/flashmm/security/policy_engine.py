"""
FlashMM Security Policy Engine

Comprehensive security policy management system that defines, enforces, and monitors
security policies across all system components with violation detection and response.
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Set, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import re
import secrets
import hashlib

from flashmm.config.settings import get_config
from flashmm.utils.logging import SecurityLogger
from flashmm.utils.exceptions import SecurityError

logger = SecurityLogger()


class PolicyType(Enum):
    """Types of security policies."""
    ACCESS_CONTROL = "access_control"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    DATA_PROTECTION = "data_protection"
    NETWORK_SECURITY = "network_security"
    COMPLIANCE = "compliance"
    OPERATIONAL = "operational"
    EMERGENCY = "emergency"
    AUDIT = "audit"
    RISK_MANAGEMENT = "risk_management"


class PolicySeverity(Enum):
    """Severity levels for policy violations."""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class PolicyStatus(Enum):
    """Policy lifecycle status."""
    DRAFT = "draft"
    ACTIVE = "active"
    INACTIVE = "inactive"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


class EnforcementMode(Enum):
    """Policy enforcement modes."""
    MONITOR = "monitor"      # Log violations but allow
    WARN = "warn"           # Warn and log but allow
    BLOCK = "block"         # Block and log violations
    ENFORCE = "enforce"     # Strict enforcement


@dataclass
class PolicyRule:
    """Individual policy rule definition."""
    rule_id: str
    name: str
    description: str
    condition: Dict[str, Any]  # Rule condition specification
    action: str               # Action to take when rule matches
    severity: PolicySeverity
    enforcement_mode: EnforcementMode
    enabled: bool = True
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class SecurityPolicy:
    """Security policy definition."""
    policy_id: str
    name: str
    description: str
    policy_type: PolicyType
    version: str
    status: PolicyStatus
    created_at: datetime
    updated_at: datetime
    created_by: str
    rules: List[PolicyRule]
    applicable_components: List[str]
    exceptions: List[str] = None
    tags: List[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.exceptions is None:
            self.exceptions = []
        if self.tags is None:
            self.tags = []
        if self.metadata is None:
            self.metadata = {}


@dataclass
class PolicyViolation:
    """Policy violation record."""
    violation_id: str
    policy_id: str
    rule_id: str
    timestamp: datetime
    component: str
    user_id: Optional[str]
    resource: Optional[str]
    action: str
    severity: PolicySeverity
    description: str
    context: Dict[str, Any]
    resolved: bool = False
    resolution_notes: Optional[str] = None
    resolution_timestamp: Optional[datetime] = None


@dataclass
class PolicyException:
    """Policy exception definition."""
    exception_id: str
    policy_id: str
    rule_id: Optional[str]
    justification: str
    granted_by: str
    granted_to: str
    valid_from: datetime
    valid_until: datetime
    conditions: Dict[str, Any]
    approved_by: str
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class PolicyConditionEvaluator:
    """Evaluates policy rule conditions against events and contexts."""
    
    def __init__(self):
        self.operators = {
            "equals": self._op_equals,
            "not_equals": self._op_not_equals,
            "in": self._op_in,
            "not_in": self._op_not_in,
            "matches": self._op_matches,
            "contains": self._op_contains,
            "starts_with": self._op_starts_with,
            "ends_with": self._op_ends_with,
            "greater_than": self._op_greater_than,
            "less_than": self._op_less_than,
            "between": self._op_between,
            "exists": self._op_exists,
            "not_exists": self._op_not_exists,
            "all": self._op_all,
            "any": self._op_any,
            "time_window": self._op_time_window
        }
    
    def evaluate_condition(self, condition: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Evaluate a policy condition against the given context."""
        try:
            return self._evaluate_condition_recursive(condition, context)
        except Exception as e:
            # Log evaluation error but don't fail open
            logger.error(f"Policy condition evaluation error: {e}")
            return False
    
    def _evaluate_condition_recursive(self, condition: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Recursively evaluate nested conditions."""
        
        # Handle logical operators
        if "and" in condition:
            return all(
                self._evaluate_condition_recursive(subcond, context)
                for subcond in condition["and"]
            )
        
        if "or" in condition:
            return any(
                self._evaluate_condition_recursive(subcond, context)
                for subcond in condition["or"]
            )
        
        if "not" in condition:
            return not self._evaluate_condition_recursive(condition["not"], context)
        
        # Handle field-based conditions
        field = condition.get("field")
        operator = condition.get("operator")
        value = condition.get("value")
        
        if not field or not operator:
            return False
        
        if operator not in self.operators:
            return False
        
        field_value = self._get_field_value(field, context)
        return self.operators[operator](field_value, value, context)
    
    def _get_field_value(self, field: str, context: Dict[str, Any]) -> Any:
        """Extract field value from context using dot notation."""
        parts = field.split(".")
        value = context
        
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return None
        
        return value
    
    def _op_equals(self, field_val: Any, expected: Any, context: Dict[str, Any]) -> bool:
        return field_val == expected
    
    def _op_not_equals(self, field_val: Any, expected: Any, context: Dict[str, Any]) -> bool:
        return field_val != expected
    
    def _op_in(self, field_val: Any, expected: List[Any], context: Dict[str, Any]) -> bool:
        return field_val in expected if expected else False
    
    def _op_not_in(self, field_val: Any, expected: List[Any], context: Dict[str, Any]) -> bool:
        return field_val not in expected if expected else True
    
    def _op_matches(self, field_val: Any, pattern: str, context: Dict[str, Any]) -> bool:
        if not isinstance(field_val, str) or not pattern:
            return False
        return bool(re.search(pattern, field_val))
    
    def _op_contains(self, field_val: Any, substring: str, context: Dict[str, Any]) -> bool:
        if not isinstance(field_val, str) or not substring:
            return False
        return substring in field_val
    
    def _op_starts_with(self, field_val: Any, prefix: str, context: Dict[str, Any]) -> bool:
        if not isinstance(field_val, str) or not prefix:
            return False
        return field_val.startswith(prefix)
    
    def _op_ends_with(self, field_val: Any, suffix: str, context: Dict[str, Any]) -> bool:
        if not isinstance(field_val, str) or not suffix:
            return False
        return field_val.endswith(suffix)
    
    def _op_greater_than(self, field_val: Any, threshold: Union[int, float], context: Dict[str, Any]) -> bool:
        try:
            return float(field_val) > float(threshold)
        except (ValueError, TypeError):
            return False
    
    def _op_less_than(self, field_val: Any, threshold: Union[int, float], context: Dict[str, Any]) -> bool:
        try:
            return float(field_val) < float(threshold)
        except (ValueError, TypeError):
            return False
    
    def _op_between(self, field_val: Any, range_vals: List[Union[int, float]], context: Dict[str, Any]) -> bool:
        if not isinstance(range_vals, list) or len(range_vals) != 2:
            return False
        try:
            val = float(field_val)
            return range_vals[0] <= val <= range_vals[1]
        except (ValueError, TypeError):
            return False
    
    def _op_exists(self, field_val: Any, expected: Any, context: Dict[str, Any]) -> bool:
        return field_val is not None
    
    def _op_not_exists(self, field_val: Any, expected: Any, context: Dict[str, Any]) -> bool:
        return field_val is None
    
    def _op_all(self, field_val: Any, conditions: List[Dict[str, Any]], context: Dict[str, Any]) -> bool:
        if not isinstance(conditions, list):
            return False
        return all(
            self._evaluate_condition_recursive(cond, context)
            for cond in conditions
        )
    
    def _op_any(self, field_val: Any, conditions: List[Dict[str, Any]], context: Dict[str, Any]) -> bool:
        if not isinstance(conditions, list):
            return False
        return any(
            self._evaluate_condition_recursive(cond, context)
            for cond in conditions
        )
    
    def _op_time_window(self, field_val: Any, window_spec: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Check if field timestamp is within specified time window."""
        try:
            if isinstance(field_val, str):
                field_time = datetime.fromisoformat(field_val.replace('Z', '+00:00'))
            elif isinstance(field_val, (int, float)):
                field_time = datetime.fromtimestamp(field_val)
            else:
                return False
            
            now = datetime.utcnow()
            
            if "last_hours" in window_spec:
                cutoff = now - timedelta(hours=window_spec["last_hours"])
                return field_time >= cutoff
            
            if "last_minutes" in window_spec:
                cutoff = now - timedelta(minutes=window_spec["last_minutes"])
                return field_time >= cutoff
            
            if "between_hours" in window_spec:
                hours = window_spec["between_hours"]
                if len(hours) == 2:
                    return hours[0] <= field_time.hour <= hours[1]
            
            return False
            
        except Exception:
            return False


class PolicyEnforcer:
    """Enforces security policies and handles violations."""
    
    def __init__(self, policy_engine):
        self.policy_engine = policy_engine
        self.enforcement_actions = {
            "log": self._action_log,
            "alert": self._action_alert,
            "block": self._action_block,
            "quarantine": self._action_quarantine,
            "terminate_session": self._action_terminate_session,
            "disable_user": self._action_disable_user,
            "escalate": self._action_escalate,
            "custom": self._action_custom
        }
    
    async def enforce_policy_violation(self, violation: PolicyViolation, 
                                     policy: SecurityPolicy, 
                                     rule: PolicyRule) -> Dict[str, Any]:
        """Enforce policy violation based on rule configuration."""
        
        enforcement_result = {
            "violation_id": violation.violation_id,
            "policy_id": policy.policy_id,
            "rule_id": rule.rule_id,
            "enforcement_mode": rule.enforcement_mode.value,
            "actions_taken": [],
            "blocked": False,
            "escalated": False
        }
        
        # Determine enforcement actions based on mode
        if rule.enforcement_mode == EnforcementMode.MONITOR:
            await self._action_log(violation, policy, rule)
            enforcement_result["actions_taken"].append("logged")
        
        elif rule.enforcement_mode == EnforcementMode.WARN:
            await self._action_log(violation, policy, rule)
            await self._action_alert(violation, policy, rule)
            enforcement_result["actions_taken"].extend(["logged", "alert_sent"])
        
        elif rule.enforcement_mode == EnforcementMode.BLOCK:
            await self._action_log(violation, policy, rule)
            await self._action_alert(violation, policy, rule)
            block_result = await self._action_block(violation, policy, rule)
            enforcement_result["actions_taken"].extend(["logged", "alert_sent", "blocked"])
            enforcement_result["blocked"] = block_result.get("blocked", False)
        
        elif rule.enforcement_mode == EnforcementMode.ENFORCE:
            await self._action_log(violation, policy, rule)
            await self._action_alert(violation, policy, rule)
            
            # Execute rule-specific action
            if rule.action in self.enforcement_actions:
                action_result = await self.enforcement_actions[rule.action](violation, policy, rule)
                enforcement_result["actions_taken"].append(rule.action)
                
                if rule.action == "block":
                    enforcement_result["blocked"] = action_result.get("blocked", False)
                elif rule.action == "escalate":
                    enforcement_result["escalated"] = action_result.get("escalated", False)
        
        # Log enforcement action
        await logger.log_critical_event(
            "policy_violation_enforced",
            "policy_engine",
            {
                "violation_id": violation.violation_id,
                "policy_id": policy.policy_id,
                "rule_id": rule.rule_id,
                "enforcement_mode": rule.enforcement_mode.value,
                "actions_taken": enforcement_result["actions_taken"],
                "severity": violation.severity.value
            }
        )
        
        return enforcement_result
    
    async def _action_log(self, violation: PolicyViolation, 
                         policy: SecurityPolicy, rule: PolicyRule) -> Dict[str, Any]:
        """Log policy violation."""
        await logger.log_critical_event(
            "policy_violation",
            violation.user_id or "system",
            {
                "violation_id": violation.violation_id,
                "policy_name": policy.name,
                "rule_name": rule.name,
                "component": violation.component,
                "action": violation.action,
                "severity": violation.severity.value,
                "description": violation.description,
                "context": violation.context
            }
        )
        
        return {"logged": True}
    
    async def _action_alert(self, violation: PolicyViolation, 
                           policy: SecurityPolicy, rule: PolicyRule) -> Dict[str, Any]:
        """Send alert for policy violation."""
        # In production, integrate with alerting system
        await logger.log_critical_event(
            "policy_violation_alert",
            "policy_engine",
            {
                "violation_id": violation.violation_id,
                "policy_name": policy.name,
                "rule_name": rule.name,
                "severity": violation.severity.value,
                "alert_channels": ["email", "slack"]
            }
        )
        
        return {"alert_sent": True}
    
    async def _action_block(self, violation: PolicyViolation, 
                           policy: SecurityPolicy, rule: PolicyRule) -> Dict[str, Any]:
        """Block the action that caused the violation."""
        # In production, integrate with access control systems
        await logger.log_critical_event(
            "policy_violation_blocked",
            "policy_engine",
            {
                "violation_id": violation.violation_id,
                "user_id": violation.user_id,
                "component": violation.component,
                "action": violation.action,
                "resource": violation.resource
            }
        )
        
        return {"blocked": True}
    
    async def _action_quarantine(self, violation: PolicyViolation, 
                                policy: SecurityPolicy, rule: PolicyRule) -> Dict[str, Any]:
        """Quarantine user or resource."""
        await logger.log_critical_event(
            "policy_violation_quarantine",
            "policy_engine",
            {
                "violation_id": violation.violation_id,
                "user_id": violation.user_id,
                "resource": violation.resource,
                "quarantine_duration": "24h"
            }
        )
        
        return {"quarantined": True}
    
    async def _action_terminate_session(self, violation: PolicyViolation, 
                                       policy: SecurityPolicy, rule: PolicyRule) -> Dict[str, Any]:
        """Terminate user session."""
        if violation.user_id:
            await logger.log_critical_event(
                "policy_violation_session_terminated",
                "policy_engine",
                {
                    "violation_id": violation.violation_id,
                    "user_id": violation.user_id,
                    "session_terminated": True
                }
            )
            
            return {"session_terminated": True}
        
        return {"session_terminated": False}
    
    async def _action_disable_user(self, violation: PolicyViolation, 
                                  policy: SecurityPolicy, rule: PolicyRule) -> Dict[str, Any]:
        """Disable user account."""
        if violation.user_id:
            await logger.log_critical_event(
                "policy_violation_user_disabled",
                "policy_engine",
                {
                    "violation_id": violation.violation_id,
                    "user_id": violation.user_id,
                    "user_disabled": True,
                    "disable_reason": f"Policy violation: {policy.name}"
                }
            )
            
            return {"user_disabled": True}
        
        return {"user_disabled": False}
    
    async def _action_escalate(self, violation: PolicyViolation, 
                              policy: SecurityPolicy, rule: PolicyRule) -> Dict[str, Any]:
        """Escalate violation to security team."""
        await logger.log_critical_event(
            "policy_violation_escalated",
            "policy_engine",
            {
                "violation_id": violation.violation_id,
                "policy_name": policy.name,
                "rule_name": rule.name,
                "severity": violation.severity.value,
                "escalation_level": "security_team",
                "requires_investigation": True
            }
        )
        
        return {"escalated": True}
    
    async def _action_custom(self, violation: PolicyViolation, 
                            policy: SecurityPolicy, rule: PolicyRule) -> Dict[str, Any]:
        """Execute custom enforcement action."""
        # Custom actions would be implemented based on specific requirements
        custom_action = rule.metadata.get("custom_action")
        
        await logger.log_critical_event(
            "policy_violation_custom_action",
            "policy_engine",
            {
                "violation_id": violation.violation_id,
                "custom_action": custom_action,
                "executed": True
            }
        )
        
        return {"custom_action_executed": True}


class PolicyEngine:
    """Main security policy engine coordinating policy management and enforcement."""
    
    def __init__(self):
        self.config = get_config()
        
        # Initialize components
        self.condition_evaluator = PolicyConditionEvaluator()
        self.enforcer = PolicyEnforcer(self)
        
        # Policy storage
        self.policies: Dict[str, SecurityPolicy] = {}
        self.policy_exceptions: Dict[str, PolicyException] = {}
        self.violations: List[PolicyViolation] = []
        
        # Performance tracking
        self.evaluation_stats = {
            "total_evaluations": 0,
            "violations_detected": 0,
            "violations_blocked": 0,
            "average_evaluation_time_ms": 0,
            "policy_updates": 0
        }
        
        # Background tasks
        self._background_tasks: List[asyncio.Task] = []
        self._shutdown_event = asyncio.Event()
        
        # Load default policies
        self._load_default_policies()
    
    def _load_default_policies(self) -> None:
        """Load default security policies."""
        
        # Access Control Policy
        access_control_policy = SecurityPolicy(
            policy_id="pol_access_control_001",
            name="Access Control Policy",
            description="Controls user access to system resources",
            policy_type=PolicyType.ACCESS_CONTROL,
            version="1.0",
            status=PolicyStatus.ACTIVE,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            created_by="system",
            applicable_components=["api", "web", "admin"],
            rules=[
                PolicyRule(
                    rule_id="rule_admin_access",
                    name="Admin Access Control",
                    description="Restrict admin access to authorized users",
                    condition={
                        "and": [
                            {"field": "action", "operator": "contains", "value": "admin"},
                            {"field": "user.role", "operator": "not_in", "value": ["admin", "super_admin"]}
                        ]
                    },
                    action="block",
                    severity=PolicySeverity.HIGH,
                    enforcement_mode=EnforcementMode.ENFORCE
                ),
                PolicyRule(
                    rule_id="rule_off_hours_access",
                    name="Off-Hours Access Control",
                    description="Monitor access during off-hours",
                    condition={
                        "not": {
                            "field": "timestamp", 
                            "operator": "time_window", 
                            "value": {"between_hours": [8, 18]}
                        }
                    },
                    action="alert",
                    severity=PolicySeverity.MEDIUM,
                    enforcement_mode=EnforcementMode.WARN
                )
            ]
        )
        
        # Authentication Policy
        auth_policy = SecurityPolicy(
            policy_id="pol_authentication_001",
            name="Authentication Policy",
            description="Enforces authentication requirements",
            policy_type=PolicyType.AUTHENTICATION,
            version="1.0",
            status=PolicyStatus.ACTIVE,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            created_by="system",
            applicable_components=["api", "web"],
            rules=[
                PolicyRule(
                    rule_id="rule_failed_login_attempts",
                    name="Failed Login Attempts",
                    description="Monitor excessive failed login attempts",
                    condition={
                        "and": [
                            {"field": "event_type", "operator": "equals", "value": "login_attempt"},
                            {"field": "success", "operator": "equals", "value": False},
                            {"field": "failed_attempts_count", "operator": "greater_than", "value": 5}
                        ]
                    },
                    action="block",
                    severity=PolicySeverity.HIGH,
                    enforcement_mode=EnforcementMode.ENFORCE
                ),
                PolicyRule(
                    rule_id="rule_mfa_required",
                    name="MFA Required for Sensitive Operations",
                    description="Require MFA for sensitive operations",
                    condition={
                        "and": [
                            {"field": "action", "operator": "in", "value": ["transfer", "config_change", "user_create"]},
                            {"field": "mfa_verified", "operator": "not_equals", "value": True}
                        ]
                    },
                    action="block",
                    severity=PolicySeverity.CRITICAL,
                    enforcement_mode=EnforcementMode.ENFORCE
                )
            ]
        )
        
        # Data Protection Policy
        data_protection_policy = SecurityPolicy(
            policy_id="pol_data_protection_001",
            name="Data Protection Policy",
            description="Protects sensitive data access and modifications",
            policy_type=PolicyType.DATA_PROTECTION,
            version="1.0",
            status=PolicyStatus.ACTIVE,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            created_by="system",
            applicable_components=["database", "api", "backup"],
            rules=[
                PolicyRule(
                    rule_id="rule_bulk_data_access",
                    name="Bulk Data Access",
                    description="Monitor bulk data access operations",
                    condition={
                        "and": [
                            {"field": "operation", "operator": "in", "value": ["export", "download", "bulk_read"]},
                            {"field": "record_count", "operator": "greater_than", "value": 1000}
                        ]
                    },
                    action="alert",
                    severity=PolicySeverity.MEDIUM,
                    enforcement_mode=EnforcementMode.WARN
                ),
                PolicyRule(
                    rule_id="rule_sensitive_data_access",
                    name="Sensitive Data Access",
                    description="Control access to sensitive data fields",
                    condition={
                        "and": [
                            {"field": "data_classification", "operator": "equals", "value": "sensitive"},
                            {"field": "user.clearance_level", "operator": "less_than", "value": 3}
                        ]
                    },
                    action="block",
                    severity=PolicySeverity.HIGH,
                    enforcement_mode=EnforcementMode.ENFORCE
                )
            ]
        )
        
        # Store policies
        self.policies[access_control_policy.policy_id] = access_control_policy
        self.policies[auth_policy.policy_id] = auth_policy
        self.policies[data_protection_policy.policy_id] = data_protection_policy
    
    async def initialize(self) -> None:
        """Initialize the policy engine."""
        
        # Start background tasks
        self._background_tasks = [
            asyncio.create_task(self._policy_compliance_monitor()),
            asyncio.create_task(self._violation_analyzer()),
            asyncio.create_task(self._policy_metrics_collector())
        ]
        
        await logger.log_critical_event(
            "policy_engine_initialized",
            "policy_engine",
            {
                "active_policies": len([p for p in self.policies.values() if p.status == PolicyStatus.ACTIVE]),
                "total_rules": sum(len(p.rules) for p in self.policies.values()),
                "timestamp": datetime.utcnow().isoformat()
            }
        )
    
    async def shutdown(self) -> None:
        """Shutdown the policy engine."""
        self._shutdown_event.set()
        
        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()
        
        await asyncio.gather(*self._background_tasks, return_exceptions=True)
        
        await logger.log_critical_event(
            "policy_engine_shutdown",
            "policy_engine",
            {"timestamp": datetime.utcnow().isoformat()}
        )
    
    async def evaluate_event(self, component: str, action: str, context: Dict[str, Any],
                           user_id: Optional[str] = None, resource: Optional[str] = None) -> Dict[str, Any]:
        """Evaluate an event against all applicable policies."""
        
        evaluation_start = time.time()
        
        evaluation_result = {
            "component": component,
            "action": action,
            "user_id": user_id,
            "resource": resource,
            "timestamp": datetime.utcnow().isoformat(),
            "violations": [],
            "policy_decisions": {},
            "overall_decision": "allow",
            "blocked": False
        }
        
        # Get applicable policies
        applicable_policies = [
            policy for policy in self.policies.values()
            if (policy.status == PolicyStatus.ACTIVE and 
                (not policy.applicable_components or component in policy.applicable_components))
        ]
        
        # Evaluate each policy
        for policy in applicable_policies:
            policy_decision = await self._evaluate_policy(policy, component, action, context, user_id, resource)
            evaluation_result["policy_decisions"][policy.policy_id] = policy_decision
            
            # Collect violations
            if policy_decision["violations"]:
                evaluation_result["violations"].extend(policy_decision["violations"])
            
            # Update overall decision
            if policy_decision["decision"] == "block":
                evaluation_result["overall_decision"] = "block"
                evaluation_result["blocked"] = True
        
        # Update stats
        evaluation_time = (time.time() - evaluation_start) * 1000
        self.evaluation_stats["total_evaluations"] += 1
        self.evaluation_stats["average_evaluation_time_ms"] = (
            (self.evaluation_stats["average_evaluation_time_ms"] * (self.evaluation_stats["total_evaluations"] - 1) + evaluation_time) /
            self.evaluation_stats["total_evaluations"]
        )
        
        if evaluation_result["violations"]:
            self.evaluation_stats["violations_detected"] += len(evaluation_result["violations"])
        
        if evaluation_result["blocked"]:
            self.evaluation_stats["violations_blocked"] += 1
        
        return evaluation_result
    
    async def _evaluate_policy(self, policy: SecurityPolicy, component: str, action: str, 
                              context: Dict[str, Any], user_id: Optional[str], 
                              resource: Optional[str]) -> Dict[str, Any]:
        """Evaluate a single policy against an event."""
        
        policy_result = {
            "policy_id": policy.policy_id,
            "policy_name": policy.name,
            "decision": "allow",
            "violations": [],
            "rules_evaluated": 0,
            "rules_matched": 0
        }
        
        # Check if there's an active exception for this policy
        if await self._has_active_exception(policy.policy_id, user_id, resource, context):
            policy_result["decision"] = "allow"
            policy_result["exception_applied"] = True
            return policy_result
        
        # Evaluate each rule in the policy
        for rule in policy.rules:
            if not rule.enabled:
                continue
            
            policy_result["rules_evaluated"] += 1
            
            # Evaluate rule condition
            if self.condition_evaluator.evaluate_condition(rule.condition, context):
                policy_result["rules_matched"] += 1
                
                # Create violation record
                violation = PolicyViolation(
                    violation_id=f"viol_{secrets.token_hex(8)}",
                    policy_id=policy.policy_id,
                    rule_id=rule.rule_id,
                    timestamp=datetime.utcnow(),
                    component=component,
                    user_id=user_id,
                    resource=resource,
                    action=action,
                    severity=rule.severity,
                    description=f"Policy violation: {rule.name}",
                    context=context.copy()
                )
                
                # Store violation
                self.violations.append(violation)
                policy_result["violations"].append(violation)
                
                # Enforce policy if required
                if rule.enforcement_mode in [EnforcementMode.BLOCK, EnforcementMode.ENFORCE]:
                    enforcement_result = await self.enforcer.enforce_policy_violation(violation, policy, rule)
                    
                    if enforcement_result.get("blocked"):
                        policy_result["decision"] = "block"
        
        return policy_result
    
    async def _has_active_exception(self, policy_id: str, user_id: Optional[str],
                                  resource: Optional[str], context: Dict[str, Any]) -> bool:
        """Check if there's an active exception for this policy evaluation."""
        now = datetime.utcnow()
        
        for exception in self.policy_exceptions.values():
            if (exception.policy_id == policy_id and
                exception.valid_from <= now <= exception.valid_until):
                
                # Check if exception applies to this user/resource
                if exception.granted_to and user_id != exception.granted_to:
                    continue
                
                # Check exception conditions
                if exception.conditions:
                    if not self.condition_evaluator.evaluate_condition(exception.conditions, context):
                        continue
                
                return True
        
        return False
    
    def get_policy_status(self) -> Dict[str, Any]:
        """Get comprehensive policy engine status."""
        active_policies = [p for p in self.policies.values() if p.status == PolicyStatus.ACTIVE]
        unresolved_violations = [v for v in self.violations if not v.resolved]
        
        return {
            "total_policies": len(self.policies),
            "active_policies": len(active_policies),
            "total_rules": sum(len(p.rules) for p in active_policies),
            "total_violations": len(self.violations),
            "unresolved_violations": len(unresolved_violations),
            "policy_exceptions": len(self.policy_exceptions),
            "evaluation_stats": self.evaluation_stats,
            "policy_types": {
                policy_type.value: len([
                    p for p in active_policies
                    if p.policy_type == policy_type
                ])
                for policy_type in PolicyType
            }
        }
    
    async def _policy_compliance_monitor(self) -> None:
        """Background task to monitor policy compliance."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(3600)  # Check every hour
                
                now = datetime.utcnow()
                last_24h = now - timedelta(hours=24)
                
                recent_violations = [
                    v for v in self.violations
                    if v.timestamp > last_24h
                ]
                
                compliance_stats = {
                    "timestamp": now.isoformat(),
                    "violations_last_24h": len(recent_violations),
                    "unresolved_violations": len([v for v in recent_violations if not v.resolved]),
                    "critical_violations": len([
                        v for v in recent_violations
                        if v.severity == PolicySeverity.CRITICAL
                    ]),
                    "policies_active": len([
                        p for p in self.policies.values()
                        if p.status == PolicyStatus.ACTIVE
                    ])
                }
                
                await logger.log_critical_event(
                    "policy_compliance_report",
                    "policy_engine",
                    compliance_stats
                )
                
            except Exception as e:
                await logger.log_critical_event(
                    "policy_compliance_monitor_error",
                    "policy_engine",
                    {"error": str(e)}
                )
    
    async def _violation_analyzer(self) -> None:
        """Background task to analyze violation patterns."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(1800)  # Analyze every 30 minutes
                
                now = datetime.utcnow()
                last_hour = now - timedelta(hours=1)
                
                recent_violations = [
                    v for v in self.violations
                    if v.timestamp > last_hour
                ]
                
                if len(recent_violations) > 10:
                    user_violations = {}
                    for violation in recent_violations:
                        if violation.user_id:
                            if violation.user_id not in user_violations:
                                user_violations[violation.user_id] = []
                            user_violations[violation.user_id].append(violation)
                    
                    for user_id, violations in user_violations.items():
                        if len(violations) >= 5:
                            await logger.log_critical_event(
                                "policy_violation_pattern_detected",
                                "policy_engine",
                                {
                                    "pattern_type": "user_multiple_violations",
                                    "user_id": user_id,
                                    "violation_count": len(violations),
                                    "timespan": "1_hour"
                                }
                            )
                
            except Exception as e:
                await logger.log_critical_event(
                    "violation_analyzer_error",
                    "policy_engine",
                    {"error": str(e)}
                )
    
    async def _policy_metrics_collector(self) -> None:
        """Background task to collect policy metrics."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(300)  # Update every 5 minutes
                self.evaluation_stats["timestamp"] = datetime.utcnow().isoformat()
                
            except Exception as e:
                await logger.log_critical_event(
                    "policy_metrics_collector_error",
                    "policy_engine",
                    {"error": str(e)}
                )
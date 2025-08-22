"""
FlashMM Security Orchestrator

Main security service coordinating all security components.
Provides comprehensive security management, threat detection, and incident response.
"""

import asyncio
import hashlib
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

from flashmm.config.settings import get_config
from flashmm.security.auth import AuthenticationManager, AuthorizationManager
from flashmm.security.encryption import DataEncryption
from flashmm.security.key_manager import (
    ColdKeyManager,
    HotKeyManager,
    KeyRotationManager,
    WarmKeyManager,
)
from flashmm.utils.exceptions import AuthenticationError
from flashmm.utils.logging import SecurityLogger


class SecurityLevel(Enum):
    """Security alert levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ThreatType(Enum):
    """Types of security threats."""
    AUTHENTICATION_FAILURE = "auth_failure"
    AUTHORIZATION_VIOLATION = "authz_violation"
    KEY_COMPROMISE = "key_compromise"
    ANOMALOUS_BEHAVIOR = "anomalous_behavior"
    SYSTEM_INTRUSION = "system_intrusion"
    DATA_BREACH = "data_breach"
    DDOS_ATTACK = "ddos_attack"
    MALICIOUS_REQUEST = "malicious_request"


@dataclass
class SecurityThreat:
    """Security threat information."""
    threat_id: str
    threat_type: ThreatType
    severity: SecurityLevel
    source_ip: str | None
    user_id: str | None
    component: str
    description: str
    timestamp: datetime
    metadata: dict[str, Any]
    resolved: bool = False
    resolution_actions: list[str] | None = None

    def __post_init__(self):
        if self.resolution_actions is None:
            self.resolution_actions = []


class SecurityState(Enum):
    """System security states."""
    NORMAL = "normal"
    ELEVATED = "elevated"
    HIGH_ALERT = "high_alert"
    LOCKDOWN = "lockdown"
    EMERGENCY = "emergency"


@dataclass
class SecurityMetrics:
    """Security system metrics."""
    total_auth_attempts: int = 0
    failed_auth_attempts: int = 0
    blocked_requests: int = 0
    threats_detected: int = 0
    threats_resolved: int = 0
    active_sessions: int = 0
    key_rotations: int = 0
    security_violations: int = 0
    last_update: datetime | None = None

    def __post_init__(self):
        if self.last_update is None:
            self.last_update = datetime.utcnow()


class SecurityOrchestrator:
    """Main security service coordinating all security components."""

    def __init__(self):
        self.config = get_config()
        self.logger = SecurityLogger()

        # Initialize security components
        self.auth_manager = AuthenticationManager()
        self.authz_manager = AuthorizationManager()
        self.hot_key_manager = HotKeyManager()
        self.warm_key_manager = WarmKeyManager()
        self.cold_key_manager = ColdKeyManager()
        self.key_rotation_manager = KeyRotationManager()

        # Security state management
        self.current_state = SecurityState.NORMAL
        self.active_threats: dict[str, SecurityThreat] = {}
        self.threat_patterns: dict[str, list[str]] = {}
        self.blocked_ips: set[str] = set()
        self.security_metrics = SecurityMetrics()

        # Rate limiting tracking
        self.rate_limits: dict[str, dict[str, list[float]]] = {}

        # Security policies
        self.security_policies = self._load_security_policies()

        # Initialize encryption
        master_key = self.config.get("security.master_key", "default_master_key")
        self.data_encryption = DataEncryption(master_key)

        # Start background tasks
        self._background_tasks: list[asyncio.Task] = []
        self._shutdown_event = asyncio.Event()

    def _load_security_policies(self) -> dict[str, Any]:
        """Load security policies configuration."""
        return {
            "max_auth_failures": self.config.get("security.max_auth_failures", 5),
            "auth_failure_window": self.config.get("security.auth_failure_window", 300),
            "lockout_duration": self.config.get("security.lockout_duration", 3600),
            "rate_limit_requests": self.config.get("security.rate_limit_requests", 100),
            "rate_limit_window": self.config.get("security.rate_limit_window", 60),
            "session_timeout": self.config.get("security.session_timeout", 3600),
            "key_rotation_interval": self.config.get("security.key_rotation_interval", 86400),
            "threat_response_delay": self.config.get("security.threat_response_delay", 5),
            "emergency_contacts": self.config.get("security.emergency_contacts", []),
            "allowed_countries": self.config.get("security.allowed_countries", []),
            "blocked_user_agents": self.config.get("security.blocked_user_agents", [])
        }

    async def start(self) -> None:
        """Start the security orchestrator and background services."""
        await self.logger.log_critical_event(
            "security_orchestrator_started",
            "system",
            {"timestamp": datetime.utcnow().isoformat()}
        )

        # Start background monitoring tasks
        self._background_tasks = [
            asyncio.create_task(self._monitor_security_state()),
            asyncio.create_task(self._monitor_key_rotation()),
            asyncio.create_task(self._monitor_threat_patterns()),
            asyncio.create_task(self._cleanup_expired_data()),
            asyncio.create_task(self._generate_security_reports())
        ]

    async def stop(self) -> None:
        """Stop the security orchestrator and cleanup resources."""
        self._shutdown_event.set()

        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()

        # Wait for tasks to complete
        await asyncio.gather(*self._background_tasks, return_exceptions=True)

        await self.logger.log_critical_event(
            "security_orchestrator_stopped",
            "system",
            {"timestamp": datetime.utcnow().isoformat()}
        )

    async def authenticate_request(self,
                                 request_data: dict[str, Any]) -> dict[str, Any]:
        """Authenticate and authorize a request through the security pipeline."""
        request_id = request_data.get("request_id", self._generate_request_id())
        source_ip = request_data.get("source_ip")
        user_agent = request_data.get("user_agent")

        # Check if IP is blocked
        if source_ip in self.blocked_ips:
            await self._handle_blocked_request(request_id, source_ip, "blocked_ip")
            raise AuthenticationError("Access denied: IP blocked")

        # Check rate limits
        if source_ip and not await self._check_rate_limit(source_ip):
            await self._handle_rate_limit_exceeded(request_id, source_ip)
            raise AuthenticationError("Rate limit exceeded")

        # Check user agent blocking
        if self._is_blocked_user_agent(user_agent):
            await self._handle_blocked_request(request_id, source_ip, "blocked_user_agent")
            raise AuthenticationError("Access denied: User agent blocked")

        # Perform authentication
        auth_result = await self._perform_authentication(request_data)

        # Update security metrics
        self.security_metrics.total_auth_attempts += 1
        if not auth_result.get("authenticated"):
            self.security_metrics.failed_auth_attempts += 1
            await self._handle_authentication_failure(request_id, source_ip, auth_result)

        return auth_result

    async def _perform_authentication(self, request_data: dict[str, Any]) -> dict[str, Any]:
        """Perform authentication using configured methods."""
        auth_type = request_data.get("auth_type", "api_key")

        if auth_type == "api_key":
            api_key = request_data.get("api_key")
            if not api_key:
                return {"authenticated": False, "reason": "missing_api_key"}

            auth_result = await self.auth_manager.verify_api_key(api_key)
            if auth_result:
                role = auth_result.get("role")
                # Simplified permissions handling without relying on UserRole enum
                permissions = ["read", "write"] if role == "admin" else ["read"]
                token = self.auth_manager.create_access_token(
                    subject=f"api_user_{role}",
                    permissions=permissions
                )
                return {
                    "authenticated": True,
                    "role": role,
                    "token": token,
                    "expires_at": (datetime.utcnow() + self.auth_manager.access_token_expire).isoformat()
                }
            else:
                return {"authenticated": False, "reason": "invalid_api_key"}

        elif auth_type == "jwt":
            token = request_data.get("token")
            if not token:
                return {"authenticated": False, "reason": "missing_token"}

            try:
                payload = self.auth_manager.verify_token(token)
                return {
                    "authenticated": True,
                    "payload": payload,
                    "role": payload.get("role"),
                    "expires_at": datetime.fromtimestamp(payload.get("exp", 0)).isoformat()
                }
            except AuthenticationError as e:
                return {"authenticated": False, "reason": str(e)}

        return {"authenticated": False, "reason": "unsupported_auth_type"}

    async def _check_rate_limit(self, identifier: str) -> bool:
        """Check if request is within rate limits."""
        current_time = time.time()
        window = self.security_policies["rate_limit_window"]
        max_requests = self.security_policies["rate_limit_requests"]

        if identifier not in self.rate_limits:
            self.rate_limits[identifier] = {"requests": []}

        # Clean old requests outside window
        self.rate_limits[identifier]["requests"] = [
            req_time for req_time in self.rate_limits[identifier]["requests"]
            if current_time - req_time < window
        ]

        # Check if within limit
        if len(self.rate_limits[identifier]["requests"]) >= max_requests:
            return False

        # Add current request
        self.rate_limits[identifier]["requests"].append(current_time)
        return True

    def _is_blocked_user_agent(self, user_agent: str | None) -> bool:
        """Check if user agent is blocked."""
        if not user_agent:
            return False

        blocked_agents = self.security_policies.get("blocked_user_agents", [])
        return any(blocked in user_agent.lower() for blocked in blocked_agents)

    async def _handle_authentication_failure(self,
                                           request_id: str,
                                           source_ip: str | None,
                                           auth_result: dict[str, Any]) -> None:
        """Handle authentication failure and track patterns."""
        threat = SecurityThreat(
            threat_id=f"auth_fail_{request_id}",
            threat_type=ThreatType.AUTHENTICATION_FAILURE,
            severity=SecurityLevel.MEDIUM,
            source_ip=source_ip,
            user_id=None,
            component="authentication",
            description=f"Authentication failure: {auth_result.get('reason')}",
            timestamp=datetime.utcnow(),
            metadata={"request_id": request_id, "auth_result": auth_result}
        )

        await self._register_threat(threat)

        # Check for brute force patterns
        if source_ip:
            await self._check_brute_force_pattern(source_ip)

    async def _handle_blocked_request(self,
                                    request_id: str,
                                    source_ip: str | None,
                                    reason: str) -> None:
        """Handle blocked request."""
        self.security_metrics.blocked_requests += 1

        threat = SecurityThreat(
            threat_id=f"blocked_{request_id}",
            threat_type=ThreatType.MALICIOUS_REQUEST,
            severity=SecurityLevel.HIGH,
            source_ip=source_ip,
            user_id=None,
            component="access_control",
            description=f"Request blocked: {reason}",
            timestamp=datetime.utcnow(),
            metadata={"request_id": request_id, "reason": reason}
        )

        await self._register_threat(threat)

    async def _handle_rate_limit_exceeded(self,
                                        request_id: str,
                                        source_ip: str | None) -> None:
        """Handle rate limit exceeded."""
        threat = SecurityThreat(
            threat_id=f"rate_limit_{request_id}",
            threat_type=ThreatType.DDOS_ATTACK,
            severity=SecurityLevel.HIGH,
            source_ip=source_ip,
            user_id=None,
            component="rate_limiter",
            description="Rate limit exceeded - potential DDoS",
            timestamp=datetime.utcnow(),
            metadata={"request_id": request_id}
        )

        await self._register_threat(threat)

        # Consider temporarily blocking the IP
        if source_ip and await self._should_block_ip(source_ip):
            await self._block_ip(source_ip, duration=timedelta(hours=1))

    async def _register_threat(self, threat: SecurityThreat) -> None:
        """Register a new security threat."""
        self.active_threats[threat.threat_id] = threat
        self.security_metrics.threats_detected += 1

        await self.logger.log_critical_event(
            "security_threat_detected",
            "security_orchestrator",
            {
                "threat_id": threat.threat_id,
                "threat_type": threat.threat_type.value,
                "severity": threat.severity.value,
                "source_ip": threat.source_ip,
                "component": threat.component,
                "description": threat.description,
                "metadata": threat.metadata
            }
        )

        # Trigger automated response
        await self._respond_to_threat(threat)

    async def _respond_to_threat(self, threat: SecurityThreat) -> None:
        """Automated threat response."""
        response_actions = []

        if threat.severity == SecurityLevel.CRITICAL:
            # Escalate security state
            await self._escalate_security_state(SecurityState.HIGH_ALERT)
            response_actions.append("escalated_security_state")

            # Consider emergency procedures
            if threat.threat_type in [ThreatType.SYSTEM_INTRUSION, ThreatType.DATA_BREACH]:
                await self._trigger_emergency_procedures(threat)
                response_actions.append("triggered_emergency_procedures")

        elif threat.severity == SecurityLevel.HIGH:
            if threat.source_ip:
                await self._block_ip(threat.source_ip, duration=timedelta(hours=24))
                response_actions.append(f"blocked_ip_{threat.source_ip}")

        # Update threat with response actions
        if threat.resolution_actions is not None:
            threat.resolution_actions.extend(response_actions)
        else:
            threat.resolution_actions = response_actions

        await self.logger.log_critical_event(
            "automated_threat_response",
            "security_orchestrator",
            {
                "threat_id": threat.threat_id,
                "actions": response_actions
            }
        )

    async def _block_ip(self, ip_address: str, duration: timedelta) -> None:
        """Block an IP address for specified duration."""
        self.blocked_ips.add(ip_address)

        # Schedule unblocking
        asyncio.create_task(self._schedule_ip_unblock(ip_address, duration))

        await self.logger.log_critical_event(
            "ip_blocked",
            "security_orchestrator",
            {
                "ip_address": ip_address,
                "duration_seconds": int(duration.total_seconds())
            }
        )

    async def _schedule_ip_unblock(self, ip_address: str, duration: timedelta) -> None:
        """Schedule IP unblocking after duration."""
        await asyncio.sleep(duration.total_seconds())

        if ip_address in self.blocked_ips:
            self.blocked_ips.remove(ip_address)

            await self.logger.log_critical_event(
                "ip_unblocked",
                "security_orchestrator",
                {"ip_address": ip_address}
            )

    async def _should_block_ip(self, ip_address: str) -> bool:
        """Determine if IP should be blocked based on threat patterns."""
        # Count recent threats from this IP
        recent_threats = [
            threat for threat in self.active_threats.values()
            if threat.source_ip == ip_address
            and datetime.utcnow() - threat.timestamp < timedelta(minutes=15)
        ]

        return len(recent_threats) >= 3

    async def _check_brute_force_pattern(self, source_ip: str) -> None:
        """Check for brute force attack patterns."""
        window = timedelta(minutes=self.security_policies["auth_failure_window"] // 60)
        max_failures = self.security_policies["max_auth_failures"]

        # Count recent auth failures from this IP
        recent_failures = [
            threat for threat in self.active_threats.values()
            if (threat.source_ip == source_ip
                and threat.threat_type == ThreatType.AUTHENTICATION_FAILURE
                and datetime.utcnow() - threat.timestamp < window)
        ]

        if len(recent_failures) >= max_failures:
            # Trigger brute force response
            threat = SecurityThreat(
                threat_id=f"brute_force_{source_ip}_{int(time.time())}",
                threat_type=ThreatType.ANOMALOUS_BEHAVIOR,
                severity=SecurityLevel.HIGH,
                source_ip=source_ip,
                user_id=None,
                component="brute_force_detector",
                description=f"Brute force attack detected from {source_ip}",
                timestamp=datetime.utcnow(),
                metadata={"failure_count": len(recent_failures)}
            )

            await self._register_threat(threat)

    def _generate_request_id(self) -> str:
        """Generate unique request ID."""
        return hashlib.sha256(
            f"{datetime.utcnow().isoformat()}_{time.time()}".encode()
        ).hexdigest()[:16]

    async def _escalate_security_state(self, new_state: SecurityState) -> None:
        """Escalate system security state."""
        if new_state.value > self.current_state.value:
            old_state = self.current_state
            self.current_state = new_state

            await self.logger.log_critical_event(
                "security_state_escalated",
                "security_orchestrator",
                {
                    "old_state": old_state.value,
                    "new_state": new_state.value,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )

    async def _trigger_emergency_procedures(self, threat: SecurityThreat) -> None:
        """Trigger emergency procedures for critical threats."""
        await self.logger.log_critical_event(
            "emergency_procedures_triggered",
            "security_orchestrator",
            {
                "threat_id": threat.threat_id,
                "threat_type": threat.threat_type.value,
                "timestamp": datetime.utcnow().isoformat()
            }
        )

        # Emergency contacts notification would be implemented here
        # Emergency shutdown procedures would be coordinated here

    async def _monitor_security_state(self) -> None:
        """Background task to monitor and adjust security state."""
        while not self._shutdown_event.is_set():
            try:
                # Check if we should de-escalate security state
                if self.current_state != SecurityState.NORMAL:
                    active_high_threats = [
                        threat for threat in self.active_threats.values()
                        if (threat.severity in [SecurityLevel.HIGH, SecurityLevel.CRITICAL]
                            and not threat.resolved)
                    ]

                    if not active_high_threats:
                        await self._de_escalate_security_state()

                await asyncio.sleep(30)  # Check every 30 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                await self.logger.log_critical_event(
                    "security_monitor_error",
                    "security_orchestrator",
                    {"error": str(e)}
                )

    async def _de_escalate_security_state(self) -> None:
        """De-escalate security state when threats are resolved."""
        if self.current_state != SecurityState.NORMAL:
            old_state = self.current_state
            self.current_state = SecurityState.NORMAL

            await self.logger.log_critical_event(
                "security_state_de_escalated",
                "security_orchestrator",
                {
                    "old_state": old_state.value,
                    "new_state": SecurityState.NORMAL.value,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )

    async def _monitor_key_rotation(self) -> None:
        """Background task to monitor key rotation needs."""
        while not self._shutdown_event.is_set():
            try:
                if self.hot_key_manager.needs_rotation():
                    await self.logger.log_critical_event(
                        "key_rotation_needed",
                        "security_orchestrator",
                        {"key_type": "hot_keys"}
                    )

                    # In production, would trigger automated key rotation
                    # await self.key_rotation_manager.rotate_hot_keys()

                await asyncio.sleep(3600)  # Check every hour
            except asyncio.CancelledError:
                break
            except Exception as e:
                await self.logger.log_critical_event(
                    "key_monitor_error",
                    "security_orchestrator",
                    {"error": str(e)}
                )

    async def _monitor_threat_patterns(self) -> None:
        """Background task to analyze threat patterns."""
        while not self._shutdown_event.is_set():
            try:
                # Analyze patterns in active threats
                await self._analyze_threat_patterns()

                await asyncio.sleep(300)  # Analyze every 5 minutes
            except asyncio.CancelledError:
                break
            except Exception as e:
                await self.logger.log_critical_event(
                    "threat_pattern_monitor_error",
                    "security_orchestrator",
                    {"error": str(e)}
                )

    async def _analyze_threat_patterns(self) -> None:
        """Analyze patterns in security threats."""
        # Group threats by type and source
        threat_groups = {}
        for threat in self.active_threats.values():
            key = f"{threat.threat_type.value}_{threat.source_ip or 'unknown'}"
            if key not in threat_groups:
                threat_groups[key] = []
            threat_groups[key].append(threat)

        # Look for concerning patterns
        for group_key, threats in threat_groups.items():
            if len(threats) >= 5:  # 5 or more similar threats
                recent_threats = [
                    t for t in threats
                    if datetime.utcnow() - t.timestamp < timedelta(hours=1)
                ]

                if len(recent_threats) >= 3:
                    await self.logger.log_critical_event(
                        "threat_pattern_detected",
                        "security_orchestrator",
                        {
                            "pattern": group_key,
                            "threat_count": len(recent_threats),
                            "timespan": "1_hour"
                        }
                    )

    async def _cleanup_expired_data(self) -> None:
        """Background task to cleanup expired security data."""
        while not self._shutdown_event.is_set():
            try:
                current_time = datetime.utcnow()

                # Clean up old resolved threats (older than 24 hours)
                expired_threats = [
                    tid for tid, threat in self.active_threats.items()
                    if (threat.resolved
                        and current_time - threat.timestamp > timedelta(hours=24))
                ]

                for tid in expired_threats:
                    del self.active_threats[tid]

                # Clean up old rate limit data
                cutoff_time = time.time() - self.security_policies["rate_limit_window"]
                for identifier in list(self.rate_limits.keys()):
                    self.rate_limits[identifier]["requests"] = [
                        req_time for req_time in self.rate_limits[identifier]["requests"]
                        if req_time > cutoff_time
                    ]

                    # Remove empty rate limit entries
                    if not self.rate_limits[identifier]["requests"]:
                        del self.rate_limits[identifier]

                await asyncio.sleep(3600)  # Cleanup every hour
            except asyncio.CancelledError:
                break
            except Exception as e:
                await self.logger.log_critical_event(
                    "cleanup_error",
                    "security_orchestrator",
                    {"error": str(e)}
                )

    async def _generate_security_reports(self) -> None:
        """Background task to generate security reports."""
        while not self._shutdown_event.is_set():
            try:
                # Generate hourly security metrics
                report = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "security_state": self.current_state.value,
                    "metrics": asdict(self.security_metrics),
                    "active_threats": len([t for t in self.active_threats.values() if not t.resolved]),
                    "blocked_ips": len(self.blocked_ips),
                    "rate_limited_clients": len(self.rate_limits)
                }

                await self.logger.log_critical_event(
                    "security_metrics_report",
                    "security_orchestrator",
                    report
                )

                # Reset counters
                self.security_metrics.last_update = datetime.utcnow()

                await asyncio.sleep(3600)  # Report every hour
            except asyncio.CancelledError:
                break
            except Exception as e:
                await self.logger.log_critical_event(
                    "security_report_error",
                    "security_orchestrator",
                    {"error": str(e)}
                )

    def get_security_status(self) -> dict[str, Any]:
        """Get current security system status."""
        return {
            "security_state": self.current_state.value,
            "active_threats": len([t for t in self.active_threats.values() if not t.resolved]),
            "blocked_ips": len(self.blocked_ips),
            "metrics": asdict(self.security_metrics),
            "key_rotation_status": self.key_rotation_manager.get_rotation_status(),
            "policies": self.security_policies
        }

    async def resolve_threat(self, threat_id: str, resolution_notes: str = "") -> bool:
        """Manually resolve a security threat."""
        if threat_id in self.active_threats:
            threat = self.active_threats[threat_id]
            threat.resolved = True
            if threat.resolution_actions is not None:
                threat.resolution_actions.append(f"manual_resolution: {resolution_notes}")
            else:
                threat.resolution_actions = [f"manual_resolution: {resolution_notes}"]

            self.security_metrics.threats_resolved += 1

            await self.logger.log_critical_event(
                "threat_resolved",
                "security_orchestrator",
                {
                    "threat_id": threat_id,
                    "resolution_notes": resolution_notes,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )

            return True

        return False

    async def emergency_shutdown(self, reason: str, user: str) -> dict[str, Any]:
        """Trigger emergency shutdown procedures."""
        await self.logger.log_critical_event(
            "emergency_shutdown_initiated",
            user,
            {
                "reason": reason,
                "timestamp": datetime.utcnow().isoformat(),
                "security_state": self.current_state.value
            }
        )

        # Set security state to emergency
        self.current_state = SecurityState.EMERGENCY

        # Stop all background tasks
        await self.stop()

        return {
            "status": "emergency_shutdown_complete",
            "reason": reason,
            "timestamp": datetime.utcnow().isoformat()
        }

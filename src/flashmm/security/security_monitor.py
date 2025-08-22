"""
FlashMM Security Monitoring System

Comprehensive security monitoring with threat detection, behavioral analysis,
intrusion detection, and automated response capabilities.
"""

import asyncio
import hashlib
import ipaddress
import json
import re
import statistics
import time
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

from flashmm.config.settings import get_config
from flashmm.utils.logging import SecurityLogger

logger = SecurityLogger()


class ThreatLevel(Enum):
    """Threat severity levels."""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class MonitoringEvent(Enum):
    """Types of monitoring events."""
    LOGIN_ATTEMPT = "login_attempt"
    API_REQUEST = "api_request"
    DATA_ACCESS = "data_access"
    SYSTEM_COMMAND = "system_command"
    NETWORK_CONNECTION = "network_connection"
    FILE_OPERATION = "file_operation"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    CONFIGURATION_CHANGE = "configuration_change"


class AnomalyType(Enum):
    """Types of behavioral anomalies."""
    FREQUENCY_ANOMALY = "frequency_anomaly"
    PATTERN_ANOMALY = "pattern_anomaly"
    GEOGRAPHIC_ANOMALY = "geographic_anomaly"
    TIME_ANOMALY = "time_anomaly"
    VOLUME_ANOMALY = "volume_anomaly"
    SEQUENCE_ANOMALY = "sequence_anomaly"


@dataclass
class SecurityEvent:
    """Security event data structure."""
    event_id: str
    event_type: MonitoringEvent
    timestamp: datetime
    source_ip: str | None
    user_id: str | None
    component: str
    action: str
    resource: str | None
    success: bool
    metadata: dict[str, Any]
    risk_score: float = 0.0
    anomaly_indicators: list[str] | None = None

    def __post_init__(self):
        if self.anomaly_indicators is None:
            self.anomaly_indicators = []


@dataclass
class ThreatIndicator:
    """Threat intelligence indicator."""
    indicator_id: str
    indicator_type: str  # ip, domain, hash, pattern
    value: str
    threat_level: ThreatLevel
    description: str
    source: str
    created_at: datetime
    expires_at: datetime | None
    metadata: dict[str, Any]


@dataclass
class BehavioralBaseline:
    """Behavioral baseline for anomaly detection."""
    entity_id: str  # user_id, ip_address, etc.
    metric_name: str
    baseline_value: float
    standard_deviation: float
    sample_count: int
    last_updated: datetime
    confidence_level: float


class IntrusionDetector:
    """Intrusion detection and prevention system."""

    def __init__(self):
        self.config = get_config()

        # Signature-based detection patterns
        self.attack_patterns = {
            "sql_injection": [
                r"(?i)(union\s+select|or\s+1\s*=\s*1|';\s*drop\s+table)",
                r"(?i)(exec\s*\(|sp_executesql|xp_cmdshell)",
                r"(?i)(\'\s*or\s*\'\w*\'\s*=\s*\'\w*)"
            ],
            "xss": [
                r"(?i)(<script|javascript:|on\w+\s*=)",
                r"(?i)(alert\s*\(|document\.cookie|window\.location)"
            ],
            "command_injection": [
                r"(?i)(;\s*cat\s+|;\s*ls\s+|;\s*rm\s+)",
                r"(?i)(\|\s*nc\s+|\|\s*wget\s+|\|\s*curl\s+)",
                r"(?i)(&&\s*whoami|&&\s*id|&&\s*uname)"
            ],
            "path_traversal": [
                r"(?i)(\.\.\/|\.\.\\|%2e%2e%2f|%2e%2e%5c)",
                r"(?i)(\/etc\/passwd|\/etc\/shadow|boot\.ini)"
            ],
            "brute_force": [
                r"(?i)(admin|administrator|root|test|guest)",
                r"(?i)(password|123456|qwerty|letmein)"
            ]
        }

        # Rate limiting thresholds
        self.rate_limits = {
            "login_attempts": {"threshold": 10, "window": 300},  # 10 attempts in 5 min
            "api_requests": {"threshold": 1000, "window": 60},   # 1000 requests per minute
            "failed_requests": {"threshold": 50, "window": 300}  # 50 failures in 5 min
        }

        # Blocked IPs and patterns
        self.blocked_ips: set[str] = set()
        self.suspicious_ips: dict[str, dict[str, Any]] = {}

        # Geographic IP database (simplified)
        self.geo_ip_database = {}  # In production, use actual GeoIP database

    def analyze_event(self, event: SecurityEvent) -> list[str]:
        """Analyze event for intrusion indicators."""
        threats = []

        # Signature-based detection
        threats.extend(self._check_attack_signatures(event))

        # Rate limiting checks
        threats.extend(self._check_rate_limits(event))

        # Geographic anomalies
        threats.extend(self._check_geographic_anomalies(event))

        # Known threat indicators
        threats.extend(self._check_threat_indicators(event))

        return threats

    def _check_attack_signatures(self, event: SecurityEvent) -> list[str]:
        """Check for known attack signatures."""
        threats = []

        # Analyze request data for attack patterns
        request_data = json.dumps(event.metadata).lower()

        for attack_type, patterns in self.attack_patterns.items():
            for pattern in patterns:
                if re.search(pattern, request_data):
                    threats.append(f"signature_match_{attack_type}")
                    break

        return threats

    def _check_rate_limits(self, event: SecurityEvent) -> list[str]:
        """Check for rate limiting violations."""
        threats = []

        if not event.source_ip:
            return threats

        # Track rate limits per IP
        if event.source_ip not in self.suspicious_ips:
            self.suspicious_ips[event.source_ip] = {
                "events": deque(maxlen=1000),
                "login_attempts": deque(maxlen=100),
                "failed_requests": deque(maxlen=200)
            }

        ip_data = self.suspicious_ips[event.source_ip]
        current_time = time.time()

        # Track different types of events
        if event.event_type == MonitoringEvent.LOGIN_ATTEMPT:
            ip_data["login_attempts"].append(current_time)

            # Check login attempt rate
            recent_logins = [
                t for t in ip_data["login_attempts"]
                if current_time - t < self.rate_limits["login_attempts"]["window"]
            ]

            if len(recent_logins) > self.rate_limits["login_attempts"]["threshold"]:
                threats.append("excessive_login_attempts")

        if not event.success:
            ip_data["failed_requests"].append(current_time)

            # Check failed request rate
            recent_failures = [
                t for t in ip_data["failed_requests"]
                if current_time - t < self.rate_limits["failed_requests"]["window"]
            ]

            if len(recent_failures) > self.rate_limits["failed_requests"]["threshold"]:
                threats.append("excessive_failed_requests")

        # General API request rate
        ip_data["events"].append(current_time)
        recent_events = [
            t for t in ip_data["events"]
            if current_time - t < self.rate_limits["api_requests"]["window"]
        ]

        if len(recent_events) > self.rate_limits["api_requests"]["threshold"]:
            threats.append("api_rate_limit_exceeded")

        return threats

    def _check_geographic_anomalies(self, event: SecurityEvent) -> list[str]:
        """Check for geographic anomalies."""
        threats = []

        if not event.source_ip or not event.user_id:
            return threats

        # In production, use actual GeoIP lookup
        # For now, simulate geographic checks
        try:
            ip_obj = ipaddress.ip_address(event.source_ip)
            if ip_obj.is_private:
                return threats  # Skip private IPs

            # Simulate unusual geographic access
            if event.user_id and self._is_unusual_geographic_access(event.user_id, event.source_ip):
                threats.append("unusual_geographic_access")

        except ValueError:
            # Invalid IP address
            threats.append("invalid_source_ip")

        return threats

    def _is_unusual_geographic_access(self, user_id: str, ip_address: str) -> bool:
        """Check if geographic access is unusual for user."""
        # In production, implement actual geographic analysis
        # For now, return False to avoid false positives
        return False

    def _check_threat_indicators(self, event: SecurityEvent) -> list[str]:
        """Check against threat intelligence indicators."""
        threats = []

        # Check IP against threat intel
        if event.source_ip and self._is_known_threat_ip(event.source_ip):
            threats.append("known_threat_ip")

        # Check for malicious patterns in metadata
        request_str = json.dumps(event.metadata)
        if self._contains_threat_indicators(request_str):
            threats.append("threat_indicator_match")

        return threats

    def _is_known_threat_ip(self, ip_address: str) -> bool:
        """Check if IP is in threat intelligence database."""
        # In production, check against threat intel feeds
        return ip_address in self.blocked_ips

    def _contains_threat_indicators(self, text: str) -> bool:
        """Check if text contains threat indicators."""
        # Simple threat indicators check
        threat_keywords = [
            "malware", "botnet", "c2", "command_control",
            "exploit", "payload", "shellcode"
        ]

        text_lower = text.lower()
        return any(keyword in text_lower for keyword in threat_keywords)


class BehavioralAnalyzer:
    """Behavioral anomaly detection system."""

    def __init__(self):
        self.config = get_config()

        # Behavioral baselines
        self.baselines: dict[str, BehavioralBaseline] = {}

        # Event history for pattern analysis
        self.event_history: dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))

        # Anomaly detection thresholds
        self.anomaly_threshold = 2.5  # Standard deviations
        self.min_samples = 10  # Minimum samples for baseline

    def analyze_behavior(self, event: SecurityEvent) -> list[str]:
        """Analyze event for behavioral anomalies."""
        anomalies = []

        # Analyze different behavioral aspects
        anomalies.extend(self._check_frequency_anomalies(event))
        anomalies.extend(self._check_pattern_anomalies(event))
        anomalies.extend(self._check_time_anomalies(event))
        anomalies.extend(self._check_volume_anomalies(event))

        # Update behavioral baselines
        self._update_baselines(event)

        return anomalies

    def _check_frequency_anomalies(self, event: SecurityEvent) -> list[str]:
        """Check for frequency-based anomalies."""
        anomalies = []

        entity_key = f"{event.user_id}_{event.component}_{event.action}"
        baseline_key = f"frequency_{entity_key}"

        if baseline_key in self.baselines:
            baseline = self.baselines[baseline_key]

            # Calculate current frequency
            current_time = time.time()
            recent_events = [
                e for e in self.event_history[entity_key]
                if current_time - e["timestamp"] < 3600  # Last hour
            ]

            current_frequency = len(recent_events)

            # Check if frequency is anomalous
            if baseline.sample_count >= self.min_samples:
                z_score = (current_frequency - baseline.baseline_value) / max(baseline.standard_deviation, 0.1)

                if abs(z_score) > self.anomaly_threshold:
                    anomalies.append(f"frequency_anomaly_z{z_score:.2f}")

        # Record event for future analysis
        self.event_history[entity_key].append({
            "timestamp": time.time(),
            "event": event
        })

        return anomalies

    def _check_pattern_anomalies(self, event: SecurityEvent) -> list[str]:
        """Check for pattern-based anomalies."""
        anomalies = []

        if not event.user_id:
            return anomalies

        user_history = self.event_history[f"user_{event.user_id}"]

        # Check for unusual action sequences
        if len(user_history) >= 5:
            recent_actions = [e["event"].action for e in list(user_history)[-5:]]

            # Look for suspicious patterns
            if self._is_suspicious_action_sequence(recent_actions):
                anomalies.append("suspicious_action_sequence")

        # Check for unusual resource access patterns
        if event.resource:
            resource_history = [
                e for e in user_history
                if e["event"].resource == event.resource
            ]

            if len(resource_history) == 1 and event.resource.startswith("admin"):
                anomalies.append("unusual_admin_resource_access")

        return anomalies

    def _is_suspicious_action_sequence(self, actions: list[str]) -> bool:
        """Check if action sequence is suspicious."""
        # Define suspicious patterns
        suspicious_sequences = [
            ["login", "admin_access", "config_change", "user_create", "logout"],
            ["login", "data_export", "data_export", "data_export", "logout"],
            ["failed_login", "failed_login", "login", "privilege_escalation"]
        ]

        for suspicious_seq in suspicious_sequences:
            if len(actions) >= len(suspicious_seq):
                if actions[-len(suspicious_seq):] == suspicious_seq:
                    return True

        return False

    def _check_time_anomalies(self, event: SecurityEvent) -> list[str]:
        """Check for time-based anomalies."""
        anomalies = []

        if not event.user_id:
            return anomalies

        # Check for unusual access times
        event_hour = event.timestamp.hour

        # Define normal business hours (configurable)
        business_hours = range(8, 18)  # 8 AM to 6 PM

        if event_hour not in business_hours:
            # Check if user normally accesses during off-hours
            user_history = self.event_history[f"user_{event.user_id}"]

            if len(user_history) >= 20:  # Enough history
                off_hours_count = sum(
                    1 for e in user_history
                    if e["event"].timestamp.hour not in business_hours
                )

                off_hours_ratio = off_hours_count / len(user_history)

                if off_hours_ratio < 0.1:  # Usually works during business hours
                    anomalies.append("unusual_access_time")

        return anomalies

    def _check_volume_anomalies(self, event: SecurityEvent) -> list[str]:
        """Check for volume-based anomalies."""
        anomalies = []

        # Check data volume in event metadata
        if "data_size" in event.metadata:
            data_size = event.metadata["data_size"]

            volume_key = f"volume_{event.user_id}_{event.action}"

            if volume_key in self.baselines:
                baseline = self.baselines[volume_key]

                if baseline.sample_count >= self.min_samples:
                    z_score = (data_size - baseline.baseline_value) / max(baseline.standard_deviation, 1.0)

                    if z_score > self.anomaly_threshold:
                        anomalies.append(f"volume_anomaly_z{z_score:.2f}")

        return anomalies

    def _update_baselines(self, event: SecurityEvent) -> None:
        """Update behavioral baselines with new event data."""
        current_time = time.time()

        # Update frequency baseline
        entity_key = f"{event.user_id}_{event.component}_{event.action}"
        baseline_key = f"frequency_{entity_key}"

        recent_events = [
            e for e in self.event_history[entity_key]
            if current_time - e["timestamp"] < 86400  # Last 24 hours
        ]

        if len(recent_events) >= self.min_samples:
            frequencies = []

            # Calculate hourly frequencies
            for hour in range(24):
                hour_events = [
                    e for e in recent_events
                    if datetime.fromtimestamp(e["timestamp"]).hour == hour
                ]
                frequencies.append(len(hour_events))

            if frequencies:
                baseline_value = statistics.mean(frequencies)
                std_dev = statistics.stdev(frequencies) if len(frequencies) > 1 else 0.0

                self.baselines[baseline_key] = BehavioralBaseline(
                    entity_id=entity_key,
                    metric_name="hourly_frequency",
                    baseline_value=baseline_value,
                    standard_deviation=std_dev,
                    sample_count=len(frequencies),
                    last_updated=datetime.utcnow(),
                    confidence_level=min(len(frequencies) / 100.0, 1.0)
                )


class SecurityMonitor:
    """Main security monitoring system."""

    def __init__(self):
        self.config = get_config()

        # Initialize subsystems
        self.intrusion_detector = IntrusionDetector()
        self.behavioral_analyzer = BehavioralAnalyzer()

        # Event processing
        self.event_queue: asyncio.Queue = asyncio.Queue()
        self.processed_events: list[SecurityEvent] = []

        # Threat intelligence
        self.threat_indicators: dict[str, ThreatIndicator] = {}

        # Response handlers
        self.response_handlers: dict[str, Callable] = {}

        # Monitoring state
        self.monitoring_active = False
        self._background_tasks: list[asyncio.Task] = []

        # Performance metrics
        self.processing_stats = {
            "events_processed": 0,
            "threats_detected": 0,
            "false_positives": 0,
            "response_time_ms": []
        }

    async def start_monitoring(self) -> None:
        """Start the security monitoring system."""
        self.monitoring_active = True

        # Start background tasks
        self._background_tasks = [
            asyncio.create_task(self._event_processor()),
            asyncio.create_task(self._threat_correlator()),
            asyncio.create_task(self._baseline_updater()),
            asyncio.create_task(self._threat_intel_updater()),
            asyncio.create_task(self._cleanup_old_data())
        ]

        await logger.log_critical_event(
            "security_monitoring_started",
            "security_monitor",
            {"timestamp": datetime.utcnow().isoformat()}
        )

    async def stop_monitoring(self) -> None:
        """Stop the security monitoring system."""
        self.monitoring_active = False

        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()

        await asyncio.gather(*self._background_tasks, return_exceptions=True)

        await logger.log_critical_event(
            "security_monitoring_stopped",
            "security_monitor",
            {"timestamp": datetime.utcnow().isoformat()}
        )

    async def process_event(self, event: SecurityEvent) -> None:
        """Process a security event."""
        if not self.monitoring_active:
            return

        await self.event_queue.put(event)

    async def _event_processor(self) -> None:
        """Background task to process security events."""
        while self.monitoring_active:
            try:
                # Process events from queue
                event = await asyncio.wait_for(self.event_queue.get(), timeout=1.0)

                start_time = time.time()

                # Analyze event for threats
                intrusion_threats = self.intrusion_detector.analyze_event(event)
                behavioral_anomalies = self.behavioral_analyzer.analyze_behavior(event)

                # Combine all threats and anomalies
                all_threats = intrusion_threats + behavioral_anomalies
                event.anomaly_indicators = all_threats

                # Calculate risk score
                event.risk_score = self._calculate_risk_score(event, all_threats)

                # Store processed event
                self.processed_events.append(event)

                # Trigger responses if necessary
                if event.risk_score > 0.7:  # High risk threshold
                    await self._trigger_automated_response(event)

                # Update statistics
                processing_time = (time.time() - start_time) * 1000
                self.processing_stats["events_processed"] += 1
                self.processing_stats["response_time_ms"].append(processing_time)

                if all_threats:
                    self.processing_stats["threats_detected"] += 1

                # Log significant events
                if event.risk_score > 0.5:
                    await logger.log_critical_event(
                        "security_threat_detected",
                        "security_monitor",
                        {
                            "event_id": event.event_id,
                            "risk_score": event.risk_score,
                            "threats": all_threats,
                            "source_ip": event.source_ip,
                            "user_id": event.user_id
                        }
                    )

            except TimeoutError:
                continue
            except Exception as e:
                await logger.log_critical_event(
                    "event_processing_error",
                    "security_monitor",
                    {"error": str(e)}
                )

    def _calculate_risk_score(self, event: SecurityEvent, threats: list[str]) -> float:
        """Calculate risk score for an event."""
        base_score = 0.0

        # Base score based on event type and success
        if not event.success:
            base_score += 0.2

        if event.event_type in [MonitoringEvent.PRIVILEGE_ESCALATION,
                               MonitoringEvent.CONFIGURATION_CHANGE]:
            base_score += 0.3

        # Add score for each threat
        threat_scores = {
            "sql_injection": 0.8,
            "xss": 0.6,
            "command_injection": 0.9,
            "path_traversal": 0.7,
            "brute_force": 0.5,
            "excessive_login_attempts": 0.6,
            "api_rate_limit_exceeded": 0.4,
            "known_threat_ip": 0.8,
            "frequency_anomaly": 0.3,
            "pattern_anomaly": 0.5,
            "time_anomaly": 0.2,
            "volume_anomaly": 0.4
        }

        for threat in threats:
            # Extract base threat type (remove suffixes like _z2.5)
            threat_base = threat.split('_')[0] + '_' + threat.split('_')[1] if '_' in threat else threat
            score = threat_scores.get(threat_base, 0.3)
            base_score += score

        # Cap at 1.0
        return min(base_score, 1.0)

    async def _trigger_automated_response(self, event: SecurityEvent) -> None:
        """Trigger automated response to high-risk events."""
        response_actions = []

        # Determine appropriate responses
        if event.risk_score > 0.9:  # Critical threat
            if event.source_ip:
                await self._block_ip_address(event.source_ip, duration=timedelta(hours=24))
                response_actions.append(f"blocked_ip_{event.source_ip}")

            if event.user_id:
                await self._suspend_user_session(event.user_id)
                response_actions.append(f"suspended_user_{event.user_id}")

        elif event.risk_score > 0.7:  # High threat
            if event.source_ip and "excessive" in str(event.anomaly_indicators):
                await self._rate_limit_ip(event.source_ip, duration=timedelta(minutes=30))
                response_actions.append(f"rate_limited_ip_{event.source_ip}")

        if response_actions:
            await logger.log_critical_event(
                "automated_security_response",
                "security_monitor",
                {
                    "event_id": event.event_id,
                    "risk_score": event.risk_score,
                    "actions": response_actions
                }
            )

    async def _block_ip_address(self, ip_address: str, duration: timedelta) -> None:
        """Block an IP address."""
        self.intrusion_detector.blocked_ips.add(ip_address)

        # Schedule unblocking
        asyncio.create_task(self._schedule_ip_unblock(ip_address, duration))

        await logger.log_critical_event(
            "ip_address_blocked",
            "security_monitor",
            {
                "ip_address": ip_address,
                "duration_seconds": int(duration.total_seconds())
            }
        )

    async def _schedule_ip_unblock(self, ip_address: str, duration: timedelta) -> None:
        """Schedule IP unblocking after duration."""
        await asyncio.sleep(duration.total_seconds())

        if ip_address in self.intrusion_detector.blocked_ips:
            self.intrusion_detector.blocked_ips.remove(ip_address)

            await logger.log_critical_event(
                "ip_address_unblocked",
                "security_monitor",
                {"ip_address": ip_address}
            )

    async def _rate_limit_ip(self, ip_address: str, duration: timedelta) -> None:
        """Apply rate limiting to an IP address."""
        # In production, integrate with rate limiting system
        await logger.log_critical_event(
            "ip_rate_limited",
            "security_monitor",
            {
                "ip_address": ip_address,
                "duration_seconds": int(duration.total_seconds())
            }
        )

    async def _suspend_user_session(self, user_id: str) -> None:
        """Suspend user sessions."""
        # In production, integrate with session management
        await logger.log_critical_event(
            "user_session_suspended",
            "security_monitor",
            {"user_id": user_id}
        )

    async def _threat_correlator(self) -> None:
        """Background task to correlate threats across events."""
        while self.monitoring_active:
            try:
                # Correlate events every 5 minutes
                await asyncio.sleep(300)

                # Analyze recent events for patterns
                now = datetime.utcnow()
                recent_events = [
                    e for e in self.processed_events
                    if (now - e.timestamp).total_seconds() < 3600  # Last hour
                ]

                # Group events by source IP
                ip_groups = defaultdict(list)
                for event in recent_events:
                    if event.source_ip:
                        ip_groups[event.source_ip].append(event)

                # Look for coordinated attacks
                for ip, events in ip_groups.items():
                    if len(events) > 20:  # High volume from single IP
                        high_risk_events = [e for e in events if e.risk_score > 0.5]

                        if len(high_risk_events) > 5:
                            await logger.log_critical_event(
                                "coordinated_attack_detected",
                                "security_monitor",
                                {
                                    "source_ip": ip,
                                    "event_count": len(events),
                                    "high_risk_count": len(high_risk_events)
                                }
                            )

            except Exception as e:
                await logger.log_critical_event(
                    "threat_correlation_error",
                    "security_monitor",
                    {"error": str(e)}
                )

    async def _baseline_updater(self) -> None:
        """Background task to update behavioral baselines."""
        while self.monitoring_active:
            try:
                # Update baselines every hour
                await asyncio.sleep(3600)

                # Trigger baseline recalculation
                for baseline in self.behavioral_analyzer.baselines.values():
                    baseline.last_updated = datetime.utcnow()

                await logger.log_critical_event(
                    "behavioral_baselines_updated",
                    "security_monitor",
                    {"baseline_count": len(self.behavioral_analyzer.baselines)}
                )

            except Exception as e:
                await logger.log_critical_event(
                    "baseline_update_error",
                    "security_monitor",
                    {"error": str(e)}
                )

    async def _threat_intel_updater(self) -> None:
        """Background task to update threat intelligence."""
        while self.monitoring_active:
            try:
                # Update threat intel every 6 hours
                await asyncio.sleep(21600)

                # In production, fetch from threat intel feeds
                await self._update_threat_indicators()

            except Exception as e:
                await logger.log_critical_event(
                    "threat_intel_update_error",
                    "security_monitor",
                    {"error": str(e)}
                )

    async def _update_threat_indicators(self) -> None:
        """Update threat intelligence indicators."""
        # In production, integrate with threat intelligence feeds
        # For now, maintain basic indicators

        sample_indicators = [
            ThreatIndicator(
                indicator_id="malicious_ip_1",
                indicator_type="ip",
                value="192.168.1.100",
                threat_level=ThreatLevel.HIGH,
                description="Known botnet C2 server",
                source="internal_analysis",
                created_at=datetime.utcnow(),
                expires_at=datetime.utcnow() + timedelta(days=30),
                metadata={"confidence": 0.9}
            )
        ]

        for indicator in sample_indicators:
            self.threat_indicators[indicator.indicator_id] = indicator

        await logger.log_critical_event(
            "threat_indicators_updated",
            "security_monitor",
            {"indicator_count": len(self.threat_indicators)}
        )

    async def _cleanup_old_data(self) -> None:
        """Background task to cleanup old monitoring data."""
        while self.monitoring_active:
            try:
                # Cleanup every 12 hours
                await asyncio.sleep(43200)

                now = datetime.utcnow()

                # Remove old processed events (keep 7 days)
                cutoff_time = now - timedelta(days=7)
                self.processed_events = [
                    e for e in self.processed_events
                    if e.timestamp > cutoff_time
                ]

                # Remove expired threat indicators
                expired_indicators = [
                    tid for tid, indicator in self.threat_indicators.items()
                    if indicator.expires_at and now > indicator.expires_at
                ]

                for tid in expired_indicators:
                    del self.threat_indicators[tid]

                # Cleanup behavioral analyzer data
                for entity_key in list(self.behavioral_analyzer.event_history.keys()):
                    old_events = [
                        e for e in self.behavioral_analyzer.event_history[entity_key]
                        if now - datetime.fromtimestamp(e["timestamp"]) > timedelta(days=7)
                    ]

                    # Remove old events
                    for old_event in old_events:
                        self.behavioral_analyzer.event_history[entity_key].remove(old_event)

                # Cleanup old baselines
                old_baselines = [
                    key for key, baseline in self.behavioral_analyzer.baselines.items()
                    if now - baseline.last_updated > timedelta(days=30)
                ]

                for key in old_baselines:
                    del self.behavioral_analyzer.baselines[key]

                # Trim performance stats
                if len(self.processing_stats["response_time_ms"]) > 10000:
                    self.processing_stats["response_time_ms"] = \
                        self.processing_stats["response_time_ms"][-5000:]

                await logger.log_critical_event(
                    "monitoring_data_cleanup",
                    "security_monitor",
                    {
                        "events_remaining": len(self.processed_events),
                        "indicators_remaining": len(self.threat_indicators),
                        "baselines_remaining": len(self.behavioral_analyzer.baselines)
                    }
                )

            except Exception as e:
                await logger.log_critical_event(
                    "cleanup_error",
                    "security_monitor",
                    {"error": str(e)}
                )

    def register_response_handler(self, threat_type: str, handler: Callable) -> None:
        """Register custom response handler for threat type."""
        self.response_handlers[threat_type] = handler

    def get_monitoring_statistics(self) -> dict[str, Any]:
        """Get comprehensive monitoring statistics."""
        response_times = self.processing_stats["response_time_ms"]

        stats = {
            "monitoring_active": self.monitoring_active,
            "events_processed": self.processing_stats["events_processed"],
            "threats_detected": self.processing_stats["threats_detected"],
            "false_positives": self.processing_stats["false_positives"],
            "detection_rate": (
                self.processing_stats["threats_detected"] /
                max(self.processing_stats["events_processed"], 1)
            ),
            "queue_size": self.event_queue.qsize(),
            "processed_events_count": len(self.processed_events),
            "threat_indicators_count": len(self.threat_indicators),
            "behavioral_baselines_count": len(self.behavioral_analyzer.baselines),
            "blocked_ips_count": len(self.intrusion_detector.blocked_ips)
        }

        if response_times:
            stats.update({
                "avg_response_time_ms": statistics.mean(response_times),
                "max_response_time_ms": max(response_times),
                "min_response_time_ms": min(response_times)
            })

        return stats

    def get_recent_threats(self, hours: int = 24) -> list[dict[str, Any]]:
        """Get recent security threats."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)

        recent_threats = [
            {
                "event_id": event.event_id,
                "timestamp": event.timestamp.isoformat(),
                "source_ip": event.source_ip,
                "user_id": event.user_id,
                "risk_score": event.risk_score,
                "threats": event.anomaly_indicators,
                "component": event.component,
                "action": event.action
            }
            for event in self.processed_events
            if event.timestamp > cutoff_time and event.risk_score > 0.3
        ]

        return sorted(recent_threats, key=lambda x: x["risk_score"], reverse=True)

    def get_threat_summary(self) -> dict[str, Any]:
        """Get high-level threat summary."""
        now = datetime.utcnow()
        last_24h = now - timedelta(hours=24)

        recent_events = [
            e for e in self.processed_events
            if e.timestamp > last_24h
        ]

        if not recent_events:
            return {
                "total_events": 0,
                "threat_events": 0,
                "critical_threats": 0,
                "top_threat_types": [],
                "top_source_ips": []
            }

        threat_events = [e for e in recent_events if e.anomaly_indicators]
        critical_threats = [e for e in recent_events if e.risk_score > 0.8]

        # Count threat types
        threat_type_counts = defaultdict(int)
        for event in threat_events:
            if event.anomaly_indicators:
                for threat in event.anomaly_indicators:
                    threat_type_counts[threat] += 1

        # Count source IPs
        ip_counts = defaultdict(int)
        for event in recent_events:
            if event.source_ip:
                ip_counts[event.source_ip] += 1

        return {
            "total_events": len(recent_events),
            "threat_events": len(threat_events),
            "critical_threats": len(critical_threats),
            "threat_rate": len(threat_events) / len(recent_events) if recent_events else 0,
            "top_threat_types": sorted(
                threat_type_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10],
            "top_source_ips": sorted(
                ip_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
        }

    async def create_security_event(self,
                                   event_type: MonitoringEvent,
                                   component: str,
                                   action: str,
                                   source_ip: str | None = None,
                                   user_id: str | None = None,
                                   resource: str | None = None,
                                   success: bool = True,
                                   metadata: dict[str, Any] | None = None) -> SecurityEvent:
        """Create and process a security event."""

        event_id = hashlib.sha256(
            f"{datetime.utcnow().isoformat()}{component}{action}{source_ip or ''}".encode()
        ).hexdigest()[:16]

        event = SecurityEvent(
            event_id=event_id,
            event_type=event_type,
            timestamp=datetime.utcnow(),
            source_ip=source_ip,
            user_id=user_id,
            component=component,
            action=action,
            resource=resource,
            success=success,
            metadata=metadata or {}
        )

        await self.process_event(event)
        return event

    async def manual_threat_response(self, event_id: str,
                                   response_action: str,
                                   user_id: str) -> bool:
        """Manually trigger threat response."""
        # Find the event
        event = None
        for e in self.processed_events:
            if e.event_id == event_id:
                event = e
                break

        if not event:
            return False

        # Execute manual response
        if response_action == "block_ip" and event.source_ip:
            await self._block_ip_address(event.source_ip, timedelta(hours=24))
        elif response_action == "suspend_user" and event.user_id:
            await self._suspend_user_session(event.user_id)
        elif response_action == "mark_false_positive":
            self.processing_stats["false_positives"] += 1
            event.risk_score = 0.0
            event.anomaly_indicators = []

        await logger.log_critical_event(
            "manual_threat_response",
            user_id,
            {
                "event_id": event_id,
                "response_action": response_action,
                "timestamp": datetime.utcnow().isoformat()
            }
        )

        return True

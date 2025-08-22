"""
Tests for FlashMM Security Orchestrator

Comprehensive tests for the security orchestrator including threat detection,
automated response, security state management, and integration testing.
"""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

from flashmm.security.security_orchestrator import (
    SecurityLevel,
    SecurityMetrics,
    SecurityOrchestrator,
    SecurityState,
    SecurityThreat,
    ThreatType,
)
from flashmm.utils.exceptions import AuthenticationError


class TestSecurityOrchestrator:
    """Test suite for Security Orchestrator."""

    @pytest.fixture
    async def orchestrator(self):
        """Create security orchestrator for testing."""
        orchestrator = SecurityOrchestrator()
        await orchestrator.start()
        yield orchestrator
        await orchestrator.stop()

    @pytest.fixture
    def sample_request(self):
        """Sample request data for testing."""
        return {
            "request_id": "test_request_123",
            "source_ip": "192.168.1.100",
            "user_agent": "TestAgent/1.0",
            "auth_type": "api_key",
            "api_key": "test_api_key"
        }

    async def test_orchestrator_initialization(self, orchestrator):
        """Test security orchestrator initialization."""
        assert orchestrator.current_state == SecurityState.NORMAL
        assert len(orchestrator.active_threats) == 0
        assert isinstance(orchestrator.security_metrics, SecurityMetrics)
        assert len(orchestrator._background_tasks) > 0

    async def test_request_authentication_success(self, orchestrator, sample_request):
        """Test successful request authentication."""
        with patch.object(orchestrator.auth_manager, 'verify_api_key', return_value="admin"):
            with patch.object(orchestrator.auth_manager, 'create_access_token', return_value="test_token"):
                result = await orchestrator.authenticate_request(sample_request)

                assert result["authenticated"] is True
                assert result["role"] == "admin"
                assert "token" in result
                assert orchestrator.security_metrics.total_auth_attempts == 1
                assert orchestrator.security_metrics.failed_auth_attempts == 0

    async def test_request_authentication_failure(self, orchestrator, sample_request):
        """Test failed request authentication."""
        with patch.object(orchestrator.auth_manager, 'verify_api_key', return_value=None):
            result = await orchestrator.authenticate_request(sample_request)

            assert result["authenticated"] is False
            assert result["reason"] == "invalid_api_key"
            assert orchestrator.security_metrics.failed_auth_attempts == 1

    async def test_blocked_ip_authentication(self, orchestrator, sample_request):
        """Test authentication with blocked IP."""
        # Block the IP first
        orchestrator.blocked_ips.add("192.168.1.100")

        with pytest.raises(AuthenticationError, match="Access denied: IP blocked"):
            await orchestrator.authenticate_request(sample_request)

    async def test_rate_limiting(self, orchestrator, sample_request):
        """Test rate limiting functionality."""
        # Simulate many requests from same IP
        orchestrator.security_policies["rate_limit_requests"] = 3

        # First few requests should pass
        for _ in range(3):
            with patch.object(orchestrator.auth_manager, 'verify_api_key', return_value="admin"):
                with patch.object(orchestrator.auth_manager, 'create_access_token', return_value="token"):
                    result = await orchestrator.authenticate_request(sample_request)
                    assert result["authenticated"] is True

        # Next request should fail due to rate limiting
        with pytest.raises(AuthenticationError, match="Rate limit exceeded"):
            await orchestrator.authenticate_request(sample_request)

    async def test_threat_registration(self, orchestrator):
        """Test threat registration and handling."""
        threat = SecurityThreat(
            threat_id="test_threat_001",
            threat_type=ThreatType.AUTHENTICATION_FAILURE,
            severity=SecurityLevel.HIGH,
            source_ip="192.168.1.100",
            user_id="test_user",
            component="authentication",
            description="Test threat",
            timestamp=datetime.utcnow(),
            metadata={"test": True}
        )

        await orchestrator._register_threat(threat)

        assert "test_threat_001" in orchestrator.active_threats
        assert orchestrator.security_metrics.threats_detected == 1
        assert orchestrator.current_state == SecurityState.HIGH_ALERT  # Should escalate for HIGH threat

    async def test_brute_force_detection(self, orchestrator, sample_request):
        """Test brute force attack detection."""
        orchestrator.security_policies["max_auth_failures"] = 3

        # Simulate multiple failed authentication attempts
        with patch.object(orchestrator.auth_manager, 'verify_api_key', return_value=None):
            for _i in range(5):  # Exceed the limit
                await orchestrator.authenticate_request(sample_request)

        # Should detect brute force pattern
        brute_force_threats = [
            threat for threat in orchestrator.active_threats.values()
            if threat.threat_type == ThreatType.ANOMALOUS_BEHAVIOR
        ]

        assert len(brute_force_threats) > 0
        assert "192.168.1.100" in orchestrator.blocked_ips  # Should be blocked

    async def test_security_state_management(self, orchestrator):
        """Test security state escalation and de-escalation."""
        # Create high severity threat
        high_threat = SecurityThreat(
            threat_id="high_threat",
            threat_type=ThreatType.SYSTEM_INTRUSION,
            severity=SecurityLevel.HIGH,
            source_ip=None,
            user_id=None,
            component="system",
            description="High severity threat",
            timestamp=datetime.utcnow(),
            metadata={}
        )

        await orchestrator._register_threat(high_threat)
        assert orchestrator.current_state == SecurityState.HIGH_ALERT

        # Resolve threat
        high_threat.resolved = True

        # Should de-escalate after monitoring cycle
        await orchestrator._de_escalate_security_state()
        assert orchestrator.current_state == SecurityState.NORMAL

    async def test_emergency_shutdown(self, orchestrator):
        """Test emergency shutdown procedure."""
        result = await orchestrator.emergency_shutdown(
            reason="Security test",
            user="test_admin"
        )

        assert result["status"] == "emergency_shutdown_complete"
        assert result["reason"] == "Security test"
        assert orchestrator.current_state == SecurityState.EMERGENCY

    async def test_background_monitoring_tasks(self, orchestrator):
        """Test that background monitoring tasks are running."""
        # Background tasks should be running
        running_tasks = [task for task in orchestrator._background_tasks if not task.done()]
        assert len(running_tasks) > 0

        # Wait a short time and verify tasks are still running
        await asyncio.sleep(0.1)
        still_running = [task for task in orchestrator._background_tasks if not task.done()]
        assert len(still_running) == len(running_tasks)

    def test_security_metrics_tracking(self, orchestrator):
        """Test security metrics tracking."""
        # Initial state
        assert orchestrator.security_metrics.total_auth_attempts == 0
        assert orchestrator.security_metrics.threats_detected == 0

        # Update metrics
        orchestrator.security_metrics.total_auth_attempts = 10
        orchestrator.security_metrics.failed_auth_attempts = 2
        orchestrator.security_metrics.threats_detected = 1

        status = orchestrator.get_security_status()
        assert status["metrics"]["total_auth_attempts"] == 10
        assert status["metrics"]["failed_auth_attempts"] == 2
        assert status["metrics"]["threats_detected"] == 1

    async def test_threat_response_automation(self, orchestrator):
        """Test automated threat response."""
        # Create critical threat
        critical_threat = SecurityThreat(
            threat_id="critical_threat",
            threat_type=ThreatType.DATA_BREACH,
            severity=SecurityLevel.CRITICAL,
            source_ip="192.168.1.200",
            user_id="suspicious_user",
            component="data_access",
            description="Critical data breach detected",
            timestamp=datetime.utcnow(),
            metadata={}
        )

        await orchestrator._register_threat(critical_threat)

        # Verify automated response
        assert orchestrator.current_state == SecurityState.HIGH_ALERT
        assert critical_threat.resolution_actions and len(critical_threat.resolution_actions) > 0

        # Should trigger emergency procedures for data breach
        emergency_actions = [action for action in critical_threat.resolution_actions
                           if "emergency" in action]
        assert len(emergency_actions) > 0

    async def test_ip_blocking_and_unblocking(self, orchestrator):
        """Test IP blocking and automatic unblocking."""
        test_ip = "192.168.1.300"
        block_duration = timedelta(seconds=1)  # Short duration for testing

        await orchestrator._block_ip(test_ip, block_duration)
        assert test_ip in orchestrator.blocked_ips

        # Wait for unblocking
        await asyncio.sleep(1.5)  # Wait slightly longer than block duration

        # IP should be automatically unblocked
        assert test_ip not in orchestrator.blocked_ips

    def test_security_policy_loading(self, orchestrator):
        """Test security policy loading and configuration."""
        policies = orchestrator.security_policies

        # Verify required policies are loaded
        required_policies = [
            "max_auth_failures", "rate_limit_requests", "session_timeout",
            "key_rotation_interval", "emergency_contacts"
        ]

        for policy in required_policies:
            assert policy in policies

        # Verify default values
        assert isinstance(policies["max_auth_failures"], int)
        assert policies["max_auth_failures"] > 0
        assert isinstance(policies["rate_limit_requests"], int)
        assert policies["rate_limit_requests"] > 0


class TestSecurityThreat:
    """Test suite for SecurityThreat data structure."""

    def test_threat_creation(self):
        """Test threat creation with all fields."""
        threat = SecurityThreat(
            threat_id="test_threat",
            threat_type=ThreatType.AUTHENTICATION_FAILURE,
            severity=SecurityLevel.MEDIUM,
            source_ip="192.168.1.1",
            user_id="test_user",
            component="auth",
            description="Test description",
            timestamp=datetime.utcnow(),
            metadata={"key": "value"}
        )

        assert threat.threat_id == "test_threat"
        assert threat.threat_type == ThreatType.AUTHENTICATION_FAILURE
        assert threat.severity == SecurityLevel.MEDIUM
        assert threat.resolved is False
        assert not threat.resolution_actions or len(threat.resolution_actions) == 0

    def test_threat_resolution_actions(self):
        """Test threat resolution actions tracking."""
        threat = SecurityThreat(
            threat_id="test_threat",
            threat_type=ThreatType.AUTHENTICATION_FAILURE,
            severity=SecurityLevel.MEDIUM,
            source_ip=None,
            user_id=None,
            component="auth",
            description="Test",
            timestamp=datetime.utcnow(),
            metadata={}
        )

        # Add resolution actions (ensure resolution_actions is initialized)
        if threat.resolution_actions is None:
            threat.resolution_actions = []
        threat.resolution_actions.append("blocked_ip")
        threat.resolution_actions.append("escalated_alert")

        assert threat.resolution_actions is not None and len(threat.resolution_actions) == 2
        assert threat.resolution_actions is not None and "blocked_ip" in threat.resolution_actions


class TestSecurityMetrics:
    """Test suite for SecurityMetrics tracking."""

    def test_metrics_initialization(self):
        """Test metrics initialization."""
        metrics = SecurityMetrics()

        assert metrics.total_auth_attempts == 0
        assert metrics.failed_auth_attempts == 0
        assert metrics.threats_detected == 0
        assert isinstance(metrics.last_update, datetime)

    def test_metrics_updates(self):
        """Test metrics updates."""
        metrics = SecurityMetrics()

        # Update metrics
        metrics.total_auth_attempts = 100
        metrics.failed_auth_attempts = 5
        metrics.threats_detected = 2
        metrics.threats_resolved = 1

        assert metrics.total_auth_attempts == 100
        assert metrics.failed_auth_attempts == 5
        assert metrics.threats_detected == 2
        assert metrics.threats_resolved == 1


@pytest.mark.asyncio
class TestSecurityIntegration:
    """Integration tests for security orchestrator with other components."""

    async def test_auth_manager_integration(self):
        """Test integration with authentication manager."""
        orchestrator = SecurityOrchestrator()

        # Test that auth manager is properly initialized
        assert orchestrator.auth_manager is not None
        assert orchestrator.authz_manager is not None

        # Test API key verification integration
        with patch.object(orchestrator.auth_manager, 'verify_api_key', return_value="test_role"):
            request_data = {
                "auth_type": "api_key",
                "api_key": "test_key",
                "source_ip": "127.0.0.1"
            }

            result = await orchestrator._perform_authentication(request_data)
            assert result["authenticated"] is True
            assert result["role"] == "test_role"

    async def test_key_manager_integration(self):
        """Test integration with key managers."""
        orchestrator = SecurityOrchestrator()

        # Test that key managers are initialized
        assert orchestrator.hot_key_manager is not None
        assert orchestrator.warm_key_manager is not None
        assert orchestrator.cold_key_manager is not None
        assert orchestrator.key_rotation_manager is not None

        # Test key rotation status retrieval
        rotation_status = orchestrator.key_rotation_manager.get_rotation_status()
        assert "last_rotation" in rotation_status
        assert "needs_rotation" in rotation_status

    async def test_security_state_persistence(self):
        """Test security state changes persist across operations."""
        orchestrator = SecurityOrchestrator()

        # Change security state
        await orchestrator._escalate_security_state(SecurityState.HIGH_ALERT)
        assert orchestrator.current_state == SecurityState.HIGH_ALERT

        # Create new threat
        threat = SecurityThreat(
            threat_id="persist_test",
            threat_type=ThreatType.SYSTEM_INTRUSION,
            severity=SecurityLevel.HIGH,
            source_ip=None,
            user_id=None,
            component="test",
            description="Test persistence",
            timestamp=datetime.utcnow(),
            metadata={}
        )

        await orchestrator._register_threat(threat)

        # State should remain elevated
        assert orchestrator.current_state == SecurityState.HIGH_ALERT
        assert len(orchestrator.active_threats) == 1


@pytest.mark.performance
class TestSecurityPerformance:
    """Performance tests for security orchestrator."""

    async def test_authentication_performance(self):
        """Test authentication performance under load."""
        orchestrator = SecurityOrchestrator()

        # Mock successful authentication
        with patch.object(orchestrator.auth_manager, 'verify_api_key', return_value="test_role"):
            with patch.object(orchestrator.auth_manager, 'create_access_token', return_value="test_token"):

                start_time = datetime.utcnow()

                # Perform multiple authentications
                tasks = []
                for i in range(100):
                    request_data = {
                        "request_id": f"perf_test_{i}",
                        "auth_type": "api_key",
                        "api_key": "test_key",
                        "source_ip": f"192.168.1.{i % 255}"
                    }
                    tasks.append(orchestrator.authenticate_request(request_data))

                results = await asyncio.gather(*tasks, return_exceptions=True)

                end_time = datetime.utcnow()
                duration = (end_time - start_time).total_seconds()

                # Verify performance requirements (<10ms per auth)
                avg_time_per_auth = duration / 100
                assert avg_time_per_auth < 0.010  # Less than 10ms

                # Verify all authentications succeeded
                successful_auths = [r for r in results if isinstance(r, dict) and r.get("authenticated")]
                assert len(successful_auths) == 100

    async def test_threat_processing_performance(self):
        """Test threat processing performance."""
        orchestrator = SecurityOrchestrator()

        start_time = datetime.utcnow()

        # Create multiple threats
        for i in range(50):
            threat = SecurityThreat(
                threat_id=f"perf_threat_{i}",
                threat_type=ThreatType.AUTHENTICATION_FAILURE,
                severity=SecurityLevel.MEDIUM,
                source_ip=f"192.168.1.{i}",
                user_id=f"user_{i}",
                component="test",
                description=f"Performance test threat {i}",
                timestamp=datetime.utcnow(),
                metadata={"test": True}
            )

            await orchestrator._register_threat(threat)

        end_time = datetime.utcnow()
        duration = (end_time - start_time).total_seconds()

        # Verify threat processing is fast
        avg_time_per_threat = duration / 50
        assert avg_time_per_threat < 0.005  # Less than 5ms per threat

        assert len(orchestrator.active_threats) == 50
        assert orchestrator.security_metrics.threats_detected == 50


@pytest.mark.integration
class TestSecurityOrchestoratorIntegration:
    """End-to-end integration tests for security orchestrator."""

    async def test_complete_security_workflow(self):
        """Test complete security workflow from threat detection to resolution."""
        orchestrator = SecurityOrchestrator()
        await orchestrator.start()

        try:
            # 1. Normal authentication
            request_data = {
                "auth_type": "api_key",
                "api_key": "valid_key",
                "source_ip": "192.168.1.10"
            }

            with patch.object(orchestrator.auth_manager, 'verify_api_key', return_value="user"):
                with patch.object(orchestrator.auth_manager, 'create_access_token', return_value="token"):
                    result = await orchestrator.authenticate_request(request_data)
                    assert result["authenticated"] is True

            # 2. Multiple failed attempts (trigger brute force detection)
            with patch.object(orchestrator.auth_manager, 'verify_api_key', return_value=None):
                for _i in range(6):  # Exceed threshold
                    await orchestrator.authenticate_request({
                        "auth_type": "api_key",
                        "api_key": "invalid_key",
                        "source_ip": "192.168.1.50"
                    })

            # 3. Verify threat detection and response
            assert len(orchestrator.active_threats) > 0
            assert "192.168.1.50" in orchestrator.blocked_ips
            assert orchestrator.current_state in [SecurityState.ELEVATED, SecurityState.HIGH_ALERT]

            # 4. Resolve threats
            for threat_id in list(orchestrator.active_threats.keys()):
                await orchestrator.resolve_threat(threat_id, "Resolved for testing")

            # 5. Verify system returns to normal
            await orchestrator._de_escalate_security_state()
            assert orchestrator.current_state == SecurityState.NORMAL

        finally:
            await orchestrator.stop()

    async def test_security_metrics_reporting(self):
        """Test security metrics reporting functionality."""
        orchestrator = SecurityOrchestrator()

        # Generate some activity
        orchestrator.security_metrics.total_auth_attempts = 1000
        orchestrator.security_metrics.failed_auth_attempts = 50
        orchestrator.security_metrics.threats_detected = 5
        orchestrator.security_metrics.threats_resolved = 3
        orchestrator.blocked_ips.add("192.168.1.100")

        # Get status report
        status = orchestrator.get_security_status()

        assert status["security_state"] == SecurityState.NORMAL.value
        assert status["active_threats"] == 0  # No active threats
        assert status["blocked_ips"] == 1
        assert status["metrics"]["total_auth_attempts"] == 1000
        assert status["metrics"]["failed_auth_attempts"] == 50
        assert "key_rotation_status" in status
        assert "policies" in status

    async def test_concurrent_threat_handling(self):
        """Test handling multiple concurrent threats."""
        orchestrator = SecurityOrchestrator()

        # Create multiple threats concurrently
        threat_tasks = []
        for i in range(20):
            threat = SecurityThreat(
                threat_id=f"concurrent_threat_{i}",
                threat_type=ThreatType.AUTHENTICATION_FAILURE,
                severity=SecurityLevel.MEDIUM,
                source_ip=f"192.168.1.{100 + i}",
                user_id=f"user_{i}",
                component="auth",
                description=f"Concurrent threat {i}",
                timestamp=datetime.utcnow(),
                metadata={"concurrent_test": True}
            )

            threat_tasks.append(orchestrator._register_threat(threat))

        # Wait for all threats to be processed
        await asyncio.gather(*threat_tasks)

        # Verify all threats were registered
        assert len(orchestrator.active_threats) == 20
        assert orchestrator.security_metrics.threats_detected == 20

        # Verify system handled concurrent threats properly
        concurrent_threats = [
            t for t in orchestrator.active_threats.values()
            if t.metadata.get("concurrent_test")
        ]
        assert len(concurrent_threats) == 20

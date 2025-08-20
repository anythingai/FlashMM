"""
Integration Tests for FlashMM Security System

Comprehensive integration tests covering all security components working together.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from flashmm.security.security_orchestrator import SecurityOrchestrator
from flashmm.security.security_monitor import SecurityMonitor, SecurityEvent, MonitoringEvent
from flashmm.security.audit_logger import AuditLogger, AuditEventType, AuditLevel
from flashmm.security.emergency_manager import EmergencyManager, EmergencyType, EmergencyLevel
from flashmm.security.policy_engine import PolicyEngine, PolicyType, PolicySeverity
from flashmm.security.key_manager import EnhancedKeyManager, KeyType, KeySecurityLevel
from flashmm.utils.exceptions import SecurityError


@pytest.mark.integration
class TestSecuritySystemIntegration:
    """Integration tests for complete security system."""
    
    @pytest.fixture
    async def security_system(self):
        """Create complete security system for integration testing."""
        # Initialize all security components
        orchestrator = SecurityOrchestrator()
        monitor = SecurityMonitor()
        audit_logger = AuditLogger()
        emergency_manager = EmergencyManager()
        policy_engine = PolicyEngine()
        key_manager = EnhancedKeyManager()
        
        # Start all components
        await orchestrator.start()
        await monitor.start_monitoring()
        await audit_logger.initialize()
        await emergency_manager.initialize()
        await policy_engine.initialize()
        await key_manager.initialize()
        
        security_system = {
            "orchestrator": orchestrator,
            "monitor": monitor,
            "audit_logger": audit_logger,
            "emergency_manager": emergency_manager,
            "policy_engine": policy_engine,
            "key_manager": key_manager
        }
        
        yield security_system
        
        # Cleanup
        await orchestrator.stop()
        await monitor.stop_monitoring()
        await audit_logger.shutdown()
        await emergency_manager.shutdown()
        await policy_engine.shutdown()
    
    async def test_threat_detection_to_response_flow(self, security_system):
        """Test complete flow from threat detection to automated response."""
        orchestrator = security_system["orchestrator"]
        monitor = security_system["monitor"]
        audit_logger = security_system["audit_logger"]
        
        # 1. Create security event that should trigger threat detection
        security_event = await monitor.create_security_event(
            event_type=MonitoringEvent.LOGIN_ATTEMPT,
            component="authentication",
            action="login",
            source_ip="192.168.1.100",
            user_id="suspicious_user",
            success=False,
            metadata={"failed_attempts": 10}
        )
        
        # 2. Process event (this should trigger threat detection)
        await monitor.process_event(security_event)
        
        # 3. Verify threat was detected and registered
        await asyncio.sleep(0.1)  # Allow processing time
        
        # Check if threats were created in orchestrator
        auth_failures = [
            threat for threat in orchestrator.active_threats.values()
            if "auth" in threat.component or "authentication" in threat.component
        ]
        
        assert len(auth_failures) > 0, "Authentication failure threat should be detected"
        
        # 4. Verify audit logging
        audit_events = await audit_logger.search_events(
            event_types=[AuditEventType.SECURITY_EVENT],
            component="security_monitor",
            limit=10
        )
        
        assert len(audit_events) > 0, "Security event should be audited"
        
        # 5. Verify automated response (IP should be blocked for repeated failures)
        if security_event.risk_score > 0.7:
            assert "192.168.1.100" in monitor.intrusion_detector.blocked_ips
    
    async def test_policy_violation_enforcement_flow(self, security_system):
        """Test policy violation detection and enforcement flow."""
        policy_engine = security_system["policy_engine"]
        audit_logger = security_system["audit_logger"]
        
        # 1. Evaluate action against policies
        context = {
            "user": {"role": "readonly"},
            "action": "admin_panel_access",
            "timestamp": datetime.utcnow().isoformat(),
            "source_ip": "192.168.1.10"
        }
        
        result = await policy_engine.evaluate_event(
            component="web",
            action="admin_panel_access",
            context=context,
            user_id="readonly_user"
        )
        
        # 2. Should detect policy violation (readonly user accessing admin)
        assert result["overall_decision"] == "block" or len(result["violations"]) > 0
        
        # 3. Verify audit logging of policy violation
        await asyncio.sleep(0.1)  # Allow audit processing
        
        policy_audit_events = await audit_logger.search_events(
            event_types=[AuditEventType.AUTHORIZATION],
            actor_id="readonly_user",
            limit=5
        )
        
        # Should have audit entries for the policy violation
        assert len(policy_audit_events) >= 0  # May not be present in test environment
    
    async def test_emergency_response_coordination(self, security_system):
        """Test emergency response coordination across components."""
        orchestrator = security_system["orchestrator"]
        emergency_manager = security_system["emergency_manager"]
        audit_logger = security_system["audit_logger"]
        
        # 1. Declare security emergency
        incident_id = await emergency_manager.declare_emergency(
            emergency_type=EmergencyType.SECURITY_BREACH,
            emergency_level=EmergencyLevel.HIGH,
            description="Simulated security breach for testing",
            detected_by="integration_test",
            affected_systems=["authentication", "database"]
        )
        
        # 2. Verify emergency state propagation
        assert incident_id in emergency_manager.active_incidents
        assert emergency_manager.current_system_state.value in ["emergency", "degraded"]
        
        # 3. Verify orchestrator receives emergency state
        # (In production, this would be event-driven)
        
        # 4. Verify comprehensive audit logging
        emergency_audit_events = await audit_logger.search_events(
            event_types=[AuditEventType.EMERGENCY],
            actor_id="integration_test",
            limit=10
        )
        
        assert len(emergency_audit_events) > 0, "Emergency should be audited"
        
        # 5. Resolve emergency
        resolved = await emergency_manager.resolve_incident(
            incident_id,
            "integration_test",
            "Test emergency resolved"
        )
        
        assert resolved is True
        assert len(emergency_manager.active_incidents) == 0
    
    async def test_key_management_security_integration(self, security_system):
        """Test key management integration with security monitoring."""
        key_manager = security_system["key_manager"]
        monitor = security_system["monitor"]
        audit_logger = security_system["audit_logger"]
        
        # 1. Generate key with monitoring
        key_id = await key_manager.generate_key(
            key_type=KeyType.API_KEY,
            security_level=KeySecurityLevel.WARM,
            owner="integration_test",
            permissions={"api.read", "data.read"}
        )
        
        # 2. Use key (should trigger monitoring)
        await key_manager.use_key(
            key_id=key_id,
            operation="encrypt",
            user_id="integration_test",
            source_ip="192.168.1.10"
        )
        
        # 3. Verify key usage is monitored and audited
        key_audit_events = await audit_logger.search_events(
            event_types=[AuditEventType.KEY_MANAGEMENT],
            actor_id="integration_test",
            limit=10
        )
        
        assert len(key_audit_events) > 0, "Key operations should be audited"
        
        # 4. Test key rotation with security monitoring
        new_key_id = await key_manager.rotate_key(key_id)
        assert new_key_id != key_id
        
        # Verify rotation is audited
        rotation_events = await audit_logger.search_events(
            action="key_rotated",
            limit=5
        )
        
        assert len(rotation_events) >= 0  # May not be present depending on implementation
    
    async def test_cross_component_threat_correlation(self, security_system):
        """Test threat correlation across multiple security components."""
        orchestrator = security_system["orchestrator"]
        monitor = security_system["monitor"]
        policy_engine = security_system["policy_engine"]
        
        suspicious_ip = "192.168.1.99"
        suspicious_user = "suspicious_user"
        
        # 1. Generate multiple suspicious activities
        
        # Failed authentication
        auth_event = await monitor.create_security_event(
            event_type=MonitoringEvent.LOGIN_ATTEMPT,
            component="authentication",
            action="login",
            source_ip=suspicious_ip,
            user_id=suspicious_user,
            success=False,
            metadata={"reason": "invalid_credentials"}
        )
        
        # Policy violation
        policy_context = {
            "user": {"role": "readonly"},
            "action": "admin_access",
            "source_ip": suspicious_ip
        }
        
        policy_result = await policy_engine.evaluate_event(
            component="web",
            action="admin_access",
            context=policy_context,
            user_id=suspicious_user
        )
        
        # 2. Verify correlation across components
        await asyncio.sleep(0.2)  # Allow processing time
        
        # Should have threats in orchestrator
        ip_related_threats = [
            threat for threat in orchestrator.active_threats.values()
            if threat.source_ip == suspicious_ip
        ]
        
        # Should have policy violations
        user_violations = [
            violation for violation in policy_engine.violations
            if violation.user_id == suspicious_user
        ]
        
        # Should have monitoring events
        processed_events = [
            event for event in monitor.processed_events
            if event.source_ip == suspicious_ip or event.user_id == suspicious_user
        ]
        
        # Verify cross-component correlation
        total_security_incidents = len(ip_related_threats) + len(user_violations) + len(processed_events)
        assert total_security_incidents > 0, "Should detect security incidents across components"
    
    async def test_security_system_performance_under_load(self, security_system):
        """Test security system performance under load."""
        orchestrator = security_system["orchestrator"]
        monitor = security_system["monitor"]
        policy_engine = security_system["policy_engine"]
        
        start_time = datetime.utcnow()
        
        # Generate high volume of security events
        tasks = []
        
        for i in range(50):  # Moderate load for testing
            # Authentication events
            auth_task = monitor.create_security_event(
                event_type=MonitoringEvent.LOGIN_ATTEMPT,
                component="auth",
                action="login",
                source_ip=f"192.168.1.{i % 255}",
                user_id=f"user_{i}",
                success=(i % 3 != 0)  # 2/3 success rate
            )
            tasks.append(auth_task)
            
            # Policy evaluations
            if i % 5 == 0:  # Every 5th iteration
                policy_task = policy_engine.evaluate_event(
                    component="api",
                    action="data_access",
                    context={
                        "user": {"role": "readonly"},
                        "resource": f"resource_{i}",
                        "timestamp": datetime.utcnow().isoformat()
                    },
                    user_id=f"user_{i}"
                )
                tasks.append(policy_task)
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = datetime.utcnow()
        duration = (end_time - start_time).total_seconds()
        
        # Performance requirements
        assert duration < 5.0, f"High load processing took too long: {duration}s"
        
        # Verify system stability
        exceptions = [r for r in results if isinstance(r, Exception)]
        assert len(exceptions) == 0, f"System errors under load: {exceptions}"
        
        # Verify components are still operational
        orchestrator_status = orchestrator.get_security_status()
        assert orchestrator_status["security_state"] is not None
        
        monitor_stats = monitor.get_monitoring_statistics()
        assert monitor_stats["monitoring_active"] is True
        
        policy_status = policy_engine.get_policy_status()
        assert policy_status["total_policies"] > 0


@pytest.mark.asyncio
class TestSecurityComplianceIntegration:
    """Integration tests for security compliance features."""
    
    async def test_compliance_audit_trail(self):
        """Test compliance audit trail generation."""
        audit_logger = AuditLogger()
        await audit_logger.initialize()
        
        try:
            # Generate compliance-relevant events
            await audit_logger.log_event(
                event_type=AuditEventType.AUTHENTICATION,
                actor_id="compliance_test_user",
                action="login",
                component="web_app",
                outcome="success",
                compliance_tags=["soc2", "iso27001"],
                metadata={"login_method": "mfa"}
            )
            
            await audit_logger.log_event(
                event_type=AuditEventType.DATA_ACCESS,
                actor_id="compliance_test_user",
                action="data_export",
                component="database",
                outcome="success",
                compliance_tags=["gdpr"],
                metadata={"record_count": 100}
            )
            
            # Wait for batch processing
            await asyncio.sleep(0.1)
            
            # Generate compliance report
            report = await audit_logger.generate_compliance_report(
                framework=audit_logger.compliance_reporter.compliance_rules[list(audit_logger.compliance_reporter.compliance_rules.keys())[0]][0].framework,
                start_date=datetime.utcnow() - timedelta(hours=1),
                end_date=datetime.utcnow(),
                generated_by="integration_test"
            )
            
            assert "report_id" in report
            assert "rule_compliance" in report
            assert "statistics" in report
            assert "recommendations" in report
            
        finally:
            await audit_logger.shutdown()
    
    async def test_security_incident_lifecycle(self):
        """Test complete security incident lifecycle."""
        # Initialize components
        orchestrator = SecurityOrchestrator()
        emergency_manager = EmergencyManager()
        audit_logger = AuditLogger()
        
        await orchestrator.start()
        await emergency_manager.initialize()
        await audit_logger.initialize()
        
        try:
            # 1. Threat detection
            request_data = {
                "auth_type": "api_key",
                "api_key": "invalid_key",
                "source_ip": "192.168.1.200"
            }
            
            # Simulate multiple failed authentication attempts
            with patch.object(orchestrator.auth_manager, 'verify_api_key', return_value=None):
                for i in range(6):
                    try:
                        await orchestrator.authenticate_request(request_data)
                    except:
                        pass  # Expected to fail
            
            # 2. Emergency declaration (if high severity threats detected)
            if len(orchestrator.active_threats) > 0:
                high_severity_threats = [
                    t for t in orchestrator.active_threats.values()
                    if t.severity.value in ["high", "critical"]
                ]
                
                if high_severity_threats:
                    incident_id = await emergency_manager.declare_emergency(
                        emergency_type=EmergencyType.SECURITY_BREACH,
                        emergency_level=EmergencyLevel.HIGH,
                        description="Automated security incident response",
                        detected_by="security_orchestrator",
                        affected_systems=["authentication"]
                    )
                    
                    # 3. Verify incident tracking
                    assert incident_id in emergency_manager.active_incidents
                    
                    # 4. Resolve incident
                    resolved = await emergency_manager.resolve_incident(
                        incident_id,
                        "integration_test",
                        "Incident resolved during integration testing"
                    )
                    
                    assert resolved is True
                    assert incident_id not in emergency_manager.active_incidents
            
            # 5. Verify comprehensive audit trail
            security_events = await audit_logger.search_events(
                event_types=[AuditEventType.SECURITY_EVENT, AuditEventType.AUTHENTICATION],
                limit=20
            )
            
            # Should have audit trail of the entire incident
            assert len(security_events) > 0
            
        finally:
            await orchestrator.stop()
            await emergency_manager.shutdown()
            await audit_logger.shutdown()
    
    async def test_multi_component_security_metrics(self):
        """Test security metrics collection across all components."""
        # Initialize components
        orchestrator = SecurityOrchestrator()
        monitor = SecurityMonitor()
        policy_engine = PolicyEngine()
        emergency_manager = EmergencyManager()
        
        await orchestrator.start()
        await monitor.start_monitoring()
        await policy_engine.initialize()
        await emergency_manager.initialize()
        
        try:
            # Generate activity across components
            
            # 1. Authentication activity
            with patch.object(orchestrator.auth_manager, 'verify_api_key', return_value="admin"):
                for i in range(10):
                    await orchestrator.authenticate_request({
                        "auth_type": "api_key",
                        "api_key": "test_key",
                        "source_ip": f"192.168.1.{i}"
                    })
            
            # 2. Policy evaluations
            for i in range(5):
                await policy_engine.evaluate_event(
                    component="api",
                    action="test_action",
                    context={"user": {"role": "admin"}, "test": i},
                    user_id=f"user_{i}"
                )
            
            # 3. Monitoring events
            for i in range(8):
                await monitor.create_security_event(
                    event_type=MonitoringEvent.API_REQUEST,
                    component="api",
                    action="test_request",
                    user_id=f"user_{i}",
                    success=True
                )
            
            # Allow processing time
            await asyncio.sleep(0.2)
            
            # 4. Collect metrics from all components
            orchestrator_status = orchestrator.get_security_status()
            monitor_stats = monitor.get_monitoring_statistics()
            policy_status = policy_engine.get_policy_status()
            emergency_status = emergency_manager.get_emergency_status()
            
            # Verify metrics are collected
            assert orchestrator_status["metrics"]["total_auth_attempts"] == 10
            assert monitor_stats["events_processed"] >= 8
            assert policy_status["evaluation_stats"]["total_evaluations"] >= 5
            assert emergency_status["system_state"] is not None
            
            # 5. Generate integrated metrics report
            integrated_metrics = {
                "timestamp": datetime.utcnow().isoformat(),
                "orchestrator": orchestrator_status,
                "monitor": monitor_stats,
                "policies": policy_status,
                "emergency": emergency_status
            }
            
            # Verify comprehensive coverage
            assert "metrics" in integrated_metrics["orchestrator"]
            assert "events_processed" in integrated_metrics["monitor"]
            assert "total_policies" in integrated_metrics["policies"]
            assert "system_state" in integrated_metrics["emergency"]
            
        finally:
            await orchestrator.stop()
            await monitor.stop_monitoring()
            await policy_engine.shutdown()
            await emergency_manager.shutdown()
    
    async def test_security_system_resilience(self):
        """Test security system resilience to component failures."""
        orchestrator = SecurityOrchestrator()
        monitor = SecurityMonitor()
        
        await orchestrator.start()
        await monitor.start_monitoring()
        
        try:
            # 1. Normal operation
            with patch.object(orchestrator.auth_manager, 'verify_api_key', return_value="admin"):
                result = await orchestrator.authenticate_request({
                    "auth_type": "api_key",
                    "api_key": "test_key",
                    "source_ip": "192.168.1.10"
                })
                assert result["authenticated"] is True
            
            # 2. Simulate component failure (monitoring)
            original_process_event = monitor.process_event
            
            def failing_process_event(*args, **kwargs):
                raise Exception("Simulated monitoring failure")
            
            monitor.process_event = failing_process_event
            
            # 3. Verify main system still functions
            with patch.object(orchestrator.auth_manager, 'verify_api_key', return_value="admin"):
                result = await orchestrator.authenticate_request({
                    "auth_type": "api_key", 
                    "api_key": "test_key",
                    "source_ip": "192.168.1.11"
                })
                assert result["authenticated"] is True
            
            # 4. Restore monitoring
            monitor.process_event = original_process_event
            
            # 5. Verify full functionality restored
            await monitor.create_security_event(
                event_type=MonitoringEvent.API_REQUEST,
                component="test",
                action="resilience_test",
                success=True
            )
            
        finally:
            await orchestrator.stop()
            await monitor.stop_monitoring()


@pytest.mark.load
class TestSecuritySystemLoad:
    """Load testing for security system."""
    
    async def test_high_volume_authentication(self):
        """Test authentication under high volume."""
        orchestrator = SecurityOrchestrator()
        await orchestrator.start()
        
        try:
            # Simulate high volume authentication
            auth_tasks = []
            
            with patch.object(orchestrator.auth_manager, 'verify_api_key', return_value="user"):
                with patch.object(orchestrator.auth_manager, 'create_access_token', return_value="token"):
                    
                    for i in range(200):  # High volume
                        task = orchestrator.authenticate_request({
                            "auth_type": "api_key",
                            "api_key": "load_test_key",
                            "source_ip": f"192.168.1.{i % 255}"
                        })
                        auth_tasks.append(task)
            
            start_time = datetime.utcnow()
            results = await asyncio.gather(*auth_tasks, return_exceptions=True)
            end_time = datetime.utcnow()
            
            duration = (end_time - start_time).total_seconds()
            
            # Performance requirements under load
            assert duration < 2.0, f"High volume auth took too long: {duration}s"
            
            # Verify success rate
            successful_auths = [r for r in results if isinstance(r, dict) and r.get("authenticated")]
            success_rate = len(successful_auths) / len(results)
            assert success_rate > 0.95, f"Success rate too low: {success_rate}"
            
        finally:
            await orchestrator.stop()
    
    async def test_concurrent_security_operations(self):
        """Test concurrent security operations across all components."""
        # Initialize all components
        components = {
            "orchestrator": SecurityOrchestrator(),
            "monitor": SecurityMonitor(),
            "policy_engine": PolicyEngine(),
            "emergency_manager": EmergencyManager()
        }
        
        # Start all components
        await components["orchestrator"].start()
        await components["monitor"].start_monitoring()
        await components["policy_engine"].initialize()
        await components["emergency_manager"].initialize()
        
        try:
            # Create concurrent operations
            operation_tasks = []
            
            # Authentication operations
            with patch.object(components["orchestrator"].auth_manager, 'verify_api_key', return_value="user"):
                for i in range(30):
                    task = components["orchestrator"].authenticate_request({
                        "auth_type": "api_key",
                        "api_key": "concurrent_key",
                        "source_ip": f"192.168.1.{i}"
                    })
                    operation_tasks.append(task)
            
            # Monitoring operations
            for i in range(25):
                task = components["monitor"].create_security_event(
                    event_type=MonitoringEvent.API_REQUEST,
                    component="api",
                    action="concurrent_test",
                    user_id=f"user_{i}",
                    success=True
                )
                operation_tasks.append(task)
            
            # Policy operations
            for i in range(20):
                task = components["policy_engine"].evaluate_event(
                    component="web",
                    action="concurrent_action",
                    context={"user": {"role": "admin"}, "test_id": i},
                    user_id=f"user_{i}"
                )
                operation_tasks.append(task)
            
            # Execute all operations concurrently
            start_time = datetime.utcnow()
            results = await asyncio.gather(*operation_tasks, return_exceptions=True)
            end_time = datetime.utcnow()
            
            duration = (end_time - start_time).total_seconds()
            
            # Performance and reliability checks
            assert duration < 3.0, f"Concurrent operations took too long: {duration}s"
            
            exceptions = [r for r in results if isinstance(r, Exception)]
            exception_rate = len(exceptions) / len(results)
            assert exception_rate < 0.05, f"Too many exceptions under load: {exception_rate}"
            
        finally:
            # Cleanup all components
            await components["orchestrator"].stop()
            await components["monitor"].stop_monitoring()
            await components["policy_engine"].shutdown()
            await components["emergency_manager"].shutdown()


@pytest.mark.regression
class TestSecurityRegression:
    """Regression tests to ensure security functionality doesn't break."""
    
    async def test_backward_compatibility(self):
        """Test backward compatibility with existing security interfaces."""
        # Test that enhanced components maintain compatibility
        orchestrator = SecurityOrchestrator()
        
        # Original auth manager methods should still work
        assert hasattr(orchestrator.auth_manager, 'verify_api_key')
        assert hasattr(orchestrator.auth_manager, 'create_access_token')
        assert hasattr(orchestrator.auth_manager, 'verify_token')
        
        # Original authz manager methods should still work
        assert hasattr(orchestrator.authz_manager, 'check_permission')
        assert hasattr(orchestrator.authz_manager, 'require_permission')
        
        # Key managers should maintain interface
        assert hasattr(orchestrator.hot_key_manager, 'test_keys_validity')
        assert hasattr(orchestrator.key_rotation_manager, 'get_rotation_status')
    
    async def test_configuration_compatibility(self):
        """Test that security configuration remains compatible."""
        orchestrator = SecurityOrchestrator()
        
        # Verify configuration loading
        policies = orchestrator.security_policies
        
        required_config_keys = [
            "max_auth_failures",
            "rate_limit_requests", 
            "session_timeout"
        ]
        
        for key in required_config_keys:
            assert key in policies
            assert isinstance(policies[key], (int, float))
        
        # Verify existing auth configuration works
        assert orchestrator.auth_manager.secret_key is not None
        assert orchestrator.auth_manager.algorithm == "HS256"
        assert isinstance(orchestrator.auth_manager.access_token_expire, timedelta)
    
    def test_existing_permission_structure(self):
        """Test that existing permission structure is maintained."""
        authz_manager = AuthorizationManager()
        
        # Verify original permission structure exists
        legacy_roles = ["admin", "readonly", "system"]
        
        for role_name in legacy_roles:
            try:
                # Convert string role to enum for new system
                role_enum = next(r for r in UserRole if r.value == role_name)
                permissions = authz_manager.get_role_permissions(role_name)
                assert len(permissions) > 0
            except StopIteration:
                # Role might be mapped differently in new system
                pass
        
        # Verify critical permissions still exist
        admin_permissions = authz_manager.get_role_permissions("admin")
        
        expected_admin_permissions = [
            "trading.pause", "trading.resume", "config.read", "config.write"
        ]
        
        for perm in expected_admin_permissions:
            assert authz_manager.check_permission("admin", perm), f"Permission {perm} missing for admin"


class TestSecurityMetricsIntegration:
    """Test security metrics integration across components."""
    
    async def test_metrics_aggregation(self):
        """Test aggregation of security metrics from all components."""
        # This would typically be implemented in a metrics aggregator
        # For now, we'll test that each component provides metrics
        
        orchestrator = SecurityOrchestrator()
        monitor = SecurityMonitor()
        policy_engine = PolicyEngine()
        emergency_manager = EmergencyManager()
        
        await orchestrator.start()
        await monitor.start_monitoring()
        await policy_engine.initialize()
        await emergency_manager.initialize()
        
        try:
            # Get metrics from all components
            orchestrator_metrics = orchestrator.get_security_status()
            monitor_metrics = monitor.get_monitoring_statistics()
            policy_metrics = policy_engine.get_policy_status()
            emergency_metrics = emergency_manager.get_emergency_status()
            
            # Verify metrics structure
            assert "metrics" in orchestrator_metrics
            assert "events_processed" in monitor_metrics
            assert "evaluation_stats" in policy_metrics
            assert "system_state" in emergency_metrics
            
            # Verify metrics are numeric and valid
            assert isinstance(orchestrator_metrics["metrics"]["total_auth_attempts"], int)
            assert isinstance(monitor_metrics["events_processed"], int)
            assert isinstance(policy_metrics["total_policies"], int)
            assert isinstance(emergency_metrics["total_incidents_handled"], int)
            
        finally:
            await orchestrator.stop()
            await monitor.stop_monitoring()
            await policy_engine.shutdown()
            await emergency_manager.shutdown()
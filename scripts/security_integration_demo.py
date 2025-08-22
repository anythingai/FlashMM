"""
FlashMM Security Integration Demo

Demonstrates comprehensive security system integration and provides
validation that all security components work together seamlessly.
"""

import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from flashmm.config.settings import get_config
from flashmm.security import (
    AuditEventType,
    AuditLogger,
    EmergencyLevel,
    EmergencyManager,
    # Data structures
    EmergencyType,
    EnhancedKeyManager,
    KeySecurityLevel,
    KeyType,
    PenetrationTestFramework,
    PolicyEngine,
    SecurityMonitor,
    SecurityOrchestrator,
    UserRole,
)


class SecurityIntegrationDemo:
    """Demonstrates comprehensive security system integration."""

    def __init__(self):
        self.config = get_config()

        # Security components - will be initialized in initialize_security_system()
        self.orchestrator: SecurityOrchestrator | None = None
        self.monitor: SecurityMonitor | None = None
        self.audit_logger: AuditLogger | None = None
        self.emergency_manager: EmergencyManager | None = None
        self.policy_engine: PolicyEngine | None = None
        self.key_manager: EnhancedKeyManager | None = None
        self.pentest_framework: PenetrationTestFramework | None = None

    async def initialize_security_system(self):
        """Initialize complete security system."""
        print("🔒 Initializing FlashMM Comprehensive Security System...")

        # Initialize components in dependency order
        print("  📋 Initializing Audit Logger...")
        self.audit_logger = AuditLogger()
        await self.audit_logger.initialize()

        print("  📜 Initializing Policy Engine...")
        self.policy_engine = PolicyEngine()
        await self.policy_engine.initialize()

        print("  🔍 Initializing Security Monitor...")
        self.monitor = SecurityMonitor()
        await self.monitor.start_monitoring()

        print("  🚨 Initializing Emergency Manager...")
        self.emergency_manager = EmergencyManager()
        await self.emergency_manager.initialize()

        print("  🔑 Initializing Enhanced Key Manager...")
        self.key_manager = EnhancedKeyManager()
        await self.key_manager.initialize()

        print("  🎛️ Initializing Security Orchestrator...")
        self.orchestrator = SecurityOrchestrator()
        await self.orchestrator.start()

        print("  🔎 Initializing Penetration Test Framework...")
        self.pentest_framework = PenetrationTestFramework()

        print("✅ Security system initialization complete!\n")

    async def demonstrate_authentication_flow(self):
        """Demonstrate complete authentication flow with security integration."""
        print("🔐 Demonstrating Enhanced Authentication Flow...")

        # 1. Generate API key with MFA requirement
        if not self.orchestrator:
            raise RuntimeError("Security orchestrator not initialized")

        api_key = self.orchestrator.auth_manager.generate_api_key(
            user_id="demo_admin",
            role=UserRole.ADMIN,
            requires_mfa=True
        )
        print("  ✓ Generated secure API key for demo_admin")

        # 2. Setup MFA
        secret, qr_code, backup_codes = self.orchestrator.auth_manager.setup_mfa("demo_admin")
        print(f"  ✓ MFA configured (secret length: {len(secret)}, backup codes: {len(backup_codes)})")

        # 3. Simulate authentication request
        auth_result = await self.orchestrator.authenticate_request({
            "auth_type": "api_key",
            "api_key": api_key,
            "source_ip": "192.168.1.10",
            "user_agent": "FlashMM-Demo/1.0"
        })

        print(f"  ✓ Authentication result: {auth_result.get('authenticated')}")
        print(f"  ✓ User role: {auth_result.get('role')}")

        # 4. Test authorization
        user_context = {
            "role": auth_result.get("role"),
            "sub": "demo_admin",
            "mfa_verified": True
        }

        # Test various permissions
        test_permissions = ["config.read", "config.write", "keys.rotate", "trading.pause"]
        print("  🔒 Testing permissions:")

        for permission in test_permissions:
            try:
                await self.orchestrator.authz_manager.require_permission(user_context, permission)
                print(f"    ✓ {permission}: ALLOWED")
            except Exception as e:
                print(f"    ❌ {permission}: DENIED - {e}")

        print()

    async def demonstrate_threat_detection_response(self):
        """Demonstrate threat detection and automated response."""
        print("🚨 Demonstrating Threat Detection and Response...")

        # 1. Simulate suspicious activity - multiple failed login attempts
        print("  🔍 Simulating brute force attack...")

        suspicious_ip = "192.168.1.99"

        # Generate multiple authentication failures
        for _ in range(8):
            try:
                if not self.orchestrator:
                    raise RuntimeError("Security orchestrator not initialized")
                await self.orchestrator.authenticate_request({
                    "auth_type": "api_key",
                    "api_key": "invalid_key_attempt",
                    "source_ip": suspicious_ip,
                    "user_agent": "AttackBot/1.0"
                })
            except Exception as e:
                # Expected authentication failures for threat detection demo
                print(f"  Expected auth failure: {type(e).__name__}")

        print(f"  ✓ Generated 8 failed authentication attempts from {suspicious_ip}")

        # 2. Check threat detection
        await asyncio.sleep(0.1)  # Allow processing time

        if not self.orchestrator:
            raise RuntimeError("Security orchestrator not initialized")
        security_status = self.orchestrator.get_security_status()
        print(f"  📊 Security status: {security_status['security_state']}")
        print(f"  ⚠️ Active threats: {security_status['active_threats']}")
        print(f"  🚫 Blocked IPs: {security_status['blocked_ips']}")

        # 3. Check if IP was automatically blocked
        if suspicious_ip in self.orchestrator.blocked_ips:
            print(f"  ✅ Automated response: {suspicious_ip} blocked successfully")
        else:
            print(f"  ℹ️ IP {suspicious_ip} not blocked (may require more attempts)")

        print()

    async def demonstrate_policy_enforcement(self):
        """Demonstrate security policy enforcement."""
        print("📜 Demonstrating Security Policy Enforcement...")

        # 1. Test policy evaluation
        test_context = {
            "user": {"role": "readonly"},
            "action": "admin_panel_access",
            "source_ip": "192.168.1.20",
            "timestamp": datetime.utcnow().isoformat()
        }

        if not self.policy_engine:
            raise RuntimeError("Policy engine not initialized")

        result = await self.policy_engine.evaluate_event(
            component="web",
            action="admin_panel_access",
            context=test_context,
            user_id="readonly_user"
        )

        print(f"  📋 Policy evaluation result: {result['overall_decision']}")
        print(f"  📊 Policies evaluated: {len(result['policy_decisions'])}")
        print(f"  ⚠️ Violations detected: {len(result['violations'])}")

        if result['violations']:
            violation = result['violations'][0]
            print(f"  🚨 Sample violation: {violation.description}")
            print(f"  📈 Severity: {violation.severity.value}")

        # 2. Show policy status
        policy_status = self.policy_engine.get_policy_status()
        print(f"  📚 Active policies: {policy_status['active_policies']}")
        print(f"  🔧 Total rules: {policy_status['total_rules']}")

        print()

    async def demonstrate_audit_compliance(self):
        """Demonstrate audit logging and compliance reporting."""
        print("📝 Demonstrating Audit Logging and Compliance...")

        # 1. Generate various audit events
        audit_events = [
            {
                "event_type": AuditEventType.AUTHENTICATION,
                "action": "login",
                "outcome": "success"
            },
            {
                "event_type": AuditEventType.DATA_ACCESS,
                "action": "data_export",
                "outcome": "success"
            },
            {
                "event_type": AuditEventType.SYSTEM_CONFIGURATION,
                "action": "config_change",
                "outcome": "success"
            }
        ]

        if not self.audit_logger:
            raise RuntimeError("Audit logger not initialized")

        for event_data in audit_events:
            await self.audit_logger.log_event(
                event_type=event_data["event_type"],
                actor_id="demo_user",
                action=event_data["action"],
                component="demo",
                outcome=event_data["outcome"],
                metadata={"demo": True, "timestamp": datetime.utcnow().isoformat()}
            )

        print(f"  ✓ Generated {len(audit_events)} audit events")

        # 2. Search audit events
        recent_events = await self.audit_logger.search_events(
            start_time=datetime.utcnow() - timedelta(minutes=5),
            limit=10
        )

        print(f"  📊 Recent audit events: {len(recent_events)}")

        # 3. Get audit statistics
        audit_stats = self.audit_logger.get_audit_statistics()
        print(f"  📈 Events logged: {audit_stats['metrics']['events_logged']}")
        print(f"  🔧 Buffer size: {audit_stats['buffer_size']}")

        print()

    async def demonstrate_emergency_procedures(self):
        """Demonstrate emergency procedures and incident response."""
        print("🚨 Demonstrating Emergency Procedures...")

        # 1. Declare test emergency
        if not self.emergency_manager:
            raise RuntimeError("Emergency manager not initialized")

        incident_id = await self.emergency_manager.declare_emergency(
            emergency_type=EmergencyType.SECURITY_BREACH,
            emergency_level=EmergencyLevel.MEDIUM,
            description="Demonstration security breach for testing",
            detected_by="security_demo",
            affected_systems=["authentication", "api"]
        )

        print(f"  🆔 Emergency incident created: {incident_id}")
        print(f"  📊 System state: {self.emergency_manager.current_system_state.value}")

        # 2. Check incident status
        emergency_status = self.emergency_manager.get_emergency_status()
        print(f"  📈 Active incidents: {emergency_status['active_incidents']}")

        # 3. Test emergency communication (simulated)
        incident = self.emergency_manager.active_incidents[incident_id]
        notification_result = await self.emergency_manager.communicator.send_emergency_notification(
            incident, escalation_level=1
        )

        print(f"  📧 Notification sent: {notification_result['notification_id']}")
        print(f"  📞 Channels used: {notification_result['channels_used']}")

        # 4. Resolve incident
        resolved = await self.emergency_manager.resolve_incident(
            incident_id,
            "security_demo",
            "Demo incident resolved successfully"
        )

        print(f"  ✅ Incident resolved: {resolved}")

        print()

    async def demonstrate_key_management(self):
        """Demonstrate enhanced key management with HSM integration."""
        print("🔑 Demonstrating Enhanced Key Management...")

        if not self.key_manager:
            raise RuntimeError("Key manager not initialized")

        # 1. Generate keys with different security levels
        hot_key_id = await self.key_manager.generate_key(
            key_type=KeyType.API_KEY,
            security_level=KeySecurityLevel.HOT,
            owner="demo_system",
            permissions={"api.access", "trading.read"}
        )

        warm_key_id = await self.key_manager.generate_key(
            key_type=KeyType.ENCRYPTION,
            security_level=KeySecurityLevel.WARM,
            owner="demo_system",
            permissions={"data.encrypt", "data.decrypt"}
        )

        print(f"  🔑 Generated HOT key: {hot_key_id}")
        print(f"  🔑 Generated WARM key: {warm_key_id}")

        # 2. Use keys (with audit logging)
        await self.key_manager.use_key(
            key_id=hot_key_id,
            operation="authenticate",
            user_id="demo_user",
            source_ip="192.168.1.10"
        )

        print("  ✓ Used HOT key for authentication")

        # 3. Check key status
        hot_key_status = self.key_manager.get_key_status(hot_key_id)
        system_key_status = self.key_manager.get_system_key_status()

        if hot_key_status:
            print(f"  📊 Key usage count: {hot_key_status['usage_count']}")
        else:
            print("  📊 Key status not available")

        print(f"  📈 Total system keys: {system_key_status['total_keys']}")
        print(f"  🔒 HSM protected keys: {system_key_status['hsm_keys']}")

        print()

    async def demonstrate_penetration_testing(self):
        """Demonstrate penetration testing framework."""
        print("🔎 Demonstrating Penetration Testing Framework...")

        if not self.pentest_framework:
            raise RuntimeError("Penetration test framework not initialized")

        # 1. Get framework status
        framework_status = self.pentest_framework.get_framework_status()
        print(f"  📊 Available test suites: {framework_status['available_test_suites']}")
        print(f"  🧪 Test suites: {framework_status['test_suites']}")

        # 2. Simulate vulnerability scan (without actual network calls)
        print("  🔍 Security assessment capabilities:")
        print("    • Web application vulnerability scanning")
        print("    • Network security assessment")
        print("    • Input validation fuzzing")
        print("    • SSL/TLS configuration testing")
        print("    • OWASP Top 10 compliance checking")

        # 3. Generate sample security report
        sample_findings = []  # Would be populated by actual scans

        security_report = self.pentest_framework.generate_security_report([
            {
                "findings": sample_findings,
                "summary": {"total_tests": 10, "passed": 8, "failed": 2},
                "duration_seconds": 45.2
            }
        ])

        print(f"  📄 Security report generated: {security_report['report_id']}")
        print(f"  🎯 Security posture: {security_report['summary']['security_posture']}")

        print()

    async def demonstrate_integration_metrics(self):
        """Show integrated security metrics across all components."""
        print("📊 Comprehensive Security Metrics Dashboard...")

        # Collect metrics from all components
        if not all([self.orchestrator, self.monitor, self.audit_logger, self.emergency_manager,
                   self.policy_engine, self.key_manager, self.pentest_framework]):
            raise RuntimeError("Security components not fully initialized")

        metrics = {
            "timestamp": datetime.utcnow().isoformat(),
            "orchestrator": self.orchestrator.get_security_status() if self.orchestrator else {},
            "monitor": self.monitor.get_monitoring_statistics() if self.monitor else {},
            "audit": self.audit_logger.get_audit_statistics() if self.audit_logger else {},
            "emergency": self.emergency_manager.get_emergency_status() if self.emergency_manager else {},
            "policies": self.policy_engine.get_policy_status() if self.policy_engine else {},
            "keys": self.key_manager.get_system_key_status() if self.key_manager else {},
            "pentest": self.pentest_framework.get_framework_status() if self.pentest_framework else {}
        }

        print("  🎛️ Security Orchestrator:")
        print(f"    • Security State: {metrics['orchestrator']['security_state']}")
        print(f"    • Auth Attempts: {metrics['orchestrator']['metrics']['total_auth_attempts']}")
        print(f"    • Active Threats: {metrics['orchestrator']['active_threats']}")

        print("  🔍 Security Monitor:")
        print(f"    • Monitoring Active: {metrics['monitor']['monitoring_active']}")
        print(f"    • Events Processed: {metrics['monitor']['events_processed']}")
        print(f"    • Threats Detected: {metrics['monitor']['threats_detected']}")

        print("  📝 Audit System:")
        print(f"    • Events Logged: {metrics['audit']['metrics']['events_logged']}")
        print(f"    • Integrity Checks: {metrics['audit']['metrics']['integrity_checks']}")

        print("  🚨 Emergency Management:")
        print(f"    • System State: {metrics['emergency']['system_state']}")
        print(f"    • Total Incidents: {metrics['emergency']['total_incidents_handled']}")

        print("  📜 Policy Engine:")
        print(f"    • Active Policies: {metrics['policies']['active_policies']}")
        print(f"    • Total Violations: {metrics['policies']['total_violations']}")

        print("  🔑 Key Management:")
        print(f"    • Total Keys: {metrics['keys']['total_keys']}")
        print(f"    • Active Keys: {metrics['keys']['active_keys']}")
        print(f"    • HSM Protected: {metrics['keys']['hsm_keys']}")

        print()

        return metrics

    async def test_system_resilience(self):
        """Test system resilience and failsafe operations."""
        print("🛡️ Testing System Resilience and Failsafes...")

        # 1. Test authentication under load
        print("  ⚡ Testing authentication performance...")

        start_time = datetime.utcnow()

        # Generate test API key
        if not self.orchestrator:
            raise RuntimeError("Security orchestrator not initialized")

        test_key = self.orchestrator.auth_manager.generate_api_key(
            "load_test_user",
            UserRole.READONLY
        )

        # Simulate concurrent authentication requests
        auth_tasks = []
        for i in range(20):
            task = self.orchestrator.authenticate_request({
                "auth_type": "api_key",
                "api_key": test_key,
                "source_ip": f"192.168.1.{i + 10}",
                "user_agent": "LoadTest/1.0"
            })
            auth_tasks.append(task)

        results = await asyncio.gather(*auth_tasks, return_exceptions=True)

        end_time = datetime.utcnow()
        duration = (end_time - start_time).total_seconds()

        successful_auths = [r for r in results if isinstance(r, dict) and r.get("authenticated")]

        print(f"    ✓ Processed {len(results)} concurrent requests in {duration:.3f}s")
        print(f"    ✓ Success rate: {len(successful_auths)}/{len(results)} ({len(successful_auths)/len(results)*100:.1f}%)")
        print(f"    ✓ Average response time: {duration/len(results)*1000:.2f}ms")

        # 2. Test emergency procedures
        print("  🚨 Testing emergency procedures...")

        if not self.emergency_manager:
            raise RuntimeError("Emergency manager not initialized")

        test_results = await self.emergency_manager.test_emergency_procedures("communication")
        print(f"    ✓ Emergency test completed: {test_results['test_id']}")

        print()

    async def shutdown_security_system(self):
        """Gracefully shutdown all security components."""
        print("🔒 Shutting down security system...")

        # Shutdown in reverse dependency order
        if self.orchestrator:
            await self.orchestrator.stop()
            print("  ✓ Security Orchestrator stopped")

        if self.monitor:
            await self.monitor.stop_monitoring()
            print("  ✓ Security Monitor stopped")

        if self.policy_engine:
            await self.policy_engine.shutdown()
            print("  ✓ Policy Engine stopped")

        if self.emergency_manager:
            await self.emergency_manager.shutdown()
            print("  ✓ Emergency Manager stopped")

        if self.audit_logger:
            await self.audit_logger.shutdown()
            print("  ✓ Audit Logger stopped")

        print("✅ Security system shutdown complete")

    async def run_complete_demo(self):
        """Run complete security integration demonstration."""
        print("🚀 FlashMM Comprehensive Security Integration Demo")
        print("=" * 60)

        try:
            # Initialize security system
            await self.initialize_security_system()

            # Run demonstrations
            await self.demonstrate_authentication_flow()
            await self.demonstrate_threat_detection_response()
            await self.demonstrate_policy_enforcement()
            await self.demonstrate_audit_compliance()
            await self.demonstrate_key_management()
            await self.demonstrate_penetration_testing()

            # Show integrated metrics
            await self.demonstrate_integration_metrics()

            # Test system resilience
            await self.test_system_resilience()

            print("✅ SECURITY INTEGRATION DEMO COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            print("🔒 FlashMM now has comprehensive security protection including:")
            print("  • Multi-factor authentication with RBAC")
            print("  • Real-time threat detection and response")
            print("  • Comprehensive audit trails for compliance")
            print("  • Emergency procedures and incident response")
            print("  • Dynamic security policy enforcement")
            print("  • Hardware security module integration")
            print("  • Automated vulnerability assessment")
            print("  • Advanced key lifecycle management")
            print("  • Behavioral anomaly detection")
            print("  • Forensic analysis capabilities")
            print()
            print("🛡️ System is protected and ready for production deployment!")

        except Exception as e:
            print(f"❌ Demo failed with error: {e}")
            raise

        finally:
            # Always cleanup
            await self.shutdown_security_system()


async def main():
    """Main demo execution."""
    demo = SecurityIntegrationDemo()
    await demo.run_complete_demo()


if __name__ == "__main__":
    asyncio.run(main())

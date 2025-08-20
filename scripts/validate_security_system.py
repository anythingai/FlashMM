"""
FlashMM Security System Validation Script

Validates that all security components are properly integrated and functional.
Used for deployment validation and system health checks.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from flashmm.security import (
    SecurityOrchestrator, SecurityMonitor, AuditLogger, EmergencyManager,
    PolicyEngine, EnhancedKeyManager, PenetrationTestFramework
)
from flashmm.utils.logging import get_logger

logger = get_logger(__name__)


async def validate_security_components():
    """Validate all security components can be initialized and are functional."""
    
    print("🔒 FlashMM Security System Validation")
    print("=" * 50)
    
    validation_results = {
        "security_orchestrator": False,
        "security_monitor": False, 
        "audit_logger": False,
        "emergency_manager": False,
        "policy_engine": False,
        "key_manager": False,
        "pentest_framework": False
    }
    
    # Test Security Orchestrator
    print("🎛️ Validating Security Orchestrator...")
    try:
        orchestrator = SecurityOrchestrator()
        await orchestrator.start()
        
        # Test basic functionality
        status = orchestrator.get_security_status()
        assert "security_state" in status
        assert "metrics" in status
        
        await orchestrator.stop()
        validation_results["security_orchestrator"] = True
        print("  ✅ Security Orchestrator: PASS")
    except Exception as e:
        print(f"  ❌ Security Orchestrator: FAIL - {e}")
    
    # Test Security Monitor
    print("🔍 Validating Security Monitor...")
    try:
        monitor = SecurityMonitor()
        await monitor.start_monitoring()
        
        # Test event creation
        await monitor.create_security_event(
            event_type=MonitoringEvent.API_REQUEST,
            component="validation",
            action="test",
            success=True
        )
        
        stats = monitor.get_monitoring_statistics()
        assert "events_processed" in stats
        
        await monitor.stop_monitoring()
        validation_results["security_monitor"] = True
        print("  ✅ Security Monitor: PASS")
    except Exception as e:
        print(f"  ❌ Security Monitor: FAIL - {e}")
    
    # Test Audit Logger
    print("📝 Validating Audit Logger...")
    try:
        audit_logger = AuditLogger()
        await audit_logger.initialize()
        
        # Test event logging
        event_id = await audit_logger.log_event(
            event_type=AuditEventType.SYSTEM_CONFIGURATION,
            actor_id="validation_test",
            action="system_validation",
            component="validation_script",
            outcome="success"
        )
        
        assert event_id is not None
        
        stats = audit_logger.get_audit_statistics()
        assert "metrics" in stats
        
        await audit_logger.shutdown()
        validation_results["audit_logger"] = True
        print("  ✅ Audit Logger: PASS")
    except Exception as e:
        print(f"  ❌ Audit Logger: FAIL - {e}")
    
    # Test Emergency Manager
    print("🚨 Validating Emergency Manager...")
    try:
        emergency_manager = EmergencyManager()
        await emergency_manager.initialize()
        
        # Test emergency procedures
        test_results = await emergency_manager.test_emergency_procedures("communication")
        assert "test_id" in test_results
        
        status = emergency_manager.get_emergency_status()
        assert "system_state" in status
        
        await emergency_manager.shutdown()
        validation_results["emergency_manager"] = True
        print("  ✅ Emergency Manager: PASS")
    except Exception as e:
        print(f"  ❌ Emergency Manager: FAIL - {e}")
    
    # Test Policy Engine
    print("📜 Validating Policy Engine...")
    try:
        policy_engine = PolicyEngine()
        await policy_engine.initialize()
        
        # Test policy evaluation
        result = await policy_engine.evaluate_event(
            component="validation",
            action="test_action",
            context={"test": True},
            user_id="validation_user"
        )
        
        assert "overall_decision" in result
        
        status = policy_engine.get_policy_status()
        assert "total_policies" in status
        
        await policy_engine.shutdown()
        validation_results["policy_engine"] = True
        print("  ✅ Policy Engine: PASS")
    except Exception as e:
        print(f"  ❌ Policy Engine: FAIL - {e}")
    
    # Test Enhanced Key Manager  
    print("🔑 Validating Enhanced Key Manager...")
    try:
        key_manager = EnhancedKeyManager()
        await key_manager.initialize()
        
        # Test key generation
        key_id = await key_manager.generate_key(
            key_type=KeyType.SESSION_KEY,
            security_level=KeySecurityLevel.WARM,
            owner="validation_test"
        )
        
        assert key_id is not None
        
        status = key_manager.get_system_key_status()
        assert "total_keys" in status
        
        validation_results["key_manager"] = True
        print("  ✅ Enhanced Key Manager: PASS")
    except Exception as e:
        print(f"  ❌ Enhanced Key Manager: FAIL - {e}")
    
    # Test Penetration Test Framework
    print("🔎 Validating Penetration Test Framework...")
    try:
        pentest_framework = PenetrationTestFramework()
        
        status = pentest_framework.get_framework_status()
        assert "available_test_suites" in status
        assert len(pentest_framework.test_suites) > 0
        
        validation_results["pentest_framework"] = True
        print("  ✅ Penetration Test Framework: PASS")
    except Exception as e:
        print(f"  ❌ Penetration Test Framework: FAIL - {e}")
    
    # Summary
    print("\n📊 Validation Summary:")
    print("=" * 30)
    
    passed = sum(validation_results.values())
    total = len(validation_results)
    
    for component, result in validation_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {component.replace('_', ' ').title()}: {status}")
    
    print(f"\nOverall Result: {passed}/{total} components validated successfully")
    
    if passed == total:
        print("🎉 ALL SECURITY COMPONENTS VALIDATED SUCCESSFULLY!")
        print("🛡️ FlashMM security system is fully operational and ready for deployment.")
        return True
    else:
        print("⚠️ Some security components failed validation.")
        print("🔧 Please review and fix the failing components before deployment.")
        return False


if __name__ == "__main__":
    success = asyncio.run(validate_security_components())
    sys.exit(0 if success else 1)
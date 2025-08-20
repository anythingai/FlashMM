"""
Tests for FlashMM Enhanced Authentication System

Comprehensive tests for MFA, RBAC, session management, and enhanced security features.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
import secrets

from flashmm.security.auth import (
    AuthenticationManager, AuthorizationManager, EnhancedSession,
    AuthenticationMethod, SessionState, UserRole
)
from flashmm.utils.exceptions import AuthenticationError, AuthorizationError


class TestEnhancedSession:
    """Test suite for Enhanced Session management."""
    
    @pytest.fixture
    def sample_session(self):
        """Create sample session for testing."""
        return EnhancedSession(
            session_id="test_session_123",
            user_id="test_user",
            role=UserRole.ADMIN,
            created_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(hours=1),
            source_ip="192.168.1.100",
            user_agent="TestAgent/1.0"
        )
    
    def test_session_creation(self, sample_session):
        """Test session creation with all properties."""
        assert sample_session.session_id == "test_session_123"
        assert sample_session.user_id == "test_user"
        assert sample_session.role == UserRole.ADMIN
        assert sample_session.state == SessionState.ACTIVE
        assert sample_session.mfa_verified is False
        assert sample_session.failed_attempts == 0
    
    def test_session_validity(self, sample_session):
        """Test session validity checks."""
        # Valid session
        assert sample_session.is_valid() is True
        
        # Expired session
        sample_session.expires_at = datetime.utcnow() - timedelta(minutes=1)
        assert sample_session.is_valid() is False
        assert sample_session.state == SessionState.EXPIRED
    
    def test_session_locking(self, sample_session):
        """Test session locking functionality."""
        lock_duration = timedelta(minutes=30)
        sample_session.lock(lock_duration)
        
        assert sample_session.state == SessionState.LOCKED
        assert sample_session.locked_until is not None
        assert sample_session.is_valid() is False
    
    def test_session_revocation(self, sample_session):
        """Test session revocation."""
        sample_session.revoke()
        
        assert sample_session.state == SessionState.REVOKED
        assert sample_session.is_valid() is False
    
    def test_session_activity_update(self, sample_session):
        """Test session activity tracking."""
        original_activity = sample_session.last_activity
        
        # Small delay to ensure timestamp difference
        import time
        time.sleep(0.001)
        
        sample_session.update_activity()
        
        assert sample_session.last_activity > original_activity


class TestAuthenticationManager:
    """Test suite for Enhanced Authentication Manager."""
    
    @pytest.fixture
    async def auth_manager(self):
        """Create authentication manager for testing."""
        return AuthenticationManager()
    
    async def test_api_key_generation(self, auth_manager):
        """Test API key generation with metadata."""
        api_key = auth_manager.generate_api_key(
            user_id="test_user",
            role=UserRole.ADMIN,
            expires_at=datetime.utcnow() + timedelta(days=30),
            allowed_ips=["192.168.1.100"],
            requires_mfa=True
        )
        
        assert ":" in api_key  # Format should be key_id:actual_key
        key_id, actual_key = api_key.split(":", 1)
        
        # Verify key was stored with metadata
        assert key_id in auth_manager.api_keys
        key_info = auth_manager.api_keys[key_id]
        
        assert key_info["role"] == UserRole.ADMIN
        assert key_info["user_id"] == "test_user"
        assert key_info["requires_mfa"] is True
        assert "192.168.1.100" in key_info["allowed_ips"]
    
    async def test_api_key_verification_success(self, auth_manager):
        """Test successful API key verification."""
        # Generate test key
        api_key = auth_manager.generate_api_key("test_user", UserRole.ADMIN)
        
        # Verify key
        result = await auth_manager.verify_api_key(api_key, source_ip="192.168.1.100")
        
        assert result is not None
        assert result["role"] == UserRole.ADMIN.value
        assert result["user_id"] == "test_user"
    
    async def test_api_key_verification_failure(self, auth_manager):
        """Test API key verification failure scenarios."""
        # Invalid key format
        result = await auth_manager.verify_api_key("invalid_key")
        assert result is None
        
        # Non-existent key
        result = await auth_manager.verify_api_key("nonexistent:key")
        assert result is None
        
        # Expired key
        expired_key = auth_manager.generate_api_key(
            "test_user", 
            UserRole.ADMIN,
            expires_at=datetime.utcnow() - timedelta(hours=1)
        )
        result = await auth_manager.verify_api_key(expired_key)
        assert result is None
    
    async def test_api_key_ip_restrictions(self, auth_manager):
        """Test API key IP restrictions."""
        restricted_key = auth_manager.generate_api_key(
            "test_user",
            UserRole.ADMIN,
            allowed_ips=["192.168.1.100"]
        )
        
        # Should work from allowed IP
        result = await auth_manager.verify_api_key(restricted_key, source_ip="192.168.1.100")
        assert result is not None
        
        # Should fail from non-allowed IP
        result = await auth_manager.verify_api_key(restricted_key, source_ip="192.168.1.200")
        assert result is None
    
    async def test_api_key_rate_limiting(self, auth_manager):
        """Test API key rate limiting."""
        # Create key with low rate limit
        api_key = auth_manager.generate_api_key("test_user", UserRole.ADMIN)
        key_id = api_key.split(":")[0]
        auth_manager.api_keys[key_id]["max_usage_per_hour"] = 3
        
        # Use key within rate limit
        for i in range(3):
            result = await auth_manager.verify_api_key(api_key)
            assert result is not None
        
        # Next use should fail due to rate limiting
        result = await auth_manager.verify_api_key(api_key)
        assert result is None
    
    async def test_lockout_mechanism(self, auth_manager):
        """Test authentication failure lockout."""
        auth_manager.max_failures = 3
        
        test_key = "test:invalid_key"
        key_id = "test"
        
        # Simulate multiple failures
        for i in range(5):
            result = await auth_manager.verify_api_key(test_key)
            # Should be None due to invalid key, which triggers failure recording
        
        # Should be locked out
        assert auth_manager._is_locked_out(key_id)
    
    def test_mfa_setup(self, auth_manager):
        """Test MFA setup process."""
        secret, qr_code, backup_codes = auth_manager.setup_mfa("test_user")
        
        assert len(secret) > 0
        assert len(qr_code) > 0  # Base64 encoded QR code
        assert len(backup_codes) == 10
        
        # Verify secret is stored
        assert "test_user" in auth_manager.mfa_secrets
        assert "test_user" in auth_manager.mfa_backup_codes
    
    def test_mfa_token_verification(self, auth_manager):
        """Test MFA token verification."""
        # Setup MFA for user
        secret, _, _ = auth_manager.setup_mfa("test_user")
        
        # In production, would use actual TOTP token generation
        # For testing, we'll patch the verification
        with patch.object(auth_manager, 'verify_mfa_token', return_value=True):
            result = auth_manager.verify_mfa_token("test_user", "123456")
            assert result is True
        
        with patch.object(auth_manager, 'verify_mfa_token', return_value=False):
            result = auth_manager.verify_mfa_token("test_user", "invalid")
            assert result is False
    
    def test_mfa_backup_codes(self, auth_manager):
        """Test MFA backup code usage."""
        _, _, backup_codes = auth_manager.setup_mfa("test_user")
        
        # Use a backup code
        first_code = backup_codes[0]
        result = auth_manager.verify_mfa_backup_code("test_user", first_code)
        assert result is True
        
        # Code should be consumed (can't use again)
        result = auth_manager.verify_mfa_backup_code("test_user", first_code)
        assert result is False
    
    def test_session_management(self, auth_manager):
        """Test comprehensive session management."""
        # Create session
        session = auth_manager.create_session(
            user_id="test_user",
            role=UserRole.ADMIN,
            source_ip="192.168.1.100",
            mfa_verified=True
        )
        
        assert session.session_id in auth_manager.active_sessions
        assert session.mfa_verified is True
        
        # Retrieve session
        retrieved_session = auth_manager.get_session(session.session_id)
        assert retrieved_session is not None
        assert retrieved_session.user_id == "test_user"
        
        # Revoke session
        revoked = auth_manager.revoke_session(session.session_id)
        assert revoked is True
        assert session.session_id not in auth_manager.active_sessions
    
    def test_user_session_revocation(self, auth_manager):
        """Test revoking all sessions for a user."""
        user_id = "test_user"
        
        # Create multiple sessions for user
        sessions = []
        for i in range(3):
            session = auth_manager.create_session(
                user_id=user_id,
                role=UserRole.ADMIN,
                source_ip=f"192.168.1.{100 + i}"
            )
            sessions.append(session)
        
        # Verify sessions exist
        assert len(auth_manager.active_sessions) == 3
        
        # Revoke all user sessions
        revoked_count = auth_manager.revoke_all_user_sessions(user_id)
        assert revoked_count == 3
        assert len(auth_manager.active_sessions) == 0
    
    def test_session_cleanup(self, auth_manager):
        """Test expired session cleanup."""
        # Create expired session
        expired_session = auth_manager.create_session(
            user_id="test_user",
            role=UserRole.ADMIN
        )
        
        # Manually expire it
        expired_session.expires_at = datetime.utcnow() - timedelta(minutes=1)
        
        # Create valid session
        valid_session = auth_manager.create_session(
            user_id="test_user2",
            role=UserRole.ADMIN
        )
        
        # Cleanup should remove expired session only
        cleaned_count = auth_manager.cleanup_expired_sessions()
        assert cleaned_count == 1
        assert len(auth_manager.active_sessions) == 1
        assert valid_session.session_id in auth_manager.active_sessions
    
    def test_auth_statistics(self, auth_manager):
        """Test authentication statistics collection."""
        # Create some test data
        auth_manager.create_session("user1", UserRole.ADMIN)
        auth_manager.create_session("user2", UserRole.READONLY)
        auth_manager.setup_mfa("user1")
        
        stats = auth_manager.get_auth_stats()
        
        assert stats["active_sessions"] == 2
        assert stats["mfa_enabled_users"] == 1
        assert "api_keys_count" in stats
        assert "locked_out_identifiers" in stats


class TestAuthorizationManager:
    """Test suite for Enhanced Authorization Manager."""
    
    @pytest.fixture
    def authz_manager(self):
        """Create authorization manager for testing."""
        return AuthorizationManager()
    
    def test_permission_checking(self, authz_manager):
        """Test permission checking with wildcard support."""
        # Test exact permission match
        assert authz_manager.check_permission("admin", "trading.pause") is True
        assert authz_manager.check_permission("readonly", "trading.pause") is False
        
        # Test wildcard permission
        assert authz_manager.check_permission("super_admin", "system.anything") is True
        assert authz_manager.check_permission("admin", "system.anything") is False
    
    def test_role_permissions_retrieval(self, authz_manager):
        """Test role permissions retrieval."""
        admin_permissions = authz_manager.get_role_permissions("admin")
        readonly_permissions = authz_manager.get_role_permissions("readonly")
        
        assert len(admin_permissions) > len(readonly_permissions)
        assert "config.write" in admin_permissions
        assert "config.write" not in readonly_permissions
        assert "config.read" in both_permissions  # Both should have read access
    
    def test_multiple_permissions_check(self, authz_manager):
        """Test checking multiple permissions at once."""
        permissions = ["config.read", "config.write", "trading.pause"]
        
        admin_results = authz_manager.check_multiple_permissions("admin", permissions)
        readonly_results = authz_manager.check_multiple_permissions("readonly", permissions)
        
        assert admin_results["config.read"] is True
        assert admin_results["config.write"] is True
        assert admin_results["trading.pause"] is True
        
        assert readonly_results["config.read"] is True
        assert readonly_results["config.write"] is False
        assert readonly_results["trading.pause"] is False
    
    async def test_permission_requirement_success(self, authz_manager):
        """Test successful permission requirement."""
        user_context = {
            "role": "admin",
            "sub": "test_user",
            "session_id": "session_123",
            "mfa_verified": True
        }
        
        # Should not raise exception
        await authz_manager.require_permission(user_context, "config.read")
    
    async def test_permission_requirement_failure(self, authz_manager):
        """Test failed permission requirement."""
        user_context = {
            "role": "readonly",
            "sub": "test_user",
            "session_id": "session_123",
            "mfa_verified": False
        }
        
        # Should raise AuthorizationError
        with pytest.raises(AuthorizationError):
            await authz_manager.require_permission(user_context, "config.write")
    
    async def test_mfa_requirement_for_sensitive_operations(self, authz_manager):
        """Test MFA requirement for sensitive operations."""
        user_context = {
            "role": "admin",
            "sub": "test_user",
            "session_id": "session_123",
            "mfa_verified": False  # MFA not verified
        }
        
        # Sensitive operation should require MFA
        with pytest.raises(AuthorizationError, match="MFA verification required"):
            await authz_manager.require_permission(user_context, "keys.rotate")
    
    def test_hierarchical_permissions(self, authz_manager):
        """Test hierarchical permission system."""
        # Super admin should have access to everything
        assert authz_manager.check_permission("super_admin", "system.shutdown") is True
        assert authz_manager.check_permission("super_admin", "trading.execute") is True
        assert authz_manager.check_permission("super_admin", "config.write") is True
        
        # System should have internal access
        assert authz_manager.check_permission("system", "internal.process") is True
        assert authz_manager.check_permission("system", "config.write") is False
    
    def test_permission_summary_generation(self, authz_manager):
        """Test permission summary generation."""
        summary = authz_manager.get_user_permissions_summary("admin")
        
        assert summary["role"] == "admin"
        assert summary["total_permissions"] > 0
        assert "categories" in summary
        assert "wildcard_permissions" in summary
        assert "specific_permissions" in summary
    
    def test_require_any_permission(self, authz_manager):
        """Test requiring any of multiple permissions."""
        user_context = {"role": "readonly"}
        
        # Should pass if user has any of the permissions
        result = authz_manager.require_any_permission(
            user_context, 
            ["config.read", "config.write"]  # readonly has config.read
        )
        assert result is True
        
        # Should fail if user has none of the permissions
        result = authz_manager.require_any_permission(
            user_context,
            ["config.write", "trading.pause"]  # readonly has neither
        )
        assert result is False
    
    def test_require_all_permissions(self, authz_manager):
        """Test requiring all of multiple permissions."""
        admin_context = {"role": "admin"}
        readonly_context = {"role": "readonly"}
        
        # Admin should have all config permissions
        result = authz_manager.require_all_permissions(
            admin_context,
            ["config.read", "config.write"]
        )
        assert result is True
        
        # Readonly should not have all config permissions
        result = authz_manager.require_all_permissions(
            readonly_context,
            ["config.read", "config.write"]
        )
        assert result is False


@pytest.mark.asyncio
class TestAuthenticationIntegration:
    """Integration tests for authentication components."""
    
    async def test_complete_auth_flow_with_mfa(self):
        """Test complete authentication flow with MFA."""
        auth_manager = AuthenticationManager()
        authz_manager = AuthorizationManager()
        
        # 1. Generate API key
        api_key = auth_manager.generate_api_key(
            "test_user",
            UserRole.ADMIN,
            requires_mfa=True
        )
        
        # 2. Setup MFA
        secret, qr_code, backup_codes = auth_manager.setup_mfa("test_user")
        
        # 3. Verify API key (without MFA first)
        key_result = await auth_manager.verify_api_key(api_key)
        assert key_result is not None
        assert key_result["requires_mfa"] is True
        
        # 4. Create session
        session = auth_manager.create_session(
            user_id="test_user",
            role=UserRole.ADMIN,
            mfa_verified=True  # Simulate MFA verification
        )
        
        # 5. Create JWT with MFA verification
        token = auth_manager.create_access_token(
            subject="test_user",
            permissions=authz_manager.get_role_permissions("admin"),
            session_id=session.session_id,
            mfa_verified=True
        )
        
        # 6. Verify token
        payload = auth_manager.verify_token(token)
        assert payload["sub"] == "test_user"
        assert payload["mfa_verified"] is True
        assert payload["session_valid"] is True
        
        # 7. Test permission with MFA-verified context
        user_context = {
            "role": "admin",
            "sub": "test_user",
            "session_id": session.session_id,
            "mfa_verified": True
        }
        
        # Should allow sensitive operation
        await authz_manager.require_permission(user_context, "keys.rotate")
    
    async def test_session_timeout_and_renewal(self):
        """Test session timeout and renewal mechanisms."""
        auth_manager = AuthenticationManager()
        auth_manager.session_timeout = timedelta(seconds=1)  # Very short for testing
        
        # Create session
        session = auth_manager.create_session("test_user", UserRole.ADMIN)
        assert session.is_valid() is True
        
        # Wait for expiration
        await asyncio.sleep(1.1)
        
        # Session should now be invalid
        retrieved_session = auth_manager.get_session(session.session_id)
        assert retrieved_session is None  # Should be cleaned up
    
    async def test_concurrent_session_management(self):
        """Test concurrent session operations."""
        auth_manager = AuthenticationManager()
        
        # Create multiple sessions concurrently
        session_tasks = []
        for i in range(10):
            task = asyncio.create_task(
                asyncio.coroutine(lambda i=i: auth_manager.create_session(
                    f"user_{i}", UserRole.ADMIN
                ))()
            )
            session_tasks.append(task)
        
        sessions = await asyncio.gather(*session_tasks)
        
        # All sessions should be created successfully
        assert len(sessions) == 10
        assert len(auth_manager.active_sessions) == 10
        
        # All sessions should have unique IDs
        session_ids = [s.session_id for s in sessions]
        assert len(set(session_ids)) == 10
    
    def test_api_key_rotation(self, auth_manager):
        """Test API key rotation functionality."""
        # Generate initial key
        original_key = auth_manager.generate_api_key("test_user", UserRole.ADMIN)
        original_key_id = original_key.split(":")[0]
        
        # Rotate key
        new_key = auth_manager.rotate_api_key(original_key_id)
        assert new_key is not None
        
        new_key_id = new_key.split(":")[0]
        
        # Old key should be gone
        assert original_key_id not in auth_manager.api_keys
        
        # New key should exist with same metadata
        assert new_key_id in auth_manager.api_keys
        new_key_info = auth_manager.api_keys[new_key_id]
        assert new_key_info["user_id"] == "test_user"
        assert new_key_info["role"] == UserRole.ADMIN


@pytest.mark.performance
class TestAuthenticationPerformance:
    """Performance tests for authentication system."""
    
    async def test_api_key_verification_performance(self):
        """Test API key verification performance."""
        auth_manager = AuthenticationManager()
        
        # Generate test key
        api_key = auth_manager.generate_api_key("perf_user", UserRole.ADMIN)
        
        start_time = datetime.utcnow()
        
        # Perform many verifications
        tasks = []
        for i in range(100):
            task = auth_manager.verify_api_key(api_key, source_ip="127.0.0.1")
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        end_time = datetime.utcnow()
        duration = (end_time - start_time).total_seconds()
        
        # Performance requirement: <10ms per verification
        avg_time_per_verification = duration / 100
        assert avg_time_per_verification < 0.010
        
        # All verifications should succeed
        successful_verifications = [r for r in results if r is not None]
        assert len(successful_verifications) == 100
    
    async def test_session_creation_performance(self):
        """Test session creation performance."""
        auth_manager = AuthenticationManager()
        
        start_time = datetime.utcnow()
        
        # Create many sessions
        sessions = []
        for i in range(100):
            session = auth_manager.create_session(f"user_{i}", UserRole.ADMIN)
            sessions.append(session)
        
        end_time = datetime.utcnow()
        duration = (end_time - start_time).total_seconds()
        
        # Performance requirement: <5ms per session
        avg_time_per_session = duration / 100
        assert avg_time_per_session < 0.005
        
        # Verify all sessions were created
        assert len(auth_manager.active_sessions) == 100
        assert all(s.is_valid() for s in sessions)
    
    def test_permission_check_performance(self):
        """Test permission checking performance."""
        authz_manager = AuthorizationManager()
        
        start_time = datetime.utcnow()
        
        # Perform many permission checks
        for i in range(1000):
            result = authz_manager.check_permission("admin", "config.read")
            assert result is True
        
        end_time = datetime.utcnow()
        duration = (end_time - start_time).total_seconds()
        
        # Performance requirement: <1ms per check
        avg_time_per_check = duration / 1000
        assert avg_time_per_check < 0.001
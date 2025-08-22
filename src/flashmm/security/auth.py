"""
FlashMM Enhanced Authentication and Authorization

JWT-based authentication with MFA, comprehensive RBAC, and session management.
"""

import base64
import hashlib
import io
import secrets
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import jwt
import pyotp
import qrcode
from jwt.exceptions import DecodeError, ExpiredSignatureError, InvalidTokenError
from passlib.context import CryptContext

from flashmm.config.settings import get_config
from flashmm.utils.exceptions import AuthenticationError, AuthorizationError
from flashmm.utils.logging import SecurityLogger

logger = SecurityLogger()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class AuthenticationMethod(Enum):
    """Authentication methods supported."""
    API_KEY = "api_key"
    JWT = "jwt"
    MFA_TOTP = "mfa_totp"
    # This is a token type identifier constant, not a hardcoded password
    EMERGENCY_TOKEN = "emergency_token"  # noqa: S105


class SessionState(Enum):
    """Session states."""
    ACTIVE = "active"
    EXPIRED = "expired"
    REVOKED = "revoked"
    LOCKED = "locked"


class UserRole(Enum):
    """Enhanced user roles with hierarchical permissions."""
    SUPER_ADMIN = "super_admin"
    ADMIN = "admin"
    OPERATOR = "operator"
    READONLY = "readonly"
    SYSTEM = "system"
    API_CLIENT = "api_client"
    EMERGENCY = "emergency"


class EnhancedSession:
    """Enhanced session with security features."""

    def __init__(self, session_id: str, user_id: str, role: UserRole,
                 created_at: datetime, expires_at: datetime,
                 source_ip: str | None = None, user_agent: str | None = None):
        self.session_id = session_id
        self.user_id = user_id
        self.role = role
        self.created_at = created_at
        self.expires_at = expires_at
        self.last_activity = created_at
        self.source_ip = source_ip
        self.user_agent = user_agent
        self.state = SessionState.ACTIVE
        self.mfa_verified = False
        self.permissions: set[str] = set()
        self.failed_attempts = 0
        self.locked_until: datetime | None = None

    def is_valid(self) -> bool:
        """Check if session is valid."""
        now = datetime.utcnow()

        if self.state != SessionState.ACTIVE:
            return False

        if now > self.expires_at:
            self.state = SessionState.EXPIRED
            return False

        if self.locked_until and now < self.locked_until:
            return False

        return True

    def update_activity(self) -> None:
        """Update last activity timestamp."""
        self.last_activity = datetime.utcnow()

    def lock(self, duration: timedelta) -> None:
        """Lock session for specified duration."""
        self.state = SessionState.LOCKED
        self.locked_until = datetime.utcnow() + duration

    def revoke(self) -> None:
        """Revoke session."""
        self.state = SessionState.REVOKED


class AuthenticationManager:
    """Enhanced authentication manager with MFA and comprehensive security."""

    def __init__(self):
        self.config = get_config()
        self.secret_key = self.config.get("secret_key", "dev_secret_key")
        self.algorithm = "HS256"
        self.access_token_expire = timedelta(
            hours=self.config.get("security.jwt_expire_hours", 1)
        )
        self.session_timeout = timedelta(
            seconds=self.config.get("security.session_timeout", 3600)
        )

        # Enhanced API key management
        self.api_keys: dict[str, dict[str, Any]] = {}
        self.api_key_usage: dict[str, list[float]] = {}

        # Session management
        self.active_sessions: dict[str, EnhancedSession] = {}

        # MFA management
        self.mfa_secrets: dict[str, str] = {}
        self.mfa_backup_codes: dict[str, list[str]] = {}

        # Authentication failure tracking
        self.auth_failures: dict[str, list[datetime]] = {}
        self.lockout_duration = timedelta(minutes=30)
        self.max_failures = 5
        self.failure_window = timedelta(minutes=15)

        # Initialize default API keys
        self._initialize_api_keys()

    def _initialize_api_keys(self) -> None:
        """Initialize default API keys with enhanced metadata."""
        default_keys = {
            "admin": {
                "key_hash": self._hash_password(self.config.get("api_auth_token", "admin_key")),
                "role": UserRole.ADMIN,
                "created_at": datetime.utcnow(),
                "last_used": None,
                "usage_count": 0,
                "max_usage_per_hour": 1000,
                "allowed_ips": [],  # Empty means all IPs allowed
                "requires_mfa": True,
                "expires_at": None,  # No expiration
                "permissions": set()
            },
            "readonly": {
                "key_hash": self._hash_password("readonly_key_placeholder"),
                "role": UserRole.READONLY,
                "created_at": datetime.utcnow(),
                "last_used": None,
                "usage_count": 0,
                "max_usage_per_hour": 500,
                "allowed_ips": [],
                "requires_mfa": False,
                "expires_at": None,
                "permissions": set()
            }
        }
        self.api_keys.update(default_keys)

    def _hash_password(self, password: str) -> str:
        """Hash password using bcrypt."""
        return pwd_context.hash(password)

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash."""
        return pwd_context.verify(plain_password, hashed_password)

    def generate_api_key(self, user_id: str, role: UserRole,
                        expires_at: datetime | None = None,
                        allowed_ips: list[str] | None = None,
                        requires_mfa: bool = False) -> str:
        """Generate a new API key with enhanced security metadata."""
        api_key = secrets.token_urlsafe(32)
        key_id = hashlib.sha256(api_key.encode()).hexdigest()[:16]

        self.api_keys[key_id] = {
            "key_hash": self._hash_password(api_key),
            "role": role,
            "user_id": user_id,
            "created_at": datetime.utcnow(),
            "last_used": None,
            "usage_count": 0,
            "max_usage_per_hour": 1000,
            "allowed_ips": allowed_ips or [],
            "requires_mfa": requires_mfa,
            "expires_at": expires_at,
            "permissions": set()
        }

        return f"{key_id}:{api_key}"

    def revoke_api_key(self, key_id: str) -> bool:
        """Revoke an API key."""
        if key_id in self.api_keys:
            del self.api_keys[key_id]
            return True
        return False

    def rotate_api_key(self, old_key_id: str) -> str | None:
        """Rotate an API key while preserving metadata."""
        if old_key_id not in self.api_keys:
            return None

        old_metadata = self.api_keys[old_key_id].copy()
        user_id = old_metadata.get("user_id", "unknown")
        role = old_metadata["role"]

        # Generate new key
        new_key = self.generate_api_key(
            user_id=user_id,
            role=role,
            expires_at=old_metadata["expires_at"],
            allowed_ips=old_metadata["allowed_ips"],
            requires_mfa=old_metadata["requires_mfa"]
        )

        # Remove old key
        del self.api_keys[old_key_id]

        return new_key

    def setup_mfa(self, user_id: str) -> tuple[str, str, list[str]]:
        """Setup MFA for a user and return secret, QR code, and backup codes."""
        # Generate TOTP secret
        secret = pyotp.random_base32()
        self.mfa_secrets[user_id] = secret

        # Generate backup codes
        backup_codes = [secrets.token_hex(8) for _ in range(10)]
        self.mfa_backup_codes[user_id] = [
            self._hash_password(code) for code in backup_codes
        ]

        # Generate QR code
        totp = pyotp.TOTP(secret)
        provisioning_uri = totp.provisioning_uri(
            name=user_id,
            issuer_name="FlashMM"
        )

        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(provisioning_uri)
        qr.make(fit=True)

        img = qr.make_image(fill_color="black", back_color="white")
        img_buffer = io.BytesIO()
        img.save(img_buffer, 'PNG')
        img_buffer.seek(0)
        qr_code_b64 = base64.b64encode(img_buffer.getvalue()).decode()

        return secret, qr_code_b64, backup_codes

    def verify_mfa_token(self, user_id: str, token: str) -> bool:
        """Verify MFA TOTP token."""
        if user_id not in self.mfa_secrets:
            return False

        secret = self.mfa_secrets[user_id]
        totp = pyotp.TOTP(secret)

        try:
            return totp.verify(token, valid_window=1)
        except Exception:
            return False

    def verify_mfa_backup_code(self, user_id: str, backup_code: str) -> bool:
        """Verify and consume MFA backup code."""
        if user_id not in self.mfa_backup_codes:
            return False

        backup_codes = self.mfa_backup_codes[user_id]

        for i, hashed_code in enumerate(backup_codes):
            if self.verify_password(backup_code, hashed_code):
                # Remove used backup code
                del backup_codes[i]
                return True

        return False

    def create_session(self, user_id: str, role: UserRole,
                      source_ip: str | None = None,
                      user_agent: str | None = None,
                      mfa_verified: bool = False) -> EnhancedSession:
        """Create a new enhanced session."""
        session_id = secrets.token_urlsafe(32)
        expires_at = datetime.utcnow() + self.session_timeout

        session = EnhancedSession(
            session_id=session_id,
            user_id=user_id,
            role=role,
            created_at=datetime.utcnow(),
            expires_at=expires_at,
            source_ip=source_ip,
            user_agent=user_agent
        )

        session.mfa_verified = mfa_verified
        self.active_sessions[session_id] = session

        return session

    def get_session(self, session_id: str) -> EnhancedSession | None:
        """Get session by ID if valid."""
        session = self.active_sessions.get(session_id)

        if session and session.is_valid():
            session.update_activity()
            return session
        elif session:
            # Clean up invalid session
            del self.active_sessions[session_id]

        return None

    def revoke_session(self, session_id: str) -> bool:
        """Revoke a specific session."""
        if session_id in self.active_sessions:
            self.active_sessions[session_id].revoke()
            del self.active_sessions[session_id]
            return True
        return False

    def revoke_all_user_sessions(self, user_id: str) -> int:
        """Revoke all sessions for a specific user."""
        revoked_count = 0

        sessions_to_revoke = [
            sid for sid, session in self.active_sessions.items()
            if session.user_id == user_id
        ]

        for session_id in sessions_to_revoke:
            self.revoke_session(session_id)
            revoked_count += 1

        return revoked_count

    def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions."""
        expired_sessions = [
            sid for sid, session in self.active_sessions.items()
            if not session.is_valid()
        ]

        for session_id in expired_sessions:
            del self.active_sessions[session_id]

        return len(expired_sessions)

    def create_access_token(self, subject: str, permissions: list[str],
                           session_id: str | None = None,
                           mfa_verified: bool = False) -> str:
        """Create JWT access token with enhanced claims."""
        expire = datetime.utcnow() + self.access_token_expire
        to_encode = {
            "sub": subject,
            "permissions": permissions,
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "access",
            "session_id": session_id,
            "mfa_verified": mfa_verified,
            "jti": secrets.token_hex(16)  # JWT ID for tracking
        }

        return jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)

    def verify_token(self, token: str) -> dict[str, Any]:
        """Verify JWT token and return payload with session validation."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])

            if payload.get("type") != "access":
                raise AuthenticationError("Invalid token type")

            # Validate session if present
            session_id = payload.get("session_id")
            if session_id:
                session = self.get_session(session_id)
                if not session:
                    raise AuthenticationError("Session expired or invalid")

                # Update payload with current session info
                payload["session_valid"] = True
                payload["session_info"] = {
                    "created_at": session.created_at.isoformat(),
                    "last_activity": session.last_activity.isoformat(),
                    "source_ip": session.source_ip,
                    "mfa_verified": session.mfa_verified
                }

            return payload

        except ExpiredSignatureError:
            raise AuthenticationError("Token expired") from None
        except (DecodeError, InvalidTokenError) as e:
            raise AuthenticationError(f"Invalid token: {e}") from e

    def _is_locked_out(self, identifier: str) -> bool:
        """Check if identifier is locked out due to failed attempts."""
        if identifier not in self.auth_failures:
            return False

        now = datetime.utcnow()

        # Clean old failures outside window
        self.auth_failures[identifier] = [
            failure_time for failure_time in self.auth_failures[identifier]
            if now - failure_time < self.failure_window
        ]

        recent_failures = len(self.auth_failures[identifier])

        if recent_failures >= self.max_failures:
            # Check if lockout period has passed
            last_failure = max(self.auth_failures[identifier])
            if now - last_failure < self.lockout_duration:
                return True
            else:
                # Reset failures after lockout period
                self.auth_failures[identifier] = []

        return False

    def _record_auth_failure(self, identifier: str) -> None:
        """Record authentication failure."""
        if identifier not in self.auth_failures:
            self.auth_failures[identifier] = []

        self.auth_failures[identifier].append(datetime.utcnow())

    def _clear_auth_failures(self, identifier: str) -> None:
        """Clear authentication failures for identifier."""
        if identifier in self.auth_failures:
            del self.auth_failures[identifier]

    async def verify_api_key(self, api_key: str, source_ip: str | None = None) -> dict[str, Any] | None:
        """Enhanced API key verification with rate limiting and IP restrictions."""
        try:
            # Parse key format: key_id:actual_key
            if ":" in api_key:
                key_id, actual_key = api_key.split(":", 1)
            else:
                # Legacy format - hash the key to find it
                key_id = None
                actual_key = api_key
                for kid, key_info in self.api_keys.items():
                    if self.verify_password(actual_key, key_info["key_hash"]):
                        key_id = kid
                        break

                if not key_id:
                    await logger.log_authentication_event(
                        "api_key_auth",
                        "unknown",
                        False,
                        source_ip,
                        reason="key_not_found"
                    )
                    return None

            if key_id not in self.api_keys:
                await logger.log_authentication_event(
                    "api_key_auth",
                    "unknown",
                    False,
                    source_ip,
                    reason="invalid_key_id"
                )
                return None

            # Check if locked out
            if self._is_locked_out(key_id):
                await logger.log_authentication_event(
                    "api_key_auth",
                    key_id,
                    False,
                    source_ip,
                    reason="locked_out"
                )
                return None

            key_info = self.api_keys[key_id]

            # Verify key
            if not self.verify_password(actual_key, key_info["key_hash"]):
                self._record_auth_failure(key_id)
                await logger.log_authentication_event(
                    "api_key_auth",
                    key_id,
                    False,
                    source_ip,
                    reason="invalid_key"
                )
                return None

            # Check expiration
            if key_info["expires_at"] and datetime.utcnow() > key_info["expires_at"]:
                await logger.log_authentication_event(
                    "api_key_auth",
                    key_id,
                    False,
                    source_ip,
                    reason="expired_key"
                )
                return None

            # Check IP restrictions
            if key_info["allowed_ips"] and source_ip not in key_info["allowed_ips"]:
                await logger.log_authentication_event(
                    "api_key_auth",
                    key_id,
                    False,
                    source_ip,
                    reason="ip_not_allowed"
                )
                return None

            # Check rate limiting
            now = time.time()
            if key_id not in self.api_key_usage:
                self.api_key_usage[key_id] = []

            # Clean old usage records (1 hour window)
            self.api_key_usage[key_id] = [
                usage_time for usage_time in self.api_key_usage[key_id]
                if now - usage_time < 3600
            ]

            if len(self.api_key_usage[key_id]) >= key_info["max_usage_per_hour"]:
                await logger.log_authentication_event(
                    "api_key_auth",
                    key_id,
                    False,
                    source_ip,
                    reason="rate_limit_exceeded"
                )
                return None

            # Record successful authentication
            self._clear_auth_failures(key_id)
            self.api_key_usage[key_id].append(now)
            key_info["last_used"] = datetime.utcnow()
            key_info["usage_count"] += 1

            await logger.log_authentication_event(
                "api_key_auth",
                key_id,
                True,
                source_ip,
                user_id=key_info.get("user_id"),
                role=key_info["role"].value
            )

            return {
                "key_id": key_id,
                "role": key_info["role"].value,
                "user_id": key_info.get("user_id"),
                "requires_mfa": key_info["requires_mfa"],
                "permissions": list(key_info["permissions"])
            }

        except Exception as e:
            await logger.log_authentication_event(
                "api_key_auth",
                "unknown",
                False,
                source_ip,
                reason="exception",
                error=str(e)
            )
            return None

    def get_auth_stats(self) -> dict[str, Any]:
        """Get authentication statistics."""
        now = datetime.utcnow()

        return {
            "active_sessions": len(self.active_sessions),
            "api_keys_count": len(self.api_keys),
            "locked_out_identifiers": len([
                identifier for identifier in self.auth_failures.keys()
                if self._is_locked_out(identifier)
            ]),
            "recent_failures": sum(
                len([
                    failure for failure in failures
                    if now - failure < self.failure_window
                ])
                for failures in self.auth_failures.values()
            ),
            "mfa_enabled_users": len(self.mfa_secrets)
        }


class AuthorizationManager:
    """Enhanced role-based access control with granular permissions."""

    # Enhanced permission system with hierarchical structure
    PERMISSIONS = {
        UserRole.SUPER_ADMIN: [
            "system.*",
            "admin.*",
            "trading.*",
            "config.*",
            "metrics.*",
            "keys.*",
            "security.*",
            "emergency.*"
        ],
        UserRole.ADMIN: [
            "trading.pause",
            "trading.resume",
            "trading.emergency_stop",
            "config.read",
            "config.write",
            "metrics.read",
            "keys.rotate",
            "security.read",
            "security.manage_users",
            "admin.moderate"
        ],
        UserRole.OPERATOR: [
            "trading.pause",
            "trading.resume",
            "config.read",
            "metrics.read",
            "security.read"
        ],
        UserRole.READONLY: [
            "metrics.read",
            "config.read",
            "trading.view"
        ],
        UserRole.SYSTEM: [
            "trading.execute",
            "metrics.write",
            "internal.*",
            "system.automated"
        ],
        UserRole.API_CLIENT: [
            "api.read",
            "metrics.read"
        ],
        UserRole.EMERGENCY: [
            "emergency.*",
            "trading.emergency_stop",
            "system.shutdown"
        ]
    }

    # Permission dependencies
    PERMISSION_DEPENDENCIES = {
        "config.write": ["config.read"],
        "keys.rotate": ["keys.read"],
        "security.manage_users": ["security.read"],
        "admin.moderate": ["admin.read"]
    }

    def __init__(self):
        self.permission_cache: dict[str, set[str]] = {}
        self._build_permission_cache()

    def _build_permission_cache(self) -> None:
        """Build permission cache with expanded wildcards."""
        for role, permissions in self.PERMISSIONS.items():
            expanded_permissions = set()

            for permission in permissions:
                if permission.endswith(".*"):
                    # Add wildcard permission and expand common sub-permissions
                    expanded_permissions.add(permission)
                    prefix = permission[:-2]

                    # Add common sub-permissions for wildcard
                    common_suffixes = ["read", "write", "execute", "delete", "manage"]
                    for suffix in common_suffixes:
                        expanded_permissions.add(f"{prefix}.{suffix}")
                else:
                    expanded_permissions.add(permission)

                # Add dependencies
                if permission in self.PERMISSION_DEPENDENCIES:
                    expanded_permissions.update(self.PERMISSION_DEPENDENCIES[permission])

            self.permission_cache[role.value] = expanded_permissions

    def get_role_permissions(self, role: str) -> set[str]:
        """Get all permissions for a role."""
        return self.permission_cache.get(role, set())

    def check_permission(self, role: str, permission: str) -> bool:
        """Enhanced permission checking with wildcard support."""
        role_permissions = self.get_role_permissions(role)

        # Check exact match
        if permission in role_permissions:
            return True

        # Check wildcard permissions
        permission_parts = permission.split(".")
        for i in range(len(permission_parts)):
            wildcard_perm = ".".join(permission_parts[:i+1]) + ".*"
            if wildcard_perm in role_permissions:
                return True

        return False

    def check_multiple_permissions(self, role: str, permissions: list[str]) -> dict[str, bool]:
        """Check multiple permissions at once."""
        return {
            permission: self.check_permission(role, permission)
            for permission in permissions
        }

    def require_any_permission(self, user_context: dict[str, Any],
                              permissions: list[str]) -> bool:
        """Require any of the specified permissions."""
        role = user_context.get("role")
        if not role:
            return False

        return any(self.check_permission(role, perm) for perm in permissions)

    def require_all_permissions(self, user_context: dict[str, Any],
                               permissions: list[str]) -> bool:
        """Require all of the specified permissions."""
        role = user_context.get("role")
        if not role:
            return False

        return all(self.check_permission(role, perm) for perm in permissions)

    async def require_permission(self, user_context: dict[str, Any], permission: str) -> None:
        """Enhanced permission requirement with detailed logging."""
        role = user_context.get("role")
        user = user_context.get("sub", "unknown")
        session_id = user_context.get("session_id")
        mfa_verified = user_context.get("mfa_verified", False)

        if not role:
            await logger.log_authorization_event(
                user,
                permission,
                "permission_check",
                False,
                reason="no_role",
                session_id=session_id
            )
            raise AuthorizationError("No role assigned")

        # Check if MFA is required for sensitive operations
        sensitive_operations = [
            "system.*", "emergency.*", "keys.*", "security.manage_users",
            "config.write", "trading.emergency_stop"
        ]

        requires_mfa = any(
            permission.startswith(op.replace(".*", "")) or permission == op
            for op in sensitive_operations
        )

        if requires_mfa and not mfa_verified:
            await logger.log_authorization_event(
                user,
                permission,
                "mfa_required",
                False,
                session_id=session_id
            )
            raise AuthorizationError(
                "MFA verification required for sensitive operation",
                user=user,
                required_permission=permission
            )

        if not self.check_permission(role, permission):
            await logger.log_authorization_event(
                user,
                permission,
                "permission_check",
                False,
                role=role,
                session_id=session_id
            )
            raise AuthorizationError(
                f"Permission denied: {permission}",
                user=user,
                required_permission=permission
            )

        await logger.log_authorization_event(
            user,
            permission,
            "permission_check",
            True,
            role=role,
            session_id=session_id,
            mfa_verified=mfa_verified
        )

    def get_user_permissions_summary(self, role: str) -> dict[str, Any]:
        """Get comprehensive permissions summary for a role."""
        permissions = self.get_role_permissions(role)

        # Group permissions by category
        categories = {}
        for permission in permissions:
            if "." in permission:
                category = permission.split(".")[0]
                if category not in categories:
                    categories[category] = []
                categories[category].append(permission)
            else:
                if "other" not in categories:
                    categories["other"] = []
                categories["other"].append(permission)

        return {
            "role": role,
            "total_permissions": len(permissions),
            "categories": categories,
            "wildcard_permissions": [p for p in permissions if p.endswith(".*")],
            "specific_permissions": [p for p in permissions if not p.endswith(".*")]
        }

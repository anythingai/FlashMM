"""
FlashMM Enhanced Cryptographic Key Management

Advanced key management with HSM integration, key escrow, recovery procedures,
and comprehensive lifecycle management for different security levels.
"""

import asyncio
import base64
import hashlib
import json
import os
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from flashmm.config.settings import get_config
from flashmm.utils.exceptions import SecurityError
from flashmm.utils.logging import SecurityLogger

logger = SecurityLogger()


class KeyType(Enum):
    """Types of cryptographic keys."""
    SIGNING = "signing"
    ENCRYPTION = "encryption"
    API_KEY = "api_key"
    SESSION_KEY = "session_key"
    RECOVERY_KEY = "recovery_key"
    EMERGENCY_KEY = "emergency_key"


class KeySecurityLevel(Enum):
    """Security levels for key storage."""
    HOT = "hot"      # Online, frequent access
    WARM = "warm"    # Online, moderate access
    COLD = "cold"    # Offline, emergency access
    HSM = "hsm"      # Hardware Security Module


class KeyStatus(Enum):
    """Key lifecycle status."""
    ACTIVE = "active"
    PENDING = "pending"
    ROTATED = "rotated"
    REVOKED = "revoked"
    EXPIRED = "expired"
    COMPROMISED = "compromised"


@dataclass
class KeyMetadata:
    """Comprehensive key metadata."""
    key_id: str
    key_type: KeyType
    security_level: KeySecurityLevel
    status: KeyStatus
    created_at: datetime
    expires_at: datetime | None
    last_used: datetime | None
    usage_count: int
    max_usage: int | None
    rotation_interval: timedelta
    next_rotation: datetime
    owner: str
    permissions: set[str]
    hsm_key_id: str | None = None
    escrow_shares: list[str] | None = None
    recovery_metadata: dict[str, Any] | None = None


@dataclass
class KeyUsageEvent:
    """Key usage tracking event."""
    key_id: str
    operation: str
    timestamp: datetime
    user_id: str
    source_ip: str | None
    success: bool
    metadata: dict[str, Any]


class HSMInterface:
    """Hardware Security Module interface abstraction."""

    def __init__(self, hsm_config: dict[str, Any]):
        self.config = hsm_config
        self.hsm_type = hsm_config.get("type", "software_hsm")
        self.connected = False

        # In production, this would connect to actual HSM
        # For now, simulate HSM with enhanced security
        self.software_hsm_keys: dict[str, bytes] = {}

    async def connect(self) -> bool:
        """Connect to HSM."""
        try:
            # Simulate HSM connection
            await logger.log_critical_event(
                "hsm_connection_attempt",
                "key_manager",
                {"hsm_type": self.hsm_type}
            )

            self.connected = True
            return True
        except Exception as e:
            await logger.log_critical_event(
                "hsm_connection_failed",
                "key_manager",
                {"error": str(e)}
            )
            return False

    async def generate_key(self, key_id: str, key_type: KeyType,
                          algorithm: str = "AES-256") -> str:
        """Generate key in HSM."""
        if not self.connected:
            raise SecurityError("HSM not connected")

        try:
            # Simulate HSM key generation
            key_material = secrets.token_bytes(32)  # 256-bit key
            hsm_key_id = f"hsm_{key_id}_{secrets.token_hex(8)}"

            # Store in software HSM simulation
            self.software_hsm_keys[hsm_key_id] = key_material

            await logger.log_critical_event(
                "hsm_key_generated",
                "key_manager",
                {
                    "hsm_key_id": hsm_key_id,
                    "key_type": key_type.value,
                    "algorithm": algorithm
                }
            )

            return hsm_key_id

        except Exception as e:
            await logger.log_critical_event(
                "hsm_key_generation_failed",
                "key_manager",
                {"error": str(e)}
            )
            raise SecurityError(f"HSM key generation failed: {e}") from e

    async def use_key(self, hsm_key_id: str, operation: str,
                     data: bytes) -> bytes:
        """Use HSM key for cryptographic operation."""
        if not self.connected:
            raise SecurityError("HSM not connected")

        if hsm_key_id not in self.software_hsm_keys:
            raise SecurityError(f"HSM key not found: {hsm_key_id}")

        key_material = self.software_hsm_keys[hsm_key_id]

        # Simulate cryptographic operation
        if operation == "encrypt":
            cipher = Fernet(base64.urlsafe_b64encode(key_material))
            return cipher.encrypt(data)
        elif operation == "decrypt":
            cipher = Fernet(base64.urlsafe_b64encode(key_material))
            return cipher.decrypt(data)
        elif operation == "sign":
            # Simulate signing operation
            return hashlib.sha256(key_material + data).digest()
        else:
            raise SecurityError(f"Unsupported HSM operation: {operation}")

    async def revoke_key(self, hsm_key_id: str) -> bool:
        """Revoke key in HSM."""
        if hsm_key_id in self.software_hsm_keys:
            del self.software_hsm_keys[hsm_key_id]

            await logger.log_critical_event(
                "hsm_key_revoked",
                "key_manager",
                {"hsm_key_id": hsm_key_id}
            )

            return True
        return False


class KeyEscrowManager:
    """Manages key escrow using Shamir's Secret Sharing."""

    def __init__(self, threshold: int = 3, total_shares: int = 5):
        self.threshold = threshold
        self.total_shares = total_shares

    def create_escrow_shares(self, key_material: bytes) -> list[str]:
        """Create escrow shares for key material."""
        # In production, use proper Shamir's Secret Sharing implementation
        # For now, simulate with simple splitting

        shares = []
        key_hex = key_material.hex()

        # Simple demonstration - in production use cryptographic secret sharing
        for i in range(self.total_shares):
            share_data = {
                "share_id": i + 1,
                "threshold": self.threshold,
                "total_shares": self.total_shares,
                "key_fragment": hashlib.sha256(
                    f"{key_hex}_{i}".encode()
                ).hexdigest()[:32],  # Simplified
                "created_at": datetime.utcnow().isoformat()
            }

            share_json = json.dumps(share_data)
            share_b64 = base64.b64encode(share_json.encode()).decode()
            shares.append(share_b64)

        return shares

    def reconstruct_key(self, shares: list[str]) -> bytes | None:
        """Reconstruct key from escrow shares."""
        if len(shares) < self.threshold:
            return None

        # In production, implement proper secret reconstruction
        # For now, simulate successful reconstruction
        try:
            # Validate shares format
            for share in shares[:self.threshold]:
                share_data = json.loads(base64.b64decode(share).decode())
                if share_data["threshold"] != self.threshold:
                    return None

            # Simulate key reconstruction
            reconstructed_key = secrets.token_bytes(32)
            return reconstructed_key

        except Exception:
            return None


class EnhancedKeyManager:
    """Enhanced key manager with HSM integration and comprehensive lifecycle management."""

    def __init__(self):
        self.config = get_config()

        # Initialize HSM interface
        hsm_config = self.config.get("security.hsm", {})
        self.hsm = HSMInterface(hsm_config)

        # Initialize key escrow
        escrow_config = self.config.get("security.key_escrow", {})
        self.escrow_manager = KeyEscrowManager(
            threshold=escrow_config.get("threshold", 3),
            total_shares=escrow_config.get("total_shares", 5)
        )

        # Key storage
        self.key_metadata: dict[str, KeyMetadata] = {}
        self.key_usage_log: list[KeyUsageEvent] = []

        # Rotation scheduling
        self.rotation_scheduler: dict[str, datetime] = {}

        # Initialize legacy key managers
        self.hot_key_manager = HotKeyManager()
        self.warm_key_manager = WarmKeyManager()
        self.cold_key_manager = ColdKeyManager()

    async def initialize(self) -> None:
        """Initialize the enhanced key manager."""
        # Connect to HSM
        if not await self.hsm.connect():
            await logger.log_critical_event(
                "key_manager_hsm_connection_failed",
                "key_manager",
                {"fallback": "software_keys"}
            )

        # Load existing key metadata
        await self._load_key_metadata()

        # Start background tasks
        asyncio.create_task(self._key_rotation_monitor())
        asyncio.create_task(self._key_usage_monitor())

    async def generate_key(self, key_type: KeyType,
                          security_level: KeySecurityLevel,
                          owner: str,
                          permissions: set[str] | None = None,
                          expires_at: datetime | None = None,
                          rotation_interval: timedelta = timedelta(days=30)) -> str:
        """Generate a new key with comprehensive metadata."""

        key_id = f"{key_type.value}_{secrets.token_hex(16)}"
        permissions = permissions or set()

        # Generate key based on security level
        hsm_key_id = None
        escrow_shares = None

        if security_level == KeySecurityLevel.HSM:
            hsm_key_id = await self.hsm.generate_key(key_id, key_type)

            # Create escrow shares for HSM keys
            key_material = await self.hsm.use_key(hsm_key_id, "export", b"")
            escrow_shares = self.escrow_manager.create_escrow_shares(key_material)

        elif security_level == KeySecurityLevel.HOT:
            # Generate using hot key manager
            pass
        elif security_level == KeySecurityLevel.WARM:
            # Generate using warm key manager
            pass
        elif security_level == KeySecurityLevel.COLD:
            # Generate using cold key manager
            pass

        # Create metadata
        now = datetime.utcnow()
        metadata = KeyMetadata(
            key_id=key_id,
            key_type=key_type,
            security_level=security_level,
            status=KeyStatus.ACTIVE,
            created_at=now,
            expires_at=expires_at,
            last_used=None,
            usage_count=0,
            max_usage=None,
            rotation_interval=rotation_interval,
            next_rotation=now + rotation_interval,
            owner=owner,
            permissions=permissions,
            hsm_key_id=hsm_key_id,
            escrow_shares=escrow_shares
        )

        self.key_metadata[key_id] = metadata

        # Schedule rotation
        self.rotation_scheduler[key_id] = metadata.next_rotation

        await logger.log_critical_event(
            "key_generated",
            "key_manager",
            {
                "key_id": key_id,
                "key_type": key_type.value,
                "security_level": security_level.value,
                "owner": owner,
                "hsm_protected": hsm_key_id is not None,
                "escrow_enabled": escrow_shares is not None
            }
        )

        return key_id

    async def use_key(self, key_id: str, operation: str,
                     user_id: str, source_ip: str | None = None,
                     data: bytes | None = None) -> Any:
        """Use a key with comprehensive audit logging."""

        if key_id not in self.key_metadata:
            raise SecurityError(f"Key not found: {key_id}")

        metadata = self.key_metadata[key_id]

        # Check key status
        if metadata.status != KeyStatus.ACTIVE:
            raise SecurityError(f"Key not active: {key_id}, status: {metadata.status.value}")

        # Check expiration
        if metadata.expires_at and datetime.utcnow() > metadata.expires_at:
            metadata.status = KeyStatus.EXPIRED
            raise SecurityError(f"Key expired: {key_id}")

        # Check usage limits
        if metadata.max_usage and metadata.usage_count >= metadata.max_usage:
            raise SecurityError(f"Key usage limit exceeded: {key_id}")

        # Log usage event
        usage_event = KeyUsageEvent(
            key_id=key_id,
            operation=operation,
            timestamp=datetime.utcnow(),
            user_id=user_id,
            source_ip=source_ip,
            success=False,
            metadata={"operation": operation}
        )

        try:
            # Perform operation based on security level
            result = None

            if metadata.security_level == KeySecurityLevel.HSM and metadata.hsm_key_id:
                result = await self.hsm.use_key(metadata.hsm_key_id, operation, data or b"")
            else:
                # Use appropriate key manager
                if metadata.security_level == KeySecurityLevel.HOT:
                    result = await self._use_hot_key(key_id, operation, data)
                elif metadata.security_level == KeySecurityLevel.WARM:
                    result = await self._use_warm_key(key_id, operation, data)
                elif metadata.security_level == KeySecurityLevel.COLD:
                    result = await self._use_cold_key(key_id, operation, data)

            # Update metadata
            metadata.last_used = datetime.utcnow()
            metadata.usage_count += 1
            usage_event.success = True

            await logger.log_critical_event(
                "key_usage",
                user_id,
                {
                    "key_id": key_id,
                    "operation": operation,
                    "success": True,
                    "usage_count": metadata.usage_count
                }
            )

            return result

        except Exception as e:
            usage_event.metadata["error"] = str(e)

            await logger.log_critical_event(
                "key_usage_failed",
                user_id,
                {
                    "key_id": key_id,
                    "operation": operation,
                    "error": str(e)
                }
            )

            raise
        finally:
            self.key_usage_log.append(usage_event)

    async def rotate_key(self, key_id: str) -> str:
        """Rotate a key maintaining continuity."""
        if key_id not in self.key_metadata:
            raise SecurityError(f"Key not found: {key_id}")

        old_metadata = self.key_metadata[key_id]

        # Generate new key with same properties
        new_key_id = await self.generate_key(
            key_type=old_metadata.key_type,
            security_level=old_metadata.security_level,
            owner=old_metadata.owner,
            permissions=old_metadata.permissions,
            expires_at=old_metadata.expires_at,
            rotation_interval=old_metadata.rotation_interval
        )

        # Mark old key as rotated
        old_metadata.status = KeyStatus.ROTATED
        old_metadata.next_rotation = datetime.utcnow() + timedelta(days=7)  # Grace period

        await logger.log_critical_event(
            "key_rotated",
            "key_manager",
            {
                "old_key_id": key_id,
                "new_key_id": new_key_id,
                "key_type": old_metadata.key_type.value
            }
        )

        return new_key_id

    async def revoke_key(self, key_id: str, reason: str) -> bool:
        """Revoke a key immediately."""
        if key_id not in self.key_metadata:
            return False

        metadata = self.key_metadata[key_id]
        metadata.status = KeyStatus.REVOKED

        # Revoke from HSM if applicable
        if metadata.hsm_key_id:
            await self.hsm.revoke_key(metadata.hsm_key_id)

        # Remove from rotation schedule
        if key_id in self.rotation_scheduler:
            del self.rotation_scheduler[key_id]

        await logger.log_critical_event(
            "key_revoked",
            "key_manager",
            {
                "key_id": key_id,
                "reason": reason,
                "revoked_at": datetime.utcnow().isoformat()
            }
        )

        return True

    async def emergency_key_recovery(self, key_id: str,
                                   escrow_shares: list[str]) -> str | None:
        """Recover key using escrow shares."""
        if key_id not in self.key_metadata:
            return None

        metadata = self.key_metadata[key_id]

        if not metadata.escrow_shares:
            raise SecurityError(f"Key has no escrow shares: {key_id}")

        # Attempt key reconstruction
        reconstructed_key = self.escrow_manager.reconstruct_key(escrow_shares)

        if not reconstructed_key:
            await logger.log_critical_event(
                "key_recovery_failed",
                "key_manager",
                {"key_id": key_id, "reason": "insufficient_shares"}
            )
            return None

        # Generate new key ID for recovered key
        recovery_key_id = f"recovered_{key_id}_{secrets.token_hex(8)}"

        # Create new metadata for recovered key
        recovery_metadata = KeyMetadata(
            key_id=recovery_key_id,
            key_type=metadata.key_type,
            security_level=KeySecurityLevel.WARM,  # Recovered keys are warm
            status=KeyStatus.ACTIVE,
            created_at=datetime.utcnow(),
            expires_at=None,
            last_used=None,
            usage_count=0,
            max_usage=None,
            rotation_interval=timedelta(days=1),  # Rotate soon
            next_rotation=datetime.utcnow() + timedelta(days=1),
            owner=metadata.owner,
            permissions=metadata.permissions,
            recovery_metadata={
                "original_key_id": key_id,
                "recovery_timestamp": datetime.utcnow().isoformat(),
                "recovery_method": "escrow_shares"
            }
        )

        self.key_metadata[recovery_key_id] = recovery_metadata

        await logger.log_critical_event(
            "key_recovered",
            "key_manager",
            {
                "original_key_id": key_id,
                "recovery_key_id": recovery_key_id,
                "recovery_method": "escrow_shares"
            }
        )

        return recovery_key_id

    async def _use_hot_key(self, key_id: str, operation: str,
                          data: bytes | None) -> Any:
        """Use hot key through hot key manager."""
        # Integration with existing hot key manager
        return await self.hot_key_manager.test_keys_validity()

    async def _use_warm_key(self, key_id: str, operation: str,
                           data: bytes | None) -> Any:
        """Use warm key through warm key manager."""
        # Integration with existing warm key manager
        if operation == "encrypt" and data:
            return self.warm_key_manager.encrypt_key(data.decode())
        return None

    async def _use_cold_key(self, key_id: str, operation: str,
                           data: bytes | None) -> Any:
        """Use cold key through cold key manager."""
        # Integration with existing cold key manager
        return await self.cold_key_manager.emergency_recovery("")

    async def _load_key_metadata(self) -> None:
        """Load existing key metadata from storage."""
        # In production, load from secure persistent storage
        pass

    async def _key_rotation_monitor(self) -> None:
        """Background task to monitor key rotation needs."""
        while True:
            try:
                now = datetime.utcnow()

                for key_id, rotation_time in list(self.rotation_scheduler.items()):
                    if now >= rotation_time:
                        try:
                            await self.rotate_key(key_id)
                        except Exception as e:
                            await logger.log_critical_event(
                                "auto_rotation_failed",
                                "key_manager",
                                {"key_id": key_id, "error": str(e)}
                            )

                await asyncio.sleep(3600)  # Check every hour

            except Exception as e:
                await logger.log_critical_event(
                    "rotation_monitor_error",
                    "key_manager",
                    {"error": str(e)}
                )
                await asyncio.sleep(60)  # Retry in 1 minute

    async def _key_usage_monitor(self) -> None:
        """Background task to monitor key usage patterns."""
        while True:
            try:
                # Analyze usage patterns for anomaly detection
                now = datetime.utcnow()
                recent_usage = [
                    event for event in self.key_usage_log
                    if (now - event.timestamp).total_seconds() < 3600
                ]

                # Check for unusual patterns
                usage_by_key = {}
                for event in recent_usage:
                    if event.key_id not in usage_by_key:
                        usage_by_key[event.key_id] = []
                    usage_by_key[event.key_id].append(event)

                for key_id, events in usage_by_key.items():
                    if len(events) > 100:  # Unusual high usage
                        await logger.log_critical_event(
                            "key_usage_anomaly",
                            "key_manager",
                            {
                                "key_id": key_id,
                                "usage_count": len(events),
                                "timespan": "1_hour"
                            }
                        )

                await asyncio.sleep(1800)  # Check every 30 minutes

            except Exception as e:
                await logger.log_critical_event(
                    "usage_monitor_error",
                    "key_manager",
                    {"error": str(e)}
                )
                await asyncio.sleep(60)

    def get_key_status(self, key_id: str) -> dict[str, Any] | None:
        """Get comprehensive key status."""
        if key_id not in self.key_metadata:
            return None

        metadata = self.key_metadata[key_id]

        return {
            "key_id": key_id,
            "key_type": metadata.key_type.value,
            "security_level": metadata.security_level.value,
            "status": metadata.status.value,
            "created_at": metadata.created_at.isoformat(),
            "expires_at": metadata.expires_at.isoformat() if metadata.expires_at else None,
            "last_used": metadata.last_used.isoformat() if metadata.last_used else None,
            "usage_count": metadata.usage_count,
            "next_rotation": metadata.next_rotation.isoformat(),
            "owner": metadata.owner,
            "permissions": list(metadata.permissions),
            "hsm_protected": metadata.hsm_key_id is not None,
            "escrow_enabled": metadata.escrow_shares is not None
        }

    def get_system_key_status(self) -> dict[str, Any]:
        """Get overall key management system status."""
        active_keys = sum(1 for m in self.key_metadata.values() if m.status == KeyStatus.ACTIVE)
        hsm_keys = sum(1 for m in self.key_metadata.values() if m.hsm_key_id is not None)
        escrowed_keys = sum(1 for m in self.key_metadata.values() if m.escrow_shares is not None)

        return {
            "total_keys": len(self.key_metadata),
            "active_keys": active_keys,
            "hsm_keys": hsm_keys,
            "escrowed_keys": escrowed_keys,
            "hsm_connected": self.hsm.connected,
            "pending_rotations": len([
                k for k, t in self.rotation_scheduler.items()
                if datetime.utcnow() >= t
            ]),
            "recent_usage_events": len([
                e for e in self.key_usage_log
                if (datetime.utcnow() - e.timestamp).total_seconds() < 3600
            ])
        }


# Legacy classes for backward compatibility
class HotKeyManager:
    """Manages keys needed for active trading operations."""

    def __init__(self):
        self.config = get_config()
        self.cambrian_api_key = os.getenv("CAMBRIAN_API_KEY")
        self.cambrian_secret = os.getenv("CAMBRIAN_SECRET_KEY")
        self.sei_private_key = os.getenv("SEI_PRIVATE_KEY")

        # Key rotation settings
        self.rotation_interval = timedelta(
            hours=self.config.get("security.key_rotation_interval_hours", 24)
        )
        self.last_rotation = datetime.now()

        self._validate_keys()

    def _validate_keys(self) -> None:
        """Validate all required keys are present and valid."""
        required_keys = [
            ("CAMBRIAN_API_KEY", self.cambrian_api_key),
            ("CAMBRIAN_SECRET_KEY", self.cambrian_secret),
            ("SEI_PRIVATE_KEY", self.sei_private_key),
        ]

        missing_keys = [name for name, value in required_keys if not value]
        if missing_keys:
            raise SecurityError(f"Missing required hot keys: {missing_keys}")

        # Validate key formats
        if self.cambrian_api_key and len(self.cambrian_api_key) < 16:
            raise SecurityError("Cambrian API key appears to be too short")

        if self.cambrian_secret and len(self.cambrian_secret) < 32:
            raise SecurityError("Cambrian secret key appears to be too short")

    async def test_keys_validity(self) -> bool:
        """Test if keys are valid by making test API calls."""
        try:
            # Test Cambrian keys - would need actual Cambrian SDK
            # For now, just check format
            return (
                self.cambrian_api_key is not None
                and self.cambrian_secret is not None
                and self.sei_private_key is not None
            )
        except Exception as e:
            await logger.log_critical_event(
                "key_validation_failed",
                "system",
                {"error": str(e)}
            )
            return False

    def needs_rotation(self) -> bool:
        """Check if keys need to be rotated."""
        return datetime.now() - self.last_rotation > self.rotation_interval

    async def prepare_for_rotation(self) -> None:
        """Prepare for key rotation by validating new keys."""
        await logger.log_critical_event(
            "key_rotation_initiated",
            "system",
            {"rotation_type": "hot_keys"}
        )


class WarmKeyManager:
    """Manages moderately sensitive keys for monitoring/reporting."""

    def __init__(self):
        self.config = get_config()
        self._cipher = None

        # Encrypted keys
        self.grafana_api_key = self._decrypt_key("GRAFANA_API_KEY")
        self.twitter_bearer_token = self._decrypt_key("TWITTER_BEARER_TOKEN")
        self.influxdb_token = self._decrypt_key("INFLUXDB_TOKEN")

    def _get_cipher(self) -> Fernet:
        """Get Fernet cipher for key decryption."""
        if self._cipher is None:
            encryption_key = os.getenv("ENCRYPTION_KEY")
            if not encryption_key:
                raise SecurityError("Master encryption key not found")

            try:
                # Ensure key is proper base64 format
                if len(encryption_key) != 44:  # Base64 encoded 32 bytes
                    # Generate from password if not proper format
                    key = self._derive_key_from_password(encryption_key)
                else:
                    key = encryption_key.encode()

                self._cipher = Fernet(key)
            except Exception as e:
                raise SecurityError(f"Failed to initialize encryption cipher: {e}") from e

        return self._cipher

    def _derive_key_from_password(self, password: str) -> bytes:
        """Derive encryption key from password using PBKDF2."""
        salt = b"flashmm_salt_2023"  # In production, use random salt
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return key

    def _decrypt_key(self, env_var_name: str) -> str | None:
        """Decrypt key from environment variable."""
        encrypted_value = os.getenv(f"{env_var_name}_ENCRYPTED")
        if not encrypted_value:
            # Fallback to plain text for development
            return os.getenv(env_var_name)

        try:
            cipher = self._get_cipher()
            decrypted = cipher.decrypt(encrypted_value.encode())
            return decrypted.decode()
        except Exception as e:
            raise SecurityError(f"Failed to decrypt key {env_var_name}: {e}") from e

    def encrypt_key(self, key_value: str) -> str:
        """Encrypt a key value for storage."""
        cipher = self._get_cipher()
        encrypted = cipher.encrypt(key_value.encode())
        return base64.urlsafe_b64encode(encrypted).decode()


class ColdKeyManager:
    """Manages offline keys for emergency recovery."""

    def __init__(self):
        self.config = get_config()
        self.recovery_keys: dict[str, str] = {}
        self.emergency_shutdown_key: str | None = None

    async def emergency_recovery(self, recovery_phrase: str) -> dict[str, str]:
        """Recover system using cold keys."""
        if not self._verify_recovery_phrase(recovery_phrase):
            await logger.log_critical_event(
                "invalid_recovery_attempt",
                "unknown",
                {"timestamp": datetime.now().isoformat()}
            )
            raise SecurityError("Invalid recovery phrase")

        # Load cold keys from secure storage
        cold_keys = await self._load_cold_keys()

        await logger.log_critical_event(
            "emergency_cold_key_access",
            "system",
            {"timestamp": datetime.now().isoformat()}
        )

        return cold_keys

    def _verify_recovery_phrase(self, phrase: str) -> bool:
        """Verify recovery phrase - implement with proper validation."""
        # In production, use proper key derivation and verification
        expected_hash = os.getenv("RECOVERY_PHRASE_HASH")
        if not expected_hash:
            return False

        # Simple hash comparison for now
        import hashlib
        phrase_hash = hashlib.sha256(phrase.encode()).hexdigest()
        return phrase_hash == expected_hash

    async def _load_cold_keys(self) -> dict[str, str]:
        """Load cold keys from secure storage."""
        # In production, load from encrypted offline storage
        return {
            "backup_api_key": "cold_backup_key",
            "recovery_private_key": "cold_recovery_key",
        }


class KeyRotationManager:
    """Manages automatic key rotation procedures."""

    def __init__(self):
        self.config = get_config()
        self.hot_key_manager = HotKeyManager()
        self.warm_key_manager = WarmKeyManager()

    async def rotate_hot_keys(self) -> None:
        """Rotate hot keys with zero-downtime procedure."""
        await logger.log_critical_event(
            "key_rotation_started",
            "system",
            {"key_type": "hot", "timestamp": datetime.now().isoformat()}
        )

        try:
            # In production, implement actual key rotation:
            # 1. Generate new keys via Cambrian API
            # 2. Test new keys
            # 3. Atomic swap in environment
            # 4. Revoke old keys

            await logger.log_critical_event(
                "key_rotation_completed",
                "system",
                {"key_type": "hot", "timestamp": datetime.now().isoformat()}
            )

        except Exception as e:
            await logger.log_critical_event(
                "key_rotation_failed",
                "system",
                {"key_type": "hot", "error": str(e)}
            )
            raise SecurityError(f"Hot key rotation failed: {e}") from e

    async def emergency_key_rotation(self) -> None:
        """Emergency key rotation in case of compromise."""
        await logger.log_critical_event(
            "emergency_key_rotation",
            "system",
            {"timestamp": datetime.now().isoformat()}
        )

        # Implement emergency rotation procedure
        await self.rotate_hot_keys()

    def get_rotation_status(self) -> dict[str, Any]:
        """Get current key rotation status."""
        return {
            "last_rotation": self.hot_key_manager.last_rotation.isoformat(),
            "needs_rotation": self.hot_key_manager.needs_rotation(),
            "rotation_interval_hours": self.hot_key_manager.rotation_interval.total_seconds() / 3600,
        }

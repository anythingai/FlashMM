"""
FlashMM Data Encryption

Encryption utilities for sensitive data at rest and in transit.
"""

import base64
import json
from typing import Any

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from flashmm.utils.exceptions import SecurityError
from flashmm.utils.logging import get_logger

logger = get_logger(__name__)


class DataEncryption:
    """Handles encryption and decryption of sensitive data."""

    def __init__(self, master_key: str):
        self.master_key = master_key
        self._cipher = None

    def _get_cipher(self) -> Fernet:
        """Get or create Fernet cipher instance."""
        if self._cipher is None:
            try:
                if len(self.master_key) == 44:  # Base64 encoded 32 bytes
                    key = self.master_key.encode()
                else:
                    key = self._derive_key_from_password(self.master_key)

                self._cipher = Fernet(key)
            except Exception as e:
                raise SecurityError(f"Failed to initialize cipher: {e}") from e

        return self._cipher

    def _derive_key_from_password(self, password: str) -> bytes:
        """Derive encryption key from password using PBKDF2."""
        salt = b"flashmm_salt_2023"  # In production, use random salt per key
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return key

    def encrypt_string(self, plaintext: str) -> str:
        """Encrypt a string value."""
        try:
            cipher = self._get_cipher()
            encrypted = cipher.encrypt(plaintext.encode())
            return base64.urlsafe_b64encode(encrypted).decode()
        except Exception as e:
            raise SecurityError(f"Encryption failed: {e}") from e

    def decrypt_string(self, encrypted_data: str) -> str:
        """Decrypt a string value."""
        try:
            cipher = self._get_cipher()
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode())
            decrypted = cipher.decrypt(encrypted_bytes)
            return decrypted.decode()
        except Exception as e:
            raise SecurityError(f"Decryption failed: {e}") from e

    def encrypt_dict(self, data: dict[str, Any]) -> str:
        """Encrypt a dictionary as JSON."""
        try:
            json_str = json.dumps(data, sort_keys=True)
            return self.encrypt_string(json_str)
        except Exception as e:
            raise SecurityError(f"Dictionary encryption failed: {e}") from e

    def decrypt_dict(self, encrypted_data: str) -> dict[str, Any]:
        """Decrypt a dictionary from JSON."""
        try:
            json_str = self.decrypt_string(encrypted_data)
            return json.loads(json_str)
        except Exception as e:
            raise SecurityError(f"Dictionary decryption failed: {e}") from e

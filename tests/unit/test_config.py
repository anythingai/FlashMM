"""
Unit tests for FlashMM configuration management.
"""

from unittest.mock import patch

import pytest

from flashmm.config.constants import DEFAULT_CONFIG
from flashmm.config.environments import detect_environment
from flashmm.config.settings import ConfigManager


class TestConfigManager:
    """Test configuration manager functionality."""

    @pytest.mark.asyncio
    async def test_config_manager_initialization(self, mock_config):
        """Test ConfigManager initialization."""
        config = ConfigManager()
        assert config is not None
        assert config._config_cache == {}

    def test_get_default_values(self):
        """Test getting default configuration values."""
        config = ConfigManager()

        # Test getting a known default value
        app_name = config.get("app.name", "unknown")
        assert app_name in ["FlashMM", "unknown"]  # Should be either default or fallback

    def test_environment_detection(self):
        """Test environment detection logic."""
        # Test default case
        env = detect_environment()
        assert env in ["development", "testnet", "production"]

    @patch.dict('os.environ', {'ENVIRONMENT': 'testnet'})
    def test_explicit_environment(self):
        """Test explicit environment setting."""
        env = detect_environment()
        assert env == "testnet"

    def test_default_config_structure(self):
        """Test that default config has expected structure."""
        assert "app" in DEFAULT_CONFIG
        assert "trading" in DEFAULT_CONFIG
        assert "api" in DEFAULT_CONFIG

        # Test specific values
        assert DEFAULT_CONFIG["app"]["name"] == "FlashMM"
        assert DEFAULT_CONFIG["app"]["version"] == "1.0.0"


@pytest.mark.unit
class TestConfigConstants:
    """Test configuration constants."""

    def test_default_config_completeness(self):
        """Test that all required config sections exist."""
        required_sections = [
            "app", "api", "trading", "ml", "sei", "cambrian",
            "storage", "monitoring", "security", "data_ingestion"
        ]

        for section in required_sections:
            assert section in DEFAULT_CONFIG, f"Missing config section: {section}"

    def test_trading_config_safety(self):
        """Test that trading is disabled by default for safety."""
        assert DEFAULT_CONFIG["trading"]["enable_trading"] is False

    def test_api_config_defaults(self):
        """Test API configuration defaults."""
        api_config = DEFAULT_CONFIG["api"]
        # Default to localhost for security - can be overridden in production config
        assert api_config["host"] == "127.0.0.1"
        assert api_config["port"] == 8000
        assert isinstance(api_config["cors_origins"], list)

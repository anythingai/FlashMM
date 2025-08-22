"""
FlashMM Configuration Management

Main configuration manager with hierarchical loading:
1. Default Values (code)
2. Environment-specific files (config/environments/)
3. Environment variables
4. Runtime overrides (Redis config cache)
"""

import os
from datetime import datetime
from functools import lru_cache
from typing import Any

import redis.asyncio as redis
import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from flashmm.config.constants import DEFAULT_CONFIG
from flashmm.config.environments import detect_environment
from flashmm.utils.exceptions import ConfigurationError


class Settings(BaseSettings):
    """Main configuration class using Pydantic for validation."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )

    # App settings
    environment: str = Field(default="development", alias="ENVIRONMENT")
    debug: bool = Field(default=False, alias="FLASHMM_DEBUG")
    log_level: str = Field(default="INFO", alias="FLASHMM_LOG_LEVEL")

    # Trading settings
    max_position_usdc: float = Field(default=2000.0, alias="TRADING_MAX_POSITION_USDC")
    quote_frequency_hz: float = Field(default=5.0, alias="TRADING_QUOTE_FREQUENCY_HZ")
    enable_trading: bool = Field(default=False, alias="TRADING_ENABLED")

    # Sei network
    sei_rpc_url: str = Field(default="https://sei-testnet-rpc.polkachu.com", alias="SEI_RPC_URL")
    sei_ws_url: str = Field(default="wss://sei-testnet-rpc.polkachu.com/websocket", alias="SEI_WS_URL")
    sei_chain_id: str = Field(default="atlantic-2", alias="SEI_CHAIN_ID")

    # Storage
    redis_url: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")
    redis_password: str | None = Field(default=None, alias="REDIS_PASSWORD")
    influxdb_url: str = Field(default="http://localhost:8086", alias="INFLUXDB_URL")
    influxdb_token: str | None = Field(default=None, alias="INFLUXDB_TOKEN")
    influxdb_org: str = Field(default="flashmm", alias="INFLUXDB_ORG")
    influxdb_bucket: str = Field(default="metrics", alias="INFLUXDB_BUCKET")

    # API Keys (sensitive)
    cambrian_api_key: str | None = Field(default=None, alias="CAMBRIAN_API_KEY")
    cambrian_secret_key: str | None = Field(default=None, alias="CAMBRIAN_SECRET_KEY")
    sei_private_key: str | None = Field(default=None, alias="SEI_PRIVATE_KEY")

    # External APIs
    grafana_api_key: str | None = Field(default=None, alias="GRAFANA_API_KEY")
    grafana_url: str | None = Field(default=None, alias="GRAFANA_URL")
    twitter_bearer_token: str | None = Field(default=None, alias="TWITTER_BEARER_TOKEN")
    twitter_api_key: str | None = Field(default=None, alias="TWITTER_API_KEY")
    twitter_api_secret: str | None = Field(default=None, alias="TWITTER_API_SECRET")

    # Security
    secret_key: str | None = Field(default=None, alias="SECRET_KEY")
    encryption_key: str | None = Field(default=None, alias="ENCRYPTION_KEY")
    api_auth_token: str | None = Field(default=None, alias="API_AUTH_TOKEN")

    # ML Model
    ml_model_path: str = Field(default="/app/models/latest.pt", alias="ML_MODEL_PATH")
    ml_confidence_threshold: float = Field(default=0.6, alias="ML_CONFIDENCE_THRESHOLD")
    ml_inference_timeout_ms: int = Field(default=5, alias="ML_INFERENCE_TIMEOUT_MS")


class ConfigManager:
    """Configuration manager with hierarchical loading and runtime updates."""

    def __init__(self):
        self.settings = Settings()
        self.redis_client: redis.Redis | None = None
        self._config_cache: dict[str, Any] = {}
        self._environment_config: dict[str, Any] = {}
        self._last_reload = datetime.now()

    async def initialize(self) -> None:
        """Initialize configuration manager."""
        try:
            # Detect environment if not explicitly set
            if not self.settings.environment:
                self.settings.environment = detect_environment()

            # Load environment-specific configuration
            await self._load_environment_config()

            # Setup Redis client for runtime config
            await self._setup_redis_client()

            # Load runtime configuration from Redis
            await self._load_runtime_config()

            # Validate critical configuration
            self._validate_configuration()

        except Exception as e:
            raise ConfigurationError(f"Failed to initialize configuration: {e}") from e

    async def _load_environment_config(self) -> None:
        """Load environment-specific YAML configuration."""
        env = self.settings.environment
        config_path = f"config/environments/{env}.yml"

        if os.path.exists(config_path):
            try:
                with open(config_path, encoding='utf-8') as f:
                    self._environment_config = yaml.safe_load(f) or {}

            except (OSError, yaml.YAMLError) as e:
                raise ConfigurationError(f"Failed to load environment config {config_path}: {e}") from e

    async def _setup_redis_client(self) -> None:
        """Setup Redis client for runtime configuration."""
        try:
            redis_kwargs = {
                "decode_responses": True,
                "socket_timeout": 5,
                "socket_connect_timeout": 5,
                "retry_on_timeout": True,
            }

            if self.settings.redis_password:
                redis_kwargs["password"] = self.settings.redis_password

            self.redis_client = redis.from_url(
                self.settings.redis_url,
                **redis_kwargs
            )

            # Test connection
            await self.redis_client.ping()

        except Exception as e:
            # Redis is optional for basic operation
            self.redis_client = None
            print(f"Redis not available for runtime config: {e}")

    async def _load_runtime_config(self) -> None:
        """Load runtime configuration from Redis."""
        if not self.redis_client:
            return

        try:
            pattern = "flashmm:config:*"
            keys = await self.redis_client.keys(pattern)

            for key in keys:
                value = await self.redis_client.get(key)
                if value:
                    config_path = key.replace("flashmm:config:", "").replace(":", ".")
                    self._config_cache[config_path] = self._parse_value(value)

        except Exception as e:
            print(f"Failed to load runtime config from Redis: {e}")

    def _parse_value(self, value: str) -> Any:
        """Parse string value to appropriate Python type."""
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'

        try:
            # Try int first
            if '.' not in value:
                return int(value)
            # Then float
            return float(value)
        except ValueError:
            # Return as string
            return value

    def _validate_configuration(self) -> None:
        """Validate critical configuration settings."""
        required_for_trading = [
            "cambrian_api_key",
            "cambrian_secret_key",
            "sei_private_key"
        ]

        if self.get("trading.enable_trading", False):
            missing = [key for key in required_for_trading if not getattr(self.settings, key)]
            if missing:
                raise ConfigurationError(f"Trading enabled but missing required keys: {missing}")

        # Validate trading parameters
        max_position = self.get("trading.max_position_usdc", 0)
        if max_position <= 0 or max_position > 100000:
            raise ConfigurationError("Invalid max_position_usdc value")

        quote_freq = self.get("trading.quote_frequency_hz", 0)
        if quote_freq <= 0 or quote_freq > 10:
            raise ConfigurationError("Invalid quote_frequency_hz value")

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with fallback hierarchy."""
        # 1. Check runtime config (Redis cache)
        if key in self._config_cache:
            return self._config_cache[key]

        # 2. Check environment variables
        env_key = key.upper().replace(".", "_")
        env_value = os.getenv(env_key)
        if env_value is not None:
            return self._parse_value(env_value)

        # 3. Check environment-specific config
        value = self._get_nested_value(self._environment_config, key)
        if value is not None:
            return value

        # 4. Check Pydantic settings
        settings_key = key.replace(".", "_")
        if hasattr(self.settings, settings_key):
            return getattr(self.settings, settings_key)

        # 5. Check default config
        value = self._get_nested_value(DEFAULT_CONFIG, key)
        if value is not None:
            return value

        # 6. Return provided default
        return default

    def _get_nested_value(self, data: dict[str, Any], key: str) -> Any:
        """Get nested dictionary value using dot notation."""
        keys = key.split('.')
        current = data

        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return None

        return current

    async def set_runtime_config(self, key: str, value: Any) -> None:
        """Set runtime configuration value."""
        if self.redis_client:
            try:
                redis_key = f"flashmm:config:{key.replace('.', ':')}"
                await self.redis_client.set(redis_key, str(value))
                self._config_cache[key] = value

                # Publish configuration change
                await self.redis_client.publish("flashmm:config:updates", key)

            except Exception as e:
                raise ConfigurationError(f"Failed to set runtime config {key}: {e}") from e
        else:
            # Store in local cache if Redis not available
            self._config_cache[key] = value

    async def reload(self) -> None:
        """Reload configuration from all sources."""
        await self._load_environment_config()
        await self._load_runtime_config()
        self._last_reload = datetime.now()

    async def reload_key(self, key: str) -> None:
        """Reload specific configuration key from Redis."""
        if self.redis_client:
            try:
                redis_key = f"flashmm:config:{key.replace('.', ':')}"
                value = await self.redis_client.get(redis_key)
                if value:
                    self._config_cache[key] = self._parse_value(value)

            except Exception as e:
                print(f"Failed to reload config key {key}: {e}")

    def get_all(self) -> dict[str, Any]:
        """Get all configuration as a flat dictionary."""
        result = {}

        # Start with defaults
        def flatten_dict(d: dict[str, Any], parent_key: str = '') -> None:
            for k, v in d.items():
                new_key = f"{parent_key}.{k}" if parent_key else k
                if isinstance(v, dict):
                    flatten_dict(v, new_key)
                else:
                    result[new_key] = v

        flatten_dict(DEFAULT_CONFIG)
        flatten_dict(self._environment_config)

        # Override with runtime config
        result.update(self._config_cache)

        return result


# Global configuration instance
_config_manager: ConfigManager | None = None


@lru_cache
def get_config() -> ConfigManager:
    """Get global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager

"""
FlashMM Environment Detection and Management

Handles automatic environment detection and environment-specific behaviors.
"""

import os
from typing import Any


def detect_environment() -> str:
    """Auto-detect environment based on various signals."""

    # Explicit environment variable
    if env := os.getenv("ENVIRONMENT"):
        return env.lower()

    # Docker container detection
    if os.path.exists("/.dockerenv"):
        hostname = os.getenv("HOSTNAME", "")
        if "testnet" in hostname:
            return "testnet"
        elif "prod" in hostname or "production" in hostname:
            return "production"
        else:
            return "testnet"  # Default for containers

    # Kubernetes detection
    if os.getenv("KUBERNETES_SERVICE_HOST"):
        namespace = os.getenv("NAMESPACE", "")
        if "prod" in namespace:
            return "production"
        elif "staging" in namespace:
            return "testnet"
        else:
            return "testnet"

    # Development indicators
    dev_indicators = [
        os.getenv("PYTHONPATH"),
        os.path.exists("pyproject.toml"),
        os.path.exists(".git"),
        os.getenv("VIRTUAL_ENV"),
    ]

    if any(dev_indicators):
        return "development"

    # Safe default
    return "development"


def get_environment_config(environment: str) -> dict[str, Any]:
    """Get environment-specific configuration overrides."""

    configs = {
        "development": {
            "app": {
                "debug": True,
                "log_level": "DEBUG",
            },
            "trading": {
                "max_position_usdc": 100.0,
                "quote_frequency_hz": 1.0,
                "enable_trading": False,  # Safety in dev
            },
            "monitoring": {
                "grafana_enabled": False,
                "twitter_enabled": False,
                "metrics_collection_interval_seconds": 30,
            },
            "security": {
                "encryption_required": False,
                "ip_whitelist_enabled": False,
            },
        },

        "testnet": {
            "app": {
                "debug": False,
                "log_level": "INFO",
            },
            "trading": {
                "max_position_usdc": 2000.0,
                "quote_frequency_hz": 5.0,
                "enable_trading": True,
            },
            "sei": {
                "network": "testnet",
                "rpc_url": "https://rpc.sei-apis.com",
                "ws_url": "wss://rpc.sei-apis.com/websocket",
                "chain_id": "pacific-1",
            },
            "monitoring": {
                "grafana_enabled": True,
                "twitter_enabled": True,
                "alert_thresholds": {
                    "max_drawdown_percent": 5.0,
                    "min_uptime_percent": 95.0,
                },
            },
            "security": {
                "encryption_required": False,
                "audit_logging": True,
            },
        },

        "production": {
            "app": {
                "debug": False,
                "log_level": "WARNING",
            },
            "trading": {
                "max_position_usdc": 10000.0,
                "quote_frequency_hz": 5.0,
                "risk_multiplier": 1.5,
                "enable_trading": True,
            },
            "sei": {
                "network": "mainnet",
                "rpc_url": "https://rpc.sei-apis.com",
                "ws_url": "wss://rpc.sei-apis.com/websocket",
                "chain_id": "pacific-1",
            },
            "storage": {
                "redis": {
                    "password_required": True,
                },
                "influxdb": {
                    "auth_required": True,
                },
            },
            "monitoring": {
                "grafana_enabled": True,
                "twitter_enabled": True,
                "alert_thresholds": {
                    "max_drawdown_percent": 2.0,
                    "min_uptime_percent": 99.0,
                },
            },
            "security": {
                "encryption_required": True,
                "audit_logging": True,
                "ip_whitelist_enabled": True,
                "key_rotation_interval_hours": 4,
            },
        },
    }

    return configs.get(environment, configs["development"])


def is_development() -> bool:
    """Check if running in development environment."""
    return detect_environment() == "development"


def is_production() -> bool:
    """Check if running in production environment."""
    return detect_environment() == "production"


def is_testnet() -> bool:
    """Check if running in testnet environment."""
    return detect_environment() == "testnet"


def get_log_level(environment: str) -> str:
    """Get appropriate log level for environment."""
    levels = {
        "development": "DEBUG",
        "testnet": "INFO",
        "production": "WARNING",
    }
    return levels.get(environment, "INFO")


def should_enable_debug(environment: str) -> bool:
    """Check if debug mode should be enabled."""
    return environment == "development"


def get_cors_origins(environment: str) -> list[str]:
    """Get CORS origins for environment."""
    if environment == "development":
        return [
            "http://localhost:3000",
            "http://localhost:8080",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:8080",
        ]
    elif environment == "testnet":
        return [
            "https://testnet.flashmm.ai",
            "https://dashboard-testnet.flashmm.ai",
        ]
    else:  # production
        return [
            "https://flashmm.ai",
            "https://dashboard.flashmm.ai",
        ]

"""
FlashMM Default Configuration Constants

Contains all default configuration values embedded in the application code.
These serve as the base layer in the hierarchical configuration system.
"""

from typing import Dict, Any

# Default configuration values
DEFAULT_CONFIG: Dict[str, Any] = {
    "app": {
        "name": "FlashMM",
        "version": "1.0.0",
        "debug": False,
        "log_level": "INFO",
        "environment": "development",
    },
    "api": {
        "host": "0.0.0.0",
        "port": 8000,
        "cors_origins": ["*"],
        "rate_limit_per_minute": 60,
        "websocket_heartbeat_interval": 30,
    },
    "trading": {
        "max_position_percent": 2.0,
        "max_position_usdc": 2000.0,
        "quote_frequency_hz": 5.0,
        "max_quote_levels": 3,
        "spread_buffer_bps": 5.0,
        "min_spread_bps": 2.0,
        "inventory_target_percent": 50.0,
        "risk_multiplier": 1.0,
        "enable_trading": False,  # Safety default
    },
    "ml": {
        "inference_timeout_ms": 5,
        "batch_size": 1,
        "confidence_threshold": 0.6,
        "model_path": "/app/models/latest.pt",
        "feature_window_size": 100,
        "prediction_horizon_seconds": 30,
    },
    "sei": {
        "network": "testnet",
        "rpc_url": "https://sei-testnet-rpc.polkachu.com",
        "ws_url": "wss://sei-testnet-rpc.polkachu.com/websocket",
        "chain_id": "atlantic-2",
        "gas_price": "0.02usei",
        "gas_limit": 300000,
    },
    "cambrian": {
        "base_url": "https://api.cambrian.com",
        "timeout_seconds": 10,
        "retry_attempts": 3,
        "retry_delay_seconds": 1,
    },
    "storage": {
        "redis": {
            "host": "localhost",
            "port": 6379,
            "db": 0,
            "password": None,
            "socket_timeout": 5,
            "socket_connect_timeout": 5,
            "retry_on_timeout": True,
            "max_connections": 20,
        },
        "influxdb": {
            "host": "localhost",
            "port": 8086,
            "database": "flashmm",
            "org": "flashmm",
            "bucket": "metrics",
            "timeout": 10,
            "retry_count": 3,
        },
    },
    "monitoring": {
        "grafana_enabled": False,
        "twitter_enabled": False,
        "metrics_collection_interval_seconds": 10,
        "health_check_interval_seconds": 30,
        "performance_tracking_enabled": True,
        "alert_thresholds": {
            "max_drawdown_percent": 5.0,
            "min_uptime_percent": 95.0,
            "max_latency_ms": 350,
            "error_rate_threshold": 0.05,
        },
    },
    "security": {
        "encryption_required": False,
        "audit_logging": True,
        "ip_whitelist_enabled": False,
        "jwt_expire_hours": 1,
        "key_rotation_interval_hours": 24,
        "max_failed_login_attempts": 5,
    },
    "data_ingestion": {
        "websocket_reconnect_delay": 5,
        "max_reconnect_attempts": 10,
        "heartbeat_interval": 30,
        "message_buffer_size": 1000,
        "data_validation_enabled": True,
    },
}

# Trading pairs configuration
SUPPORTED_TRADING_PAIRS = [
    "SEI/USDC",
    "ETH/USDC",
    "BTC/USDC",
]

# Risk management constants
RISK_LIMITS = {
    "max_position_size_usdc": 10000.0,
    "max_daily_volume_usdc": 100000.0,
    "max_orders_per_second": 10,
    "max_open_orders": 50,
    "circuit_breaker_loss_percent": 10.0,
}

# ML model configuration
ML_MODEL_CONFIG = {
    "model_types": ["transformer", "lstm", "ensemble"],
    "max_model_size_mb": 5,
    "inference_timeout_ms": 5,
    "feature_dimensions": 64,
    "sequence_length": 100,
}

# API endpoints configuration
API_ENDPOINTS = {
    "health": "/health",
    "metrics": "/metrics",
    "trading_status": "/trading/status",
    "positions": "/trading/positions",
    "orders": "/trading/orders",
    "admin_pause": "/admin/pause",
    "admin_resume": "/admin/resume",
    "websocket_metrics": "/ws/metrics",
}

# Logging configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "json": {
            "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
            "format": "%(asctime)s %(name)s %(levelname)s %(message)s",
        },
        "structured": {
            "()": "structlog.stdlib.ProcessorFormatter",
            "processor": "structlog.processors.add_logger_name",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "structured",
            "stream": "ext://sys.stdout",
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": "json",
            "filename": "/app/logs/flashmm.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
        },
    },
    "loggers": {
        "flashmm": {
            "level": "INFO",
            "handlers": ["console", "file"],
            "propagate": False,
        },
        "uvicorn": {
            "level": "INFO",
            "handlers": ["console"],
            "propagate": False,
        },
    },
    "root": {
        "level": "WARNING",
        "handlers": ["console"],
    },
}
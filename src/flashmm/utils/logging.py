"""
FlashMM Logging Configuration

Structured logging setup using structlog with JSON output for production
and human-readable output for development.
"""

import logging
import logging.config
import sys
from typing import Any

import structlog


def setup_logging(
    log_level: str = "INFO",
    environment: str = "development",
    enable_json: bool = False,
) -> None:
    """Setup structured logging configuration."""

    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper()),
    )

    # Configure structlog processors
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    # Add environment-specific processors
    if environment == "production" or enable_json:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=True))

    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a configured logger instance."""
    return structlog.get_logger(name)


class TradingEventLogger:
    """Specialized logger for trading events with structured data."""

    def __init__(self):
        self.logger = get_logger("flashmm.trading")

    async def log_order_event(
        self,
        event_type: str,
        order_id: str,
        symbol: str,
        side: str,
        price: float,
        size: float,
        **kwargs
    ) -> None:
        """Log trading order events."""
        self.logger.info(
            "order_event",
            event_type=event_type,
            order_id=order_id,
            symbol=symbol,
            side=side,
            price=price,
            size=size,
            **kwargs
        )

    async def log_fill_event(
        self,
        order_id: str,
        symbol: str,
        side: str,
        price: float,
        size: float,
        fee: float,
        **kwargs
    ) -> None:
        """Log trade fill events."""
        self.logger.info(
            "fill_event",
            order_id=order_id,
            symbol=symbol,
            side=side,
            price=price,
            size=size,
            fee=fee,
            **kwargs
        )

    async def log_pnl_event(
        self,
        symbol: str,
        realized_pnl: float,
        unrealized_pnl: float,
        total_pnl: float,
        **kwargs
    ) -> None:
        """Log P&L events."""
        self.logger.info(
            "pnl_event",
            symbol=symbol,
            realized_pnl=realized_pnl,
            unrealized_pnl=unrealized_pnl,
            total_pnl=total_pnl,
            **kwargs
        )


class PerformanceLogger:
    """Logger for performance metrics and latency tracking."""

    def __init__(self):
        self.logger = get_logger("flashmm.performance")

    async def log_latency(
        self,
        operation: str,
        latency_ms: float,
        success: bool = True,
        **kwargs
    ) -> None:
        """Log operation latency."""
        self.logger.info(
            "latency_measurement",
            operation=operation,
            latency_ms=latency_ms,
            success=success,
            **kwargs
        )

    async def log_throughput(
        self,
        operation: str,
        count: int,
        duration_seconds: float,
        **kwargs
    ) -> None:
        """Log throughput metrics."""
        rate = count / duration_seconds if duration_seconds > 0 else 0
        self.logger.info(
            "throughput_measurement",
            operation=operation,
            count=count,
            duration_seconds=duration_seconds,
            rate_per_second=rate,
            **kwargs
        )

    async def log_system_metrics(
        self,
        cpu_percent: float,
        memory_percent: float,
        disk_usage_percent: float,
        **kwargs
    ) -> None:
        """Log system resource metrics."""
        self.logger.info(
            "system_metrics",
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            disk_usage_percent=disk_usage_percent,
            **kwargs
        )


class SecurityLogger:
    """Logger for security events and audit trails."""

    def __init__(self):
        self.logger = get_logger("flashmm.security")

    async def log_authentication_event(
        self,
        event_type: str,
        user: str,
        success: bool,
        ip_address: str | None = None,
        **kwargs
    ) -> None:
        """Log authentication events."""
        self.logger.info(
            "authentication_event",
            event_type=event_type,
            user=user,
            success=success,
            ip_address=ip_address,
            **kwargs
        )

    async def log_authorization_event(
        self,
        user: str,
        action: str,
        resource: str,
        success: bool,
        **kwargs
    ) -> None:
        """Log authorization events."""
        self.logger.info(
            "authorization_event",
            user=user,
            action=action,
            resource=resource,
            success=success,
            **kwargs
        )

    async def log_critical_event(
        self,
        event_type: str,
        user: str,
        details: dict[str, Any],
        **kwargs
    ) -> None:
        """Log critical security events."""
        self.logger.critical(
            "critical_security_event",
            event_type=event_type,
            user=user,
            details=details,
            **kwargs
        )


class DataLogger:
    """Logger for data ingestion and processing events."""

    def __init__(self):
        self.logger = get_logger("flashmm.data")

    async def log_websocket_event(
        self,
        event_type: str,
        symbol: str,
        message_type: str,
        latency_ms: float | None = None,
        **kwargs
    ) -> None:
        """Log WebSocket data events."""
        self.logger.info(
            "websocket_event",
            event_type=event_type,
            symbol=symbol,
            message_type=message_type,
            latency_ms=latency_ms,
            **kwargs
        )

    async def log_data_validation_event(
        self,
        validation_type: str,
        success: bool,
        error_details: str | None = None,
        **kwargs
    ) -> None:
        """Log data validation events."""
        level = "info" if success else "warning"
        getattr(self.logger, level)(
            "data_validation_event",
            validation_type=validation_type,
            success=success,
            error_details=error_details,
            **kwargs
        )

"""
FlashMM Custom Exceptions

Defines custom exception classes for different error categories in FlashMM.
"""

from typing import Any


class FlashMMError(Exception):
    """Base exception class for FlashMM-specific errors."""

    def __init__(
        self,
        message: str,
        error_code: str | None = None,
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}

    def __str__(self) -> str:
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message

    def to_dict(self) -> dict[str, Any]:
        """Convert exception to dictionary for JSON serialization."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "details": self.details,
        }


class ConfigurationError(FlashMMError):
    """Raised when there's an error in configuration management."""

    def __init__(
        self,
        message: str,
        config_key: str | None = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.config_key = config_key


class TradingError(FlashMMError):
    """Base class for trading-related errors."""

    def __init__(
        self,
        message: str,
        symbol: str | None = None,
        order_id: str | None = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.symbol = symbol
        self.order_id = order_id


class OrderError(TradingError):
    """Raised when there's an error with order operations."""
    pass


class PositionError(TradingError):
    """Raised when there's an error with position management."""

    def __init__(
        self,
        message: str,
        current_position: float | None = None,
        requested_position: float | None = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.current_position = current_position
        self.requested_position = requested_position


class RiskError(TradingError):
    """Raised when risk limits are exceeded."""

    def __init__(
        self,
        message: str,
        risk_metric: str | None = None,
        current_value: float | None = None,
        limit_value: float | None = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.risk_metric = risk_metric
        self.current_value = current_value
        self.limit_value = limit_value


class SecurityError(FlashMMError):
    """Raised for security-related errors."""

    def __init__(
        self,
        message: str,
        security_event_type: str | None = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.security_event_type = security_event_type


class AuthenticationError(SecurityError):
    """Raised when authentication fails."""

    def __init__(
        self,
        message: str,
        user: str | None = None,
        **kwargs
    ):
        super().__init__(message, security_event_type="authentication_failure", **kwargs)
        self.user = user


class AuthorizationError(SecurityError):
    """Raised when authorization fails."""

    def __init__(
        self,
        message: str,
        user: str | None = None,
        required_permission: str | None = None,
        **kwargs
    ):
        super().__init__(message, security_event_type="authorization_failure", **kwargs)
        self.user = user
        self.required_permission = required_permission


class DataValidationError(FlashMMError):
    """Raised when data validation fails."""

    def __init__(
        self,
        message: str,
        field_name: str | None = None,
        field_value: Any | None = None,
        validation_rule: str | None = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.field_name = field_name
        self.field_value = field_value
        self.validation_rule = validation_rule


class DataIngestionError(FlashMMError):
    """Raised when there's an error in data ingestion."""

    def __init__(
        self,
        message: str,
        source: str | None = None,
        data_type: str | None = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.source = source
        self.data_type = data_type


class WebSocketError(DataIngestionError):
    """Raised when there's a WebSocket connection error."""

    def __init__(
        self,
        message: str,
        url: str | None = None,
        reconnect_attempts: int | None = None,
        **kwargs
    ):
        super().__init__(message, source="websocket", **kwargs)
        self.url = url
        self.reconnect_attempts = reconnect_attempts


class ModelError(FlashMMError):
    """Raised when there's an error with ML models."""

    def __init__(
        self,
        message: str,
        model_name: str | None = None,
        model_version: str | None = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.model_name = model_name
        self.model_version = model_version


class InferenceError(ModelError):
    """Raised when model inference fails."""

    def __init__(
        self,
        message: str,
        input_shape: tuple | None = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.input_shape = input_shape


class ExternalAPIError(FlashMMError):
    """Raised when external API calls fail."""

    def __init__(
        self,
        message: str,
        api_name: str | None = None,
        status_code: int | None = None,
        response_data: dict[str, Any] | None = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.api_name = api_name
        self.status_code = status_code
        self.response_data = response_data


class CambrianAPIError(ExternalAPIError):
    """Raised when Cambrian SDK API calls fail."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, api_name="cambrian", **kwargs)


class SeiRPCError(ExternalAPIError):
    """Raised when Sei RPC calls fail."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, api_name="sei_rpc", **kwargs)


class StorageError(FlashMMError):
    """Raised when storage operations fail."""

    def __init__(
        self,
        message: str,
        storage_type: str | None = None,
        operation: str | None = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.storage_type = storage_type
        self.operation = operation


class RedisError(StorageError):
    """Raised when Redis operations fail."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, storage_type="redis", **kwargs)


class InfluxDBError(StorageError):
    """Raised when InfluxDB operations fail."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, storage_type="influxdb", **kwargs)


class CircuitBreakerError(FlashMMError):
    """Raised when circuit breaker is triggered."""

    def __init__(
        self,
        message: str,
        breaker_name: str | None = None,
        trigger_reason: str | None = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.breaker_name = breaker_name
        self.trigger_reason = trigger_reason


class RateLimitError(FlashMMError):
    """Raised when rate limits are exceeded."""

    def __init__(
        self,
        message: str,
        limit_type: str | None = None,
        current_rate: float | None = None,
        limit_rate: float | None = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.limit_type = limit_type
        self.current_rate = current_rate
        self.limit_rate = limit_rate


class TimeoutError(FlashMMError):
    """Raised when operations timeout."""

    def __init__(
        self,
        message: str,
        operation: str | None = None,
        timeout_seconds: float | None = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.operation = operation
        self.timeout_seconds = timeout_seconds


class BlockchainError(FlashMMError):
    """Raised when blockchain operations fail."""

    def __init__(
        self,
        message: str,
        chain_id: str | None = None,
        transaction_hash: str | None = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.chain_id = chain_id
        self.transaction_hash = transaction_hash


class ServiceError(FlashMMError):
    """Raised when service operations fail."""

    def __init__(
        self,
        message: str,
        service_name: str | None = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.service_name = service_name


class ValidationError(FlashMMError):
    """Raised when validation fails."""

    def __init__(
        self,
        message: str,
        field_name: str | None = None,
        validation_rule: str | None = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.field_name = field_name
        self.validation_rule = validation_rule


class OperationalError(FlashMMError):
    """Raised when operational issues occur."""

    def __init__(
        self,
        message: str,
        operation: str | None = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.operation = operation


class EmergencyStopError(FlashMMError):
    """Raised when emergency stop is triggered."""

    def __init__(
        self,
        message: str,
        reason: str | None = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.reason = reason

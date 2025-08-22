"""
FlashMM Utility Decorators

Common decorators for retry logic, timeout handling, latency measurement,
and trading controls.
"""

import asyncio
import builtins
import time
from collections.abc import Callable
from functools import wraps
from typing import Any

from flashmm.utils.exceptions import TimeoutError, TradingError
from flashmm.utils.logging import get_logger

logger = get_logger(__name__)


def retry_async(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,),
) -> Callable:
    """Async retry decorator with exponential backoff."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            current_delay = delay

            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_attempts - 1:
                        break

                    logger.warning(
                        f"Attempt {attempt + 1}/{max_attempts} failed for {func.__name__}: {e}"
                    )
                    await asyncio.sleep(current_delay)
                    current_delay *= backoff

            logger.error(f"All {max_attempts} attempts failed for {func.__name__}")
            if last_exception is not None:
                raise last_exception
            else:
                raise Exception(f"All {max_attempts} attempts failed for {func.__name__}")

        return wrapper
    return decorator


def timeout_async(timeout_seconds: float) -> Callable:
    """Async timeout decorator."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            try:
                return await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=timeout_seconds
                )
            except builtins.TimeoutError as e:
                raise TimeoutError(
                    f"Function {func.__name__} timed out after {timeout_seconds}s",
                    operation=func.__name__,
                    timeout_seconds=timeout_seconds,
                ) from e

        return wrapper
    return decorator


def measure_latency(operation_name: str | None = None) -> Callable:
    """Decorator to measure and log function execution latency."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            op_name = operation_name or func.__name__
            start_time = time.perf_counter()
            success = False

            try:
                result = await func(*args, **kwargs)
                success = True
                return result
            except Exception:
                success = False
                raise
            finally:
                end_time = time.perf_counter()
                latency_ms = (end_time - start_time) * 1000

                logger.info(
                    f"latency_measurement: {op_name} took {latency_ms:.2f}ms, success={success}"
                )

        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            op_name = operation_name or func.__name__
            start_time = time.perf_counter()
            success = False

            try:
                result = func(*args, **kwargs)
                success = True
                return result
            except Exception:
                success = False
                raise
            finally:
                end_time = time.perf_counter()
                latency_ms = (end_time - start_time) * 1000

                logger.info(
                    f"latency_measurement: {op_name} took {latency_ms:.2f}ms, success={success}"
                )

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


def require_trading_enabled(func: Callable) -> Callable:
    """Decorator to ensure trading is enabled before executing trading operations."""

    @wraps(func)
    async def async_wrapper(*args, **kwargs) -> Any:
        # Import here to avoid circular imports
        from flashmm.config.settings import get_config

        config = get_config()
        if not config.get("trading.enable_trading", False):
            raise TradingError(
                "Trading is disabled - operation not allowed",
                error_code="TRADING_DISABLED"
            )

        return await func(*args, **kwargs)

    @wraps(func)
    def sync_wrapper(*args, **kwargs) -> Any:
        # Import here to avoid circular imports
        from flashmm.config.settings import get_config

        config = get_config()
        if not config.get("trading.enable_trading", False):
            raise TradingError(
                "Trading is disabled - operation not allowed",
                error_code="TRADING_DISABLED"
            )

        return func(*args, **kwargs)

    # Return appropriate wrapper based on function type
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper


def circuit_breaker(
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0,
    expected_exception: type[Exception] = Exception,
) -> Callable:
    """Circuit breaker decorator to prevent cascading failures."""

    def decorator(func: Callable) -> Callable:
        failure_count = 0
        last_failure_time = 0
        circuit_open = False

        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            nonlocal failure_count, last_failure_time, circuit_open

            # Check if circuit should be reset
            if circuit_open and time.time() - last_failure_time > recovery_timeout:
                circuit_open = False
                failure_count = 0
                logger.info(f"Circuit breaker reset for {func.__name__}")

            # If circuit is open, fail fast
            if circuit_open:
                raise TradingError(
                    f"Circuit breaker open for {func.__name__}",
                    error_code="CIRCUIT_BREAKER_OPEN"
                )

            try:
                result = await func(*args, **kwargs)
                # Reset failure count on success
                failure_count = 0
                return result

            except expected_exception as e:
                failure_count += 1
                last_failure_time = time.time()

                if failure_count >= failure_threshold:
                    circuit_open = True
                    logger.error(
                        f"Circuit breaker opened for {func.__name__} "
                        f"after {failure_count} failures"
                    )

                raise e

        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            nonlocal failure_count, last_failure_time, circuit_open

            # Check if circuit should be reset
            if circuit_open and time.time() - last_failure_time > recovery_timeout:
                circuit_open = False
                failure_count = 0
                logger.info(f"Circuit breaker reset for {func.__name__}")

            # If circuit is open, fail fast
            if circuit_open:
                raise TradingError(
                    f"Circuit breaker open for {func.__name__}",
                    error_code="CIRCUIT_BREAKER_OPEN"
                )

            try:
                result = func(*args, **kwargs)
                # Reset failure count on success
                failure_count = 0
                return result

            except expected_exception as e:
                failure_count += 1
                last_failure_time = time.time()

                if failure_count >= failure_threshold:
                    circuit_open = True
                    logger.error(
                        f"Circuit breaker opened for {func.__name__} "
                        f"after {failure_count} failures"
                    )

                raise e

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


def rate_limit(calls_per_second: float) -> Callable:
    """Rate limiting decorator."""

    def decorator(func: Callable) -> Callable:
        min_interval = 1.0 / calls_per_second
        last_called = 0

        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            nonlocal last_called

            elapsed = time.time() - last_called
            if elapsed < min_interval:
                sleep_time = min_interval - elapsed
                await asyncio.sleep(sleep_time)

            last_called = time.time()
            return await func(*args, **kwargs)

        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            nonlocal last_called

            elapsed = time.time() - last_called
            if elapsed < min_interval:
                sleep_time = min_interval - elapsed
                time.sleep(sleep_time)

            last_called = time.time()
            return func(*args, **kwargs)

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator

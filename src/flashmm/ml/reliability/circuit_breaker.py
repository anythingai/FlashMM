"""
FlashMM Circuit Breaker Implementation

Circuit breaker pattern for Azure OpenAI API reliability with automatic
failover to rule-based predictions when API failures exceed thresholds.
"""

import asyncio
import time
from typing import Dict, Any, Callable, Optional, List
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field

from flashmm.utils.logging import get_logger
from flashmm.data.storage.data_models import OrderBookSnapshot, Trade, MarketStats

logger = get_logger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"        # Normal operation
    OPEN = "open"           # Circuit open, failing fast
    HALF_OPEN = "half_open" # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""
    failure_threshold: int = 5          # Failures before opening circuit
    recovery_timeout: int = 30          # Seconds to wait before testing recovery
    success_threshold: int = 3          # Successes needed to close circuit
    timeout_threshold: float = 5.0      # Max response time before considering failure (seconds)
    
    # Sliding window for failure counting
    window_size: int = 60               # Window size in seconds
    max_requests_in_window: int = 100   # Max requests to track in window


@dataclass
class RequestResult:
    """Result of a circuit breaker protected request."""
    timestamp: datetime
    success: bool
    response_time: float
    error: Optional[str] = None


class CircuitBreakerMetrics:
    """Metrics tracking for circuit breaker."""
    
    def __init__(self, window_size: int = 60):
        """Initialize metrics tracker.
        
        Args:
            window_size: Sliding window size in seconds
        """
        self.window_size = window_size
        self.requests: List[RequestResult] = []
        self.state_changes: List[Dict[str, Any]] = []
        self._lock = asyncio.Lock()
    
    async def add_request(self, result: RequestResult) -> None:
        """Add request result to metrics."""
        async with self._lock:
            self.requests.append(result)
            await self._cleanup_old_requests()
    
    async def add_state_change(self, old_state: CircuitState, new_state: CircuitState, reason: str) -> None:
        """Record state change."""
        async with self._lock:
            self.state_changes.append({
                'timestamp': datetime.utcnow(),
                'old_state': old_state.value,
                'new_state': new_state.value,
                'reason': reason
            })
            
            # Keep only recent state changes
            if len(self.state_changes) > 100:
                self.state_changes = self.state_changes[-50:]
    
    async def _cleanup_old_requests(self) -> None:
        """Remove requests outside the sliding window."""
        cutoff_time = datetime.utcnow() - timedelta(seconds=self.window_size)
        self.requests = [req for req in self.requests if req.timestamp > cutoff_time]
        
        # Limit total requests tracked
        if len(self.requests) > 1000:
            self.requests = self.requests[-500:]
    
    async def get_failure_count(self) -> int:
        """Get failure count in current window."""
        async with self._lock:
            return sum(1 for req in self.requests if not req.success)
    
    async def get_success_count(self) -> int:
        """Get success count in current window."""
        async with self._lock:
            return sum(1 for req in self.requests if req.success)
    
    async def get_success_rate(self) -> float:
        """Get success rate in current window."""
        async with self._lock:
            if not self.requests:
                return 0.0
            
            success_count = sum(1 for req in self.requests if req.success)
            return success_count / len(self.requests)
    
    async def get_average_response_time(self) -> float:
        """Get average response time in current window."""
        async with self._lock:
            if not self.requests:
                return 0.0
            
            total_time = sum(req.response_time for req in self.requests)
            return total_time / len(self.requests)
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        async with self._lock:
            if not self.requests:
                return {
                    'total_requests': 0,
                    'success_count': 0,
                    'failure_count': 0,
                    'success_rate': 0.0,
                    'average_response_time': 0.0,
                    'state_changes': len(self.state_changes)
                }
            
            success_count = sum(1 for req in self.requests if req.success)
            failure_count = len(self.requests) - success_count
            avg_response_time = sum(req.response_time for req in self.requests) / len(self.requests)
            
            return {
                'total_requests': len(self.requests),
                'success_count': success_count,
                'failure_count': failure_count,
                'success_rate': success_count / len(self.requests),
                'average_response_time': avg_response_time,
                'state_changes': len(self.state_changes),
                'window_size_seconds': self.window_size
            }


class CircuitBreaker:
    """Circuit breaker for protecting against failing services."""
    
    def __init__(self, 
                 name: str,
                 config: Optional[CircuitBreakerConfig] = None,
                 fallback_func: Optional[Callable] = None):
        """Initialize circuit breaker.
        
        Args:
            name: Circuit breaker name for logging
            config: Circuit breaker configuration
            fallback_func: Fallback function to call when circuit is open
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.fallback_func = fallback_func
        
        self.state = CircuitState.CLOSED
        self.last_failure_time: Optional[datetime] = None
        self.consecutive_successes = 0
        self.consecutive_failures = 0
        
        self.metrics = CircuitBreakerMetrics(self.config.window_size)
        self._lock = asyncio.Lock()
        
        logger.info(f"Circuit breaker '{name}' initialized with config: {self.config}")
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection.
        
        Args:
            func: Function to execute
            *args: Positional arguments for function
            **kwargs: Keyword arguments for function
            
        Returns:
            Function result or fallback result
            
        Raises:
            Exception: If circuit is open and no fallback available
        """
        async with self._lock:
            # Check if circuit should be opened or closed
            await self._update_state()
        
        if self.state == CircuitState.OPEN:
            if self.fallback_func:
                logger.warning(f"Circuit breaker '{self.name}' is OPEN, using fallback")
                return await self._execute_fallback(*args, **kwargs)
            else:
                raise Exception(f"Circuit breaker '{self.name}' is OPEN and no fallback available")
        
        # Execute the protected function
        start_time = time.time()
        success = False
        error = None
        
        try:
            result = await func(*args, **kwargs)
            success = True
            response_time = time.time() - start_time
            
            # Check if response time is acceptable
            if response_time > self.config.timeout_threshold:
                success = False
                error = f"Response time {response_time:.2f}s exceeded threshold {self.config.timeout_threshold}s"
                logger.warning(f"Circuit breaker '{self.name}': {error}")
            
            # Record success
            await self._record_result(success, response_time, error)
            
            if success:
                await self._on_success()
            else:
                await self._on_failure(error or "Timeout")
            
            return result
            
        except Exception as e:
            response_time = time.time() - start_time
            error = str(e)
            
            await self._record_result(False, response_time, error)
            await self._on_failure(error)
            
            if self.state == CircuitState.OPEN and self.fallback_func:
                logger.warning(f"Circuit breaker '{self.name}' opened due to failure, using fallback")
                return await self._execute_fallback(*args, **kwargs)
            
            raise
    
    async def _execute_fallback(self, *args, **kwargs) -> Any:
        """Execute fallback function."""
        try:
            if asyncio.iscoroutinefunction(self.fallback_func):
                return await self.fallback_func(*args, **kwargs)
            else:
                return self.fallback_func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Circuit breaker '{self.name}' fallback failed: {e}")
            raise
    
    async def _record_result(self, success: bool, response_time: float, error: Optional[str] = None) -> None:
        """Record request result."""
        result = RequestResult(
            timestamp=datetime.utcnow(),
            success=success,
            response_time=response_time,
            error=error
        )
        
        await self.metrics.add_request(result)
    
    async def _on_success(self) -> None:
        """Handle successful request."""
        async with self._lock:
            self.consecutive_failures = 0
            
            if self.state == CircuitState.HALF_OPEN:
                self.consecutive_successes += 1
                
                if self.consecutive_successes >= self.config.success_threshold:
                    await self._close_circuit("Sufficient successes in HALF_OPEN state")
    
    async def _on_failure(self, error: str) -> None:
        """Handle failed request."""
        async with self._lock:
            self.consecutive_successes = 0
            self.consecutive_failures += 1
            self.last_failure_time = datetime.utcnow()
            
            logger.warning(f"Circuit breaker '{self.name}' failure #{self.consecutive_failures}: {error}")
            
            if self.state in [CircuitState.CLOSED, CircuitState.HALF_OPEN]:
                failure_count = await self.metrics.get_failure_count()
                
                if failure_count >= self.config.failure_threshold:
                    await self._open_circuit(f"Failure threshold reached: {failure_count}")
    
    async def _update_state(self) -> None:
        """Update circuit state based on current conditions."""
        if self.state == CircuitState.OPEN:
            # Check if recovery timeout has passed
            if self.last_failure_time:
                time_since_failure = (datetime.utcnow() - self.last_failure_time).total_seconds()
                
                if time_since_failure >= self.config.recovery_timeout:
                    await self._half_open_circuit("Recovery timeout elapsed")
    
    async def _open_circuit(self, reason: str) -> None:
        """Open the circuit."""
        old_state = self.state
        self.state = CircuitState.OPEN
        self.consecutive_successes = 0
        
        await self.metrics.add_state_change(old_state, self.state, reason)
        logger.warning(f"Circuit breaker '{self.name}' OPENED: {reason}")
    
    async def _half_open_circuit(self, reason: str) -> None:
        """Set circuit to half-open state."""
        old_state = self.state
        self.state = CircuitState.HALF_OPEN
        self.consecutive_successes = 0
        
        await self.metrics.add_state_change(old_state, self.state, reason)
        logger.info(f"Circuit breaker '{self.name}' HALF_OPEN: {reason}")
    
    async def _close_circuit(self, reason: str) -> None:
        """Close the circuit."""
        old_state = self.state
        self.state = CircuitState.CLOSED
        self.consecutive_failures = 0
        self.consecutive_successes = 0
        
        await self.metrics.add_state_change(old_state, self.state, reason)
        logger.info(f"Circuit breaker '{self.name}' CLOSED: {reason}")
    
    async def force_open(self, reason: str = "Manual override") -> None:
        """Manually force circuit open."""
        async with self._lock:
            await self._open_circuit(reason)
    
    async def force_close(self, reason: str = "Manual override") -> None:
        """Manually force circuit closed."""
        async with self._lock:
            await self._close_circuit(reason)
    
    async def reset(self) -> None:
        """Reset circuit breaker to initial state."""
        async with self._lock:
            old_state = self.state
            self.state = CircuitState.CLOSED
            self.consecutive_failures = 0
            self.consecutive_successes = 0
            self.last_failure_time = None
            
            await self.metrics.add_state_change(old_state, self.state, "Manual reset")
            logger.info(f"Circuit breaker '{self.name}' reset")
    
    async def get_status(self) -> Dict[str, Any]:
        """Get current circuit breaker status."""
        async with self._lock:
            stats = await self.metrics.get_stats()
            
            return {
                'name': self.name,
                'state': self.state.value,
                'consecutive_failures': self.consecutive_failures,
                'consecutive_successes': self.consecutive_successes,
                'last_failure_time': self.last_failure_time.isoformat() if self.last_failure_time else None,
                'config': {
                    'failure_threshold': self.config.failure_threshold,
                    'recovery_timeout': self.config.recovery_timeout,
                    'success_threshold': self.config.success_threshold,
                    'timeout_threshold': self.config.timeout_threshold
                },
                'metrics': stats
            }
    
    def is_closed(self) -> bool:
        """Check if circuit is closed (normal operation)."""
        return self.state == CircuitState.CLOSED
    
    def is_open(self) -> bool:
        """Check if circuit is open (failing fast)."""
        return self.state == CircuitState.OPEN
    
    def is_half_open(self) -> bool:
        """Check if circuit is half-open (testing recovery)."""
        return self.state == CircuitState.HALF_OPEN


class AzureOpenAICircuitBreaker(CircuitBreaker):
    """Specialized circuit breaker for Azure OpenAI API calls."""
    
    def __init__(self, fallback_engine=None):
        """Initialize Azure OpenAI circuit breaker.
        
        Args:
            fallback_engine: Rule-based engine for fallback predictions
        """
        # Configure for Azure OpenAI API characteristics
        config = CircuitBreakerConfig(
            failure_threshold=5,        # Open after 5 failures
            recovery_timeout=30,        # Test recovery after 30 seconds  
            success_threshold=3,        # Close after 3 successes
            timeout_threshold=10.0,     # 10 second timeout threshold
            window_size=120,            # 2-minute sliding window
            max_requests_in_window=200  # Track up to 200 requests
        )
        
        super().__init__(
            name="azure-openai",
            config=config,
            fallback_func=self._fallback_prediction if fallback_engine else None
        )
        
        self.fallback_engine = fallback_engine
    
    async def _fallback_prediction(
        self,
        order_book: OrderBookSnapshot,
        recent_trades: List[Trade],
        market_stats: Optional[MarketStats] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Fallback prediction using rule-based engine."""
        if not self.fallback_engine:
            return {
                'direction': 'neutral',
                'confidence': 0.5,
                'price_change_bps': 0.0,
                'magnitude': 'low',
                'reasoning': 'Circuit breaker fallback: No rule engine available',
                'key_factors': ['circuit_breaker_fallback'],
                'timestamp': datetime.utcnow().isoformat(),
                'model_version': 'circuit-breaker-fallback',
                'symbol': order_book.symbol,
                'api_success': False,
                'circuit_breaker_state': self.state.value
            }
        
        try:
            prediction = await self.fallback_engine.predict(
                order_book=order_book,
                recent_trades=recent_trades,
                market_stats=market_stats
            )
            
            # Add circuit breaker metadata
            prediction['api_success'] = False
            prediction['circuit_breaker_state'] = self.state.value
            prediction['fallback_reason'] = 'Azure OpenAI circuit breaker open'
            
            return prediction
            
        except Exception as e:
            logger.error(f"Circuit breaker fallback failed: {e}")
            
            return {
                'direction': 'neutral',
                'confidence': 0.3,
                'price_change_bps': 0.0,
                'magnitude': 'low',
                'reasoning': f'Circuit breaker fallback error: {str(e)}',
                'key_factors': ['fallback_error'],
                'timestamp': datetime.utcnow().isoformat(),
                'model_version': 'circuit-breaker-error-fallback',
                'symbol': order_book.symbol,
                'api_success': False,
                'circuit_breaker_state': self.state.value
            }
    
    async def predict_with_fallback(
        self,
        prediction_func: Callable,
        order_book: OrderBookSnapshot,
        recent_trades: List[Trade],
        market_stats: Optional[MarketStats] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Make prediction with circuit breaker protection and fallback.
        
        Args:
            prediction_func: Azure OpenAI prediction function
            order_book: Current order book
            recent_trades: Recent trades
            market_stats: Market statistics
            **kwargs: Additional arguments
            
        Returns:
            Prediction result (from API or fallback)
        """
        return await self.call(
            prediction_func,
            order_book,
            recent_trades,
            market_stats,
            **kwargs
        )


class CircuitBreakerManager:
    """Manager for multiple circuit breakers."""
    
    def __init__(self):
        """Initialize circuit breaker manager."""
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._lock = asyncio.Lock()
    
    async def register_circuit_breaker(self, circuit_breaker: CircuitBreaker) -> None:
        """Register a circuit breaker."""
        async with self._lock:
            self.circuit_breakers[circuit_breaker.name] = circuit_breaker
            logger.info(f"Registered circuit breaker: {circuit_breaker.name}")
    
    async def get_circuit_breaker(self, name: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker by name.""" 
        async with self._lock:
            return self.circuit_breakers.get(name)
    
    async def get_all_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all circuit breakers."""
        async with self._lock:
            status = {}
            for name, cb in self.circuit_breakers.items():
                status[name] = await cb.get_status()
            return status
    
    async def reset_all(self) -> None:
        """Reset all circuit breakers."""
        async with self._lock:
            for cb in self.circuit_breakers.values():
                await cb.reset()
            logger.info("All circuit breakers reset")
    
    async def get_health_summary(self) -> Dict[str, Any]:
        """Get health summary of all circuit breakers."""
        async with self._lock:
            total_breakers = len(self.circuit_breakers)
            open_breakers = sum(1 for cb in self.circuit_breakers.values() if cb.is_open())
            half_open_breakers = sum(1 for cb in self.circuit_breakers.values() if cb.is_half_open())
            closed_breakers = sum(1 for cb in self.circuit_breakers.values() if cb.is_closed())
            
            return {
                'total_circuit_breakers': total_breakers,
                'open': open_breakers,
                'half_open': half_open_breakers,
                'closed': closed_breakers,
                'health_score': closed_breakers / max(total_breakers, 1),
                'timestamp': datetime.utcnow().isoformat()
            }


# Global circuit breaker manager instance
_circuit_breaker_manager: Optional[CircuitBreakerManager] = None


def get_circuit_breaker_manager() -> CircuitBreakerManager:
    """Get global circuit breaker manager instance."""
    global _circuit_breaker_manager
    if _circuit_breaker_manager is None:
        _circuit_breaker_manager = CircuitBreakerManager()
    return _circuit_breaker_manager
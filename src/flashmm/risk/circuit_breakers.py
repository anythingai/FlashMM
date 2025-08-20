"""
FlashMM Circuit Breaker System

Multi-layered circuit breaker system with specialized breakers for different risk types:
- Price circuit breakers for extreme market moves
- Volume circuit breakers for unusual trading activity
- P&L circuit breakers for loss protection
- Latency circuit breakers for performance degradation
- System circuit breakers for operational issues
"""

import asyncio
import time
from typing import Dict, Any, Optional, List, Callable, Union
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from dataclasses import dataclass, field
import numpy as np

from flashmm.config.settings import get_config
from flashmm.utils.logging import get_logger, TradingEventLogger
from flashmm.utils.exceptions import CircuitBreakerError, RiskError
from flashmm.utils.decorators import measure_latency, timeout_async

logger = get_logger(__name__)
trading_logger = TradingEventLogger()


class BreakerType(Enum):
    """Circuit breaker types."""
    PRICE = "price"
    VOLUME = "volume"
    PNL = "pnl"
    LATENCY = "latency"
    SYSTEM = "system"
    CUSTOM = "custom"


class BreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"        # Normal operation
    OPEN = "open"           # Circuit tripped, blocking operations
    HALF_OPEN = "half_open" # Testing recovery
    COOLDOWN = "cooldown"   # Gradual re-entry phase


@dataclass
class BreakerEvent:
    """Circuit breaker event record."""
    timestamp: datetime
    breaker_name: str
    breaker_type: BreakerType
    event_type: str  # 'trip', 'reset', 'test', 'recover'
    trigger_value: float
    threshold: float
    message: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BreakerConfig:
    """Base circuit breaker configuration."""
    name: str
    breaker_type: BreakerType
    enabled: bool = True
    
    # Thresholds
    threshold: float = 0.0
    warning_threshold: float = 0.0
    
    # Timing
    cooldown_seconds: int = 60
    recovery_test_interval_seconds: int = 30
    gradual_recovery_steps: int = 5
    
    # Response
    halt_trading: bool = True
    reduce_position_size: bool = False
    widen_spreads: bool = False
    cancel_orders: bool = True
    
    # Metadata
    description: str = ""
    priority: int = 1  # 1=highest, 10=lowest


class BaseCircuitBreaker:
    """Base circuit breaker implementation."""
    
    def __init__(self, config: BreakerConfig):
        self.config = config
        self.state = BreakerState.CLOSED
        
        # State tracking
        self.trip_time: Optional[datetime] = None
        self.last_test_time: Optional[datetime] = None
        self.recovery_step = 0
        self.consecutive_failures = 0
        self.consecutive_successes = 0
        
        # Event history
        self.events: List[BreakerEvent] = []
        self.trip_count = 0
        
        # Metrics
        self.total_activations = 0
        self.false_positives = 0
        self.successful_recoveries = 0
        
        self._lock = asyncio.Lock()
        
        logger.info(f"Initialized {self.config.breaker_type.value} circuit breaker: {self.config.name}")
    
    async def check_condition(self, value: float, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Check if circuit breaker should trip.
        
        Args:
            value: Current value to check against threshold
            metadata: Additional context data
            
        Returns:
            True if circuit should trip, False otherwise
        """
        async with self._lock:
            return await self._evaluate_condition(value, metadata or {})
    
    async def _evaluate_condition(self, value: float, metadata: Dict[str, Any]) -> bool:
        """Evaluate if condition should trigger circuit breaker."""
        if not self.config.enabled:
            return False
        
        # Update state based on time
        await self._update_state()
        
        # If already open, don't re-evaluate
        if self.state == BreakerState.OPEN:
            return True
        
        # Check threshold
        should_trip = await self._should_trip(value, metadata)
        
        if should_trip and self.state == BreakerState.CLOSED:
            await self._trip_breaker(value, metadata)
            return True
        elif should_trip and self.state == BreakerState.HALF_OPEN:
            # Failed recovery test
            await self._failed_recovery(value, metadata)
            return True
        elif not should_trip and self.state == BreakerState.HALF_OPEN:
            # Successful recovery test
            await self._successful_recovery(value, metadata)
            return False
        
        return self.state == BreakerState.OPEN
    
    async def _should_trip(self, value: float, metadata: Dict[str, Any]) -> bool:
        """Determine if breaker should trip based on value and conditions."""
        # Base implementation - override in subclasses
        return abs(value) > self.config.threshold
    
    async def _trip_breaker(self, trigger_value: float, metadata: Dict[str, Any]) -> None:
        """Trip the circuit breaker."""
        self.state = BreakerState.OPEN
        self.trip_time = datetime.now()
        self.trip_count += 1
        self.total_activations += 1
        self.consecutive_failures += 1
        self.consecutive_successes = 0
        
        # Log event
        event = BreakerEvent(
            timestamp=datetime.now(),
            breaker_name=self.config.name,
            breaker_type=self.config.breaker_type,
            event_type='trip',
            trigger_value=trigger_value,
            threshold=self.config.threshold,
            message=f"Circuit breaker tripped: {trigger_value} exceeded threshold {self.config.threshold}",
            metadata=metadata
        )
        
        self.events.append(event)
        
        logger.critical(f"🔴 CIRCUIT BREAKER TRIPPED: {self.config.name} - {event.message}")
        
        # Log to trading event logger
        await trading_logger.log_pnl_event(
            "SYSTEM",
            0.0,
            0.0,
            0.0,
            circuit_breaker_reason=f"{self.config.name}: {event.message}"
        )
    
    async def _failed_recovery(self, trigger_value: float, metadata: Dict[str, Any]) -> None:
        """Handle failed recovery attempt."""
        self.state = BreakerState.OPEN
        self.recovery_step = 0
        self.consecutive_failures += 1
        self.consecutive_successes = 0
        
        event = BreakerEvent(
            timestamp=datetime.now(),
            breaker_name=self.config.name,
            breaker_type=self.config.breaker_type,
            event_type='failed_recovery',
            trigger_value=trigger_value,
            threshold=self.config.threshold,
            message=f"Recovery failed: {trigger_value} still exceeds threshold",
            metadata=metadata
        )
        
        self.events.append(event)
        logger.warning(f"🔄 Circuit breaker recovery failed: {self.config.name}")
    
    async def _successful_recovery(self, test_value: float, metadata: Dict[str, Any]) -> None:
        """Handle successful recovery."""
        self.consecutive_successes += 1
        self.consecutive_failures = 0
        
        if self.consecutive_successes >= 3:  # Require multiple successes
            await self._reset_breaker("successful_recovery", metadata)
        
        event = BreakerEvent(
            timestamp=datetime.now(),
            breaker_name=self.config.name,
            breaker_type=self.config.breaker_type,
            event_type='recovery_test_passed',
            trigger_value=test_value,
            threshold=self.config.threshold,
            message=f"Recovery test passed: {test_value} within threshold",
            metadata=metadata
        )
        
        self.events.append(event)
        logger.info(f"✅ Circuit breaker recovery test passed: {self.config.name}")
    
    async def _reset_breaker(self, reason: str, metadata: Dict[str, Any]) -> None:
        """Reset circuit breaker to normal operation."""
        old_state = self.state
        self.state = BreakerState.CLOSED
        self.trip_time = None
        self.recovery_step = 0
        self.consecutive_failures = 0
        self.consecutive_successes = 0
        
        if old_state == BreakerState.OPEN:
            self.successful_recoveries += 1
        
        event = BreakerEvent(
            timestamp=datetime.now(),
            breaker_name=self.config.name,
            breaker_type=self.config.breaker_type,
            event_type='reset',
            trigger_value=0.0,
            threshold=self.config.threshold,
            message=f"Circuit breaker reset: {reason}",
            metadata=metadata
        )
        
        self.events.append(event)
        logger.info(f"🟢 Circuit breaker reset: {self.config.name} - {reason}")
    
    async def _update_state(self) -> None:
        """Update circuit breaker state based on time."""
        if self.state != BreakerState.OPEN or not self.trip_time:
            return
        
        time_since_trip = (datetime.now() - self.trip_time).total_seconds()
        
        # Check if we should enter half-open state for testing
        if time_since_trip >= self.config.recovery_test_interval_seconds:
            if not self.last_test_time or \
               (datetime.now() - self.last_test_time).total_seconds() >= self.config.recovery_test_interval_seconds:
                
                self.state = BreakerState.HALF_OPEN
                self.last_test_time = datetime.now()
                
                logger.info(f"🔄 Circuit breaker entering recovery test: {self.config.name}")
    
    async def manual_reset(self, reason: str = "Manual override") -> None:
        """Manually reset the circuit breaker."""
        async with self._lock:
            await self._reset_breaker(reason, {"manual_reset": True})
    
    async def manual_trip(self, reason: str = "Manual override") -> None:
        """Manually trip the circuit breaker."""
        async with self._lock:
            await self._trip_breaker(0.0, {"manual_trip": True, "reason": reason})
    
    def get_status(self) -> Dict[str, Any]:
        """Get current circuit breaker status."""
        return {
            'name': self.config.name,
            'type': self.config.breaker_type.value,
            'state': self.state.value,
            'enabled': self.config.enabled,
            'threshold': self.config.threshold,
            'warning_threshold': self.config.warning_threshold,
            'trip_count': self.trip_count,
            'total_activations': self.total_activations,
            'successful_recoveries': self.successful_recoveries,
            'false_positives': self.false_positives,
            'consecutive_failures': self.consecutive_failures,
            'consecutive_successes': self.consecutive_successes,
            'trip_time': self.trip_time.isoformat() if self.trip_time else None,
            'time_since_trip_seconds': (datetime.now() - self.trip_time).total_seconds() if self.trip_time else None,
            'recovery_step': self.recovery_step,
            'description': self.config.description,
            'priority': self.config.priority
        }
    
    def is_tripped(self) -> bool:
        """Check if circuit breaker is currently tripped."""
        return self.state == BreakerState.OPEN
    
    def is_testing_recovery(self) -> bool:
        """Check if circuit breaker is testing recovery."""
        return self.state == BreakerState.HALF_OPEN


class PriceCircuitBreaker(BaseCircuitBreaker):
    """Circuit breaker for extreme price movements."""
    
    def __init__(self, symbol: str, config: Optional[BreakerConfig] = None):
        if config is None:
            config = BreakerConfig(
                name=f"price_breaker_{symbol}",
                breaker_type=BreakerType.PRICE,
                threshold=5.0,  # 5 sigma price move
                warning_threshold=3.0,  # 3 sigma warning
                cooldown_seconds=300,  # 5 minutes
                description=f"Price movement circuit breaker for {symbol}"
            )
        
        super().__init__(config)
        self.symbol = symbol
        self.price_history: List[float] = []
        self.volatility_estimate = 0.02  # 2% daily volatility estimate
        
    async def _should_trip(self, price_change_pct: float, metadata: Dict[str, Any]) -> bool:
        """Check if price movement exceeds sigma threshold."""
        # Update volatility estimate if we have recent prices
        current_price = metadata.get('current_price', 0.0)
        if current_price > 0:
            self.price_history.append(current_price)
            if len(self.price_history) > 100:
                self.price_history = self.price_history[-50:]  # Keep last 50 prices
                
            # Calculate rolling volatility
            if len(self.price_history) >= 10:
                returns = np.diff(np.log(self.price_history))
                self.volatility_estimate = np.std(returns) * np.sqrt(288)  # Scale to daily (5min intervals)
        
        # Calculate z-score
        z_score = abs(price_change_pct) / (self.volatility_estimate * 100)  # Convert to percentage
        
        return z_score > self.config.threshold


class VolumeCircuitBreaker(BaseCircuitBreaker):
    """Circuit breaker for unusual volume spikes."""
    
    def __init__(self, symbol: str, config: Optional[BreakerConfig] = None):
        if config is None:
            config = BreakerConfig(
                name=f"volume_breaker_{symbol}",
                breaker_type=BreakerType.VOLUME,
                threshold=10.0,  # 10x normal volume
                warning_threshold=5.0,  # 5x normal volume
                cooldown_seconds=120,  # 2 minutes
                description=f"Volume spike circuit breaker for {symbol}"
            )
        
        super().__init__(config)
        self.symbol = symbol
        self.volume_history: List[float] = []
        self.normal_volume_estimate = 1000.0  # Initial estimate
        
    async def _should_trip(self, current_volume: float, metadata: Dict[str, Any]) -> bool:
        """Check if volume spike exceeds threshold."""
        # Update normal volume estimate
        self.volume_history.append(current_volume)
        if len(self.volume_history) > 144:  # Keep 12 hours of 5-min intervals
            self.volume_history = self.volume_history[-72:]  # Keep last 6 hours
            
        if len(self.volume_history) >= 10:
            # Use median to avoid outlier influence
            self.normal_volume_estimate = np.median(self.volume_history[:-1])  # Exclude current
        
        # Check volume ratio
        if self.normal_volume_estimate > 0:
            volume_ratio = current_volume / self.normal_volume_estimate
            return volume_ratio > self.config.threshold
        
        return False


class PnLCircuitBreaker(BaseCircuitBreaker):
    """Circuit breaker for P&L drawdown protection."""
    
    def __init__(self, config: Optional[BreakerConfig] = None):
        if config is None:
            config = BreakerConfig(
                name="pnl_drawdown_breaker",
                breaker_type=BreakerType.PNL,
                threshold=5000.0,  # $5000 daily loss
                warning_threshold=2500.0,  # $2500 warning
                cooldown_seconds=1800,  # 30 minutes
                description="P&L drawdown protection circuit breaker"
            )
        
        super().__init__(config)
        self.daily_start_pnl = 0.0
        self.peak_pnl = 0.0
        self.max_drawdown = 0.0
        
    async def _should_trip(self, current_pnl: float, metadata: Dict[str, Any]) -> bool:
        """Check if P&L drawdown exceeds threshold."""
        # Track peak P&L for drawdown calculation
        self.peak_pnl = max(self.peak_pnl, current_pnl)
        
        # Calculate current drawdown
        current_drawdown = self.peak_pnl - current_pnl
        self.max_drawdown = max(self.max_drawdown, current_drawdown)
        
        # Calculate daily P&L loss
        daily_pnl = current_pnl - self.daily_start_pnl
        
        # Trip on either daily loss or drawdown
        return (abs(daily_pnl) > self.config.threshold) or (current_drawdown > self.config.threshold)
    
    def reset_daily_pnl(self, starting_pnl: float = 0.0):
        """Reset daily P&L tracking."""
        self.daily_start_pnl = starting_pnl
        self.peak_pnl = starting_pnl
        self.max_drawdown = 0.0


class LatencyCircuitBreaker(BaseCircuitBreaker):
    """Circuit breaker for system latency degradation."""
    
    def __init__(self, config: Optional[BreakerConfig] = None):
        if config is None:
            config = BreakerConfig(
                name="latency_breaker",
                breaker_type=BreakerType.LATENCY,
                threshold=1000.0,  # 1000ms
                warning_threshold=500.0,  # 500ms
                cooldown_seconds=60,  # 1 minute
                description="System latency circuit breaker"
            )
        
        super().__init__(config)
        self.latency_history: List[float] = []
        self.baseline_latency = 50.0  # 50ms baseline
        
    async def _should_trip(self, current_latency_ms: float, metadata: Dict[str, Any]) -> bool:
        """Check if latency exceeds threshold."""
        self.latency_history.append(current_latency_ms)
        if len(self.latency_history) > 100:
            self.latency_history = self.latency_history[-50:]
            
        # Calculate baseline from recent history
        if len(self.latency_history) >= 10:
            self.baseline_latency = np.percentile(self.latency_history[:-5], 50)  # 50th percentile
        
        # Trip if current latency is much higher than baseline
        return current_latency_ms > max(self.config.threshold, self.baseline_latency * 5)


class CircuitBreakerSystem:
    """Comprehensive circuit breaker management system."""
    
    def __init__(self):
        self.config = get_config()
        self.breakers: Dict[str, BaseCircuitBreaker] = {}
        self.system_halted = False
        self.halt_reason = ""
        self.halt_time: Optional[datetime] = None
        
        # Callbacks for different actions
        self.halt_trading_callback: Optional[Callable] = None
        self.cancel_orders_callback: Optional[Callable] = None
        self.reduce_position_callback: Optional[Callable] = None
        self.widen_spreads_callback: Optional[Callable] = None
        
        self._lock = asyncio.Lock()
        
    async def initialize(self) -> None:
        """Initialize circuit breaker system."""
        try:
            # Create default circuit breakers
            symbols = self.config.get("trading.symbols", ["SEI/USDC"])
            
            for symbol in symbols:
                # Price circuit breakers
                price_breaker = PriceCircuitBreaker(symbol)
                await self.register_breaker(price_breaker)
                
                # Volume circuit breakers
                volume_breaker = VolumeCircuitBreaker(symbol)
                await self.register_breaker(volume_breaker)
            
            # System-wide circuit breakers
            pnl_breaker = PnLCircuitBreaker()
            await self.register_breaker(pnl_breaker)
            
            latency_breaker = LatencyCircuitBreaker()
            await self.register_breaker(latency_breaker)
            
            logger.info(f"Circuit breaker system initialized with {len(self.breakers)} breakers")
            
        except Exception as e:
            logger.error(f"Failed to initialize circuit breaker system: {e}")
            raise RiskError(f"Circuit breaker system initialization failed: {e}")
    
    async def register_breaker(self, breaker: BaseCircuitBreaker) -> None:
        """Register a circuit breaker."""
        async with self._lock:
            self.breakers[breaker.config.name] = breaker
            logger.info(f"Registered circuit breaker: {breaker.config.name}")
    
    async def check_all_breakers(self, market_data: Dict[str, Any]) -> bool:
        """Check all circuit breakers and take action if any trip.
        
        Args:
            market_data: Dictionary containing current market data
            
        Returns:
            True if any breaker is tripped and system is halted
        """
        async with self._lock:
            any_tripped = False
            triggered_breakers = []
            
            for name, breaker in self.breakers.items():
                try:
                    # Extract relevant data for each breaker type
                    should_halt = False
                    
                    if breaker.config.breaker_type == BreakerType.PRICE:
                        symbol = getattr(breaker, 'symbol', 'UNKNOWN')
                        price_data = market_data.get(symbol, {})
                        price_change_pct = price_data.get('price_change_pct', 0.0)
                        current_price = price_data.get('current_price', 0.0)
                        
                        should_halt = await breaker.check_condition(
                            price_change_pct,
                            {'current_price': current_price, 'symbol': symbol}
                        )
                    
                    elif breaker.config.breaker_type == BreakerType.VOLUME:
                        symbol = getattr(breaker, 'symbol', 'UNKNOWN')
                        volume_data = market_data.get(symbol, {})
                        current_volume = volume_data.get('volume', 0.0)
                        
                        should_halt = await breaker.check_condition(
                            current_volume,
                            {'symbol': symbol}
                        )
                    
                    elif breaker.config.breaker_type == BreakerType.PNL:
                        current_pnl = market_data.get('total_pnl', 0.0)
                        
                        should_halt = await breaker.check_condition(
                            current_pnl,
                            {'total_pnl': current_pnl}
                        )
                    
                    elif breaker.config.breaker_type == BreakerType.LATENCY:
                        current_latency = market_data.get('system_latency_ms', 0.0)
                        
                        should_halt = await breaker.check_condition(
                            current_latency,
                            {'system_latency_ms': current_latency}
                        )
                    
                    if should_halt and not any_tripped:
                        any_tripped = True
                        triggered_breakers.append(breaker)
                    elif should_halt:
                        triggered_breakers.append(breaker)
                
                except Exception as e:
                    logger.error(f"Error checking circuit breaker {name}: {e}")
            
            # Take action if any breakers tripped
            if any_tripped and not self.system_halted:
                await self._handle_circuit_breaker_trip(triggered_breakers)
            
            return self.system_halted
    
    async def _handle_circuit_breaker_trip(self, triggered_breakers: List[BaseCircuitBreaker]) -> None:
        """Handle circuit breaker trip by taking appropriate actions."""
        # Find highest priority breaker
        priority_breaker = min(triggered_breakers, key=lambda b: b.config.priority)
        
        self.system_halted = True
        self.halt_time = datetime.now()
        self.halt_reason = f"Circuit breaker triggered: {priority_breaker.config.name}"
        
        logger.critical(f"🚨 SYSTEM HALT: {self.halt_reason}")
        
        # Execute configured responses
        try:
            if priority_breaker.config.cancel_orders and self.cancel_orders_callback:
                await self.cancel_orders_callback("Circuit breaker triggered")
            
            if priority_breaker.config.halt_trading and self.halt_trading_callback:
                await self.halt_trading_callback(self.halt_reason)
            
            if priority_breaker.config.reduce_position_size and self.reduce_position_callback:
                await self.reduce_position_callback(0.5)  # Reduce by 50%
            
            if priority_breaker.config.widen_spreads and self.widen_spreads_callback:
                await self.widen_spreads_callback(2.0)  # Double spreads
        
        except Exception as e:
            logger.error(f"Error executing circuit breaker response: {e}")
    
    async def manual_system_reset(self, reason: str = "Manual reset") -> None:
        """Manually reset the entire circuit breaker system."""
        async with self._lock:
            # Reset all breakers
            for breaker in self.breakers.values():
                if breaker.is_tripped():
                    await breaker.manual_reset(reason)
            
            # Reset system halt
            self.system_halted = False
            self.halt_reason = ""
            self.halt_time = None
            
            logger.warning(f"🔄 Circuit breaker system manually reset: {reason}")
    
    def set_callbacks(self,
                      halt_trading: Optional[Callable] = None,
                      cancel_orders: Optional[Callable] = None,
                      reduce_position: Optional[Callable] = None,
                      widen_spreads: Optional[Callable] = None) -> None:
        """Set callback functions for circuit breaker actions."""
        self.halt_trading_callback = halt_trading
        self.cancel_orders_callback = cancel_orders
        self.reduce_position_callback = reduce_position
        self.widen_spreads_callback = widen_spreads
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        breaker_statuses = {name: breaker.get_status() for name, breaker in self.breakers.items()}
        
        active_breakers = sum(1 for b in self.breakers.values() if b.is_tripped())
        testing_breakers = sum(1 for b in self.breakers.values() if b.is_testing_recovery())
        
        return {
            'system_halted': self.system_halted,
            'halt_reason': self.halt_reason,
            'halt_time': self.halt_time.isoformat() if self.halt_time else None,
            'time_since_halt_seconds': (datetime.now() - self.halt_time).total_seconds() if self.halt_time else None,
            'total_breakers': len(self.breakers),
            'active_breakers': active_breakers,
            'testing_recovery_breakers': testing_breakers,
            'healthy_breakers': len(self.breakers) - active_breakers - testing_breakers,
            'breakers': breaker_statuses
        }
    
    async def cleanup(self) -> None:
        """Cleanup circuit breaker system."""
        logger.info("Circuit breaker system cleanup completed")
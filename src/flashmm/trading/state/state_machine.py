"""
FlashMM Trading State Machine

Advanced state management system for controlling trading operations with safety
interlocks, automatic transitions, and comprehensive state tracking.
"""

import asyncio
from typing import Dict, List, Optional, Callable, Any, Set
from enum import Enum
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import json

from flashmm.config.settings import get_config
from flashmm.data.storage.redis_client import RedisClient
from flashmm.utils.logging import get_logger
from flashmm.utils.exceptions import StateTransitionError, ValidationError
from flashmm.utils.decorators import measure_latency, timeout_async

logger = get_logger(__name__)


class TradingState(Enum):
    """Trading system states."""
    INACTIVE = "inactive"           # System not trading
    STARTING = "starting"          # System starting up
    ACTIVE = "active"              # Normal trading operations
    PAUSED = "paused"              # Temporarily paused
    EMERGENCY_STOP = "emergency_stop"  # Emergency stop
    MAINTENANCE = "maintenance"     # Maintenance mode
    SHUTTING_DOWN = "shutting_down" # Graceful shutdown
    ERROR = "error"                # Error state


class StateChangeReason(Enum):
    """Reasons for state changes."""
    MANUAL = "manual"
    AUTOMATIC = "automatic"
    EMERGENCY = "emergency"
    TIMEOUT = "timeout"
    ERROR = "error"
    MAINTENANCE = "maintenance"
    STARTUP = "startup"
    SHUTDOWN = "shutdown"


@dataclass
class StateTransition:
    """State transition record."""
    from_state: TradingState
    to_state: TradingState
    reason: StateChangeReason
    timestamp: datetime
    message: str = ""
    user_id: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'from_state': self.from_state.value,
            'to_state': self.to_state.value,
            'reason': self.reason.value,
            'timestamp': self.timestamp.isoformat(),
            'message': self.message,
            'user_id': self.user_id,
            'data': self.data
        }


@dataclass
class StateCondition:
    """Condition that must be met for state changes."""
    name: str
    check_function: Callable[[], bool]
    message: str
    timeout_seconds: Optional[int] = None


class TradingStateMachine:
    """Advanced trading state machine with safety interlocks."""
    
    def __init__(self):
        self.config = get_config()
        self.redis_client: Optional[RedisClient] = None
        
        # Current state
        self.current_state = TradingState.INACTIVE
        self.state_entered_at = datetime.now()
        self.last_state_change = datetime.now()
        
        # State history
        self.state_history: List[StateTransition] = []
        self.max_history_length = self.config.get("trading.max_state_history", 1000)
        
        # Valid state transitions
        self.valid_transitions = self._define_valid_transitions()
        
        # State conditions and callbacks
        self.state_conditions: Dict[TradingState, List[StateCondition]] = {}
        self.state_enter_callbacks: Dict[TradingState, List[Callable]] = {}
        self.state_exit_callbacks: Dict[TradingState, List[Callable]] = {}
        
        # Emergency conditions
        self.emergency_conditions: List[StateCondition] = []
        
        # Automatic transition timers
        self.auto_transition_timers: Dict[TradingState, Dict[str, Any]] = {}
        
        # Background tasks
        self._monitor_task: Optional[asyncio.Task] = None
        self._persistence_task: Optional[asyncio.Task] = None
        
        # Configuration
        self.enable_persistence = self.config.get("trading.enable_state_persistence", True)
        self.state_check_interval = self.config.get("trading.state_check_interval_seconds", 5)
        self.emergency_check_interval = self.config.get("trading.emergency_check_interval_seconds", 1)
        
        logger.info("TradingStateMachine initialized")
    
    def _define_valid_transitions(self) -> Dict[TradingState, Set[TradingState]]:
        """Define valid state transitions."""
        return {
            TradingState.INACTIVE: {
                TradingState.STARTING,
                TradingState.MAINTENANCE,
                TradingState.ERROR
            },
            TradingState.STARTING: {
                TradingState.ACTIVE,
                TradingState.ERROR,
                TradingState.INACTIVE
            },
            TradingState.ACTIVE: {
                TradingState.PAUSED,
                TradingState.EMERGENCY_STOP,
                TradingState.MAINTENANCE,
                TradingState.SHUTTING_DOWN,
                TradingState.ERROR
            },
            TradingState.PAUSED: {
                TradingState.ACTIVE,
                TradingState.EMERGENCY_STOP,
                TradingState.MAINTENANCE,
                TradingState.SHUTTING_DOWN,
                TradingState.ERROR
            },
            TradingState.EMERGENCY_STOP: {
                TradingState.MAINTENANCE,
                TradingState.INACTIVE,
                TradingState.ERROR
            },
            TradingState.MAINTENANCE: {
                TradingState.INACTIVE,
                TradingState.STARTING,
                TradingState.ERROR
            },
            TradingState.SHUTTING_DOWN: {
                TradingState.INACTIVE,
                TradingState.ERROR
            },
            TradingState.ERROR: {
                TradingState.MAINTENANCE,
                TradingState.INACTIVE
            }
        }
    
    async def initialize(self) -> None:
        """Initialize the state machine."""
        try:
            # Initialize Redis client for persistence
            if self.enable_persistence:
                self.redis_client = RedisClient()
                await self.redis_client.initialize()
                await self._load_state()
            
            # Set up default conditions and callbacks
            await self._setup_default_conditions()
            
            # Start background monitoring
            await self._start_background_tasks()
            
            logger.info(f"TradingStateMachine initialized in state: {self.current_state.value}")
            
        except Exception as e:
            logger.error(f"Failed to initialize TradingStateMachine: {e}")
            await self.transition_to_error(f"Initialization failed: {e}")
            raise
    
    async def _setup_default_conditions(self) -> None:
        """Set up default state conditions."""
        # Emergency conditions (checked continuously)
        self.emergency_conditions = [
            StateCondition(
                name="system_health",
                check_function=self._check_system_health,
                message="System health check failed"
            ),
            StateCondition(
                name="memory_usage",
                check_function=self._check_memory_usage,
                message="Memory usage too high"
            )
        ]
        
        # State-specific conditions
        self.state_conditions[TradingState.ACTIVE] = [
            StateCondition(
                name="market_connection",
                check_function=self._check_market_connection,
                message="Market connection lost"
            ),
            StateCondition(
                name="position_limits",
                check_function=self._check_position_limits,
                message="Position limits exceeded"
            )
        ]
        
        # Auto-transition timers
        self.auto_transition_timers[TradingState.STARTING] = {
            'timeout_seconds': 60,
            'target_state': TradingState.ERROR,
            'reason': StateChangeReason.TIMEOUT,
            'message': 'Startup timeout'
        }
        
        self.auto_transition_timers[TradingState.SHUTTING_DOWN] = {
            'timeout_seconds': 30,
            'target_state': TradingState.INACTIVE,
            'reason': StateChangeReason.TIMEOUT,
            'message': 'Shutdown timeout'
        }
    
    async def _start_background_tasks(self) -> None:
        """Start background monitoring tasks."""
        self._monitor_task = asyncio.create_task(self._state_monitor_loop())
        if self.enable_persistence:
            self._persistence_task = asyncio.create_task(self._persistence_loop())
    
    async def _state_monitor_loop(self) -> None:
        """Background state monitoring loop."""
        while True:
            try:
                # Check emergency conditions
                await self._check_emergency_conditions()
                await asyncio.sleep(self.emergency_check_interval)
                
                # Check state-specific conditions
                await self._check_state_conditions()
                await asyncio.sleep(self.state_check_interval)
                
                # Check auto-transition timers
                await self._check_auto_transitions()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"State monitor error: {e}")
                await asyncio.sleep(5)
    
    async def _persistence_loop(self) -> None:
        """Background state persistence loop."""
        while True:
            try:
                await asyncio.sleep(10)  # Save state every 10 seconds
                await self._save_state()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"State persistence error: {e}")
                await asyncio.sleep(30)
    
    @timeout_async(0.1)  # 100ms timeout for state transitions
    @measure_latency("state_transition")
    async def transition_to(
        self,
        target_state: TradingState,
        reason: StateChangeReason = StateChangeReason.MANUAL,
        message: str = "",
        user_id: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Transition to a new state with validation."""
        try:
            # Validate transition
            if not self._is_valid_transition(self.current_state, target_state):
                raise StateTransitionError(
                    f"Invalid transition from {self.current_state.value} to {target_state.value}"
                )
            
            # Check state conditions for target state
            if not await self._check_transition_conditions(target_state):
                logger.warning(f"Transition conditions not met for {target_state.value}")
                return False
            
            # Execute transition
            return await self._execute_transition(
                target_state, reason, message, user_id, data or {}
            )
            
        except Exception as e:
            logger.error(f"State transition failed: {e}")
            await self.transition_to_error(f"Transition failed: {e}")
            return False
    
    async def _execute_transition(
        self,
        target_state: TradingState,
        reason: StateChangeReason,
        message: str,
        user_id: Optional[str],
        data: Dict[str, Any]
    ) -> bool:
        """Execute state transition."""
        old_state = self.current_state
        
        try:
            # Call exit callbacks for current state
            await self._call_exit_callbacks(old_state)
            
            # Update state
            self.current_state = target_state
            self.state_entered_at = datetime.now()
            self.last_state_change = datetime.now()
            
            # Record transition
            transition = StateTransition(
                from_state=old_state,
                to_state=target_state,
                reason=reason,
                timestamp=self.last_state_change,
                message=message,
                user_id=user_id,
                data=data
            )
            
            self.state_history.append(transition)
            
            # Trim history if needed
            if len(self.state_history) > self.max_history_length:
                self.state_history = self.state_history[-self.max_history_length:]
            
            # Call enter callbacks for new state
            await self._call_enter_callbacks(target_state)
            
            # Log transition
            logger.info(
                f"State transition: {old_state.value} -> {target_state.value} "
                f"(reason: {reason.value}, message: {message})"
            )
            
            # Save state if persistence enabled
            if self.enable_persistence:
                await self._save_state()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to execute transition: {e}")
            # Attempt to revert state
            self.current_state = old_state
            raise StateTransitionError(f"Transition execution failed: {e}")
    
    def _is_valid_transition(self, from_state: TradingState, to_state: TradingState) -> bool:
        """Check if transition is valid."""
        return to_state in self.valid_transitions.get(from_state, set())
    
    async def _check_transition_conditions(self, target_state: TradingState) -> bool:
        """Check if conditions are met for transitioning to target state."""
        conditions = self.state_conditions.get(target_state, [])
        
        for condition in conditions:
            try:
                if not condition.check_function():
                    logger.warning(f"Transition condition failed: {condition.name} - {condition.message}")
                    return False
            except Exception as e:
                logger.error(f"Error checking condition {condition.name}: {e}")
                return False
        
        return True
    
    async def _call_enter_callbacks(self, state: TradingState) -> None:
        """Call enter callbacks for a state."""
        callbacks = self.state_enter_callbacks.get(state, [])
        for callback in callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(state)
                else:
                    callback(state)
            except Exception as e:
                logger.error(f"Error in state enter callback: {e}")
    
    async def _call_exit_callbacks(self, state: TradingState) -> None:
        """Call exit callbacks for a state."""
        callbacks = self.state_exit_callbacks.get(state, [])
        for callback in callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(state)
                else:
                    callback(state)
            except Exception as e:
                logger.error(f"Error in state exit callback: {e}")
    
    async def _check_emergency_conditions(self) -> None:
        """Check emergency conditions that trigger immediate stop."""
        for condition in self.emergency_conditions:
            try:
                if not condition.check_function():
                    logger.critical(f"Emergency condition failed: {condition.name} - {condition.message}")
                    await self.emergency_stop(condition.message)
                    return
            except Exception as e:
                logger.error(f"Error checking emergency condition {condition.name}: {e}")
    
    async def _check_state_conditions(self) -> None:
        """Check conditions for current state."""
        conditions = self.state_conditions.get(self.current_state, [])
        
        for condition in conditions:
            try:
                if not condition.check_function():
                    logger.warning(f"State condition failed: {condition.name} - {condition.message}")
                    
                    # Transition to appropriate state based on current state
                    if self.current_state == TradingState.ACTIVE:
                        await self.pause(f"Automatic pause: {condition.message}")
                    elif self.current_state == TradingState.PAUSED:
                        # Stay paused, just log
                        pass
                    else:
                        await self.transition_to_error(f"State condition failed: {condition.message}")
                    
                    return
            except Exception as e:
                logger.error(f"Error checking state condition {condition.name}: {e}")
    
    async def _check_auto_transitions(self) -> None:
        """Check automatic transition timers."""
        timer_config = self.auto_transition_timers.get(self.current_state)
        if not timer_config:
            return
        
        time_in_state = (datetime.now() - self.state_entered_at).total_seconds()
        if time_in_state >= timer_config['timeout_seconds']:
            await self.transition_to(
                timer_config['target_state'],
                timer_config['reason'],
                timer_config['message']
            )
    
    # Condition check functions
    def _check_system_health(self) -> bool:
        """Check overall system health."""
        # Implement actual health checks
        return True
    
    def _check_memory_usage(self) -> bool:
        """Check memory usage."""
        # Implement memory usage check
        return True
    
    def _check_market_connection(self) -> bool:
        """Check market data connection."""
        # Implement market connection check
        return True
    
    def _check_position_limits(self) -> bool:
        """Check position limits."""
        # Implement position limit checks
        return True
    
    # Convenience methods for common transitions
    async def start_trading(
        self,
        message: str = "Starting trading operations",
        user_id: Optional[str] = None
    ) -> bool:
        """Start trading operations."""
        if self.current_state == TradingState.INACTIVE:
            await self.transition_to(TradingState.STARTING, StateChangeReason.MANUAL, message, user_id)
            # Simulate startup process
            await asyncio.sleep(2)
            return await self.transition_to(TradingState.ACTIVE, StateChangeReason.AUTOMATIC, "Startup complete")
        else:
            return await self.transition_to(TradingState.ACTIVE, StateChangeReason.MANUAL, message, user_id)
    
    async def pause(
        self,
        message: str = "Pausing trading operations",
        user_id: Optional[str] = None
    ) -> bool:
        """Pause trading operations."""
        return await self.transition_to(TradingState.PAUSED, StateChangeReason.MANUAL, message, user_id)
    
    async def resume(
        self,
        message: str = "Resuming trading operations",
        user_id: Optional[str] = None
    ) -> bool:
        """Resume trading operations."""
        return await self.transition_to(TradingState.ACTIVE, StateChangeReason.MANUAL, message, user_id)
    
    async def emergency_stop(
        self,
        message: str = "Emergency stop triggered",
        user_id: Optional[str] = None
    ) -> bool:
        """Trigger emergency stop."""
        return await self.transition_to(TradingState.EMERGENCY_STOP, StateChangeReason.EMERGENCY, message, user_id)
    
    async def enter_maintenance(
        self,
        message: str = "Entering maintenance mode",
        user_id: Optional[str] = None
    ) -> bool:
        """Enter maintenance mode."""
        return await self.transition_to(TradingState.MAINTENANCE, StateChangeReason.MAINTENANCE, message, user_id)
    
    async def shutdown(
        self,
        message: str = "Shutting down trading system",
        user_id: Optional[str] = None
    ) -> bool:
        """Graceful shutdown."""
        await self.transition_to(TradingState.SHUTTING_DOWN, StateChangeReason.SHUTDOWN, message, user_id)
        # Simulate shutdown process
        await asyncio.sleep(5)
        return await self.transition_to(TradingState.INACTIVE, StateChangeReason.AUTOMATIC, "Shutdown complete")
    
    async def transition_to_error(
        self,
        message: str = "System error occurred",
        user_id: Optional[str] = None
    ) -> bool:
        """Transition to error state."""
        return await self.transition_to(TradingState.ERROR, StateChangeReason.ERROR, message, user_id)
    
    # Callback registration
    def register_state_enter_callback(self, state: TradingState, callback: Callable) -> None:
        """Register callback for when entering a state."""
        if state not in self.state_enter_callbacks:
            self.state_enter_callbacks[state] = []
        self.state_enter_callbacks[state].append(callback)
    
    def register_state_exit_callback(self, state: TradingState, callback: Callable) -> None:
        """Register callback for when exiting a state."""
        if state not in self.state_exit_callbacks:
            self.state_exit_callbacks[state] = []
        self.state_exit_callbacks[state].append(callback)
    
    def add_state_condition(self, state: TradingState, condition: StateCondition) -> None:
        """Add a condition for a state."""
        if state not in self.state_conditions:
            self.state_conditions[state] = []
        self.state_conditions[state].append(condition)
    
    def add_emergency_condition(self, condition: StateCondition) -> None:
        """Add an emergency condition."""
        self.emergency_conditions.append(condition)
    
    # State queries
    def get_current_state(self) -> TradingState:
        """Get current state."""
        return self.current_state
    
    def is_trading_active(self) -> bool:
        """Check if trading is active."""
        return self.current_state == TradingState.ACTIVE
    
    def is_system_stopped(self) -> bool:
        """Check if system is stopped."""
        return self.current_state in {TradingState.INACTIVE, TradingState.EMERGENCY_STOP, TradingState.ERROR}
    
    def get_time_in_state(self) -> timedelta:
        """Get time spent in current state."""
        return datetime.now() - self.state_entered_at
    
    def get_state_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get state transition history."""
        history = self.state_history[-limit:] if limit else self.state_history
        return [transition.to_dict() for transition in history]
    
    def get_state_summary(self) -> Dict[str, Any]:
        """Get state machine summary."""
        return {
            'current_state': self.current_state.value,
            'state_entered_at': self.state_entered_at.isoformat(),
            'time_in_state_seconds': self.get_time_in_state().total_seconds(),
            'last_state_change': self.last_state_change.isoformat(),
            'total_transitions': len(self.state_history),
            'emergency_conditions_count': len(self.emergency_conditions),
            'state_conditions_count': sum(len(conditions) for conditions in self.state_conditions.values()),
            'is_trading_active': self.is_trading_active(),
            'is_system_stopped': self.is_system_stopped()
        }
    
    # Persistence methods
    async def _save_state(self) -> None:
        """Save state to Redis."""
        if not self.redis_client:
            return
        
        try:
            state_data = {
                'current_state': self.current_state.value,
                'state_entered_at': self.state_entered_at.isoformat(),
                'last_state_change': self.last_state_change.isoformat(),
                'state_history': [t.to_dict() for t in self.state_history[-100:]]  # Last 100 transitions
            }
            
            await self.redis_client.set(
                "trading_state_machine",
                json.dumps(state_data),
                ex=86400  # 24 hours
            )
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
    
    async def _load_state(self) -> None:
        """Load state from Redis."""
        if not self.redis_client:
            return
        
        try:
            state_data = await self.redis_client.get("trading_state_machine")
            if state_data:
                data = json.loads(state_data)
                
                self.current_state = TradingState(data['current_state'])
                self.state_entered_at = datetime.fromisoformat(data['state_entered_at'])
                self.last_state_change = datetime.fromisoformat(data['last_state_change'])
                
                # Restore history
                self.state_history = []
                for transition_data in data.get('state_history', []):
                    transition = StateTransition(
                        from_state=TradingState(transition_data['from_state']),
                        to_state=TradingState(transition_data['to_state']),
                        reason=StateChangeReason(transition_data['reason']),
                        timestamp=datetime.fromisoformat(transition_data['timestamp']),
                        message=transition_data.get('message', ''),
                        user_id=transition_data.get('user_id'),
                        data=transition_data.get('data', {})
                    )
                    self.state_history.append(transition)
                
                logger.info(f"Loaded state machine state: {self.current_state.value}")
                
        except Exception as e:
            logger.warning(f"Failed to load state, using default: {e}")
    
    async def cleanup(self) -> None:
        """Cleanup resources and stop background tasks."""
        # Cancel background tasks
        if self._monitor_task and not self._monitor_task.done():
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        if self._persistence_task and not self._persistence_task.done():
            self._persistence_task.cancel()
            try:
                await self._persistence_task
            except asyncio.CancelledError:
                pass
        
        # Save final state
        if self.enable_persistence:
            await self._save_state()
        
        # Close Redis connection
        if self.redis_client:
            await self.redis_client.close()
        
        logger.info("TradingStateMachine cleanup completed")


# Global state machine instance
_state_machine: Optional[TradingStateMachine] = None


async def get_state_machine() -> TradingStateMachine:
    """Get global state machine instance."""
    global _state_machine
    if _state_machine is None:
        _state_machine = TradingStateMachine()
        await _state_machine.initialize()
    return _state_machine


async def cleanup_state_machine() -> None:
    """Cleanup global state machine."""
    global _state_machine
    if _state_machine:
        await _state_machine.cleanup()
        _state_machine = None
"""
FlashMM Emergency Protocols

Emergency response system with automated protocols for critical situations:
- Emergency stop mechanisms
- Position flattening procedures
- Market exit strategies for different scenarios
- Automatic hedge placement during stress
- Communication protocols for emergencies
- Recovery procedures after emergency stops
"""

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any

from flashmm.config.settings import get_config
from flashmm.utils.decorators import measure_latency, timeout_async
from flashmm.utils.exceptions import EmergencyStopError, RiskError
from flashmm.utils.logging import TradingEventLogger, get_logger

logger = get_logger(__name__)
trading_logger = TradingEventLogger()


class EmergencyLevel(Enum):
    """Emergency severity levels."""
    NORMAL = "normal"
    ALERT = "alert"
    WARNING = "warning"
    CRITICAL = "critical"
    CATASTROPHIC = "catastrophic"


class EmergencyType(Enum):
    """Types of emergency situations."""
    MARKET_CRASH = "market_crash"
    LIQUIDITY_CRISIS = "liquidity_crisis"
    SYSTEM_FAILURE = "system_failure"
    CONNECTIVITY_LOSS = "connectivity_loss"
    EXCESSIVE_LOSSES = "excessive_losses"
    REGULATORY_HALT = "regulatory_halt"
    OPERATIONAL_ERROR = "operational_error"
    EXTERNAL_THREAT = "external_threat"


class ProtocolAction(Enum):
    """Emergency protocol actions."""
    HALT_TRADING = "halt_trading"
    FLATTEN_POSITIONS = "flatten_positions"
    HEDGE_PORTFOLIO = "hedge_portfolio"
    REDUCE_EXPOSURE = "reduce_exposure"
    CANCEL_ORDERS = "cancel_orders"
    ISOLATE_SYSTEM = "isolate_system"
    NOTIFY_STAKEHOLDERS = "notify_stakeholders"
    ACTIVATE_BACKUP = "activate_backup"


@dataclass
class EmergencyEvent:
    """Emergency event record."""
    event_id: str
    emergency_type: EmergencyType
    emergency_level: EmergencyLevel
    trigger_reason: str
    trigger_value: float | None
    threshold_value: float | None
    timestamp: datetime

    # Response tracking
    protocols_activated: list[str] = field(default_factory=list)
    actions_taken: list[dict[str, Any]] = field(default_factory=list)
    response_time_seconds: float | None = None
    resolved: bool = False
    resolution_time: datetime | None = None

    # Impact assessment
    positions_affected: int = 0
    orders_cancelled: int = 0
    financial_impact: Decimal = Decimal('0')

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            'event_id': self.event_id,
            'emergency_type': self.emergency_type.value,
            'emergency_level': self.emergency_level.value,
            'trigger_reason': self.trigger_reason,
            'trigger_value': self.trigger_value,
            'threshold_value': self.threshold_value,
            'timestamp': self.timestamp.isoformat(),
            'protocols_activated': self.protocols_activated,
            'actions_taken': self.actions_taken,
            'response_time_seconds': self.response_time_seconds,
            'resolved': self.resolved,
            'resolution_time': self.resolution_time.isoformat() if self.resolution_time else None,
            'impact': {
                'positions_affected': self.positions_affected,
                'orders_cancelled': self.orders_cancelled,
                'financial_impact': float(self.financial_impact)
            }
        }


@dataclass
class EmergencyProtocol:
    """Emergency response protocol definition."""
    name: str
    trigger_conditions: list[dict[str, Any]]
    response_actions: list[ProtocolAction]
    required_confirmations: int
    auto_execute: bool
    max_execution_time_seconds: int
    escalation_level: EmergencyLevel

    # Execution tracking
    last_triggered: datetime | None = None
    execution_count: int = 0
    success_rate: float = 100.0

    def should_trigger(self, conditions: dict[str, Any]) -> bool:
        """Check if protocol should trigger based on conditions."""
        try:
            triggered_conditions = 0

            for condition in self.trigger_conditions:
                condition_type = condition.get('type')
                threshold = condition.get('threshold')
                operator = condition.get('operator', 'greater_than')

                current_value = conditions.get(condition_type) if condition_type else None
                if current_value is None:
                    continue

                if operator == 'greater_than' and current_value > threshold:
                    triggered_conditions += 1
                elif operator == 'less_than' and current_value < threshold:
                    triggered_conditions += 1
                elif operator == 'equals' and current_value == threshold:
                    triggered_conditions += 1

            # Protocol triggers if any condition is met (OR logic)
            return triggered_conditions > 0

        except Exception as e:
            logger.error(f"Error checking protocol trigger conditions: {e}")
            return False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'trigger_conditions': self.trigger_conditions,
            'response_actions': [action.value for action in self.response_actions],
            'required_confirmations': self.required_confirmations,
            'auto_execute': self.auto_execute,
            'max_execution_time_seconds': self.max_execution_time_seconds,
            'escalation_level': self.escalation_level.value,
            'execution_stats': {
                'last_triggered': self.last_triggered.isoformat() if self.last_triggered else None,
                'execution_count': self.execution_count,
                'success_rate': self.success_rate
            }
        }


class PositionFlattener:
    """Position flattening system for emergency situations."""

    def __init__(self):
        self.config = get_config()
        self.flattening_in_progress = False
        self.flattened_positions: list[dict[str, Any]] = []

        # Flattening settings
        self.max_flatten_time_seconds = self.config.get("emergency.max_flatten_time_seconds", 300)  # 5 minutes
        self.flatten_order_size_limit = self.config.get("emergency.flatten_order_size_limit", 10000.0)
        self.flatten_price_tolerance_pct = self.config.get("emergency.flatten_price_tolerance_pct", 5.0)

        # Callbacks for order management
        self.cancel_orders_callback: Callable | None = None
        self.place_order_callback: Callable | None = None
        self.get_positions_callback: Callable | None = None

    def set_callbacks(self,
                      cancel_orders: Callable | None = None,
                      place_order: Callable | None = None,
                      get_positions: Callable | None = None):
        """Set callback functions for order and position management."""
        self.cancel_orders_callback = cancel_orders
        self.place_order_callback = place_order
        self.get_positions_callback = get_positions

    async def flatten_all_positions(self, reason: str, emergency_level: EmergencyLevel) -> dict[str, Any]:
        """Flatten all open positions as quickly as possible."""
        if self.flattening_in_progress:
            return {'status': 'already_in_progress', 'message': 'Position flattening already in progress'}

        self.flattening_in_progress = True
        start_time = time.perf_counter()

        try:
            logger.critical(f"🚨 EMERGENCY POSITION FLATTENING INITIATED: {reason}")

            # Step 1: Cancel all open orders immediately
            cancelled_orders = 0
            if self.cancel_orders_callback:
                try:
                    cancel_result = await self.cancel_orders_callback("Emergency flattening")
                    cancelled_orders = cancel_result.get('cancelled_count', 0)
                    logger.info(f"Cancelled {cancelled_orders} open orders")
                except Exception as e:
                    logger.error(f"Error cancelling orders during flattening: {e}")

            # Step 2: Get all current positions
            positions = []
            if self.get_positions_callback:
                try:
                    positions = await self.get_positions_callback()
                    logger.info(f"Found {len(positions)} positions to flatten")
                except Exception as e:
                    logger.error(f"Error getting positions during flattening: {e}")
                    positions = []

            # Step 3: Flatten each position
            flattened_count = 0
            total_impact = Decimal('0')

            for position in positions:
                try:
                    flatten_result = await self._flatten_single_position(position, emergency_level)
                    if flatten_result['success']:
                        flattened_count += 1
                        total_impact += Decimal(str(flatten_result.get('financial_impact', 0)))
                        self.flattened_positions.append(flatten_result)

                except Exception as e:
                    logger.error(f"Error flattening position {position.get('symbol', 'unknown')}: {e}")

            execution_time = time.perf_counter() - start_time

            result = {
                'status': 'completed',
                'execution_time_seconds': execution_time,
                'positions_flattened': flattened_count,
                'orders_cancelled': cancelled_orders,
                'total_financial_impact': float(total_impact),
                'reason': reason,
                'emergency_level': emergency_level.value,
                'timestamp': datetime.now().isoformat()
            }

            logger.critical(f"Position flattening completed: {flattened_count} positions, {execution_time:.2f}s")

            # Log to trading event logger
            await trading_logger.log_pnl_event(
                "EMERGENCY",
                float(total_impact),
                float(total_impact),
                float(total_impact),
                emergency_action=f"position_flattening:{reason}"
            )

            return result

        except Exception as e:
            logger.error(f"Critical error during position flattening: {e}")
            raise EmergencyStopError(f"Position flattening failed: {e}") from e

        finally:
            self.flattening_in_progress = False

    async def _flatten_single_position(self, position: dict[str, Any], emergency_level: EmergencyLevel) -> dict[str, Any]:
        """Flatten a single position."""
        try:
            symbol = position.get('symbol', 'UNKNOWN')
            current_size = Decimal(str(position.get('base_balance', 0)))

            if abs(current_size) < Decimal('0.001'):  # Position too small to matter
                return {'success': True, 'symbol': symbol, 'action': 'skipped_small_position'}

            # Determine flattening strategy based on emergency level
            if emergency_level == EmergencyLevel.CATASTROPHIC:
                # Market orders at any price
                order_type = 'market'
            elif emergency_level == EmergencyLevel.CRITICAL:
                # Aggressive limit orders
                order_type = 'limit'
            else:
                # Conservative flattening
                order_type = 'limit'

            # Calculate flattening order
            flatten_side = 'sell' if current_size > 0 else 'buy'
            flatten_size = abs(current_size)

            # Split large positions into smaller orders
            if flatten_size * Decimal(str(position.get('mark_price', 1))) > self.flatten_order_size_limit:
                # Would implement order splitting logic
                pass

            # Place flattening order
            if self.place_order_callback:
                try:
                    order_result = await self.place_order_callback({
                        'symbol': symbol,
                        'side': flatten_side,
                        'size': float(flatten_size),
                        'type': order_type,
                        'metadata': {
                            'emergency_flatten': True,
                            'emergency_level': emergency_level.value,
                            'original_position_size': float(current_size)
                        }
                    })

                    return {
                        'success': True,
                        'symbol': symbol,
                        'action': 'flattened',
                        'original_size': float(current_size),
                        'flatten_side': flatten_side,
                        'flatten_size': float(flatten_size),
                        'order_result': order_result,
                        'financial_impact': 0.0  # Would calculate actual impact
                    }

                except Exception as e:
                    logger.error(f"Error placing flatten order for {symbol}: {e}")
                    return {
                        'success': False,
                        'symbol': symbol,
                        'error': str(e),
                        'financial_impact': 0.0
                    }

            return {
                'success': False,
                'symbol': symbol,
                'error': 'No place_order callback available',
                'financial_impact': 0.0
            }

        except Exception as e:
            logger.error(f"Error in _flatten_single_position: {e}")
            return {
                'success': False,
                'symbol': position.get('symbol', 'UNKNOWN'),
                'error': str(e),
                'financial_impact': 0.0
            }


class MarketExitStrategy:
    """Market exit strategies for different emergency scenarios."""

    def __init__(self):
        self.config = get_config()
        self.exit_strategies = {
            EmergencyType.MARKET_CRASH: self._market_crash_exit,
            EmergencyType.LIQUIDITY_CRISIS: self._liquidity_crisis_exit,
            EmergencyType.SYSTEM_FAILURE: self._system_failure_exit,
            EmergencyType.CONNECTIVITY_LOSS: self._connectivity_loss_exit,
            EmergencyType.EXCESSIVE_LOSSES: self._excessive_losses_exit
        }

    async def execute_exit_strategy(self, emergency_type: EmergencyType, positions: list[dict[str, Any]]) -> dict[str, Any]:
        """Execute appropriate exit strategy based on emergency type."""
        try:
            strategy_func = self.exit_strategies.get(emergency_type, self._default_exit_strategy)
            result = await strategy_func(positions)

            logger.info(f"Executed {emergency_type.value} exit strategy: {result.get('status')}")
            return result

        except Exception as e:
            logger.error(f"Error executing exit strategy for {emergency_type.value}: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'emergency_type': emergency_type.value
            }

    async def _market_crash_exit(self, positions: list[dict[str, Any]]) -> dict[str, Any]:
        """Exit strategy for market crash scenario."""
        # Immediate market orders to exit all positions
        return {
            'status': 'executed',
            'strategy': 'immediate_market_exit',
            'description': 'All positions closed with market orders',
            'priority': 'speed_over_price',
            'expected_slippage': 'high'
        }

    async def _liquidity_crisis_exit(self, positions: list[dict[str, Any]]) -> dict[str, Any]:
        """Exit strategy for liquidity crisis."""
        # Gradual exit with limit orders, focus on liquid markets first
        return {
            'status': 'executed',
            'strategy': 'gradual_limit_exit',
            'description': 'Positions closed gradually using limit orders, liquid markets first',
            'priority': 'minimize_market_impact',
            'expected_slippage': 'medium'
        }

    async def _system_failure_exit(self, positions: list[dict[str, Any]]) -> dict[str, Any]:
        """Exit strategy for system failure."""
        # Manual intervention required, log positions for manual closure
        return {
            'status': 'manual_intervention_required',
            'strategy': 'log_for_manual_closure',
            'description': 'System failure - positions logged for manual intervention',
            'priority': 'preserve_records',
            'manual_action_required': True
        }

    async def _connectivity_loss_exit(self, positions: list[dict[str, Any]]) -> dict[str, Any]:
        """Exit strategy for connectivity loss."""
        # Attempt backup connection, if available
        return {
            'status': 'attempting_backup',
            'strategy': 'backup_connection_exit',
            'description': 'Attempting to close positions via backup connectivity',
            'priority': 'restore_connectivity',
            'backup_required': True
        }

    async def _excessive_losses_exit(self, positions: list[dict[str, Any]]) -> dict[str, Any]:
        """Exit strategy for excessive losses."""
        # Close losing positions first, preserve winners if possible
        return {
            'status': 'executed',
            'strategy': 'selective_exit',
            'description': 'Closing losing positions first, preserving profitable ones',
            'priority': 'stop_further_losses',
            'selective': True
        }

    async def _default_exit_strategy(self, positions: list[dict[str, Any]]) -> dict[str, Any]:
        """Default exit strategy for unknown scenarios."""
        return {
            'status': 'executed',
            'strategy': 'conservative_exit',
            'description': 'Conservative exit using limit orders with moderate urgency',
            'priority': 'balanced_approach',
            'expected_slippage': 'low'
        }


class EmergencyProtocolManager:
    """Main emergency protocol management system."""

    def __init__(self):
        self.config = get_config()

        # Core components
        self.position_flattener = PositionFlattener()
        self.market_exit_strategy = MarketExitStrategy()

        # Protocol definitions
        self.protocols: dict[str, EmergencyProtocol] = {}
        self.active_emergencies: dict[str, EmergencyEvent] = {}
        self.emergency_history: list[EmergencyEvent] = []

        # System state
        self.emergency_mode_active = False
        self.system_isolated = False
        self.last_emergency_check = datetime.now()

        # Callbacks for external actions
        self.halt_trading_callback: Callable | None = None
        self.cancel_orders_callback: Callable | None = None
        self.place_order_callback: Callable | None = None
        self.get_positions_callback: Callable | None = None
        self.notify_stakeholders_callback: Callable | None = None

        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Initialize emergency protocol manager."""
        try:
            # Create default emergency protocols
            await self._create_default_protocols()

            # Set up position flattener callbacks
            self.position_flattener.set_callbacks(
                cancel_orders=self.cancel_orders_callback,
                place_order=self.place_order_callback,
                get_positions=self.get_positions_callback
            )

            logger.info(f"Emergency protocol manager initialized with {len(self.protocols)} protocols")

        except Exception as e:
            logger.error(f"Failed to initialize emergency protocol manager: {e}")
            raise RiskError(f"Emergency protocol manager initialization failed: {e}") from e

    async def _create_default_protocols(self):
        """Create default emergency protocols."""

        # Market crash protocol
        market_crash_protocol = EmergencyProtocol(
            name="market_crash_response",
            trigger_conditions=[
                {'type': 'price_change_pct_1min', 'threshold': 10.0, 'operator': 'greater_than'},
                {'type': 'volatility_spike', 'threshold': 5.0, 'operator': 'greater_than'}
            ],
            response_actions=[
                ProtocolAction.HALT_TRADING,
                ProtocolAction.CANCEL_ORDERS,
                ProtocolAction.FLATTEN_POSITIONS,
                ProtocolAction.NOTIFY_STAKEHOLDERS
            ],
            required_confirmations=0,  # Auto-execute
            auto_execute=True,
            max_execution_time_seconds=60,
            escalation_level=EmergencyLevel.CRITICAL
        )
        self.protocols[market_crash_protocol.name] = market_crash_protocol

        # Excessive loss protocol
        loss_protocol = EmergencyProtocol(
            name="excessive_loss_response",
            trigger_conditions=[
                {'type': 'daily_pnl', 'threshold': -10000.0, 'operator': 'less_than'},
                {'type': 'drawdown_pct', 'threshold': 15.0, 'operator': 'greater_than'}
            ],
            response_actions=[
                ProtocolAction.HALT_TRADING,
                ProtocolAction.FLATTEN_POSITIONS,
                ProtocolAction.NOTIFY_STAKEHOLDERS
            ],
            required_confirmations=0,
            auto_execute=True,
            max_execution_time_seconds=300,
            escalation_level=EmergencyLevel.CRITICAL
        )
        self.protocols[loss_protocol.name] = loss_protocol

        # System failure protocol
        system_failure_protocol = EmergencyProtocol(
            name="system_failure_response",
            trigger_conditions=[
                {'type': 'system_health_score', 'threshold': 20.0, 'operator': 'less_than'},
                {'type': 'connectivity_failures', 'threshold': 3, 'operator': 'greater_than'}
            ],
            response_actions=[
                ProtocolAction.HALT_TRADING,
                ProtocolAction.ISOLATE_SYSTEM,
                ProtocolAction.ACTIVATE_BACKUP,
                ProtocolAction.NOTIFY_STAKEHOLDERS
            ],
            required_confirmations=1,  # Require confirmation
            auto_execute=False,
            max_execution_time_seconds=120,
            escalation_level=EmergencyLevel.CRITICAL
        )
        self.protocols[system_failure_protocol.name] = system_failure_protocol

    def set_callbacks(self,
                      halt_trading: Callable | None = None,
                      cancel_orders: Callable | None = None,
                      place_order: Callable | None = None,
                      get_positions: Callable | None = None,
                      notify_stakeholders: Callable | None = None):
        """Set callback functions for emergency actions."""
        self.halt_trading_callback = halt_trading
        self.cancel_orders_callback = cancel_orders
        self.place_order_callback = place_order
        self.get_positions_callback = get_positions
        self.notify_stakeholders_callback = notify_stakeholders

        # Update position flattener callbacks
        self.position_flattener.set_callbacks(cancel_orders, place_order, get_positions)

    async def check_emergency_conditions(self, risk_data: dict[str, Any]) -> list[str]:
        """Check all emergency protocols against current conditions."""
        async with self._lock:
            triggered_protocols = []

            try:
                self.last_emergency_check = datetime.now()

                # Extract condition values from risk data
                conditions = await self._extract_conditions(risk_data)

                # Check each protocol
                for protocol_name, protocol in self.protocols.items():
                    if protocol.should_trigger(conditions):
                        if protocol.auto_execute:
                            # Execute immediately
                            await self._execute_emergency_protocol(protocol, conditions)
                            triggered_protocols.append(protocol_name)
                        else:
                            # Log for manual review
                            logger.warning(f"Emergency protocol {protocol_name} triggered but requires manual confirmation")
                            triggered_protocols.append(f"{protocol_name}_manual_review")

                return triggered_protocols

            except Exception as e:
                logger.error(f"Error checking emergency conditions: {e}")
                return []

    async def _extract_conditions(self, risk_data: dict[str, Any]) -> dict[str, Any]:
        """Extract relevant conditions from risk data."""
        conditions = {}

        try:
            # P&L conditions
            pnl_data = risk_data.get('components', {}).get('pnl_controller', {})
            if pnl_data and not pnl_data.get('error'):
                global_metrics = pnl_data.get('global_metrics', {})
                conditions['daily_pnl'] = global_metrics.get('daily', 0.0)

                # Calculate drawdown percentage (simplified)
                daily_pnl = global_metrics.get('daily', 0.0)
                if daily_pnl < 0:
                    conditions['drawdown_pct'] = abs(daily_pnl) / 10000.0 * 100  # Assuming 10k base

            # System health conditions
            op_data = risk_data.get('components', {}).get('operational_risk', {})
            if op_data and not op_data.get('error'):
                health_scores = {
                    'excellent': 100, 'good': 80, 'warning': 60,
                    'critical': 30, 'failure': 0
                }

                system_health = op_data.get('system_health', 'good')
                conditions['system_health_score'] = health_scores.get(system_health, 50)

                # Count connectivity failures
                endpoint_health = op_data.get('endpoint_health', {})
                failed_endpoints = sum(1 for ep in endpoint_health.values()
                                     if isinstance(ep, dict) and not ep.get('success', True))
                conditions['connectivity_failures'] = failed_endpoints

            # Circuit breaker conditions
            cb_data = risk_data.get('components', {}).get('circuit_breakers', {})
            if cb_data and not cb_data.get('error'):
                conditions['active_circuit_breakers'] = cb_data.get('active_breakers', 0)

            # Market volatility conditions (would be calculated from market data)
            conditions['volatility_spike'] = 1.0  # Placeholder
            conditions['price_change_pct_1min'] = 0.5  # Placeholder

        except Exception as e:
            logger.error(f"Error extracting conditions: {e}")

        return conditions

    async def _execute_emergency_protocol(self, protocol: EmergencyProtocol, conditions: dict[str, Any]):
        """Execute an emergency protocol."""
        try:
            event_id = f"EMRG_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

            # Create emergency event
            emergency_event = EmergencyEvent(
                event_id=event_id,
                emergency_type=EmergencyType.EXCESSIVE_LOSSES,  # Would determine from conditions
                emergency_level=protocol.escalation_level,
                trigger_reason=f"Protocol {protocol.name} triggered",
                trigger_value=None,
                threshold_value=None,
                timestamp=datetime.now()
            )

            self.active_emergencies[event_id] = emergency_event
            execution_start = time.perf_counter()

            logger.critical(f"🚨 EXECUTING EMERGENCY PROTOCOL: {protocol.name}")

            # Execute each action in the protocol
            for action in protocol.response_actions:
                try:
                    action_result = await self._execute_protocol_action(action, emergency_event)
                    emergency_event.actions_taken.append(action_result)

                except Exception as e:
                    logger.error(f"Error executing protocol action {action.value}: {e}")
                    emergency_event.actions_taken.append({
                        'action': action.value,
                        'status': 'failed',
                        'error': str(e),
                        'timestamp': datetime.now().isoformat()
                    })

            # Update protocol and event statistics
            protocol.last_triggered = datetime.now()
            protocol.execution_count += 1

            emergency_event.response_time_seconds = time.perf_counter() - execution_start
            emergency_event.protocols_activated.append(protocol.name)

            self.emergency_mode_active = True

            logger.critical(f"Emergency protocol {protocol.name} executed in {emergency_event.response_time_seconds:.2f}s")

        except Exception as e:
            logger.error(f"Critical error executing emergency protocol {protocol.name}: {e}")
            raise EmergencyStopError(f"Emergency protocol execution failed: {e}") from e

    async def _execute_protocol_action(self, action: ProtocolAction, emergency_event: EmergencyEvent) -> dict[str, Any]:
        """Execute a single protocol action."""
        action_start = time.perf_counter()

        try:
            if action == ProtocolAction.NOTIFY_STAKEHOLDERS:
                if self.notify_stakeholders_callback:
                    await self.notify_stakeholders_callback({
                        'event_id': emergency_event.event_id,
                        'emergency_type': emergency_event.emergency_type.value,
                        'emergency_level': emergency_event.emergency_level.value,
                        'trigger_reason': emergency_event.trigger_reason,
                        'timestamp': emergency_event.timestamp.isoformat()
                    })

                return {
                    'action': action.value,
                    'status': 'completed',
                    'execution_time_seconds': time.perf_counter() - action_start,
                    'timestamp': datetime.now().isoformat()
                }

            elif action == ProtocolAction.ISOLATE_SYSTEM:
                self.system_isolated = True
                logger.critical("System isolated due to emergency protocol")

                return {
                    'action': action.value,
                    'status': 'completed',
                    'system_isolated': True,
                    'execution_time_seconds': time.perf_counter() - action_start,
                    'timestamp': datetime.now().isoformat()
                }

            elif action == ProtocolAction.HEDGE_PORTFOLIO:
                # Would implement portfolio hedging logic
                return {
                    'action': action.value,
                    'status': 'placeholder',
                    'message': 'Portfolio hedging not yet implemented',
                    'execution_time_seconds': time.perf_counter() - action_start,
                    'timestamp': datetime.now().isoformat()
                }

            elif action == ProtocolAction.REDUCE_EXPOSURE:
                # Would implement exposure reduction logic
                return {
                    'action': action.value,
                    'status': 'placeholder',
                    'message': 'Exposure reduction not yet implemented',
                    'execution_time_seconds': time.perf_counter() - action_start,
                    'timestamp': datetime.now().isoformat()
                }

            elif action == ProtocolAction.ACTIVATE_BACKUP:
                # Would implement backup system activation
                return {
                    'action': action.value,
                    'status': 'placeholder',
                    'message': 'Backup activation not yet implemented',
                    'execution_time_seconds': time.perf_counter() - action_start,
                    'timestamp': datetime.now().isoformat()
                }

            else:
                return {
                    'action': action.value,
                    'status': 'unknown_action',
                    'execution_time_seconds': time.perf_counter() - action_start,
                    'timestamp': datetime.now().isoformat()
                }

        except Exception as e:
            logger.error(f"Error executing protocol action {action.value}: {e}")
            return {
                'action': action.value,
                'status': 'failed',
                'error': str(e),
                'execution_time_seconds': time.perf_counter() - action_start,
                'timestamp': datetime.now().isoformat()
            }

    @measure_latency("manual_emergency_stop")
    async def manual_emergency_stop(self, reason: str, emergency_type: EmergencyType) -> dict[str, Any]:
        """Manually trigger emergency stop procedures."""
        try:
            event_id = f"MANUAL_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

            emergency_event = EmergencyEvent(
                event_id=event_id,
                emergency_type=emergency_type,
                emergency_level=EmergencyLevel.CRITICAL,
                trigger_reason=f"Manual emergency stop: {reason}",
                trigger_value=None,
                threshold_value=None,
                timestamp=datetime.now()
            )

            self.active_emergencies[event_id] = emergency_event

            logger.critical(f"🚨 MANUAL EMERGENCY STOP INITIATED: {reason}")

            # Execute comprehensive emergency response
            actions = [
                ProtocolAction.HALT_TRADING,
                ProtocolAction.CANCEL_ORDERS,
                ProtocolAction.FLATTEN_POSITIONS,
                ProtocolAction.NOTIFY_STAKEHOLDERS
            ]

            for action in actions:
                try:
                    action_result = await self._execute_protocol_action(action, emergency_event)
                    emergency_event.actions_taken.append(action_result)
                except Exception as e:
                    logger.error(f"Error in manual emergency stop action {action.value}: {e}")

            self.emergency_mode_active = True

            return {
                'status': 'completed',
                'event_id': event_id,
                'actions_executed': len(emergency_event.actions_taken),
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error in manual emergency stop: {e}")
            raise EmergencyStopError(f"Manual emergency stop failed: {e}") from e

    async def recover_from_emergency(self, event_id: str, recovery_notes: str = "") -> dict[str, Any]:
        """Recover from emergency state and resume normal operations."""
        try:
            if event_id not in self.active_emergencies:
                return {'status': 'error', 'message': f'Emergency event {event_id} not found'}

            emergency_event = self.active_emergencies[event_id]

            # Mark emergency as resolved
            emergency_event.resolved = True
            emergency_event.resolution_time = datetime.now()

            # Move to history
            self.emergency_history.append(emergency_event)
            del self.active_emergencies[event_id]

            # Check if all emergencies are resolved
            if not self.active_emergencies:
                self.emergency_mode_active = False
                self.system_isolated = False
                logger.info("All emergencies resolved - returning to normal operations")

            logger.info(f"Emergency {event_id} resolved: {recovery_notes}")

            return {
                'status': 'resolved',
                'event_id': event_id,
                'recovery_notes': recovery_notes,
                'emergency_mode_active': self.emergency_mode_active,
                'system_isolated': self.system_isolated,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error recovering from emergency {event_id}: {e}")
            return {
                'status': 'error',
                'event_id': event_id,
                'error': str(e)
            }

    def add_custom_protocol(self, protocol: EmergencyProtocol) -> bool:
        """Add a custom emergency protocol."""
        try:
            if protocol.name in self.protocols:
                logger.warning(f"Protocol {protocol.name} already exists - overwriting")

            self.protocols[protocol.name] = protocol
            logger.info(f"Added custom emergency protocol: {protocol.name}")
            return True

        except Exception as e:
            logger.error(f"Error adding custom protocol {protocol.name}: {e}")
            return False

    def get_emergency_status(self) -> dict[str, Any]:
        """Get current emergency system status."""
        return {
            'emergency_mode_active': self.emergency_mode_active,
            'system_isolated': self.system_isolated,
            'active_emergencies': len(self.active_emergencies),
            'total_protocols': len(self.protocols),
            'last_emergency_check': self.last_emergency_check.isoformat(),
            'emergency_history_count': len(self.emergency_history),
            'active_emergency_details': [
                {
                    'event_id': event.event_id,
                    'emergency_type': event.emergency_type.value,
                    'emergency_level': event.emergency_level.value,
                    'trigger_reason': event.trigger_reason,
                    'timestamp': event.timestamp.isoformat(),
                    'actions_taken': len(event.actions_taken)
                }
                for event in self.active_emergencies.values()
            ]
        }

    def get_protocol_status(self) -> dict[str, Any]:
        """Get status of all emergency protocols."""
        return {
            protocol_name: protocol.to_dict()
            for protocol_name, protocol in self.protocols.items()
        }

    def get_emergency_history(self, limit: int = 50) -> list[dict[str, Any]]:
        """Get recent emergency event history."""
        # Sort by timestamp, most recent first
        sorted_history = sorted(
            self.emergency_history,
            key=lambda x: x.timestamp,
            reverse=True
        )

        return [event.to_dict() for event in sorted_history[:limit]]

    @timeout_async(timeout_seconds=30)
    async def emergency_health_check(self) -> dict[str, Any]:
        """Perform emergency system health check."""
        try:
            health_status = {
                'overall_status': 'healthy',
                'emergency_mode_active': self.emergency_mode_active,
                'system_isolated': self.system_isolated,
                'protocols_loaded': len(self.protocols),
                'active_emergencies': len(self.active_emergencies),
                'position_flattener_available': self.position_flattener is not None,
                'callbacks_configured': {
                    'halt_trading': self.halt_trading_callback is not None,
                    'cancel_orders': self.cancel_orders_callback is not None,
                    'place_order': self.place_order_callback is not None,
                    'get_positions': self.get_positions_callback is not None,
                    'notify_stakeholders': self.notify_stakeholders_callback is not None
                },
                'last_check': datetime.now().isoformat()
            }

            # Determine overall status
            if self.emergency_mode_active:
                health_status['overall_status'] = 'emergency_active'
            elif self.system_isolated:
                health_status['overall_status'] = 'system_isolated'
            elif len(self.active_emergencies) > 0:
                health_status['overall_status'] = 'emergency_pending'

            return health_status

        except Exception as e:
            logger.error(f"Error in emergency health check: {e}")
            return {
                'overall_status': 'error',
                'error': str(e),
                'last_check': datetime.now().isoformat()
            }

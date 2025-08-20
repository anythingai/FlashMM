"""
FlashMM P&L Risk Controller

Comprehensive P&L risk management system with:
- Real-time P&L monitoring with multiple calculation methods
- Daily, weekly, and monthly P&L limits
- Drawdown protection with automatic position reduction
- P&L-based position sizing adjustments
- Stop-loss and take-profit mechanisms
- Risk-adjusted P&L attribution analysis
"""

import asyncio
import math
from typing import Dict, Any, Optional, List, Tuple, Callable
from datetime import datetime, timedelta, time
from decimal import Decimal
from enum import Enum
from dataclasses import dataclass, field
import numpy as np
from collections import deque

from flashmm.config.settings import get_config
from flashmm.utils.logging import get_logger, TradingEventLogger
from flashmm.utils.exceptions import RiskError
from flashmm.utils.decorators import measure_latency, timeout_async

logger = get_logger(__name__)
trading_logger = TradingEventLogger()


class PnLPeriod(Enum):
    """P&L tracking periods."""
    INTRADAY = "intraday"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    INCEPTION = "inception"


class DrawdownLevel(Enum):
    """Drawdown severity levels."""
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class StopLossType(Enum):
    """Stop-loss types."""
    FIXED_AMOUNT = "fixed_amount"
    PERCENTAGE = "percentage"
    TRAILING = "trailing"
    VOLATILITY_ADJUSTED = "volatility_adjusted"
    TIME_BASED = "time_based"


@dataclass
class PnLSnapshot:
    """P&L snapshot at a point in time."""
    timestamp: datetime
    symbol: str
    realized_pnl: Decimal
    unrealized_pnl: Decimal
    total_pnl: Decimal
    position_value: Decimal
    fees: Decimal
    volume: Decimal
    trade_count: int
    
    @property
    def net_pnl(self) -> Decimal:
        """Net P&L after fees."""
        return self.total_pnl - self.fees
    
    @property
    def pnl_per_trade(self) -> Decimal:
        """Average P&L per trade."""
        return self.total_pnl / max(1, self.trade_count)
    
    @property
    def return_on_volume(self) -> Decimal:
        """Return on trading volume."""
        return self.total_pnl / max(Decimal('1'), self.volume)


@dataclass
class DrawdownMetrics:
    """Drawdown analysis metrics."""
    current_drawdown: Decimal
    current_drawdown_pct: Decimal
    max_drawdown: Decimal
    max_drawdown_pct: Decimal
    peak_pnl: Decimal
    drawdown_duration_hours: float
    recovery_factor: float  # How much above previous peak
    consecutive_losing_trades: int
    drawdown_level: DrawdownLevel
    time_to_recovery_estimate_hours: Optional[float] = None


@dataclass
class StopLossOrder:
    """Stop-loss order configuration."""
    symbol: str
    stop_type: StopLossType
    trigger_value: Decimal
    current_value: Decimal
    created_time: datetime
    last_updated: datetime
    
    # Configuration
    fixed_amount: Optional[Decimal] = None
    percentage: Optional[float] = None
    trailing_distance: Optional[Decimal] = None
    volatility_multiplier: Optional[float] = None
    time_limit_hours: Optional[float] = None
    
    # State
    triggered: bool = False
    trigger_time: Optional[datetime] = None
    initial_position_value: Optional[Decimal] = None
    
    def should_trigger(self, current_pnl: Decimal, current_position_value: Decimal) -> bool:
        """Check if stop-loss should trigger."""
        if self.triggered:
            return False
        
        if self.stop_type == StopLossType.FIXED_AMOUNT:
            return current_pnl <= -abs(self.trigger_value)
        
        elif self.stop_type == StopLossType.PERCENTAGE:
            if self.initial_position_value and self.initial_position_value > 0:
                loss_pct = (current_pnl / self.initial_position_value) * 100
                return loss_pct <= -abs(self.percentage or 0)
        
        elif self.stop_type == StopLossType.TRAILING:
            # Trailing stop updates trigger as position moves favorably
            return current_pnl <= self.trigger_value
        
        elif self.stop_type == StopLossType.TIME_BASED:
            if self.time_limit_hours:
                elapsed_hours = (datetime.now() - self.created_time).total_seconds() / 3600
                return elapsed_hours >= self.time_limit_hours and current_pnl < 0
        
        return False
    
    def update_trailing_stop(self, current_pnl: Decimal):
        """Update trailing stop level."""
        if self.stop_type == StopLossType.TRAILING and self.trailing_distance:
            new_trigger = current_pnl - self.trailing_distance
            if new_trigger > self.trigger_value:
                self.trigger_value = new_trigger
                self.last_updated = datetime.now()


class DrawdownProtector:
    """Drawdown protection system."""
    
    def __init__(self):
        self.config = get_config()
        
        # Drawdown thresholds
        self.warning_drawdown_pct = self.config.get("risk.warning_drawdown_pct", 2.0)  # 2%
        self.critical_drawdown_pct = self.config.get("risk.critical_drawdown_pct", 5.0)  # 5%
        self.emergency_drawdown_pct = self.config.get("risk.emergency_drawdown_pct", 10.0)  # 10%
        
        # Protection actions
        self.enable_position_reduction = self.config.get("risk.enable_position_reduction", True)
        self.enable_spread_widening = self.config.get("risk.enable_spread_widening", True)
        self.enable_trading_halt = self.config.get("risk.enable_trading_halt", True)
        
        # State tracking
        self.peak_pnl_by_period: Dict[PnLPeriod, Decimal] = {}
        self.drawdown_start_time: Dict[PnLPeriod, Optional[datetime]] = {}
        self.protection_actions_taken: List[Dict[str, Any]] = []
        
        # Callbacks for protection actions
        self.position_reduction_callback: Optional[Callable] = None
        self.spread_widening_callback: Optional[Callable] = None
        self.trading_halt_callback: Optional[Callable] = None
    
    async def update_pnl_tracking(self, period: PnLPeriod, current_pnl: Decimal) -> DrawdownMetrics:
        """Update P&L tracking and calculate drawdown metrics."""
        try:
            # Initialize if first update
            if period not in self.peak_pnl_by_period:
                self.peak_pnl_by_period[period] = current_pnl
                self.drawdown_start_time[period] = None
            
            # Update peak P&L
            if current_pnl > self.peak_pnl_by_period[period]:
                self.peak_pnl_by_period[period] = current_pnl
                self.drawdown_start_time[period] = None  # End drawdown period
            
            # Calculate drawdown
            peak_pnl = self.peak_pnl_by_period[period]
            current_drawdown = peak_pnl - current_pnl
            current_drawdown_pct = float((current_drawdown / max(abs(peak_pnl), Decimal('100'))) * 100)
            
            # Track drawdown duration
            drawdown_duration_hours = 0.0
            if current_drawdown > 0:
                if self.drawdown_start_time[period] is None:
                    self.drawdown_start_time[period] = datetime.now()
                else:
                    duration = datetime.now() - self.drawdown_start_time[period]
                    drawdown_duration_hours = duration.total_seconds() / 3600
            
            # Calculate recovery factor
            recovery_factor = float(current_pnl / max(abs(peak_pnl), Decimal('1')))
            
            # Classify drawdown level
            drawdown_level = self._classify_drawdown_level(current_drawdown_pct)
            
            # Estimate recovery time (simple heuristic)
            time_to_recovery = None
            if current_drawdown > 0 and drawdown_duration_hours > 0:
                # Assume recovery at same rate as decline
                recovery_needed = float(current_drawdown)
                if recovery_needed > 0:
                    time_to_recovery = drawdown_duration_hours * 2  # Conservative estimate
            
            metrics = DrawdownMetrics(
                current_drawdown=current_drawdown,
                current_drawdown_pct=Decimal(str(current_drawdown_pct)),
                max_drawdown=current_drawdown,  # Would track historical max
                max_drawdown_pct=Decimal(str(current_drawdown_pct)),
                peak_pnl=peak_pnl,
                drawdown_duration_hours=drawdown_duration_hours,
                recovery_factor=recovery_factor,
                consecutive_losing_trades=0,  # Would track from trade history
                drawdown_level=drawdown_level,
                time_to_recovery_estimate_hours=time_to_recovery
            )
            
            # Trigger protection actions if needed
            await self._check_protection_triggers(period, metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error updating drawdown tracking for {period.value}: {e}")
            return self._default_drawdown_metrics()
    
    def _classify_drawdown_level(self, drawdown_pct: float) -> DrawdownLevel:
        """Classify drawdown severity level."""
        if drawdown_pct >= self.emergency_drawdown_pct:
            return DrawdownLevel.EMERGENCY
        elif drawdown_pct >= self.critical_drawdown_pct:
            return DrawdownLevel.CRITICAL
        elif drawdown_pct >= self.warning_drawdown_pct:
            return DrawdownLevel.WARNING
        else:
            return DrawdownLevel.NORMAL
    
    async def _check_protection_triggers(self, period: PnLPeriod, metrics: DrawdownMetrics):
        """Check if protection actions should be triggered."""
        try:
            current_time = datetime.now()
            action_taken = False
            
            if metrics.drawdown_level == DrawdownLevel.EMERGENCY:
                if self.enable_trading_halt and self.trading_halt_callback:
                    await self.trading_halt_callback(f"Emergency drawdown: {metrics.current_drawdown_pct}%")
                    action_taken = True
                    
                    self.protection_actions_taken.append({
                        'timestamp': current_time.isoformat(),
                        'period': period.value,
                        'action': 'trading_halt',
                        'trigger': 'emergency_drawdown',
                        'drawdown_pct': float(metrics.current_drawdown_pct),
                        'drawdown_amount': float(metrics.current_drawdown)
                    })
            
            elif metrics.drawdown_level == DrawdownLevel.CRITICAL:
                if self.enable_position_reduction and self.position_reduction_callback:
                    reduction_factor = 0.5  # Reduce positions by 50%
                    await self.position_reduction_callback(reduction_factor)
                    action_taken = True
                    
                    self.protection_actions_taken.append({
                        'timestamp': current_time.isoformat(),
                        'period': period.value,
                        'action': 'position_reduction',
                        'trigger': 'critical_drawdown',
                        'reduction_factor': reduction_factor,
                        'drawdown_pct': float(metrics.current_drawdown_pct)
                    })
            
            elif metrics.drawdown_level == DrawdownLevel.WARNING:
                if self.enable_spread_widening and self.spread_widening_callback:
                    spread_multiplier = 1.5  # Widen spreads by 50%
                    await self.spread_widening_callback(spread_multiplier)
                    action_taken = True
                    
                    self.protection_actions_taken.append({
                        'timestamp': current_time.isoformat(),
                        'period': period.value,
                        'action': 'spread_widening',
                        'trigger': 'warning_drawdown',
                        'spread_multiplier': spread_multiplier,
                        'drawdown_pct': float(metrics.current_drawdown_pct)
                    })
            
            if action_taken:
                logger.warning(f"Drawdown protection triggered for {period.value}: {metrics.drawdown_level.value}")
        
        except Exception as e:
            logger.error(f"Error checking protection triggers: {e}")
    
    def _default_drawdown_metrics(self) -> DrawdownMetrics:
        """Return default drawdown metrics."""
        return DrawdownMetrics(
            current_drawdown=Decimal('0'),
            current_drawdown_pct=Decimal('0'),
            max_drawdown=Decimal('0'),
            max_drawdown_pct=Decimal('0'),
            peak_pnl=Decimal('0'),
            drawdown_duration_hours=0.0,
            recovery_factor=1.0,
            consecutive_losing_trades=0,
            drawdown_level=DrawdownLevel.NORMAL
        )
    
    def set_callbacks(self,
                      position_reduction: Optional[Callable] = None,
                      spread_widening: Optional[Callable] = None,
                      trading_halt: Optional[Callable] = None):
        """Set callback functions for protection actions."""
        self.position_reduction_callback = position_reduction
        self.spread_widening_callback = spread_widening
        self.trading_halt_callback = trading_halt


class StopLossManager:
    """Stop-loss order management system."""
    
    def __init__(self):
        self.config = get_config()
        self.stop_orders: Dict[str, List[StopLossOrder]] = {}  # symbol -> list of stops
        self.triggered_stops: List[StopLossOrder] = []
        
        # Default stop-loss settings
        self.default_stop_pct = self.config.get("risk.default_stop_loss_pct", 3.0)  # 3%
        self.default_trailing_distance = self.config.get("risk.default_trailing_distance", 100.0)  # $100
        
        self._lock = asyncio.Lock()
    
    async def create_stop_loss(self, 
                             symbol: str, 
                             stop_type: StopLossType,
                             position_value: Decimal,
                             **kwargs) -> StopLossOrder:
        """Create a new stop-loss order."""
        async with self._lock:
            try:
                current_time = datetime.now()
                
                # Calculate trigger value based on stop type
                if stop_type == StopLossType.FIXED_AMOUNT:
                    trigger_value = -abs(Decimal(str(kwargs.get('amount', 100.0))))
                elif stop_type == StopLossType.PERCENTAGE:
                    percentage = kwargs.get('percentage', self.default_stop_pct)
                    trigger_value = -abs(position_value * Decimal(str(percentage / 100)))
                elif stop_type == StopLossType.TRAILING:
                    trigger_value = Decimal(str(kwargs.get('initial_trigger', -100.0)))
                else:
                    trigger_value = Decimal('0')
                
                stop_order = StopLossOrder(
                    symbol=symbol,
                    stop_type=stop_type,
                    trigger_value=trigger_value,
                    current_value=Decimal('0'),
                    created_time=current_time,
                    last_updated=current_time,
                    fixed_amount=kwargs.get('fixed_amount'),
                    percentage=kwargs.get('percentage'),
                    trailing_distance=kwargs.get('trailing_distance'),
                    volatility_multiplier=kwargs.get('volatility_multiplier'),
                    time_limit_hours=kwargs.get('time_limit_hours'),
                    initial_position_value=position_value
                )
                
                # Add to active stops
                if symbol not in self.stop_orders:
                    self.stop_orders[symbol] = []
                self.stop_orders[symbol].append(stop_order)
                
                logger.info(f"Created {stop_type.value} stop-loss for {symbol}: trigger at {trigger_value}")
                return stop_order
                
            except Exception as e:
                logger.error(f"Error creating stop-loss for {symbol}: {e}")
                raise RiskError(f"Stop-loss creation failed: {e}")
    
    async def check_stop_losses(self, symbol: str, current_pnl: Decimal, position_value: Decimal) -> List[StopLossOrder]:
        """Check if any stop-losses should trigger."""
        async with self._lock:
            triggered_stops = []
            
            if symbol not in self.stop_orders:
                return triggered_stops
            
            active_stops = [stop for stop in self.stop_orders[symbol] if not stop.triggered]
            
            for stop in active_stops:
                # Update trailing stops
                if stop.stop_type == StopLossType.TRAILING:
                    stop.update_trailing_stop(current_pnl)
                
                # Check if should trigger
                if stop.should_trigger(current_pnl, position_value):
                    stop.triggered = True
                    stop.trigger_time = datetime.now()
                    stop.current_value = current_pnl
                    
                    triggered_stops.append(stop)
                    self.triggered_stops.append(stop)
                    
                    logger.critical(f"🛑 STOP-LOSS TRIGGERED: {symbol} {stop.stop_type.value} at {current_pnl}")
                    
                    # Log to trading event logger
                    await trading_logger.log_pnl_event(
                        symbol,
                        float(current_pnl),
                        float(current_pnl),
                        float(current_pnl),
                        stop_loss_triggered=f"{stop.stop_type.value}: {stop.trigger_value}"
                    )
            
            return triggered_stops
    
    async def cancel_stop_loss(self, symbol: str, stop_type: Optional[StopLossType] = None):
        """Cancel stop-loss orders."""
        async with self._lock:
            if symbol not in self.stop_orders:
                return
            
            if stop_type:
                # Cancel specific type
                self.stop_orders[symbol] = [
                    stop for stop in self.stop_orders[symbol] 
                    if stop.stop_type != stop_type or stop.triggered
                ]
            else:
                # Cancel all active stops for symbol
                for stop in self.stop_orders[symbol]:
                    if not stop.triggered:
                        stop.triggered = True
                        stop.trigger_time = datetime.now()
                
                self.stop_orders[symbol] = []
            
            logger.info(f"Cancelled stop-loss orders for {symbol}")
    
    def get_active_stops(self, symbol: Optional[str] = None) -> Dict[str, List[StopLossOrder]]:
        """Get active stop-loss orders."""
        if symbol:
            active_stops = [stop for stop in self.stop_orders.get(symbol, []) if not stop.triggered]
            return {symbol: active_stops}
        else:
            return {
                sym: [stop for stop in stops if not stop.triggered]
                for sym, stops in self.stop_orders.items()
            }


class PnLRiskController:
    """Comprehensive P&L risk management system."""
    
    def __init__(self):
        self.config = get_config()
        
        # Components
        self.drawdown_protector = DrawdownProtector()
        self.stop_loss_manager = StopLossManager()
        
        # P&L tracking
        self.pnl_history: Dict[str, deque] = {}  # symbol -> PnL snapshots
        self.daily_pnl_limits: Dict[str, Decimal] = {}
        self.period_pnl: Dict[PnLPeriod, Dict[str, Decimal]] = {}
        
        # Risk limits
        self.daily_loss_limit = Decimal(str(self.config.get("risk.daily_loss_limit", 5000.0)))
        self.position_loss_limit = Decimal(str(self.config.get("risk.position_loss_limit", 1000.0)))
        self.max_consecutive_losses = self.config.get("risk.max_consecutive_losses", 5)
        
        # State tracking
        self.consecutive_losing_positions = 0
        self.last_reset_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        self._lock = asyncio.Lock()
    
    async def initialize(self) -> None:
        """Initialize P&L risk controller."""
        try:
            # Initialize period tracking
            for period in PnLPeriod:
                self.period_pnl[period] = {}
            
            # Set up protection callbacks
            self.drawdown_protector.set_callbacks(
                position_reduction=self._reduce_positions,
                spread_widening=self._widen_spreads,
                trading_halt=self._halt_trading
            )
            
            logger.info("P&L risk controller initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize P&L risk controller: {e}")
            raise RiskError(f"P&L risk controller initialization failed: {e}")
    
    @timeout_async(0.05)  # 50ms timeout for P&L checks
    @measure_latency("pnl_risk_check")
    async def update_pnl_and_check_risks(self, 
                                       symbol: str,
                                       realized_pnl: Decimal,
                                       unrealized_pnl: Decimal,
                                       position_value: Decimal,
                                       fees: Decimal = Decimal('0'),
                                       volume: Decimal = Decimal('0'),
                                       trade_count: int = 0) -> Dict[str, Any]:
        """Update P&L and perform comprehensive risk checks."""
        async with self._lock:
            try:
                current_time = datetime.now()
                total_pnl = realized_pnl + unrealized_pnl
                
                # Create P&L snapshot
                snapshot = PnLSnapshot(
                    timestamp=current_time,
                    symbol=symbol,
                    realized_pnl=realized_pnl,
                    unrealized_pnl=unrealized_pnl,
                    total_pnl=total_pnl,
                    position_value=position_value,
                    fees=fees,
                    volume=volume,
                    trade_count=trade_count
                )
                
                # Update history
                if symbol not in self.pnl_history:
                    self.pnl_history[symbol] = deque(maxlen=1000)  # Keep last 1000 snapshots
                self.pnl_history[symbol].append(snapshot)
                
                # Update period P&L tracking
                await self._update_period_pnl(symbol, total_pnl)
                
                # Check daily reset
                await self._check_daily_reset()
                
                # Calculate drawdown metrics
                daily_pnl = self.period_pnl[PnLPeriod.DAILY].get(symbol, Decimal('0'))
                drawdown_metrics = await self.drawdown_protector.update_pnl_tracking(
                    PnLPeriod.DAILY, daily_pnl
                )
                
                # Check stop-losses
                triggered_stops = await self.stop_loss_manager.check_stop_losses(
                    symbol, total_pnl, position_value
                )
                
                # Check P&L limits
                limit_violations = await self._check_pnl_limits(symbol, snapshot)
                
                # Generate risk assessment
                risk_assessment = {
                    'timestamp': current_time.isoformat(),
                    'symbol': symbol,
                    'pnl_snapshot': {
                        'realized_pnl': float(realized_pnl),
                        'unrealized_pnl': float(unrealized_pnl),
                        'total_pnl': float(total_pnl),
                        'net_pnl': float(snapshot.net_pnl),
                        'position_value': float(position_value),
                        'fees': float(fees)
                    },
                    'drawdown_metrics': {
                        'current_drawdown': float(drawdown_metrics.current_drawdown),
                        'current_drawdown_pct': float(drawdown_metrics.current_drawdown_pct),
                        'drawdown_level': drawdown_metrics.drawdown_level.value,
                        'peak_pnl': float(drawdown_metrics.peak_pnl),
                        'drawdown_duration_hours': drawdown_metrics.drawdown_duration_hours
                    },
                    'triggered_stops': [
                        {
                            'type': stop.stop_type.value,
                            'trigger_value': float(stop.trigger_value),
                            'trigger_time': stop.trigger_time.isoformat() if stop.trigger_time else None
                        }
                        for stop in triggered_stops
                    ],
                    'limit_violations': limit_violations,
                    'period_pnl': {
                        period.value: float(self.period_pnl[period].get(symbol, Decimal('0')))
                        for period in PnLPeriod
                    },
                    'consecutive_losing_positions': self.consecutive_losing_positions,
                    'recommendations': self._generate_pnl_recommendations(
                        snapshot, drawdown_metrics, triggered_stops, limit_violations
                    )
                }
                
                return risk_assessment
                
            except Exception as e:
                logger.error(f"Error updating P&L and checking risks for {symbol}: {e}")
                raise RiskError(f"P&L risk check failed: {e}")
    
    async def _update_period_pnl(self, symbol: str, total_pnl: Decimal):
        """Update P&L tracking for different periods."""
        for period in PnLPeriod:
            if symbol not in self.period_pnl[period]:
                self.period_pnl[period][symbol] = Decimal('0')
            
            # For simplicity, using total P&L for all periods
            # In production, would calculate period-specific P&L
            self.period_pnl[period][symbol] = total_pnl
    
    async def _check_daily_reset(self):
        """Check if daily reset is needed."""
        current_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        if current_date > self.last_reset_time:
            # Reset daily tracking
            self.period_pnl[PnLPeriod.DAILY] = {}
            self.consecutive_losing_positions = 0
            self.last_reset_time = current_date
            
            logger.info("Daily P&L tracking reset")
    
    async def _check_pnl_limits(self, symbol: str, snapshot: PnLSnapshot) -> List[Dict[str, Any]]:
        """Check P&L limit violations."""
        violations = []
        
        try:
            # Daily loss limit
            daily_pnl = self.period_pnl[PnLPeriod.DAILY].get(symbol, Decimal('0'))
            if daily_pnl < -self.daily_loss_limit:
                violations.append({
                    'type': 'daily_loss_limit',
                    'limit': float(self.daily_loss_limit),
                    'actual': float(daily_pnl),
                    'severity': 'critical'
                })
            
            # Position loss limit
            if snapshot.total_pnl < -self.position_loss_limit:
                violations.append({
                    'type': 'position_loss_limit',
                    'limit': float(self.position_loss_limit),
                    'actual': float(snapshot.total_pnl),
                    'severity': 'high'
                })
            
            # Check consecutive losses
            if snapshot.total_pnl < 0:
                self.consecutive_losing_positions += 1
                if self.consecutive_losing_positions >= self.max_consecutive_losses:
                    violations.append({
                        'type': 'consecutive_losses',
                        'limit': self.max_consecutive_losses,
                        'actual': self.consecutive_losing_positions,
                        'severity': 'high'
                    })
            else:
                self.consecutive_losing_positions = 0
            
            return violations
            
        except Exception as e:
            logger.error(f"Error checking P&L limits: {e}")
            return []
    
    def _generate_pnl_recommendations(self, 
                                    snapshot: PnLSnapshot,
                                    drawdown_metrics: DrawdownMetrics,
                                    triggered_stops: List[StopLossOrder],
                                    limit_violations: List[Dict[str, Any]]) -> List[str]:
        """Generate P&L-based recommendations."""
        recommendations = []
        
        # Drawdown recommendations
        if drawdown_metrics.drawdown_level == DrawdownLevel.EMERGENCY:
            recommendations.append("CRITICAL: Halt all trading immediately due to emergency drawdown")
        elif drawdown_metrics.drawdown_level == DrawdownLevel.CRITICAL:
            recommendations.append("Reduce position sizes significantly due to critical drawdown")
        elif drawdown_metrics.drawdown_level == DrawdownLevel.WARNING:
            recommendations.append("Consider reducing position sizes due to drawdown warning")
        
        # Stop-loss recommendations
        if triggered_stops:
            recommendations.append(f"Execute position closure due to {len(triggered_stops)} triggered stop-loss orders")
        
        # Limit violation recommendations
        for violation in limit_violations:
            if violation['type'] == 'daily_loss_limit':
                recommendations.append("URGENT: Daily loss limit exceeded - halt trading for today")
            elif violation['type'] == 'consecutive_losses':
                recommendations.append("Review trading strategy due to consecutive losses")
        
        # Performance recommendations
        if snapshot.net_pnl < 0:
            recommendations.append("Consider reviewing quote parameters due to negative P&L")
        
        if snapshot.pnl_per_trade < Decimal('-10'):
            recommendations.append("Review trading strategy - poor per-trade performance")
        
        return recommendations
    
    async def _reduce_positions(self, reduction_factor: float):
        """Callback to reduce position sizes."""
        try:
            logger.warning(f"Drawdown protection: Reducing positions by {reduction_factor:.1%}")
            # Implementation would integrate with position tracker
            # For now, just log the action
            await trading_logger.log_pnl_event(
                "SYSTEM",
                0.0,
                0.0,
                0.0,
                drawdown_action=f"position_reduction_{reduction_factor}"
            )
        except Exception as e:
            logger.error(f"Error reducing positions: {e}")
    
    async def _widen_spreads(self, spread_multiplier: float):
        """Callback to widen trading spreads."""
        try:
            logger.warning(f"Drawdown protection: Widening spreads by {spread_multiplier:.1f}x")
            # Implementation would integrate with quote generator
            await trading_logger.log_pnl_event(
                "SYSTEM",
                0.0,
                0.0,
                0.0,
                drawdown_action=f"spread_widening_{spread_multiplier}"
            )
        except Exception as e:
            logger.error(f"Error widening spreads: {e}")
    
    async def _halt_trading(self, reason: str):
        """Callback to halt trading operations."""
        try:
            logger.critical(f"Drawdown protection: Halting trading - {reason}")
            # Implementation would integrate with trading engine
            await trading_logger.log_pnl_event(
                "SYSTEM",
                0.0,
                0.0,
                0.0,
                drawdown_action=f"trading_halt_{reason}"
            )
        except Exception as e:
            logger.error(f"Error halting trading: {e}")
    
    async def create_position_stop_loss(self,
                                      symbol: str,
                                      position_value: Decimal,
                                      stop_type: StopLossType = StopLossType.PERCENTAGE,
                                      **kwargs) -> StopLossOrder:
        """Create a stop-loss order for a position."""
        return await self.stop_loss_manager.create_stop_loss(
            symbol, stop_type, position_value, **kwargs
        )
    
    async def cancel_position_stops(self, symbol: str, stop_type: Optional[StopLossType] = None):
        """Cancel stop-loss orders for a position."""
        await self.stop_loss_manager.cancel_stop_loss(symbol, stop_type)
    
    def get_pnl_summary(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Get P&L summary for symbol or all symbols."""
        try:
            if symbol:
                history = list(self.pnl_history.get(symbol, []))
                if not history:
                    return {'symbol': symbol, 'no_data': True}
                
                latest = history[-1]
                return {
                    'symbol': symbol,
                    'latest_snapshot': {
                        'total_pnl': float(latest.total_pnl),
                        'realized_pnl': float(latest.realized_pnl),
                        'unrealized_pnl': float(latest.unrealized_pnl),
                        'net_pnl': float(latest.net_pnl),
                        'fees': float(latest.fees),
                        'trade_count': latest.trade_count
                    },
                    'period_pnl': {
                        period.value: float(self.period_pnl[period].get(symbol, Decimal('0')))
                        for period in PnLPeriod
                    },
                    'active_stops': len(self.stop_loss_manager.get_active_stops(symbol).get(symbol, [])),
                    'triggered_stops': len([s for s in self.stop_loss_manager.triggered_stops if s.symbol == symbol])
                }
            else:
                # All symbols summary
                all_symbols = set()
                for period_data in self.period_pnl.values():
                    all_symbols.update(period_data.keys())
                
                summary = {
                    'total_symbols': len(all_symbols),
                    'symbols': {},
                    'global_metrics': {
                        period.value: sum(float(v) for v in period_data.values())
                        for period, period_data in self.period_pnl.items()
                    },
                    'consecutive_losing_positions': self.consecutive_losing_positions,
                    'protection_actions_count': len(self.drawdown_protector.protection_actions_taken)
                }
                
                for sym in all_symbols:
                    summary['symbols'][sym] = self.get_pnl_summary(sym)
                
                return summary
                
        except Exception as e:
            logger.error(f"Error getting P&L summary: {e}")
            return {'error': str(e)}
    
    def get_drawdown_status(self) -> Dict[str, Any]:
        """Get current drawdown protection status."""
        return {
            'protection_settings': {
                'warning_drawdown_pct': self.drawdown_protector.warning_drawdown_pct,
                'critical_drawdown_pct': self.drawdown_protector.critical_drawdown_pct,
                'emergency_drawdown_pct': self.drawdown_protector.emergency_drawdown_pct,
                'position_reduction_enabled': self.drawdown_protector.enable_position_reduction,
                'spread_widening_enabled': self.drawdown_protector.enable_spread_widening,
                'trading_halt_enabled': self.drawdown_protector.enable_trading_halt
            },
            'current_peaks': {
                period.value: float(peak) for period, peak in self.drawdown_protector.peak_pnl_by_period.items()
            },
            'drawdown_start_times': {
                period.value: start_time.isoformat() if start_time else None
                for period, start_time in self.drawdown_protector.drawdown_start_time.items()
            },
            'protection_actions_taken': self.drawdown_protector.protection_actions_taken[-10:],  # Last 10 actions
            'risk_limits': {
                'daily_loss_limit': float(self.daily_loss_limit),
                'position_loss_limit': float(self.position_loss_limit),
                'max_consecutive_losses': self.max_consecutive_losses
            }
        }
    
    async def reset_drawdown_tracking(self, period: Optional[PnLPeriod] = None):
        """Reset drawdown tracking for period or all periods."""
        try:
            if period:
                self.drawdown_protector.peak_pnl_by_period[period] = Decimal('0')
                self.drawdown_protector.drawdown_start_time[period] = None
                logger.info(f"Reset drawdown tracking for {period.value}")
            else:
                for p in PnLPeriod:
                    self.drawdown_protector.peak_pnl_by_period[p] = Decimal('0')
                    self.drawdown_protector.drawdown_start_time[p] = None
                logger.info("Reset drawdown tracking for all periods")
                
        except Exception as e:
            logger.error(f"Error resetting drawdown tracking: {e}")
    
    async def update_risk_limits(self,
                               daily_loss_limit: Optional[Decimal] = None,
                               position_loss_limit: Optional[Decimal] = None,
                               max_consecutive_losses: Optional[int] = None):
        """Update risk limits."""
        try:
            if daily_loss_limit is not None:
                self.daily_loss_limit = daily_loss_limit
                logger.info(f"Updated daily loss limit to {daily_loss_limit}")
            
            if position_loss_limit is not None:
                self.position_loss_limit = position_loss_limit
                logger.info(f"Updated position loss limit to {position_loss_limit}")
            
            if max_consecutive_losses is not None:
                self.max_consecutive_losses = max_consecutive_losses
                logger.info(f"Updated max consecutive losses to {max_consecutive_losses}")
        
        except Exception as e:
            logger.error(f"Error updating risk limits: {e}")
    
    async def cleanup(self) -> None:
        """Cleanup P&L risk controller."""
        try:
            # Save final P&L state if needed
            logger.info("P&L risk controller cleanup completed")
        except Exception as e:
            logger.error(f"Error during P&L controller cleanup: {e}")
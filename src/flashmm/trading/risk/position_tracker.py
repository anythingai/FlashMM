"""
FlashMM Enhanced Position Tracker

Advanced position management with real-time P&L calculation, inventory control,
risk metrics, and hedging recommendations for market making operations.
"""

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any

from flashmm.config.settings import get_config
from flashmm.data.storage.redis_client import RedisClient
from flashmm.utils.decorators import measure_latency, timeout_async
from flashmm.utils.exceptions import PositionError
from flashmm.utils.logging import get_logger

logger = get_logger(__name__)


class PositionStatus(Enum):
    """Position status enumeration."""
    NORMAL = "normal"
    WARNING = "warning"      # Approaching limits
    CRITICAL = "critical"    # Exceeding limits
    EMERGENCY = "emergency"  # Requires immediate action


class HedgingRecommendation(Enum):
    """Hedging recommendation types."""
    NONE = "none"
    REDUCE_LONG = "reduce_long"
    REDUCE_SHORT = "reduce_short"
    HEDGE_FULL = "hedge_full"
    HEDGE_PARTIAL = "hedge_partial"


@dataclass
class Trade:
    """Individual trade record."""
    trade_id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    size: Decimal
    price: Decimal
    fee: Decimal
    fee_currency: str
    timestamp: datetime
    order_id: str | None = None
    is_maker: bool = True

    @property
    def notional_value(self) -> Decimal:
        """Calculate notional value of the trade."""
        return self.size * self.price

    def to_dict(self) -> dict[str, Any]:
        """Convert trade to dictionary."""
        return {
            'trade_id': self.trade_id,
            'symbol': self.symbol,
            'side': self.side,
            'size': float(self.size),
            'price': float(self.price),
            'fee': float(self.fee),
            'fee_currency': self.fee_currency,
            'timestamp': self.timestamp.isoformat(),
            'order_id': self.order_id,
            'is_maker': self.is_maker,
            'notional_value': float(self.notional_value)
        }


@dataclass
class PnLBreakdown:
    """Detailed P&L breakdown."""
    realized_pnl: Decimal = Decimal('0')
    unrealized_pnl: Decimal = Decimal('0')
    total_pnl: Decimal = field(init=False)

    # Fee breakdown
    maker_fees_earned: Decimal = Decimal('0')
    taker_fees_paid: Decimal = Decimal('0')
    net_fees: Decimal = field(init=False)

    # Volume metrics
    total_volume: Decimal = Decimal('0')
    buy_volume: Decimal = Decimal('0')
    sell_volume: Decimal = Decimal('0')

    # Trade counts
    total_trades: int = 0
    maker_trades: int = 0
    taker_trades: int = 0

    def __post_init__(self):
        """Calculate derived values."""
        self.total_pnl = self.realized_pnl + self.unrealized_pnl
        self.net_fees = self.maker_fees_earned - self.taker_fees_paid

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            'realized_pnl': float(self.realized_pnl),
            'unrealized_pnl': float(self.unrealized_pnl),
            'total_pnl': float(self.total_pnl),
            'maker_fees_earned': float(self.maker_fees_earned),
            'taker_fees_paid': float(self.taker_fees_paid),
            'net_fees': float(self.net_fees),
            'total_volume': float(self.total_volume),
            'buy_volume': float(self.buy_volume),
            'sell_volume': float(self.sell_volume),
            'total_trades': self.total_trades,
            'maker_trades': self.maker_trades,
            'taker_trades': self.taker_trades
        }


@dataclass
class RiskMetrics:
    """Risk metrics for a position."""
    var_1d: Decimal = Decimal('0')      # 1-day Value at Risk
    var_1d_pct: float = 0.0             # 1-day VaR as percentage
    max_drawdown: Decimal = Decimal('0') # Maximum drawdown
    sharpe_ratio: float = 0.0           # Sharpe ratio
    volatility: float = 0.0             # Position volatility
    beta: float = 0.0                   # Beta vs market
    inventory_ratio: float = 0.0        # Current inventory as % of limit
    time_weighted_exposure: Decimal = Decimal('0')  # Time-weighted exposure

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            'var_1d': float(self.var_1d),
            'var_1d_pct': self.var_1d_pct,
            'max_drawdown': float(self.max_drawdown),
            'sharpe_ratio': self.sharpe_ratio,
            'volatility': self.volatility,
            'beta': self.beta,
            'inventory_ratio': self.inventory_ratio,
            'time_weighted_exposure': float(self.time_weighted_exposure)
        }


@dataclass
class Position:
    """Enhanced position with comprehensive tracking."""
    symbol: str
    base_balance: Decimal = Decimal('0')
    quote_balance: Decimal = Decimal('0')

    # Position metrics
    average_price: Decimal = Decimal('0')
    mark_price: Decimal = Decimal('0')
    notional_value: Decimal = Decimal('0')
    inventory_target: Decimal = Decimal('0')
    inventory_deviation: Decimal = field(init=False)

    # P&L and risk
    pnl: PnLBreakdown = field(default_factory=PnLBreakdown)
    risk_metrics: RiskMetrics = field(default_factory=RiskMetrics)

    # Status and recommendations
    status: PositionStatus = PositionStatus.NORMAL
    hedging_recommendation: HedgingRecommendation = HedgingRecommendation.NONE

    # Trade history
    trades: list[Trade] = field(default_factory=list)

    # Timestamps
    last_trade_time: datetime | None = None
    last_updated: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        """Calculate derived values."""
        self.inventory_deviation = self.base_balance - self.inventory_target
        self.notional_value = abs(self.base_balance * self.mark_price)

    @property
    def is_long(self) -> bool:
        """Check if position is long."""
        return self.base_balance > 0

    @property
    def is_short(self) -> bool:
        """Check if position is short."""
        return self.base_balance < 0

    @property
    def is_flat(self) -> bool:
        """Check if position is flat."""
        return abs(self.base_balance) < Decimal('0.001')  # Small tolerance

    def add_trade(self, trade: Trade) -> None:
        """Add a trade and update position."""
        self.trades.append(trade)
        self.last_trade_time = trade.timestamp
        self.last_updated = datetime.now()

        # Update balances
        if trade.side == 'buy':
            self.base_balance += trade.size
            self.quote_balance -= trade.notional_value
        else:  # sell
            self.base_balance -= trade.size
            self.quote_balance += trade.notional_value

        # Update P&L metrics
        self.pnl.total_trades += 1
        self.pnl.total_volume += trade.notional_value

        if trade.side == 'buy':
            self.pnl.buy_volume += trade.notional_value
        else:
            self.pnl.sell_volume += trade.notional_value

        if trade.is_maker:
            self.pnl.maker_trades += 1
            self.pnl.maker_fees_earned += trade.fee
        else:
            self.pnl.taker_trades += 1
            self.pnl.taker_fees_paid += trade.fee

        # Recalculate average price
        self._update_average_price()

        # Recalculate derived values
        self.inventory_deviation = self.base_balance - self.inventory_target
        self.notional_value = abs(self.base_balance * self.mark_price)

    def _update_average_price(self) -> None:
        """Update average price based on trade history."""
        if not self.trades or self.base_balance == 0:
            self.average_price = Decimal('0')
            return

        # Calculate weighted average price
        total_cost = Decimal('0')
        total_size = Decimal('0')

        for trade in self.trades:
            if trade.side == 'buy':
                total_cost += trade.notional_value
                total_size += trade.size
            else:  # sell
                total_cost -= trade.notional_value
                total_size -= trade.size

        if total_size != 0:
            self.average_price = abs(total_cost / total_size)

    def update_mark_price(self, new_mark_price: Decimal) -> None:
        """Update mark price and recalculate unrealized P&L."""
        self.mark_price = new_mark_price
        self.last_updated = datetime.now()

        # Calculate unrealized P&L
        if not self.is_flat and self.average_price > 0:
            if self.is_long:
                self.pnl.unrealized_pnl = self.base_balance * (self.mark_price - self.average_price)
            else:
                self.pnl.unrealized_pnl = abs(self.base_balance) * (self.average_price - self.mark_price)
        else:
            self.pnl.unrealized_pnl = Decimal('0')

        # Update total P&L
        self.pnl.total_pnl = self.pnl.realized_pnl + self.pnl.unrealized_pnl

        # Update notional value
        self.notional_value = abs(self.base_balance * self.mark_price)

    def to_dict(self) -> dict[str, Any]:
        """Convert position to dictionary."""
        return {
            'symbol': self.symbol,
            'base_balance': float(self.base_balance),
            'quote_balance': float(self.quote_balance),
            'average_price': float(self.average_price),
            'mark_price': float(self.mark_price),
            'notional_value': float(self.notional_value),
            'inventory_target': float(self.inventory_target),
            'inventory_deviation': float(self.inventory_deviation),
            'is_long': self.is_long,
            'is_short': self.is_short,
            'is_flat': self.is_flat,
            'pnl': self.pnl.to_dict(),
            'risk_metrics': self.risk_metrics.to_dict(),
            'status': self.status.value,
            'hedging_recommendation': self.hedging_recommendation.value,
            'last_trade_time': self.last_trade_time.isoformat() if self.last_trade_time else None,
            'last_updated': self.last_updated.isoformat(),
            'trade_count': len(self.trades)
        }


class InventoryController:
    """Inventory control and skewing logic."""

    def __init__(self, position_tracker: 'PositionTracker'):
        self.position_tracker = position_tracker
        self.config = get_config()

        # Inventory limits
        self.max_inventory_ratio = self.config.get("trading.max_inventory_ratio", 0.02)  # 2%
        self.warning_inventory_ratio = self.config.get("trading.warning_inventory_ratio", 0.015)  # 1.5%
        self.max_inventory_usdc = self.config.get("trading.max_inventory_usdc", 2000.0)

        # Skewing parameters
        self.max_skew_bps = self.config.get("trading.max_skew_bps", 20.0)
        self.skew_sensitivity = self.config.get("trading.skew_sensitivity", 2.0)

    def calculate_inventory_skew(self, position: Position) -> float:
        """Calculate inventory skew for quote adjustment."""
        if self.max_inventory_usdc == 0:
            return 0.0

        # Calculate inventory ratio
        inventory_ratio = float(position.notional_value) / self.max_inventory_usdc

        # Direction of skew (positive = skew quotes away from position)
        if position.is_long:
            skew_direction = 1.0  # Widen asks, tighten bids
        elif position.is_short:
            skew_direction = -1.0  # Widen bids, tighten asks
        else:
            return 0.0

        # Calculate skew magnitude
        skew_magnitude = min(inventory_ratio * self.skew_sensitivity, 1.0)

        return skew_direction * skew_magnitude

    def get_position_status(self, position: Position) -> PositionStatus:
        """Determine position status based on inventory levels."""
        inventory_ratio = float(position.notional_value) / self.max_inventory_usdc

        if inventory_ratio >= self.max_inventory_ratio:
            return PositionStatus.CRITICAL
        elif inventory_ratio >= self.warning_inventory_ratio:
            return PositionStatus.WARNING
        else:
            return PositionStatus.NORMAL

    def get_hedging_recommendation(self, position: Position) -> HedgingRecommendation:
        """Get hedging recommendation based on position size."""
        inventory_ratio = float(position.notional_value) / self.max_inventory_usdc

        if inventory_ratio < self.warning_inventory_ratio:
            return HedgingRecommendation.NONE

        if position.is_long:
            if inventory_ratio >= self.max_inventory_ratio:
                return HedgingRecommendation.HEDGE_FULL
            else:
                return HedgingRecommendation.REDUCE_LONG
        elif position.is_short:
            if inventory_ratio >= self.max_inventory_ratio:
                return HedgingRecommendation.HEDGE_FULL
            else:
                return HedgingRecommendation.REDUCE_SHORT

        return HedgingRecommendation.NONE


class RiskCalculator:
    """Risk metrics calculation."""

    def __init__(self, position_tracker: 'PositionTracker'):
        self.position_tracker = position_tracker
        self.price_history: dict[str, list[tuple[datetime, Decimal]]] = {}

    async def calculate_risk_metrics(self, position: Position) -> RiskMetrics:
        """Calculate comprehensive risk metrics."""
        metrics = RiskMetrics()

        try:
            # Calculate basic metrics
            metrics.inventory_ratio = float(position.notional_value) / self.position_tracker.max_inventory_usdc

            # Calculate VaR (simplified)
            if position.notional_value > 0:
                # Use 2% as daily volatility estimate (would use historical data in production)
                daily_vol = 0.02
                z_score = 1.645  # 95% confidence

                metrics.var_1d = position.notional_value * Decimal(str(daily_vol * z_score))
                metrics.var_1d_pct = daily_vol * z_score * 100
                metrics.volatility = daily_vol

            # Calculate time-weighted exposure
            if position.trades:
                metrics.time_weighted_exposure = self._calculate_time_weighted_exposure(position)

            # Update price history for the symbol
            self._update_price_history(position.symbol, position.mark_price)

            # Calculate max drawdown
            metrics.max_drawdown = self._calculate_max_drawdown(position)

            return metrics

        except Exception as e:
            logger.error(f"Risk calculation failed for {position.symbol}: {e}")
            return metrics

    def _calculate_time_weighted_exposure(self, position: Position) -> Decimal:
        """Calculate time-weighted exposure."""
        if not position.trades:
            return Decimal('0')

        # Simplified calculation - would be more sophisticated in production
        current_time = datetime.now()
        total_exposure = Decimal('0')

        for trade in position.trades[-10:]:  # Last 10 trades
            time_weight = max(1.0, (current_time - trade.timestamp).total_seconds() / 3600)  # Hours
            exposure = trade.notional_value / Decimal(str(time_weight))
            total_exposure += exposure

        return total_exposure

    def _update_price_history(self, symbol: str, price: Decimal) -> None:
        """Update price history for volatility calculations."""
        if symbol not in self.price_history:
            self.price_history[symbol] = []

        self.price_history[symbol].append((datetime.now(), price))

        # Keep only last 24 hours of data
        cutoff_time = datetime.now() - timedelta(hours=24)
        self.price_history[symbol] = [
            (ts, p) for ts, p in self.price_history[symbol] if ts > cutoff_time
        ]

    def _calculate_max_drawdown(self, position: Position) -> Decimal:
        """Calculate maximum drawdown."""
        if not position.trades:
            return Decimal('0')

        # Simplified calculation
        peak_pnl = Decimal('0')
        max_drawdown = Decimal('0')
        running_pnl = Decimal('0')

        for trade in position.trades:
            if trade.side == 'buy':
                running_pnl -= trade.notional_value
            else:
                running_pnl += trade.notional_value

            peak_pnl = max(peak_pnl, running_pnl)
            drawdown = peak_pnl - running_pnl
            max_drawdown = max(max_drawdown, drawdown)

        return max_drawdown


class PositionTracker:
    """Enhanced position tracker with comprehensive risk management."""

    def __init__(self):
        self.config = get_config()
        self.redis_client: RedisClient | None = None

        # Position storage
        self.positions: dict[str, Position] = {}

        # Components
        self.inventory_controller = InventoryController(self)
        self.risk_calculator = RiskCalculator(self)

        # Configuration
        self.max_inventory_usdc = self.config.get("trading.max_inventory_usdc", 2000.0)
        self.update_interval_seconds = self.config.get("trading.position_update_interval_seconds", 10)
        self.enable_redis_persistence = self.config.get("trading.enable_redis_persistence", True)

        # Background tasks
        self._update_task: asyncio.Task | None = None

        # Statistics
        self.total_trades_processed = 0
        self.total_volume_processed = Decimal('0')
        self.alerts_generated = 0

    async def initialize(self) -> None:
        """Initialize the enhanced position tracker."""
        try:
            # Initialize Redis client if persistence is enabled
            if self.enable_redis_persistence:
                self.redis_client = RedisClient()
                await self.redis_client.initialize()
                await self._load_positions()

            # Start background tasks
            await self._start_background_tasks()

            logger.info("Enhanced PositionTracker initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize PositionTracker: {e}")
            raise PositionError(f"PositionTracker initialization failed: {e}") from e

    async def _start_background_tasks(self) -> None:
        """Start background position update tasks."""
        self._update_task = asyncio.create_task(self._position_update_loop())

    async def _position_update_loop(self) -> None:
        """Background position update loop."""
        while True:
            try:
                await asyncio.sleep(self.update_interval_seconds)
                await self._update_all_positions()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Position update loop error: {e}")
                await asyncio.sleep(5)

    async def _update_all_positions(self) -> None:
        """Update all positions with current market data."""
        for symbol, position in self.positions.items():
            try:
                # In production, get real market price
                # For now, simulate small price movements
                if position.mark_price > 0:
                    import secrets
                    price_change = (secrets.randbelow(200) - 100) / 10000  # ±1%
                    new_price = position.mark_price * Decimal(str(1 + price_change))
                    await self.update_mark_price(symbol, float(new_price))
            except Exception as e:
                logger.error(f"Failed to update position for {symbol}: {e}")

    async def _load_positions(self) -> None:
        """Load positions from Redis storage."""
        if not self.redis_client:
            return

        try:
            keys = await self.redis_client.keys("position:*")
            for key in keys:
                symbol = key.split(":")[1]
                position_data = await self.redis_client.get(key)
                if position_data:
                    data = json.loads(position_data)
                    position = self._deserialize_position(data)
                    self.positions[symbol] = position

            logger.info(f"Loaded {len(self.positions)} positions from storage")

        except Exception as e:
            logger.error(f"Failed to load positions: {e}")

    async def _save_position(self, position: Position) -> None:
        """Save position to Redis storage."""
        if not self.redis_client:
            return

        try:
            position_data = json.dumps(self._serialize_position(position))
            await self.redis_client.set(
                f"position:{position.symbol}",
                position_data,
                expire=86400  # 24 hours
            )
        except Exception as e:
            logger.error(f"Failed to save position for {position.symbol}: {e}")

    def _serialize_position(self, position: Position) -> dict[str, Any]:
        """Serialize position for storage."""
        data = position.to_dict()
        # Add trade history
        data['trades'] = [trade.to_dict() for trade in position.trades]
        return data

    def _deserialize_position(self, data: dict[str, Any]) -> Position:
        """Deserialize position from storage."""
        # Create position with basic data
        position = Position(
            symbol=data['symbol'],
            base_balance=Decimal(str(data['base_balance'])),
            quote_balance=Decimal(str(data['quote_balance'])),
            average_price=Decimal(str(data['average_price'])),
            mark_price=Decimal(str(data['mark_price'])),
            inventory_target=Decimal(str(data.get('inventory_target', 0))),
            last_updated=datetime.fromisoformat(data['last_updated'])
        )

        # Restore trade history
        if 'trades' in data:
            for trade_data in data['trades']:
                trade = Trade(
                    trade_id=trade_data['trade_id'],
                    symbol=trade_data['symbol'],
                    side=trade_data['side'],
                    size=Decimal(str(trade_data['size'])),
                    price=Decimal(str(trade_data['price'])),
                    fee=Decimal(str(trade_data['fee'])),
                    fee_currency=trade_data['fee_currency'],
                    timestamp=datetime.fromisoformat(trade_data['timestamp']),
                    order_id=trade_data.get('order_id'),
                    is_maker=trade_data.get('is_maker', True)
                )
                position.trades.append(trade)

        return position

    @timeout_async(0.05)  # 50ms timeout for position updates
    @measure_latency("position_update")
    async def process_trade(
        self,
        symbol: str,
        side: str,
        size: float,
        price: float,
        fee: float = 0.0,
        fee_currency: str = "USDC",
        order_id: str | None = None,
        is_maker: bool = True
    ) -> None:
        """Process a trade and update position."""
        try:
            # Get or create position
            position = self.positions.get(symbol)
            if not position:
                position = Position(
                    symbol=symbol,
                    mark_price=Decimal(str(price))
                )
                self.positions[symbol] = position

            # Create trade record
            trade = Trade(
                trade_id=f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
                symbol=symbol,
                side=side,
                size=Decimal(str(size)),
                price=Decimal(str(price)),
                fee=Decimal(str(fee)),
                fee_currency=fee_currency,
                timestamp=datetime.now(),
                order_id=order_id,
                is_maker=is_maker
            )

            # Add trade to position
            position.add_trade(trade)

            # Update mark price
            position.update_mark_price(Decimal(str(price)))

            # Update inventory control metrics
            position.status = self.inventory_controller.get_position_status(position)
            position.hedging_recommendation = self.inventory_controller.get_hedging_recommendation(position)

            # Update risk metrics
            position.risk_metrics = await self.risk_calculator.calculate_risk_metrics(position)

            # Generate alerts if necessary
            await self._check_position_alerts(position)

            # Save position
            if self.enable_redis_persistence:
                await self._save_position(position)

            # Update statistics
            self.total_trades_processed += 1
            self.total_volume_processed += trade.notional_value

            logger.debug(f"Processed trade for {symbol}: {side} {size} @ {price}")

        except Exception as e:
            logger.error(f"Failed to process trade for {symbol}: {e}")
            raise PositionError(f"Trade processing failed: {e}") from e

    async def update_mark_price(self, symbol: str, price: float) -> None:
        """Update mark price for a position."""
        position = self.positions.get(symbol)
        if position:
            position.update_mark_price(Decimal(str(price)))

            # Update risk metrics
            position.risk_metrics = await self.risk_calculator.calculate_risk_metrics(position)

            # Save updated position
            if self.enable_redis_persistence:
                await self._save_position(position)

    async def _check_position_alerts(self, position: Position) -> None:
        """Check position for alert conditions."""
        alerts = []

        # Inventory limit alerts
        if position.status == PositionStatus.CRITICAL:
            alerts.append({
                'type': 'inventory_critical',
                'symbol': position.symbol,
                'message': f'Inventory critical: {position.notional_value} USDC',
                'recommendation': position.hedging_recommendation.value
            })
        elif position.status == PositionStatus.WARNING:
            alerts.append({
                'type': 'inventory_warning',
                'symbol': position.symbol,
                'message': f'Inventory warning: {position.notional_value} USDC'
            })

        # P&L alerts
        if position.pnl.total_pnl < -100:  # $100 loss
            alerts.append({
                'type': 'pnl_loss',
                'symbol': position.symbol,
                'message': f'Position loss: ${position.pnl.total_pnl}'
            })

        # Log alerts
        for alert in alerts:
            logger.warning(f"Position alert: {alert}")
            self.alerts_generated += 1

    # Public interface methods
    async def update_position(self, symbol: str, side: str, size: float, price: float) -> None:
        """Update position based on trade execution."""
        try:
            await self.process_trade(
                symbol=symbol,
                side=side,
                size=size,
                price=price
            )
        except Exception as e:
            logger.error(f"Error updating position for {symbol}: {e}")
            raise

    async def get_position(self, symbol: str) -> dict[str, Any]:
        """Get current position for symbol."""
        position = self.positions.get(symbol)
        if position:
            return position.to_dict()
        else:
            # Return empty position
            return Position(symbol=symbol).to_dict()

    async def get_all_positions(self) -> dict[str, dict[str, Any]]:
        """Get all current positions."""
        return {symbol: pos.to_dict() for symbol, pos in self.positions.items()}

    def get_inventory_skew(self, symbol: str) -> float:
        """Get inventory skew recommendation for symbol."""
        position = self.positions.get(symbol)
        if position:
            return self.inventory_controller.calculate_inventory_skew(position)
        return 0.0

    def get_hedging_recommendation(self, symbol: str) -> str:
        """Get hedging recommendation for symbol."""
        position = self.positions.get(symbol)
        if position:
            return position.hedging_recommendation.value
        return HedgingRecommendation.NONE.value

    def get_portfolio_summary(self) -> dict[str, Any]:
        """Get portfolio-level summary."""
        total_pnl = sum(pos.pnl.total_pnl for pos in self.positions.values())
        total_volume = sum(pos.pnl.total_volume for pos in self.positions.values())
        total_notional = sum(pos.notional_value for pos in self.positions.values())
        total_fees = sum(pos.pnl.net_fees for pos in self.positions.values())

        # Count position statuses
        status_counts = {}
        for position in self.positions.values():
            status = position.status.value
            status_counts[status] = status_counts.get(status, 0) + 1

        return {
            'total_positions': len(self.positions),
            'total_pnl': float(total_pnl),
            'total_volume': float(total_volume),
            'total_notional_exposure': float(total_notional),
            'total_net_fees': float(total_fees),
            'total_trades_processed': self.total_trades_processed,
            'alerts_generated': self.alerts_generated,
            'position_status_counts': status_counts,
            'max_inventory_limit': self.max_inventory_usdc,
            'portfolio_utilization': float(total_notional) / self.max_inventory_usdc if self.max_inventory_usdc > 0 else 0.0
        }

    def get_performance_stats(self) -> dict[str, Any]:
        """Get position tracker performance statistics."""
        return {
            'total_trades_processed': self.total_trades_processed,
            'total_volume_processed': float(self.total_volume_processed),
            'alerts_generated': self.alerts_generated,
            'positions_tracked': len(self.positions),
            'update_interval_seconds': self.update_interval_seconds,
            'redis_persistence_enabled': self.enable_redis_persistence,
            'max_inventory_usdc': self.max_inventory_usdc
        }

    async def set_inventory_target(self, symbol: str, target: float) -> None:
        """Set inventory target for a symbol."""
        position = self.positions.get(symbol)
        if not position:
            position = Position(symbol=symbol)
            self.positions[symbol] = position

        position.inventory_target = Decimal(str(target))
        position.inventory_deviation = position.base_balance - position.inventory_target

        logger.info(f"Set inventory target for {symbol}: {target}")

    async def reset_position(self, symbol: str) -> None:
        """Reset position for a symbol (used for testing/maintenance)."""
        if symbol in self.positions:
            del self.positions[symbol]

            # Remove from Redis
            if self.redis_client:
                await self.redis_client.delete(f"position:{symbol}")

            logger.info(f"Reset position for {symbol}")

    def check_inventory_compliance(self, symbol: str) -> dict[str, Any]:
        """Check inventory compliance for a symbol."""
        position = self.positions.get(symbol)
        if not position:
            return {
                'compliant': True,
                'inventory_ratio': 0.0,
                'limit_utilization': 0.0,
                'recommendation': 'none'
            }

        inventory_ratio = float(position.notional_value) / self.max_inventory_usdc
        is_compliant = inventory_ratio <= self.inventory_controller.max_inventory_ratio

        return {
            'compliant': is_compliant,
            'inventory_ratio': inventory_ratio,
            'limit_utilization': inventory_ratio / self.inventory_controller.max_inventory_ratio,
            'status': position.status.value,
            'recommendation': position.hedging_recommendation.value,
            'notional_value': float(position.notional_value),
            'inventory_limit': self.max_inventory_usdc
        }

    async def cleanup(self) -> None:
        """Cleanup resources and stop background tasks."""
        # Cancel background tasks
        if self._update_task and not self._update_task.done():
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass

        # Close Redis connection
        if self.redis_client:
            await self.redis_client.close()

        logger.info("PositionTracker cleanup completed")


# Legacy compatibility methods
async def update_position(symbol: str, side: str, size: float, price: float) -> None:
    """Legacy method for backward compatibility."""
    # This would be called by the global position tracker instance
    pass


async def get_position_legacy(symbol: str) -> dict[str, Any]:
    """Legacy method for backward compatibility."""
    # This would be called by the global position tracker instance
    return {
        "symbol": symbol,
        "base_balance": 0.0,
        "quote_balance": 0.0,
        "value_usdc": 0.0,
        "unrealized_pnl": 0.0,
        "last_updated": datetime.now().isoformat(),
    }

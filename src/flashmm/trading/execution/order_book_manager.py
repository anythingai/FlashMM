"""
FlashMM Order Book Manager

Manages order book state synchronization, order replacement strategies,
and performance analytics for market making operations.
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any

from flashmm.config.settings import get_config
from flashmm.trading.execution.order_router import Order, OrderRouter, OrderStatus
from flashmm.utils.decorators import measure_latency
from flashmm.utils.logging import get_logger

logger = get_logger(__name__)


class OrderReplacementStrategy(Enum):
    """Order replacement strategies."""
    CANCEL_REPLACE = "cancel_replace"  # Cancel then place new order
    MODIFY_ORDER = "modify_order"      # Modify existing order in place
    HYBRID = "hybrid"                  # Choose best strategy per situation


class OrderAgingPolicy(Enum):
    """Order aging policies."""
    TIME_BASED = "time_based"          # Replace based on time elapsed
    MARKET_BASED = "market_based"      # Replace based on market conditions
    ADAPTIVE = "adaptive"              # Combine time and market factors


@dataclass
class OrderBookLevel:
    """Order book level tracking our orders."""
    price: Decimal
    our_size: Decimal
    market_size: Decimal
    order_ids: set[str]
    last_updated: datetime
    competition_score: float = 0.0  # How competitive our orders are


@dataclass
class OrderBookState:
    """Current order book state for a symbol."""
    symbol: str
    bid_levels: dict[str, OrderBookLevel]  # price -> level
    ask_levels: dict[str, OrderBookLevel]  # price -> level
    best_bid: Decimal | None
    best_ask: Decimal | None
    our_best_bid: Decimal | None
    our_best_ask: Decimal | None
    mid_price: Decimal | None
    spread_bps: float
    last_updated: datetime
    sequence_number: int = 0

    def get_our_orders_count(self) -> int:
        """Get total count of our orders."""
        count = 0
        for level in self.bid_levels.values():
            count += len(level.order_ids)
        for level in self.ask_levels.values():
            count += len(level.order_ids)
        return count

    def get_our_total_size(self) -> tuple[Decimal, Decimal]:
        """Get our total bid and ask sizes."""
        bid_size = sum(level.our_size for level in self.bid_levels.values())
        ask_size = sum(level.our_size for level in self.ask_levels.values())
        return bid_size if bid_size else Decimal('0'), ask_size if ask_size else Decimal('0')


@dataclass
class OrderPerformanceMetrics:
    """Order performance metrics."""
    order_id: str
    symbol: str
    placed_at: datetime
    current_status: OrderStatus
    time_to_acknowledge_ms: float | None = None
    time_in_market_seconds: float = 0.0
    fill_rate: float = 0.0
    average_fill_price: Decimal | None = None
    total_fees: Decimal = Decimal('0')
    profitability: float = 0.0
    market_impact_bps: float = 0.0
    adverse_selection_score: float = 0.0
    replacement_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            'order_id': self.order_id,
            'symbol': self.symbol,
            'placed_at': self.placed_at.isoformat(),
            'current_status': self.current_status.value,
            'time_to_acknowledge_ms': self.time_to_acknowledge_ms,
            'time_in_market_seconds': self.time_in_market_seconds,
            'fill_rate': self.fill_rate,
            'average_fill_price': float(self.average_fill_price) if self.average_fill_price else None,
            'total_fees': float(self.total_fees),
            'profitability': self.profitability,
            'market_impact_bps': self.market_impact_bps,
            'adverse_selection_score': self.adverse_selection_score,
            'replacement_count': self.replacement_count
        }


@dataclass
class OrderConflictDetection:
    """Order conflict detection result."""
    has_conflicts: bool
    conflicting_orders: list[str]
    conflict_type: str
    resolution_strategy: str
    priority_order: str | None = None


class OrderConflictResolver:
    """Resolves order conflicts and race conditions."""

    def __init__(self, order_book_manager: 'OrderBookManager'):
        self.manager = order_book_manager
        self.conflict_history: list[dict[str, Any]] = []

    async def detect_conflicts(self, symbol: str) -> OrderConflictDetection:
        """Detect order conflicts for a symbol."""
        try:
            state = self.manager.get_order_book_state(symbol)
            if not state:
                return OrderConflictDetection(False, [], "none", "none")

            conflicts = []
            conflict_type = "none"

            # Check for price level conflicts (multiple orders at same price)
            for _price, level in state.bid_levels.items():
                if len(level.order_ids) > 1:
                    conflicts.extend(level.order_ids)
                    conflict_type = "price_level_duplicate"

            for _price, level in state.ask_levels.items():
                if len(level.order_ids) > 1:
                    conflicts.extend(level.order_ids)
                    conflict_type = "price_level_duplicate"

            # Check for crossing orders (bid >= ask)
            if state.our_best_bid and state.our_best_ask:
                if state.our_best_bid >= state.our_best_ask:
                    # Find crossing orders
                    crossing_orders = []
                    for level in state.bid_levels.values():
                        if level.price >= state.our_best_ask:
                            crossing_orders.extend(level.order_ids)
                    for level in state.ask_levels.values():
                        if level.price <= state.our_best_bid:
                            crossing_orders.extend(level.order_ids)

                    conflicts.extend(crossing_orders)
                    conflict_type = "crossing_orders"

            # Determine resolution strategy
            resolution_strategy = "cancel_all" if conflict_type == "crossing_orders" else "cancel_duplicates"

            has_conflicts = len(conflicts) > 0

            if has_conflicts:
                logger.warning(f"Order conflicts detected for {symbol}: {conflict_type}")
                self.conflict_history.append({
                    'symbol': symbol,
                    'timestamp': datetime.now(),
                    'conflict_type': conflict_type,
                    'conflicting_orders': conflicts.copy(),
                    'resolution_strategy': resolution_strategy
                })

            return OrderConflictDetection(
                has_conflicts=has_conflicts,
                conflicting_orders=conflicts,
                conflict_type=conflict_type,
                resolution_strategy=resolution_strategy
            )

        except Exception as e:
            logger.error(f"Conflict detection failed for {symbol}: {e}")
            return OrderConflictDetection(False, [], "error", "none")

    async def resolve_conflicts(self, conflict: OrderConflictDetection) -> bool:
        """Resolve detected conflicts."""
        try:
            if not conflict.has_conflicts:
                return True

            if conflict.resolution_strategy == "cancel_all":
                # Cancel all conflicting orders
                for order_id in conflict.conflicting_orders:
                    await self.manager.order_router.cancel_order(order_id)
                return True

            elif conflict.resolution_strategy == "cancel_duplicates":
                # Keep the most recent order at each price level
                orders_to_cancel = conflict.conflicting_orders[:-1]  # Keep last one
                for order_id in orders_to_cancel:
                    await self.manager.order_router.cancel_order(order_id)
                return True

            return False

        except Exception as e:
            logger.error(f"Conflict resolution failed: {e}")
            return False


class OrderAgingManager:
    """Manages order aging and refresh logic."""

    def __init__(self, order_book_manager: 'OrderBookManager'):
        self.manager = order_book_manager
        self.config = get_config()

        # Aging configuration
        self.max_order_age_seconds = self.config.get("trading.max_order_age_seconds", 300)  # 5 minutes
        self.refresh_threshold_seconds = self.config.get("trading.refresh_threshold_seconds", 60)  # 1 minute
        self.market_change_threshold_bps = self.config.get("trading.market_change_threshold_bps", 5.0)

    async def check_aging_orders(self, symbol: str) -> list[str]:
        """Check for orders that need aging/refresh."""
        try:
            aging_orders = []
            current_time = datetime.now()

            # Get active orders for symbol
            active_orders = self.manager.order_router.get_orders_for_symbol(
                symbol, {OrderStatus.ACTIVE, OrderStatus.PARTIALLY_FILLED}
            )

            for order in active_orders:
                age_seconds = (current_time - order.created_at).total_seconds()

                # Check time-based aging
                if age_seconds > self.max_order_age_seconds:
                    aging_orders.append(order.order_id)
                    continue

                # Check market-based aging
                if age_seconds > self.refresh_threshold_seconds:
                    if await self._should_refresh_for_market_changes(order):
                        aging_orders.append(order.order_id)

            return aging_orders

        except Exception as e:
            logger.error(f"Order aging check failed for {symbol}: {e}")
            return []

    async def _should_refresh_for_market_changes(self, order: Order) -> bool:
        """Check if order should be refreshed due to market changes."""
        try:
            state = self.manager.get_order_book_state(order.symbol)
            if not state or not state.mid_price:
                return False

            # Calculate price deviation from current mid
            price_deviation = abs(order.price - state.mid_price) / state.mid_price * 10000

            # Refresh if price has moved significantly
            return price_deviation > self.market_change_threshold_bps

        except Exception as e:
            logger.error(f"Market change check failed for order {order.order_id}: {e}")
            return False

    async def refresh_aging_orders(self, symbol: str, aging_orders: list[str]) -> int:
        """Refresh aging orders."""
        refreshed_count = 0

        for order_id in aging_orders:
            try:
                success = await self._refresh_single_order(order_id)
                if success:
                    refreshed_count += 1
            except Exception as e:
                logger.error(f"Failed to refresh order {order_id}: {e}")

        if refreshed_count > 0:
            logger.info(f"Refreshed {refreshed_count}/{len(aging_orders)} aging orders for {symbol}")

        return refreshed_count

    async def _refresh_single_order(self, order_id: str) -> bool:
        """Refresh a single order."""
        order = self.manager.order_router.get_order(order_id)
        if not order:
            return False

        # Create replacement order with updated parameters
        new_order_data = {
            'symbol': order.symbol,
            'side': order.side,
            'price': float(order.price),  # Could update price based on current market
            'size': float(order.remaining_size),  # Use remaining size
            'order_type': order.order_type.value,
            'time_in_force': order.time_in_force.value,
            'tags': {**order.tags, 'refresh_parent': order_id}
        }

        # Use order replacement
        replacement = await self.manager.order_router._replace_order(order_id, new_order_data)
        return replacement is not None


class OrderBookManager:
    """Advanced order book manager with state synchronization."""

    def __init__(self, order_router: OrderRouter):
        self.order_router = order_router
        self.config = get_config()

        # State tracking
        self.order_book_states: dict[str, OrderBookState] = {}
        self.performance_metrics: dict[str, OrderPerformanceMetrics] = {}

        # Components
        self.conflict_resolver = OrderConflictResolver(self)
        self.aging_manager = OrderAgingManager(self)

        # Configuration
        self.sync_interval_seconds = self.config.get("trading.order_sync_interval_seconds", 5)
        self.performance_tracking_enabled = self.config.get("trading.enable_performance_tracking", True)

        # Background tasks
        self._sync_task: asyncio.Task | None = None
        self._aging_task: asyncio.Task | None = None

        # Statistics
        self.sync_operations = 0
        self.conflicts_detected = 0
        self.conflicts_resolved = 0
        self.orders_refreshed = 0

    async def initialize(self) -> None:
        """Initialize the order book manager."""
        try:
            # Start background tasks
            await self._start_background_tasks()

            logger.info("OrderBookManager initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize OrderBookManager: {e}")
            raise

    async def _start_background_tasks(self) -> None:
        """Start background synchronization tasks."""
        self._sync_task = asyncio.create_task(self._synchronization_loop())
        self._aging_task = asyncio.create_task(self._aging_loop())

    async def _synchronization_loop(self) -> None:
        """Background order book synchronization loop."""
        while True:
            try:
                await asyncio.sleep(self.sync_interval_seconds)
                await self._sync_all_symbols()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Synchronization loop error: {e}")
                await asyncio.sleep(1)

    async def _aging_loop(self) -> None:
        """Background order aging management loop."""
        while True:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                await self._process_aging_orders()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Aging loop error: {e}")
                await asyncio.sleep(5)

    @measure_latency("order_book_sync")
    async def _sync_all_symbols(self) -> None:
        """Synchronize order book state for all symbols."""
        symbols = set()
        for order in self.order_router.orders.values():
            if order.is_active():
                symbols.add(order.symbol)

        for symbol in symbols:
            try:
                await self.sync_order_book_state(symbol)
            except Exception as e:
                logger.error(f"Failed to sync {symbol}: {e}")

        self.sync_operations += 1

    async def sync_order_book_state(self, symbol: str) -> OrderBookState:
        """Synchronize order book state for a specific symbol."""
        try:
            # Get our active orders for this symbol
            our_orders = self.order_router.get_orders_for_symbol(
                symbol, {OrderStatus.ACTIVE, OrderStatus.PARTIALLY_FILLED}
            )

            # Create or update order book state
            state = self.order_book_states.get(symbol)
            if not state:
                state = OrderBookState(
                    symbol=symbol,
                    bid_levels={},
                    ask_levels={},
                    best_bid=None,
                    best_ask=None,
                    our_best_bid=None,
                    our_best_ask=None,
                    mid_price=None,
                    spread_bps=0.0,
                    last_updated=datetime.now()
                )
                self.order_book_states[symbol] = state

            # Update levels with our orders
            await self._update_order_levels(state, our_orders)

            # Update market data (would come from market data feed in production)
            await self._update_market_data(state)

            # Calculate derived values
            self._calculate_derived_values(state)

            # Check for conflicts
            conflict = await self.conflict_resolver.detect_conflicts(symbol)
            if conflict.has_conflicts:
                self.conflicts_detected += 1
                resolved = await self.conflict_resolver.resolve_conflicts(conflict)
                if resolved:
                    self.conflicts_resolved += 1

            # Update performance metrics
            if self.performance_tracking_enabled:
                await self._update_performance_metrics(our_orders)

            state.last_updated = datetime.now()
            state.sequence_number += 1

            return state

        except Exception as e:
            logger.error(f"Order book sync failed for {symbol}: {e}")
            raise

    async def _update_order_levels(self, state: OrderBookState, orders: list[Order]) -> None:
        """Update order book levels with our orders."""
        # Clear existing levels
        state.bid_levels.clear()
        state.ask_levels.clear()

        # Group orders by price and side
        for order in orders:
            price_str = str(order.price)

            if order.side == 'buy':
                if price_str not in state.bid_levels:
                    state.bid_levels[price_str] = OrderBookLevel(
                        price=order.price,
                        our_size=Decimal('0'),
                        market_size=Decimal('0'),  # Would be updated from market data
                        order_ids=set(),
                        last_updated=datetime.now()
                    )
                level = state.bid_levels[price_str]
                level.our_size += order.remaining_size
                level.order_ids.add(order.order_id)
                level.last_updated = max(level.last_updated, order.last_updated_at)

            else:  # sell
                if price_str not in state.ask_levels:
                    state.ask_levels[price_str] = OrderBookLevel(
                        price=order.price,
                        our_size=Decimal('0'),
                        market_size=Decimal('0'),  # Would be updated from market data
                        order_ids=set(),
                        last_updated=datetime.now()
                    )
                level = state.ask_levels[price_str]
                level.our_size += order.remaining_size
                level.order_ids.add(order.order_id)
                level.last_updated = max(level.last_updated, order.last_updated_at)

    async def _update_market_data(self, state: OrderBookState) -> None:
        """Update market data (simulated - would come from market data feed)."""
        # In production, this would subscribe to market data feed
        # For now, simulate market data

        if state.bid_levels:
            state.best_bid = max(Decimal(price) for price in state.bid_levels.keys())
            state.our_best_bid = state.best_bid

        if state.ask_levels:
            state.best_ask = min(Decimal(price) for price in state.ask_levels.keys())
            state.our_best_ask = state.best_ask

        # Simulate market best bid/ask (slightly better than ours)
        if state.our_best_bid:
            state.best_bid = state.our_best_bid + Decimal('0.0001')
        if state.our_best_ask:
            state.best_ask = state.our_best_ask - Decimal('0.0001')

    def _calculate_derived_values(self, state: OrderBookState) -> None:
        """Calculate derived values like mid price and spread."""
        if state.best_bid and state.best_ask:
            state.mid_price = (state.best_bid + state.best_ask) / 2
            spread = state.best_ask - state.best_bid
            state.spread_bps = float(spread / state.mid_price * 10000)
        else:
            state.mid_price = None
            state.spread_bps = 0.0

    async def _update_performance_metrics(self, orders: list[Order]) -> None:
        """Update performance metrics for orders."""
        current_time = datetime.now()

        for order in orders:
            if order.order_id not in self.performance_metrics:
                # Create new metrics
                self.performance_metrics[order.order_id] = OrderPerformanceMetrics(
                    order_id=order.order_id,
                    symbol=order.symbol,
                    placed_at=order.created_at,
                    current_status=order.status
                )

            metrics = self.performance_metrics[order.order_id]

            # Update basic metrics
            metrics.current_status = order.status
            metrics.time_in_market_seconds = (current_time - order.created_at).total_seconds()
            metrics.fill_rate = float(order.filled_size / order.size) if order.size > 0 else 0.0
            metrics.average_fill_price = order.average_fill_price
            metrics.total_fees = order.total_fees

            # Calculate time to acknowledge
            if order.acknowledged_at and not metrics.time_to_acknowledge_ms:
                metrics.time_to_acknowledge_ms = (
                    order.acknowledged_at - order.submitted_at
                ).total_seconds() * 1000 if order.submitted_at else None

            # Update replacement count
            metrics.replacement_count = len(order.child_order_ids)

    async def _process_aging_orders(self) -> None:
        """Process aging orders for all symbols."""
        symbols = list(self.order_book_states.keys())

        for symbol in symbols:
            try:
                aging_orders = await self.aging_manager.check_aging_orders(symbol)
                if aging_orders:
                    refreshed = await self.aging_manager.refresh_aging_orders(symbol, aging_orders)
                    self.orders_refreshed += refreshed
            except Exception as e:
                logger.error(f"Aging processing failed for {symbol}: {e}")

    # Public interface methods
    def get_order_book_state(self, symbol: str) -> OrderBookState | None:
        """Get current order book state for symbol."""
        return self.order_book_states.get(symbol)

    def get_performance_metrics(self, order_id: str) -> OrderPerformanceMetrics | None:
        """Get performance metrics for an order."""
        return self.performance_metrics.get(order_id)

    def get_all_performance_metrics(self, symbol: str | None = None) -> list[OrderPerformanceMetrics]:
        """Get all performance metrics, optionally filtered by symbol."""
        metrics = list(self.performance_metrics.values())
        if symbol:
            metrics = [m for m in metrics if m.symbol == symbol]
        return metrics

    async def force_sync(self, symbol: str) -> OrderBookState:
        """Force immediate synchronization for a symbol."""
        return await self.sync_order_book_state(symbol)

    async def check_order_conflicts(self, symbol: str) -> OrderConflictDetection:
        """Manually check for order conflicts."""
        return await self.conflict_resolver.detect_conflicts(symbol)

    def get_statistics(self) -> dict[str, Any]:
        """Get order book manager statistics."""
        return {
            'sync_operations': self.sync_operations,
            'conflicts_detected': self.conflicts_detected,
            'conflicts_resolved': self.conflicts_resolved,
            'conflict_resolution_rate': self.conflicts_resolved / max(self.conflicts_detected, 1),
            'orders_refreshed': self.orders_refreshed,
            'symbols_tracked': len(self.order_book_states),
            'total_performance_metrics': len(self.performance_metrics),
            'last_sync_times': {
                symbol: state.last_updated.isoformat()
                for symbol, state in self.order_book_states.items()
            }
        }

    async def place_order(
        self,
        symbol: str,
        side: str,
        price: float,
        size: float,
        order_type: str = "limit",
        time_in_force: str = "GTC"
    ) -> Order | None:
        """Place order via order router (wrapper method)."""
        return await self.order_router.place_order(
            symbol=symbol,
            side=side,
            price=price,
            size=size,
            order_type=order_type,
            time_in_force=time_in_force
        )

    async def manage_quote_lifecycle(self) -> None:
        """Manage quote lifecycle including aging and conflicts."""
        try:
            symbols = list(self.order_book_states.keys())

            for symbol in symbols:
                # Check aging orders
                aging_orders = await self.aging_manager.check_aging_orders(symbol)
                if aging_orders:
                    await self.aging_manager.refresh_aging_orders(symbol, aging_orders)

                # Check conflicts
                conflict = await self.conflict_resolver.detect_conflicts(symbol)
                if conflict.has_conflicts:
                    await self.conflict_resolver.resolve_conflicts(conflict)

        except Exception as e:
            logger.error(f"Quote lifecycle management error: {e}")

    async def cleanup(self) -> None:
        """Cleanup resources and stop background tasks."""
        # Cancel background tasks
        if self._sync_task and not self._sync_task.done():
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass

        if self._aging_task and not self._aging_task.done():
            self._aging_task.cancel()
            try:
                await self._aging_task
            except asyncio.CancelledError:
                pass

        logger.info("OrderBookManager cleanup completed")

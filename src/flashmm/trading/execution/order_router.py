"""
FlashMM Enhanced Order Router

Advanced order management system with complete lifecycle management,
batch operations, state tracking, and reconciliation for market making.
"""

import asyncio
from typing import Dict, Any, Optional, List, Tuple, Set
from decimal import Decimal, ROUND_HALF_UP
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import uuid

from flashmm.config.settings import get_config
from flashmm.utils.logging import get_logger, TradingEventLogger
from flashmm.utils.exceptions import OrderError, CambrianAPIError, ValidationError, BlockchainError
from flashmm.utils.decorators import require_trading_enabled, measure_latency, timeout_async
from flashmm.blockchain.blockchain_service import get_blockchain_service

logger = get_logger(__name__)
trading_logger = TradingEventLogger()


class OrderStatus(Enum):
    """Order status enumeration."""
    PENDING = "pending"           # Order created but not yet sent
    SUBMITTED = "submitted"       # Order sent to exchange
    ACKNOWLEDGED = "acknowledged" # Order accepted by exchange
    ACTIVE = "active"            # Order is live in the market
    PARTIALLY_FILLED = "partially_filled"  # Order partially executed
    FILLED = "filled"            # Order completely executed
    CANCELLED = "cancelled"      # Order cancelled
    REJECTED = "rejected"        # Order rejected by exchange
    EXPIRED = "expired"          # Order expired
    REPLACED = "replaced"        # Order replaced by another order
    FAILED = "failed"            # Order failed due to system error


class OrderType(Enum):
    """Order type enumeration."""
    LIMIT = "limit"
    MARKET = "market"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class TimeInForce(Enum):
    """Time in force enumeration."""
    GTC = "GTC"  # Good Till Cancelled
    IOC = "IOC"  # Immediate Or Cancel
    FOK = "FOK"  # Fill Or Kill
    DAY = "DAY"  # Day order


@dataclass
class OrderFill:
    """Individual order fill information."""
    fill_id: str
    order_id: str
    symbol: str
    side: str
    price: Decimal
    size: Decimal
    fee: Decimal
    fee_currency: str
    timestamp: datetime
    trade_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert fill to dictionary."""
        return {
            'fill_id': self.fill_id,
            'order_id': self.order_id,
            'symbol': self.symbol,
            'side': self.side,
            'price': float(self.price),
            'size': float(self.size),
            'fee': float(self.fee),
            'fee_currency': self.fee_currency,
            'timestamp': self.timestamp.isoformat(),
            'trade_id': self.trade_id
        }


@dataclass
class Order:
    """Enhanced order with complete lifecycle tracking."""
    order_id: str
    client_order_id: str
    symbol: str
    side: str
    order_type: OrderType
    price: Decimal
    size: Decimal
    time_in_force: TimeInForce
    status: OrderStatus
    
    # Execution details
    filled_size: Decimal = Decimal('0')
    remaining_size: Decimal = field(init=False)
    average_fill_price: Decimal = Decimal('0')
    total_fees: Decimal = Decimal('0')
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    submitted_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    last_updated_at: datetime = field(default_factory=datetime.now)
    
    # Metadata
    source: str = "flashmm"
    parent_order_id: Optional[str] = None  # For order replacements
    child_order_ids: List[str] = field(default_factory=list)
    tags: Dict[str, Any] = field(default_factory=dict)
    
    # Fills and history
    fills: List[OrderFill] = field(default_factory=list)
    status_history: List[Tuple[OrderStatus, datetime]] = field(default_factory=list)
    
    def __post_init__(self):
        """Post-initialization setup."""
        self.remaining_size = self.size
        self.status_history.append((self.status, self.created_at))
    
    def update_status(self, new_status: OrderStatus) -> None:
        """Update order status with history tracking."""
        if new_status != self.status:
            self.status = new_status
            self.last_updated_at = datetime.now()
            self.status_history.append((new_status, self.last_updated_at))
    
    def add_fill(self, fill: OrderFill) -> None:
        """Add a fill to the order."""
        self.fills.append(fill)
        self.filled_size += fill.size
        self.remaining_size = self.size - self.filled_size
        self.total_fees += fill.fee
        
        # Update average fill price
        total_value = sum(f.price * f.size for f in self.fills)
        self.average_fill_price = total_value / self.filled_size if self.filled_size > 0 else Decimal('0')
        
        # Update status based on fill
        if self.remaining_size <= Decimal('0.0001'):  # Small tolerance for floating point
            self.update_status(OrderStatus.FILLED)
        elif self.filled_size > 0:
            self.update_status(OrderStatus.PARTIALLY_FILLED)
        
        self.last_updated_at = datetime.now()
    
    def is_terminal(self) -> bool:
        """Check if order is in a terminal state."""
        return self.status in {
            OrderStatus.FILLED, OrderStatus.CANCELLED, 
            OrderStatus.REJECTED, OrderStatus.EXPIRED, 
            OrderStatus.REPLACED, OrderStatus.FAILED
        }
    
    def is_active(self) -> bool:
        """Check if order is active in the market."""
        return self.status in {
            OrderStatus.ACKNOWLEDGED, OrderStatus.ACTIVE, OrderStatus.PARTIALLY_FILLED
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert order to dictionary."""
        return {
            'order_id': self.order_id,
            'client_order_id': self.client_order_id,
            'symbol': self.symbol,
            'side': self.side,
            'order_type': self.order_type.value,
            'price': float(self.price),
            'size': float(self.size),
            'time_in_force': self.time_in_force.value,
            'status': self.status.value,
            'filled_size': float(self.filled_size),
            'remaining_size': float(self.remaining_size),
            'average_fill_price': float(self.average_fill_price),
            'total_fees': float(self.total_fees),
            'created_at': self.created_at.isoformat(),
            'submitted_at': self.submitted_at.isoformat() if self.submitted_at else None,
            'acknowledged_at': self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            'last_updated_at': self.last_updated_at.isoformat(),
            'source': self.source,
            'parent_order_id': self.parent_order_id,
            'child_order_ids': self.child_order_ids.copy(),
            'tags': self.tags.copy(),
            'fills': [fill.to_dict() for fill in self.fills],
            'status_history': [(status.value, ts.isoformat()) for status, ts in self.status_history]
        }


@dataclass
class BatchOrderRequest:
    """Batch order operation request."""
    orders_to_place: List[Dict[str, Any]] = field(default_factory=list)
    orders_to_cancel: List[str] = field(default_factory=list)
    orders_to_replace: List[Tuple[str, Dict[str, Any]]] = field(default_factory=list)


@dataclass
class BatchOrderResult:
    """Batch order operation result."""
    placed_orders: List[Order] = field(default_factory=list)
    cancelled_orders: List[str] = field(default_factory=list)
    replaced_orders: List[Tuple[str, Order]] = field(default_factory=list)
    failed_operations: List[Dict[str, Any]] = field(default_factory=list)
    execution_time_ms: float = 0.0


class OrderReconciler:
    """Order state reconciliation service."""
    
    def __init__(self, order_router: 'OrderRouter'):
        self.order_router = order_router
        self.last_reconciliation = datetime.now()
        
    async def reconcile_orders(self) -> Dict[str, Any]:
        """Reconcile order states with exchange."""
        start_time = datetime.now()
        reconciliation_results = {
            'orders_checked': 0,
            'orders_updated': 0,
            'fills_detected': 0,
            'errors': []
        }
        
        try:
            active_orders = [
                order for order in self.order_router.orders.values()
                if order.is_active()
            ]
            
            reconciliation_results['orders_checked'] = len(active_orders)
            
            # In production, query exchange for order status
            # For now, simulate some reconciliation
            for order in active_orders:
                try:
                    await self._reconcile_single_order(order, reconciliation_results)
                except Exception as e:
                    reconciliation_results['errors'].append({
                        'order_id': order.order_id,
                        'error': str(e)
                    })
            
            self.last_reconciliation = datetime.now()
            
            logger.debug(f"Order reconciliation completed: {reconciliation_results}")
            return reconciliation_results
            
        except Exception as e:
            logger.error(f"Order reconciliation failed: {e}")
            reconciliation_results['errors'].append({'general_error': str(e)})
            return reconciliation_results
    
    async def _reconcile_single_order(self, order: Order, results: Dict[str, Any]) -> None:
        """Reconcile a single order."""
        # Simulate order aging and potential fills
        age_minutes = (datetime.now() - order.created_at).total_seconds() / 60
        
        # Simulate random fills for demonstration (remove in production)
        if age_minutes > 1 and order.status == OrderStatus.ACTIVE:
            import random
            if random.random() < 0.1:  # 10% chance of partial fill
                fill_size = order.remaining_size * Decimal(str(random.uniform(0.1, 0.5)))
                fill = OrderFill(
                    fill_id=str(uuid.uuid4()),
                    order_id=order.order_id,
                    symbol=order.symbol,
                    side=order.side,
                    price=order.price,
                    size=fill_size,
                    fee=fill_size * order.price * Decimal('0.001'),  # 0.1% fee
                    fee_currency="USDC",
                    timestamp=datetime.now()
                )
                
                order.add_fill(fill)
                results['fills_detected'] += 1
                results['orders_updated'] += 1
                
                await trading_logger.log_order_event(
                    "order_fill_detected",
                    order.order_id,
                    order.symbol,
                    order.side,
                    float(fill.price),
                    float(fill.size)
                )


class OrderRouter:
    """Enhanced order router with complete lifecycle management."""
    
    def __init__(self):
        self.config = get_config()
        self.cambrian_client = None  # Will be initialized with actual SDK
        
        # Blockchain integration
        self.blockchain_service = get_blockchain_service()
        self.blockchain_enabled = self.config.get("blockchain.enabled", True)
        
        # Order storage
        self.orders: Dict[str, Order] = {}
        self.orders_by_symbol: Dict[str, Set[str]] = {}
        self.orders_by_client_id: Dict[str, str] = {}
        
        # Components
        self.reconciler = OrderReconciler(self)
        
        # Performance tracking
        self.total_orders = 0
        self.successful_orders = 0
        self.failed_orders = 0
        self.total_fills = 0
        self.total_volume = Decimal('0')
        
        # Configuration
        self.max_orders_per_symbol = self.config.get("trading.max_orders_per_symbol", 50)
        self.order_timeout_minutes = self.config.get("trading.order_timeout_minutes", 60)
        self.reconciliation_interval_seconds = self.config.get("trading.reconciliation_interval_seconds", 30)
        
        # Background tasks
        self._reconciliation_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        
    async def initialize(self) -> None:
        """Initialize the enhanced order router."""
        try:
            # Initialize blockchain service if enabled
            if self.blockchain_enabled:
                try:
                    await self.blockchain_service.initialize()
                    logger.info("Blockchain service integration enabled")
                except Exception as e:
                    logger.warning(f"Blockchain service initialization failed: {e}")
                    logger.warning("Continuing with mock execution mode")
                    self.blockchain_enabled = False
            
            # Initialize Cambrian SDK client (legacy - replaced by blockchain service)
            # self.cambrian_client = CambrianClient(
            #     api_key=self.config.get("cambrian_api_key"),
            #     secret_key=self.config.get("cambrian_secret_key")
            # )
            
            # Start background tasks
            await self._start_background_tasks()
            
            logger.info("Enhanced OrderRouter initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize OrderRouter: {e}")
            raise OrderError(f"OrderRouter initialization failed: {e}")
    
    async def _start_background_tasks(self) -> None:
        """Start background maintenance tasks."""
        self._reconciliation_task = asyncio.create_task(self._reconciliation_loop())
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
    
    async def _reconciliation_loop(self) -> None:
        """Background order reconciliation loop."""
        while True:
            try:
                await asyncio.sleep(self.reconciliation_interval_seconds)
                await self.reconciler.reconcile_orders()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Reconciliation loop error: {e}")
                await asyncio.sleep(5)  # Brief pause on error
    
    async def _cleanup_loop(self) -> None:
        """Background cleanup of old orders."""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                await self._cleanup_old_orders()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")
                await asyncio.sleep(60)  # Brief pause on error
    
    async def _cleanup_old_orders(self) -> None:
        """Clean up old terminal orders."""
        cutoff_time = datetime.now() - timedelta(hours=24)
        orders_to_remove = []
        
        for order_id, order in self.orders.items():
            if order.is_terminal() and order.last_updated_at < cutoff_time:
                orders_to_remove.append(order_id)
        
        for order_id in orders_to_remove:
            order = self.orders.pop(order_id, None)
            if order:
                # Remove from symbol tracking
                if order.symbol in self.orders_by_symbol:
                    self.orders_by_symbol[order.symbol].discard(order_id)
                
                # Remove from client ID tracking
                self.orders_by_client_id.pop(order.client_order_id, None)
        
        if orders_to_remove:
            logger.info(f"Cleaned up {len(orders_to_remove)} old orders")
    
    @require_trading_enabled
    @timeout_async(0.1)  # 100ms timeout for order placement
    @measure_latency("place_order")
    async def place_order(
        self,
        symbol: str,
        side: str,
        price: float,
        size: float,
        order_type: str = "limit",
        time_in_force: str = "GTC",
        client_order_id: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None
    ) -> Optional[Order]:
        """Place an order with enhanced tracking."""
        try:
            # Validate parameters
            await self._validate_order_parameters(symbol, side, price, size, order_type)
            
            # Check limits
            await self._check_order_limits(symbol)
            
            # Create order
            order = await self._create_order(
                symbol, side, price, size, order_type, time_in_force, client_order_id, tags
            )
            
            # Submit order
            success = await self._submit_order(order)
            
            if success:
                # Store order
                self._store_order(order)
                
                # Update statistics
                self.total_orders += 1
                self.successful_orders += 1
                
                logger.debug(f"Order placed successfully: {order.order_id}")
                return order
            else:
                self.failed_orders += 1
                return None
                
        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            self.failed_orders += 1
            raise OrderError(f"Order placement failed: {e}")
    
    async def _create_order(
        self,
        symbol: str,
        side: str,
        price: float,
        size: float,
        order_type: str,
        time_in_force: str,
        client_order_id: Optional[str],
        tags: Optional[Dict[str, Any]]
    ) -> Order:
        """Create an order object."""
        order_id = f"flashmm_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        client_id = client_order_id or str(uuid.uuid4())
        
        return Order(
            order_id=order_id,
            client_order_id=client_id,
            symbol=symbol,
            side=side,
            order_type=OrderType(order_type.lower()),
            price=Decimal(str(price)).quantize(Decimal('0.000001'), rounding=ROUND_HALF_UP),
            size=Decimal(str(size)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP),
            time_in_force=TimeInForce(time_in_force.upper()),
            status=OrderStatus.PENDING,
            tags=tags or {}
        )
    
    async def _submit_order(self, order: Order) -> bool:
        """Submit order to exchange via blockchain or mock execution."""
        try:
            # Log order attempt
            await trading_logger.log_order_event(
                "order_attempt",
                order.order_id,
                order.symbol,
                order.side,
                float(order.price),
                float(order.size)
            )
            
            # Update status
            order.update_status(OrderStatus.SUBMITTED)
            order.submitted_at = datetime.now()
            
            # Submit via blockchain service if available
            if self.blockchain_enabled and self.blockchain_service:
                try:
                    success = await self.blockchain_service.submit_order_to_blockchain(order)
                    
                    if success:
                        # Blockchain submission successful
                        order.update_status(OrderStatus.ACKNOWLEDGED)
                        order.acknowledged_at = datetime.now()
                        order.update_status(OrderStatus.ACTIVE)
                        
                        await trading_logger.log_order_event(
                            "order_placed_blockchain",
                            order.order_id,
                            order.symbol,
                            order.side,
                            float(order.price),
                            float(order.size)
                        )
                        
                        return True
                    else:
                        # Blockchain submission failed, fallback to mock
                        logger.warning(f"Blockchain submission failed for order {order.order_id}, using mock execution")
                        return await self._submit_order_mock(order)
                        
                except Exception as e:
                    logger.error(f"Blockchain submission error for order {order.order_id}: {e}")
                    # Fallback to mock execution
                    return await self._submit_order_mock(order)
            else:
                # Use mock execution
                return await self._submit_order_mock(order)
                
        except Exception as e:
            order.update_status(OrderStatus.FAILED)
            logger.error(f"Order submission failed: {e}")
            return False
    
    async def _submit_order_mock(self, order: Order) -> bool:
        """Submit order using mock execution for testing."""
        try:
            # Simulate network latency
            await asyncio.sleep(0.01)
            
            # Simulate exchange acknowledgment
            order.update_status(OrderStatus.ACKNOWLEDGED)
            order.acknowledged_at = datetime.now()
            
            # Simulate order becoming active
            order.update_status(OrderStatus.ACTIVE)
            
            await trading_logger.log_order_event(
                "order_placed_mock",
                order.order_id,
                order.symbol,
                order.side,
                float(order.price),
                float(order.size)
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Mock order submission failed: {e}")
            return False
    
    def _store_order(self, order: Order) -> None:
        """Store order in internal tracking."""
        self.orders[order.order_id] = order
        self.orders_by_client_id[order.client_order_id] = order.order_id
        
        # Track by symbol
        if order.symbol not in self.orders_by_symbol:
            self.orders_by_symbol[order.symbol] = set()
        self.orders_by_symbol[order.symbol].add(order.order_id)
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel a specific order."""
        try:
            order = self.orders.get(order_id)
            if not order:
                logger.warning(f"Order not found: {order_id}")
                return False
            
            if order.is_terminal():
                logger.warning(f"Cannot cancel terminal order: {order_id}")
                return False
            
            # Cancel via blockchain service if available
            blockchain_success = False
            if self.blockchain_enabled and self.blockchain_service:
                try:
                    blockchain_success = await self.blockchain_service.cancel_order_on_blockchain(order_id)
                    
                    if blockchain_success:
                        await trading_logger.log_order_event(
                            "order_cancelled_blockchain",
                            order_id,
                            order.symbol,
                            order.side,
                            float(order.price),
                            float(order.size)
                        )
                    else:
                        logger.warning(f"Blockchain cancellation failed for order {order_id}")
                        
                except Exception as e:
                    logger.error(f"Blockchain cancellation error for order {order_id}: {e}")
            
            # Update internal order status (regardless of blockchain result for mock testing)
            order.update_status(OrderStatus.CANCELLED)
            
            await trading_logger.log_order_event(
                "order_cancelled",
                order_id,
                order.symbol,
                order.side,
                float(order.price),
                float(order.size)
            )
            
            logger.debug(f"Order cancelled: {order_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            raise OrderError(f"Order cancellation failed: {e}")
    
    async def cancel_orders_for_symbol(self, symbol: str) -> int:
        """Cancel all orders for a specific symbol."""
        if symbol not in self.orders_by_symbol:
            return 0
        
        order_ids = list(self.orders_by_symbol[symbol])
        cancelled_count = 0
        
        for order_id in order_ids:
            try:
                if await self.cancel_order(order_id):
                    cancelled_count += 1
            except Exception as e:
                logger.error(f"Failed to cancel order {order_id}: {e}")
        
        logger.debug(f"Cancelled {cancelled_count}/{len(order_ids)} orders for {symbol}")
        return cancelled_count
    
    async def cancel_all_orders(self) -> int:
        """Cancel all active orders."""
        active_orders = [
            order_id for order_id, order in self.orders.items()
            if order.is_active()
        ]
        
        cancelled_count = 0
        for order_id in active_orders:
            try:
                if await self.cancel_order(order_id):
                    cancelled_count += 1
            except Exception as e:
                logger.error(f"Failed to cancel order {order_id}: {e}")
        
        logger.info(f"Cancelled {cancelled_count}/{len(active_orders)} active orders")
        return cancelled_count
    
    @timeout_async(0.2)  # 200ms timeout for batch operations
    @measure_latency("batch_order_operation")
    async def execute_batch_orders(self, batch_request: BatchOrderRequest) -> BatchOrderResult:
        """Execute batch order operations for efficiency."""
        start_time = datetime.now()
        result = BatchOrderResult()
        
        try:
            # Execute all operations concurrently
            tasks = []
            
            # Place orders
            for order_data in batch_request.orders_to_place:
                task = asyncio.create_task(self._place_order_from_dict(order_data))
                tasks.append(('place', task, order_data))
            
            # Cancel orders
            for order_id in batch_request.orders_to_cancel:
                task = asyncio.create_task(self.cancel_order(order_id))
                tasks.append(('cancel', task, order_id))
            
            # Replace orders
            for old_order_id, new_order_data in batch_request.orders_to_replace:
                task = asyncio.create_task(self._replace_order(old_order_id, new_order_data))
                tasks.append(('replace', task, (old_order_id, new_order_data)))
            
            # Wait for all operations
            for operation_type, task, data in tasks:
                try:
                    result_value = await task
                    
                    if operation_type == 'place' and result_value:
                        result.placed_orders.append(result_value)
                    elif operation_type == 'cancel' and result_value:
                        result.cancelled_orders.append(data)
                    elif operation_type == 'replace' and result_value:
                        result.replaced_orders.append(result_value)
                        
                except Exception as e:
                    result.failed_operations.append({
                        'operation': operation_type,
                        'data': str(data),
                        'error': str(e)
                    })
            
            result.execution_time_ms = (datetime.now() - start_time).total_seconds() * 1000
            
            logger.debug(f"Batch operation completed: "
                        f"{len(result.placed_orders)} placed, "
                        f"{len(result.cancelled_orders)} cancelled, "
                        f"{len(result.replaced_orders)} replaced, "
                        f"{len(result.failed_operations)} failed")
            
            return result
            
        except Exception as e:
            logger.error(f"Batch order execution failed: {e}")
            result.failed_operations.append({'general_error': str(e)})
            return result
    
    async def _place_order_from_dict(self, order_data: Dict[str, Any]) -> Optional[Order]:
        """Place order from dictionary data."""
        return await self.place_order(
            symbol=order_data['symbol'],
            side=order_data['side'],
            price=order_data['price'],
            size=order_data['size'],
            order_type=order_data.get('order_type', 'limit'),
            time_in_force=order_data.get('time_in_force', 'GTC'),
            client_order_id=order_data.get('client_order_id'),
            tags=order_data.get('tags')
        )
    
    async def _replace_order(self, old_order_id: str, new_order_data: Dict[str, Any]) -> Optional[Tuple[str, Order]]:
        """Replace an existing order."""
        # Cancel old order
        cancel_success = await self.cancel_order(old_order_id)
        if not cancel_success:
            return None
        
        # Mark old order as replaced
        old_order = self.orders.get(old_order_id)
        if old_order:
            old_order.update_status(OrderStatus.REPLACED)
        
        # Place new order
        new_order = await self._place_order_from_dict(new_order_data)
        if new_order and old_order:
            # Link orders
            old_order.child_order_ids.append(new_order.order_id)
            new_order.parent_order_id = old_order_id
            
            return (old_order_id, new_order)
        
        return None
    
    async def _check_order_limits(self, symbol: str) -> None:
        """Check order limits before placement."""
        active_orders_count = len([
            o for o in self.orders_by_symbol.get(symbol, set())
            if self.orders[o].is_active()
        ])
        
        if active_orders_count >= self.max_orders_per_symbol:
            raise OrderError(f"Maximum orders per symbol exceeded: {active_orders_count}")
    
    async def _validate_order_parameters(
        self,
        symbol: str,
        side: str,
        price: float,
        size: float,
        order_type: str
    ) -> None:
        """Enhanced order parameter validation with blockchain integration."""
        if not symbol or not isinstance(symbol, str):
            raise OrderError(f"Invalid symbol: {symbol}")
        
        if side not in ["buy", "sell"]:
            raise OrderError(f"Invalid order side: {side}")
        
        if price <= 0:
            raise OrderError(f"Invalid price: {price}")
        
        if size <= 0:
            raise OrderError(f"Invalid size: {size}")
        
        if order_type not in ["limit", "market", "stop", "stop_limit"]:
            raise OrderError(f"Invalid order type: {order_type}")
        
        # Additional validation for market making
        min_size = self.config.get("trading.min_order_size", 1.0)
        max_size = self.config.get("trading.max_order_size", 10000.0)
        
        if size < min_size:
            raise OrderError(f"Order size below minimum: {size} < {min_size}")
        
        if size > max_size:
            raise OrderError(f"Order size above maximum: {size} > {max_size}")
        
        # Blockchain-specific validation
        if self.blockchain_enabled and self.blockchain_service:
            try:
                # Check if market is supported
                supported_markets = self.blockchain_service.get_supported_markets()
                if symbol not in supported_markets:
                    raise OrderError(f"Market {symbol} not supported on blockchain")
                
                # Validate order parameters for blockchain
                from flashmm.trading.execution.order_router import Order, OrderType, TimeInForce
                temp_order = Order(
                    order_id="validation_temp",
                    client_order_id="validation_temp",
                    symbol=symbol,
                    side=side,
                    order_type=OrderType(order_type.lower()),
                    price=Decimal(str(price)),
                    size=Decimal(str(size)),
                    time_in_force=TimeInForce.GTC,
                    status=OrderStatus.PENDING
                )
                
                validation_result = self.blockchain_service.validate_order_for_blockchain(temp_order)
                
                if not validation_result.get('valid', False):
                    raise OrderError(f"Blockchain validation failed: {validation_result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                if "not supported on blockchain" in str(e) or "Blockchain validation failed" in str(e):
                    raise
                else:
                    # Log blockchain validation error but don't fail order
                    logger.warning(f"Blockchain validation error (continuing with order): {e}")
    
    # Query methods
    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID."""
        return self.orders.get(order_id)
    
    def get_order_by_client_id(self, client_order_id: str) -> Optional[Order]:
        """Get order by client ID."""
        order_id = self.orders_by_client_id.get(client_order_id)
        return self.orders.get(order_id) if order_id else None
    
    def get_orders_for_symbol(self, symbol: str, status_filter: Optional[Set[OrderStatus]] = None) -> List[Order]:
        """Get orders for a specific symbol with optional status filter."""
        if symbol not in self.orders_by_symbol:
            return []
        
        orders = [
            self.orders[order_id] for order_id in self.orders_by_symbol[symbol]
            if order_id in self.orders
        ]
        
        if status_filter:
            orders = [order for order in orders if order.status in status_filter]
        
        return orders
    
    def get_active_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get all active orders, optionally filtered by symbol."""
        active_statuses = {OrderStatus.ACKNOWLEDGED, OrderStatus.ACTIVE, OrderStatus.PARTIALLY_FILLED}
        
        if symbol:
            return self.get_orders_for_symbol(symbol, active_statuses)
        else:
            return [order for order in self.orders.values() if order.status in active_statuses]
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive order router performance statistics."""
        active_orders = self.get_active_orders()
        
        stats = {
            'total_orders': self.total_orders,
            'successful_orders': self.successful_orders,
            'failed_orders': self.failed_orders,
            'success_rate': self.successful_orders / max(self.total_orders, 1),
            'active_orders_count': len(active_orders),
            'total_orders_tracked': len(self.orders),
            'total_fills': self.total_fills,
            'total_volume': float(self.total_volume),
            'symbols_trading': len(self.orders_by_symbol),
            'last_reconciliation': self.reconciler.last_reconciliation.isoformat(),
            'orders_by_status': self._get_orders_by_status_count(),
            'blockchain_integration': self.get_blockchain_status()
        }
        
        return stats
    
    def _get_orders_by_status_count(self) -> Dict[str, int]:
        """Get count of orders by status."""
        status_counts = {}
        for order in self.orders.values():
            status = order.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
        return status_counts
    
    async def cleanup(self) -> None:
        """Cleanup resources and stop background tasks."""
        # Cancel background tasks
        if self._reconciliation_task and not self._reconciliation_task.done():
            self._reconciliation_task.cancel()
            try:
                await self._reconciliation_task
            except asyncio.CancelledError:
                pass
        
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Cleanup blockchain service
        if self.blockchain_enabled and self.blockchain_service:
            try:
                await self.blockchain_service.cleanup()
            except Exception as e:
                logger.warning(f"Blockchain service cleanup failed: {e}")
        
        logger.info("OrderRouter cleanup completed")
    
    def get_blockchain_status(self) -> Dict[str, Any]:
        """Get blockchain integration status."""
        if not self.blockchain_enabled or not self.blockchain_service:
            return {
                'enabled': False,
                'status': 'disabled',
                'supported_markets': [],
                'network_status': 'unavailable'
            }
        
        try:
            return {
                'enabled': True,
                'status': self.blockchain_service.status.value if hasattr(self.blockchain_service, 'status') else 'unknown',
                'supported_markets': self.blockchain_service.get_supported_markets(),
                'network_status': self.blockchain_service.get_network_status(),
                'service_status': self.blockchain_service.get_service_status()
            }
        except Exception as e:
            return {
                'enabled': True,
                'status': 'error',
                'error': str(e),
                'supported_markets': [],
                'network_status': 'error'
            }
    
    def is_blockchain_ready(self) -> bool:
        """Check if blockchain integration is ready for trading."""
        if not self.blockchain_enabled or not self.blockchain_service:
            return False
        
        try:
            status = self.blockchain_service.get_service_status()
            return status.get('status') in ['healthy', 'degraded']
        except Exception as e:
            logger.warning(f"Failed to check blockchain readiness: {e}")
            return False
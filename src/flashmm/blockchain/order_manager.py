"""
Blockchain Order Manager

Integrates Cambrian SDK with existing order management system,
handles order validation, blockchain submission, and state synchronization.
"""

import asyncio
from typing import Dict, Any, Optional, List, Tuple, Set
from decimal import Decimal
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import uuid

try:
    from cambrian_sdk import CambrianClient
    from cambrian_sdk.types import Order as CambrianOrder, OrderSide, OrderType as CambrianOrderType
    from cambrian_sdk.exceptions import CambrianAPIError
    CAMBRIAN_AVAILABLE = True
except ImportError:
    # Mock classes for development when SDK is not available
    class CambrianClient:
        pass
    class CambrianOrder:
        pass
    class OrderSide:
        BUY = "buy"
        SELL = "sell"
    class CambrianOrderType:
        LIMIT = "limit"
        MARKET = "market"
    class CambrianAPIError(Exception):
        pass
    CAMBRIAN_AVAILABLE = False

from flashmm.config.settings import get_config
from flashmm.utils.logging import get_logger, TradingEventLogger
from flashmm.utils.exceptions import OrderError, ValidationError, BlockchainError
from flashmm.utils.decorators import timeout_async, measure_latency, retry_async
from flashmm.trading.execution.order_router import Order, OrderStatus, OrderType, TimeInForce, OrderFill
from flashmm.blockchain.sei_client import SeiClient
from flashmm.blockchain.market_config import MarketConfigManager, MarketConfig

logger = get_logger(__name__)
trading_logger = TradingEventLogger()


@dataclass
class BlockchainOrderMapping:
    """Mapping between internal orders and blockchain orders."""
    internal_order_id: str
    blockchain_order_id: Optional[str]
    client_order_id: str
    submission_timestamp: datetime
    confirmation_timestamp: Optional[datetime]
    blockchain_status: str
    last_sync_timestamp: datetime


class OrderSyncStatus(Enum):
    """Order synchronization status."""
    PENDING_SUBMISSION = "pending_submission"
    SUBMITTED = "submitted"
    CONFIRMED = "confirmed"
    SYNCED = "synced"
    FAILED = "failed"
    CONFLICT = "conflict"


class BlockchainOrderManager:
    """Manager for blockchain order operations and synchronization."""
    
    def __init__(self, sei_client: SeiClient, market_config_manager: MarketConfigManager):
        self.sei_client = sei_client
        self.market_config_manager = market_config_manager
        self.config = get_config()
        
        # Cambrian SDK client
        self.cambrian_client: Optional[CambrianClient] = None
        
        # Order mappings and tracking
        self.order_mappings: Dict[str, BlockchainOrderMapping] = {}
        self.blockchain_to_internal: Dict[str, str] = {}
        
        # Configuration
        self.max_retries = self.config.get("blockchain.max_retries", 3)
        self.retry_delay_seconds = self.config.get("blockchain.retry_delay_seconds", 1.0)
        self.sync_interval_seconds = self.config.get("blockchain.sync_interval_seconds", 10)
        self.order_timeout_seconds = self.config.get("blockchain.order_timeout_seconds", 300)
        
        # Performance tracking
        self.total_submissions = 0
        self.successful_submissions = 0
        self.failed_submissions = 0
        self.sync_operations = 0
        
        # Background tasks
        self._sync_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        
    async def initialize(self) -> None:
        """Initialize the blockchain order manager."""
        try:
            logger.info("Initializing blockchain order manager")
            
            # Initialize Cambrian SDK client
            await self._initialize_cambrian_client()
            
            # Start background tasks
            await self._start_background_tasks()
            
            logger.info("Blockchain order manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize blockchain order manager: {e}")
            raise BlockchainError(f"Order manager initialization failed: {e}")
    
    async def _initialize_cambrian_client(self) -> None:
        """Initialize Cambrian SDK client."""
        if not CAMBRIAN_AVAILABLE:
            logger.warning("Cambrian SDK not available - using mock implementation")
            return
        
        try:
            api_key = self.config.get("cambrian_api_key")
            secret_key = self.config.get("cambrian_secret_key")
            
            if not api_key or not secret_key:
                logger.warning("Cambrian API credentials not configured")
                return
            
            # Initialize Cambrian client
            self.cambrian_client = CambrianClient(
                api_key=api_key,
                secret_key=secret_key,
                testnet=True,  # Use testnet for Sei V2
                timeout=10.0
            )
            
            # Test connection
            await self.cambrian_client.get_account_info()
            
            logger.info("Cambrian SDK client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Cambrian client: {e}")
            # Continue with mock implementation for development
            self.cambrian_client = None
    
    async def _start_background_tasks(self) -> None:
        """Start background maintenance tasks."""
        self._sync_task = asyncio.create_task(self._sync_loop())
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
    
    async def _sync_loop(self) -> None:
        """Background order synchronization loop."""
        while True:
            try:
                await asyncio.sleep(self.sync_interval_seconds)
                await self._sync_orders()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Order sync loop error: {e}")
                await asyncio.sleep(30)  # Brief pause on error
    
    async def _cleanup_loop(self) -> None:
        """Background cleanup of old order mappings."""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                await self._cleanup_old_mappings()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")
                await asyncio.sleep(300)  # Brief pause on error
    
    @timeout_async(10.0)
    @measure_latency("blockchain_order_submission")
    @retry_async(max_retries=3, delay=1.0)
    async def submit_order_to_blockchain(self, order: Order) -> bool:
        """Submit order to blockchain via Cambrian SDK."""
        try:
            # Validate order before submission
            validation_result = await self._validate_order_for_blockchain(order)
            if not validation_result['valid']:
                logger.error(f"Order validation failed: {validation_result['error']}")
                return False
            
            # Convert to blockchain order format
            blockchain_order = await self._convert_to_blockchain_order(order)
            
            # Submit via Cambrian SDK
            blockchain_order_id = await self._submit_via_cambrian(blockchain_order)
            
            if blockchain_order_id:
                # Create order mapping
                mapping = BlockchainOrderMapping(
                    internal_order_id=order.order_id,
                    blockchain_order_id=blockchain_order_id,
                    client_order_id=order.client_order_id,
                    submission_timestamp=datetime.now(),
                    confirmation_timestamp=None,
                    blockchain_status="submitted",
                    last_sync_timestamp=datetime.now()
                )
                
                # Store mapping
                self.order_mappings[order.order_id] = mapping
                self.blockchain_to_internal[blockchain_order_id] = order.order_id
                
                # Update statistics
                self.total_submissions += 1
                self.successful_submissions += 1
                
                await trading_logger.log_order_event(
                    "order_submitted_blockchain",
                    order.order_id,
                    order.symbol,
                    order.side,
                    float(order.price),
                    float(order.size),
                    {"blockchain_order_id": blockchain_order_id}
                )
                
                logger.debug(f"Order submitted to blockchain: {order.order_id} -> {blockchain_order_id}")
                return True
            else:
                self.failed_submissions += 1
                return False
                
        except Exception as e:
            logger.error(f"Failed to submit order to blockchain: {e}")
            self.failed_submissions += 1
            raise OrderError(f"Blockchain order submission failed: {e}")
    
    async def _validate_order_for_blockchain(self, order: Order) -> Dict[str, Any]:
        """Validate order against blockchain and market requirements."""
        try:
            # Get market configuration
            market = self.market_config_manager.get_market(order.symbol)
            if not market:
                return {'valid': False, 'error': f'Unsupported market: {order.symbol}'}
            
            # Validate market order parameters
            market_validation = self.market_config_manager.validate_market_order(
                order.symbol, order.price, order.size
            )
            
            if not market_validation['valid']:
                return market_validation
            
            # Check account balances (if possible)
            balance_check = await self._check_account_balance(order, market)
            if not balance_check['valid']:
                return balance_check
            
            return {'valid': True, 'market_validation': market_validation}
            
        except Exception as e:
            logger.error(f"Order validation failed: {e}")
            return {'valid': False, 'error': f'Validation error: {e}'}
    
    async def _check_account_balance(self, order: Order, market: MarketConfig) -> Dict[str, Any]:
        """Check if account has sufficient balance for order."""
        try:
            if not self.sei_client.wallet:
                # Skip balance check if no wallet configured
                return {'valid': True}
            
            # Get account info
            account_info = await self.sei_client.get_account_info()
            balances = account_info.balances
            
            # Calculate required balance
            if order.side.lower() == "buy":
                # Need quote currency for buy orders
                required_currency = market.quote_asset.lower()
                required_amount = order.price * order.size
            else:
                # Need base currency for sell orders
                required_currency = market.base_asset.lower()
                required_amount = order.size
            
            # Check balance
            available_balance = balances.get(required_currency, Decimal('0'))
            if available_balance < required_amount:
                return {
                    'valid': False,
                    'error': f'Insufficient {required_currency.upper()} balance: {available_balance} < {required_amount}'
                }
            
            return {'valid': True}
            
        except Exception as e:
            logger.warning(f"Balance check failed: {e}")
            # Continue without balance check on error
            return {'valid': True}
    
    async def _convert_to_blockchain_order(self, order: Order) -> Dict[str, Any]:
        """Convert internal order to blockchain order format."""
        # Get market configuration for proper formatting
        market = self.market_config_manager.get_market(order.symbol)
        if not market:
            raise ValidationError(f"Market configuration not found: {order.symbol}")
        
        # Round price and size to market requirements
        rounded_price = market.round_price(order.price)
        rounded_size = market.round_size(order.size)
        
        # Convert order type
        blockchain_order_type = self._convert_order_type(order.order_type)
        
        # Convert side
        blockchain_side = OrderSide.BUY if order.side.lower() == "buy" else OrderSide.SELL
        
        return {
            'client_order_id': order.client_order_id,
            'market_id': market.market_id,
            'side': blockchain_side,
            'type': blockchain_order_type,
            'price': str(rounded_price),
            'size': str(rounded_size),
            'time_in_force': order.time_in_force.value,
            'metadata': {
                'internal_order_id': order.order_id,
                'source': 'flashmm',
                'timestamp': order.created_at.isoformat()
            }
        }
    
    def _convert_order_type(self, order_type: OrderType) -> str:
        """Convert internal order type to blockchain order type."""
        type_mapping = {
            OrderType.LIMIT: CambrianOrderType.LIMIT,
            OrderType.MARKET: CambrianOrderType.MARKET,
            # Add more mappings as needed
        }
        return type_mapping.get(order_type, CambrianOrderType.LIMIT)
    
    async def _submit_via_cambrian(self, blockchain_order: Dict[str, Any]) -> Optional[str]:
        """Submit order via Cambrian SDK."""
        if not self.cambrian_client:
            # Mock implementation for development
            mock_order_id = f"blockchain_{uuid.uuid4().hex[:8]}"
            await asyncio.sleep(0.1)  # Simulate network latency
            return mock_order_id
        
        try:
            # Submit order via Cambrian SDK
            result = await self.cambrian_client.place_order(**blockchain_order)
            return result.get('order_id')
            
        except CambrianAPIError as e:
            logger.error(f"Cambrian API error: {e}")
            raise OrderError(f"Cambrian order submission failed: {e}")
    
    @timeout_async(5.0)
    @measure_latency("blockchain_order_cancellation")
    async def cancel_order_on_blockchain(self, order_id: str) -> bool:
        """Cancel order on blockchain."""
        try:
            mapping = self.order_mappings.get(order_id)
            if not mapping or not mapping.blockchain_order_id:
                logger.warning(f"No blockchain mapping found for order: {order_id}")
                return False
            
            # Cancel via Cambrian SDK
            success = await self._cancel_via_cambrian(mapping.blockchain_order_id)
            
            if success:
                await trading_logger.log_order_event(
                    "order_cancelled_blockchain",
                    order_id,
                    "",  # Symbol not available in mapping
                    "",  # Side not available in mapping
                    0.0,
                    0.0,
                    {"blockchain_order_id": mapping.blockchain_order_id}
                )
                
                logger.debug(f"Order cancelled on blockchain: {order_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to cancel order on blockchain: {e}")
            return False
    
    async def _cancel_via_cambrian(self, blockchain_order_id: str) -> bool:
        """Cancel order via Cambrian SDK."""
        if not self.cambrian_client:
            # Mock implementation for development
            await asyncio.sleep(0.05)  # Simulate network latency
            return True
        
        try:
            result = await self.cambrian_client.cancel_order(blockchain_order_id)
            return result.get('success', False)
            
        except CambrianAPIError as e:
            logger.error(f"Cambrian cancellation error: {e}")
            return False
    
    async def _sync_orders(self) -> None:
        """Synchronize order states with blockchain."""
        try:
            if not self.order_mappings:
                return
            
            synced_count = 0
            error_count = 0
            
            # Sync active orders
            active_mappings = [
                mapping for mapping in self.order_mappings.values()
                if mapping.blockchain_order_id and mapping.blockchain_status not in ['filled', 'cancelled', 'rejected']
            ]
            
            for mapping in active_mappings:
                try:
                    await self._sync_single_order(mapping)
                    synced_count += 1
                except Exception as e:
                    logger.warning(f"Failed to sync order {mapping.internal_order_id}: {e}")
                    error_count += 1
            
            self.sync_operations += 1
            
            if synced_count > 0 or error_count > 0:
                logger.debug(f"Order sync completed: {synced_count} synced, {error_count} errors")
                
        except Exception as e:
            logger.error(f"Order synchronization failed: {e}")
    
    async def _sync_single_order(self, mapping: BlockchainOrderMapping) -> None:
        """Synchronize single order with blockchain state."""
        if not mapping.blockchain_order_id:
            return
        
        try:
            # Query order status from blockchain
            blockchain_status = await self._query_blockchain_order_status(mapping.blockchain_order_id)
            
            if blockchain_status:
                # Update mapping
                old_status = mapping.blockchain_status
                mapping.blockchain_status = blockchain_status.get('status', old_status)
                mapping.last_sync_timestamp = datetime.now()
                
                # Check for fills
                fills = blockchain_status.get('fills', [])
                if fills:
                    await self._process_order_fills(mapping, fills)
                
                # Update confirmation timestamp if needed
                if not mapping.confirmation_timestamp and mapping.blockchain_status in ['active', 'partially_filled']:
                    mapping.confirmation_timestamp = datetime.now()
                
        except Exception as e:
            logger.warning(f"Failed to sync order {mapping.internal_order_id}: {e}")
    
    async def _query_blockchain_order_status(self, blockchain_order_id: str) -> Optional[Dict[str, Any]]:
        """Query order status from blockchain."""
        if not self.cambrian_client:
            # Mock implementation for development
            return {
                'status': 'active',
                'filled_size': '0',
                'remaining_size': '100',
                'fills': []
            }
        
        try:
            return await self.cambrian_client.get_order_status(blockchain_order_id)
            
        except CambrianAPIError as e:
            logger.warning(f"Failed to query order status: {e}")
            return None
    
    async def _process_order_fills(self, mapping: BlockchainOrderMapping, fills: List[Dict[str, Any]]) -> None:
        """Process fills for an order."""
        try:
            for fill_data in fills:
                # Create OrderFill object
                fill = OrderFill(
                    fill_id=fill_data.get('fill_id', str(uuid.uuid4())),
                    order_id=mapping.internal_order_id,
                    symbol=fill_data.get('symbol', ''),
                    side=fill_data.get('side', ''),
                    price=Decimal(str(fill_data.get('price', '0'))),
                    size=Decimal(str(fill_data.get('size', '0'))),
                    fee=Decimal(str(fill_data.get('fee', '0'))),
                    fee_currency=fill_data.get('fee_currency', 'USDC'),
                    timestamp=datetime.now(),
                    trade_id=fill_data.get('trade_id')
                )
                
                # Log fill detection
                await trading_logger.log_order_event(
                    "order_fill_blockchain",
                    mapping.internal_order_id,
                    fill.symbol,
                    fill.side,
                    float(fill.price),
                    float(fill.size),
                    {
                        'fill_id': fill.fill_id,
                        'trade_id': fill.trade_id,
                        'fee': float(fill.fee)
                    }
                )
                
        except Exception as e:
            logger.error(f"Failed to process fills for order {mapping.internal_order_id}: {e}")
    
    async def _cleanup_old_mappings(self) -> None:
        """Clean up old order mappings."""
        cutoff_time = datetime.now() - timedelta(hours=24)
        mappings_to_remove = []
        
        for order_id, mapping in self.order_mappings.items():
            # Remove old terminal orders
            if (mapping.blockchain_status in ['filled', 'cancelled', 'rejected'] and 
                mapping.last_sync_timestamp < cutoff_time):
                mappings_to_remove.append(order_id)
        
        for order_id in mappings_to_remove:
            mapping = self.order_mappings.pop(order_id, None)
            if mapping and mapping.blockchain_order_id:
                self.blockchain_to_internal.pop(mapping.blockchain_order_id, None)
        
        if mappings_to_remove:
            logger.info(f"Cleaned up {len(mappings_to_remove)} old order mappings")
    
    def get_order_mapping(self, order_id: str) -> Optional[BlockchainOrderMapping]:
        """Get blockchain mapping for internal order ID."""
        return self.order_mappings.get(order_id)
    
    def get_internal_order_id(self, blockchain_order_id: str) -> Optional[str]:
        """Get internal order ID from blockchain order ID."""
        return self.blockchain_to_internal.get(blockchain_order_id)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get order manager performance statistics."""
        active_mappings = len([
            m for m in self.order_mappings.values()
            if m.blockchain_status not in ['filled', 'cancelled', 'rejected']
        ])
        
        return {
            'total_submissions': self.total_submissions,
            'successful_submissions': self.successful_submissions,
            'failed_submissions': self.failed_submissions,
            'success_rate': self.successful_submissions / max(self.total_submissions, 1),
            'sync_operations': self.sync_operations,
            'active_mappings': active_mappings,
            'total_mappings': len(self.order_mappings),
            'cambrian_available': CAMBRIAN_AVAILABLE and self.cambrian_client is not None
        }
    
    async def cleanup(self) -> None:
        """Cleanup resources and stop background tasks."""
        # Cancel background tasks
        if self._sync_task and not self._sync_task.done():
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass
        
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Close Cambrian client if available
        if self.cambrian_client and hasattr(self.cambrian_client, 'close'):
            await self.cambrian_client.close()
        
        logger.info("Blockchain order manager cleanup completed")
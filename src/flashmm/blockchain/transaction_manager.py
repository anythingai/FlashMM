"""
Transaction Manager

Handles transaction batching, optimization, priority management,
and failure recovery for Sei blockchain operations.
"""

import asyncio
import json
import uuid
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any

from flashmm.blockchain.sei_client import SeiClient, TransactionResult, TransactionStatus
from flashmm.config.settings import get_config
from flashmm.utils.decorators import measure_latency, timeout_async
from flashmm.utils.exceptions import BlockchainError
from flashmm.utils.logging import get_logger

logger = get_logger(__name__)


class TransactionPriority(Enum):
    """Transaction priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class BatchTransactionType(Enum):
    """Types of batch transactions."""
    ORDER_PLACEMENT = "order_placement"
    ORDER_CANCELLATION = "order_cancellation"
    ORDER_MODIFICATION = "order_modification"
    POSITION_UPDATE = "position_update"
    BALANCE_ADJUSTMENT = "balance_adjustment"


@dataclass
class TransactionRequest:
    """Individual transaction request."""
    tx_id: str
    tx_type: BatchTransactionType
    priority: TransactionPriority
    payload: dict[str, Any]
    created_at: datetime = field(default_factory=datetime.now)
    max_gas: int | None = None
    gas_price: Decimal | None = None
    timeout_seconds: int = 30
    retry_count: int = 0
    max_retries: int = 3
    callback: Callable[[TransactionResult], None] | None = None


@dataclass
class BatchTransaction:
    """Batch of related transactions."""
    batch_id: str
    batch_type: BatchTransactionType
    priority: TransactionPriority
    transactions: list[TransactionRequest]
    created_at: datetime = field(default_factory=datetime.now)
    estimated_gas: int | None = None
    estimated_fee: Decimal | None = None
    deadline: datetime | None = None


@dataclass
class TransactionStats:
    """Transaction execution statistics."""
    total_transactions: int = 0
    successful_transactions: int = 0
    failed_transactions: int = 0
    total_gas_used: int = 0
    total_fees_paid: Decimal = field(default_factory=lambda: Decimal('0'))
    avg_confirmation_time: float = 0.0
    batch_efficiency: float = 0.0


class TransactionManager:
    """Advanced transaction manager with batching and optimization."""

    def __init__(self, sei_client: SeiClient):
        self.sei_client = sei_client
        self.config = get_config()

        # Transaction queues by priority
        self.transaction_queues: dict[TransactionPriority, deque] = {
            priority: deque() for priority in TransactionPriority
        }

        # Batch management
        self.pending_batches: dict[str, BatchTransaction] = {}
        self.active_transactions: dict[str, TransactionRequest] = {}
        self.completed_transactions: dict[str, TransactionResult] = {}

        # Configuration
        self.max_batch_size = self.config.get("blockchain.max_batch_size", 10)
        self.batch_timeout_seconds = self.config.get("blockchain.batch_timeout_seconds", 5)
        self.max_concurrent_batches = self.config.get("blockchain.max_concurrent_batches", 3)
        self.gas_optimization_enabled = self.config.get("blockchain.gas_optimization", True)
        self.mev_protection_enabled = self.config.get("blockchain.mev_protection", True)

        # Gas management
        self.base_gas_price = Decimal(self.config.get("blockchain.base_gas_price", "0.025"))
        self.gas_price_multipliers = {
            TransactionPriority.LOW: Decimal("0.8"),
            TransactionPriority.NORMAL: Decimal("1.0"),
            TransactionPriority.HIGH: Decimal("1.5"),
            TransactionPriority.CRITICAL: Decimal("2.0")
        }

        # Statistics
        self.stats = TransactionStats()

        # Background tasks
        self._processor_task: asyncio.Task | None = None
        self._gas_monitor_task: asyncio.Task | None = None
        self._cleanup_task: asyncio.Task | None = None

        # MEV protection
        self._recent_transactions: deque = deque(maxlen=100)
        self._nonce_manager: dict[str, int] = {}

    async def initialize(self) -> None:
        """Initialize the transaction manager."""
        try:
            logger.info("Initializing transaction manager")

            # Start background tasks
            await self._start_background_tasks()

            # Initialize gas price monitoring
            await self._update_gas_prices()

            logger.info("Transaction manager initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize transaction manager: {e}")
            raise BlockchainError(f"Transaction manager initialization failed: {e}") from e

    async def _start_background_tasks(self) -> None:
        """Start background processing tasks."""
        self._processor_task = asyncio.create_task(self._transaction_processor())
        self._gas_monitor_task = asyncio.create_task(self._gas_price_monitor())
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def _transaction_processor(self) -> None:
        """Main transaction processing loop."""
        while True:
            try:
                # Process transactions by priority
                await self._process_transaction_queues()

                # Process pending batches
                await self._process_pending_batches()

                await asyncio.sleep(0.1)  # Small delay to prevent busy waiting

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Transaction processor error: {e}")
                await asyncio.sleep(1)  # Brief pause on error

    async def _gas_price_monitor(self) -> None:
        """Monitor and update gas prices."""
        while True:
            try:
                await asyncio.sleep(30)  # Update every 30 seconds
                await self._update_gas_prices()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Gas price monitor error: {e}")
                await asyncio.sleep(60)  # Longer pause on error

    async def _cleanup_loop(self) -> None:
        """Cleanup old transactions and batches."""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                await self._cleanup_old_transactions()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")
                await asyncio.sleep(60)

    @timeout_async(1.0)
    async def submit_transaction(
        self,
        tx_type: BatchTransactionType,
        payload: dict[str, Any],
        priority: TransactionPriority = TransactionPriority.NORMAL,
        callback: Callable[[TransactionResult], None] | None = None
    ) -> str:
        """Submit a transaction for processing."""
        try:
            # Create transaction request
            tx_request = TransactionRequest(
                tx_id=f"tx_{uuid.uuid4().hex[:8]}",
                tx_type=tx_type,
                priority=priority,
                payload=payload,
                callback=callback
            )

            # Add to appropriate queue
            self.transaction_queues[priority].append(tx_request)

            logger.debug(f"Transaction queued: {tx_request.tx_id} (priority: {priority.name})")
            return tx_request.tx_id

        except Exception as e:
            logger.error(f"Failed to submit transaction: {e}")
            raise BlockchainError(f"Transaction submission failed: {e}") from e

    @timeout_async(2.0)
    async def submit_batch(
        self,
        batch_type: BatchTransactionType,
        transactions: list[dict[str, Any]],
        priority: TransactionPriority = TransactionPriority.NORMAL,
        deadline: datetime | None = None
    ) -> str:
        """Submit a batch of related transactions."""
        try:
            # Create transaction requests
            tx_requests = []
            for tx_data in transactions:
                tx_request = TransactionRequest(
                    tx_id=f"tx_{uuid.uuid4().hex[:8]}",
                    tx_type=batch_type,
                    priority=priority,
                    payload=tx_data
                )
                tx_requests.append(tx_request)

            # Create batch
            batch = BatchTransaction(
                batch_id=f"batch_{uuid.uuid4().hex[:8]}",
                batch_type=batch_type,
                priority=priority,
                transactions=tx_requests,
                deadline=deadline
            )

            # Estimate gas for batch
            await self._estimate_batch_gas(batch)

            # Store batch
            self.pending_batches[batch.batch_id] = batch

            logger.debug(f"Batch queued: {batch.batch_id} with {len(tx_requests)} transactions")
            return batch.batch_id

        except Exception as e:
            logger.error(f"Failed to submit batch: {e}")
            raise BlockchainError(f"Batch submission failed: {e}") from e

    async def _process_transaction_queues(self) -> None:
        """Process transactions from priority queues."""
        # Check if we can process more batches
        if len(self.active_transactions) >= self.max_concurrent_batches * self.max_batch_size:
            return

        # Process by priority (highest first)
        for priority in reversed(list(TransactionPriority)):
            queue = self.transaction_queues[priority]

            if not queue:
                continue

            # Collect transactions for batching
            batch_transactions = []
            while queue and len(batch_transactions) < self.max_batch_size:
                tx_request = queue.popleft()
                batch_transactions.append(tx_request)

            if batch_transactions:
                await self._create_and_process_batch(batch_transactions)
                break  # Process one priority level at a time

    async def _create_and_process_batch(self, transactions: list[TransactionRequest]) -> None:
        """Create and process a batch from individual transactions."""
        try:
            # Group by transaction type for better batching
            type_groups = defaultdict(list)
            for tx in transactions:
                type_groups[tx.tx_type].append(tx)

            # Process each group as a separate batch
            for tx_type, tx_group in type_groups.items():
                if not tx_group:
                    continue

                batch = BatchTransaction(
                    batch_id=f"batch_{uuid.uuid4().hex[:8]}",
                    batch_type=tx_type,
                    priority=tx_group[0].priority,  # Use first transaction's priority
                    transactions=tx_group
                )

                # Estimate gas
                await self._estimate_batch_gas(batch)

                # Process batch
                await self._process_batch(batch)

        except Exception as e:
            logger.error(f"Failed to create and process batch: {e}")
            # Return transactions to queue for retry
            for tx in transactions:
                if tx.retry_count < tx.max_retries:
                    tx.retry_count += 1
                    self.transaction_queues[tx.priority].append(tx)

    async def _process_pending_batches(self) -> None:
        """Process batches that are ready for execution."""
        current_time = datetime.now()
        ready_batches = []

        for batch_id, batch in list(self.pending_batches.items()):
            # Check if batch is ready
            batch_age = (current_time - batch.created_at).total_seconds()

            if (batch_age >= self.batch_timeout_seconds or
                len(batch.transactions) >= self.max_batch_size or
                (batch.deadline and current_time >= batch.deadline)):

                ready_batches.append(batch)
                del self.pending_batches[batch_id]

        # Process ready batches
        for batch in ready_batches:
            await self._process_batch(batch)

    @measure_latency("transaction_batch_processing")
    async def _process_batch(self, batch: BatchTransaction) -> None:
        """Process a transaction batch."""
        try:
            logger.debug(f"Processing batch {batch.batch_id} with {len(batch.transactions)} transactions")

            # Apply MEV protection
            if self.mev_protection_enabled:
                await self._apply_mev_protection(batch)

            # Optimize gas prices
            if self.gas_optimization_enabled:
                await self._optimize_batch_gas(batch)

            # Execute batch
            results = await self._execute_batch(batch)

            # Process results
            await self._process_batch_results(batch, results)

            logger.debug(f"Batch {batch.batch_id} processing completed")

        except Exception as e:
            logger.error(f"Failed to process batch {batch.batch_id}: {e}")
            await self._handle_batch_failure(batch, str(e))

    async def _estimate_batch_gas(self, batch: BatchTransaction) -> None:
        """Estimate gas requirements for batch."""
        try:
            total_gas = 0

            for tx_request in batch.transactions:
                # Estimate gas based on transaction type
                estimated_gas = await self._estimate_transaction_gas(tx_request)
                total_gas += estimated_gas

                if not tx_request.max_gas:
                    tx_request.max_gas = estimated_gas

            batch.estimated_gas = total_gas

            # Calculate estimated fee
            gas_price = await self._get_gas_price_for_priority(batch.priority)
            batch.estimated_fee = Decimal(total_gas) * gas_price

        except Exception as e:
            logger.warning(f"Failed to estimate batch gas: {e}")
            # Use default values
            batch.estimated_gas = len(batch.transactions) * 100000  # Default gas per tx
            batch.estimated_fee = Decimal("1.0")  # Default fee

    async def _estimate_transaction_gas(self, tx_request: TransactionRequest) -> int:
        """Estimate gas for individual transaction."""
        # Base gas estimates by transaction type
        gas_estimates = {
            BatchTransactionType.ORDER_PLACEMENT: 150000,
            BatchTransactionType.ORDER_CANCELLATION: 100000,
            BatchTransactionType.ORDER_MODIFICATION: 120000,
            BatchTransactionType.POSITION_UPDATE: 80000,
            BatchTransactionType.BALANCE_ADJUSTMENT: 60000
        }

        base_gas = gas_estimates.get(tx_request.tx_type, 100000)

        # Adjust based on payload complexity
        payload_complexity = len(json.dumps(tx_request.payload))
        complexity_adjustment = min(payload_complexity // 100, 50000)  # Cap adjustment

        return base_gas + complexity_adjustment

    async def _apply_mev_protection(self, batch: BatchTransaction) -> None:
        """Apply MEV protection strategies."""
        try:
            # Add random delay for high-value transactions
            if batch.priority in [TransactionPriority.HIGH, TransactionPriority.CRITICAL]:
                delay = 0.1 + (hash(batch.batch_id) % 100) / 1000  # 0.1-0.2 second delay
                await asyncio.sleep(delay)

            # Randomize transaction order within batch (if applicable)
            if len(batch.transactions) > 1 and batch.batch_type == BatchTransactionType.ORDER_PLACEMENT:
                import random
                random.shuffle(batch.transactions)

            # Record transaction for monitoring
            self._recent_transactions.append({
                'batch_id': batch.batch_id,
                'timestamp': datetime.now(),
                'type': batch.batch_type.value,
                'priority': batch.priority.value
            })

        except Exception as e:
            logger.warning(f"MEV protection failed for batch {batch.batch_id}: {e}")

    async def _optimize_batch_gas(self, batch: BatchTransaction) -> None:
        """Optimize gas prices for batch."""
        try:
            # Get current network gas price
            network_gas_price = await self.sei_client.get_gas_price()

            # Calculate optimal gas price based on priority
            priority_multiplier = self.gas_price_multipliers[batch.priority]
            optimal_gas_price = network_gas_price * priority_multiplier

            # Apply gas price to all transactions in batch
            for tx_request in batch.transactions:
                if not tx_request.gas_price:
                    tx_request.gas_price = optimal_gas_price

            # Update batch fee estimate
            if batch.estimated_gas:
                batch.estimated_fee = Decimal(batch.estimated_gas) * optimal_gas_price

        except Exception as e:
            logger.warning(f"Gas optimization failed for batch {batch.batch_id}: {e}")

    async def _execute_batch(self, batch: BatchTransaction) -> list[TransactionResult]:
        """Execute transaction batch."""
        results = []

        try:
            # Execute transactions concurrently (if safe)
            if self._can_execute_concurrently(batch):
                tasks = [
                    self._execute_transaction(tx_request)
                    for tx_request in batch.transactions
                ]
                gathered_results = await asyncio.gather(*tasks, return_exceptions=True)

                # Filter out exceptions and convert to TransactionResult
                for result in gathered_results:
                    if isinstance(result, TransactionResult):
                        results.append(result)
                    elif isinstance(result, Exception):
                        results.append(TransactionResult(
                            tx_hash="",
                            status=TransactionStatus.FAILED,
                            error_message=str(result)
                        ))
            else:
                # Execute sequentially
                for tx_request in batch.transactions:
                    result = await self._execute_transaction(tx_request)
                    results.append(result)

            return results

        except Exception as e:
            logger.error(f"Batch execution failed: {e}")
            # Return error results for all transactions
            return [
                TransactionResult(
                    tx_hash="",
                    status=TransactionStatus.FAILED,
                    error_message=str(e)
                ) for _ in batch.transactions
            ]

    def _can_execute_concurrently(self, batch: BatchTransaction) -> bool:
        """Determine if batch transactions can be executed concurrently."""
        # Cancellations can be concurrent
        if batch.batch_type == BatchTransactionType.ORDER_CANCELLATION:
            return True

        # Same-market orders might conflict, execute sequentially
        if batch.batch_type == BatchTransactionType.ORDER_PLACEMENT:
            markets = set()
            for tx in batch.transactions:
                market = tx.payload.get('market', '')
                if market in markets:
                    return False  # Same market, potential conflict
                markets.add(market)
            return True

        # Default to sequential for safety
        return False

    async def _execute_transaction(self, tx_request: TransactionRequest) -> TransactionResult:
        """Execute individual transaction."""
        try:
            # Mark as active
            self.active_transactions[tx_request.tx_id] = tx_request

            # Convert transaction to blockchain format
            tx_bytes = await self._convert_to_blockchain_transaction(tx_request)

            # Broadcast transaction
            result = await self.sei_client.broadcast_transaction(tx_bytes)

            # Update statistics
            self._update_transaction_stats(result)

            return result

        except Exception as e:
            logger.error(f"Transaction execution failed: {e}")
            return TransactionResult(
                tx_hash="",
                status=TransactionStatus.FAILED,
                error_message=str(e)
            )
        finally:
            # Remove from active transactions
            self.active_transactions.pop(tx_request.tx_id, None)

    async def _convert_to_blockchain_transaction(self, tx_request: TransactionRequest) -> bytes:
        """Convert transaction request to blockchain transaction bytes."""
        # This is a simplified implementation
        # In production, this would create proper Sei transaction bytes

        # Mock transaction bytes for development
        tx_data = {
            'type': tx_request.tx_type.value,
            'payload': tx_request.payload,
            'gas': tx_request.max_gas or 100000,
            'gas_price': str(tx_request.gas_price or self.base_gas_price)
        }

        return json.dumps(tx_data).encode('utf-8')

    async def _process_batch_results(self, batch: BatchTransaction, results: list[TransactionResult]) -> None:
        """Process batch execution results."""
        try:
            successful_count = 0
            failed_count = 0

            for tx_request, result in zip(batch.transactions, results, strict=False):
                # Store result
                self.completed_transactions[tx_request.tx_id] = result

                # Update counters
                if result.status == TransactionStatus.CONFIRMED:
                    successful_count += 1
                else:
                    failed_count += 1

                # Call callback if provided
                if tx_request.callback:
                    try:
                        tx_request.callback(result)  # Callback is sync, not async
                    except Exception as e:
                        logger.warning(f"Transaction callback failed: {e}")

                # Handle retries for failed transactions
                if result.status == TransactionStatus.FAILED and tx_request.retry_count < tx_request.max_retries:
                    tx_request.retry_count += 1
                    self.transaction_queues[tx_request.priority].append(tx_request)

            logger.info(f"Batch {batch.batch_id} completed: {successful_count} successful, {failed_count} failed")

        except Exception as e:
            logger.error(f"Failed to process batch results: {e}")

    async def _handle_batch_failure(self, batch: BatchTransaction, error_message: str) -> None:
        """Handle batch processing failure."""
        try:
            # Return transactions to queue for retry
            for tx_request in batch.transactions:
                if tx_request.retry_count < tx_request.max_retries:
                    tx_request.retry_count += 1
                    self.transaction_queues[tx_request.priority].append(tx_request)
                else:
                    # Max retries reached, mark as failed
                    result = TransactionResult(
                        tx_hash="",
                        status=TransactionStatus.FAILED,
                        error_message=f"Max retries exceeded: {error_message}"
                    )
                    self.completed_transactions[tx_request.tx_id] = result

                    if tx_request.callback:
                        tx_request.callback(result)  # Callback is sync, not async

            logger.warning(f"Batch {batch.batch_id} failed: {error_message}")

        except Exception as e:
            logger.error(f"Failed to handle batch failure: {e}")

    async def _update_gas_prices(self) -> None:
        """Update gas price information."""
        try:
            current_gas_price = await self.sei_client.get_gas_price()

            # Update base gas price if significantly different
            price_change = abs(current_gas_price - self.base_gas_price) / self.base_gas_price
            if price_change > 0.1:  # 10% change threshold
                logger.info(f"Gas price updated: {self.base_gas_price} -> {current_gas_price}")
                self.base_gas_price = current_gas_price

        except Exception as e:
            logger.warning(f"Failed to update gas prices: {e}")

    async def _get_gas_price_for_priority(self, priority: TransactionPriority) -> Decimal:
        """Get gas price for specific priority level."""
        multiplier = self.gas_price_multipliers.get(priority, Decimal("1.0"))
        return self.base_gas_price * multiplier

    def _update_transaction_stats(self, result: TransactionResult) -> None:
        """Update transaction statistics."""
        self.stats.total_transactions += 1

        if result.status == TransactionStatus.CONFIRMED:
            self.stats.successful_transactions += 1

            if result.gas_used:
                self.stats.total_gas_used += result.gas_used

            if result.fee_amount:
                self.stats.total_fees_paid += result.fee_amount
        else:
            self.stats.failed_transactions += 1

        # Update success rate
        if self.stats.total_transactions > 0:
            self.stats.batch_efficiency = (
                self.stats.successful_transactions / self.stats.total_transactions
            )

    async def _cleanup_old_transactions(self) -> None:
        """Clean up old completed transactions."""
        cutoff_time = datetime.now() - timedelta(hours=1)
        old_tx_ids = [
            tx_id for tx_id, result in self.completed_transactions.items()
            if result.timestamp and result.timestamp < cutoff_time
        ]

        for tx_id in old_tx_ids:
            self.completed_transactions.pop(tx_id, None)

        if old_tx_ids:
            logger.debug(f"Cleaned up {len(old_tx_ids)} old transactions")

    def get_transaction_status(self, tx_id: str) -> TransactionResult | None:
        """Get status of transaction."""
        return self.completed_transactions.get(tx_id)

    def get_queue_status(self) -> dict[str, Any]:
        """Get current queue status."""
        return {
            'pending_by_priority': {
                priority.name: len(queue)
                for priority, queue in self.transaction_queues.items()
            },
            'pending_batches': len(self.pending_batches),
            'active_transactions': len(self.active_transactions),
            'completed_transactions': len(self.completed_transactions)
        }

    def get_performance_stats(self) -> dict[str, Any]:
        """Get transaction manager performance statistics."""
        return {
            'total_transactions': self.stats.total_transactions,
            'successful_transactions': self.stats.successful_transactions,
            'failed_transactions': self.stats.failed_transactions,
            'success_rate': self.stats.batch_efficiency,
            'total_gas_used': self.stats.total_gas_used,
            'total_fees_paid': float(self.stats.total_fees_paid),
            'avg_gas_price': float(self.base_gas_price),
            'queue_status': self.get_queue_status()
        }

    async def cleanup(self) -> None:
        """Cleanup resources and stop background tasks."""
        # Cancel background tasks
        tasks = [self._processor_task, self._gas_monitor_task, self._cleanup_task]

        for task in tasks:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        logger.info("Transaction manager cleanup completed")

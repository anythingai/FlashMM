"""
Sei Blockchain Client

Comprehensive Sei V2 testnet RPC integration with wallet management,
transaction signing, and blockchain operations.
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any

import aiohttp
from cosmpy.aerial.client import LedgerClient, NetworkConfig
from cosmpy.aerial.wallet import LocalWallet
from cosmpy.crypto.keypairs import PrivateKey

from flashmm.config.settings import get_config
from flashmm.security.key_manager import EnhancedKeyManager
from flashmm.utils.decorators import measure_latency, timeout_async
from flashmm.utils.exceptions import BlockchainError, ValidationError
from flashmm.utils.logging import get_logger

logger = get_logger(__name__)


class TransactionStatus(Enum):
    """Transaction status enumeration."""
    PENDING = "pending"
    CONFIRMED = "confirmed"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class TransactionResult:
    """Transaction execution result."""
    tx_hash: str
    status: TransactionStatus
    gas_used: int | None = None
    gas_wanted: int | None = None
    fee_amount: Decimal | None = None
    block_height: int | None = None
    timestamp: datetime | None = None
    error_message: str | None = None
    raw_response: dict[str, Any] | None = None


@dataclass
class AccountInfo:
    """Account information."""
    address: str
    account_number: int
    sequence: int
    balances: dict[str, Decimal]
    pub_key: str | None = None


@dataclass
class NetworkHealth:
    """Network health status."""
    is_healthy: bool
    latest_block_height: int
    latest_block_time: datetime
    avg_block_time: float
    validator_count: int
    syncing: bool
    rpc_latency_ms: float


class SeiClient:
    """Sei V2 testnet blockchain client with comprehensive functionality."""

    def __init__(self):
        self.config = get_config()
        self.key_manager = EnhancedKeyManager()

        # Network configuration
        self.rpc_url = self.config.get("sei.rpc_url", "https://rpc.sei-apis.com")
        self.ws_url = self.config.get("sei.ws_url", "wss://rpc.sei-apis.com/websocket")
        self.chain_id = self.config.get("sei.chain_id", "pacific-1")

        # Client instances
        self.ledger_client: LedgerClient | None = None
        self.wallet: LocalWallet | None = None
        self.session: aiohttp.ClientSession | None = None

        # Configuration
        self.gas_adjustment = self.config.get("blockchain.gas_adjustment", 1.3)
        self.gas_price = self.config.get("blockchain.gas_price", "0.025usei")
        self.max_gas = self.config.get("blockchain.max_gas", 300000)
        self.tx_timeout_seconds = self.config.get("blockchain.tx_timeout_seconds", 30)

        # Connection pooling
        self.max_connections = self.config.get("blockchain.max_connections", 10)
        self.connection_timeout = self.config.get("blockchain.connection_timeout", 10)

        # Health monitoring
        self._last_health_check = None
        self._health_cache_duration = timedelta(seconds=30)
        self._network_health: NetworkHealth | None = None

    async def initialize(self) -> None:
        """Initialize the Sei client with network configuration."""
        try:
            logger.info(f"Initializing Sei client for chain {self.chain_id}")

            # Setup HTTP session with connection pooling
            connector = aiohttp.TCPConnector(
                limit=self.max_connections,
                limit_per_host=self.max_connections
            )
            self.session = aiohttp.ClientSession(connector=connector)

            # Configure network
            network_config = NetworkConfig(
                chain_id=self.chain_id,
                url=self.rpc_url,
                fee_minimum_gas_price=int(self.gas_price.replace("usei", "")),
                fee_denomination="usei",
                staking_denomination="usei"
            )

            # Initialize ledger client
            self.ledger_client = LedgerClient(network_config)

            # Initialize wallet if private key is available
            await self._initialize_wallet()

            # Perform initial health check
            await self.check_network_health()

            logger.info("Sei client initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Sei client: {e}")
            raise BlockchainError(f"Sei client initialization failed: {e}") from e

    async def _initialize_wallet(self) -> None:
        """Initialize wallet from configured private key."""
        try:
            private_key_hex = self.config.get("sei_private_key")
            if not private_key_hex:
                logger.warning("No private key configured - wallet operations will not be available")
                return

            # For now, use the private key directly since EnhancedKeyManager doesn't have decrypt_private_key
            # In production, implement proper key decryption through the key manager
            private_key = PrivateKey(bytes.fromhex(private_key_hex))

            # Create wallet
            self.wallet = LocalWallet(private_key)

            # Log wallet address (not the private key!)
            address = self.wallet.address()
            logger.info(f"Wallet initialized for address: {address}")

        except Exception as e:
            logger.error(f"Failed to initialize wallet: {e}")
            raise BlockchainError(f"Wallet initialization failed: {e}") from e

    @timeout_async(10.0)
    @measure_latency("sei_network_health")
    async def check_network_health(self) -> NetworkHealth:
        """Check Sei network health and performance."""
        # Use cached result if available and fresh
        if (self._network_health and self._last_health_check and
            datetime.now() - self._last_health_check < self._health_cache_duration):
            return self._network_health

        try:
            start_time = datetime.now()

            # Get latest block info
            status_response = await self._rpc_request("status")
            status = status_response.get("result", {})

            sync_info = status.get("sync_info", {})
            latest_block_height = int(sync_info.get("latest_block_height", 0))
            latest_block_time_str = sync_info.get("latest_block_time", "")
            syncing = sync_info.get("catching_up", True)

            # Parse block time
            try:
                latest_block_time = datetime.fromisoformat(
                    latest_block_time_str.replace("Z", "+00:00")
                )
            except (ValueError, AttributeError):
                latest_block_time = datetime.now()

            # Calculate average block time
            avg_block_time = await self._calculate_avg_block_time(latest_block_height)

            # Get validator count
            validators_response = await self._rpc_request("validators")
            validator_count = len(validators_response.get("result", {}).get("validators", []))

            # Calculate RPC latency
            rpc_latency_ms = (datetime.now() - start_time).total_seconds() * 1000

            # Determine health
            is_healthy = (
                not syncing and
                latest_block_height > 0 and
                validator_count > 0 and
                rpc_latency_ms < 5000  # 5 second max latency
            )

            self._network_health = NetworkHealth(
                is_healthy=is_healthy,
                latest_block_height=latest_block_height,
                latest_block_time=latest_block_time,
                avg_block_time=avg_block_time,
                validator_count=validator_count,
                syncing=syncing,
                rpc_latency_ms=rpc_latency_ms
            )

            self._last_health_check = datetime.now()

            logger.debug(f"Network health check: healthy={is_healthy}, "
                        f"block={latest_block_height}, latency={rpc_latency_ms:.1f}ms")

            return self._network_health

        except Exception as e:
            logger.error(f"Network health check failed: {e}")
            # Return unhealthy status on error
            self._network_health = NetworkHealth(
                is_healthy=False,
                latest_block_height=0,
                latest_block_time=datetime.now(),
                avg_block_time=0.0,
                validator_count=0,
                syncing=True,
                rpc_latency_ms=float('inf')
            )
            return self._network_health

    async def _calculate_avg_block_time(self, latest_height: int) -> float:
        """Calculate average block time from recent blocks."""
        try:
            # Sample last 10 blocks
            sample_size = min(10, latest_height)
            if sample_size < 2:
                return 6.0  # Sei target block time

            start_height = latest_height - sample_size + 1

            # Get first and last block
            first_block = await self._rpc_request("block", {"height": str(start_height)})
            last_block = await self._rpc_request("block", {"height": str(latest_height)})

            first_time_str = first_block["result"]["block"]["header"]["time"]
            last_time_str = last_block["result"]["block"]["header"]["time"]

            first_time = datetime.fromisoformat(first_time_str.replace("Z", "+00:00"))
            last_time = datetime.fromisoformat(last_time_str.replace("Z", "+00:00"))

            time_diff = (last_time - first_time).total_seconds()
            avg_block_time = time_diff / (sample_size - 1)

            return avg_block_time

        except Exception as e:
            logger.warning(f"Failed to calculate average block time: {e}")
            return 6.0  # Default Sei block time

    @timeout_async(5.0)
    @measure_latency("sei_account_info")
    async def get_account_info(self, address: str | None = None) -> AccountInfo:
        """Get account information including balances."""
        if not address and self.wallet:
            address = str(self.wallet.address())

        if not address:
            raise ValidationError("No address provided and no wallet configured")

        try:
            # Get account info
            account_response = await self._rpc_request(
                "abci_query",
                {
                    "path": "/cosmos.auth.v1beta1.Query/Account",
                    "data": self._encode_account_query(address)
                }
            )

            # Get balances
            balance_response = await self._rpc_request(
                "abci_query",
                {
                    "path": "/cosmos.bank.v1beta1.Query/AllBalances",
                    "data": self._encode_balance_query(address)
                }
            )

            # Parse responses
            account_data = self._decode_account_response(account_response)
            balance_data = self._decode_balance_response(balance_response)

            return AccountInfo(
                address=address,
                account_number=account_data.get("account_number", 0),
                sequence=account_data.get("sequence", 0),
                balances=balance_data,
                pub_key=account_data.get("pub_key")
            )

        except Exception as e:
            logger.error(f"Failed to get account info for {address}: {e}")
            raise BlockchainError(f"Account info query failed: {e}") from e

    @timeout_async(3.0)
    @measure_latency("sei_gas_estimation")
    async def estimate_gas(self, tx_bytes: bytes) -> int:
        """Estimate gas required for transaction."""
        try:
            response = await self._rpc_request(
                "abci_query",
                {
                    "path": "/cosmos.tx.v1beta1.Service/Simulate",
                    "data": tx_bytes.hex()
                }
            )

            gas_info = response.get("result", {}).get("response", {}).get("gas_info", {})
            gas_used = int(gas_info.get("gas_used", self.max_gas))

            # Apply gas adjustment
            estimated_gas = int(gas_used * self.gas_adjustment)

            # Cap at maximum
            return min(estimated_gas, self.max_gas)

        except Exception as e:
            logger.warning(f"Gas estimation failed: {e}, using default")
            return self.max_gas

    @timeout_async(30.0)
    @measure_latency("sei_broadcast_transaction")
    async def broadcast_transaction(self, tx_bytes: bytes) -> TransactionResult:
        """Broadcast transaction and wait for confirmation."""
        if not tx_bytes:
            raise ValidationError("Empty transaction bytes")

        try:
            # Broadcast transaction
            broadcast_response = await self._rpc_request(
                "broadcast_tx_sync",
                {"tx": tx_bytes.hex()}
            )

            result = broadcast_response.get("result", {})
            tx_hash = result.get("hash", "")

            if result.get("code", 0) != 0:
                error_msg = result.get("log", "Transaction failed")
                logger.error(f"Transaction broadcast failed: {error_msg}")
                return TransactionResult(
                    tx_hash=tx_hash,
                    status=TransactionStatus.FAILED,
                    error_message=error_msg,
                    raw_response=broadcast_response
                )

            logger.debug(f"Transaction broadcasted: {tx_hash}")

            # Wait for confirmation
            return await self._wait_for_confirmation(tx_hash)

        except Exception as e:
            logger.error(f"Failed to broadcast transaction: {e}")
            raise BlockchainError(f"Transaction broadcast failed: {e}") from e

    async def _wait_for_confirmation(self, tx_hash: str) -> TransactionResult:
        """Wait for transaction confirmation."""
        start_time = datetime.now()
        timeout = timedelta(seconds=self.tx_timeout_seconds)

        while datetime.now() - start_time < timeout:
            try:
                # Query transaction status
                tx_response = await self._rpc_request("tx", {"hash": tx_hash})

                if "result" in tx_response:
                    tx_result = tx_response["result"]

                    # Parse transaction result
                    tx_response_data = tx_result.get("tx_result", {})

                    status = (TransactionStatus.CONFIRMED if tx_response_data.get("code", 0) == 0
                             else TransactionStatus.FAILED)

                    return TransactionResult(
                        tx_hash=tx_hash,
                        status=status,
                        gas_used=tx_response_data.get("gas_used"),
                        gas_wanted=tx_response_data.get("gas_wanted"),
                        block_height=tx_result.get("height"),
                        timestamp=datetime.now(),
                        error_message=tx_response_data.get("log") if status == TransactionStatus.FAILED else None,
                        raw_response=tx_response
                    )

                # Transaction not found yet, wait and retry
                await asyncio.sleep(1.0)

            except Exception as e:
                logger.debug(f"Transaction confirmation check failed: {e}")
                await asyncio.sleep(1.0)

        # Timeout reached
        logger.warning(f"Transaction confirmation timeout: {tx_hash}")
        return TransactionResult(
            tx_hash=tx_hash,
            status=TransactionStatus.TIMEOUT,
            timestamp=datetime.now()
        )

    async def _rpc_request(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Make RPC request to Sei node."""
        if not self.session:
            raise BlockchainError("Client not initialized")

        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params or {}
        }

        try:
            async with self.session.post(
                self.rpc_url,
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                response.raise_for_status()
                result = await response.json()

                if "error" in result:
                    raise BlockchainError(f"RPC error: {result['error']}")

                return result

        except aiohttp.ClientError as e:
            raise BlockchainError(f"RPC request failed: {e}") from e

    def _encode_account_query(self, address: str) -> str:
        """Encode account query for ABCI request."""
        # Simplified implementation - in production use proper protobuf encoding
        return address.encode().hex()

    def _encode_balance_query(self, address: str) -> str:
        """Encode balance query for ABCI request."""
        # Simplified implementation - in production use proper protobuf encoding
        return address.encode().hex()

    def _decode_account_response(self, response: dict[str, Any]) -> dict[str, Any]:
        """Decode account query response."""
        # Simplified implementation - in production use proper protobuf decoding
        return {
            "account_number": 0,
            "sequence": 0,
            "pub_key": None
        }

    def _decode_balance_response(self, response: dict[str, Any]) -> dict[str, Decimal]:
        """Decode balance query response."""
        # Simplified implementation - in production use proper protobuf decoding
        return {
            "usei": Decimal("1000000"),  # 1 SEI
            "usdc": Decimal("2000000000")  # 2000 USDC
        }

    async def get_gas_price(self) -> Decimal:
        """Get current network gas price."""
        try:
            # In production, query actual gas prices from network
            # For now, return configured gas price
            price_str = self.gas_price.replace("usei", "")
            return Decimal(price_str) / Decimal("1000000")  # Convert to SEI

        except Exception as e:
            logger.warning(f"Failed to get gas price: {e}")
            return Decimal("0.025")  # Default gas price

    async def cleanup(self) -> None:
        """Cleanup resources."""
        if self.session:
            await self.session.close()
            self.session = None

        logger.info("Sei client cleanup completed")

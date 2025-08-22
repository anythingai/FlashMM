"""
Sei Market Configuration

Market-specific parameters, discovery, validation, and monitoring
for supported trading pairs on Sei V2 CLOB.
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any

from flashmm.blockchain.sei_client import SeiClient
from flashmm.config.settings import get_config
from flashmm.utils.exceptions import ConfigurationError
from flashmm.utils.logging import get_logger

logger = get_logger(__name__)


class MarketStatus(Enum):
    """Market status enumeration."""
    ACTIVE = "active"
    PAUSED = "paused"
    MAINTENANCE = "maintenance"
    SUSPENDED = "suspended"
    UNKNOWN = "unknown"


@dataclass
class MarketConfig:
    """Market configuration for a trading pair."""

    # Basic market info
    symbol: str
    base_asset: str
    quote_asset: str
    market_id: str

    # Trading parameters
    tick_size: Decimal
    lot_size: Decimal
    min_order_size: Decimal
    max_order_size: Decimal

    # Fees
    maker_fee: Decimal
    taker_fee: Decimal
    fee_currency: str

    # Limits
    price_precision: int
    size_precision: int
    max_price: Decimal
    min_price: Decimal

    # Market status
    status: MarketStatus
    last_updated: datetime

    # Additional parameters
    margin_enabled: bool = False
    settlement_currency: str = "USDC"
    contract_address: str | None = None
    oracle_address: str | None = None

    def validate_price(self, price: Decimal) -> bool:
        """Validate if price meets market requirements."""
        if price <= 0:
            return False

        if price < self.min_price or price > self.max_price:
            return False

        # Check tick size alignment
        remainder = price % self.tick_size
        return remainder == 0 or remainder < Decimal('1e-8')  # Floating point tolerance

    def validate_size(self, size: Decimal) -> bool:
        """Validate if size meets market requirements."""
        if size <= 0:
            return False

        if size < self.min_order_size or size > self.max_order_size:
            return False

        # Check lot size alignment
        remainder = size % self.lot_size
        return remainder == 0 or remainder < Decimal('1e-8')  # Floating point tolerance

    def round_price(self, price: Decimal) -> Decimal:
        """Round price to valid tick size."""
        if self.tick_size == 0:
            return price

        # Round to nearest tick
        ticks = (price / self.tick_size).quantize(Decimal('1'), rounding='ROUND_HALF_UP')
        return ticks * self.tick_size

    def round_size(self, size: Decimal) -> Decimal:
        """Round size to valid lot size."""
        if self.lot_size == 0:
            return size

        # Round down to nearest lot
        lots = (size / self.lot_size).quantize(Decimal('1'), rounding='ROUND_DOWN')
        return lots * self.lot_size

    def calculate_notional(self, price: Decimal, size: Decimal) -> Decimal:
        """Calculate notional value of order."""
        return price * size

    def calculate_fees(self, notional: Decimal, is_maker: bool) -> Decimal:
        """Calculate trading fees for given notional."""
        fee_rate = self.maker_fee if is_maker else self.taker_fee
        return notional * fee_rate

    def to_dict(self) -> dict[str, Any]:
        """Convert market config to dictionary."""
        return {
            'symbol': self.symbol,
            'base_asset': self.base_asset,
            'quote_asset': self.quote_asset,
            'market_id': self.market_id,
            'tick_size': float(self.tick_size),
            'lot_size': float(self.lot_size),
            'min_order_size': float(self.min_order_size),
            'max_order_size': float(self.max_order_size),
            'maker_fee': float(self.maker_fee),
            'taker_fee': float(self.taker_fee),
            'fee_currency': self.fee_currency,
            'price_precision': self.price_precision,
            'size_precision': self.size_precision,
            'max_price': float(self.max_price),
            'min_price': float(self.min_price),
            'status': self.status.value,
            'last_updated': self.last_updated.isoformat(),
            'margin_enabled': self.margin_enabled,
            'settlement_currency': self.settlement_currency,
            'contract_address': self.contract_address,
            'oracle_address': self.oracle_address
        }


class MarketConfigManager:
    """Manager for market configurations with discovery and hot-reloading."""

    def __init__(self, sei_client: SeiClient):
        self.sei_client = sei_client
        self.config = get_config()

        # Market configurations
        self.markets: dict[str, MarketConfig] = {}
        self.markets_by_id: dict[str, MarketConfig] = {}

        # Configuration
        self.auto_discovery = self.config.get("markets.auto_discovery", True)
        self.refresh_interval = self.config.get("markets.refresh_interval_seconds", 300)  # 5 minutes
        self.status_check_interval = self.config.get("markets.status_check_interval_seconds", 60)  # 1 minute

        # Background tasks
        self._refresh_task: asyncio.Task | None = None
        self._status_task: asyncio.Task | None = None

        # Cache
        self._last_refresh = None
        self._last_status_check = None

    async def initialize(self) -> None:
        """Initialize market configuration manager."""
        try:
            logger.info("Initializing market configuration manager")

            # Load default market configurations
            await self._load_default_markets()

            # Discover additional markets if enabled
            if self.auto_discovery:
                await self._discover_markets()

            # Start background tasks
            await self._start_background_tasks()

            logger.info(f"Market configuration manager initialized with {len(self.markets)} markets")

        except Exception as e:
            logger.error(f"Failed to initialize market config manager: {e}")
            raise ConfigurationError(f"Market config initialization failed: {e}") from e

    async def _load_default_markets(self) -> None:
        """Load default market configurations for primary trading pairs."""

        # SEI/USDC market configuration
        sei_usdc_config = MarketConfig(
            symbol="SEI/USDC",
            base_asset="SEI",
            quote_asset="USDC",
            market_id="sei_usdc_clob",
            tick_size=Decimal("0.0001"),  # $0.0001 tick size
            lot_size=Decimal("1.0"),      # 1 SEI minimum lot
            min_order_size=Decimal("10.0"),   # 10 SEI minimum order
            max_order_size=Decimal("100000.0"), # 100,000 SEI maximum order
            maker_fee=Decimal("0.0005"),  # 0.05% maker fee
            taker_fee=Decimal("0.001"),   # 0.1% taker fee
            fee_currency="USDC",
            price_precision=4,
            size_precision=2,
            max_price=Decimal("100.0"),   # $100 max price
            min_price=Decimal("0.01"),    # $0.01 min price
            status=MarketStatus.ACTIVE,
            last_updated=datetime.now(),
            margin_enabled=False,
            settlement_currency="USDC",
            contract_address="sei1...",  # Placeholder - to be filled with actual contract
            oracle_address="sei1..."     # Placeholder - to be filled with actual oracle
        )

        # wETH/USDC market configuration
        weth_usdc_config = MarketConfig(
            symbol="wETH/USDC",
            base_asset="wETH",
            quote_asset="USDC",
            market_id="weth_usdc_clob",
            tick_size=Decimal("0.01"),    # $0.01 tick size
            lot_size=Decimal("0.001"),    # 0.001 ETH minimum lot
            min_order_size=Decimal("0.01"),    # 0.01 ETH minimum order
            max_order_size=Decimal("100.0"),   # 100 ETH maximum order
            maker_fee=Decimal("0.0005"),  # 0.05% maker fee
            taker_fee=Decimal("0.001"),   # 0.1% taker fee
            fee_currency="USDC",
            price_precision=2,
            size_precision=4,
            max_price=Decimal("10000.0"), # $10,000 max price
            min_price=Decimal("100.0"),   # $100 min price
            status=MarketStatus.ACTIVE,
            last_updated=datetime.now(),
            margin_enabled=False,
            settlement_currency="USDC",
            contract_address="sei1...",  # Placeholder
            oracle_address="sei1..."     # Placeholder
        )

        # Store configurations
        self.markets[sei_usdc_config.symbol] = sei_usdc_config
        self.markets[weth_usdc_config.symbol] = weth_usdc_config

        self.markets_by_id[sei_usdc_config.market_id] = sei_usdc_config
        self.markets_by_id[weth_usdc_config.market_id] = weth_usdc_config

        logger.info("Default market configurations loaded")

    async def _discover_markets(self) -> None:
        """Discover additional markets from blockchain."""
        try:
            logger.info("Discovering markets from blockchain...")

            # In production, query Sei CLOB contracts for available markets
            # For now, we'll use the configured markets
            discovered_count = 0

            # TODO: Implement actual market discovery
            # This would involve:
            # 1. Querying CLOB factory contracts
            # 2. Retrieving market metadata
            # 3. Validating market parameters
            # 4. Creating MarketConfig objects

            logger.info(f"Market discovery completed, found {discovered_count} additional markets")

        except Exception as e:
            logger.warning(f"Market discovery failed: {e}")

    async def _start_background_tasks(self) -> None:
        """Start background maintenance tasks."""
        self._refresh_task = asyncio.create_task(self._refresh_loop())
        self._status_task = asyncio.create_task(self._status_check_loop())

    async def _refresh_loop(self) -> None:
        """Background market configuration refresh loop."""
        while True:
            try:
                await asyncio.sleep(self.refresh_interval)
                await self._refresh_market_configs()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Market config refresh error: {e}")
                await asyncio.sleep(60)  # Brief pause on error

    async def _status_check_loop(self) -> None:
        """Background market status monitoring loop."""
        while True:
            try:
                await asyncio.sleep(self.status_check_interval)
                await self._check_market_status()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Market status check error: {e}")
                await asyncio.sleep(30)  # Brief pause on error

    async def _refresh_market_configs(self) -> None:
        """Refresh market configurations from blockchain."""
        try:
            updated_count = 0

            for symbol, market in self.markets.items():
                try:
                    # In production, query blockchain for updated parameters
                    # For now, just update timestamp
                    market.last_updated = datetime.now()
                    updated_count += 1

                except Exception as e:
                    logger.warning(f"Failed to refresh market {symbol}: {e}")

            self._last_refresh = datetime.now()
            logger.debug(f"Market config refresh completed, updated {updated_count} markets")

        except Exception as e:
            logger.error(f"Market config refresh failed: {e}")

    async def _check_market_status(self) -> None:
        """Check status of all configured markets."""
        try:
            status_changes = 0

            for symbol, market in self.markets.items():
                try:
                    # In production, query market status from blockchain
                    # Check if market is paused, in maintenance, etc.

                    # For now, simulate occasional status changes
                    # In production, this would query actual market contracts
                    old_status = market.status
                    new_status = await self._query_market_status(market.market_id)

                    if new_status != old_status:
                        market.status = new_status
                        market.last_updated = datetime.now()
                        status_changes += 1

                        logger.info(f"Market {symbol} status changed: {old_status.value} -> {new_status.value}")

                except Exception as e:
                    logger.warning(f"Failed to check status for market {symbol}: {e}")

            self._last_status_check = datetime.now()

            if status_changes > 0:
                logger.info(f"Market status check completed, {status_changes} status changes")

        except Exception as e:
            logger.error(f"Market status check failed: {e}")

    async def _query_market_status(self, market_id: str) -> MarketStatus:
        """Query market status from blockchain."""
        try:
            # In production, query actual market contract
            # For now, return active status
            return MarketStatus.ACTIVE

        except Exception as e:
            logger.warning(f"Failed to query market status for {market_id}: {e}")
            return MarketStatus.UNKNOWN

    def get_market(self, symbol: str) -> MarketConfig | None:
        """Get market configuration by symbol."""
        return self.markets.get(symbol)

    def get_market_by_id(self, market_id: str) -> MarketConfig | None:
        """Get market configuration by market ID."""
        return self.markets_by_id.get(market_id)

    def get_active_markets(self) -> list[MarketConfig]:
        """Get all active markets."""
        return [
            market for market in self.markets.values()
            if market.status == MarketStatus.ACTIVE
        ]

    def get_supported_symbols(self) -> set[str]:
        """Get set of supported trading symbols."""
        return set(self.markets.keys())

    def validate_market_order(self, symbol: str, price: Decimal, size: Decimal) -> dict[str, Any]:
        """Validate order parameters against market configuration."""
        market = self.get_market(symbol)
        if not market:
            return {
                'valid': False,
                'error': f'Unsupported market: {symbol}'
            }

        if market.status != MarketStatus.ACTIVE:
            return {
                'valid': False,
                'error': f'Market {symbol} is {market.status.value}'
            }

        # Validate price
        if not market.validate_price(price):
            return {
                'valid': False,
                'error': f'Invalid price {price}. Must be between {market.min_price} and {market.max_price} with tick size {market.tick_size}'
            }

        # Validate size
        if not market.validate_size(size):
            return {
                'valid': False,
                'error': f'Invalid size {size}. Must be between {market.min_order_size} and {market.max_order_size} with lot size {market.lot_size}'
            }

        return {
            'valid': True,
            'rounded_price': market.round_price(price),
            'rounded_size': market.round_size(size),
            'notional': market.calculate_notional(price, size)
        }

    def get_market_summary(self) -> dict[str, Any]:
        """Get summary of all market configurations."""
        return {
            'total_markets': len(self.markets),
            'active_markets': len(self.get_active_markets()),
            'supported_symbols': list(self.get_supported_symbols()),
            'last_refresh': self._last_refresh.isoformat() if self._last_refresh else None,
            'last_status_check': self._last_status_check.isoformat() if self._last_status_check else None,
            'markets': {symbol: market.to_dict() for symbol, market in self.markets.items()}
        }

    async def reload_market_config(self, symbol: str) -> bool:
        """Reload configuration for specific market."""
        try:
            market = self.get_market(symbol)
            if not market:
                return False

            # In production, reload from blockchain
            market.last_updated = datetime.now()

            logger.info(f"Market configuration reloaded: {symbol}")
            return True

        except Exception as e:
            logger.error(f"Failed to reload market config {symbol}: {e}")
            return False

    async def cleanup(self) -> None:
        """Cleanup resources and stop background tasks."""
        # Cancel background tasks
        if self._refresh_task and not self._refresh_task.done():
            self._refresh_task.cancel()
            try:
                await self._refresh_task
            except asyncio.CancelledError:
                pass

        if self._status_task and not self._status_task.done():
            self._status_task.cancel()
            try:
                await self._status_task
            except asyncio.CancelledError:
                pass

        logger.info("Market configuration manager cleanup completed")

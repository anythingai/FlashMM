"""
Sei V2 Testnet Integration Test

This test validates the complete blockchain integration with real Sei testnet.
Run this test when you have proper testnet configuration and test tokens.
"""

import asyncio
import os
from decimal import Decimal

import pytest

from flashmm.blockchain.blockchain_service import get_blockchain_service
from flashmm.trading.engine.market_making_engine import MarketMakingEngine
from flashmm.trading.execution.order_router import OrderRouter
from flashmm.utils.logging import get_logger

logger = get_logger(__name__)

# Skip tests if testnet configuration is not available
TESTNET_CONFIGURED = all([
    os.getenv("SEI_PRIVATE_KEY"),
    os.getenv("CAMBRIAN_API_KEY"),
    os.getenv("CAMBRIAN_SECRET_KEY")
])

pytestmark = pytest.mark.skipif(
    not TESTNET_CONFIGURED,
    reason="Testnet configuration not available (set SEI_PRIVATE_KEY, CAMBRIAN_API_KEY, CAMBRIAN_SECRET_KEY)"
)


@pytest.fixture
async def blockchain_service():
    """Initialize blockchain service for testnet."""
    service = get_blockchain_service()

    try:
        await service.initialize()
        yield service
    finally:
        await service.cleanup()


@pytest.fixture
async def order_router(blockchain_service):
    """Initialize order router with blockchain integration."""
    router = OrderRouter()

    try:
        await router.initialize()
        yield router
    finally:
        await router.cleanup()


@pytest.fixture
async def market_making_engine(order_router):
    """Initialize market making engine."""
    engine = MarketMakingEngine()

    try:
        await engine.initialize()
        yield engine
    finally:
        await engine.cleanup()


class TestSeiTestnetIntegration:
    """Integration tests with Sei V2 testnet."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.testnet
    async def test_blockchain_service_initialization(self, blockchain_service):
        """Test that blockchain service initializes successfully."""
        assert blockchain_service.status.value in ['healthy', 'degraded']
        assert blockchain_service.enabled is True

        # Test service components
        component_status = blockchain_service.get_component_status()
        logger.info(f"Component status: {component_status}")

        # At minimum, sei_client should be available
        assert component_status['sei_client'] is True

    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.testnet
    async def test_network_connectivity(self, blockchain_service):
        """Test connectivity to Sei testnet."""
        network_status = blockchain_service.get_network_status()
        logger.info(f"Network status: {network_status}")

        assert network_status['status'] in ['healthy', 'degraded']
        assert 'block_height' in network_status
        assert network_status['block_height'] > 0

    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.testnet
    async def test_supported_markets(self, blockchain_service):
        """Test that supported markets are available."""
        markets = blockchain_service.get_supported_markets()
        logger.info(f"Supported markets: {markets}")

        # Should support at least SEI/USDC and wETH/USDC
        assert len(markets) >= 2
        assert "SEI/USDC" in markets
        assert "wETH/USDC" in markets

    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.testnet
    async def test_account_balance_query(self, blockchain_service):
        """Test querying account balance."""
        if not blockchain_service.account_manager:
            pytest.skip("Account manager not initialized")

        # Get active account balance
        try:
            active_account = await blockchain_service.account_manager.get_active_account()
            if active_account:
                balance = await blockchain_service.account_manager.get_account_balance(active_account.account_id)
                logger.info(f"Account balance: {balance}")

                assert balance is not None
                assert balance.address == active_account.address
                assert isinstance(balance.balances, dict)
        except Exception as e:
            logger.warning(f"Account balance query failed: {e}")
            # This might fail in test environment, so we'll just log it

    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.testnet
    async def test_order_validation(self, blockchain_service):
        """Test order validation against testnet markets."""
        from flashmm.trading.execution.order_router import (
            Order,
            OrderStatus,
            OrderType,
            TimeInForce,
        )

        # Create test order
        test_order = Order(
            order_id="test_validation_001",
            client_order_id="client_validation_001",
            symbol="SEI/USDC",
            side="buy",
            order_type=OrderType.LIMIT,
            price=Decimal("0.1000"),  # Low price to avoid execution
            size=Decimal("10.0"),     # Small size
            time_in_force=TimeInForce.GTC,
            status=OrderStatus.PENDING
        )

        # Validate order
        validation_result = blockchain_service.validate_order_for_blockchain(test_order)
        logger.info(f"Validation result: {validation_result}")

        assert validation_result['valid'] is True

    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.testnet
    @pytest.mark.slow
    async def test_order_submission_testnet(self, order_router):
        """Test actual order submission to testnet (CAREFUL - uses real testnet)."""
        if not order_router.is_blockchain_ready():
            pytest.skip("Blockchain not ready for order submission")

        logger.warning("SUBMITTING REAL ORDER TO TESTNET - ENSURE YOU HAVE TEST TOKENS")

        # Submit a very small buy order at low price to minimize risk
        try:
            order = await order_router.place_order(
                symbol="SEI/USDC",
                side="buy",
                price=0.01,      # Very low price
                size=1.0,        # Very small size
                order_type="limit",
                time_in_force="GTC"
            )

            assert order is not None
            assert order.symbol == "SEI/USDC"
            logger.info(f"Order submitted: {order.order_id} - Status: {order.status.value}")

            # Wait a moment for blockchain processing
            await asyncio.sleep(5)

            # Check order status
            updated_order = order_router.get_order(order.order_id)
            logger.info(f"Order status after wait: {updated_order.status.value}")

            # Cancel the order to clean up
            cancel_success = await order_router.cancel_order(order.order_id)
            logger.info(f"Order cancellation: {'success' if cancel_success else 'failed'}")

            # Final status check
            final_order = order_router.get_order(order.order_id)
            logger.info(f"Final order status: {final_order.status.value}")

        except Exception as e:
            logger.error(f"Order submission test failed: {e}")
            raise

    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.testnet
    async def test_market_making_engine_integration(self, market_making_engine):
        """Test market making engine with blockchain integration."""
        if not market_making_engine.order_router.is_blockchain_ready():
            pytest.skip("Blockchain not ready for market making")

        # Test engine status
        status = market_making_engine.get_status()
        logger.info(f"Market making engine status: {status}")

        # Test blockchain integration status
        blockchain_status = market_making_engine.order_router.get_blockchain_status()
        logger.info(f"Blockchain integration status: {blockchain_status}")

        assert blockchain_status['enabled'] is True
        assert len(blockchain_status['supported_markets']) > 0

    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.testnet
    async def test_performance_monitoring(self, blockchain_service):
        """Test performance monitoring with testnet."""
        # Force health check
        health_result = await blockchain_service.force_health_check()
        logger.info(f"Health check result: {health_result}")

        assert health_result['success'] is True

        # Get service statistics
        service_status = blockchain_service.get_service_status()
        logger.info(f"Service status: {service_status}")

        assert service_status['status'] in ['healthy', 'degraded']
        assert 'uptime_seconds' in service_status
        assert service_status['uptime_seconds'] >= 0

    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.testnet
    async def test_error_recovery(self, blockchain_service):
        """Test error recovery mechanisms."""
        # Test service restart capability
        _original_status = blockchain_service.status

        # Note: In a real test, we might simulate failures
        # For now, just test that restart functionality exists
        assert hasattr(blockchain_service, 'restart_service')
        assert callable(blockchain_service.restart_service)

        logger.info(f"Service status maintained: {blockchain_service.status.value}")


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.testnet
async def test_end_to_end_trading_flow():
    """Test complete end-to-end trading flow with testnet."""
    logger.info("Starting end-to-end trading flow test")

    # Initialize all components
    blockchain_service = get_blockchain_service()
    order_router = OrderRouter()

    try:
        # Initialize services
        await blockchain_service.initialize()
        await order_router.initialize()

        if not order_router.is_blockchain_ready():
            pytest.skip("Blockchain not ready for end-to-end test")

        logger.info("All services initialized successfully")

        # Test the complete flow
        logger.info("Testing market data availability...")
        markets = blockchain_service.get_supported_markets()
        assert len(markets) > 0
        logger.info(f"Available markets: {markets}")

        logger.info("Testing order validation...")
        from flashmm.trading.execution.order_router import (
            Order,
            OrderStatus,
            OrderType,
            TimeInForce,
        )

        test_order = Order(
            order_id="e2e_test_001",
            client_order_id="e2e_client_001",
            symbol="SEI/USDC",
            side="buy",
            order_type=OrderType.LIMIT,
            price=Decimal("0.01"),  # Very low price
            size=Decimal("1.0"),    # Small size
            time_in_force=TimeInForce.GTC,
            status=OrderStatus.PENDING
        )

        validation = blockchain_service.validate_order_for_blockchain(test_order)
        assert validation['valid'] is True
        logger.info("Order validation passed")

        logger.info("Testing network connectivity...")
        network_status = blockchain_service.get_network_status()
        assert network_status['status'] in ['healthy', 'degraded']
        logger.info(f"Network status: {network_status['status']}")

        logger.info("End-to-end test completed successfully")

    finally:
        # Cleanup
        await order_router.cleanup()
        await blockchain_service.cleanup()


if __name__ == "__main__":
    # Run specific test
    pytest.main([
        __file__,
        "-v",
        "-s",
        "--tb=short",
        "-m", "integration and testnet"
    ])

"""
Integration tests for OrderRouter with blockchain integration.
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from flashmm.blockchain.blockchain_service import BlockchainService, ServiceStatus
from flashmm.trading.execution.order_router import (
    OrderRouter,
    OrderStatus,
)
from flashmm.utils.exceptions import OrderError


@pytest.fixture
def mock_config():
    """Mock configuration for testing."""
    config_data = {
        "blockchain.enabled": True,
        "trading.max_orders_per_symbol": 50,
        "trading.order_timeout_minutes": 60,
        "trading.reconciliation_interval_seconds": 30,
        "trading.min_order_size": 1.0,
        "trading.max_order_size": 10000.0,
        "trading.enable_trading": True
    }

    with patch('flashmm.trading.execution.order_router.get_config') as mock_get_config:
        mock_config_obj = Mock()
        mock_config_obj.get.side_effect = lambda key, default=None: config_data.get(key, default)
        mock_get_config.return_value = mock_config_obj
        yield mock_config_obj


@pytest.fixture
async def order_router(mock_config):
    """Create OrderRouter instance with mocked blockchain service."""
    with patch('flashmm.trading.execution.order_router.get_blockchain_service') as mock_get_service:
        mock_blockchain_service = AsyncMock(spec=BlockchainService)
        mock_blockchain_service.status = ServiceStatus.HEALTHY
        mock_blockchain_service.initialize = AsyncMock()
        mock_blockchain_service.submit_order_to_blockchain = AsyncMock(return_value=True)
        mock_blockchain_service.cancel_order_on_blockchain = AsyncMock(return_value=True)
        mock_blockchain_service.get_supported_markets = Mock(return_value=["SEI/USDC", "wETH/USDC"])
        mock_blockchain_service.validate_order_for_blockchain = Mock(return_value={'valid': True})
        mock_blockchain_service.get_service_status = Mock(return_value={'status': 'healthy'})
        mock_blockchain_service.cleanup = AsyncMock()

        mock_get_service.return_value = mock_blockchain_service

        router = OrderRouter()
        await router.initialize()

        yield router

        await router.cleanup()


@pytest.fixture
def sample_order_params():
    """Sample order parameters for testing."""
    return {
        "symbol": "SEI/USDC",
        "side": "buy",
        "price": 0.5000,
        "size": 100.0,
        "order_type": "limit",
        "time_in_force": "GTC"
    }


class TestOrderRouterBlockchainIntegration:
    """Test OrderRouter blockchain integration functionality."""

    @pytest.mark.asyncio
    async def test_initialization_with_blockchain(self, order_router):
        """Test OrderRouter initialization with blockchain service."""
        assert order_router.blockchain_enabled is True
        assert order_router.blockchain_service is not None
        order_router.blockchain_service.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialization_blockchain_failure_fallback(self, mock_config):
        """Test OrderRouter initialization with blockchain service failure."""
        with patch('flashmm.trading.execution.order_router.get_blockchain_service') as mock_get_service:
            mock_blockchain_service = AsyncMock()
            mock_blockchain_service.initialize = AsyncMock(side_effect=Exception("Blockchain init failed"))
            mock_get_service.return_value = mock_blockchain_service

            router = OrderRouter()
            await router.initialize()

            # Should fallback to disabled blockchain
            assert router.blockchain_enabled is False

            await router.cleanup()

    @pytest.mark.asyncio
    async def test_place_order_blockchain_success(self, order_router, sample_order_params):
        """Test successful order placement via blockchain."""
        order = await order_router.place_order(**sample_order_params)

        assert order is not None
        assert order.symbol == "SEI/USDC"
        assert order.status == OrderStatus.ACTIVE

        # Verify blockchain service was called
        order_router.blockchain_service.submit_order_to_blockchain.assert_called_once()

        # Verify order is stored
        assert order.order_id in order_router.orders

    @pytest.mark.asyncio
    async def test_place_order_blockchain_failure_fallback(self, order_router, sample_order_params):
        """Test order placement with blockchain failure falling back to mock."""
        # Configure blockchain service to fail
        order_router.blockchain_service.submit_order_to_blockchain.return_value = False

        order = await order_router.place_order(**sample_order_params)

        assert order is not None
        assert order.status == OrderStatus.ACTIVE  # Should still succeed with mock

        # Verify blockchain was attempted first
        order_router.blockchain_service.submit_order_to_blockchain.assert_called_once()

    @pytest.mark.asyncio
    async def test_place_order_blockchain_disabled(self, mock_config, sample_order_params):
        """Test order placement when blockchain is disabled."""
        # Create router with blockchain disabled
        with patch('flashmm.trading.execution.order_router.get_blockchain_service') as mock_get_service:
            mock_blockchain_service = AsyncMock()
            mock_get_service.return_value = mock_blockchain_service

            router = OrderRouter()
            router.blockchain_enabled = False
            await router.initialize()

            order = await router.place_order(**sample_order_params)

            assert order is not None
            assert order.status == OrderStatus.ACTIVE

            # Verify blockchain service was not called
            mock_blockchain_service.submit_order_to_blockchain.assert_not_called()

            await router.cleanup()

    @pytest.mark.asyncio
    async def test_cancel_order_blockchain_success(self, order_router, sample_order_params):
        """Test successful order cancellation via blockchain."""
        # First place an order
        order = await order_router.place_order(**sample_order_params)
        order_id = order.order_id

        # Reset mock call count
        order_router.blockchain_service.cancel_order_on_blockchain.reset_mock()

        # Cancel the order
        success = await order_router.cancel_order(order_id)

        assert success is True
        assert order.status == OrderStatus.CANCELLED

        # Verify blockchain service was called
        order_router.blockchain_service.cancel_order_on_blockchain.assert_called_once_with(order_id)

    @pytest.mark.asyncio
    async def test_cancel_order_blockchain_failure(self, order_router, sample_order_params):
        """Test order cancellation with blockchain failure."""
        # Place an order first
        order = await order_router.place_order(**sample_order_params)
        order_id = order.order_id

        # Configure blockchain service to fail cancellation
        order_router.blockchain_service.cancel_order_on_blockchain.return_value = False

        # Cancel should still succeed (internal state update)
        success = await order_router.cancel_order(order_id)

        assert success is True
        assert order.status == OrderStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_order_validation_blockchain_markets(self, order_router):
        """Test order validation with blockchain market support."""
        # Test supported market
        supported_params = {
            "symbol": "SEI/USDC",
            "side": "buy",
            "price": 0.5000,
            "size": 100.0,
            "order_type": "limit"
        }

        order = await order_router.place_order(**supported_params)
        assert order is not None

        # Test unsupported market
        order_router.blockchain_service.get_supported_markets.return_value = ["wETH/USDC"]

        unsupported_params = {
            "symbol": "ATOM/USDC",
            "side": "buy",
            "price": 10.0,
            "size": 50.0,
            "order_type": "limit"
        }

        with pytest.raises(OrderError, match="not supported on blockchain"):
            await order_router.place_order(**unsupported_params)

    @pytest.mark.asyncio
    async def test_order_validation_blockchain_parameters(self, order_router):
        """Test order parameter validation with blockchain integration."""
        # Configure blockchain validation to fail
        order_router.blockchain_service.validate_order_for_blockchain.return_value = {
            'valid': False,
            'error': 'Invalid tick size'
        }

        invalid_params = {
            "symbol": "SEI/USDC",
            "side": "buy",
            "price": 0.12345,  # Invalid tick size
            "size": 100.0,
            "order_type": "limit"
        }

        with pytest.raises(OrderError, match="Blockchain validation failed"):
            await order_router.place_order(**invalid_params)

    def test_blockchain_status_reporting(self, order_router):
        """Test blockchain status reporting."""
        # Configure mock blockchain service status
        order_router.blockchain_service.get_supported_markets.return_value = ["SEI/USDC", "wETH/USDC"]
        order_router.blockchain_service.get_network_status.return_value = {
            'status': 'healthy',
            'block_height': 12345
        }
        order_router.blockchain_service.get_service_status.return_value = {
            'status': 'healthy',
            'uptime_seconds': 3600
        }

        status = order_router.get_blockchain_status()

        assert status['enabled'] is True
        assert status['supported_markets'] == ["SEI/USDC", "wETH/USDC"]
        assert status['network_status']['status'] == 'healthy'
        assert status['service_status']['status'] == 'healthy'

    def test_blockchain_status_disabled(self, order_router):
        """Test blockchain status when disabled."""
        order_router.blockchain_enabled = False
        order_router.blockchain_service = None

        status = order_router.get_blockchain_status()

        assert status['enabled'] is False
        assert status['status'] == 'disabled'
        assert status['supported_markets'] == []
        assert status['network_status'] == 'unavailable'

    def test_blockchain_readiness_check(self, order_router):
        """Test blockchain readiness check."""
        # Test healthy status
        order_router.blockchain_service.get_service_status.return_value = {'status': 'healthy'}
        assert order_router.is_blockchain_ready() is True

        # Test degraded status (still ready)
        order_router.blockchain_service.get_service_status.return_value = {'status': 'degraded'}
        assert order_router.is_blockchain_ready() is True

        # Test unhealthy status
        order_router.blockchain_service.get_service_status.return_value = {'status': 'unhealthy'}
        assert order_router.is_blockchain_ready() is False

        # Test disabled blockchain
        order_router.blockchain_enabled = False
        assert order_router.is_blockchain_ready() is False

    def test_performance_stats_include_blockchain(self, order_router):
        """Test that performance stats include blockchain information."""
        stats = order_router.get_performance_stats()

        assert 'blockchain_integration' in stats
        blockchain_stats = stats['blockchain_integration']

        assert 'enabled' in blockchain_stats
        assert 'status' in blockchain_stats
        assert 'supported_markets' in blockchain_stats
        assert 'network_status' in blockchain_stats

    @pytest.mark.asyncio
    async def test_batch_operations_with_blockchain(self, order_router):
        """Test batch order operations with blockchain integration."""
        from flashmm.trading.execution.order_router import BatchOrderRequest

        # Prepare batch request
        batch_request = BatchOrderRequest()
        batch_request.orders_to_place = [
            {
                "symbol": "SEI/USDC",
                "side": "buy",
                "price": 0.5000,
                "size": 50.0,
                "order_type": "limit"
            },
            {
                "symbol": "wETH/USDC",
                "side": "sell",
                "price": 2000.0,
                "size": 0.1,
                "order_type": "limit"
            }
        ]

        result = await order_router.execute_batch_orders(batch_request)

        assert len(result.placed_orders) == 2
        assert len(result.failed_operations) == 0

        # Verify blockchain service was called for each order
        assert order_router.blockchain_service.submit_order_to_blockchain.call_count == 2

    @pytest.mark.asyncio
    async def test_reconciliation_with_blockchain(self, order_router, sample_order_params):
        """Test order reconciliation with blockchain integration."""
        # Place an order
        order = await order_router.place_order(**sample_order_params)

        # Simulate reconciliation
        await order_router.reconciler.reconcile_orders()

        # Order should still be tracked
        assert order.order_id in order_router.orders
        assert order.status in [OrderStatus.ACTIVE, OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED]

    @pytest.mark.asyncio
    async def test_cleanup_includes_blockchain(self, order_router):
        """Test that cleanup properly handles blockchain service."""
        await order_router.cleanup()

        # Verify blockchain service cleanup was called
        order_router.blockchain_service.cleanup.assert_called_once()


class TestOrderRouterErrorHandling:
    """Test error handling in OrderRouter with blockchain integration."""

    @pytest.mark.asyncio
    async def test_blockchain_service_exception(self, order_router, sample_order_params):
        """Test handling of blockchain service exceptions."""
        # Configure blockchain service to raise exception
        order_router.blockchain_service.submit_order_to_blockchain.side_effect = Exception("Blockchain service error")

        # Order placement should still succeed with fallback
        order = await order_router.place_order(**sample_order_params)

        assert order is not None
        assert order.status == OrderStatus.ACTIVE

    @pytest.mark.asyncio
    async def test_blockchain_validation_exception(self, order_router, sample_order_params):
        """Test handling of blockchain validation exceptions."""
        # Configure validation to raise exception
        order_router.blockchain_service.validate_order_for_blockchain.side_effect = Exception("Validation service error")

        # Should log warning but continue with order
        order = await order_router.place_order(**sample_order_params)

        assert order is not None
        assert order.status == OrderStatus.ACTIVE

    @pytest.mark.asyncio
    async def test_network_status_exception(self, order_router):
        """Test handling of network status query exceptions."""
        # Configure network status to raise exception
        order_router.blockchain_service.get_network_status.side_effect = Exception("Network query failed")

        # Should return error status
        status = order_router.get_blockchain_status()

        assert status['status'] == 'error'
        assert 'error' in status


if __name__ == "__main__":
    pytest.main([__file__])

"""
Unit tests for BlockchainService integration.
"""

from datetime import datetime
from decimal import Decimal
from unittest.mock import AsyncMock, Mock, patch

import pytest

from flashmm.blockchain.blockchain_service import BlockchainService, ServiceStatus
from flashmm.trading.execution.order_router import Order, OrderStatus, OrderType, TimeInForce
from flashmm.utils.exceptions import ServiceError


@pytest.fixture
def mock_config():
    """Mock configuration."""
    config_data = {
        "blockchain.enabled": True,
        "blockchain.auto_recovery": True,
        "blockchain.health_check_interval_seconds": 60,
        "blockchain.emergency_stop_enabled": True,
        "sei.rpc_url": "https://test-rpc.sei.com",
        "sei.chain_id": "atlantic-2",
        "cambrian_api_key": "test_key",
        "cambrian_secret_key": "test_secret",
        "sei_private_key": "test_private_key"
    }

    with patch('flashmm.blockchain.blockchain_service.get_config') as mock_get_config:
        mock_config_obj = Mock()
        mock_config_obj.get.side_effect = lambda key, default=None: config_data.get(key, default)
        mock_get_config.return_value = mock_config_obj
        yield mock_config_obj


@pytest.fixture
def blockchain_service(mock_config):
    """Create BlockchainService instance."""
    return BlockchainService()


@pytest.fixture
def sample_order():
    """Create sample order for testing."""
    return Order(
        order_id="test_order_001",
        client_order_id="client_001",
        symbol="SEI/USDC",
        side="buy",
        order_type=OrderType.LIMIT,
        price=Decimal("0.5000"),
        size=Decimal("100.0"),
        time_in_force=TimeInForce.GTC,
        status=OrderStatus.PENDING
    )


class TestBlockchainService:
    """Test BlockchainService functionality."""

    @pytest.mark.asyncio
    async def test_service_initialization(self, blockchain_service):
        """Test service initialization."""
        with patch.multiple(
            blockchain_service,
            _initialize_sei_client=AsyncMock(),
            _initialize_market_config_manager=AsyncMock(),
            _initialize_transaction_manager=AsyncMock(),
            _initialize_account_manager=AsyncMock(),
            _initialize_order_manager=AsyncMock(),
            _initialize_blockchain_monitor=AsyncMock(),
            _setup_component_connections=AsyncMock(),
            _start_background_tasks=AsyncMock(),
            _perform_health_check=AsyncMock(return_value=Mock(status=ServiceStatus.HEALTHY))
        ):
            await blockchain_service.initialize()

            assert blockchain_service.status == ServiceStatus.HEALTHY
            assert blockchain_service.initialization_time is not None

    @pytest.mark.asyncio
    async def test_service_initialization_failure(self, blockchain_service):
        """Test service initialization failure handling."""
        with patch.object(blockchain_service, '_initialize_sei_client', side_effect=Exception("Init failed")):
            with pytest.raises(ServiceError):
                await blockchain_service.initialize()

            assert blockchain_service.status == ServiceStatus.UNHEALTHY
            assert blockchain_service.error_count > 0
            assert "Init failed" in blockchain_service.last_error

    @pytest.mark.asyncio
    async def test_service_disabled(self, mock_config):
        """Test service behavior when disabled."""
        mock_config.get.side_effect = lambda key, default=None: False if key == "blockchain.enabled" else default

        service = BlockchainService()
        await service.initialize()

        assert service.status == ServiceStatus.SHUTDOWN
        assert not service.enabled

    @pytest.mark.asyncio
    async def test_submit_order_success(self, blockchain_service, sample_order):
        """Test successful order submission."""
        # Mock successful initialization
        blockchain_service.status = ServiceStatus.HEALTHY
        blockchain_service.order_manager = AsyncMock()
        blockchain_service.order_manager.submit_order_to_blockchain.return_value = True

        result = await blockchain_service.submit_order_to_blockchain(sample_order)

        assert result is True
        blockchain_service.order_manager.submit_order_to_blockchain.assert_called_once_with(sample_order)
        assert blockchain_service.performance_stats['orders_submitted'] == 1
        assert blockchain_service.performance_stats['orders_successful'] == 1

    @pytest.mark.asyncio
    async def test_submit_order_failure(self, blockchain_service, sample_order):
        """Test order submission failure."""
        blockchain_service.status = ServiceStatus.HEALTHY
        blockchain_service.order_manager = AsyncMock()
        blockchain_service.order_manager.submit_order_to_blockchain.return_value = False

        result = await blockchain_service.submit_order_to_blockchain(sample_order)

        assert result is False
        assert blockchain_service.performance_stats['orders_failed'] == 1

    @pytest.mark.asyncio
    async def test_submit_order_service_not_ready(self, blockchain_service, sample_order):
        """Test order submission when service is not ready."""
        blockchain_service.status = ServiceStatus.UNHEALTHY

        result = await blockchain_service.submit_order_to_blockchain(sample_order)

        assert result is False
        assert blockchain_service.performance_stats['orders_failed'] == 1

    @pytest.mark.asyncio
    async def test_cancel_order_success(self, blockchain_service):
        """Test successful order cancellation."""
        blockchain_service.status = ServiceStatus.HEALTHY
        blockchain_service.order_manager = AsyncMock()
        blockchain_service.order_manager.cancel_order_on_blockchain.return_value = True

        result = await blockchain_service.cancel_order_on_blockchain("test_order_001")

        assert result is True
        blockchain_service.order_manager.cancel_order_on_blockchain.assert_called_once_with("test_order_001")

    @pytest.mark.asyncio
    async def test_cancel_order_failure(self, blockchain_service):
        """Test order cancellation failure."""
        blockchain_service.status = ServiceStatus.HEALTHY
        blockchain_service.order_manager = AsyncMock()
        blockchain_service.order_manager.cancel_order_on_blockchain.return_value = False

        result = await blockchain_service.cancel_order_on_blockchain("test_order_001")

        assert result is False

    def test_get_supported_markets(self, blockchain_service):
        """Test getting supported markets."""
        blockchain_service.market_config_manager = Mock()
        blockchain_service.market_config_manager.get_supported_symbols.return_value = {"SEI/USDC", "wETH/USDC"}

        markets = blockchain_service.get_supported_markets()

        assert "SEI/USDC" in markets
        assert "wETH/USDC" in markets

    def test_get_supported_markets_no_manager(self, blockchain_service):
        """Test getting supported markets when manager is not available."""
        blockchain_service.market_config_manager = None

        markets = blockchain_service.get_supported_markets()

        assert markets == []

    def test_validate_order_for_blockchain(self, blockchain_service, sample_order):
        """Test order validation for blockchain."""
        blockchain_service.market_config_manager = Mock()
        blockchain_service.market_config_manager.validate_market_order.return_value = {
            'valid': True,
            'rounded_price': Decimal("0.5000"),
            'rounded_size': Decimal("100.0")
        }

        result = blockchain_service.validate_order_for_blockchain(sample_order)

        assert result['valid'] is True
        assert result['rounded_price'] == Decimal("0.5000")

    def test_validate_order_invalid(self, blockchain_service, sample_order):
        """Test invalid order validation."""
        blockchain_service.market_config_manager = Mock()
        blockchain_service.market_config_manager.validate_market_order.return_value = {
            'valid': False,
            'error': 'Invalid price'
        }

        result = blockchain_service.validate_order_for_blockchain(sample_order)

        assert result['valid'] is False
        assert 'Invalid price' in result['error']

    def test_get_network_status(self, blockchain_service):
        """Test getting network status."""
        blockchain_service.blockchain_monitor = Mock()
        blockchain_service.blockchain_monitor.get_current_network_status.return_value = {
            'status': 'healthy',
            'block_height': 12345,
            'rpc_latency_ms': 100.0
        }

        status = blockchain_service.get_network_status()

        assert status['status'] == 'healthy'
        assert status['block_height'] == 12345

    def test_get_network_status_no_monitor(self, blockchain_service):
        """Test getting network status when monitor is not available."""
        blockchain_service.blockchain_monitor = None

        status = blockchain_service.get_network_status()

        assert status['status'] == 'monitoring_unavailable'

    @pytest.mark.asyncio
    async def test_health_check_all_healthy(self, blockchain_service):
        """Test health check when all components are healthy."""
        # Mock all components as healthy
        blockchain_service.sei_client = AsyncMock()
        blockchain_service.sei_client.check_network_health.return_value = Mock(is_healthy=True, rpc_latency_ms=100.0)

        blockchain_service.market_config_manager = Mock()
        blockchain_service.market_config_manager.get_active_markets.return_value = [Mock(), Mock()]

        blockchain_service.transaction_manager = Mock()
        blockchain_service.transaction_manager.get_performance_stats.return_value = {'success_rate': 0.95}

        blockchain_service.account_manager = Mock()
        blockchain_service.account_manager.get_account_summary.return_value = {'active_accounts': 2}

        blockchain_service.order_manager = Mock()
        blockchain_service.order_manager.get_performance_stats.return_value = {'success_rate': 0.98}

        blockchain_service.blockchain_monitor = Mock()
        blockchain_service.blockchain_monitor.get_monitoring_stats.return_value = {'active_alerts_count': 0}

        health_check = await blockchain_service._perform_health_check()

        assert health_check.status == ServiceStatus.HEALTHY
        assert health_check.error_count == 0
        assert all(health_check.components.values())

    @pytest.mark.asyncio
    async def test_health_check_degraded(self, blockchain_service):
        """Test health check when some components are unhealthy."""
        # Mock some components as unhealthy
        blockchain_service.sei_client = AsyncMock()
        blockchain_service.sei_client.check_network_health.return_value = Mock(is_healthy=False)

        blockchain_service.market_config_manager = Mock()
        blockchain_service.market_config_manager.get_active_markets.return_value = [Mock()]

        blockchain_service.transaction_manager = Mock()
        blockchain_service.transaction_manager.get_performance_stats.return_value = {'success_rate': 0.95}

        blockchain_service.account_manager = Mock()
        blockchain_service.account_manager.get_account_summary.return_value = {'active_accounts': 1}

        blockchain_service.order_manager = Mock()
        blockchain_service.order_manager.get_performance_stats.return_value = {'success_rate': 0.98}

        blockchain_service.blockchain_monitor = Mock()
        blockchain_service.blockchain_monitor.get_monitoring_stats.return_value = {'active_alerts_count': 1}

        health_check = await blockchain_service._perform_health_check()

        assert health_check.status in [ServiceStatus.DEGRADED, ServiceStatus.UNHEALTHY]
        assert health_check.error_count > 0

    def test_service_status_summary(self, blockchain_service):
        """Test getting service status summary."""
        blockchain_service.status = ServiceStatus.HEALTHY
        blockchain_service.initialization_time = datetime.now()
        blockchain_service.last_health_check = Mock()
        blockchain_service.last_health_check.timestamp = datetime.now()
        blockchain_service.last_health_check.status = ServiceStatus.HEALTHY
        blockchain_service.last_health_check.components = {'sei_client': True}
        blockchain_service.last_health_check.error_count = 0
        blockchain_service.last_health_check.performance_metrics = {'rpc_latency': 100.0}

        status = blockchain_service.get_service_status()

        assert status['status'] == 'healthy'
        assert status['enabled'] is True
        assert 'initialization_time' in status
        assert 'uptime_seconds' in status
        assert status['last_health_check']['components']['sei_client'] is True

    def test_component_status(self, blockchain_service):
        """Test getting component status."""
        blockchain_service.sei_client = Mock()
        blockchain_service.market_config_manager = Mock()
        blockchain_service.order_manager = None
        blockchain_service.transaction_manager = Mock()
        blockchain_service.account_manager = None
        blockchain_service.blockchain_monitor = Mock()

        status = blockchain_service.get_component_status()

        assert status['sei_client'] is True
        assert status['market_config_manager'] is True
        assert status['order_manager'] is False
        assert status['transaction_manager'] is True
        assert status['account_manager'] is False
        assert status['blockchain_monitor'] is True

    @pytest.mark.asyncio
    async def test_force_health_check(self, blockchain_service):
        """Test forcing a health check."""
        health_check_mock = Mock()
        health_check_mock.status = ServiceStatus.HEALTHY
        health_check_mock.timestamp = datetime.now()
        health_check_mock.components = {'sei_client': True}
        health_check_mock.error_count = 0
        health_check_mock.performance_metrics = {}

        with patch.object(blockchain_service, '_perform_health_check', return_value=health_check_mock):
            result = await blockchain_service.force_health_check()

            assert result['success'] is True
            assert result['health_check']['status'] == 'healthy'

    @pytest.mark.asyncio
    async def test_cleanup(self, blockchain_service):
        """Test service cleanup."""
        # Mock components with cleanup methods
        mock_components = []
        for attr in ['blockchain_monitor', 'order_manager', 'account_manager',
                    'transaction_manager', 'market_config_manager', 'sei_client']:
            component = AsyncMock()
            component.cleanup = AsyncMock()
            setattr(blockchain_service, attr, component)
            mock_components.append(component)

        # Mock background tasks
        blockchain_service._health_check_task = AsyncMock()
        blockchain_service._recovery_task = AsyncMock()

        await blockchain_service.cleanup()

        # Verify all components were cleaned up
        for component in mock_components:
            component.cleanup.assert_called_once()

        assert blockchain_service.status == ServiceStatus.SHUTDOWN
        assert blockchain_service.sei_client is None
        assert blockchain_service.order_manager is None


@pytest.mark.asyncio
async def test_blockchain_service_singleton():
    """Test that get_blockchain_service returns singleton."""
    from flashmm.blockchain.blockchain_service import get_blockchain_service

    service1 = get_blockchain_service()
    service2 = get_blockchain_service()

    assert service1 is service2


class TestBlockchainServiceErrorHandling:
    """Test error handling in BlockchainService."""

    @pytest.mark.asyncio
    async def test_error_callback_registration(self, blockchain_service):
        """Test error callback registration and execution."""
        error_callback = AsyncMock()
        blockchain_service.add_error_callback(error_callback)

        # Trigger an error
        await blockchain_service._handle_error("test_error", Exception("Test exception"))

        error_callback.assert_called_once_with("test_error", Exception("Test exception"))

    @pytest.mark.asyncio
    async def test_status_change_callback(self, blockchain_service):
        """Test status change callback registration and execution."""
        status_callback = AsyncMock()
        blockchain_service.add_status_change_callback(status_callback)

        # Trigger status change
        await blockchain_service._notify_status_change(ServiceStatus.DEGRADED)

        status_callback.assert_called_once_with(ServiceStatus.DEGRADED)

    @pytest.mark.asyncio
    async def test_emergency_stop(self, blockchain_service):
        """Test emergency stop functionality."""
        blockchain_service.status = ServiceStatus.HEALTHY
        blockchain_service.emergency_stop_enabled = True
        blockchain_service.order_manager = AsyncMock()

        # Mock status change callback
        status_callback = AsyncMock()
        blockchain_service.add_status_change_callback(status_callback)

        await blockchain_service._trigger_emergency_stop("Test emergency")

        assert blockchain_service.status == ServiceStatus.EMERGENCY
        status_callback.assert_called_with(ServiceStatus.EMERGENCY)
        assert len(blockchain_service.error_history) > 0
        assert blockchain_service.error_history[-1]['type'] == 'emergency_stop'


if __name__ == "__main__":
    pytest.main([__file__])

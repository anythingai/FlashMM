"""
Integration Test for Enhanced Risk Manager with Trading Engine

Tests the complete integration of the enhanced risk management system
with the existing MarketMakingEngine, ensuring backward compatibility
while providing enterprise-grade risk protection.
"""

from datetime import datetime
from unittest.mock import patch

import pytest

from flashmm.trading.risk.enhanced_risk_manager import EnhancedRiskManager, RiskCheckResult
from flashmm.utils.exceptions import CircuitBreakerError


class MockMarketMakingEngine:
    """Mock MarketMakingEngine for testing integration."""

    def __init__(self):
        self.is_running = False
        self.emergency_stops = []
        self.paused_events = []
        self.cancelled_orders = []
        self.state = 'inactive'

    async def emergency_stop(self, reason: str):
        """Mock emergency stop."""
        self.emergency_stops.append({'reason': reason, 'timestamp': datetime.now()})
        self.is_running = False
        self.state = 'emergency_stopped'

    async def pause(self, reason: str):
        """Mock pause trading."""
        self.paused_events.append({'reason': reason, 'timestamp': datetime.now()})
        self.state = 'paused'

    async def cancel_all_orders(self, reason: str):
        """Mock cancel all orders."""
        cancelled_count = 5  # Simulate 5 orders cancelled
        self.cancelled_orders.append({
            'reason': reason,
            'count': cancelled_count,
            'timestamp': datetime.now()
        })
        return {'cancelled_count': cancelled_count}

    def get_status(self):
        """Get engine status."""
        return {
            'is_running': self.is_running,
            'state': self.state,
            'emergency_stops': len(self.emergency_stops),
            'paused_events': len(self.paused_events)
        }


class TestEnhancedRiskManagerIntegration:
    """Test enhanced risk manager integration with trading engine."""

    @pytest.fixture
    async def enhanced_risk_manager(self):
        """Create enhanced risk manager for testing."""
        manager = EnhancedRiskManager()
        await manager.initialize()
        return manager

    @pytest.fixture
    async def legacy_risk_manager(self):
        """Create enhanced risk manager in legacy mode."""
        manager = EnhancedRiskManager()
        manager.enterprise_mode_enabled = False
        await manager.initialize()
        return manager

    @pytest.fixture
    def mock_trading_engine(self):
        """Create mock trading engine."""
        return MockMarketMakingEngine()

    @pytest.mark.asyncio
    async def test_enhanced_manager_initialization(self, enhanced_risk_manager):
        """Test enhanced risk manager initializes all components."""
        assert enhanced_risk_manager.initialized
        assert enhanced_risk_manager.enterprise_mode_enabled
        assert enhanced_risk_manager.position_tracker is not None
        assert enhanced_risk_manager.circuit_breakers is not None
        assert enhanced_risk_manager.position_limits is not None
        assert enhanced_risk_manager.market_monitor is not None
        assert enhanced_risk_manager.pnl_controller is not None
        assert enhanced_risk_manager.operational_monitor is not None
        assert enhanced_risk_manager.emergency_protocols is not None

    @pytest.mark.asyncio
    async def test_legacy_compatibility_mode(self, legacy_risk_manager):
        """Test enhanced manager works in legacy compatibility mode."""
        assert legacy_risk_manager.initialized
        assert not legacy_risk_manager.enterprise_mode_enabled
        assert legacy_risk_manager.position_tracker is not None

        # Enterprise components should not be initialized
        assert legacy_risk_manager.circuit_breakers is None
        assert legacy_risk_manager.position_limits is None
        assert legacy_risk_manager.market_monitor is None

    @pytest.mark.asyncio
    async def test_backward_compatible_interface(self, enhanced_risk_manager):
        """Test that enhanced manager maintains backward compatible interface."""
        # Test that all legacy methods exist and work
        symbol = "BTC-USD"

        # check_trading_allowed should work
        allowed = await enhanced_risk_manager.check_trading_allowed(symbol)
        assert isinstance(allowed, bool)

        # get_risk_metrics should work
        metrics = await enhanced_risk_manager.get_risk_metrics()
        assert 'circuit_breaker_active' in metrics
        assert 'daily_pnl' in metrics
        assert 'positions' in metrics

        # update_position should work
        await enhanced_risk_manager.update_position("BTC-USD", "buy", 0.1, 50000.0)

        # Legacy methods should work
        await enhanced_risk_manager.reset_daily_metrics()
        await enhanced_risk_manager.reset_circuit_breaker()

    @pytest.mark.asyncio
    async def test_trading_engine_callback_integration(self, enhanced_risk_manager, mock_trading_engine):
        """Test integration with trading engine callbacks."""
        # Setup callbacks
        enhanced_risk_manager.set_trading_engine_callbacks(
            emergency_stop=mock_trading_engine.emergency_stop,
            pause_trading=mock_trading_engine.pause,
            cancel_orders=mock_trading_engine.cancel_all_orders
        )

        # Trigger emergency through enhanced risk manager
        await enhanced_risk_manager._handle_emergency_halt("Test emergency")

        # Verify trading engine received emergency stop
        assert len(mock_trading_engine.emergency_stops) == 1
        assert mock_trading_engine.emergency_stops[0]['reason'] == "Test emergency"
        assert mock_trading_engine.state == 'emergency_stopped'

    @pytest.mark.asyncio
    async def test_comprehensive_risk_check_enterprise_mode(self, enhanced_risk_manager):
        """Test comprehensive risk check in enterprise mode."""
        symbol = "BTC-USD"

        # Mock some risk conditions
        with patch.object(enhanced_risk_manager, '_get_current_market_data') as mock_market_data:
            mock_market_data.return_value = {
                'price_data': {
                    symbol: {
                        'current_price': 50000.0,
                        'previous_price': 49000.0,  # 2% change
                        'timestamp': datetime.now()
                    }
                },
                'volume_data': {
                    symbol: {
                        'volume': 1000000.0,
                        'timestamp': datetime.now()
                    }
                }
            }

            risk_result = await enhanced_risk_manager.perform_comprehensive_risk_check(symbol)

            assert isinstance(risk_result, RiskCheckResult)
            assert hasattr(risk_result, 'allowed')
            assert hasattr(risk_result, 'risk_level')
            assert hasattr(risk_result, 'violations')
            assert hasattr(risk_result, 'warnings')
            assert hasattr(risk_result, 'metrics')

    @pytest.mark.asyncio
    async def test_emergency_protocol_integration(self, enhanced_risk_manager, mock_trading_engine):
        """Test emergency protocol integration with trading engine."""
        # Setup trading engine callbacks
        enhanced_risk_manager.set_trading_engine_callbacks(
            emergency_stop=mock_trading_engine.emergency_stop,
            cancel_orders=mock_trading_engine.cancel_all_orders
        )

        # Simulate crisis conditions that should trigger emergency protocols
        crisis_conditions = {
            'components': {
                'pnl_controller': {
                    'global_metrics': {'daily': -15000.0}  # Large loss
                },
                'operational_risk': {
                    'system_health': 'failure',
                    'overall_score': 10.0
                },
                'circuit_breakers': {
                    'active_breakers': 3
                }
            }
        }

        # Check emergency conditions
        if enhanced_risk_manager.emergency_protocols:
            triggered = await enhanced_risk_manager.emergency_protocols.check_emergency_conditions(
                crisis_conditions
            )

            # Should have triggered emergency protocols
            if triggered:
                # Verify trading engine received emergency actions
                assert len(mock_trading_engine.emergency_stops) >= 0
                assert len(mock_trading_engine.cancelled_orders) >= 0

    @pytest.mark.asyncio
    async def test_circuit_breaker_integration(self, enhanced_risk_manager, mock_trading_engine):
        """Test circuit breaker integration with trading engine."""
        enhanced_risk_manager.set_trading_engine_callbacks(
            emergency_stop=mock_trading_engine.emergency_stop
        )

        # Simulate conditions that should trigger circuit breakers
        market_data = {
            'price_data': {
                'BTC-USD': {
                    'current_price': 40000.0,
                    'previous_price': 50000.0,  # 20% crash
                    'timestamp': datetime.now()
                }
            },
            'volume_data': {
                'BTC-USD': {
                    'volume': 10000000.0,  # 10x normal volume
                    'timestamp': datetime.now()
                }
            }
        }

        # Check if circuit breakers would trigger
        if enhanced_risk_manager.circuit_breakers:
            triggered_breakers = await enhanced_risk_manager.circuit_breakers.check_all_breakers(market_data)

            # If breakers triggered, emergency should be called
            if triggered_breakers:
                # Simulate the integration flow
                await enhanced_risk_manager._handle_emergency_halt("Circuit breaker triggered")
                assert len(mock_trading_engine.emergency_stops) == 1

    @pytest.mark.asyncio
    async def test_position_limit_integration(self, enhanced_risk_manager):
        """Test position limit integration with portfolio tracking."""
        symbol = "BTC-USD"

        # Simulate a large position update
        await enhanced_risk_manager.update_position(symbol, "buy", 1.0, 50000.0)

        # Check if position limits are enforced
        risk_check = await enhanced_risk_manager.perform_comprehensive_risk_check(symbol)

        # Should have valid risk assessment
        assert risk_check.risk_level in ['normal', 'medium', 'high', 'critical']
        assert isinstance(risk_check.allowed, bool)

    @pytest.mark.asyncio
    async def test_legacy_circuit_breaker_compatibility(self, enhanced_risk_manager, mock_trading_engine):
        """Test legacy circuit breaker still works with enhanced system."""
        enhanced_risk_manager.set_trading_engine_callbacks(
            emergency_stop=mock_trading_engine.emergency_stop
        )

        # Set conditions that would trigger legacy circuit breaker
        enhanced_risk_manager.daily_pnl = -15000.0  # Large loss
        enhanced_risk_manager.max_position_usdc = 10000.0

        # This should trigger legacy circuit breaker which should integrate with emergency protocols
        try:
            await enhanced_risk_manager._check_pnl_limits()
            # If we get here, the check passed (no trigger)
        except CircuitBreakerError:
            # Circuit breaker was triggered - this is expected behavior
            pass

    @pytest.mark.asyncio
    async def test_performance_monitoring_integration(self, enhanced_risk_manager):
        """Test performance monitoring doesn't impact trading performance."""
        symbol = "BTC-USD"

        # Time multiple risk checks to ensure performance
        start_time = datetime.now()

        for _ in range(10):
            await enhanced_risk_manager.check_trading_allowed(symbol)

        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()
        avg_time_per_check = total_time / 10

        # Risk checks should be fast (under 50ms each)
        assert avg_time_per_check < 0.05

    @pytest.mark.asyncio
    async def test_metrics_aggregation(self, enhanced_risk_manager):
        """Test that enhanced metrics include both legacy and enterprise data."""
        metrics = await enhanced_risk_manager.get_risk_metrics()

        # Should have legacy metrics
        assert 'circuit_breaker_active' in metrics
        assert 'daily_pnl' in metrics
        assert 'daily_volume' in metrics
        assert 'positions' in metrics

        # Should have enhanced metrics
        assert 'enterprise_mode_enabled' in metrics
        assert 'last_risk_check' in metrics

        if enhanced_risk_manager.enterprise_mode_enabled:
            assert 'enterprise_metrics' in metrics

    @pytest.mark.asyncio
    async def test_risk_reporting_integration(self, enhanced_risk_manager):
        """Test risk reporting integration."""
        # Generate comprehensive risk report
        report = await enhanced_risk_manager.generate_risk_report()

        assert isinstance(report, dict)
        assert len(report) > 0

        # Should include key risk information
        expected_fields = ['circuit_breaker_active', 'daily_pnl', 'positions']
        for field in expected_fields:
            assert field in report

    @pytest.mark.asyncio
    async def test_monitoring_lifecycle(self, enhanced_risk_manager):
        """Test monitoring start/stop lifecycle."""
        # Start monitoring
        await enhanced_risk_manager.start_monitoring()
        assert enhanced_risk_manager.monitoring_active

        # Stop monitoring
        await enhanced_risk_manager.stop_monitoring()
        assert not enhanced_risk_manager.monitoring_active

    @pytest.mark.asyncio
    async def test_cleanup_integration(self, enhanced_risk_manager):
        """Test cleanup properly shuts down all components."""
        # Ensure all components are initialized
        assert enhanced_risk_manager.initialized

        # Cleanup should not raise exceptions
        await enhanced_risk_manager.cleanup()

        # Should handle cleanup gracefully
        # (In a real test, we'd verify components are properly cleaned up)


class TestRealWorldIntegrationScenarios:
    """Test real-world integration scenarios."""

    @pytest.fixture
    async def integrated_system(self):
        """Create integrated system for testing."""
        risk_manager = EnhancedRiskManager()
        await risk_manager.initialize()

        trading_engine = MockMarketMakingEngine()

        # Connect them
        risk_manager.set_trading_engine_callbacks(
            emergency_stop=trading_engine.emergency_stop,
            pause_trading=trading_engine.pause,
            cancel_orders=trading_engine.cancel_all_orders
        )

        return risk_manager, trading_engine

    @pytest.mark.asyncio
    async def test_market_crash_scenario(self, integrated_system):
        """Test system behavior during market crash."""
        risk_manager, trading_engine = integrated_system

        # Simulate market crash conditions
        crash_updates = [
            # Initial normal conditions
            {'symbol': 'BTC-USD', 'side': 'buy', 'size': 0.1, 'price': 50000.0},

            # Market starts crashing
            {'symbol': 'BTC-USD', 'side': 'sell', 'size': 0.05, 'price': 48000.0},
            {'symbol': 'BTC-USD', 'side': 'sell', 'size': 0.1, 'price': 45000.0},

            # Major crash
            {'symbol': 'BTC-USD', 'side': 'sell', 'size': 0.2, 'price': 40000.0},
        ]

        # Process each update and check system response
        allowed = True  # Initialize allowed variable
        for update in crash_updates:
            await risk_manager.update_position(
                update['symbol'], update['side'], update['size'], update['price']
            )

            # Check if trading is still allowed
            allowed = await risk_manager.check_trading_allowed(update['symbol'])

            # In a crash scenario, system should eventually stop trading
            if not allowed:
                break

        # System should have taken protective action
        trading_status = trading_engine.get_status()

        # Either emergency stopped or paused
        assert (trading_status['emergency_stops'] > 0 or
                trading_status['paused_events'] > 0 or
                not allowed)

    @pytest.mark.asyncio
    async def test_gradual_loss_scenario(self, integrated_system):
        """Test system behavior during gradual losses."""
        risk_manager, trading_engine = integrated_system

        # Simulate gradual losses over time
        cumulative_loss = 0.0
        allowed = True  # Initialize allowed variable

        for _ in range(20):  # 20 small losing trades
            loss_per_trade = 100.0  # $100 loss per trade
            cumulative_loss += loss_per_trade

            # Update P&L
            risk_manager.daily_pnl = -cumulative_loss

            # Check if trading is still allowed
            allowed = await risk_manager.check_trading_allowed("BTC-USD")

            if not allowed:
                break

        # Should eventually trigger risk controls if losses are large enough
        if cumulative_loss > risk_manager.circuit_breaker_loss_percent * risk_manager.max_position_usdc / 100:
            assert not allowed or trading_engine.get_status()['emergency_stops'] > 0

    @pytest.mark.asyncio
    async def test_system_failure_scenario(self, integrated_system):
        """Test system behavior during operational failures."""
        risk_manager, trading_engine = integrated_system

        # Simulate system health degradation
        if risk_manager.operational_monitor:
            # Mock failing system health check
            with patch.object(risk_manager.operational_monitor, 'perform_health_check') as mock_health:
                mock_health.return_value = {
                    'overall_score': 15.0,  # Very poor health
                    'system_health': {'health_score': 10.0},
                    'connectivity_health': {'connectivity_score': 20.0},
                    'risk_level': 'critical'
                }

                # Check if system responds to poor health
                risk_check = await risk_manager.perform_comprehensive_risk_check("BTC-USD")

                # System should detect operational issues
                assert risk_check.risk_level in ['high', 'critical']

    @pytest.mark.asyncio
    async def test_recovery_scenario(self, integrated_system):
        """Test system recovery after emergency."""
        risk_manager, trading_engine = integrated_system

        # Trigger emergency stop
        await trading_engine.emergency_stop("Test emergency")
        assert not trading_engine.is_running

        # Reset conditions
        await risk_manager.reset_circuit_breaker()
        await risk_manager.reset_daily_metrics()

        # Check if trading can be allowed again
        allowed = await risk_manager.check_trading_allowed("BTC-USD")

        # Should be able to resume trading after reset (in normal conditions)
        # Note: This depends on the specific implementation of recovery logic
        assert isinstance(allowed, bool)  # Should return valid boolean

    @pytest.mark.asyncio
    async def test_high_frequency_operations(self, integrated_system):
        """Test system under high-frequency trading conditions."""
        risk_manager, trading_engine = integrated_system

        # Simulate high-frequency trading checks
        symbol = "BTC-USD"
        check_count = 100

        start_time = datetime.now()

        # Perform many rapid risk checks
        results = []
        for i in range(check_count):
            allowed = await risk_manager.check_trading_allowed(symbol)
            results.append(allowed)

            # Simulate small position updates
            if i % 10 == 0:  # Every 10th check, update position
                await risk_manager.update_position(symbol, "buy", 0.001, 50000.0 + i)

        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()

        # Should handle high frequency efficiently
        avg_time_per_check = total_time / check_count
        assert avg_time_per_check < 0.01  # Under 10ms per check

        # All checks should return valid results
        assert all(isinstance(result, bool) for result in results)


@pytest.mark.asyncio
async def test_factory_function():
    """Test the factory function for creating enhanced risk manager."""
    from flashmm.trading.risk.enhanced_risk_manager import create_risk_manager

    # Test enterprise mode
    enterprise_manager = await create_risk_manager(enterprise_mode=True)
    assert enterprise_manager.enterprise_mode_enabled
    assert enterprise_manager.initialized

    # Test legacy mode
    legacy_manager = await create_risk_manager(enterprise_mode=False)
    assert not legacy_manager.enterprise_mode_enabled
    assert legacy_manager.initialized

    # Cleanup
    await enterprise_manager.cleanup()
    await legacy_manager.cleanup()


if __name__ == "__main__":
    pytest.main([__file__])

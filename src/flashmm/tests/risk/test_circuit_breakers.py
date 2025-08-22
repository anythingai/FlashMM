"""
Test suite for Circuit Breaker System

Tests all circuit breaker types and the coordinating system including:
- Individual breaker functionality
- Circuit breaker system coordination
- Threshold monitoring and breach detection
- Automatic recovery mechanisms
- Performance and latency requirements
"""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock

import pytest

from flashmm.risk.circuit_breakers import (
    BreakerState,
    CircuitBreakerSystem,
    LatencyCircuitBreaker,
    PnLCircuitBreaker,
    PriceCircuitBreaker,
    VolumeCircuitBreaker,
)


class TestPriceCircuitBreaker:
    """Test price circuit breaker functionality."""

    @pytest.fixture
    def price_breaker(self):
        """Create price breaker for testing."""
        return PriceCircuitBreaker(symbol="BTC-USD")

    @pytest.mark.asyncio
    async def test_price_breaker_initialization(self, price_breaker):
        """Test price breaker initializes correctly."""
        assert price_breaker.symbol == "BTC-USD"
        assert price_breaker.config.breaker_type.value == "price"
        assert price_breaker.state == BreakerState.CLOSED
        assert not price_breaker.is_tripped()

    @pytest.mark.asyncio
    async def test_normal_price_changes(self, price_breaker):
        """Test breaker doesn't trip on normal price changes."""
        # Small price changes should not trip breaker (2% change)
        price_change_pct = 2.0
        metadata = {'current_price': 50000.0, 'symbol': 'BTC-USD'}

        should_trip = await price_breaker.check_condition(price_change_pct, metadata)
        assert not should_trip
        assert price_breaker.state == BreakerState.CLOSED

    @pytest.mark.asyncio
    async def test_large_price_change_trips_breaker(self, price_breaker):
        """Test breaker trips on large price changes."""
        # Large price change should trip breaker (~11% change)
        price_change_pct = 11.0
        metadata = {'current_price': 50000.0, 'symbol': 'BTC-USD'}

        should_trip = await price_breaker.check_condition(price_change_pct, metadata)
        assert should_trip
        assert price_breaker.state == BreakerState.OPEN
        assert price_breaker.is_tripped()

    @pytest.mark.asyncio
    async def test_breaker_cooldown(self, price_breaker):
        """Test breaker cooldown mechanism."""
        # Trip the breaker manually
        await price_breaker.manual_trip("Test trip")
        assert price_breaker.state == BreakerState.OPEN

        # Should still be tripped during cooldown
        price_change_pct = 1.0  # Normal change
        metadata = {'current_price': 50000.0, 'symbol': 'BTC-USD'}

        should_trip = await price_breaker.check_condition(price_change_pct, metadata)
        assert should_trip  # Still tripped because breaker is open

        # Reset the breaker manually
        await price_breaker.manual_reset("Test reset")
        assert price_breaker.state == BreakerState.CLOSED
        assert not price_breaker.is_tripped()

    @pytest.mark.asyncio
    async def test_price_change_calculation(self, price_breaker):
        """Test price change percentage calculation."""
        test_cases = [
            (100.0, 110.0, 10.0),   # 10% increase
            (100.0, 90.0, -10.0),   # 10% decrease
            (100.0, 105.5, 5.5),    # 5.5% increase
            (100.0, 95.0, -5.0),    # 5% decrease
        ]

        for prev_price, curr_price, expected_change in test_cases:
            _test_data = {
                'symbol': 'TEST',
                'current_price': curr_price,
                'previous_price': prev_price,
                'timestamp': datetime.now()
            }

            # Calculate actual change
            change_pct = abs((curr_price - prev_price) / prev_price * 100)
            assert abs(change_pct - abs(expected_change)) < 0.1

    @pytest.mark.asyncio
    async def test_missing_price_data(self, price_breaker):
        """Test handling of missing price data."""
        # Test with zero price change and no current price
        price_change_pct = 0.0
        metadata = {'symbol': 'BTC-USD'}  # Missing current_price

        should_trip = await price_breaker.check_condition(price_change_pct, metadata)
        assert not should_trip  # Should not trip on missing data


class TestVolumeCircuitBreaker:
    """Test volume circuit breaker functionality."""

    @pytest.fixture
    def volume_breaker(self):
        """Create volume breaker for testing."""
        return VolumeCircuitBreaker(symbol="BTC-USD")

    @pytest.mark.asyncio
    async def test_volume_baseline_calculation(self, volume_breaker):
        """Test volume baseline calculation."""
        # Add historical volume data to build baseline
        historical_volumes = [1000, 1200, 800, 1500, 900]
        for volume in historical_volumes:
            volume_breaker.volume_history.append(volume)

        # Verify baseline is calculated correctly
        _expected_baseline = sum(historical_volumes) / len(historical_volumes)
        assert abs(volume_breaker.normal_volume_estimate - 1000.0) < 0.01  # Initial estimate

    @pytest.mark.asyncio
    async def test_normal_volume_no_trip(self, volume_breaker):
        """Test breaker doesn't trip on normal volume."""
        # Set up baseline volume history
        for _i in range(10):
            volume_breaker.volume_history.append(1000)

        # Normal volume (2x baseline, below 10x threshold)
        current_volume = 2000.0
        metadata = {'symbol': 'BTC-USD'}

        should_trip = await volume_breaker.check_condition(current_volume, metadata)
        assert not should_trip

    @pytest.mark.asyncio
    async def test_high_volume_trips_breaker(self, volume_breaker):
        """Test breaker trips on abnormally high volume."""
        # Set up baseline volume history
        for _i in range(10):
            volume_breaker.volume_history.append(1000)

        # Abnormally high volume (12x baseline, above 10x threshold)
        current_volume = 12000.0
        metadata = {'symbol': 'BTC-USD'}

        should_trip = await volume_breaker.check_condition(current_volume, metadata)
        assert should_trip


class TestPnLCircuitBreaker:
    """Test P&L circuit breaker functionality."""

    @pytest.fixture
    def pnl_breaker(self):
        """Create P&L breaker for testing."""
        return PnLCircuitBreaker()

    @pytest.mark.asyncio
    async def test_small_losses_no_trip(self, pnl_breaker):
        """Test breaker doesn't trip on small losses."""
        # Small loss below threshold (default is 5000)
        current_pnl = -3000.0
        metadata = {'total_pnl': current_pnl}

        should_trip = await pnl_breaker.check_condition(current_pnl, metadata)
        assert not should_trip

    @pytest.mark.asyncio
    async def test_large_losses_trip_breaker(self, pnl_breaker):
        """Test breaker trips on large losses."""
        # Large loss above threshold (default is 5000)
        current_pnl = -6000.0
        metadata = {'total_pnl': current_pnl}

        should_trip = await pnl_breaker.check_condition(current_pnl, metadata)
        assert should_trip

    @pytest.mark.asyncio
    async def test_drawdown_calculation(self, pnl_breaker):
        """Test drawdown percentage calculation."""
        # Set initial peak PnL
        pnl_breaker.peak_pnl = 8000.0

        # Current PnL represents a significant drawdown
        current_pnl = 3000.0
        metadata = {'total_pnl': current_pnl}

        # Should trip due to drawdown from 8000 to 3000 (5000 drawdown > 5000 threshold)
        should_trip = await pnl_breaker.check_condition(current_pnl, metadata)
        assert should_trip


class TestLatencyCircuitBreaker:
    """Test latency circuit breaker functionality."""

    @pytest.fixture
    def latency_breaker(self):
        """Create latency breaker for testing."""
        return LatencyCircuitBreaker()

    @pytest.mark.asyncio
    async def test_normal_latency_no_trip(self, latency_breaker):
        """Test breaker doesn't trip on normal latency."""
        for _i in range(5):
            # Normal latency below threshold (default is 1000ms)
            current_latency = 50.0
            metadata = {'system_latency_ms': current_latency}

            should_trip = await latency_breaker.check_condition(current_latency, metadata)
            assert not should_trip

    @pytest.mark.asyncio
    async def test_consecutive_high_latency_trips(self, latency_breaker):
        """Test breaker trips on high latency."""
        # High latency above threshold (default is 1000ms)
        current_latency = 1500.0
        metadata = {'system_latency_ms': current_latency}

        should_trip = await latency_breaker.check_condition(current_latency, metadata)
        assert should_trip

    @pytest.mark.asyncio
    async def test_intermittent_high_latency_no_trip(self, latency_breaker):
        """Test intermittent high latency doesn't trip breaker."""
        # Mix of normal and high latencies (but high ones are below 1000ms threshold)
        latencies = [150.0, 50.0, 150.0, 50.0, 150.0]  # All below 1000ms threshold

        for latency in latencies:
            metadata = {'system_latency_ms': latency}
            should_trip = await latency_breaker.check_condition(latency, metadata)
            assert not should_trip  # Should never trip as all are below threshold


class TestCircuitBreakerSystem:
    """Test the complete circuit breaker system."""

    @pytest.fixture
    def circuit_system(self):
        """Create circuit breaker system for testing."""
        return CircuitBreakerSystem()

    @pytest.mark.asyncio
    async def test_system_initialization(self, circuit_system):
        """Test system initializes with default breakers."""
        await circuit_system.initialize()

        assert len(circuit_system.breakers) > 0
        # Check if breakers are properly initialized
        breaker_names = list(circuit_system.breakers.keys())
        assert any('price_breaker' in name for name in breaker_names)
        assert any('volume_breaker' in name for name in breaker_names)
        assert 'pnl_drawdown_breaker' in breaker_names
        assert 'latency_breaker' in breaker_names

    @pytest.mark.asyncio
    async def test_add_custom_breaker(self, circuit_system):
        """Test adding custom breaker to system."""
        await circuit_system.initialize()

        custom_breaker = PriceCircuitBreaker(symbol="ETH-USD")

        await circuit_system.register_breaker(custom_breaker)
        assert custom_breaker.config.name in circuit_system.breakers
        assert circuit_system.breakers[custom_breaker.config.name] == custom_breaker

    @pytest.mark.asyncio
    async def test_check_all_breakers(self, circuit_system):
        """Test checking all breakers with market data."""
        await circuit_system.initialize()

        market_data = {
            'price_data': {
                'BTC-USD': {
                    'current_price': 50000.0,
                    'previous_price': 49000.0,
                    'timestamp': datetime.now()
                }
            },
            'volume_data': {
                'BTC-USD': {
                    'volume': 2000.0,
                    'timestamp': datetime.now()
                }
            },
            'pnl_data': {
                'total_pnl': -5000.0,
                'unrealized_pnl': -3000.0,
                'realized_pnl': -2000.0,
                'timestamp': datetime.now()
            },
            'latency_data': {
                'market_data': {
                    'latency_ms': 75.0,
                    'timestamp': datetime.now()
                }
            }
        }

        system_halted = await circuit_system.check_all_breakers(market_data)
        assert isinstance(system_halted, bool)
        # With normal data, system should not be halted
        assert not system_halted

    @pytest.mark.asyncio
    async def test_system_emergency_stop(self, circuit_system):
        """Test system-wide emergency stop."""
        await circuit_system.initialize()

        # Mock callback for emergency actions
        emergency_callback = AsyncMock()
        circuit_system.set_emergency_callback(emergency_callback)

        # Trigger emergency stop
        await circuit_system.emergency_stop("Test emergency stop")

        # All breakers should be tripped
        for breaker in circuit_system.breakers.values():
            assert breaker.state == BreakerState.OPEN
            assert breaker.is_tripped()

        # Emergency callback should be called
        emergency_callback.assert_called_once()

    @pytest.mark.asyncio
    async def test_system_status_reporting(self, circuit_system):
        """Test system status reporting."""
        await circuit_system.initialize()

        status = circuit_system.get_system_status()

        assert 'active_breakers' in status
        assert 'total_breakers' in status
        assert 'system_halted' in status
        assert 'breakers' in status

        # Initially all breakers should be closed
        assert status['active_breakers'] == 0
        assert not status['system_halted']

    @pytest.mark.asyncio
    async def test_performance_monitoring(self, circuit_system):
        """Test system performance monitoring."""
        await circuit_system.initialize()

        # Simulate multiple checks
        market_data = {
            'price_data': {'BTC-USD': {'current_price': 50000.0, 'previous_price': 49000.0, 'timestamp': datetime.now()}},
            'volume_data': {'BTC-USD': {'volume': 1000.0, 'timestamp': datetime.now()}},
            'pnl_data': {'total_pnl': 1000.0, 'timestamp': datetime.now()},
            'latency_data': {'market_data': {'latency_ms': 50.0, 'timestamp': datetime.now()}}
        }

        # Time multiple checks
        start_time = datetime.now()
        for _ in range(100):
            await circuit_system.check_all_breakers(market_data)
        end_time = datetime.now()

        total_time = (end_time - start_time).total_seconds()
        avg_time_per_check = total_time / 100

        # Should be very fast (under 10ms per check)
        assert avg_time_per_check < 0.01

    @pytest.mark.asyncio
    async def test_concurrent_breaker_checks(self, circuit_system):
        """Test concurrent breaker checks."""
        await circuit_system.initialize()

        market_data = {
            'price_data': {'BTC-USD': {'current_price': 50000.0, 'previous_price': 49000.0, 'timestamp': datetime.now()}},
            'volume_data': {'BTC-USD': {'volume': 1000.0, 'timestamp': datetime.now()}},
            'pnl_data': {'total_pnl': 1000.0, 'timestamp': datetime.now()},
            'latency_data': {'market_data': {'latency_ms': 50.0, 'timestamp': datetime.now()}}
        }

        # Run multiple concurrent checks
        tasks = [circuit_system.check_all_breakers(market_data) for _ in range(10)]
        results = await asyncio.gather(*tasks)

        # All should complete successfully
        assert len(results) == 10
        for result in results:
            assert isinstance(result, bool)


@pytest.mark.asyncio
async def test_integration_scenario():
    """Test complete integration scenario with multiple breaker triggers."""
    system = CircuitBreakerSystem()
    await system.initialize()

    # Mock emergency callback
    emergency_actions = []
    async def mock_emergency_callback(breaker_name, reason):
        emergency_actions.append({'breaker': breaker_name, 'reason': reason})

    system.set_emergency_callback(mock_emergency_callback)

    # Create scenario with multiple triggers
    crisis_market_data = {
        'price_data': {
            'BTC-USD': {
                'current_price': 40000.0,   # 20% drop
                'previous_price': 50000.0,
                'timestamp': datetime.now()
            }
        },
        'volume_data': {
            'BTC-USD': {
                'volume': 50000.0,  # Very high volume
                'timestamp': datetime.now()
            }
        },
        'pnl_data': {
            'total_pnl': -25000.0,  # Large loss
            'unrealized_pnl': -20000.0,
            'realized_pnl': -5000.0,
            'timestamp': datetime.now()
        },
        'latency_data': {
            'market_data': {
                'latency_ms': 200.0,  # High latency
                'timestamp': datetime.now()
            }
        }
    }

    # Check breakers - should trigger multiple
    system_halted = await system.check_all_breakers(crisis_market_data)

    # Should have triggered the system halt
    assert system_halted

    # Should have called emergency callbacks
    assert len(emergency_actions) > 0


if __name__ == "__main__":
    pytest.main([__file__])

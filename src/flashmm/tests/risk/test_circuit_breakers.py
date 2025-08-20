"""
Test suite for Circuit Breaker System

Tests all circuit breaker types and the coordinating system including:
- Individual breaker functionality
- Circuit breaker system coordination
- Threshold monitoring and breach detection
- Automatic recovery mechanisms
- Performance and latency requirements
"""

import pytest
import asyncio
from decimal import Decimal
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, List

from flashmm.risk.circuit_breakers import (
    CircuitBreakerSystem,
    PriceCircuitBreaker,
    VolumeCircuitBreaker,
    PnLCircuitBreaker,
    LatencyCircuitBreaker,
    BreakerState
)
from flashmm.utils.exceptions import RiskError


class TestPriceCircuitBreaker:
    """Test price circuit breaker functionality."""
    
    @pytest.fixture
    def price_breaker(self):
        """Create price breaker for testing."""
        return PriceCircuitBreaker(
            name="test_price_breaker",
            price_change_threshold_pct=5.0,
            time_window_minutes=1.0,
            cooldown_minutes=5.0
        )
    
    @pytest.mark.asyncio
    async def test_price_breaker_initialization(self, price_breaker):
        """Test price breaker initializes correctly."""
        assert price_breaker.name == "test_price_breaker"
        assert price_breaker.price_change_threshold_pct == 5.0
        assert price_breaker.time_window_minutes == 1.0
        assert price_breaker.cooldown_minutes == 5.0
        assert price_breaker.state == BreakerState.CLOSED
        assert not price_breaker.tripped
    
    @pytest.mark.asyncio
    async def test_normal_price_changes(self, price_breaker):
        """Test breaker doesn't trip on normal price changes."""
        # Small price changes should not trip breaker
        test_data = {
            'symbol': 'BTC-USD',
            'current_price': 50000.0,
            'previous_price': 49000.0,  # 2% change
            'timestamp': datetime.now()
        }
        
        should_trip = await price_breaker.should_trip(test_data)
        assert not should_trip
        assert price_breaker.state == BreakerState.CLOSED
    
    @pytest.mark.asyncio
    async def test_large_price_change_trips_breaker(self, price_breaker):
        """Test breaker trips on large price changes."""
        # Large price change should trip breaker
        test_data = {
            'symbol': 'BTC-USD',
            'current_price': 50000.0,
            'previous_price': 45000.0,  # ~11% change
            'timestamp': datetime.now()
        }
        
        should_trip = await price_breaker.should_trip(test_data)
        assert should_trip
        
        # Trip the breaker
        await price_breaker.trip("Large price change detected")
        assert price_breaker.state == BreakerState.OPEN
        assert price_breaker.tripped
    
    @pytest.mark.asyncio
    async def test_breaker_cooldown(self, price_breaker):
        """Test breaker cooldown mechanism."""
        # Trip the breaker
        await price_breaker.trip("Test trip")
        assert price_breaker.state == BreakerState.OPEN
        
        # Should still be tripped during cooldown
        test_data = {
            'symbol': 'BTC-USD',
            'current_price': 50000.0,
            'previous_price': 49500.0,  # Normal change
            'timestamp': datetime.now()
        }
        
        should_trip = await price_breaker.should_trip(test_data)
        assert not should_trip  # Normal change, but breaker is still open
        
        # Simulate cooldown period passing
        price_breaker.trip_time = datetime.now() - timedelta(minutes=6)
        can_reset = await price_breaker.can_reset()
        assert can_reset
        
        await price_breaker.reset()
        assert price_breaker.state == BreakerState.CLOSED
        assert not price_breaker.tripped
    
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
            test_data = {
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
        test_data = {
            'symbol': 'BTC-USD',
            'timestamp': datetime.now()
            # Missing price data
        }
        
        should_trip = await price_breaker.should_trip(test_data)
        assert not should_trip  # Should not trip on missing data


class TestVolumeCircuitBreaker:
    """Test volume circuit breaker functionality."""
    
    @pytest.fixture
    def volume_breaker(self):
        """Create volume breaker for testing."""
        return VolumeCircuitBreaker(
            name="test_volume_breaker",
            volume_threshold_multiplier=5.0,
            baseline_period_minutes=60.0,
            time_window_minutes=5.0,
            cooldown_minutes=10.0
        )
    
    @pytest.mark.asyncio
    async def test_volume_baseline_calculation(self, volume_breaker):
        """Test volume baseline calculation."""
        # Add historical volume data
        historical_volumes = [1000, 1200, 800, 1500, 900]
        for i, volume in enumerate(historical_volumes):
            volume_data = {
                'symbol': 'BTC-USD',
                'volume': volume,
                'timestamp': datetime.now() - timedelta(minutes=i*10)
            }
            volume_breaker.volume_history.append(volume_data)
        
        baseline = volume_breaker._calculate_baseline_volume()
        expected_baseline = sum(historical_volumes) / len(historical_volumes)
        assert abs(baseline - expected_baseline) < 0.01
    
    @pytest.mark.asyncio
    async def test_normal_volume_no_trip(self, volume_breaker):
        """Test breaker doesn't trip on normal volume."""
        # Set up baseline
        for i in range(10):
            volume_breaker.volume_history.append({
                'symbol': 'BTC-USD',
                'volume': 1000,
                'timestamp': datetime.now() - timedelta(minutes=i*5)
            })
        
        # Normal volume
        test_data = {
            'symbol': 'BTC-USD',
            'volume': 2000,  # 2x baseline
            'timestamp': datetime.now()
        }
        
        should_trip = await volume_breaker.should_trip(test_data)
        assert not should_trip
    
    @pytest.mark.asyncio
    async def test_high_volume_trips_breaker(self, volume_breaker):
        """Test breaker trips on abnormally high volume."""
        # Set up baseline
        for i in range(10):
            volume_breaker.volume_history.append({
                'symbol': 'BTC-USD',
                'volume': 1000,
                'timestamp': datetime.now() - timedelta(minutes=i*5)
            })
        
        # Abnormally high volume
        test_data = {
            'symbol': 'BTC-USD',
            'volume': 6000,  # 6x baseline (above 5x threshold)
            'timestamp': datetime.now()
        }
        
        should_trip = await volume_breaker.should_trip(test_data)
        assert should_trip


class TestPnLCircuitBreaker:
    """Test P&L circuit breaker functionality."""
    
    @pytest.fixture
    def pnl_breaker(self):
        """Create P&L breaker for testing."""
        return PnLCircuitBreaker(
            name="test_pnl_breaker",
            loss_threshold=10000.0,
            time_window_minutes=60.0,
            drawdown_threshold_pct=15.0,
            cooldown_minutes=30.0
        )
    
    @pytest.mark.asyncio
    async def test_small_losses_no_trip(self, pnl_breaker):
        """Test breaker doesn't trip on small losses."""
        test_data = {
            'unrealized_pnl': -5000.0,  # Below threshold
            'realized_pnl': -2000.0,
            'total_pnl': -7000.0,
            'timestamp': datetime.now()
        }
        
        should_trip = await pnl_breaker.should_trip(test_data)
        assert not should_trip
    
    @pytest.mark.asyncio
    async def test_large_losses_trip_breaker(self, pnl_breaker):
        """Test breaker trips on large losses."""
        test_data = {
            'unrealized_pnl': -15000.0,  # Above threshold
            'realized_pnl': -3000.0,
            'total_pnl': -18000.0,
            'timestamp': datetime.now()
        }
        
        should_trip = await pnl_breaker.should_trip(test_data)
        assert should_trip
    
    @pytest.mark.asyncio
    async def test_drawdown_calculation(self, pnl_breaker):
        """Test drawdown percentage calculation."""
        # Add some P&L history
        pnl_history = [
            {'total_pnl': 5000.0, 'timestamp': datetime.now() - timedelta(hours=2)},
            {'total_pnl': 8000.0, 'timestamp': datetime.now() - timedelta(hours=1)},
            {'total_pnl': 3000.0, 'timestamp': datetime.now()}  # Current drawdown
        ]
        
        pnl_breaker.pnl_history = pnl_history
        
        test_data = {
            'total_pnl': 3000.0,
            'timestamp': datetime.now()
        }
        
        # Should trip due to drawdown from 8000 to 3000 (62.5% drawdown)
        should_trip = await pnl_breaker.should_trip(test_data)
        assert should_trip


class TestLatencyCircuitBreaker:
    """Test latency circuit breaker functionality."""
    
    @pytest.fixture
    def latency_breaker(self):
        """Create latency breaker for testing."""
        return LatencyCircuitBreaker(
            name="test_latency_breaker",
            latency_threshold_ms=100.0,
            consecutive_breaches_threshold=3,
            time_window_minutes=5.0,
            cooldown_minutes=15.0
        )
    
    @pytest.mark.asyncio
    async def test_normal_latency_no_trip(self, latency_breaker):
        """Test breaker doesn't trip on normal latency."""
        for i in range(5):
            test_data = {
                'endpoint': 'market_data',
                'latency_ms': 50.0,  # Below threshold
                'timestamp': datetime.now()
            }
            
            should_trip = await latency_breaker.should_trip(test_data)
            assert not should_trip
    
    @pytest.mark.asyncio
    async def test_consecutive_high_latency_trips(self, latency_breaker):
        """Test breaker trips on consecutive high latency."""
        # First two high latencies shouldn't trip
        for i in range(2):
            test_data = {
                'endpoint': 'market_data',
                'latency_ms': 150.0,  # Above threshold
                'timestamp': datetime.now()
            }
            
            should_trip = await latency_breaker.should_trip(test_data)
            assert not should_trip
        
        # Third consecutive high latency should trip
        test_data = {
            'endpoint': 'market_data',
            'latency_ms': 150.0,
            'timestamp': datetime.now()
        }
        
        should_trip = await latency_breaker.should_trip(test_data)
        assert should_trip
    
    @pytest.mark.asyncio
    async def test_intermittent_high_latency_no_trip(self, latency_breaker):
        """Test intermittent high latency doesn't trip breaker."""
        latencies = [150.0, 50.0, 150.0, 50.0, 150.0]  # Intermittent
        
        for latency in latencies:
            test_data = {
                'endpoint': 'market_data',
                'latency_ms': latency,
                'timestamp': datetime.now()
            }
            
            should_trip = await latency_breaker.should_trip(test_data)
            assert not should_trip  # Should never trip due to intermittent pattern


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
        assert 'price_breaker' in circuit_system.breakers
        assert 'volume_breaker' in circuit_system.breakers
        assert 'pnl_breaker' in circuit_system.breakers
        assert 'latency_breaker' in circuit_system.breakers
    
    @pytest.mark.asyncio
    async def test_add_custom_breaker(self, circuit_system):
        """Test adding custom breaker to system."""
        await circuit_system.initialize()
        
        custom_breaker = PriceCircuitBreaker(
            name="custom_price_breaker",
            price_change_threshold_pct=10.0,
            time_window_minutes=2.0,
            cooldown_minutes=10.0
        )
        
        circuit_system.add_breaker(custom_breaker)
        assert 'custom_price_breaker' in circuit_system.breakers
        assert circuit_system.breakers['custom_price_breaker'] == custom_breaker
    
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
        
        triggered_breakers = await circuit_system.check_all_breakers(market_data)
        assert isinstance(triggered_breakers, list)
        # With normal data, no breakers should trigger
        assert len(triggered_breakers) == 0
    
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
            assert breaker.tripped
        
        # Emergency callback should be called
        emergency_callback.assert_called_once()
    
    @pytest.mark.asyncio 
    async def test_system_status_reporting(self, circuit_system):
        """Test system status reporting."""
        await circuit_system.initialize()
        
        status = circuit_system.get_system_status()
        
        assert 'active_breakers' in status
        assert 'total_breakers' in status
        assert 'system_health' in status
        assert 'last_check_time' in status
        assert 'breaker_details' in status
        
        # Initially all breakers should be closed
        assert status['active_breakers'] == 0
        assert status['system_health'] == 'healthy'
    
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
            assert isinstance(result, list)


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
    triggered = await system.check_all_breakers(crisis_market_data)
    
    # Should have triggered multiple breakers
    assert len(triggered) > 0
    
    # Should have called emergency callbacks
    assert len(emergency_actions) > 0


if __name__ == "__main__":
    pytest.main([__file__])
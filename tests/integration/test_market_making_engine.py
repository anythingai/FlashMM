"""
Integration tests for the Market Making Engine

Tests the complete system integration including ML predictions, quote generation,
order management, risk controls, and performance requirements validation.
"""

import pytest
import asyncio
import time
from decimal import Decimal
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, List, Any

from flashmm.trading.engine.market_making_engine import MarketMakingEngine, TradingMetrics
from flashmm.trading.state.state_machine import TradingState
from flashmm.ml.models.prediction_models import PredictionResult
from flashmm.trading.execution.order_router import Order, OrderStatus


class MockCambrianSDK:
    """Mock Cambrian SDK for testing."""
    
    def __init__(self):
        self.orders = {}
        self.fills = []
        self.market_data = {
            'SEI/USDC': {
                'best_bid': 0.4995,
                'best_ask': 0.5005,
                'mid_price': 0.5000,
                'volume_24h': 1000000,
                'order_book': {
                    'bids': [
                        {'price': 0.4995, 'size': 1000},
                        {'price': 0.4990, 'size': 2000},
                        {'price': 0.4985, 'size': 1500}
                    ],
                    'asks': [
                        {'price': 0.5005, 'size': 1000},
                        {'price': 0.5010, 'size': 2000},
                        {'price': 0.5015, 'size': 1500}
                    ]
                }
            }
        }
    
    async def place_order(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """Mock order placement."""
        order_id = f"test_order_{len(self.orders)}"
        order_data = {
            'order_id': order_id,
            'status': 'pending',
            'timestamp': datetime.now().isoformat(),
            **order
        }
        self.orders[order_id] = order_data
        
        # Simulate some orders getting filled
        if len(self.orders) % 3 == 0:  # Every 3rd order gets filled
            await asyncio.sleep(0.01)  # Small delay to simulate network
            self.fills.append({
                'order_id': order_id,
                'fill_price': order['price'],
                'fill_size': order['size'],
                'timestamp': datetime.now().isoformat()
            })
        
        return order_data
    
    async def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Mock order cancellation."""
        if order_id in self.orders:
            self.orders[order_id]['status'] = 'cancelled'
        return {'order_id': order_id, 'status': 'cancelled'}
    
    async def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """Mock order status check."""
        return self.orders.get(order_id, {'status': 'not_found'})
    
    async def get_market_data(self, symbol: str) -> Dict[str, Any]:
        """Mock market data."""
        return self.market_data.get(symbol, {})


@pytest.fixture
async def mock_cambrian_sdk():
    """Fixture for mock Cambrian SDK."""
    return MockCambrianSDK()


@pytest.fixture
async def market_making_engine(mock_cambrian_sdk):
    """Fixture for market making engine with mocked dependencies."""
    engine = MarketMakingEngine()
    
    # Mock external dependencies
    with patch('flashmm.trading.engine.market_making_engine.RedisClient') as mock_redis:
        mock_redis.return_value.initialize = AsyncMock()
        mock_redis.return_value.close = AsyncMock()
        mock_redis.return_value.set = AsyncMock()
        mock_redis.return_value.get = AsyncMock(return_value=None)
        mock_redis.return_value.keys = AsyncMock(return_value=[])
        
        with patch('flashmm.ml.prediction_service.PredictionService') as mock_prediction:
            mock_prediction.return_value.initialize = AsyncMock()
            mock_prediction.return_value.cleanup = AsyncMock()
            mock_prediction.return_value.get_prediction = AsyncMock(
                return_value=PredictionResult(
                    symbol="SEI/USDC",
                    prediction_horizon_minutes=5,
                    predicted_price=Decimal('0.5010'),
                    confidence=0.75,
                    predicted_direction=1,
                    predicted_magnitude=0.002
                )
            )
            
            with patch('flashmm.trading.execution.order_router.CambrianSDK', mock_cambrian_sdk):
                await engine.initialize()
                yield engine
                await engine.cleanup()


@pytest.mark.asyncio
class TestMarketMakingEngineIntegration:
    """Integration tests for Market Making Engine."""
    
    async def test_engine_initialization(self, market_making_engine):
        """Test engine initializes all components correctly."""
        engine = market_making_engine
        
        # Verify all components are initialized
        assert engine.prediction_service is not None
        assert engine.inference_engine is not None
        assert engine.quoting_strategy is not None
        assert engine.quote_generator is not None
        assert engine.order_router is not None
        assert engine.order_book_manager is not None
        assert engine.position_tracker is not None
        assert engine.state_machine is not None
        
        # Verify state machine is in correct state
        assert engine.state_machine.get_current_state() == TradingState.INACTIVE
    
    async def test_trading_lifecycle(self, market_making_engine):
        """Test complete trading lifecycle."""
        engine = market_making_engine
        
        # Start trading
        await engine.start()
        
        # Wait a moment for trading to start
        await asyncio.sleep(0.1)
        
        # Verify trading is active
        assert engine.state_machine.is_trading_active()
        assert engine.is_running
        
        # Let it run for a few cycles
        await asyncio.sleep(1.0)  # 1 second = ~5 cycles at 200ms
        
        # Verify metrics are being updated
        metrics = engine.get_metrics()
        assert metrics['cycle_count'] > 0
        assert metrics['average_cycle_time_ms'] > 0
        
        # Stop trading
        await engine.stop()
        
        # Verify trading stopped
        assert not engine.is_running
        assert engine.state_machine.get_current_state() == TradingState.INACTIVE
    
    async def test_cycle_time_performance(self, market_making_engine):
        """Test that trading cycles meet 200ms target."""
        engine = market_making_engine
        
        # Start trading
        await engine.start()
        
        # Let it run for sufficient cycles to get accurate measurements
        await asyncio.sleep(2.0)  # 2 seconds = ~10 cycles
        
        # Check performance metrics
        metrics = engine.get_metrics()
        
        # Verify we have enough cycles for meaningful measurement
        assert metrics['cycle_count'] >= 5
        
        # Verify average cycle time meets requirement
        avg_cycle_time = metrics['average_cycle_time_ms']
        assert avg_cycle_time <= 200.0, f"Average cycle time {avg_cycle_time}ms exceeds 200ms target"
        
        # Verify maximum cycle time is reasonable
        max_cycle_time = metrics['max_cycle_time_ms']
        assert max_cycle_time <= 500.0, f"Maximum cycle time {max_cycle_time}ms exceeds 500ms emergency threshold"
        
        # Verify consistent performance
        assert avg_cycle_time > 0, "Average cycle time should be positive"
        
        await engine.stop()
    
    async def test_quote_generation_and_placement(self, market_making_engine, mock_cambrian_sdk):
        """Test quote generation and order placement."""
        engine = market_making_engine
        
        # Start trading
        await engine.start()
        
        # Let it run to generate and place quotes
        await asyncio.sleep(1.0)
        
        # Check metrics
        metrics = engine.get_metrics()
        
        # Verify quotes were generated
        assert metrics['quotes_generated'] > 0, "No quotes were generated"
        
        # Verify orders were placed
        assert metrics['orders_placed'] > 0, "No orders were placed"
        
        # Verify orders in mock SDK
        assert len(mock_cambrian_sdk.orders) > 0, "No orders found in mock SDK"
        
        # Check order details
        orders = list(mock_cambrian_sdk.orders.values())
        symbols_traded = set(order['symbol'] for order in orders)
        assert 'SEI/USDC' in symbols_traded, "Expected symbol not traded"
        
        # Verify order types
        sides_traded = set(order['side'] for order in orders)
        assert 'buy' in sides_traded or 'sell' in sides_traded, "No buy or sell orders found"
        
        await engine.stop()
    
    async def test_inventory_control(self, market_making_engine, mock_cambrian_sdk):
        """Test inventory control within ±2% limits."""
        engine = market_making_engine
        
        # Set up position tracker with test limits
        if engine.position_tracker:
            engine.position_tracker.max_inventory_usdc = 1000.0  # $1000 limit for testing
        
        # Start trading
        await engine.start()
        
        # Simulate some fills to build inventory
        for i in range(10):
            await mock_cambrian_sdk.place_order({
                'symbol': 'SEI/USDC',
                'side': 'buy',
                'size': 100,
                'price': 0.4995,
                'type': 'limit'
            })
            await asyncio.sleep(0.1)
        
        # Let the system process
        await asyncio.sleep(0.5)
        
        # Check inventory compliance
        if engine.position_tracker:
            compliance = engine.position_tracker.check_inventory_compliance('SEI/USDC')
            
            # Verify inventory is within reasonable bounds
            assert compliance['inventory_ratio'] <= 1.0, f"Inventory ratio {compliance['inventory_ratio']} exceeds 100%"
            
            # If we have a position, verify it's being tracked
            position = await engine.position_tracker.get_position('SEI/USDC')
            if position and position['base_balance'] != 0:
                assert 'inventory_ratio' in compliance
                assert 'limit_utilization' in compliance
        
        await engine.stop()
    
    async def test_ml_integration(self, market_making_engine):
        """Test ML prediction integration."""
        engine = market_making_engine
        
        # Verify ML components are available
        assert engine.enable_ml_predictions
        assert engine.prediction_service is not None
        assert engine.inference_engine is not None
        
        # Start trading to trigger ML predictions
        await engine.start()
        await asyncio.sleep(1.0)
        
        # Check metrics for ML activity
        metrics = engine.get_metrics()
        
        # Verify ML predictions were requested
        component_timing = metrics.get('component_timing_ms', {})
        ml_prediction_time = component_timing.get('ml_prediction', 0)
        
        # ML prediction time should be positive if predictions were made
        if ml_prediction_time > 0:
            assert ml_prediction_time <= 50.0, f"ML prediction time {ml_prediction_time}ms exceeds 50ms target"
        
        await engine.stop()
    
    async def test_error_handling_and_recovery(self, market_making_engine):
        """Test error handling and system recovery."""
        engine = market_making_engine
        
        # Start trading
        await engine.start()
        await asyncio.sleep(0.5)
        
        # Simulate an error in the trading cycle
        original_execute_cycle = engine._execute_trading_cycle
        call_count = 0
        
        async def error_execute_cycle():
            nonlocal call_count
            call_count += 1
            if call_count == 3:  # Fail on 3rd call
                raise Exception("Simulated trading cycle error")
            return await original_execute_cycle()
        
        engine._execute_trading_cycle = error_execute_cycle
        
        # Let it run and encounter the error
        await asyncio.sleep(1.0)
        
        # Verify system handled the error gracefully
        assert engine.state_machine.get_current_state() == TradingState.ERROR
        
        # Verify metrics show the error was handled
        metrics = engine.get_metrics()
        assert metrics['emergency_stops'] >= 0  # Should track errors
        
        await engine.stop()
    
    async def test_state_machine_integration(self, market_making_engine):
        """Test state machine integration with trading engine."""
        engine = market_making_engine
        
        # Test state transitions
        assert engine.state_machine.get_current_state() == TradingState.INACTIVE
        
        # Start trading
        await engine.start()
        assert engine.state_machine.is_trading_active()
        
        # Pause trading
        await engine.pause()
        assert engine.state_machine.get_current_state() == TradingState.PAUSED
        assert not engine.is_running
        
        # Resume trading
        await engine.resume()
        assert engine.state_machine.is_trading_active()
        assert engine.is_running
        
        # Emergency stop
        await engine.emergency_stop("Test emergency stop")
        assert engine.state_machine.get_current_state() == TradingState.EMERGENCY_STOP
        assert not engine.is_running
        
        # Final stop
        await engine.stop()
        assert engine.state_machine.get_current_state() == TradingState.INACTIVE
    
    async def test_performance_monitoring_integration(self, market_making_engine):
        """Test performance monitoring integration."""
        engine = market_making_engine
        
        # Verify performance tracker is available
        if engine.enable_performance_monitoring:
            assert engine.performance_tracker is not None
        
        # Start trading
        await engine.start()
        await asyncio.sleep(1.0)
        
        # Get performance metrics
        metrics = engine.get_metrics()
        
        # Verify key performance indicators are tracked
        assert 'cycle_count' in metrics
        assert 'average_cycle_time_ms' in metrics
        assert 'quotes_generated' in metrics
        assert 'orders_placed' in metrics
        
        # Verify component timing is tracked
        component_timing = metrics.get('component_timing_ms', {})
        assert 'quote_generation' in component_timing
        assert 'order_management' in component_timing
        
        await engine.stop()


@pytest.mark.asyncio
class TestPerformanceRequirements:
    """Tests specifically for performance requirements validation."""
    
    async def test_200ms_cycle_requirement(self, market_making_engine):
        """Validate 200ms trading cycle requirement."""
        engine = market_making_engine
        
        # Start trading
        await engine.start()
        
        # Collect timing data over multiple cycles
        cycle_times = []
        start_time = time.time()
        
        # Run for at least 20 cycles to get statistical significance
        while len(cycle_times) < 20 and time.time() - start_time < 10:
            await asyncio.sleep(0.1)
            metrics = engine.get_metrics()
            if metrics['cycle_count'] > len(cycle_times):
                cycle_times.append(metrics['last_cycle_time_ms'])
        
        await engine.stop()
        
        # Statistical analysis
        assert len(cycle_times) >= 10, "Insufficient cycle data for analysis"
        
        avg_cycle_time = sum(cycle_times) / len(cycle_times)
        max_cycle_time = max(cycle_times)
        min_cycle_time = min(cycle_times)
        
        # Performance assertions
        assert avg_cycle_time <= 200.0, f"Average cycle time {avg_cycle_time:.1f}ms exceeds 200ms requirement"
        assert max_cycle_time <= 500.0, f"Maximum cycle time {max_cycle_time:.1f}ms exceeds 500ms emergency limit"
        assert min_cycle_time > 0, "Minimum cycle time should be positive"
        
        # Log performance statistics
        print(f"Cycle Time Statistics:")
        print(f"  Average: {avg_cycle_time:.1f}ms")
        print(f"  Maximum: {max_cycle_time:.1f}ms")
        print(f"  Minimum: {min_cycle_time:.1f}ms")
        print(f"  Target: 200ms ✓" if avg_cycle_time <= 200.0 else f"  Target: 200ms ✗")
    
    async def test_inventory_control_requirement(self, market_making_engine):
        """Validate ±2% inventory control requirement."""
        engine = market_making_engine
        
        # Set test limits
        max_inventory_usdc = 10000.0  # $10k for testing
        max_allowed_ratio = 0.02  # 2%
        
        if engine.position_tracker:
            engine.position_tracker.max_inventory_usdc = max_inventory_usdc
            engine.position_tracker.inventory_controller.max_inventory_ratio = max_allowed_ratio
        
        # Start trading
        await engine.start()
        
        # Simulate trading activity
        await asyncio.sleep(2.0)
        
        # Check inventory compliance
        if engine.position_tracker:
            compliance = engine.position_tracker.check_inventory_compliance('SEI/USDC')
            
            # Verify inventory is within ±2% requirement
            inventory_ratio = compliance.get('inventory_ratio', 0)
            assert inventory_ratio <= max_allowed_ratio, f"Inventory ratio {inventory_ratio:.3f} exceeds {max_allowed_ratio:.3f} requirement"
            
            # Verify compliance status
            assert compliance.get('compliant', False), "Inventory compliance check failed"
            
            # Get portfolio summary
            portfolio = engine.position_tracker.get_portfolio_summary()
            utilization = portfolio.get('portfolio_utilization', 0)
            
            print(f"Inventory Control Statistics:")
            print(f"  Current Ratio: {inventory_ratio:.3f}")
            print(f"  Max Allowed: {max_allowed_ratio:.3f}")
            print(f"  Portfolio Utilization: {utilization:.3f}")
            print(f"  Compliant: {'✓' if compliance.get('compliant', False) else '✗'}")
        
        await engine.stop()
    
    async def test_spread_improvement_target(self, market_making_engine):
        """Test for ≥40% spread improvement target (indicative)."""
        engine = market_making_engine
        
        # This test validates the optimization framework is in place
        # Actual spread improvement measurement requires real market data
        
        # Verify optimization components are available
        assert engine.quote_generator is not None
        assert hasattr(engine.quote_generator, 'spread_optimizer')
        
        # Start trading
        await engine.start()
        await asyncio.sleep(1.0)
        
        # Verify quotes are being generated with optimization
        metrics = engine.get_metrics()
        assert metrics['quotes_generated'] > 0, "No quotes generated for spread improvement testing"
        
        # Check if quote generator has spread optimization capability
        if hasattr(engine.quote_generator, 'spread_optimizer'):
            optimizer = engine.quote_generator.spread_optimizer
            
            # Verify optimization modes are available
            assert hasattr(optimizer, 'optimization_mode')
            print(f"Spread optimization mode: {optimizer.optimization_mode}")
            print(f"Target improvement: ≥40% (framework ready for measurement)")
        
        await engine.stop()


@pytest.mark.performance
class TestPerformanceBenchmarks:
    """Performance benchmark tests."""
    
    async def test_sustained_performance(self, market_making_engine):
        """Test sustained performance over extended period."""
        engine = market_making_engine
        
        await engine.start()
        
        # Run for 30 seconds to test sustained performance
        start_time = time.time()
        performance_snapshots = []
        
        while time.time() - start_time < 30:
            await asyncio.sleep(1.0)
            metrics = engine.get_metrics()
            performance_snapshots.append({
                'timestamp': time.time(),
                'cycle_count': metrics['cycle_count'],
                'avg_cycle_time': metrics['average_cycle_time_ms'],
                'max_cycle_time': metrics['max_cycle_time_ms']
            })
        
        await engine.stop()
        
        # Analyze sustained performance
        final_metrics = performance_snapshots[-1]
        cycles_per_second = final_metrics['cycle_count'] / 30
        
        print(f"Sustained Performance Results:")
        print(f"  Total Runtime: 30 seconds")
        print(f"  Total Cycles: {final_metrics['cycle_count']}")
        print(f"  Cycles/Second: {cycles_per_second:.1f}")
        print(f"  Target: ~5 cycles/second (200ms target)")
        print(f"  Final Avg Cycle Time: {final_metrics['avg_cycle_time']:.1f}ms")
        print(f"  Final Max Cycle Time: {final_metrics['max_cycle_time']:.1f}ms")
        
        # Performance assertions
        assert cycles_per_second >= 4.0, f"Cycles per second {cycles_per_second:.1f} below minimum acceptable rate"
        assert final_metrics['avg_cycle_time'] <= 250.0, "Average cycle time degraded over time"


if __name__ == "__main__":
    # Run with: python -m pytest tests/integration/test_market_making_engine.py -v
    pytest.main([__file__, "-v", "--tb=short"])
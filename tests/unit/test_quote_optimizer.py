"""
Unit tests for Quote Optimizer

Tests the advanced quote optimization system including market condition detection,
competition analysis, and ML-driven spread optimization.
"""

from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, patch

import pytest

from flashmm.trading.optimization.quote_optimizer import (
    CompetitionAnalyzer,
    MarketCondition,
    MarketConditionAnalyzer,
    MarketMetrics,
    MLSpreadPredictor,
    OptimizationResult,
    QuoteOptimizer,
)


@pytest.fixture
def sample_market_metrics():
    """Sample market metrics for testing."""
    return MarketMetrics(
        symbol="SEI/USDC",
        mid_price=Decimal('0.5000'),
        best_bid=Decimal('0.4995'),
        best_ask=Decimal('0.5005'),
        current_spread_bps=10.0,
        price_volatility_1m=0.02,
        price_volatility_5m=0.03,
        volume_volatility=0.15,
        bid_depth=Decimal('5000'),
        ask_depth=Decimal('4500'),
        order_book_imbalance=0.05,
        num_market_makers=3,
        recent_fills=15,
        recent_volume=Decimal('50000'),
        recent_pnl=Decimal('25.50')
    )


@pytest.fixture
def sample_market_data():
    """Sample market data for testing."""
    return {
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
        },
        'volume_volatility': 0.15
    }


@pytest.fixture
def sample_position_data():
    """Sample position data for testing."""
    return {
        'base_balance': 100.0,
        'quote_balance': 50.0,
        'recent_fills': 15,
        'recent_volume': 50000.0,
        'recent_pnl': 25.50
    }


@pytest.fixture
def sample_current_quotes():
    """Sample current quotes for testing."""
    return [
        {
            'side': 'buy',
            'price': 0.4993,
            'size': 500,
            'quote_id': 'test_bid_1'
        },
        {
            'side': 'sell',
            'price': 0.5007,
            'size': 500,
            'quote_id': 'test_ask_1'
        }
    ]


class TestMarketConditionAnalyzer:
    """Tests for market condition analyzer."""

    def test_volatility_calculation(self):
        """Test volatility calculation."""
        analyzer = MarketConditionAnalyzer()

        # Add sample price data
        symbol = "SEI/USDC"
        base_time = datetime.now()

        # Add price points with known volatility
        prices = [0.5000, 0.5010, 0.4995, 0.5015, 0.4990]
        for i, price in enumerate(prices):
            timestamp = base_time - timedelta(seconds=30 - i * 5)
            from collections import deque
            analyzer.price_history[symbol] = analyzer.price_history.get(symbol, deque())
            analyzer.price_history[symbol].append((timestamp, price))

        # Calculate volatility
        volatility = analyzer.calculate_volatility(symbol, 60)

        assert volatility >= 0, "Volatility should be non-negative"
        assert volatility < 1.0, "Volatility should be reasonable for test data"

    def test_market_condition_detection(self, sample_market_metrics):
        """Test market condition detection."""
        analyzer = MarketConditionAnalyzer()

        # Test calm market
        calm_metrics = sample_market_metrics
        calm_metrics.price_volatility_1m = 0.01  # Low volatility
        calm_metrics.current_spread_bps = 3.0    # Tight spread

        condition = analyzer.detect_market_condition(calm_metrics)
        assert condition == MarketCondition.CALM

        # Test volatile market
        volatile_metrics = sample_market_metrics
        volatile_metrics.price_volatility_1m = 0.10  # High volatility
        volatile_metrics.price_volatility_5m = 0.09  # Sustained

        condition = analyzer.detect_market_condition(volatile_metrics)
        assert condition == MarketCondition.VOLATILE

        # Test competitive market
        competitive_metrics = sample_market_metrics
        competitive_metrics.num_market_makers = 5   # Many MMs
        competitive_metrics.current_spread_bps = 2.0  # Very tight

        condition = analyzer.detect_market_condition(competitive_metrics)
        assert condition == MarketCondition.COMPETITIVE


class TestCompetitionAnalyzer:
    """Tests for competition analyzer."""

    def test_competition_analysis(self, sample_market_data):
        """Test competition analysis."""
        analyzer = CompetitionAnalyzer()

        result = analyzer.analyze_competition("SEI/USDC", sample_market_data)

        assert 'num_market_makers' in result
        assert 'avg_spread_bps' in result
        assert 'competitive_pressure' in result

        assert result['num_market_makers'] > 0
        assert result['avg_spread_bps'] >= 0
        assert 0 <= result['competitive_pressure'] <= 1

    def test_empty_order_book_handling(self):
        """Test handling of empty order book."""
        analyzer = CompetitionAnalyzer()

        empty_order_book = {'bids': [], 'asks': []}
        result = analyzer.analyze_competition("SEI/USDC", empty_order_book)

        assert result['num_market_makers'] == 0
        assert result['avg_spread_bps'] == 0
        assert result['competitive_pressure'] == 0


class TestMLSpreadPredictor:
    """Tests for ML spread predictor."""

    def test_feature_extraction(self, sample_market_metrics):
        """Test feature extraction for ML prediction."""
        predictor = MLSpreadPredictor()

        competition_data = {
            'num_market_makers': 3,
            'competitive_pressure': 0.3,
            'avg_spread_bps': 8.0
        }

        inventory_position = 0.05  # 5% inventory
        ml_prediction = {
            'confidence': 0.75,
            'predicted_direction': 1,
            'predicted_magnitude': 0.002
        }

        features = predictor._extract_features(
            sample_market_metrics,
            competition_data,
            inventory_position,
            ml_prediction
        )

        # Verify all expected features are present
        expected_features = [
            'volatility_1m', 'volatility_5m', 'volatility_ratio',
            'num_competitors', 'competitive_pressure', 'avg_competitor_spread',
            'inventory_position', 'inventory_skew',
            'order_book_imbalance', 'bid_ask_ratio',
            'recent_fill_rate', 'recent_pnl_norm',
            'ml_confidence', 'ml_direction', 'ml_magnitude'
        ]

        for feature in expected_features:
            assert feature in features, f"Missing feature: {feature}"
            assert isinstance(features[feature], int | float), f"Feature {feature} should be numeric"

    def test_spread_prediction(self, sample_market_metrics):
        """Test spread prediction."""
        predictor = MLSpreadPredictor()

        competition_data = {'num_market_makers': 3, 'competitive_pressure': 0.3, 'avg_spread_bps': 8.0}
        inventory_position = 0.0  # Neutral position

        bid_spread, ask_spread = predictor.predict_optimal_spreads(
            sample_market_metrics,
            competition_data,
            inventory_position,
            None
        )

        # Verify spreads are reasonable
        assert 1.0 <= bid_spread <= 50.0, f"Bid spread {bid_spread} outside reasonable range"
        assert 1.0 <= ask_spread <= 50.0, f"Ask spread {ask_spread} outside reasonable range"
        assert bid_spread > 0 and ask_spread > 0, "Spreads should be positive"

    def test_inventory_skewing(self, sample_market_metrics):
        """Test inventory-based spread skewing."""
        predictor = MLSpreadPredictor()

        competition_data = {'num_market_makers': 3, 'competitive_pressure': 0.3, 'avg_spread_bps': 8.0}

        # Test long position (should skew spreads)
        long_position = 0.3  # 30% long
        bid_spread_long, ask_spread_long = predictor.predict_optimal_spreads(
            sample_market_metrics, competition_data, long_position, None
        )

        # Test short position
        short_position = -0.3  # 30% short
        bid_spread_short, ask_spread_short = predictor.predict_optimal_spreads(
            sample_market_metrics, competition_data, short_position, None
        )

        # Test neutral position
        neutral_position = 0.0
        bid_spread_neutral, ask_spread_neutral = predictor.predict_optimal_spreads(
            sample_market_metrics, competition_data, neutral_position, None
        )

        # Verify skewing behavior (long position should have wider ask spread)
        assert ask_spread_long >= ask_spread_neutral, "Long position should widen ask spreads"
        assert bid_spread_short >= bid_spread_neutral, "Short position should widen bid spreads"


@pytest.mark.asyncio
class TestQuoteOptimizer:
    """Tests for quote optimizer."""

    @pytest.fixture
    async def quote_optimizer(self):
        """Quote optimizer fixture with mocked dependencies."""
        with patch('flashmm.trading.optimization.quote_optimizer.RedisClient') as mock_redis:
            mock_redis.return_value.initialize = AsyncMock()
            mock_redis.return_value.close = AsyncMock()
            mock_redis.return_value.set = AsyncMock()
            mock_redis.return_value.get = AsyncMock(return_value=None)
            mock_redis.return_value.keys = AsyncMock(return_value=[])

            optimizer = QuoteOptimizer()
            await optimizer.initialize()
            yield optimizer
            await optimizer.cleanup()

    async def test_optimization_initialization(self, quote_optimizer):
        """Test optimizer initialization."""
        optimizer = quote_optimizer

        assert optimizer.market_analyzer is not None
        assert optimizer.competition_analyzer is not None
        assert optimizer.ml_predictor is not None
        assert optimizer.optimization_mode in ['aggressive', 'conservative', 'adaptive']

    async def test_quote_optimization_process(
        self,
        quote_optimizer,
        sample_current_quotes,
        sample_market_data,
        sample_position_data
    ):
        """Test complete quote optimization process."""
        optimizer = quote_optimizer

        result = await optimizer.optimize_quotes(
            symbol="SEI/USDC",
            current_quotes=sample_current_quotes,
            market_data=sample_market_data,
            position_data=sample_position_data,
            ml_prediction={'confidence': 0.8, 'predicted_direction': 1}
        )

        # Verify optimization result
        assert isinstance(result, OptimizationResult)
        assert result.symbol == "SEI/USDC"
        assert result.optimized_bid_spread_bps > 0
        assert result.optimized_ask_spread_bps > 0
        assert 0 <= result.confidence_score <= 1
        assert isinstance(result.market_condition, MarketCondition)
        assert result.improvement_pct >= 0

    async def test_optimization_improvement_calculation(self, quote_optimizer):
        """Test improvement calculation logic."""
        optimizer = quote_optimizer

        # Test cases with known improvements
        test_cases = [
            # (original_bid, original_ask, optimized_bid, optimized_ask, expected_improvement)
            (5.0, 5.0, 4.0, 4.0, 20.0),  # 20% improvement
            (10.0, 10.0, 5.0, 5.0, 50.0),  # 50% improvement
            (3.0, 3.0, 3.0, 3.0, 0.0),    # No improvement
            (4.0, 4.0, 5.0, 5.0, 0.0),    # Worse performance (capped at 0)
        ]

        for orig_bid, orig_ask, opt_bid, opt_ask, expected in test_cases:
            improvement = optimizer._calculate_improvement(orig_bid, orig_ask, opt_bid, opt_ask)
            assert abs(improvement - expected) < 0.1, f"Expected {expected}%, got {improvement}%"

    async def test_optimization_modes(
        self,
        quote_optimizer,
        sample_current_quotes,
        sample_market_data,
        sample_position_data
    ):
        """Test different optimization modes."""
        optimizer = quote_optimizer

        results = {}

        # Test different modes
        for mode in ['aggressive', 'conservative', 'adaptive']:
            optimizer.optimization_mode = mode

            result = await optimizer.optimize_quotes(
                symbol="SEI/USDC",
                current_quotes=sample_current_quotes,
                market_data=sample_market_data,
                position_data=sample_position_data
            )

            results[mode] = result

        # Verify mode differences
        aggressive = results['aggressive']
        conservative = results['conservative']
        results['adaptive']

        # Aggressive should generally have tighter spreads than conservative
        assert (aggressive.optimized_bid_spread_bps + aggressive.optimized_ask_spread_bps) <= \
               (conservative.optimized_bid_spread_bps + conservative.optimized_ask_spread_bps)

    async def test_performance_tracking(self, quote_optimizer):
        """Test optimization performance tracking."""
        optimizer = quote_optimizer

        # Run several optimizations
        for _i in range(5):
            await optimizer.optimize_quotes(
                symbol="SEI/USDC",
                current_quotes=[],
                market_data={'order_book': {'bids': [{'price': 0.5, 'size': 1000}], 'asks': [{'price': 0.501, 'size': 1000}]}},
                position_data={'base_balance': 0, 'recent_fills': 0, 'recent_volume': 0, 'recent_pnl': 0}
            )

        # Check performance metrics
        performance = optimizer.get_optimization_performance()

        assert performance['total_optimizations'] == 5
        assert 'average_improvement_pct' in performance
        assert 'success_rate' in performance
        assert performance['success_rate'] <= 1.0

    async def test_market_condition_analysis(self, quote_optimizer):
        """Test market condition analysis."""
        optimizer = quote_optimizer

        # Run optimization to generate history
        await optimizer.optimize_quotes(
            symbol="SEI/USDC",
            current_quotes=[],
            market_data={'order_book': {'bids': [{'price': 0.5, 'size': 1000}], 'asks': [{'price': 0.501, 'size': 1000}]}},
            position_data={'base_balance': 0, 'recent_fills': 0, 'recent_volume': 0, 'recent_pnl': 0}
        )

        # Get market condition analysis
        analysis = optimizer.get_market_condition_analysis("SEI/USDC")

        assert 'current_market_condition' in analysis
        assert 'confidence_score' in analysis
        assert 'recent_improvement_pct' in analysis

    async def test_parameter_calibration(self, quote_optimizer):
        """Test automatic parameter calibration."""
        optimizer = quote_optimizer

        # Set up test history with poor performance
        symbol = "SEI/USDC"
        optimizer.optimization_mode = "aggressive"

        # Simulate multiple optimizations with poor results
        for i in range(15):
            result = OptimizationResult(
                symbol=symbol,
                original_bid_spread_bps=10.0,
                original_ask_spread_bps=10.0,
                optimized_bid_spread_bps=9.5,
                optimized_ask_spread_bps=9.5,
                improvement_pct=5.0,  # Below threshold
                confidence_score=0.5,
                market_condition=MarketCondition.CALM,
                optimization_reason="Test",
                expected_fill_rate=0.3,
                expected_pnl_improvement=1.0,
                risk_adjustment=0.1,
                timestamp=datetime.now() - timedelta(minutes=i)
            )

            if symbol not in optimizer.optimization_history:
                from collections import deque
                optimizer.optimization_history[symbol] = deque(maxlen=100)
            optimizer.optimization_history[symbol].append(result)

        # Run calibration
        original_mode = optimizer.optimization_mode
        await optimizer.calibrate_optimization_parameters(symbol)

        # Should have switched to more conservative mode
        assert optimizer.optimization_mode != original_mode or optimizer.optimization_mode == "conservative"


@pytest.mark.performance
class TestOptimizationPerformance:
    """Performance tests for quote optimization."""

    @pytest.mark.asyncio
    async def test_optimization_latency(self, sample_current_quotes, sample_market_data, sample_position_data):
        """Test optimization latency meets 50ms requirement."""
        with patch('flashmm.trading.optimization.quote_optimizer.RedisClient') as mock_redis:
            mock_redis.return_value.initialize = AsyncMock()
            mock_redis.return_value.close = AsyncMock()
            mock_redis.return_value.set = AsyncMock()
            mock_redis.return_value.get = AsyncMock(return_value=None)
            mock_redis.return_value.keys = AsyncMock(return_value=[])

            optimizer = QuoteOptimizer()
            await optimizer.initialize()

            # Measure optimization time
            import time

            latencies = []
            for _i in range(10):
                start_time = time.perf_counter()

                await optimizer.optimize_quotes(
                    symbol="SEI/USDC",
                    current_quotes=sample_current_quotes,
                    market_data=sample_market_data,
                    position_data=sample_position_data
                )

                end_time = time.perf_counter()
                latency_ms = (end_time - start_time) * 1000
                latencies.append(latency_ms)

            await optimizer.cleanup()

            # Analyze latency
            avg_latency = sum(latencies) / len(latencies)
            max_latency = max(latencies)

            print("Optimization Latency Statistics:")
            print(f"  Average: {avg_latency:.1f}ms")
            print(f"  Maximum: {max_latency:.1f}ms")
            print("  Target: 50ms")

            # Performance requirements
            assert avg_latency <= 50.0, f"Average optimization latency {avg_latency:.1f}ms exceeds 50ms target"
            assert max_latency <= 100.0, f"Maximum optimization latency {max_latency:.1f}ms too high"


if __name__ == "__main__":
    # Run with: python -m pytest tests/unit/test_quote_optimizer.py -v
    pytest.main([__file__, "-v", "--tb=short"])

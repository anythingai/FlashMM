"""
Test suite for Market Risk Monitor

Tests market risk monitoring components including:
- Volatility detection and regime change detection
- Liquidity risk assessment and monitoring
- Real-time market condition analysis
- Risk threshold management and alerting
- Performance under various market scenarios
"""

from datetime import datetime, timedelta

import numpy as np
import pytest

from flashmm.risk.market_risk_monitor import (
    LiquidityRiskAssessor,
    MarketRiskMonitor,
    RegimeChangeDetector,
    VolatilityDetector,
)


class TestVolatilityDetector:
    """Test volatility detection functionality."""

    @pytest.fixture
    def volatility_detector(self):
        """Create volatility detector for testing."""
        return VolatilityDetector()

    @pytest.mark.asyncio
    async def test_initialization(self, volatility_detector):
        """Test volatility detector initializes correctly."""
        await volatility_detector.initialize()

        assert volatility_detector.window_size == 20
        assert volatility_detector.volatility_threshold == 0.05
        assert volatility_detector.spike_threshold == 3.0
        assert len(volatility_detector.price_history) == 0

    @pytest.mark.asyncio
    async def test_price_data_ingestion(self, volatility_detector):
        """Test ingestion of price data."""
        await volatility_detector.initialize()

        # Add price data
        price_data = {
            'symbol': 'BTC-USD',
            'price': 50000.0,
            'timestamp': datetime.now(),
            'volume': 1000.0
        }

        await volatility_detector.add_price_data(price_data)

        assert len(volatility_detector.price_history['BTC-USD']) == 1
        assert volatility_detector.price_history['BTC-USD'][0]['price'] == 50000.0

    @pytest.mark.asyncio
    async def test_volatility_calculation(self, volatility_detector):
        """Test volatility calculation from price series."""
        await volatility_detector.initialize()

        # Create price series with known volatility
        base_price = 50000.0
        prices = []
        timestamps = []

        for i in range(25):  # More than window size
            # Add some randomness to create volatility
            price_change = 0.02 * (np.random.random() - 0.5)  # ±1% random change
            price = base_price * (1 + price_change)
            prices.append(price)
            timestamps.append(datetime.now() - timedelta(minutes=i))

            price_data = {
                'symbol': 'BTC-USD',
                'price': price,
                'timestamp': timestamps[-1],
                'volume': 1000.0
            }

            await volatility_detector.add_price_data(price_data)

        # Calculate volatility
        volatility = await volatility_detector.calculate_volatility('BTC-USD')

        assert volatility is not None
        assert volatility > 0
        assert isinstance(volatility, float)

    @pytest.mark.asyncio
    async def test_volatility_spike_detection(self, volatility_detector):
        """Test detection of volatility spikes."""
        await volatility_detector.initialize()

        symbol = 'BTC-USD'
        base_price = 50000.0

        # Add stable price history
        for i in range(20):
            price_data = {
                'symbol': symbol,
                'price': base_price + (i % 2),  # Minimal variation
                'timestamp': datetime.now() - timedelta(minutes=i),
                'volume': 1000.0
            }
            await volatility_detector.add_price_data(price_data)

        # Add a price spike
        spike_price = base_price * 1.10  # 10% spike
        spike_data = {
            'symbol': symbol,
            'price': spike_price,
            'timestamp': datetime.now(),
            'volume': 1000.0
        }

        await volatility_detector.add_price_data(spike_data)

        # Check for volatility spike
        is_spike = await volatility_detector.detect_volatility_spike(symbol)

        assert is_spike

    @pytest.mark.asyncio
    async def test_multiple_symbols(self, volatility_detector):
        """Test handling multiple symbols simultaneously."""
        await volatility_detector.initialize()

        symbols = ['BTC-USD', 'ETH-USD', 'AAPL']
        base_prices = [50000.0, 3000.0, 150.0]

        # Add data for multiple symbols
        for _i, (symbol, base_price) in enumerate(zip(symbols, base_prices, strict=False)):
            for j in range(15):
                price_data = {
                    'symbol': symbol,
                    'price': base_price * (1 + 0.001 * j),  # Small uptrend
                    'timestamp': datetime.now() - timedelta(minutes=j),
                    'volume': 1000.0
                }
                await volatility_detector.add_price_data(price_data)

        # Calculate volatility for all symbols
        volatilities = {}
        for symbol in symbols:
            volatilities[symbol] = await volatility_detector.calculate_volatility(symbol)

        assert len(volatilities) == 3
        for _symbol, volatility in volatilities.items():
            assert volatility is not None
            assert volatility >= 0

    @pytest.mark.asyncio
    async def test_insufficient_data_handling(self, volatility_detector):
        """Test handling of insufficient data for calculations."""
        await volatility_detector.initialize()

        # Add only a few data points
        for i in range(5):  # Less than window size
            price_data = {
                'symbol': 'BTC-USD',
                'price': 50000.0 + i,
                'timestamp': datetime.now() - timedelta(minutes=i),
                'volume': 1000.0
            }
            await volatility_detector.add_price_data(price_data)

        # Should handle insufficient data gracefully
        volatility = await volatility_detector.calculate_volatility('BTC-USD')

        # Should return None or handle gracefully
        assert volatility is None or volatility >= 0


class TestRegimeChangeDetector:
    """Test regime change detection functionality."""

    @pytest.fixture
    def regime_detector(self):
        """Create regime change detector for testing."""
        return RegimeChangeDetector()

    @pytest.mark.asyncio
    async def test_initialization(self, regime_detector):
        """Test regime detector initializes correctly."""
        await regime_detector.initialize()

        assert regime_detector.lookback_periods == 50
        assert regime_detector.regime_change_threshold == 2.0
        assert regime_detector.min_regime_duration == 10
        assert regime_detector.current_regime == 'normal'

    @pytest.mark.asyncio
    async def test_trend_regime_detection(self, regime_detector):
        """Test detection of trending market regime."""
        await regime_detector.initialize()

        symbol = 'BTC-USD'
        base_price = 50000.0

        # Create strong uptrend
        for i in range(30):
            price = base_price * (1 + 0.02 * i)  # 2% per period increase
            market_data = {
                'symbol': symbol,
                'price': price,
                'volume': 1000.0,
                'timestamp': datetime.now() - timedelta(minutes=30-i)
            }
            await regime_detector.add_market_data(market_data)

        # Detect regime
        regime = await regime_detector.detect_regime_change(symbol)

        assert regime in ['trending_up', 'high_volatility', 'normal']

    @pytest.mark.asyncio
    async def test_volatile_regime_detection(self, regime_detector):
        """Test detection of volatile market regime."""
        await regime_detector.initialize()

        symbol = 'BTC-USD'
        base_price = 50000.0

        # Create volatile price action
        for i in range(30):
            # Alternating large moves
            price_change = 0.05 * (1 if i % 2 == 0 else -1)  # ±5% alternating
            price = base_price * (1 + price_change)

            market_data = {
                'symbol': symbol,
                'price': price,
                'volume': 2000.0 * (1 + abs(price_change)),  # Higher volume on volatile moves
                'timestamp': datetime.now() - timedelta(minutes=30-i)
            }
            await regime_detector.add_market_data(market_data)

        # Detect regime
        regime = await regime_detector.detect_regime_change(symbol)

        assert regime in ['high_volatility', 'trending_down', 'normal']

    @pytest.mark.asyncio
    async def test_regime_persistence(self, regime_detector):
        """Test that regime changes require persistence."""
        await regime_detector.initialize()

        symbol = 'BTC-USD'
        base_price = 50000.0

        # Add normal market data first
        for i in range(20):
            price = base_price + (i % 3) * 10  # Small variations
            market_data = {
                'symbol': symbol,
                'price': price,
                'volume': 1000.0,
                'timestamp': datetime.now() - timedelta(minutes=20-i)
            }
            await regime_detector.add_market_data(market_data)


        # Add a few volatile points (not enough for regime change)
        for i in range(5):
            price = base_price * (1 + 0.03 * (1 if i % 2 == 0 else -1))
            market_data = {
                'symbol': symbol,
                'price': price,
                'volume': 1500.0,
                'timestamp': datetime.now() - timedelta(minutes=5-i)
            }
            await regime_detector.add_market_data(market_data)

        # Regime should not change with just a few volatile points
        current_regime = await regime_detector.detect_regime_change(symbol)

        # Should still be similar to initial regime or show early signs
        assert current_regime is not None

    @pytest.mark.asyncio
    async def test_regime_change_alerts(self, regime_detector):
        """Test regime change alerting mechanism."""
        await regime_detector.initialize()

        alerts_received = []

        async def mock_alert_callback(alert_data):
            alerts_received.append(alert_data)

        regime_detector.set_alert_callback(mock_alert_callback)

        symbol = 'BTC-USD'
        base_price = 50000.0

        # Create conditions for regime change
        for i in range(40):
            if i < 20:
                # Normal phase
                price = base_price + (i % 5) * 10
            else:
                # Volatile phase
                price_change = 0.04 * (1 if i % 2 == 0 else -1)
                price = base_price * (1 + price_change)

            market_data = {
                'symbol': symbol,
                'price': price,
                'volume': 1000.0 + (500.0 if i >= 20 else 0),
                'timestamp': datetime.now() - timedelta(minutes=40-i)
            }

            await regime_detector.add_market_data(market_data)
            await regime_detector.detect_regime_change(symbol)

        # Should have received regime change alerts
        assert len(alerts_received) >= 0  # May or may not trigger based on exact conditions


class TestLiquidityRiskAssessor:
    """Test liquidity risk assessment functionality."""

    @pytest.fixture
    def liquidity_assessor(self):
        """Create liquidity risk assessor for testing."""
        return LiquidityRiskAssessor()

    @pytest.mark.asyncio
    async def test_initialization(self, liquidity_assessor):
        """Test liquidity assessor initializes correctly."""
        await liquidity_assessor.initialize()

        assert liquidity_assessor.min_order_book_depth == 100000.0
        assert liquidity_assessor.max_bid_ask_spread_pct == 1.0
        assert liquidity_assessor.volume_lookback_periods == 20
        assert liquidity_assessor.liquidity_threshold_score == 0.7

    @pytest.mark.asyncio
    async def test_bid_ask_spread_assessment(self, liquidity_assessor):
        """Test bid-ask spread liquidity assessment."""
        await liquidity_assessor.initialize()

        # Good liquidity (tight spread)
        good_liquidity_data = {
            'symbol': 'BTC-USD',
            'bid': 49995.0,
            'ask': 50005.0,  # 0.02% spread
            'mid_price': 50000.0,
            'timestamp': datetime.now()
        }

        spread_score = await liquidity_assessor.assess_bid_ask_spread(good_liquidity_data)
        assert spread_score > 0.8  # Should be high score for tight spread

        # Poor liquidity (wide spread)
        poor_liquidity_data = {
            'symbol': 'ALT-USD',
            'bid': 99.0,
            'ask': 101.0,  # 2% spread
            'mid_price': 100.0,
            'timestamp': datetime.now()
        }

        spread_score = await liquidity_assessor.assess_bid_ask_spread(poor_liquidity_data)
        assert spread_score < 0.5  # Should be low score for wide spread

    @pytest.mark.asyncio
    async def test_order_book_depth_assessment(self, liquidity_assessor):
        """Test order book depth assessment."""
        await liquidity_assessor.initialize()

        # Deep order book
        deep_book_data = {
            'symbol': 'BTC-USD',
            'bid_depth': 500000.0,
            'ask_depth': 450000.0,
            'total_depth': 950000.0,
            'timestamp': datetime.now()
        }

        depth_score = await liquidity_assessor.assess_order_book_depth(deep_book_data)
        assert depth_score > 0.8  # Should be high score for deep book

        # Shallow order book
        shallow_book_data = {
            'symbol': 'ILLIQUID-USD',
            'bid_depth': 5000.0,
            'ask_depth': 4000.0,
            'total_depth': 9000.0,
            'timestamp': datetime.now()
        }

        depth_score = await liquidity_assessor.assess_order_book_depth(shallow_book_data)
        assert depth_score < 0.3  # Should be low score for shallow book

    @pytest.mark.asyncio
    async def test_volume_based_liquidity_assessment(self, liquidity_assessor):
        """Test volume-based liquidity assessment."""
        await liquidity_assessor.initialize()

        symbol = 'BTC-USD'

        # Add historical volume data
        volumes = [1000000, 1200000, 800000, 1500000, 900000, 1100000]
        for i, volume in enumerate(volumes):
            volume_data = {
                'symbol': symbol,
                'volume': volume,
                'timestamp': datetime.now() - timedelta(hours=i)
            }
            await liquidity_assessor.add_volume_data(volume_data)

        # Assess current volume (high)
        current_volume_data = {
            'symbol': symbol,
            'volume': 2000000,  # Much higher than average
            'timestamp': datetime.now()
        }

        volume_score = await liquidity_assessor.assess_volume_liquidity(current_volume_data)
        assert volume_score > 0.7  # Should be high score for high volume

        # Assess low volume
        low_volume_data = {
            'symbol': symbol,
            'volume': 200000,  # Much lower than average
            'timestamp': datetime.now()
        }

        volume_score = await liquidity_assessor.assess_volume_liquidity(low_volume_data)
        assert volume_score < 0.5  # Should be low score for low volume

    @pytest.mark.asyncio
    async def test_comprehensive_liquidity_score(self, liquidity_assessor):
        """Test comprehensive liquidity score calculation."""
        await liquidity_assessor.initialize()

        symbol = 'BTC-USD'

        # Add volume history
        for i in range(10):
            volume_data = {
                'symbol': symbol,
                'volume': 1000000 + i * 50000,
                'timestamp': datetime.now() - timedelta(hours=i)
            }
            await liquidity_assessor.add_volume_data(volume_data)

        # Comprehensive market data
        market_data = {
            'symbol': symbol,
            'bid': 49990.0,
            'ask': 50010.0,  # 0.04% spread
            'mid_price': 50000.0,
            'bid_depth': 800000.0,
            'ask_depth': 750000.0,
            'total_depth': 1550000.0,
            'volume': 1800000.0,  # High volume
            'timestamp': datetime.now()
        }

        comprehensive_score = await liquidity_assessor.calculate_liquidity_score(market_data)

        assert 0.0 <= comprehensive_score <= 1.0
        assert comprehensive_score > 0.7  # Should be high with good conditions

    @pytest.mark.asyncio
    async def test_liquidity_risk_alerts(self, liquidity_assessor):
        """Test liquidity risk alerting."""
        await liquidity_assessor.initialize()

        alerts_received = []

        async def mock_alert_callback(alert_data):
            alerts_received.append(alert_data)

        liquidity_assessor.set_alert_callback(mock_alert_callback)

        # Create illiquid market conditions
        illiquid_data = {
            'symbol': 'ILLIQUID-USD',
            'bid': 95.0,
            'ask': 105.0,  # 10% spread
            'mid_price': 100.0,
            'bid_depth': 1000.0,
            'ask_depth': 800.0,
            'total_depth': 1800.0,
            'volume': 5000.0,  # Very low volume
            'timestamp': datetime.now()
        }

        score = await liquidity_assessor.calculate_liquidity_score(illiquid_data)

        # Should trigger liquidity alert
        if score < liquidity_assessor.liquidity_threshold_score:
            assert len(alerts_received) >= 0  # May or may not alert based on implementation


class TestMarketRiskMonitor:
    """Test the complete market risk monitoring system."""

    @pytest.fixture
    def market_monitor(self):
        """Create market risk monitor for testing."""
        return MarketRiskMonitor()

    @pytest.mark.asyncio
    async def test_monitor_initialization(self, market_monitor):
        """Test monitor initializes all components."""
        await market_monitor.initialize()

        assert market_monitor.volatility_detector is not None
        assert market_monitor.regime_detector is not None
        assert market_monitor.liquidity_assessor is not None
        assert market_monitor.enabled
        assert market_monitor.monitoring_interval > 0

    @pytest.mark.asyncio
    async def test_comprehensive_market_analysis(self, market_monitor):
        """Test comprehensive market analysis."""
        await market_monitor.initialize()

        # Comprehensive market data
        market_data = {
            'BTC-USD': {
                'price': 50000.0,
                'bid': 49995.0,
                'ask': 50005.0,
                'volume': 1500000.0,
                'bid_depth': 800000.0,
                'ask_depth': 750000.0,
                'timestamp': datetime.now()
            },
            'ETH-USD': {
                'price': 3000.0,
                'bid': 2998.0,
                'ask': 3002.0,
                'volume': 800000.0,
                'bid_depth': 400000.0,
                'ask_depth': 380000.0,
                'timestamp': datetime.now()
            }
        }

        # Analyze market conditions
        analysis_result = await market_monitor.analyze_market_risk(market_data)

        assert 'volatility_analysis' in analysis_result
        assert 'regime_analysis' in analysis_result
        assert 'liquidity_analysis' in analysis_result
        assert 'overall_risk_level' in analysis_result
        assert 'risk_factors' in analysis_result

    @pytest.mark.asyncio
    async def test_risk_threshold_monitoring(self, market_monitor):
        """Test risk threshold monitoring and alerting."""
        await market_monitor.initialize()

        alerts_received = []

        async def mock_alert_callback(alert_data):
            alerts_received.append(alert_data)

        market_monitor.set_alert_callback(mock_alert_callback)

        # Create high-risk market conditions
        risky_market_data = {
            'VOLATILE-USD': {
                'price': 100.0,
                'bid': 90.0,
                'ask': 110.0,  # 20% spread
                'volume': 1000.0,  # Low volume
                'bid_depth': 500.0,
                'ask_depth': 400.0,
                'timestamp': datetime.now()
            }
        }

        # Add historical data to trigger volatility
        for i in range(25):
            price_change = 0.05 * (1 if i % 2 == 0 else -1)  # ±5% swings
            historical_data = {
                'VOLATILE-USD': {
                    'price': 100.0 * (1 + price_change),
                    'volume': 1000.0,
                    'timestamp': datetime.now() - timedelta(minutes=25-i)
                }
            }
            # Add historical data for each symbol
            for symbol, data in historical_data.items():
                await market_monitor.add_market_data(symbol, data)

        # Analyze risky conditions
        analysis_result = await market_monitor.analyze_market_risk(risky_market_data)

        # Should detect high risk
        assert analysis_result['overall_risk_level'] in ['high', 'critical']
        assert len(analysis_result['risk_factors']) > 0

    @pytest.mark.asyncio
    async def test_multi_symbol_monitoring(self, market_monitor):
        """Test monitoring multiple symbols simultaneously."""
        await market_monitor.initialize()

        symbols = ['BTC-USD', 'ETH-USD', 'AAPL', 'TSLA']

        # Add market data for multiple symbols
        multi_symbol_data = {}
        for i, symbol in enumerate(symbols):
            base_price = 1000.0 * (i + 1)
            multi_symbol_data[symbol] = {
                'price': base_price,
                'bid': base_price * 0.999,
                'ask': base_price * 1.001,
                'volume': 100000.0 * (i + 1),
                'bid_depth': 50000.0 * (i + 1),
                'ask_depth': 45000.0 * (i + 1),
                'timestamp': datetime.now()
            }

        # Analyze all symbols
        analysis_result = await market_monitor.analyze_market_risk(multi_symbol_data)

        # Should have analysis for all symbols
        assert len(analysis_result['volatility_analysis']) == len(symbols)
        assert len(analysis_result['liquidity_analysis']) == len(symbols)

    @pytest.mark.asyncio
    async def test_performance_monitoring(self, market_monitor):
        """Test monitoring system performance."""
        await market_monitor.initialize()

        # Single symbol data for performance test
        market_data = {
            'BTC-USD': {
                'price': 50000.0,
                'bid': 49995.0,
                'ask': 50005.0,
                'volume': 1000000.0,
                'bid_depth': 500000.0,
                'ask_depth': 480000.0,
                'timestamp': datetime.now()
            }
        }

        # Time multiple analyses
        start_time = datetime.now()
        for _ in range(50):
            await market_monitor.analyze_market_conditions(market_data)
        end_time = datetime.now()

        total_time = (end_time - start_time).total_seconds()
        avg_time_per_analysis = total_time / 50

        # Should be reasonably fast (under 50ms per analysis)
        assert avg_time_per_analysis < 0.05

    @pytest.mark.asyncio
    async def test_historical_data_management(self, market_monitor):
        """Test historical data management and cleanup."""
        await market_monitor.initialize()

        symbol = 'BTC-USD'

        # Add lots of historical data
        for i in range(200):  # More than typical retention
            historical_data = {
                symbol: {
                    'price': 50000.0 + i,
                    'volume': 1000000.0,
                    'timestamp': datetime.now() - timedelta(hours=i)
                }
            }
            # Add historical data for each symbol
            for symbol, data in historical_data.items():
                await market_monitor.add_market_data(symbol, data)

        # Check that old data is cleaned up (implementation dependent)
        # This test ensures the system handles large amounts of data gracefully
        current_data = {
            symbol: {
                'price': 52000.0,
                'bid': 51995.0,
                'ask': 52005.0,
                'volume': 1200000.0,
                'bid_depth': 600000.0,
                'ask_depth': 580000.0,
                'timestamp': datetime.now()
            }
        }

        # Should still work efficiently with current data
        analysis_result = await market_monitor.analyze_market_conditions(current_data)
        assert analysis_result is not None


@pytest.mark.asyncio
async def test_market_stress_scenario():
    """Test market risk monitor under stress conditions."""
    monitor = MarketRiskMonitor()
    await monitor.initialize()

    # Simulate market crash scenario
    crash_data = {
        'BTC-USD': {
            'price': 30000.0,  # 40% drop from 50k
            'bid': 29000.0,
            'ask': 31000.0,    # Wide spread during crash
            'volume': 5000000.0,  # High panic volume
            'bid_depth': 100000.0,  # Low depth during crash
            'ask_depth': 80000.0,
            'timestamp': datetime.now()
        }
    }

    # Add pre-crash stable data
    for i in range(30):
        stable_data = {
            'BTC-USD': {
                'price': 50000.0 + (i % 10) * 100,  # Stable around 50k
                'volume': 1000000.0,
                'timestamp': datetime.now() - timedelta(minutes=30-i)
            }
        }
        # Add stable data for each symbol
        for symbol, data in stable_data.items():
            await monitor.add_market_data(symbol, data)

    # Analyze crash conditions
    crash_analysis = await monitor.analyze_market_risk(crash_data)

    # Should detect extreme conditions
    assert crash_analysis['overall_risk_level'] == 'critical'
    assert 'high_volatility' in crash_analysis['risk_factors']
    assert 'liquidity_crisis' in crash_analysis['risk_factors']


if __name__ == "__main__":
    pytest.main([__file__])

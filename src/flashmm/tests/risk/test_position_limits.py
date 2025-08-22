"""
Test suite for Position Limits Manager

Tests position limit management, dynamic limit calculation, and concentration risk monitoring:
- Position limit enforcement
- Dynamic limit adjustments based on volatility
- Concentration risk detection and monitoring
- Real-time position tracking and validation
- Integration with market conditions
"""

from datetime import datetime
from decimal import Decimal
from unittest.mock import patch

import pytest

from flashmm.risk.position_limits import (
    ConcentrationRiskMonitor,
    DynamicLimitCalculator,
    PositionLimitsManager,
)


class TestDynamicLimitCalculator:
    """Test dynamic limit calculation based on market conditions."""

    @pytest.fixture
    def limit_calculator(self):
        """Create limit calculator for testing."""
        return DynamicLimitCalculator()

    @pytest.mark.asyncio
    async def test_initialization(self, limit_calculator):
        """Test calculator initializes with default settings."""
        await limit_calculator.initialize()

        assert limit_calculator.base_limit_pct == 2.0
        assert limit_calculator.volatility_adjustment_factor == 0.5
        assert limit_calculator.liquidity_adjustment_factor == 0.3
        assert limit_calculator.max_limit_pct == 5.0
        assert limit_calculator.min_limit_pct == 0.5

    @pytest.mark.asyncio
    async def test_base_limit_calculation(self, limit_calculator):
        """Test base limit calculation."""
        await limit_calculator.initialize()

        portfolio_value = Decimal('100000')  # $100k portfolio

        # Base limit should be 2% of portfolio
        base_limit = await limit_calculator.calculate_base_limit(portfolio_value)
        expected_limit = portfolio_value * Decimal('0.02')

        assert base_limit == expected_limit
        assert base_limit == Decimal('2000')

    @pytest.mark.asyncio
    async def test_volatility_adjustment(self, limit_calculator):
        """Test volatility-based limit adjustments."""
        await limit_calculator.initialize()

        base_limit = Decimal('2000')

        # Low volatility should increase limits
        low_vol_data = {'volatility': 0.01, 'symbol': 'BTC-USD'}  # 1% volatility
        adjusted_limit = await limit_calculator.adjust_for_volatility(base_limit, low_vol_data)
        assert adjusted_limit > base_limit

        # High volatility should decrease limits
        high_vol_data = {'volatility': 0.05, 'symbol': 'BTC-USD'}  # 5% volatility
        adjusted_limit = await limit_calculator.adjust_for_volatility(base_limit, high_vol_data)
        assert adjusted_limit < base_limit

        # Normal volatility should keep limits similar
        normal_vol_data = {'volatility': 0.02, 'symbol': 'BTC-USD'}  # 2% volatility
        adjusted_limit = await limit_calculator.adjust_for_volatility(base_limit, normal_vol_data)
        assert abs(adjusted_limit - base_limit) < base_limit * Decimal('0.1')  # Within 10%

    @pytest.mark.asyncio
    async def test_liquidity_adjustment(self, limit_calculator):
        """Test liquidity-based limit adjustments."""
        await limit_calculator.initialize()

        base_limit = Decimal('2000')

        # High liquidity should allow higher limits
        high_liquidity_data = {
            'bid_ask_spread_pct': 0.001,  # 0.1% spread
            'order_book_depth': 1000000,
            'symbol': 'BTC-USD'
        }
        adjusted_limit = await limit_calculator.adjust_for_liquidity(base_limit, high_liquidity_data)
        assert adjusted_limit >= base_limit

        # Low liquidity should reduce limits
        low_liquidity_data = {
            'bid_ask_spread_pct': 0.01,   # 1% spread
            'order_book_depth': 10000,
            'symbol': 'ALT-USD'
        }
        adjusted_limit = await limit_calculator.adjust_for_liquidity(base_limit, low_liquidity_data)
        assert adjusted_limit < base_limit

    @pytest.mark.asyncio
    async def test_combined_adjustments(self, limit_calculator):
        """Test combined volatility and liquidity adjustments."""
        await limit_calculator.initialize()

        portfolio_value = Decimal('100000')
        market_data = {
            'volatility': 0.03,           # Moderate volatility
            'bid_ask_spread_pct': 0.005,  # Moderate spread
            'order_book_depth': 500000,   # Good depth
            'symbol': 'ETH-USD'
        }

        final_limit = await limit_calculator.calculate_dynamic_limit(portfolio_value, market_data)

        # Should be a reasonable limit
        assert final_limit > Decimal('500')   # Above minimum
        assert final_limit < Decimal('5000')  # Below maximum
        assert isinstance(final_limit, Decimal)

    @pytest.mark.asyncio
    async def test_limit_bounds_enforcement(self, limit_calculator):
        """Test that limits are enforced within bounds."""
        await limit_calculator.initialize()

        portfolio_value = Decimal('100000')

        # Extreme market conditions
        extreme_conditions = [
            {
                'volatility': 0.001,         # Very low volatility
                'bid_ask_spread_pct': 0.0001, # Very tight spread
                'order_book_depth': 10000000, # Very deep
                'symbol': 'STABLE-USD'
            },
            {
                'volatility': 0.20,          # Very high volatility
                'bid_ask_spread_pct': 0.05,  # Very wide spread
                'order_book_depth': 1000,    # Very shallow
                'symbol': 'VOLATILE-USD'
            }
        ]

        for conditions in extreme_conditions:
            limit = await limit_calculator.calculate_dynamic_limit(portfolio_value, conditions)

            # Should be within bounds
            max_allowed = portfolio_value * Decimal(str(limit_calculator.max_limit_pct / 100))
            min_allowed = portfolio_value * Decimal(str(limit_calculator.min_limit_pct / 100))

            assert limit <= max_allowed
            assert limit >= min_allowed


class TestConcentrationRiskMonitor:
    """Test concentration risk monitoring functionality."""

    @pytest.fixture
    def concentration_monitor(self):
        """Create concentration monitor for testing."""
        return ConcentrationRiskMonitor()

    @pytest.mark.asyncio
    async def test_initialization(self, concentration_monitor):
        """Test monitor initializes correctly."""
        await concentration_monitor.initialize()

        assert concentration_monitor.max_single_position_pct == 10.0
        assert concentration_monitor.max_sector_concentration_pct == 25.0
        assert concentration_monitor.max_correlated_exposure_pct == 15.0
        assert concentration_monitor.correlation_threshold == 0.7

    @pytest.mark.asyncio
    async def test_single_position_concentration(self, concentration_monitor):
        """Test single position concentration calculations."""
        await concentration_monitor.initialize()

        portfolio_value = Decimal('100000')
        positions = [
            {'symbol': 'BTC-USD', 'notional_value': Decimal('8000'), 'sector': 'crypto'},
            {'symbol': 'ETH-USD', 'notional_value': Decimal('5000'), 'sector': 'crypto'},
            {'symbol': 'AAPL', 'notional_value': Decimal('3000'), 'sector': 'tech'},
        ]

        concentration_data = await concentration_monitor.calculate_concentration_risk(
            positions, portfolio_value
        )

        # Check single position concentrations
        assert 'single_position_risk' in concentration_data
        single_risks = concentration_data['single_position_risk']

        # BTC should be 8% concentration
        btc_risk = next(r for r in single_risks if r['symbol'] == 'BTC-USD')
        assert abs(btc_risk['concentration_pct'] - 8.0) < 0.01
        assert not btc_risk['exceeds_limit']  # Under 10% limit

        # Test position that exceeds limit
        large_position = {'symbol': 'LARGE-USD', 'notional_value': Decimal('12000'), 'sector': 'other'}
        positions.append(large_position)

        concentration_data = await concentration_monitor.calculate_concentration_risk(
            positions, portfolio_value
        )

        large_risk = next(r for r in concentration_data['single_position_risk'] if r['symbol'] == 'LARGE-USD')
        assert large_risk['exceeds_limit']  # Should exceed 10% limit

    @pytest.mark.asyncio
    async def test_sector_concentration(self, concentration_monitor):
        """Test sector concentration calculations."""
        await concentration_monitor.initialize()

        portfolio_value = Decimal('100000')
        positions = [
            {'symbol': 'BTC-USD', 'notional_value': Decimal('15000'), 'sector': 'crypto'},
            {'symbol': 'ETH-USD', 'notional_value': Decimal('12000'), 'sector': 'crypto'},
            {'symbol': 'ADA-USD', 'notional_value': Decimal('8000'), 'sector': 'crypto'},
            {'symbol': 'AAPL', 'notional_value': Decimal('5000'), 'sector': 'tech'},
        ]

        concentration_data = await concentration_monitor.calculate_concentration_risk(
            positions, portfolio_value
        )

        # Check sector concentrations
        assert 'sector_concentration_risk' in concentration_data
        sector_risks = concentration_data['sector_concentration_risk']

        # Crypto sector should be 35% (15k + 12k + 8k = 35k out of 100k)
        crypto_risk = next(r for r in sector_risks if r['sector'] == 'crypto')
        assert abs(crypto_risk['concentration_pct'] - 35.0) < 0.01
        assert crypto_risk['exceeds_limit']  # Should exceed 25% limit

        # Tech sector should be 5%
        tech_risk = next(r for r in sector_risks if r['sector'] == 'tech')
        assert abs(tech_risk['concentration_pct'] - 5.0) < 0.01
        assert not tech_risk['exceeds_limit']  # Under 25% limit

    @pytest.mark.asyncio
    async def test_correlation_risk_detection(self, concentration_monitor):
        """Test correlated exposure detection."""
        await concentration_monitor.initialize()

        # Mock correlation calculation
        with patch.object(concentration_monitor, '_calculate_correlation', return_value=0.85):
            portfolio_value = Decimal('100000')
            positions = [
                {'symbol': 'BTC-USD', 'notional_value': Decimal('10000'), 'sector': 'crypto'},
                {'symbol': 'ETH-USD', 'notional_value': Decimal('8000'), 'sector': 'crypto'},
            ]

            concentration_data = await concentration_monitor.calculate_concentration_risk(
                positions, portfolio_value
            )

            # Should detect high correlation risk
            assert 'correlation_risk' in concentration_data
            correlation_risks = concentration_data['correlation_risk']

            # Should find BTC-ETH pair with high correlation
            btc_eth_risk = next(
                (r for r in correlation_risks
                 if {r['symbol1'], r['symbol2']} == {'BTC-USD', 'ETH-USD'}),
                None
            )

            assert btc_eth_risk is not None
            assert btc_eth_risk['correlation'] == 0.85
            assert btc_eth_risk['combined_exposure_pct'] == 18.0  # 10k + 8k = 18k out of 100k
            assert btc_eth_risk['exceeds_limit']  # Should exceed 15% limit for correlated assets

    @pytest.mark.asyncio
    async def test_overall_risk_assessment(self, concentration_monitor):
        """Test overall concentration risk assessment."""
        await concentration_monitor.initialize()

        portfolio_value = Decimal('100000')

        # Balanced portfolio
        balanced_positions = [
            {'symbol': 'BTC-USD', 'notional_value': Decimal('5000'), 'sector': 'crypto'},
            {'symbol': 'AAPL', 'notional_value': Decimal('5000'), 'sector': 'tech'},
            {'symbol': 'SPY', 'notional_value': Decimal('5000'), 'sector': 'etf'},
            {'symbol': 'GLD', 'notional_value': Decimal('5000'), 'sector': 'commodity'},
        ]

        balanced_data = await concentration_monitor.calculate_concentration_risk(
            balanced_positions, portfolio_value
        )

        assert balanced_data['overall_risk_level'] == 'low'
        assert not balanced_data['has_concentration_violations']

        # Concentrated portfolio
        concentrated_positions = [
            {'symbol': 'BTC-USD', 'notional_value': Decimal('30000'), 'sector': 'crypto'},
            {'symbol': 'ETH-USD', 'notional_value': Decimal('20000'), 'sector': 'crypto'},
            {'symbol': 'ADA-USD', 'notional_value': Decimal('15000'), 'sector': 'crypto'},
        ]

        concentrated_data = await concentration_monitor.calculate_concentration_risk(
            concentrated_positions, portfolio_value
        )

        assert concentrated_data['overall_risk_level'] in ['high', 'critical']
        assert concentrated_data['has_concentration_violations']


class TestPositionLimitsManager:
    """Test the complete position limits management system."""

    @pytest.fixture
    def position_manager(self):
        """Create position limits manager for testing."""
        return PositionLimitsManager()

    @pytest.mark.asyncio
    async def test_manager_initialization(self, position_manager):
        """Test manager initializes all components."""
        await position_manager.initialize()

        assert position_manager.limit_calculator is not None
        assert position_manager.concentration_monitor is not None
        assert position_manager.enabled
        assert position_manager.last_update is not None

    @pytest.mark.asyncio
    async def test_position_validation_pass(self, position_manager):
        """Test position validation that should pass."""
        await position_manager.initialize()

        # Mock portfolio value
        position_manager.current_portfolio_value = Decimal('100000')

        # Small position that should be allowed
        position_request = {
            'symbol': 'BTC-USD',
            'side': 'buy',
            'size': Decimal('0.05'),  # Small size
            'price': Decimal('50000'),
            'notional_value': Decimal('2500'),  # 2.5% of portfolio
            'sector': 'crypto'
        }

        is_valid, reason = await position_manager.validate_position(position_request)
        assert is_valid
        assert reason is None

    @pytest.mark.asyncio
    async def test_position_validation_fail_size(self, position_manager):
        """Test position validation that fails due to size."""
        await position_manager.initialize()

        position_manager.current_portfolio_value = Decimal('100000')

        # Large position that should be rejected
        large_position = {
            'symbol': 'BTC-USD',
            'side': 'buy',
            'size': Decimal('0.5'),   # Large size
            'price': Decimal('50000'),
            'notional_value': Decimal('25000'),  # 25% of portfolio
            'sector': 'crypto'
        }

        is_valid, reason = await position_manager.validate_position(large_position)
        assert not is_valid
        assert 'position limit' in reason.lower()

    @pytest.mark.asyncio
    async def test_position_validation_fail_concentration(self, position_manager):
        """Test position validation that fails due to concentration."""
        await position_manager.initialize()

        position_manager.current_portfolio_value = Decimal('100000')

        # Add existing positions
        existing_positions = [
            {'symbol': 'BTC-USD', 'notional_value': Decimal('8000'), 'sector': 'crypto'},
            {'symbol': 'ETH-USD', 'notional_value': Decimal('7000'), 'sector': 'crypto'},
            {'symbol': 'ADA-USD', 'notional_value': Decimal('6000'), 'sector': 'crypto'},
        ]

        position_manager.current_positions = existing_positions

        # New crypto position that would create concentration risk
        new_crypto_position = {
            'symbol': 'DOT-USD',
            'side': 'buy',
            'size': Decimal('1000'),
            'price': Decimal('10'),
            'notional_value': Decimal('10000'),  # 10% individual, but 31% sector total
            'sector': 'crypto'
        }

        is_valid, reason = await position_manager.validate_position(new_crypto_position)
        assert not is_valid
        assert 'concentration' in reason.lower()

    @pytest.mark.asyncio
    async def test_dynamic_limit_updates(self, position_manager):
        """Test dynamic limit updates based on market conditions."""
        await position_manager.initialize()

        position_manager.current_portfolio_value = Decimal('100000')

        # Normal market conditions
        normal_market_data = {
            'BTC-USD': {
                'volatility': 0.02,
                'bid_ask_spread_pct': 0.001,
                'order_book_depth': 1000000,
                'symbol': 'BTC-USD'
            }
        }

        limits_before = await position_manager.get_current_limits()

        # Update with new market data
        await position_manager.update_limits(normal_market_data)

        limits_after = await position_manager.get_current_limits()

        # Limits should be updated
        assert limits_after != limits_before
        assert 'BTC-USD' in limits_after
        assert limits_after['BTC-USD'] > Decimal('0')

    @pytest.mark.asyncio
    async def test_portfolio_monitoring(self, position_manager):
        """Test continuous portfolio monitoring."""
        await position_manager.initialize()

        # Mock positions and portfolio value
        test_positions = [
            {'symbol': 'BTC-USD', 'notional_value': Decimal('15000'), 'sector': 'crypto'},
            {'symbol': 'AAPL', 'notional_value': Decimal('8000'), 'sector': 'tech'},
            {'symbol': 'TSLA', 'notional_value': Decimal('12000'), 'sector': 'tech'},
        ]

        portfolio_value = Decimal('100000')

        # Check portfolio
        portfolio_status = await position_manager.check_portfolio_limits(
            test_positions, portfolio_value
        )

        assert 'position_limit_violations' in portfolio_status
        assert 'concentration_violations' in portfolio_status
        assert 'overall_risk_level' in portfolio_status
        assert 'total_exposure_pct' in portfolio_status

        # Total exposure should be 35%
        assert abs(portfolio_status['total_exposure_pct'] - 35.0) < 0.01

    @pytest.mark.asyncio
    async def test_limit_adjustment_scenarios(self, position_manager):
        """Test various limit adjustment scenarios."""
        await position_manager.initialize()

        position_manager.current_portfolio_value = Decimal('100000')

        # Test different market scenarios
        scenarios = [
            {
                'name': 'high_volatility',
                'market_data': {
                    'BTC-USD': {
                        'volatility': 0.08,  # Very high volatility
                        'bid_ask_spread_pct': 0.002,
                        'order_book_depth': 500000,
                        'symbol': 'BTC-USD'
                    }
                },
                'expected_change': 'decrease'
            },
            {
                'name': 'low_volatility',
                'market_data': {
                    'BTC-USD': {
                        'volatility': 0.005,  # Very low volatility
                        'bid_ask_spread_pct': 0.0005,
                        'order_book_depth': 2000000,
                        'symbol': 'BTC-USD'
                    }
                },
                'expected_change': 'increase'
            },
            {
                'name': 'poor_liquidity',
                'market_data': {
                    'ALT-USD': {
                        'volatility': 0.02,
                        'bid_ask_spread_pct': 0.02,  # Very wide spread
                        'order_book_depth': 10000,   # Very shallow
                        'symbol': 'ALT-USD'
                    }
                },
                'expected_change': 'decrease'
            }
        ]

        base_limits = await position_manager.get_current_limits()

        for scenario in scenarios:
            await position_manager.update_limits(scenario['market_data'])
            new_limits = await position_manager.get_current_limits()

            symbol = list(scenario['market_data'].keys())[0]

            if scenario['expected_change'] == 'decrease':
                # Limit should be lower than base for this symbol
                assert new_limits.get(symbol, Decimal('0')) <= base_limits.get(symbol, Decimal('999999'))
            elif scenario['expected_change'] == 'increase':
                # Limit should be higher than base for this symbol
                assert new_limits.get(symbol, Decimal('0')) >= base_limits.get(symbol, Decimal('0'))

    @pytest.mark.asyncio
    async def test_emergency_limit_reduction(self, position_manager):
        """Test emergency limit reduction functionality."""
        await position_manager.initialize()

        # Normal limits
        normal_limits = await position_manager.get_current_limits()

        # Trigger emergency reduction
        await position_manager.emergency_limit_reduction(
            reduction_factor=0.5,  # 50% reduction
            reason="Market stress test"
        )

        # Check that limits are reduced
        emergency_limits = await position_manager.get_current_limits()

        for symbol, limit in emergency_limits.items():
            if symbol in normal_limits:
                expected_reduced_limit = normal_limits[symbol] * Decimal('0.5')
                assert abs(limit - expected_reduced_limit) < Decimal('0.01')

    @pytest.mark.asyncio
    async def test_performance_monitoring(self, position_manager):
        """Test position manager performance."""
        await position_manager.initialize()

        position_manager.current_portfolio_value = Decimal('100000')

        # Time multiple validations
        test_position = {
            'symbol': 'BTC-USD',
            'side': 'buy',
            'size': Decimal('0.02'),
            'price': Decimal('50000'),
            'notional_value': Decimal('1000'),
            'sector': 'crypto'
        }

        start_time = datetime.now()
        for _ in range(100):
            await position_manager.validate_position(test_position)
        end_time = datetime.now()

        total_time = (end_time - start_time).total_seconds()
        avg_time_per_validation = total_time / 100

        # Should be very fast (under 5ms per validation)
        assert avg_time_per_validation < 0.005


@pytest.mark.asyncio
async def test_integration_with_concentration_monitor():
    """Test integration between position limits and concentration monitoring."""
    manager = PositionLimitsManager()
    await manager.initialize()

    manager.current_portfolio_value = Decimal('100000')
    manager.current_positions = {
        'BTC-USD': {'notional_value': Decimal('20000'), 'sector': 'crypto'},
        'ETH-USD': {'notional_value': Decimal('15000'), 'sector': 'crypto'},
    }

    # New position that would create concentration risk
    risky_position = {
        'symbol': 'LTC-USD',
        'side': 'buy',
        'size': Decimal('100'),
        'price': Decimal('100'),
        'notional_value': Decimal('10000'),  # Would make crypto 45% of portfolio
        'sector': 'crypto'
    }

    validation_result = await manager.validate_position(
        symbol=risky_position['symbol'],
        side=risky_position['side'],
        size=float(risky_position['size']),
        price=float(risky_position['price'])
    )
    is_valid = validation_result['allowed']
    reason = validation_result.get('violations', ['concentration risk'])[0] if not is_valid else None

    # Should be rejected due to concentration risk
    assert not is_valid
    assert reason and ('concentration' in reason.lower() or 'sector' in reason.lower())


if __name__ == "__main__":
    pytest.main([__file__])

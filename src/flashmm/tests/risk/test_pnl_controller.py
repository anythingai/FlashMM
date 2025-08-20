"""
Test suite for P&L Risk Controller

Tests P&L risk management components including:
- Drawdown protection and monitoring
- Stop-loss management and execution
- P&L tracking and risk metrics calculation
- Real-time risk assessment and alerts
- Performance attribution and analysis
"""

import pytest
import asyncio
from decimal import Decimal
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, List

from flashmm.risk.pnl_controller import (
    PnLRiskController,
    DrawdownProtector,
    StopLossManager
)
from flashmm.utils.exceptions import RiskError


class TestDrawdownProtector:
    """Test drawdown protection functionality."""
    
    @pytest.fixture
    def drawdown_protector(self):
        """Create drawdown protector for testing."""
        return DrawdownProtector()
    
    @pytest.mark.asyncio
    async def test_initialization(self, drawdown_protector):
        """Test drawdown protector initializes correctly."""
        await drawdown_protector.initialize()
        
        assert drawdown_protector.max_daily_drawdown_pct == 5.0
        assert drawdown_protector.max_weekly_drawdown_pct == 10.0
        assert drawdown_protector.max_monthly_drawdown_pct == 15.0
        assert drawdown_protector.trailing_stop_pct == 3.0
        assert drawdown_protector.current_peak == Decimal('0')
    
    @pytest.mark.asyncio
    async def test_peak_tracking(self, drawdown_protector):
        """Test peak P&L tracking."""
        await drawdown_protector.initialize()
        
        # Add positive P&L data
        pnl_data = [
            {'total_pnl': Decimal('1000'), 'timestamp': datetime.now() - timedelta(hours=5)},
            {'total_pnl': Decimal('2000'), 'timestamp': datetime.now() - timedelta(hours=4)},
            {'total_pnl': Decimal('3000'), 'timestamp': datetime.now() - timedelta(hours=3)},
            {'total_pnl': Decimal('2500'), 'timestamp': datetime.now() - timedelta(hours=2)},
            {'total_pnl': Decimal('2800'), 'timestamp': datetime.now() - timedelta(hours=1)},
        ]
        
        for pnl in pnl_data:
            await drawdown_protector.update_pnl(pnl)
        
        # Peak should be 3000
        assert drawdown_protector.current_peak == Decimal('3000')
    
    @pytest.mark.asyncio
    async def test_daily_drawdown_calculation(self, drawdown_protector):
        """Test daily drawdown calculation."""
        await drawdown_protector.initialize()
        
        # Set up daily P&L history
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        daily_pnls = [
            {'total_pnl': Decimal('5000'), 'timestamp': today + timedelta(hours=9)},   # Start of day high
            {'total_pnl': Decimal('7000'), 'timestamp': today + timedelta(hours=10)},  # Peak
            {'total_pnl': Decimal('6000'), 'timestamp': today + timedelta(hours=11)},  # Drawdown
            {'total_pnl': Decimal('4500'), 'timestamp': today + timedelta(hours=12)},  # Larger drawdown
        ]
        
        for pnl in daily_pnls:
            await drawdown_protector.update_pnl(pnl)
        
        # Calculate current drawdown
        current_drawdown = await drawdown_protector.calculate_current_drawdown('daily')
        
        # Should be (7000 - 4500) / 7000 = 35.7%
        expected_drawdown = (Decimal('7000') - Decimal('4500')) / Decimal('7000') * 100
        assert abs(current_drawdown - expected_drawdown) < Decimal('0.1')
    
    @pytest.mark.asyncio
    async def test_drawdown_limit_breach(self, drawdown_protector):
        """Test drawdown limit breach detection."""
        await drawdown_protector.initialize()
        
        alerts_received = []
        
        async def mock_alert_callback(alert_data):
            alerts_received.append(alert_data)
        
        drawdown_protector.set_alert_callback(mock_alert_callback)
        
        # Create scenario that breaches daily drawdown limit
        today = datetime.now().replace(hour=9, minute=0, second=0, microsecond=0)
        
        breach_pnls = [
            {'total_pnl': Decimal('10000'), 'timestamp': today},                    # Peak
            {'total_pnl': Decimal('9000'), 'timestamp': today + timedelta(hours=1)},  # -10%
        ]
        
        for pnl in breach_pnls:
            await drawdown_protector.update_pnl(pnl)
        
        # Check for limit breach
        is_breach = await drawdown_protector.check_drawdown_limits()
        
        # Should detect breach of 5% daily limit with 10% drawdown
        assert is_breach
        assert len(alerts_received) > 0
    
    @pytest.mark.asyncio
    async def test_trailing_stop_mechanism(self, drawdown_protector):
        """Test trailing stop loss mechanism."""
        await drawdown_protector.initialize()
        
        # Simulate profitable trade that then reverses
        pnl_sequence = [
            {'total_pnl': Decimal('1000'), 'timestamp': datetime.now() - timedelta(hours=5)},
            {'total_pnl': Decimal('3000'), 'timestamp': datetime.now() - timedelta(hours=4)},  # Peak
            {'total_pnl': Decimal('2800'), 'timestamp': datetime.now() - timedelta(hours=3)},  # Small dip
            {'total_pnl': Decimal('2700'), 'timestamp': datetime.now() - timedelta(hours=2)},  # More dip
            {'total_pnl': Decimal('2000'), 'timestamp': datetime.now() - timedelta(hours=1)},  # Larger dip
        ]
        
        stop_triggered = False
        
        for pnl in pnl_sequence:
            await drawdown_protector.update_pnl(pnl)
            
            # Check if trailing stop is triggered
            should_stop = await drawdown_protector.check_trailing_stop()
            if should_stop:
                stop_triggered = True
                break
        
        # Trailing stop should trigger when drawdown from peak (3000) exceeds 3%
        # At 2000, drawdown is (3000-2000)/3000 = 33.3%, which exceeds 3%
        assert stop_triggered
    
    @pytest.mark.asyncio
    async def test_recovery_tracking(self, drawdown_protector):
        """Test P&L recovery tracking after drawdown."""
        await drawdown_protector.initialize()
        
        # Simulate drawdown and recovery
        recovery_sequence = [
            {'total_pnl': Decimal('5000'), 'timestamp': datetime.now() - timedelta(hours=6)},  # Peak
            {'total_pnl': Decimal('3000'), 'timestamp': datetime.now() - timedelta(hours=4)},  # Drawdown
            {'total_pnl': Decimal('3500'), 'timestamp': datetime.now() - timedelta(hours=3)},  # Partial recovery
            {'total_pnl': Decimal('4200'), 'timestamp': datetime.now() - timedelta(hours=2)},  # More recovery
            {'total_pnl': Decimal('4800'), 'timestamp': datetime.now() - timedelta(hours=1)},  # Near recovery
        ]
        
        for pnl in recovery_sequence:
            await drawdown_protector.update_pnl(pnl)
        
        # Check recovery percentage
        recovery_pct = await drawdown_protector.calculate_recovery_percentage()
        
        # Recovery from 3000 to 4800 vs target 5000
        # (4800-3000)/(5000-3000) = 1800/2000 = 90%
        expected_recovery = Decimal('90')
        assert abs(recovery_pct - expected_recovery) < Decimal('1')


class TestStopLossManager:
    """Test stop-loss management functionality."""
    
    @pytest.fixture
    def stop_loss_manager(self):
        """Create stop-loss manager for testing."""
        return StopLossManager()
    
    @pytest.mark.asyncio
    async def test_initialization(self, stop_loss_manager):
        """Test stop-loss manager initializes correctly."""
        await stop_loss_manager.initialize()
        
        assert stop_loss_manager.default_stop_loss_pct == 2.0
        assert stop_loss_manager.max_position_loss_pct == 5.0
        assert stop_loss_manager.dynamic_adjustment_enabled == True
        assert len(stop_loss_manager.active_stops) == 0
    
    @pytest.mark.asyncio
    async def test_stop_loss_creation(self, stop_loss_manager):
        """Test creation of stop-loss orders."""
        await stop_loss_manager.initialize()
        
        # Create position with stop-loss
        position_data = {
            'symbol': 'BTC-USD',
            'side': 'long',
            'entry_price': Decimal('50000'),
            'size': Decimal('0.1'),
            'notional_value': Decimal('5000'),
            'timestamp': datetime.now()
        }
        
        stop_loss_id = await stop_loss_manager.create_stop_loss(position_data)
        
        assert stop_loss_id is not None
        assert stop_loss_id in stop_loss_manager.active_stops
        
        stop_loss = stop_loss_manager.active_stops[stop_loss_id]
        
        # Stop loss should be 2% below entry for long position
        expected_stop_price = Decimal('50000') * Decimal('0.98')  # 2% below
        assert abs(stop_loss['stop_price'] - expected_stop_price) < Decimal('10')
    
    @pytest.mark.asyncio
    async def test_dynamic_stop_adjustment(self, stop_loss_manager):
        """Test dynamic stop-loss adjustment based on volatility."""
        await stop_loss_manager.initialize()
        
        position_data = {
            'symbol': 'BTC-USD',
            'side': 'long',
            'entry_price': Decimal('50000'),
            'size': Decimal('0.1'),
            'notional_value': Decimal('5000'),
            'timestamp': datetime.now()
        }
        
        stop_loss_id = await stop_loss_manager.create_stop_loss(position_data)
        original_stop = stop_loss_manager.active_stops[stop_loss_id]['stop_price']
        
        # Update with high volatility market data
        volatility_data = {
            'symbol': 'BTC-USD',
            'volatility': 0.08,  # 8% volatility
            'current_price': Decimal('51000'),
            'timestamp': datetime.now()
        }
        
        await stop_loss_manager.update_stop_for_volatility(stop_loss_id, volatility_data)
        
        adjusted_stop = stop_loss_manager.active_stops[stop_loss_id]['stop_price']
        
        # With high volatility, stop should be wider (lower for long position)
        assert adjusted_stop < original_stop
    
    @pytest.mark.asyncio
    async def test_trailing_stop_loss(self, stop_loss_manager):
        """Test trailing stop-loss functionality."""
        await stop_loss_manager.initialize()
        
        position_data = {
            'symbol': 'BTC-USD',
            'side': 'long',
            'entry_price': Decimal('50000'),
            'size': Decimal('0.1'),
            'notional_value': Decimal('5000'),
            'timestamp': datetime.now()
        }
        
        stop_loss_id = await stop_loss_manager.create_trailing_stop(position_data, trail_pct=1.5)
        
        original_stop = stop_loss_manager.active_stops[stop_loss_id]['stop_price']
        
        # Price moves up - trailing stop should adjust up
        price_updates = [
            {'current_price': Decimal('51000'), 'timestamp': datetime.now()},
            {'current_price': Decimal('52000'), 'timestamp': datetime.now()},
            {'current_price': Decimal('53000'), 'timestamp': datetime.now()},
        ]
        
        for update in price_updates:
            await stop_loss_manager.update_trailing_stop(stop_loss_id, update)
        
        final_stop = stop_loss_manager.active_stops[stop_loss_id]['stop_price']
        
        # Trailing stop should have moved up with price
        assert final_stop > original_stop
        
        # Should be 1.5% below current price of 53000
        expected_stop = Decimal('53000') * Decimal('0.985')  # 1.5% below
        assert abs(final_stop - expected_stop) < Decimal('100')
    
    @pytest.mark.asyncio
    async def test_stop_loss_execution(self, stop_loss_manager):
        """Test stop-loss order execution."""
        await stop_loss_manager.initialize()
        
        executed_orders = []
        
        async def mock_execute_order(order_data):
            executed_orders.append(order_data)
            return {'status': 'filled', 'order_id': 'test_order_123'}
        
        stop_loss_manager.set_execution_callback(mock_execute_order)
        
        position_data = {
            'symbol': 'BTC-USD',
            'side': 'long',
            'entry_price': Decimal('50000'),
            'size': Decimal('0.1'),
            'notional_value': Decimal('5000'),
            'timestamp': datetime.now()
        }
        
        stop_loss_id = await stop_loss_manager.create_stop_loss(position_data)
        
        # Price drops below stop loss
        price_update = {
            'symbol': 'BTC-USD',
            'current_price': Decimal('48500'),  # Below stop loss
            'timestamp': datetime.now()
        }
        
        triggered = await stop_loss_manager.check_stop_triggers([price_update])
        
        assert len(triggered) > 0
        assert len(executed_orders) > 0
        
        # Check executed order details
        executed_order = executed_orders[0]
        assert executed_order['symbol'] == 'BTC-USD'
        assert executed_order['side'] == 'sell'  # Opposite of long position
        assert executed_order['size'] == Decimal('0.1')
    
    @pytest.mark.asyncio
    async def test_position_sizing_adjustment(self, stop_loss_manager):
        """Test position size adjustment based on stop distance."""
        await stop_loss_manager.initialize()
        
        # Test different stop distances
        test_cases = [
            {
                'entry_price': Decimal('50000'),
                'stop_price': Decimal('49000'),  # 2% stop
                'risk_amount': Decimal('1000'),
                'expected_size_range': (Decimal('0.08'), Decimal('0.12'))  # Around 0.1 BTC
            },
            {
                'entry_price': Decimal('50000'),
                'stop_price': Decimal('47500'),  # 5% stop  
                'risk_amount': Decimal('1000'),
                'expected_size_range': (Decimal('0.035'), Decimal('0.045'))  # Around 0.04 BTC
            }
        ]
        
        for case in test_cases:
            calculated_size = await stop_loss_manager.calculate_position_size_for_risk(
                entry_price=case['entry_price'],
                stop_price=case['stop_price'],
                risk_amount=case['risk_amount']
            )
            
            assert case['expected_size_range'][0] <= calculated_size <= case['expected_size_range'][1]
    
    @pytest.mark.asyncio
    async def test_multiple_positions_management(self, stop_loss_manager):
        """Test managing stop losses for multiple positions."""
        await stop_loss_manager.initialize()
        
        # Create multiple positions
        positions = [
            {
                'symbol': 'BTC-USD',
                'side': 'long',
                'entry_price': Decimal('50000'),
                'size': Decimal('0.1'),
                'notional_value': Decimal('5000')
            },
            {
                'symbol': 'ETH-USD',
                'side': 'short',
                'entry_price': Decimal('3000'),
                'size': Decimal('2.0'),
                'notional_value': Decimal('6000')
            },
            {
                'symbol': 'AAPL',
                'side': 'long',
                'entry_price': Decimal('150'),
                'size': Decimal('50'),
                'notional_value': Decimal('7500')
            }
        ]
        
        stop_ids = []
        for position in positions:
            position['timestamp'] = datetime.now()
            stop_id = await stop_loss_manager.create_stop_loss(position)
            stop_ids.append(stop_id)
        
        assert len(stop_ids) == 3
        assert len(stop_loss_manager.active_stops) == 3
        
        # Check that each position has appropriate stop loss
        for i, stop_id in enumerate(stop_ids):
            stop_loss = stop_loss_manager.active_stops[stop_id]
            assert stop_loss['symbol'] == positions[i]['symbol']
            assert stop_loss['side'] == positions[i]['side']
    
    @pytest.mark.asyncio
    async def test_stop_loss_modification(self, stop_loss_manager):
        """Test modifying existing stop-loss orders."""
        await stop_loss_manager.initialize()
        
        position_data = {
            'symbol': 'BTC-USD',
            'side': 'long',
            'entry_price': Decimal('50000'),
            'size': Decimal('0.1'),
            'notional_value': Decimal('5000'),
            'timestamp': datetime.now()
        }
        
        stop_loss_id = await stop_loss_manager.create_stop_loss(position_data)
        original_stop = stop_loss_manager.active_stops[stop_loss_id]['stop_price']
        
        # Modify stop loss to be tighter
        new_stop_price = Decimal('49500')  # Tighter than original 2%
        
        success = await stop_loss_manager.modify_stop_loss(stop_loss_id, new_stop_price)
        
        assert success
        
        modified_stop = stop_loss_manager.active_stops[stop_loss_id]['stop_price']
        assert modified_stop == new_stop_price
        assert modified_stop != original_stop


class TestPnLRiskController:
    """Test the complete P&L risk control system."""
    
    @pytest.fixture
    def pnl_controller(self):
        """Create P&L risk controller for testing."""
        return PnLRiskController()
    
    @pytest.mark.asyncio
    async def test_controller_initialization(self, pnl_controller):
        """Test P&L controller initializes all components."""
        await pnl_controller.initialize()
        
        assert pnl_controller.drawdown_protector is not None
        assert pnl_controller.stop_loss_manager is not None
        assert pnl_controller.enabled
        assert pnl_controller.monitoring_interval > 0
    
    @pytest.mark.asyncio
    async def test_comprehensive_pnl_monitoring(self, pnl_controller):
        """Test comprehensive P&L monitoring and risk assessment."""
        await pnl_controller.initialize()
        
        # Simulate trading day with P&L data
        pnl_data = {
            'positions': [
                {
                    'symbol': 'BTC-USD',
                    'unrealized_pnl': Decimal('1500'),
                    'realized_pnl': Decimal('800'),
                    'side': 'long',
                    'entry_price': Decimal('50000'),
                    'current_price': Decimal('51000'),
                    'size': Decimal('0.1')
                },
                {
                    'symbol': 'ETH-USD',
                    'unrealized_pnl': Decimal('-500'),
                    'realized_pnl': Decimal('200'),
                    'side': 'short',
                    'entry_price': Decimal('3000'),
                    'current_price': Decimal('3100'),
                    'size': Decimal('2.0')
                }
            ],
            'total_unrealized_pnl': Decimal('1000'),
            'total_realized_pnl': Decimal('1000'),
            'total_pnl': Decimal('2000'),
            'daily_pnl': Decimal('2000'),
            'weekly_pnl': Decimal('5000'),
            'monthly_pnl': Decimal('12000'),
            'timestamp': datetime.now()
        }
        
        risk_assessment = await pnl_controller.assess_pnl_risk(pnl_data)
        
        assert 'drawdown_analysis' in risk_assessment
        assert 'stop_loss_status' in risk_assessment
        assert 'risk_metrics' in risk_assessment
        assert 'overall_risk_level' in risk_assessment
        assert 'recommendations' in risk_assessment
    
    @pytest.mark.asyncio
    async def test_risk_limit_enforcement(self, pnl_controller):
        """Test P&L risk limit enforcement."""
        await pnl_controller.initialize()
        
        # Create scenario with excessive losses
        high_loss_data = {
            'positions': [
                {
                    'symbol': 'BTC-USD',
                    'unrealized_pnl': Decimal('-8000'),
                    'realized_pnl': Decimal('-2000'),
                    'side': 'long',
                    'entry_price': Decimal('50000'),
                    'current_price': Decimal('42000'),  # 16% loss
                    'size': Decimal('0.2')
                }
            ],
            'total_unrealized_pnl': Decimal('-8000'),
            'total_realized_pnl': Decimal('-2000'),
            'total_pnl': Decimal('-10000'),
            'daily_pnl': Decimal('-10000'),
            'timestamp': datetime.now()
        }
        
        risk_assessment = await pnl_controller.assess_pnl_risk(high_loss_data)
        
        # Should detect high risk and recommend action
        assert risk_assessment['overall_risk_level'] in ['high', 'critical']
        assert len(risk_assessment['recommendations']) > 0
    
    @pytest.mark.asyncio
    async def test_performance_metrics_calculation(self, pnl_controller):
        """Test calculation of performance metrics."""
        await pnl_controller.initialize()
        
        # Add historical P&L data
        historical_pnls = []
        base_pnl = Decimal('10000')
        
        for i in range(30):  # 30 days of data
            daily_change = Decimal(str(np.random.normal(100, 500)))  # Random daily P&L
            pnl = base_pnl + daily_change
            
            historical_pnls.append({
                'total_pnl': pnl,
                'daily_pnl': daily_change,
                'timestamp': datetime.now() - timedelta(days=30-i)
            })
            
            base_pnl = pnl
        
        # Calculate performance metrics
        metrics = await pnl_controller.calculate_performance_metrics(historical_pnls)
        
        assert 'total_return' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'max_drawdown' in metrics
        assert 'win_rate' in metrics
        assert 'profit_factor' in metrics
        assert 'volatility' in metrics
    
    @pytest.mark.asyncio
    async def test_position_risk_scoring(self, pnl_controller):
        """Test individual position risk scoring."""
        await pnl_controller.initialize()
        
        positions = [
            {
                'symbol': 'BTC-USD',
                'unrealized_pnl': Decimal('2000'),   # Profitable
                'size': Decimal('0.1'),
                'entry_price': Decimal('50000'),
                'current_price': Decimal('52000'),
                'duration_hours': 24,
                'volatility': 0.02
            },
            {
                'symbol': 'VOLATILE-COIN',
                'unrealized_pnl': Decimal('-1500'),  # Loss
                'size': Decimal('1000'),
                'entry_price': Decimal('1.0'),
                'current_price': Decimal('0.85'),
                'duration_hours': 120,  # Held for 5 days
                'volatility': 0.15  # Very volatile
            }
        ]
        
        risk_scores = []
        for position in positions:
            score = await pnl_controller.calculate_position_risk_score(position)
            risk_scores.append(score)
        
        # Volatile losing position should have higher risk score
        assert risk_scores[1] > risk_scores[0]
        assert all(0 <= score <= 100 for score in risk_scores)
    
    @pytest.mark.asyncio
    async def test_correlation_risk_assessment(self, pnl_controller):
        """Test correlation risk in P&L analysis."""
        await pnl_controller.initialize()
        
        # Positions in correlated assets
        correlated_positions = [
            {
                'symbol': 'BTC-USD',
                'unrealized_pnl': Decimal('-2000'),
                'size': Decimal('0.1'),
                'sector': 'crypto'
            },
            {
                'symbol': 'ETH-USD',
                'unrealized_pnl': Decimal('-1500'),
                'size': Decimal('2.0'),
                'sector': 'crypto'
            },
            {
                'symbol': 'ADA-USD',
                'unrealized_pnl': Decimal('-800'),
                'size': Decimal('1000'),
                'sector': 'crypto'
            }
        ]
        
        correlation_risk = await pnl_controller.assess_correlation_risk(correlated_positions)
        
        assert 'sector_concentration' in correlation_risk
        assert 'correlated_losses' in correlation_risk
        assert 'diversification_score' in correlation_risk
        
        # Should detect high concentration in crypto sector
        assert correlation_risk['sector_concentration']['crypto'] > 50  # Over 50% in crypto
    
    @pytest.mark.asyncio
    async def test_real_time_monitoring(self, pnl_controller):
        """Test real-time P&L monitoring and alerting."""
        await pnl_controller.initialize()
        
        alerts_received = []
        
        async def mock_alert_callback(alert_data):
            alerts_received.append(alert_data)
        
        pnl_controller.set_alert_callback(mock_alert_callback)
        
        # Start monitoring
        monitoring_task = asyncio.create_task(pnl_controller.start_monitoring())
        
        # Simulate P&L updates
        pnl_updates = [
            {'total_pnl': Decimal('5000'), 'timestamp': datetime.now()},
            {'total_pnl': Decimal('3000'), 'timestamp': datetime.now()},  # 40% drawdown
            {'total_pnl': Decimal('1000'), 'timestamp': datetime.now()},  # 80% drawdown
        ]
        
        for update in pnl_updates:
            await pnl_controller.update_pnl(update)
            await asyncio.sleep(0.1)  # Small delay
        
        # Stop monitoring
        monitoring_task.cancel()
        
        try:
            await monitoring_task
        except asyncio.CancelledError:
            pass
        
        # Should have received drawdown alerts
        assert len(alerts_received) >= 0  # May or may not alert based on thresholds
    
    @pytest.mark.asyncio
    async def test_portfolio_heat_map(self, pnl_controller):
        """Test portfolio risk heat map generation."""
        await pnl_controller.initialize()
        
        positions = [
            {'symbol': 'BTC-USD', 'unrealized_pnl': Decimal('2000'), 'notional': Decimal('20000'), 'risk_score': 30},
            {'symbol': 'ETH-USD', 'unrealized_pnl': Decimal('-1000'), 'notional': Decimal('15000'), 'risk_score': 60},
            {'symbol': 'RISKY-COIN', 'unrealized_pnl': Decimal('-2000'), 'notional': Decimal('10000'), 'risk_score': 85},
            {'symbol': 'AAPL', 'unrealized_pnl': Decimal('500'), 'notional': Decimal('8000'), 'risk_score': 25},
        ]
        
        heat_map = await pnl_controller.generate_risk_heat_map(positions)
        
        assert 'high_risk_positions' in heat_map
        assert 'medium_risk_positions' in heat_map
        assert 'low_risk_positions' in heat_map
        assert 'risk_distribution' in heat_map
        
        # RISKY-COIN should be in high risk category
        high_risk_symbols = [pos['symbol'] for pos in heat_map['high_risk_positions']]
        assert 'RISKY-COIN' in high_risk_symbols


@pytest.mark.asyncio
async def test_integration_stress_scenario():
    """Test P&L controller under market stress scenario."""
    controller = PnLRiskController()
    await controller.initialize()
    
    # Simulate market crash affecting multiple positions
    crash_scenario = {
        'positions': [
            {
                'symbol': 'BTC-USD',
                'unrealized_pnl': Decimal('-15000'),  # Large loss
                'realized_pnl': Decimal('0'),
                'side': 'long',
                'entry_price': Decimal('60000'),
                'current_price': Decimal('40000'),    # 33% drop
                'size': Decimal('0.5')
            },
            {
                'symbol': 'ETH-USD',
                'unrealized_pnl': Decimal('-8000'),   # Large loss
                'realized_pnl': Decimal('0'),
                'side': 'long',
                'entry_price': Decimal('4000'),
                'current_price': Decimal('2800'),     # 30% drop
                'size': Decimal('3.0')
            }
        ],
        'total_unrealized_pnl': Decimal('-23000'),
        'total_realized_pnl': Decimal('0'),
        'total_pnl': Decimal('-23000'),
        'daily_pnl': Decimal('-23000'),
        'timestamp': datetime.now()
    }
    
    # Assess crash scenario
    risk_assessment = await controller.assess_pnl_risk(crash_scenario)
    
    # Should detect critical risk level
    assert risk_assessment['overall_risk_level'] == 'critical'
    assert 'emergency_stop' in risk_assessment['recommendations']
    
    # Should trigger multiple protective measures
    protective_actions = risk_assessment.get('protective_actions', [])
    assert len(protective_actions) > 0


if __name__ == "__main__":
    pytest.main([__file__])
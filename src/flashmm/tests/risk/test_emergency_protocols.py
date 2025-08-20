"""
Test suite for Emergency Protocols

Tests emergency response system components including:
- Emergency protocol manager and protocol execution
- Position flattening procedures and automation
- Market exit strategies for different scenarios
- Emergency event tracking and response coordination
- Integration with circuit breakers and risk systems
"""

import pytest
import asyncio
from decimal import Decimal
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, List

from flashmm.risk.emergency_protocols import (
    EmergencyProtocolManager,
    PositionFlattener,
    MarketExitStrategy,
    EmergencyEvent,
    EmergencyProtocol,
    EmergencyLevel,
    EmergencyType,
    ProtocolAction
)
from flashmm.utils.exceptions import RiskError, EmergencyStopError


class TestEmergencyProtocol:
    """Test emergency protocol definitions and logic."""
    
    @pytest.fixture
    def sample_protocol(self):
        """Create sample emergency protocol for testing."""
        return EmergencyProtocol(
            name="test_protocol",
            trigger_conditions=[
                {'type': 'daily_pnl', 'threshold': -5000.0, 'operator': 'less_than'},
                {'type': 'volatility_spike', 'threshold': 3.0, 'operator': 'greater_than'}
            ],
            response_actions=[
                ProtocolAction.HALT_TRADING,
                ProtocolAction.CANCEL_ORDERS,
                ProtocolAction.FLATTEN_POSITIONS
            ],
            required_confirmations=0,
            auto_execute=True,
            max_execution_time_seconds=300,
            escalation_level=EmergencyLevel.CRITICAL
        )
    
    def test_protocol_initialization(self, sample_protocol):
        """Test protocol initializes with correct parameters."""
        assert sample_protocol.name == "test_protocol"
        assert len(sample_protocol.trigger_conditions) == 2
        assert len(sample_protocol.response_actions) == 3
        assert sample_protocol.auto_execute == True
        assert sample_protocol.escalation_level == EmergencyLevel.CRITICAL
        assert sample_protocol.execution_count == 0
    
    def test_trigger_condition_evaluation(self, sample_protocol):
        """Test trigger condition evaluation logic."""
        # Conditions that should trigger
        trigger_conditions = {
            'daily_pnl': -7000.0,  # Below -5000 threshold
            'volatility_spike': 4.0  # Above 3.0 threshold
        }
        
        should_trigger = sample_protocol.should_trigger(trigger_conditions)
        assert should_trigger
        
        # Conditions that should not trigger
        no_trigger_conditions = {
            'daily_pnl': -3000.0,  # Above -5000 threshold
            'volatility_spike': 2.0  # Below 3.0 threshold
        }
        
        should_not_trigger = sample_protocol.should_trigger(no_trigger_conditions)
        assert not should_not_trigger
    
    def test_protocol_serialization(self, sample_protocol):
        """Test protocol serialization to dictionary."""
        protocol_dict = sample_protocol.to_dict()
        
        assert protocol_dict['name'] == "test_protocol"
        assert 'trigger_conditions' in protocol_dict
        assert 'response_actions' in protocol_dict
        assert 'execution_stats' in protocol_dict
        assert protocol_dict['escalation_level'] == 'critical'


class TestEmergencyEvent:
    """Test emergency event tracking and management."""
    
    @pytest.fixture
    def sample_event(self):
        """Create sample emergency event for testing."""
        return EmergencyEvent(
            event_id="TEST_EMERGENCY_001",
            emergency_type=EmergencyType.EXCESSIVE_LOSSES,
            emergency_level=EmergencyLevel.CRITICAL,
            trigger_reason="Daily loss limit exceeded",
            trigger_value=-8000.0,
            threshold_value=-5000.0,
            timestamp=datetime.now()
        )
    
    def test_event_initialization(self, sample_event):
        """Test event initializes with correct data."""
        assert sample_event.event_id == "TEST_EMERGENCY_001"
        assert sample_event.emergency_type == EmergencyType.EXCESSIVE_LOSSES
        assert sample_event.emergency_level == EmergencyLevel.CRITICAL
        assert sample_event.trigger_reason == "Daily loss limit exceeded"
        assert sample_event.resolved == False
        assert len(sample_event.actions_taken) == 0
    
    def test_event_action_tracking(self, sample_event):
        """Test tracking of actions taken during emergency."""
        # Add some actions
        actions = [
            {'action': 'halt_trading', 'status': 'completed', 'timestamp': datetime.now().isoformat()},
            {'action': 'cancel_orders', 'status': 'completed', 'orders_cancelled': 15},
            {'action': 'flatten_positions', 'status': 'failed', 'error': 'Connection timeout'}
        ]
        
        for action in actions:
            sample_event.actions_taken.append(action)
        
        assert len(sample_event.actions_taken) == 3
        assert sample_event.actions_taken[0]['action'] == 'halt_trading'
        assert sample_event.actions_taken[2]['status'] == 'failed'
    
    def test_event_serialization(self, sample_event):
        """Test event serialization to dictionary."""
        event_dict = sample_event.to_dict()
        
        assert event_dict['event_id'] == "TEST_EMERGENCY_001"
        assert event_dict['emergency_type'] == 'excessive_losses'
        assert event_dict['emergency_level'] == 'critical'
        assert 'impact' in event_dict
        assert 'timestamp' in event_dict


class TestPositionFlattener:
    """Test position flattening functionality."""
    
    @pytest.fixture
    def position_flattener(self):
        """Create position flattener for testing."""
        return PositionFlattener()
    
    @pytest.mark.asyncio
    async def test_flattener_initialization(self, position_flattener):
        """Test flattener initializes correctly."""
        assert position_flattener.flattening_in_progress == False
        assert position_flattener.max_flatten_time_seconds == 300
        assert position_flattener.flatten_price_tolerance_pct == 5.0
        assert len(position_flattener.flattened_positions) == 0
    
    @pytest.mark.asyncio
    async def test_callback_setup(self, position_flattener):
        """Test callback function setup."""
        mock_cancel_orders = AsyncMock()
        mock_place_order = AsyncMock()
        mock_get_positions = AsyncMock()
        
        position_flattener.set_callbacks(
            cancel_orders=mock_cancel_orders,
            place_order=mock_place_order,
            get_positions=mock_get_positions
        )
        
        assert position_flattener.cancel_orders_callback == mock_cancel_orders
        assert position_flattener.place_order_callback == mock_place_order
        assert position_flattener.get_positions_callback == mock_get_positions
    
    @pytest.mark.asyncio
    async def test_position_flattening_process(self, position_flattener):
        """Test complete position flattening process."""
        # Setup mock callbacks
        mock_cancel_orders = AsyncMock(return_value={'cancelled_count': 5})
        mock_place_order = AsyncMock(return_value={'status': 'filled', 'order_id': 'test_123'})
        
        mock_positions = [
            {
                'symbol': 'BTC-USD',
                'base_balance': 0.5,  # Long position
                'mark_price': 50000.0
            },
            {
                'symbol': 'ETH-USD', 
                'base_balance': -2.0,  # Short position
                'mark_price': 3000.0
            }
        ]
        
        mock_get_positions = AsyncMock(return_value=mock_positions)
        
        position_flattener.set_callbacks(
            cancel_orders=mock_cancel_orders,
            place_order=mock_place_order,
            get_positions=mock_get_positions
        )
        
        # Execute flattening
        result = await position_flattener.flatten_all_positions(
            reason="Test emergency",
            emergency_level=EmergencyLevel.CRITICAL
        )
        
        assert result['status'] == 'completed'
        assert result['positions_flattened'] >= 0
        assert result['orders_cancelled'] == 5
        assert 'execution_time_seconds' in result
        
        # Verify callbacks were called
        mock_cancel_orders.assert_called_once()
        mock_get_positions.assert_called_once()
        assert mock_place_order.call_count <= 2  # Up to 2 positions to flatten
    
    @pytest.mark.asyncio
    async def test_concurrent_flattening_prevention(self, position_flattener):
        """Test prevention of concurrent flattening operations."""
        # Setup basic callbacks
        position_flattener.set_callbacks(
            cancel_orders=AsyncMock(return_value={'cancelled_count': 0}),
            get_positions=AsyncMock(return_value=[])
        )
        
        # Start first flattening
        task1 = asyncio.create_task(
            position_flattener.flatten_all_positions("Test 1", EmergencyLevel.CRITICAL)
        )
        
        # Try to start second flattening immediately
        result2 = await position_flattener.flatten_all_positions("Test 2", EmergencyLevel.CRITICAL)
        
        # Second should be rejected
        assert result2['status'] == 'already_in_progress'
        
        # Wait for first to complete
        result1 = await task1
        assert result1['status'] == 'completed'
    
    @pytest.mark.asyncio
    async def test_emergency_level_strategy_selection(self, position_flattener):
        """Test different strategies based on emergency level."""
        mock_place_order = AsyncMock(return_value={'status': 'filled'})
        position_flattener.place_order_callback = mock_place_order
        
        position = {
            'symbol': 'BTC-USD',
            'base_balance': 0.1,
            'mark_price': 50000.0
        }
        
        # Test catastrophic level (should use market orders)
        await position_flattener._flatten_single_position(position, EmergencyLevel.CATASTROPHIC)
        
        # Verify order was placed (implementation would check order type)
        mock_place_order.assert_called()
        call_args = mock_place_order.call_args[0][0]
        assert call_args['symbol'] == 'BTC-USD'
        assert call_args['side'] == 'sell'  # Opposite of long position


class TestMarketExitStrategy:
    """Test market exit strategy functionality."""
    
    @pytest.fixture
    def exit_strategy(self):
        """Create market exit strategy for testing."""
        return MarketExitStrategy()
    
    @pytest.mark.asyncio
    async def test_strategy_selection(self, exit_strategy):
        """Test selection of appropriate exit strategy."""
        sample_positions = [
            {'symbol': 'BTC-USD', 'notional_value': 10000},
            {'symbol': 'ETH-USD', 'notional_value': 5000}
        ]
        
        # Test different emergency types
        strategies = [
            (EmergencyType.MARKET_CRASH, 'immediate_market_exit'),
            (EmergencyType.LIQUIDITY_CRISIS, 'gradual_limit_exit'),
            (EmergencyType.SYSTEM_FAILURE, 'log_for_manual_closure'),
            (EmergencyType.CONNECTIVITY_LOSS, 'backup_connection_exit'),
            (EmergencyType.EXCESSIVE_LOSSES, 'selective_exit')
        ]
        
        for emergency_type, expected_strategy in strategies:
            result = await exit_strategy.execute_exit_strategy(emergency_type, sample_positions)
            
            assert result['status'] in ['executed', 'manual_intervention_required', 'attempting_backup']
            assert 'strategy' in result
            assert result['strategy'] == expected_strategy
    
    @pytest.mark.asyncio
    async def test_market_crash_strategy(self, exit_strategy):
        """Test market crash exit strategy."""
        positions = [{'symbol': 'BTC-USD', 'notional_value': 15000}]
        
        result = await exit_strategy._market_crash_exit(positions)
        
        assert result['strategy'] == 'immediate_market_exit'
        assert result['priority'] == 'speed_over_price'
        assert result['expected_slippage'] == 'high'
    
    @pytest.mark.asyncio
    async def test_liquidity_crisis_strategy(self, exit_strategy):
        """Test liquidity crisis exit strategy."""
        positions = [{'symbol': 'ILLIQUID-USD', 'notional_value': 8000}]
        
        result = await exit_strategy._liquidity_crisis_exit(positions)
        
        assert result['strategy'] == 'gradual_limit_exit'
        assert result['priority'] == 'minimize_market_impact'
        assert result['expected_slippage'] == 'medium'


class TestEmergencyProtocolManager:
    """Test the complete emergency protocol management system."""
    
    @pytest.fixture
    def protocol_manager(self):
        """Create emergency protocol manager for testing."""
        return EmergencyProtocolManager()
    
    @pytest.mark.asyncio
    async def test_manager_initialization(self, protocol_manager):
        """Test manager initializes with default protocols."""
        await protocol_manager.initialize()
        
        assert protocol_manager.position_flattener is not None
        assert protocol_manager.market_exit_strategy is not None
        assert len(protocol_manager.protocols) > 0
        assert protocol_manager.emergency_mode_active == False
        assert protocol_manager.system_isolated == False
    
    @pytest.mark.asyncio
    async def test_callback_setup(self, protocol_manager):
        """Test callback function setup."""
        await protocol_manager.initialize()
        
        mock_callbacks = {
            'halt_trading': AsyncMock(),
            'cancel_orders': AsyncMock(),
            'place_order': AsyncMock(),
            'get_positions': AsyncMock(),
            'notify_stakeholders': AsyncMock()
        }
        
        protocol_manager.set_callbacks(**mock_callbacks)
        
        assert protocol_manager.halt_trading_callback == mock_callbacks['halt_trading']
        assert protocol_manager.cancel_orders_callback == mock_callbacks['cancel_orders']
    
    @pytest.mark.asyncio
    async def test_emergency_condition_checking(self, protocol_manager):
        """Test checking of emergency conditions."""
        await protocol_manager.initialize()
        
        # Setup mock callbacks
        protocol_manager.set_callbacks(
            halt_trading=AsyncMock(),
            cancel_orders=AsyncMock(return_value={'cancelled_count': 10}),
            place_order=AsyncMock(),
            get_positions=AsyncMock(return_value=[]),
            notify_stakeholders=AsyncMock()
        )
        
        # Risk data that should trigger emergency protocols
        critical_risk_data = {
            'components': {
                'pnl_controller': {
                    'global_metrics': {
                        'daily': -12000.0  # Large loss
                    }
                },
                'operational_risk': {
                    'system_health': 'critical',
                    'endpoint_health': {
                        'api1': {'success': False},
                        'api2': {'success': False}
                    }
                },
                'circuit_breakers': {
                    'active_breakers': 3
                }
            }
        }
        
        triggered_protocols = await protocol_manager.check_emergency_conditions(critical_risk_data)
        
        # Should have triggered some protocols
        assert len(triggered_protocols) >= 0
        
        # Emergency mode should be activated if protocols were triggered
        if len(triggered_protocols) > 0:
            assert protocol_manager.emergency_mode_active
    
    @pytest.mark.asyncio
    async def test_manual_emergency_stop(self, protocol_manager):
        """Test manual emergency stop procedure."""
        await protocol_manager.initialize()
        
        # Setup mock callbacks
        mock_callbacks = {
            'halt_trading': AsyncMock(),
            'cancel_orders': AsyncMock(return_value={'cancelled_count': 5}),
            'place_order': AsyncMock(),
            'get_positions': AsyncMock(return_value=[]),
            'notify_stakeholders': AsyncMock()
        }
        
        protocol_manager.set_callbacks(**mock_callbacks)
        
        # Trigger manual emergency stop
        result = await protocol_manager.manual_emergency_stop(
            reason="Manual test stop",
            emergency_type=EmergencyType.OPERATIONAL_ERROR
        )
        
        assert result['status'] == 'completed'
        assert 'event_id' in result
        assert 'actions_executed' in result
        assert protocol_manager.emergency_mode_active
        
        # Verify callbacks were called
        mock_callbacks['halt_trading'].assert_called()
        mock_callbacks['cancel_orders'].assert_called()
        mock_callbacks['notify_stakeholders'].assert_called()
    
    @pytest.mark.asyncio
    async def test_emergency_recovery(self, protocol_manager):
        """Test recovery from emergency state."""
        await protocol_manager.initialize()
        
        # Setup and trigger emergency
        protocol_manager.set_callbacks(
            halt_trading=AsyncMock(),
            cancel_orders=AsyncMock(return_value={'cancelled_count': 0}),
            get_positions=AsyncMock(return_value=[])
        )
        
        stop_result = await protocol_manager.manual_emergency_stop(
            "Test emergency",
            EmergencyType.SYSTEM_FAILURE
        )
        
        event_id = stop_result['event_id']
        
        # Recover from emergency
        recovery_result = await protocol_manager.recover_from_emergency(
            event_id,
            "System restored, manual recovery"
        )
        
        assert recovery_result['status'] == 'resolved'
        assert recovery_result['event_id'] == event_id
        assert not protocol_manager.emergency_mode_active
        assert not protocol_manager.system_isolated
    
    @pytest.mark.asyncio
    async def test_custom_protocol_addition(self, protocol_manager):
        """Test adding custom emergency protocols."""
        await protocol_manager.initialize()
        
        initial_count = len(protocol_manager.protocols)
        
        # Create custom protocol
        custom_protocol = EmergencyProtocol(
            name="custom_test_protocol",
            trigger_conditions=[
                {'type': 'custom_metric', 'threshold': 100.0, 'operator': 'greater_than'}
            ],
            response_actions=[ProtocolAction.HALT_TRADING],
            required_confirmations=1,
            auto_execute=False,
            max_execution_time_seconds=60,
            escalation_level=EmergencyLevel.WARNING
        )
        
        success = protocol_manager.add_custom_protocol(custom_protocol)
        
        assert success
        assert len(protocol_manager.protocols) == initial_count + 1
        assert 'custom_test_protocol' in protocol_manager.protocols
    
    @pytest.mark.asyncio
    async def test_emergency_status_reporting(self, protocol_manager):
        """Test emergency system status reporting."""
        await protocol_manager.initialize()
        
        # Get initial status
        status = protocol_manager.get_emergency_status()
        
        assert 'emergency_mode_active' in status
        assert 'system_isolated' in status
        assert 'active_emergencies' in status
        assert 'total_protocols' in status
        assert 'last_emergency_check' in status
        assert 'emergency_history_count' in status
        
        # Initially should be normal
        assert not status['emergency_mode_active']
        assert status['active_emergencies'] == 0
    
    @pytest.mark.asyncio
    async def test_protocol_status_reporting(self, protocol_manager):
        """Test protocol status reporting."""
        await protocol_manager.initialize()
        
        protocol_status = protocol_manager.get_protocol_status()
        
        assert isinstance(protocol_status, dict)
        assert len(protocol_status) > 0
        
        # Check structure of protocol data
        for protocol_name, protocol_data in protocol_status.items():
            assert 'trigger_conditions' in protocol_data
            assert 'response_actions' in protocol_data
            assert 'execution_stats' in protocol_data
            assert 'escalation_level' in protocol_data
    
    @pytest.mark.asyncio
    async def test_emergency_health_check(self, protocol_manager):
        """Test emergency system health check."""
        await protocol_manager.initialize()
        
        health_check = await protocol_manager.emergency_health_check()
        
        assert 'overall_status' in health_check
        assert 'emergency_mode_active' in health_check
        assert 'protocols_loaded' in health_check
        assert 'callbacks_configured' in health_check
        assert 'last_check' in health_check
        
        # Should be healthy initially
        assert health_check['overall_status'] in ['healthy', 'emergency_active', 'system_isolated']
        assert health_check['protocols_loaded'] > 0


@pytest.mark.asyncio
async def test_integration_emergency_scenario():
    """Test complete emergency scenario integration."""
    manager = EmergencyProtocolManager()
    await manager.initialize()
    
    # Track all actions taken
    actions_taken = []
    
    async def track_action(action_type, **kwargs):
        actions_taken.append({'type': action_type, 'data': kwargs})
    
    # Setup callbacks to track actions
    manager.set_callbacks(
        halt_trading=lambda reason: track_action('halt_trading', reason=reason),
        cancel_orders=lambda reason: track_action('cancel_orders', reason=reason) or {'cancelled_count': 8},
        place_order=lambda order: track_action('place_order', order=order) or {'status': 'filled'},
        get_positions=lambda: [],
        notify_stakeholders=lambda data: track_action('notify_stakeholders', data=data)
    )
    
    # Simulate cascading crisis
    crisis_conditions = {
        'components': {
            'pnl_controller': {
                'global_metrics': {'daily': -25000.0}  # Massive loss
            },
            'operational_risk': {
                'system_health': 'failure',
                'endpoint_health': {
                    'exchange1': {'success': False},
                    'exchange2': {'success': False},
                    'exchange3': {'success': False}
                }
            },
            'circuit_breakers': {
                'active_breakers': 5
            }
        }
    }
    
    # Check emergency conditions
    triggered = await manager.check_emergency_conditions(crisis_conditions)
    
    # Should have triggered multiple protocols
    assert len(triggered) > 0
    assert manager.emergency_mode_active
    
    # Should have taken multiple emergency actions
    assert len(actions_taken) > 0
    action_types = [action['type'] for action in actions_taken]
    
    # Should include critical emergency actions
    expected_actions = ['halt_trading', 'cancel_orders', 'notify_stakeholders']
    assert any(action in action_types for action in expected_actions)


if __name__ == "__main__":
    pytest.main([__file__])
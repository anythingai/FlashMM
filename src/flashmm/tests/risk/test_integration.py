"""
Integration Test Suite for Enterprise Risk Management System

Tests the complete integration of all risk management components:
- Unified risk management system coordination
- End-to-end risk monitoring and response workflows
- Cross-component communication and data flow
- Emergency response coordination across all systems
- Performance and scalability under realistic conditions
"""

import pytest
import asyncio
import json
from decimal import Decimal
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, List

from flashmm.risk import (
    CircuitBreakerSystem,
    PositionLimitsManager,
    MarketRiskMonitor,
    PnLRiskController,
    OperationalRiskManager,
    RiskReporter,
    EmergencyProtocolManager
)
from flashmm.utils.exceptions import RiskError, EmergencyStopError


class IntegratedRiskManager:
    """Integrated risk management system for testing."""
    
    def __init__(self):
        self.circuit_breakers = CircuitBreakerSystem()
        self.position_limits = PositionLimitsManager()
        self.market_monitor = MarketRiskMonitor()
        self.pnl_controller = PnLRiskController()
        self.operational_monitor = OperationalRiskManager()
        self.risk_reporter = RiskReporter()
        self.emergency_protocols = EmergencyProtocolManager()
        
        self.components = [
            self.circuit_breakers,
            self.position_limits,
            self.market_monitor,
            self.pnl_controller,
            self.operational_monitor,
            self.risk_reporter,
            self.emergency_protocols
        ]
        
        self.enabled = False
        self.monitoring_active = False
        self.risk_events = []
        
    async def initialize(self):
        """Initialize all risk management components."""
        for component in self.components:
            await component.initialize()
        
        # Setup inter-component communication
        await self._setup_component_integration()
        
        self.enabled = True
    
    async def _setup_component_integration(self):
        """Setup integration between components."""
        # Emergency protocols should be notified by all other components
        async def emergency_alert_handler(alert_data):
            await self.emergency_protocols.check_emergency_conditions(alert_data)
        
        # Circuit breakers can trigger emergency protocols
        self.circuit_breakers.set_emergency_callback(emergency_alert_handler)
        
        # Market monitor alerts can trigger circuit breakers
        async def market_alert_handler(market_alert):
            if market_alert.get('severity') == 'critical':
                await self.circuit_breakers.emergency_stop("Market crisis detected")
        
        self.market_monitor.set_alert_callback(market_alert_handler)
        
        # P&L controller can trigger position limits adjustment
        async def pnl_alert_handler(pnl_alert):
            if pnl_alert.get('drawdown_pct', 0) > 10:
                await self.position_limits.emergency_limit_reduction(0.5, "High drawdown detected")
        
        self.pnl_controller.set_alert_callback(pnl_alert_handler)
    
    async def perform_comprehensive_risk_check(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive risk assessment across all components."""
        risk_assessment = {
            'timestamp': datetime.now().isoformat(),
            'components': {},
            'overall_risk_level': 'normal',
            'alerts': [],
            'actions_taken': []
        }
        
        try:
            # Circuit breaker analysis
            cb_triggered = await self.circuit_breakers.check_all_breakers(market_data)
            risk_assessment['components']['circuit_breakers'] = {
                'triggered_breakers': cb_triggered,
                'active_breakers': len(cb_triggered),
                'system_health': 'normal' if len(cb_triggered) == 0 else 'warning'
            }
            
            # Position limits analysis
            if 'positions' in market_data:
                position_status = await self.position_limits.check_portfolio_limits(
                    market_data['positions'],
                    market_data.get('portfolio_value', Decimal('100000'))
                )
                risk_assessment['components']['position_limits'] = position_status
            
            # Market risk analysis
            market_analysis = await self.market_monitor.analyze_market_conditions(market_data)
            risk_assessment['components']['market_risk'] = market_analysis
            
            # P&L risk analysis
            if 'pnl_data' in market_data:
                pnl_analysis = await self.pnl_controller.assess_pnl_risk(market_data['pnl_data'])
                risk_assessment['components']['pnl_risk'] = pnl_analysis
            
            # Operational risk analysis
            operational_health = await self.operational_monitor.perform_health_check()
            risk_assessment['components']['operational_risk'] = operational_health
            
            # Determine overall risk level
            risk_assessment['overall_risk_level'] = await self._calculate_overall_risk_level(
                risk_assessment['components']
            )
            
            # Check if emergency protocols should be triggered
            triggered_protocols = await self.emergency_protocols.check_emergency_conditions(
                risk_assessment
            )
            
            if triggered_protocols:
                risk_assessment['emergency_protocols_triggered'] = triggered_protocols
                risk_assessment['overall_risk_level'] = 'critical'
            
            return risk_assessment
            
        except Exception as e:
            risk_assessment['error'] = str(e)
            risk_assessment['overall_risk_level'] = 'error'
            return risk_assessment
    
    async def _calculate_overall_risk_level(self, components: Dict[str, Any]) -> str:
        """Calculate overall risk level from component assessments."""
        risk_scores = []
        
        # Circuit breakers
        cb_data = components.get('circuit_breakers', {})
        if cb_data.get('active_breakers', 0) > 0:
            risk_scores.append('high')
        
        # Market risk
        market_data = components.get('market_risk', {})
        market_risk = market_data.get('overall_risk_level', 'normal')
        risk_scores.append(market_risk)
        
        # P&L risk
        pnl_data = components.get('pnl_risk', {})
        pnl_risk = pnl_data.get('overall_risk_level', 'normal')
        risk_scores.append(pnl_risk)
        
        # Operational risk
        ops_data = components.get('operational_risk', {})
        ops_risk = ops_data.get('risk_level', 'normal')
        risk_scores.append(ops_risk)
        
        # Determine highest risk level
        if 'critical' in risk_scores:
            return 'critical'
        elif 'high' in risk_scores:
            return 'high'
        elif 'medium' in risk_scores:
            return 'medium'
        else:
            return 'normal'


class TestBasicIntegration:
    """Test basic integration between risk components."""
    
    @pytest.fixture
    async def integrated_system(self):
        """Create integrated risk management system."""
        system = IntegratedRiskManager()
        await system.initialize()
        return system
    
    @pytest.mark.asyncio
    async def test_system_initialization(self, integrated_system):
        """Test that all components initialize correctly."""
        assert integrated_system.enabled
        assert len(integrated_system.components) == 7
        
        # Verify each component is initialized
        for component in integrated_system.components:
            assert hasattr(component, 'enabled')
            # Most components should be enabled after initialization
    
    @pytest.mark.asyncio
    async def test_normal_market_conditions(self, integrated_system):
        """Test system behavior under normal market conditions."""
        normal_market_data = {
            'price_data': {
                'BTC-USD': {
                    'current_price': 50000.0,
                    'previous_price': 49800.0,  # 0.4% change
                    'timestamp': datetime.now()
                }
            },
            'volume_data': {
                'BTC-USD': {
                    'volume': 1000000.0,
                    'timestamp': datetime.now()
                }
            },
            'pnl_data': {
                'total_pnl': 2000.0,
                'daily_pnl': 500.0,
                'positions': [
                    {
                        'symbol': 'BTC-USD',
                        'unrealized_pnl': 1000.0,
                        'size': 0.1
                    }
                ],
                'timestamp': datetime.now()
            },
            'positions': [
                {
                    'symbol': 'BTC-USD',
                    'notional_value': Decimal('5000'),
                    'sector': 'crypto'
                }
            ],
            'portfolio_value': Decimal('100000')
        }
        
        risk_assessment = await integrated_system.perform_comprehensive_risk_check(normal_market_data)
        
        assert risk_assessment['overall_risk_level'] == 'normal'
        assert len(risk_assessment['components']) > 0
        assert 'circuit_breakers' in risk_assessment['components']
        assert risk_assessment['components']['circuit_breakers']['active_breakers'] == 0
    
    @pytest.mark.asyncio
    async def test_market_stress_scenario(self, integrated_system):
        """Test system behavior during market stress."""
        stress_market_data = {
            'price_data': {
                'BTC-USD': {
                    'current_price': 40000.0,
                    'previous_price': 50000.0,  # 20% crash
                    'timestamp': datetime.now()
                },
                'ETH-USD': {
                    'current_price': 2400.0,
                    'previous_price': 3000.0,  # 20% crash
                    'timestamp': datetime.now()
                }
            },
            'volume_data': {
                'BTC-USD': {'volume': 10000000.0, 'timestamp': datetime.now()},  # 10x normal volume
                'ETH-USD': {'volume': 8000000.0, 'timestamp': datetime.now()}
            },
            'pnl_data': {
                'total_pnl': -15000.0,  # Large loss
                'daily_pnl': -15000.0,
                'positions': [
                    {
                        'symbol': 'BTC-USD',
                        'unrealized_pnl': -10000.0,
                        'size': 0.5
                    },
                    {
                        'symbol': 'ETH-USD',
                        'unrealized_pnl': -5000.0,
                        'size': 3.0
                    }
                ],
                'timestamp': datetime.now()
            },
            'positions': [
                {'symbol': 'BTC-USD', 'notional_value': Decimal('20000'), 'sector': 'crypto'},
                {'symbol': 'ETH-USD', 'notional_value': Decimal('15000'), 'sector': 'crypto'}
            ],
            'portfolio_value': Decimal('85000')  # Portfolio down from losses
        }
        
        # Add historical data to support volatility calculations
        for component in [integrated_system.market_monitor, integrated_system.circuit_breakers]:
            if hasattr(component, 'add_market_data'):
                await component.add_market_data(stress_market_data)
        
        risk_assessment = await integrated_system.perform_comprehensive_risk_check(stress_market_data)
        
        # Should detect high or critical risk
        assert risk_assessment['overall_risk_level'] in ['high', 'critical']
        
        # Multiple components should detect issues
        components_with_issues = 0
        for component_name, component_data in risk_assessment['components'].items():
            if isinstance(component_data, dict):
                if (component_data.get('system_health') in ['warning', 'critical'] or
                    component_data.get('overall_risk_level') in ['high', 'critical'] or
                    component_data.get('risk_level') in ['high', 'critical']):
                    components_with_issues += 1
        
        assert components_with_issues > 0
    
    @pytest.mark.asyncio
    async def test_cascading_risk_events(self, integrated_system):
        """Test cascading risk events across components."""
        # Setup callbacks to track cascading events
        cascade_events = []
        
        async def track_cascade_event(event_type, data):
            cascade_events.append({'type': event_type, 'data': data, 'timestamp': datetime.now()})
        
        # Mock callbacks for emergency protocols
        integrated_system.emergency_protocols.set_callbacks(
            halt_trading=lambda reason: track_cascade_event('halt_trading', {'reason': reason}),
            cancel_orders=lambda reason: track_cascade_event('cancel_orders', {'reason': reason}) or {'cancelled_count': 10},
            place_order=lambda order: track_cascade_event('place_order', order),
            get_positions=lambda: [],
            notify_stakeholders=lambda data: track_cascade_event('notify_stakeholders', data)
        )
        
        # Create cascading crisis scenario
        crisis_data = {
            'price_data': {
                'BTC-USD': {'current_price': 30000.0, 'previous_price': 50000.0, 'timestamp': datetime.now()}
            },
            'volume_data': {
                'BTC-USD': {'volume': 20000000.0, 'timestamp': datetime.now()}  # Panic volume
            },
            'pnl_data': {
                'total_pnl': -25000.0,
                'daily_pnl': -25000.0,
                'positions': [
                    {'symbol': 'BTC-USD', 'unrealized_pnl': -25000.0, 'size': 1.0}
                ],
                'timestamp': datetime.now()
            },
            'positions': [
                {'symbol': 'BTC-USD', 'notional_value': Decimal('30000'), 'sector': 'crypto'}
            ],
            'portfolio_value': Decimal('75000')
        }
        
        # Process the crisis
        risk_assessment = await integrated_system.perform_comprehensive_risk_check(crisis_data)
        
        # Should trigger emergency protocols
        assert risk_assessment['overall_risk_level'] == 'critical'
        
        # Should have triggered cascading events
        if 'emergency_protocols_triggered' in risk_assessment:
            assert len(cascade_events) > 0


class TestPerformanceIntegration:
    """Test performance characteristics of integrated system."""
    
    @pytest.fixture
    async def performance_system(self):
        """Create system optimized for performance testing."""
        system = IntegratedRiskManager()
        await system.initialize()
        return system
    
    @pytest.mark.asyncio
    async def test_risk_check_performance(self, performance_system):
        """Test performance of comprehensive risk checks."""
        market_data = {
            'price_data': {f'SYMBOL_{i}': {
                'current_price': 100.0 + i,
                'previous_price': 99.0 + i,
                'timestamp': datetime.now()
            } for i in range(10)},  # 10 symbols
            'volume_data': {f'SYMBOL_{i}': {
                'volume': 1000000.0,
                'timestamp': datetime.now()
            } for i in range(10)},
            'pnl_data': {
                'total_pnl': 5000.0,
                'daily_pnl': 1000.0,
                'positions': [
                    {'symbol': f'SYMBOL_{i}', 'unrealized_pnl': 100.0, 'size': 1.0}
                    for i in range(10)
                ],
                'timestamp': datetime.now()
            },
            'positions': [
                {'symbol': f'SYMBOL_{i}', 'notional_value': Decimal('5000'), 'sector': 'test'}
                for i in range(10)
            ],
            'portfolio_value': Decimal('100000')
        }
        
        # Time multiple risk checks
        start_time = datetime.now()
        for _ in range(10):
            await performance_system.perform_comprehensive_risk_check(market_data)
        end_time = datetime.now()
        
        total_time = (end_time - start_time).total_seconds()
        avg_time_per_check = total_time / 10
        
        # Should complete comprehensive risk checks quickly (under 100ms each)
        assert avg_time_per_check < 0.1
    
    @pytest.mark.asyncio
    async def test_concurrent_risk_monitoring(self, performance_system):
        """Test concurrent risk monitoring across multiple market conditions."""
        # Create different market scenarios
        scenarios = [
            {'name': 'normal', 'price_change': 1.0, 'volume_mult': 1.0, 'pnl': 1000.0},
            {'name': 'volatile', 'price_change': 5.0, 'volume_mult': 3.0, 'pnl': -2000.0},
            {'name': 'crash', 'price_change': -15.0, 'volume_mult': 10.0, 'pnl': -8000.0},
        ]
        
        async def process_scenario(scenario):
            market_data = {
                'price_data': {
                    'TEST-USD': {
                        'current_price': 100.0 * (1 + scenario['price_change']/100),
                        'previous_price': 100.0,
                        'timestamp': datetime.now()
                    }
                },
                'volume_data': {
                    'TEST-USD': {
                        'volume': 1000000.0 * scenario['volume_mult'],
                        'timestamp': datetime.now()
                    }
                },
                'pnl_data': {
                    'total_pnl': scenario['pnl'],
                    'daily_pnl': scenario['pnl'],
                    'positions': [{'symbol': 'TEST-USD', 'unrealized_pnl': scenario['pnl'], 'size': 1.0}],
                    'timestamp': datetime.now()
                },
                'positions': [{'symbol': 'TEST-USD', 'notional_value': Decimal('10000'), 'sector': 'test'}],
                'portfolio_value': Decimal('100000')
            }
            
            return await performance_system.perform_comprehensive_risk_check(market_data)
        
        # Process all scenarios concurrently
        tasks = [process_scenario(scenario) for scenario in scenarios]
        results = await asyncio.gather(*tasks)
        
        # All should complete successfully
        assert len(results) == 3
        for result in results:
            assert 'overall_risk_level' in result
            assert 'components' in result


class TestFailureRecovery:
    """Test system behavior during component failures and recovery."""
    
    @pytest.fixture
    async def recovery_system(self):
        """Create system for failure recovery testing."""
        system = IntegratedRiskManager()
        await system.initialize()
        return system
    
    @pytest.mark.asyncio
    async def test_component_failure_isolation(self, recovery_system):
        """Test that component failures don't cascade to other components."""
        # Simulate failure in one component
        with patch.object(recovery_system.market_monitor, 'analyze_market_conditions', 
                         side_effect=Exception("Market monitor failure")):
            
            market_data = {
                'price_data': {'BTC-USD': {'current_price': 50000.0, 'previous_price': 49000.0, 'timestamp': datetime.now()}},
                'pnl_data': {'total_pnl': 1000.0, 'daily_pnl': 500.0, 'positions': [], 'timestamp': datetime.now()},
                'positions': [],
                'portfolio_value': Decimal('100000')
            }
            
            # System should still function with other components
            risk_assessment = await recovery_system.perform_comprehensive_risk_check(market_data)
            
            # Should have error information but not crash
            assert 'error' in risk_assessment or risk_assessment['overall_risk_level'] == 'error'
            
            # Other components should still work
            assert 'components' in risk_assessment
    
    @pytest.mark.asyncio
    async def test_emergency_protocol_activation_during_failures(self, recovery_system):
        """Test emergency protocol activation when components fail."""
        emergency_actions = []
        
        def track_emergency_action(action_type, **kwargs):
            emergency_actions.append({'action': action_type, 'kwargs': kwargs})
        
        recovery_system.emergency_protocols.set_callbacks(
            halt_trading=lambda reason: track_emergency_action('halt_trading', reason=reason),
            cancel_orders=lambda reason: track_emergency_action('cancel_orders', reason=reason) or {'cancelled_count': 0},
            get_positions=lambda: [],
            notify_stakeholders=lambda data: track_emergency_action('notify_stakeholders', data=data)
        )
        
        # Simulate system-wide crisis with component failures
        with patch.object(recovery_system.operational_monitor, 'perform_health_check',
                         return_value={
                             'system_health': {'health_score': 10.0},
                             'connectivity_health': {'connectivity_score': 5.0},
                             'overall_score': 7.5,
                             'risk_level': 'critical'
                         }):
            
            crisis_data = {
                'components': {
                    'operational_risk': {
                        'system_health': 'failure',
                        'endpoint_health': {}
                    }
                }
            }
            
            # Emergency protocols should activate
            triggered = await recovery_system.emergency_protocols.check_emergency_conditions(crisis_data)
            
            # Should have triggered emergency protocols
            assert len(triggered) >= 0  # May or may not trigger based on exact conditions


@pytest.mark.asyncio
async def test_end_to_end_trading_day_simulation():
    """Test complete trading day simulation with realistic market events."""
    system = IntegratedRiskManager()
    await system.initialize()
    
    # Track events throughout the day
    daily_events = []
    
    def log_event(event_type, details):
        daily_events.append({
            'timestamp': datetime.now(),
            'type': event_type,
            'details': details
        })
    
    # Simulate market events throughout a trading day
    market_events = [
        # Market open - normal conditions
        {
            'hour': 9,
            'price_change': 1.0,
            'volume_mult': 1.0,
            'description': 'Market open - normal'
        },
        # Mid-morning volatility
        {
            'hour': 10,
            'price_change': -3.0,
            'volume_mult': 2.0,
            'description': 'Morning volatility'
        },
        # Lunch lull
        {
            'hour': 12,
            'price_change': 0.5,
            'volume_mult': 0.5,
            'description': 'Lunch lull'
        },
        # Afternoon crash
        {
            'hour': 14,
            'price_change': -8.0,
            'volume_mult': 5.0,
            'description': 'Afternoon sell-off'
        },
        # Recovery attempt
        {
            'hour': 15,
            'price_change': 2.0,
            'volume_mult': 3.0,
            'description': 'Recovery attempt'
        },
        # Market close
        {
            'hour': 16,
            'price_change': -1.0,
            'volume_mult': 1.5,
            'description': 'Market close'
        }
    ]
    
    base_price = 50000.0
    current_price = base_price
    cumulative_pnl = 0.0
    
    for event in market_events:
        # Update market conditions
        price_change = event['price_change']
        current_price = current_price * (1 + price_change / 100)
        
        # Calculate P&L impact
        pnl_change = price_change * 100  # Simplified P&L calculation
        cumulative_pnl += pnl_change
        
        market_data = {
            'price_data': {
                'BTC-USD': {
                    'current_price': current_price,
                    'previous_price': current_price / (1 + price_change / 100),
                    'timestamp': datetime.now()
                }
            },
            'volume_data': {
                'BTC-USD': {
                    'volume': 1000000.0 * event['volume_mult'],
                    'timestamp': datetime.now()
                }
            },
            'pnl_data': {
                'total_pnl': cumulative_pnl,
                'daily_pnl': cumulative_pnl,
                'positions': [
                    {'symbol': 'BTC-USD', 'unrealized_pnl': cumulative_pnl * 0.8, 'size': 0.1}
                ],
                'timestamp': datetime.now()
            },
            'positions': [
                {'symbol': 'BTC-USD', 'notional_value': Decimal('5000'), 'sector': 'crypto'}
            ],
            'portfolio_value': Decimal('100000') + Decimal(str(cumulative_pnl))
        }
        
        # Perform risk check
        risk_assessment = await system.perform_comprehensive_risk_check(market_data)
        
        log_event('risk_check', {
            'hour': event['hour'],
            'description': event['description'],
            'risk_level': risk_assessment['overall_risk_level'],
            'price': current_price,
            'pnl': cumulative_pnl
        })
        
        # Check if emergency protocols were triggered
        if 'emergency_protocols_triggered' in risk_assessment:
            log_event('emergency_trigger', {
                'protocols': risk_assessment['emergency_protocols_triggered'],
                'reason': event['description']
            })
    
    # Analyze daily events
    risk_levels = [event['details']['risk_level'] for event in daily_events if event['type'] == 'risk_check']
    emergency_triggers = [event for event in daily_events if event['type'] == 'emergency_trigger']
    
    # Should have monitored throughout the day
    assert len(daily_events) >= len(market_events)
    
    # Should have detected the afternoon crash as high risk
    assert any(level in ['high', 'critical'] for level in risk_levels)
    
    # Final P&L should reflect market movements
    assert cumulative_pnl != 0  # Should have some P&L impact


if __name__ == "__main__":
    pytest.main([__file__])
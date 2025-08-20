"""
Test utilities for risk management test suite.

Common utilities, mocks, and helper functions for testing risk components.
"""

import asyncio
import random
from decimal import Decimal
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from unittest.mock import AsyncMock, Mock


def create_mock_market_data(
    symbols: List[str] = None,
    base_price: float = 50000.0,
    price_change_pct: float = 0.0,
    volume_multiplier: float = 1.0,
    timestamp: Optional[datetime] = None
) -> Dict[str, Any]:
    """Create mock market data for testing."""
    if symbols is None:
        symbols = ['BTC-USD', 'ETH-USD']
    
    if timestamp is None:
        timestamp = datetime.now()
    
    price_data = {}
    volume_data = {}
    
    for i, symbol in enumerate(symbols):
        current_price = base_price * (1 + i * 0.1) * (1 + price_change_pct / 100)
        previous_price = base_price * (1 + i * 0.1)
        
        price_data[symbol] = {
            'current_price': current_price,
            'previous_price': previous_price,
            'bid': current_price * 0.999,
            'ask': current_price * 1.001,
            'timestamp': timestamp
        }
        
        volume_data[symbol] = {
            'volume': 1000000.0 * volume_multiplier * (1 + i * 0.2),
            'bid_depth': 500000.0,
            'ask_depth': 480000.0,
            'timestamp': timestamp
        }
    
    return {
        'price_data': price_data,
        'volume_data': volume_data,
        'timestamp': timestamp
    }


def create_mock_position_data(
    symbols: List[str] = None,
    portfolio_value: Decimal = Decimal('100000'),
    position_size_pct: float = 5.0
) -> List[Dict[str, Any]]:
    """Create mock position data for testing."""
    if symbols is None:
        symbols = ['BTC-USD', 'ETH-USD', 'AAPL']
    
    positions = []
    
    for symbol in symbols:
        sector = 'crypto' if 'USD' in symbol else 'equity'
        notional_value = portfolio_value * Decimal(str(position_size_pct / 100))
        
        positions.append({
            'symbol': symbol,
            'notional_value': notional_value,
            'sector': sector,
            'side': random.choice(['long', 'short']),
            'entry_price': Decimal(str(random.uniform(100, 60000))),
            'current_price': Decimal(str(random.uniform(100, 60000))),
            'size': Decimal(str(random.uniform(0.1, 10.0)))
        })
    
    return positions


def create_mock_pnl_data(
    total_pnl: float = 1000.0,
    daily_pnl: float = 500.0,
    positions: List[Dict[str, Any]] = None,
    timestamp: Optional[datetime] = None
) -> Dict[str, Any]:
    """Create mock P&L data for testing."""
    if timestamp is None:
        timestamp = datetime.now()
    
    if positions is None:
        positions = [
            {
                'symbol': 'BTC-USD',
                'unrealized_pnl': Decimal(str(total_pnl * 0.6)),
                'realized_pnl': Decimal(str(total_pnl * 0.4)),
                'size': Decimal('0.1'),
                'side': 'long'
            }
        ]
    
    unrealized_total = sum(pos.get('unrealized_pnl', Decimal('0')) for pos in positions)
    realized_total = sum(pos.get('realized_pnl', Decimal('0')) for pos in positions)
    
    return {
        'positions': positions,
        'total_unrealized_pnl': unrealized_total,
        'total_realized_pnl': realized_total,
        'total_pnl': unrealized_total + realized_total,
        'daily_pnl': Decimal(str(daily_pnl)),
        'weekly_pnl': Decimal(str(daily_pnl * 5)),
        'monthly_pnl': Decimal(str(daily_pnl * 20)),
        'timestamp': timestamp
    }


class MockRiskComponent:
    """Mock risk component for testing."""
    
    def __init__(self, name: str = "MockComponent"):
        self.name = name
        self.enabled = False
        self.initialized = False
        self.alerts_sent = []
        self.callbacks = {}
        self.metrics = {}
        
    async def initialize(self):
        """Initialize mock component."""
        self.initialized = True
        self.enabled = True
        
    def set_alert_callback(self, callback):
        """Set alert callback."""
        self.callbacks['alert'] = callback
        
    async def send_alert(self, alert_data):
        """Send mock alert."""
        self.alerts_sent.append(alert_data)
        if 'alert' in self.callbacks:
            await self.callbacks['alert'](alert_data)
    
    def get_status(self) -> Dict[str, Any]:
        """Get component status."""
        return {
            'name': self.name,
            'enabled': self.enabled,
            'initialized': self.initialized,
            'alerts_sent_count': len(self.alerts_sent),
            'metrics': self.metrics
        }


def create_stress_test_data(scenario: str = 'market_crash') -> Dict[str, Any]:
    """Create stress test data for various scenarios."""
    scenarios = {
        'market_crash': {
            'price_change_pct': -20.0,
            'volume_multiplier': 10.0,
            'pnl_impact': -15000.0,
            'portfolio_impact': -0.15
        },
        'flash_crash': {
            'price_change_pct': -35.0,
            'volume_multiplier': 20.0,
            'pnl_impact': -25000.0,
            'portfolio_impact': -0.25
        },
        'liquidity_crisis': {
            'price_change_pct': -8.0,
            'volume_multiplier': 0.2,
            'pnl_impact': -8000.0,
            'portfolio_impact': -0.08
        },
        'volatility_spike': {
            'price_change_pct': 15.0,
            'volume_multiplier': 5.0,
            'pnl_impact': 8000.0,
            'portfolio_impact': 0.08
        },
        'normal_trading': {
            'price_change_pct': 2.0,
            'volume_multiplier': 1.0,
            'pnl_impact': 1000.0,
            'portfolio_impact': 0.01
        }
    }
    
    if scenario not in scenarios:
        scenario = 'normal_trading'
    
    config = scenarios[scenario]
    
    # Create comprehensive stress test data
    market_data = create_mock_market_data(
        symbols=['BTC-USD', 'ETH-USD', 'AAPL', 'TSLA'],
        price_change_pct=config['price_change_pct'],
        volume_multiplier=config['volume_multiplier']
    )
    
    portfolio_value = Decimal('100000') * (1 + Decimal(str(config['portfolio_impact'])))
    positions = create_mock_position_data(
        symbols=['BTC-USD', 'ETH-USD', 'AAPL', 'TSLA'],
        portfolio_value=portfolio_value
    )
    
    pnl_data = create_mock_pnl_data(
        total_pnl=config['pnl_impact'],
        daily_pnl=config['pnl_impact'],
        positions=[
            {
                'symbol': pos['symbol'],
                'unrealized_pnl': Decimal(str(config['pnl_impact'] / len(positions))),
                'realized_pnl': Decimal('0'),
                'size': pos['size'],
                'side': pos['side']
            }
            for pos in positions
        ]
    )
    
    return {
        'scenario': scenario,
        'market_data': market_data,
        'positions': positions,
        'pnl_data': pnl_data,
        'portfolio_value': portfolio_value,
        'expected_risk_level': _determine_expected_risk_level(scenario)
    }


def _determine_expected_risk_level(scenario: str) -> str:
    """Determine expected risk level for scenario."""
    risk_levels = {
        'market_crash': 'critical',
        'flash_crash': 'critical', 
        'liquidity_crisis': 'high',
        'volatility_spike': 'medium',
        'normal_trading': 'normal'
    }
    return risk_levels.get(scenario, 'normal')


async def simulate_market_session(
    duration_minutes: int = 60,
    update_interval_seconds: int = 30,
    volatility_level: str = 'normal'
) -> List[Dict[str, Any]]:
    """Simulate a market session with realistic price movements."""
    
    volatility_configs = {
        'low': {'price_std': 0.5, 'volume_std': 0.2},
        'normal': {'price_std': 1.0, 'volume_std': 0.5},
        'high': {'price_std': 2.0, 'volume_std': 1.0},
        'extreme': {'price_std': 5.0, 'volume_std': 2.0}
    }
    
    config = volatility_configs.get(volatility_level, volatility_configs['normal'])
    
    session_data = []
    updates_count = (duration_minutes * 60) // update_interval_seconds
    
    base_price = 50000.0
    current_price = base_price
    
    for i in range(updates_count):
        # Generate price movement
        price_change = random.gauss(0, config['price_std'])
        current_price = current_price * (1 + price_change / 100)
        
        # Generate volume
        base_volume = 1000000.0
        volume_change = random.gauss(1.0, config['volume_std'])
        current_volume = base_volume * max(0.1, volume_change)
        
        # Create market snapshot
        timestamp = datetime.now() + timedelta(seconds=i * update_interval_seconds)
        
        market_snapshot = {
            'timestamp': timestamp,
            'price': current_price,
            'price_change_pct': price_change,
            'volume': current_volume,
            'session_minute': i * update_interval_seconds // 60
        }
        
        session_data.append(market_snapshot)
    
    return session_data


class PerformanceTracker:
    """Track performance metrics during testing."""
    
    def __init__(self):
        self.metrics = {}
        self.start_times = {}
        
    def start_timing(self, operation: str):
        """Start timing an operation."""
        self.start_times[operation] = datetime.now()
    
    def end_timing(self, operation: str):
        """End timing an operation and record duration."""
        if operation in self.start_times:
            duration = (datetime.now() - self.start_times[operation]).total_seconds()
            
            if operation not in self.metrics:
                self.metrics[operation] = []
            
            self.metrics[operation].append(duration)
            del self.start_times[operation]
            
            return duration
        return None
    
    def get_average_time(self, operation: str) -> Optional[float]:
        """Get average time for an operation."""
        if operation in self.metrics and self.metrics[operation]:
            return sum(self.metrics[operation]) / len(self.metrics[operation])
        return None
    
    def get_max_time(self, operation: str) -> Optional[float]:
        """Get maximum time for an operation."""
        if operation in self.metrics and self.metrics[operation]:
            return max(self.metrics[operation])
        return None
    
    def get_performance_summary(self) -> Dict[str, Dict[str, float]]:
        """Get performance summary for all operations."""
        summary = {}
        
        for operation, times in self.metrics.items():
            if times:
                summary[operation] = {
                    'count': len(times),
                    'average_ms': (sum(times) / len(times)) * 1000,
                    'max_ms': max(times) * 1000,
                    'min_ms': min(times) * 1000
                }
        
        return summary


def create_integration_test_callbacks() -> Dict[str, AsyncMock]:
    """Create mock callbacks for integration testing."""
    return {
        'halt_trading': AsyncMock(),
        'cancel_orders': AsyncMock(return_value={'cancelled_count': 10}),
        'place_order': AsyncMock(return_value={'status': 'filled', 'order_id': 'test_123'}),
        'get_positions': AsyncMock(return_value=[]),
        'notify_stakeholders': AsyncMock(),
        'execute_order': AsyncMock(return_value={'status': 'executed'}),
        'get_market_data': AsyncMock(return_value=create_mock_market_data()),
        'get_portfolio_value': AsyncMock(return_value=Decimal('100000')),
        'emergency_shutdown': AsyncMock()
    }


async def wait_for_condition(
    condition_func,
    timeout_seconds: float = 5.0,
    check_interval: float = 0.1
) -> bool:
    """Wait for a condition to become true."""
    start_time = datetime.now()
    
    while (datetime.now() - start_time).total_seconds() < timeout_seconds:
        if await condition_func() if asyncio.iscoroutinefunction(condition_func) else condition_func():
            return True
        await asyncio.sleep(check_interval)
    
    return False


def assert_risk_metrics_valid(metrics: Dict[str, Any]):
    """Assert that risk metrics are valid."""
    required_fields = ['timestamp', 'overall_risk_level']
    
    for field in required_fields:
        assert field in metrics, f"Missing required field: {field}"
    
    # Risk level should be valid
    valid_risk_levels = ['normal', 'low', 'medium', 'high', 'critical']
    assert metrics['overall_risk_level'] in valid_risk_levels
    
    # Timestamp should be recent (within last hour)
    if isinstance(metrics['timestamp'], str):
        timestamp = datetime.fromisoformat(metrics['timestamp'].replace('Z', '+00:00'))
    else:
        timestamp = metrics['timestamp']
    
    time_diff = (datetime.now() - timestamp.replace(tzinfo=None)).total_seconds()
    assert time_diff < 3600, "Timestamp is too old"


def create_test_config_override(overrides: Dict[str, Any] = None) -> Dict[str, Any]:
    """Create test configuration with overrides."""
    base_config = {
        'risk_limits': {
            'max_position_pct': 10.0,
            'max_daily_loss': 5000.0,
            'max_drawdown_pct': 15.0
        },
        'circuit_breakers': {
            'price_change_threshold': 5.0,
            'volume_threshold_multiplier': 5.0,
            'latency_threshold_ms': 100.0
        },
        'emergency_protocols': {
            'auto_execute': True,
            'max_execution_time': 300
        },
        'monitoring': {
            'check_interval': 30.0,
            'alert_thresholds': {
                'cpu_pct': 80.0,
                'memory_pct': 85.0,
                'disk_pct': 90.0
            }
        }
    }
    
    if overrides:
        # Deep merge overrides
        def deep_merge(base, override):
            for key, value in override.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    deep_merge(base[key], value)
                else:
                    base[key] = value
        
        deep_merge(base_config, overrides)
    
    return base_config
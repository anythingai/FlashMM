"""
Risk Management Test Suite

Comprehensive test suite for FlashMM enterprise risk management system.
"""

# Test utilities
from .test_utils import (
    MockRiskComponent,
    create_mock_market_data,
    create_mock_pnl_data,
    create_mock_position_data,
)

# Test configuration
TEST_CONFIG = {
    'default_timeout': 30.0,
    'test_portfolio_value': 100000.0,
    'test_symbols': ['BTC-USD', 'ETH-USD', 'AAPL', 'TSLA'],
    'mock_data_enabled': True,
    'performance_test_enabled': True,
    'integration_test_enabled': True
}

__all__ = [
    'TEST_CONFIG',
    'create_mock_market_data',
    'create_mock_position_data',
    'create_mock_pnl_data',
    'MockRiskComponent'
]

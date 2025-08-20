"""
Pytest configuration and shared fixtures for FlashMM tests.
"""

import pytest
import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock
from typing import Dict, Any

# Configure test logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_config():
    """Mock configuration for testing."""
    return {
        "trading": {
            "symbols": ["SEI/USDC"],
            "target_cycle_time_ms": 200,
            "max_cycle_time_ms": 500,
            "max_inventory_usdc": 2000.0,
            "max_inventory_ratio": 0.02,
            "warning_inventory_ratio": 0.015,
            "enable_ml_predictions": True,
            "enable_position_tracking": True,
            "enable_performance_monitoring": True,
            "position_update_interval_seconds": 10,
            "enable_redis_persistence": False  # Disable for testing
        },
        "ml": {
            "azure_openai_endpoint": "test_endpoint",
            "azure_openai_key": "test_key",
            "model_name": "o4-mini",
            "enable_fallback": True,
            "max_retries": 3,
            "timeout_seconds": 30
        },
        "optimization": {
            "min_improvement_pct": 20.0,
            "target_improvement_pct": 40.0,
            "max_spread_bps": 30.0,
            "min_spread_bps": 1.0,
            "mode": "adaptive"
        },
        "monitoring": {
            "enable_redis_publishing": False,  # Disable for testing
            "publishing_interval_seconds": 10,
            "metric_retention_hours": 24
        }
    }


@pytest.fixture
def mock_redis_client():
    """Mock Redis client for testing."""
    client = MagicMock()
    client.initialize = AsyncMock()
    client.close = AsyncMock()
    client.set = AsyncMock()
    client.get = AsyncMock(return_value=None)
    client.keys = AsyncMock(return_value=[])
    client.delete = AsyncMock()
    client.publish = AsyncMock()
    return client


@pytest.fixture
def sample_market_data():
    """Sample market data for testing."""
    return {
        "SEI/USDC": {
            "best_bid": 0.4995,
            "best_ask": 0.5005,
            "mid_price": 0.5000,
            "volume_24h": 1000000,
            "price_change_24h": 0.01,
            "order_book": {
                "bids": [
                    {"price": 0.4995, "size": 1000},
                    {"price": 0.4990, "size": 2000},
                    {"price": 0.4985, "size": 1500},
                    {"price": 0.4980, "size": 3000},
                    {"price": 0.4975, "size": 2500}
                ],
                "asks": [
                    {"price": 0.5005, "size": 1000},
                    {"price": 0.5010, "size": 2000},
                    {"price": 0.5015, "size": 1500},
                    {"price": 0.5020, "size": 3000},
                    {"price": 0.5025, "size": 2500}
                ]
            },
            "recent_trades": [
                {"price": 0.5000, "size": 500, "timestamp": "2024-01-01T12:00:00Z", "side": "buy"},
                {"price": 0.4998, "size": 300, "timestamp": "2024-01-01T12:01:00Z", "side": "sell"},
                {"price": 0.5002, "size": 800, "timestamp": "2024-01-01T12:02:00Z", "side": "buy"}
            ]
        }
    }


@pytest.fixture
def sample_ml_prediction():
    """Sample ML prediction for testing."""
    return {
        "symbol": "SEI/USDC",
        "prediction_horizon_minutes": 5,
        "predicted_price": 0.5010,
        "confidence": 0.75,
        "predicted_direction": 1,
        "predicted_magnitude": 0.002,
        "features_used": ["price", "volume", "volatility"],
        "model_version": "1.0.0",
        "timestamp": "2024-01-01T12:00:00Z"
    }


@pytest.fixture
def sample_position_data():
    """Sample position data for testing."""
    return {
        "symbol": "SEI/USDC",
        "base_balance": 100.0,
        "quote_balance": 50.0,
        "average_price": 0.4998,
        "mark_price": 0.5000,
        "notional_value": 50.0,
        "unrealized_pnl": 0.20,
        "realized_pnl": 5.50,
        "total_pnl": 5.70,
        "recent_fills": 15,
        "recent_volume": 50000.0,
        "recent_pnl": 25.50,
        "last_updated": "2024-01-01T12:00:00Z"
    }


@pytest.fixture
def performance_thresholds():
    """Performance thresholds for testing."""
    return {
        "max_cycle_time_ms": 200,
        "max_emergency_cycle_time_ms": 500,
        "max_ml_prediction_time_ms": 50,
        "max_quote_generation_time_ms": 20,
        "max_order_placement_time_ms": 30,
        "max_position_update_time_ms": 10,
        "max_inventory_ratio": 0.02,
        "min_spread_improvement_pct": 20.0,
        "target_spread_improvement_pct": 40.0,
        "min_uptime_ratio": 0.95
    }


# Pytest configuration
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "asyncio: mark test as async"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "unit: mark test as unit test"  
    )
    config.addinivalue_line(
        "markers", "performance: mark test as performance test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


def pytest_collection_modifyitems(config, items):
    """Add default markers to tests based on location."""
    for item in items:
        # Add integration marker to integration tests
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        
        # Add unit marker to unit tests
        elif "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        
        # Add performance marker to performance tests
        if "performance" in item.name.lower() or "benchmark" in item.name.lower():
            item.add_marker(pytest.mark.performance)
        
        # Add slow marker to tests that take longer than 5 seconds
        if "sustained" in item.name.lower() or "long" in item.name.lower():
            item.add_marker(pytest.mark.slow)


# Async test configuration
pytest_plugins = ['pytest_asyncio']
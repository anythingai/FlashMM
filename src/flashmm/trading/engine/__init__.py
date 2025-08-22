"""
FlashMM Trading Engine

High-performance market making engine that orchestrates ML predictions, quote generation,
order management, risk controls, and state management to achieve ≥40% spread improvement
with ±2% inventory control.
"""

from .market_making_engine import (
    MarketMakingEngine,
    TradingMetrics,
    cleanup_market_making_engine,
    get_market_making_engine,
)

__all__ = [
    'MarketMakingEngine',
    'TradingMetrics',
    'get_market_making_engine',
    'cleanup_market_making_engine'
]

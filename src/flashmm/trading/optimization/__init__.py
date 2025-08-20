"""
FlashMM Quote Optimization

Advanced quote optimization system with adaptive spread sizing, competition analysis,
and ML-driven optimization to achieve ≥40% spread improvement vs baseline.
"""

from .quote_optimizer import (
    MarketCondition,
    MarketMetrics,
    OptimizationResult,
    MarketConditionAnalyzer,
    CompetitionAnalyzer,
    MLSpreadPredictor,
    QuoteOptimizer,
    get_quote_optimizer,
    cleanup_quote_optimizer
)

__all__ = [
    'MarketCondition',
    'MarketMetrics',
    'OptimizationResult',
    'MarketConditionAnalyzer',
    'CompetitionAnalyzer',
    'MLSpreadPredictor',
    'QuoteOptimizer',
    'get_quote_optimizer',
    'cleanup_quote_optimizer'
]
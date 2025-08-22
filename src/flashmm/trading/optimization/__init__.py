"""
FlashMM Quote Optimization

Advanced quote optimization system with adaptive spread sizing, competition analysis,
and ML-driven optimization to achieve ≥40% spread improvement vs baseline.
"""

from .quote_optimizer import (
    CompetitionAnalyzer,
    MarketCondition,
    MarketConditionAnalyzer,
    MarketMetrics,
    MLSpreadPredictor,
    OptimizationResult,
    QuoteOptimizer,
    cleanup_quote_optimizer,
    get_quote_optimizer,
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

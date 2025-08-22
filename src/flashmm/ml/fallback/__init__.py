"""
FlashMM ML Fallback Module

Fallback prediction engines for high availability.
"""

from .rule_based_engine import RuleBasedEngine

__all__ = [
    'RuleBasedEngine'
]

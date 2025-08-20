"""
FlashMM - Predictive On-Chain Market Making Agent for Sei Blockchain

This package contains the core implementation of FlashMM, a high-performance,
AI-driven market-making agent optimized for the Sei blockchain ecosystem.

Author: FlashMM Team
License: MIT
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "FlashMM Team"
__email__ = "team@flashmm.ai"

from flashmm.config.settings import get_config
from flashmm.utils.logging import get_logger

# Initialize global configuration and logging
config = get_config()
logger = get_logger(__name__)

__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "config",
    "logger",
]
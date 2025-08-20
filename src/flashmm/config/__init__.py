"""
FlashMM Configuration Management Module

Provides hierarchical configuration management with support for:
- Default values embedded in code
- Environment-specific YAML files
- Environment variable overrides
- Runtime configuration updates via Redis
"""

from flashmm.config.settings import ConfigManager, get_config
from flashmm.config.constants import DEFAULT_CONFIG
from flashmm.config.environments import detect_environment

__all__ = [
    "ConfigManager",
    "get_config", 
    "DEFAULT_CONFIG",
    "detect_environment",
]
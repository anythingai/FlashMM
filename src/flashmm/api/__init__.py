"""
FlashMM API Module

FastAPI-based REST API and WebSocket endpoints for monitoring and control.
"""

from flashmm.api.app import create_app

__all__ = [
    "create_app",
]
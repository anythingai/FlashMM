"""
FlashMM API Routers

FastAPI routers for different endpoint categories.
"""

from flashmm.api.routers.health import router as health_router
from flashmm.api.routers.metrics import router as metrics_router
from flashmm.api.routers.trading import router as trading_router
from flashmm.api.routers.admin import router as admin_router

__all__ = [
    "health_router",
    "metrics_router",
    "trading_router",
    "admin_router",
]
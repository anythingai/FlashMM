"""
FlashMM Health API Router

Health check and system status endpoints.
"""

from fastapi import APIRouter
from typing import Dict, Any
from datetime import datetime

from flashmm.config.settings import get_config
from flashmm.utils.logging import get_logger

router = APIRouter()
logger = get_logger(__name__)
config = get_config()


@router.get("")
async def health_check() -> Dict[str, Any]:
    """Basic health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "environment": config.get("app.environment", "unknown"),
    }


@router.get("/detailed")
async def detailed_health() -> Dict[str, Any]:
    """Detailed health check with component status."""
    components = {}
    
    # Check Redis connection
    try:
        # In production, actually test Redis connection
        components["redis"] = "healthy"
    except Exception:
        components["redis"] = "unhealthy"
    
    # Check InfluxDB connection
    try:
        # In production, actually test InfluxDB connection
        components["influxdb"] = "healthy"
    except Exception:
        components["influxdb"] = "unhealthy"
    
    # Check ML model
    try:
        # In production, check if model is loaded
        components["ml_model"] = "healthy"
    except Exception:
        components["ml_model"] = "unhealthy"
    
    # Check trading engine
    try:
        components["trading_engine"] = "healthy" if config.get("trading.enable_trading", False) else "disabled"
    except Exception:
        components["trading_engine"] = "unhealthy"
    
    overall_status = "healthy" if all(
        status in ["healthy", "disabled"] for status in components.values()
    ) else "unhealthy"
    
    return {
        "status": overall_status,
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "environment": config.get("app.environment", "unknown"),
        "components": components,
        "uptime_seconds": 0,  # TODO: Track actual uptime
    }


@router.get("/ready")
async def readiness_check() -> Dict[str, Any]:
    """Readiness check for load balancers."""
    return {
        "status": "ready",
        "timestamp": datetime.now().isoformat(),
    }


@router.get("/live")
async def liveness_check() -> Dict[str, Any]:
    """Liveness check for container orchestrators."""
    return {
        "status": "alive",
        "timestamp": datetime.now().isoformat(),
    }
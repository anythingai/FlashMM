"""
FlashMM Admin API Router

Administrative endpoints for system management.
"""

from typing import Any

from fastapi import APIRouter

router = APIRouter()


@router.get("/status")
async def admin_status() -> dict[str, Any]:
    """Get admin system status."""
    return {
        "status": "active",
        "message": "Admin router is operational"
    }


@router.get("/config")
async def get_config() -> dict[str, Any]:
    """Get system configuration (placeholder)."""
    return {
        "message": "Configuration endpoint - implementation needed"
    }

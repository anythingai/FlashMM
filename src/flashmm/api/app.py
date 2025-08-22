"""
FlashMM FastAPI Application

Main FastAPI application with health endpoints, metrics, and WebSocket support.
"""


from datetime import datetime

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from flashmm.api.routers.health import router as health_router
from flashmm.config.settings import get_config

# Import security components
from flashmm.security import (
    AuditEventType,
    AuditLevel,
    AuditLogger,
    MonitoringEvent,
    PolicyEngine,
    SecurityMonitor,
    SecurityOrchestrator,
)
from flashmm.utils.exceptions import AuthenticationError, AuthorizationError, FlashMMError
from flashmm.utils.logging import get_logger

logger = get_logger(__name__)
config = get_config()


def create_app(security_orchestrator: SecurityOrchestrator | None = None,
               security_monitor: SecurityMonitor | None = None,
               audit_logger: AuditLogger | None = None,
               policy_engine: PolicyEngine | None = None) -> FastAPI:
    """Create and configure FastAPI application with comprehensive security integration."""

    app = FastAPI(
        title="FlashMM API",
        description="FlashMM - Predictive On-Chain Market Making Agent with Comprehensive Security",
        version="1.0.0",
        docs_url="/docs" if config.get("app.debug", False) else None,
        redoc_url="/redoc" if config.get("app.debug", False) else None,
    )

    # Store security components in app state
    app.state.security_orchestrator = security_orchestrator
    app.state.security_monitor = security_monitor
    app.state.audit_logger = audit_logger
    app.state.policy_engine = policy_engine

    # Add security middleware
    @app.middleware("http")
    async def security_middleware(request: Request, call_next):
        """Security middleware for request processing."""
        start_time = datetime.utcnow()

        # Extract request information
        client_ip = request.client.host if request.client else None
        user_agent = request.headers.get("user-agent")

        try:
            # Security policy evaluation
            if policy_engine:
                policy_result = await policy_engine.evaluate_event(
                    component="api",
                    action=f"{request.method}:{request.url.path}",
                    context={
                        "method": request.method,
                        "path": str(request.url.path),
                        "headers": dict(request.headers),
                        "source_ip": client_ip,
                        "user_agent": user_agent,
                        "timestamp": start_time.isoformat()
                    },
                    user_id=None  # Will be populated after authentication
                )

                # Block request if policy violation
                if policy_result.get("blocked"):
                    if audit_logger:
                        await audit_logger.log_event(
                            event_type=AuditEventType.API_ACCESS,
                            actor_id="unknown",
                            action="api_request_blocked",
                            component="api_middleware",
                            outcome="blocked",
                            risk_level="medium",
                            metadata={
                                "path": str(request.url.path),
                                "method": request.method,
                                "policy_violations": len(policy_result.get("violations", []))
                            }
                        )

                    return JSONResponse(
                        status_code=403,
                        content={"detail": "Request blocked by security policy"}
                    )

            # Process request
            response = await call_next(request)

            # Log successful API access
            if audit_logger:
                await audit_logger.log_event(
                    event_type=AuditEventType.API_ACCESS,
                    actor_id="api_user",  # Will be enhanced with actual user after auth
                    action="api_request",
                    component="api_middleware",
                    outcome="success",
                    risk_level="low",
                    metadata={
                        "path": str(request.url.path),
                        "method": request.method,
                        "status_code": response.status_code,
                        "response_time_ms": (datetime.utcnow() - start_time).total_seconds() * 1000
                    }
                )

            # Create security monitoring event
            if security_monitor:
                await security_monitor.create_security_event(
                    event_type=MonitoringEvent.API_REQUEST,
                    component="api",
                    action=f"{request.method}:{request.url.path}",
                    source_ip=client_ip,
                    success=200 <= response.status_code < 400,
                    metadata={
                        "method": request.method,
                        "path": str(request.url.path),
                        "status_code": response.status_code,
                        "user_agent": user_agent
                    }
                )

            return response

        except Exception as e:
            # Log security middleware error
            if audit_logger:
                try:
                    await audit_logger.log_event(
                        event_type=AuditEventType.SYSTEM_CONFIGURATION,
                        actor_id="system",
                        action="security_middleware_error",
                        component="api_middleware",
                        outcome="error",
                        risk_level="high",
                        level=AuditLevel.ERROR,
                        metadata={"error": str(e)}
                    )
                except Exception as audit_e:
                    logger.debug(f"Failed to log security middleware error: {audit_e}")

            logger.error(f"Security middleware error: {e}")

            # Continue processing but log the security failure
            return await call_next(request)

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.get("api.cors_origins", ["*"]),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Authentication dependency
    security_scheme = HTTPBearer(auto_error=False)

    async def get_current_user(request: Request,
                              credentials: HTTPAuthorizationCredentials = Depends(security_scheme)):
        """Authentication dependency with security integration."""

        # Handle API key authentication
        api_key = request.headers.get("X-API-Key")
        if api_key and security_orchestrator:
            try:
                auth_result = await security_orchestrator.authenticate_request({
                    "auth_type": "api_key",
                    "api_key": api_key,
                    "source_ip": request.client.host if request.client else None,
                    "user_agent": request.headers.get("user-agent")
                })

                if auth_result.get("authenticated"):
                    return {
                        "user_id": auth_result.get("role", "api_user"),
                        "role": auth_result.get("role"),
                        "authenticated": True,
                        "token": auth_result.get("token")
                    }
            except AuthenticationError:
                pass

        # Handle JWT authentication
        if credentials and security_orchestrator:
            try:
                auth_result = await security_orchestrator.authenticate_request({
                    "auth_type": "jwt",
                    "token": credentials.credentials,
                    "source_ip": request.client.host if request.client else None,
                    "user_agent": request.headers.get("user-agent")
                })

                if auth_result.get("authenticated"):
                    payload = auth_result.get("payload", {})
                    return {
                        "user_id": payload.get("sub"),
                        "role": payload.get("role"),
                        "authenticated": True,
                        "permissions": payload.get("permissions", []),
                        "session_id": payload.get("session_id")
                    }
            except AuthenticationError:
                pass

        # Return unauthenticated user for public endpoints
        return {
            "user_id": "anonymous",
            "role": "anonymous",
            "authenticated": False
        }

    # Add security-aware routers
    app.include_router(health_router, prefix="/health", tags=["health"])

    # Protected routers would include security dependencies
    # app.include_router(metrics_router, prefix="/metrics", tags=["metrics"], dependencies=[Depends(get_current_user)])
    # app.include_router(trading_router, prefix="/trading", tags=["trading"], dependencies=[Depends(get_current_user)])
    # app.include_router(admin_router, prefix="/admin", tags=["admin"], dependencies=[Depends(get_current_user)])

    # Add enhanced exception handlers with security logging
    @app.exception_handler(AuthenticationError)
    async def authentication_exception_handler(request: Request, exc: AuthenticationError):
        """Handle authentication errors with security logging."""

        # Log authentication failure
        if audit_logger:
            await audit_logger.log_event(
                event_type=AuditEventType.AUTHENTICATION,
                actor_id="unknown",
                action="authentication_failure",
                component="api",
                outcome="failure",
                risk_level="medium",
                metadata={
                    "path": str(request.url.path),
                    "error": str(exc),
                    "source_ip": request.client.host if request.client else None
                }
            )

        return JSONResponse(
            status_code=401,
            content={"detail": "Authentication failed", "error": str(exc)}
        )

    @app.exception_handler(AuthorizationError)
    async def authorization_exception_handler(request: Request, exc: AuthorizationError):
        """Handle authorization errors with security logging."""

        # Log authorization failure
        if audit_logger:
            await audit_logger.log_event(
                event_type=AuditEventType.AUTHORIZATION,
                actor_id=getattr(exc, 'user', 'unknown'),
                action="authorization_failure",
                component="api",
                outcome="failure",
                risk_level="high",
                metadata={
                    "path": str(request.url.path),
                    "required_permission": getattr(exc, 'required_permission', 'unknown'),
                    "user_role": getattr(exc, 'user_role', 'unknown'),
                    "source_ip": request.client.host if request.client else None
                }
            )

        return JSONResponse(
            status_code=403,
            content={"detail": "Access denied", "error": str(exc)}
        )

    @app.exception_handler(FlashMMError)
    async def flashmm_exception_handler(request: Request, exc: FlashMMError):
        """Handle FlashMM errors with security logging."""

        # Log application errors
        if audit_logger:
            await audit_logger.log_event(
                event_type=AuditEventType.SYSTEM_CONFIGURATION,
                actor_id="system",
                action="application_error",
                component="api",
                outcome="error",
                risk_level="medium",
                level=AuditLevel.ERROR,
                metadata={
                    "path": str(request.url.path),
                    "error_type": type(exc).__name__,
                    "error": str(exc)
                }
            )

        return JSONResponse(
            status_code=400,
            content=exc.to_dict()
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """Handle HTTP exceptions with logging."""
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail}
        )

    # Security-aware root endpoint
    @app.get("/")
    async def root(request: Request, current_user: dict = Depends(get_current_user)):
        """Root endpoint with security integration."""

        # Log API access
        if audit_logger:
            await audit_logger.log_event(
                event_type=AuditEventType.API_ACCESS,
                actor_id=current_user.get("user_id", "anonymous"),
                action="root_access",
                component="api",
                outcome="success",
                risk_level="low",
                metadata={
                    "authenticated": current_user.get("authenticated", False),
                    "role": current_user.get("role", "anonymous"),
                    "source_ip": request.client.host if request.client else None
                }
            )

        response_data = {
            "name": "FlashMM API",
            "version": "1.0.0",
            "status": "online",
            "security": {
                "authentication": "enabled",
                "monitoring": "active" if security_monitor else "disabled",
                "audit_logging": "active" if audit_logger else "disabled",
                "policy_enforcement": "active" if policy_engine else "disabled"
            }
        }

        # Add security status for authenticated users
        if current_user.get("authenticated") and security_orchestrator:
            try:
                security_status = security_orchestrator.get_security_status()
                response_data["security"]["system_state"] = security_status["security_state"]
                response_data["security"]["active_threats"] = security_status["active_threats"]
            except Exception as e:
                logger.debug(f"Failed to get security status: {e}")

        return response_data

    # Security status endpoint (protected)
    @app.get("/security/status")
    async def security_status(request: Request, current_user: dict = Depends(get_current_user)):
        """Get comprehensive security system status."""

        # Check authorization
        if not current_user.get("authenticated"):
            raise HTTPException(status_code=401, detail="Authentication required")

        if current_user.get("role") not in ["admin", "super_admin"]:
            raise HTTPException(status_code=403, detail="Admin access required")

        # Log security status access
        if audit_logger:
            await audit_logger.log_event(
                event_type=AuditEventType.SECURITY_EVENT,
                actor_id=current_user["user_id"],
                action="security_status_access",
                component="api",
                outcome="success",
                risk_level="medium",
                metadata={
                    "role": current_user.get("role"),
                    "source_ip": request.client.host if request.client else None
                }
            )

        # Compile security status from all components
        status = {
            "timestamp": datetime.utcnow().isoformat(),
            "components": {}
        }

        if security_orchestrator:
            status["components"]["orchestrator"] = security_orchestrator.get_security_status()

        if security_monitor:
            status["components"]["monitor"] = security_monitor.get_monitoring_statistics()

        if policy_engine:
            status["components"]["policies"] = policy_engine.get_policy_status()

        if audit_logger:
            status["components"]["audit"] = audit_logger.get_audit_statistics()

        return status

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.get("api.cors_origins", ["*"]),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add routers (health is public, others may be protected)
    app.include_router(health_router, prefix="/health", tags=["health"])

    # Note: Other routers would be added with security dependencies in production
    # app.include_router(metrics_router, prefix="/metrics", tags=["metrics"], dependencies=[Depends(get_current_user)])
    # app.include_router(trading_router, prefix="/trading", tags=["trading"], dependencies=[Depends(get_current_user)])
    # app.include_router(admin_router, prefix="/admin", tags=["admin"], dependencies=[Depends(get_current_user)])

    return app

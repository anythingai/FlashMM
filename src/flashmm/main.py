"""
FlashMM Application Entry Point

Main application module that initializes and coordinates all components
of the FlashMM market-making system.
"""

import asyncio
import signal
import sys
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI

from flashmm.api.app import create_app
from flashmm.config.settings import get_config
from flashmm.data.ingestion.feed_manager import FeedManager
from flashmm.ml.inference.inference_engine import InferenceEngine
from flashmm.monitoring.telemetry.metrics_collector import MetricsCollector
from flashmm.trading.execution.order_router import OrderRouter
from flashmm.trading.strategy.quoting_strategy import QuotingStrategy
from flashmm.utils.logging import get_logger

# Import comprehensive security components
from flashmm.security import (
    SecurityOrchestrator,
    SecurityMonitor,
    AuditLogger,
    EmergencyManager,
    PolicyEngine,
    EnhancedKeyManager
)

logger = get_logger(__name__)


class FlashMMApplication:
    """Main application class for FlashMM."""
    
    def __init__(self):
        self.config = get_config()
        self.running = False
        
        # Security components (initialized first for system protection)
        self.security_orchestrator: SecurityOrchestrator | None = None
        self.security_monitor: SecurityMonitor | None = None
        self.audit_logger: AuditLogger | None = None
        self.emergency_manager: EmergencyManager | None = None
        self.policy_engine: PolicyEngine | None = None
        self.key_manager: EnhancedKeyManager | None = None
        
        # Core components
        self.feed_manager: FeedManager | None = None
        self.inference_engine: InferenceEngine | None = None
        self.quoting_strategy: QuotingStrategy | None = None
        self.order_router: OrderRouter | None = None
        self.metrics_collector: MetricsCollector | None = None
        
        # FastAPI app
        self.app: FastAPI | None = None
    
    async def initialize(self) -> None:
        """Initialize all application components with comprehensive security."""
        logger.info("Initializing FlashMM application with enhanced security...")
        
        try:
            # Initialize configuration
            await self.config.initialize()
            
            # Initialize security components FIRST (for system protection)
            logger.info("Initializing security infrastructure...")
            
            self.audit_logger = AuditLogger()
            await self.audit_logger.initialize()
            
            self.policy_engine = PolicyEngine()
            await self.policy_engine.initialize()
            
            self.security_monitor = SecurityMonitor()
            await self.security_monitor.start_monitoring()
            
            self.emergency_manager = EmergencyManager()
            await self.emergency_manager.initialize()
            
            self.key_manager = EnhancedKeyManager()
            await self.key_manager.initialize()
            
            self.security_orchestrator = SecurityOrchestrator()
            await self.security_orchestrator.start()
            
            # Log successful security initialization
            await self.audit_logger.log_event(
                event_type=AuditEventType.SYSTEM_CONFIGURATION,
                actor_id="system",
                action="security_initialization",
                component="main_application",
                outcome="success",
                risk_level="low",
                metadata={"components": ["orchestrator", "monitor", "audit", "emergency", "policy", "keys"]}
            )
            
            logger.info("Security infrastructure initialized successfully")
            
            # Initialize core components with security integration
            self.feed_manager = FeedManager()
            await self.feed_manager.initialize()
            
            self.inference_engine = InferenceEngine()
            await self.inference_engine.initialize()
            
            self.quoting_strategy = QuotingStrategy()
            await self.quoting_strategy.initialize()
            
            self.order_router = OrderRouter()
            await self.order_router.initialize()
            
            self.metrics_collector = MetricsCollector()
            await self.metrics_collector.initialize()
            
            # Create FastAPI app with security integration
            self.app = create_app(
                security_orchestrator=self.security_orchestrator,
                security_monitor=self.security_monitor,
                audit_logger=self.audit_logger,
                policy_engine=self.policy_engine
            )
            
            logger.info("FlashMM application initialized successfully with comprehensive security")
            
        except Exception as e:
            logger.error(f"Failed to initialize application: {e}")
            
            # Log initialization failure
            if self.audit_logger:
                try:
                    await self.audit_logger.log_event(
                        event_type=AuditEventType.SYSTEM_CONFIGURATION,
                        actor_id="system",
                        action="application_initialization",
                        component="main_application",
                        outcome="failure",
                        risk_level="high",
                        level=AuditLevel.ERROR,
                        metadata={"error": str(e)}
                    )
                except:
                    pass  # Don't fail if audit logging fails during initialization error
            
            raise
    
    async def start(self) -> None:
        """Start the application and all background tasks."""
        if not self.app:
            raise RuntimeError("Application not initialized")
        
        logger.info("Starting FlashMM application...")
        self.running = True
        
        try:
            # Start background tasks
            await self._start_background_tasks()
            
            # Start web server
            config = uvicorn.Config(
                app=self.app,
                host=self.config.get("api.host", "0.0.0.0"),
                port=self.config.get("api.port", 8000),
                log_level=self.config.get("log_level", "info").lower(),
                access_log=self.config.get("debug", False),
                reload=self.config.get("debug", False),
            )
            
            server = uvicorn.Server(config)
            await server.serve()
            
        except Exception as e:
            logger.error(f"Application startup failed: {e}")
            await self.shutdown()
            raise
    
    async def _start_background_tasks(self) -> None:
        """Start all background tasks."""
        # Start data ingestion
        if self.feed_manager:
            asyncio.create_task(self.feed_manager.start())
        
        # Start metrics collection
        if self.metrics_collector:
            asyncio.create_task(self.metrics_collector.start())
        
        # Start trading strategy
        if self.quoting_strategy:
            asyncio.create_task(self.quoting_strategy.start())
    
    async def shutdown(self) -> None:
        """Graceful shutdown of the application with security audit."""
        if not self.running:
            return
        
        logger.info("Shutting down FlashMM application...")
        self.running = False
        
        try:
            # Log shutdown initiation
            if self.audit_logger:
                await self.audit_logger.log_event(
                    event_type=AuditEventType.SYSTEM_CONFIGURATION,
                    actor_id="system",
                    action="application_shutdown",
                    component="main_application",
                    outcome="initiated",
                    risk_level="low",
                    metadata={"shutdown_type": "graceful"}
                )
            
            # Stop trading first (most critical)
            if self.quoting_strategy:
                await self.quoting_strategy.stop()
            
            if self.order_router:
                await self.order_router.cancel_all_orders()
            
            # Stop data ingestion
            if self.feed_manager:
                await self.feed_manager.stop()
            
            # Stop monitoring
            if self.metrics_collector:
                await self.metrics_collector.stop()
            
            # Shutdown security components last (maintain protection during shutdown)
            if self.security_monitor:
                await self.security_monitor.stop_monitoring()
            
            if self.policy_engine:
                await self.policy_engine.shutdown()
            
            if self.emergency_manager:
                await self.emergency_manager.shutdown()
            
            if self.security_orchestrator:
                await self.security_orchestrator.stop()
            
            # Audit logger shuts down last to capture all events
            if self.audit_logger:
                await self.audit_logger.log_event(
                    event_type=AuditEventType.SYSTEM_CONFIGURATION,
                    actor_id="system",
                    action="application_shutdown",
                    component="main_application",
                    outcome="success",
                    risk_level="low",
                    metadata={"shutdown_duration": "graceful"}
                )
                await self.audit_logger.shutdown()
            
            logger.info("FlashMM application shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
            
            # Emergency shutdown if graceful fails
            if self.emergency_manager:
                try:
                    await self.emergency_manager.declare_emergency(
                        emergency_type=EmergencyType.TECHNICAL_FAILURE,
                        emergency_level=EmergencyLevel.HIGH,
                        description=f"Application shutdown error: {str(e)}",
                        detected_by="system"
                    )
                except:
                    pass  # Don't cascade failures


# Global application instance
app_instance: FlashMMApplication | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """FastAPI lifespan context manager."""
    global app_instance
    
    # Startup
    app_instance = FlashMMApplication()
    await app_instance.initialize()
    
    # Setup signal handlers
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, shutting down...")
        if app_instance:
            asyncio.create_task(app_instance.shutdown())
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    yield
    
    # Shutdown
    if app_instance:
        await app_instance.shutdown()


async def main() -> None:
    """Main application entry point."""
    try:
        app = FlashMMApplication()
        await app.initialize()
        await app.start()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    except Exception as e:
        logger.error(f"Application failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
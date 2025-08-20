"""
FlashMM Monitoring Service Orchestrator

Main orchestrator service that coordinates all monitoring components, provides health checks,
manages service lifecycle, and offers a unified API for the monitoring system.
"""

import asyncio
import json
import time
from typing import Dict, Any, List, Optional, Set, Callable, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
import traceback
from contextlib import asynccontextmanager
import weakref

from flashmm.config.settings import get_config
from flashmm.utils.logging import get_logger
from flashmm.utils.exceptions import ValidationError

# Import monitoring components
from flashmm.monitoring.telemetry.metrics_collector import MetricsCollector
from flashmm.monitoring.alerts.alert_manager import AlertManager
from flashmm.monitoring.analytics.performance_analyzer import PerformanceAnalyzer
from flashmm.monitoring.streaming.data_streamer import DataStreamer
from flashmm.monitoring.dashboards.grafana_client import GrafanaClient
from flashmm.monitoring.dashboards.dashboard_generator import DashboardGenerator
from flashmm.monitoring.social.twitter_client import TwitterClient

logger = get_logger(__name__)


class ServiceStatus(Enum):
    """Service status enumeration."""
    UNKNOWN = "unknown"
    STARTING = "starting"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    FAILED = "failed"
    STOPPING = "stopping"
    STOPPED = "stopped"


class ServiceType(Enum):
    """Service type enumeration."""
    METRICS_COLLECTOR = "metrics_collector"
    ALERT_MANAGER = "alert_manager"
    PERFORMANCE_ANALYZER = "performance_analyzer"
    DATA_STREAMER = "data_streamer"
    GRAFANA_CLIENT = "grafana_client"
    DASHBOARD_GENERATOR = "dashboard_generator"
    TWITTER_CLIENT = "twitter_client"


@dataclass
class ServiceHealth:
    """Service health information."""
    service_name: str
    service_type: ServiceType
    status: ServiceStatus
    last_check: datetime
    uptime_seconds: float
    error_count: int = 0
    warning_count: int = 0
    last_error: Optional[str] = None
    last_warning: Optional[str] = None
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    health_score: float = 1.0  # 0.0 to 1.0


@dataclass
class ServiceDependency:
    """Service dependency configuration."""
    service_name: str
    required: bool = True
    timeout_seconds: int = 30
    retry_count: int = 3
    health_check_interval: int = 60


@dataclass
class MonitoringConfig:
    """Monitoring service configuration."""
    enabled_services: Set[ServiceType] = field(default_factory=set)
    health_check_interval: int = 30
    service_timeout: int = 60
    max_restart_attempts: int = 3
    restart_delay_seconds: int = 5
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: int = 300
    auto_recovery: bool = True
    performance_monitoring: bool = True


class CircuitBreaker:
    """Circuit breaker for service resilience."""
    
    def __init__(self, failure_threshold: int = 5, timeout_seconds: int = 300):
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open
    
    def call(self, func):
        """Call function with circuit breaker protection."""
        if self.state == "open":
            if self._should_attempt_reset():
                self.state = "half-open"
            else:
                raise Exception("Circuit breaker is open")
        
        try:
            result = func()
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset."""
        return (self.last_failure_time and 
                time.time() - self.last_failure_time > self.timeout_seconds)
    
    def _on_success(self):
        """Handle successful call."""
        self.failure_count = 0
        self.state = "closed"
    
    def _on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"


class MonitoringService:
    """Main monitoring service orchestrator."""
    
    def __init__(self, config: Optional[MonitoringConfig] = None):
        self.config = config or self._load_default_config()
        self.app_config = get_config()
        
        # Service registry
        self.services: Dict[str, Any] = {}
        self.service_health: Dict[str, ServiceHealth] = {}
        self.service_dependencies: Dict[str, List[ServiceDependency]] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        
        # Service lifecycle management
        self.startup_order: List[str] = []
        self.shutdown_order: List[str] = []
        self.restart_counts: Dict[str, int] = {}
        self.last_restart_time: Dict[str, datetime] = {}
        
        # Background tasks
        self.health_check_task: Optional[asyncio.Task] = None
        self.performance_monitor_task: Optional[asyncio.Task] = None
        self.recovery_task: Optional[asyncio.Task] = None
        self.cleanup_task: Optional[asyncio.Task] = None
        
        # State management
        self.running = False
        self.start_time = datetime.now()
        self.shutdown_event = asyncio.Event()
        
        # Event handlers
        self.event_handlers: Dict[str, List[Callable]] = {
            "service_started": [],
            "service_stopped": [],
            "service_failed": [],
            "service_recovered": [],
            "health_degraded": [],
            "health_restored": []
        }
        
        # Statistics
        self.stats = {
            "total_restarts": 0,
            "total_failures": 0,
            "total_recoveries": 0,
            "uptime_seconds": 0,
            "last_full_health_check": None,
            "services_started": 0,
            "services_stopped": 0
        }
        
        logger.info("MonitoringService initialized")
    
    def _load_default_config(self) -> MonitoringConfig:
        """Load default monitoring configuration."""
        app_config = get_config()
        
        return MonitoringConfig(
            enabled_services={
                ServiceType.METRICS_COLLECTOR,
                ServiceType.ALERT_MANAGER,
                ServiceType.PERFORMANCE_ANALYZER,
                ServiceType.DATA_STREAMER,
                ServiceType.GRAFANA_CLIENT,
                ServiceType.DASHBOARD_GENERATOR,
                ServiceType.TWITTER_CLIENT
            },
            health_check_interval=app_config.get("monitoring.health_check_interval", 30),
            service_timeout=app_config.get("monitoring.service_timeout", 60),
            max_restart_attempts=app_config.get("monitoring.max_restart_attempts", 3),
            restart_delay_seconds=app_config.get("monitoring.restart_delay", 5),
            circuit_breaker_threshold=app_config.get("monitoring.circuit_breaker_threshold", 5),
            circuit_breaker_timeout=app_config.get("monitoring.circuit_breaker_timeout", 300),
            auto_recovery=app_config.get("monitoring.auto_recovery", True),
            performance_monitoring=app_config.get("monitoring.performance_monitoring", True)
        )
    
    async def initialize(self) -> None:
        """Initialize monitoring service and all components."""
        try:
            logger.info("Initializing FlashMM Monitoring Service...")
            
            # Initialize services in dependency order
            await self._initialize_services()
            
            # Setup service dependencies
            await self._setup_service_dependencies()
            
            # Start background tasks
            await self._start_background_tasks()
            
            self.running = True
            self.start_time = datetime.now()
            
            logger.info("FlashMM Monitoring Service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize monitoring service: {e}")
            await self.shutdown()
            raise
    
    async def _initialize_services(self) -> None:
        """Initialize all enabled monitoring services."""
        try:
            # Initialize services in proper order
            initialization_order = [
                (ServiceType.METRICS_COLLECTOR, self._initialize_metrics_collector),
                (ServiceType.ALERT_MANAGER, self._initialize_alert_manager),
                (ServiceType.PERFORMANCE_ANALYZER, self._initialize_performance_analyzer),
                (ServiceType.DATA_STREAMER, self._initialize_data_streamer),
                (ServiceType.GRAFANA_CLIENT, self._initialize_grafana_client),
                (ServiceType.DASHBOARD_GENERATOR, self._initialize_dashboard_generator),
                (ServiceType.TWITTER_CLIENT, self._initialize_twitter_client)
            ]
            
            for service_type, init_func in initialization_order:
                if service_type in self.config.enabled_services:
                    try:
                        service_name = service_type.value
                        logger.info(f"Initializing {service_name}...")
                        
                        service = await init_func()
                        if service:
                            self.services[service_name] = service
                            self.startup_order.append(service_name)
                            
                            # Initialize health tracking
                            self.service_health[service_name] = ServiceHealth(
                                service_name=service_name,
                                service_type=service_type,
                                status=ServiceStatus.HEALTHY,
                                last_check=datetime.now(),
                                uptime_seconds=0
                            )
                            
                            # Initialize circuit breaker
                            self.circuit_breakers[service_name] = CircuitBreaker(
                                self.config.circuit_breaker_threshold,
                                self.config.circuit_breaker_timeout
                            )
                            
                            self.stats["services_started"] += 1
                            logger.info(f"{service_name} initialized successfully")
                        else:
                            logger.warning(f"Failed to initialize {service_name}")
                    
                    except Exception as e:
                        logger.error(f"Error initializing {service_type.value}: {e}")
                        if service_type.value in self.services:
                            del self.services[service_type.value]
            
            # Reverse order for shutdown
            self.shutdown_order = list(reversed(self.startup_order))
            
            logger.info(f"Initialized {len(self.services)} monitoring services")
            
        except Exception as e:
            logger.error(f"Error initializing services: {e}")
            raise
    
    async def _initialize_metrics_collector(self) -> Optional[MetricsCollector]:
        """Initialize metrics collector service."""
        try:
            metrics_collector = MetricsCollector()
            await metrics_collector.initialize()
            return metrics_collector
        except Exception as e:
            logger.error(f"Failed to initialize metrics collector: {e}")
            return None
    
    async def _initialize_alert_manager(self) -> Optional[AlertManager]:
        """Initialize alert manager service."""
        try:
            alert_manager = AlertManager()
            await alert_manager.initialize()
            return alert_manager
        except Exception as e:
            logger.error(f"Failed to initialize alert manager: {e}")
            return None
    
    async def _initialize_performance_analyzer(self) -> Optional[PerformanceAnalyzer]:
        """Initialize performance analyzer service."""
        try:
            metrics_collector = self.services.get("metrics_collector")
            performance_analyzer = PerformanceAnalyzer(metrics_collector)
            await performance_analyzer.initialize()
            return performance_analyzer
        except Exception as e:
            logger.error(f"Failed to initialize performance analyzer: {e}")
            return None
    
    async def _initialize_data_streamer(self) -> Optional[DataStreamer]:
        """Initialize data streamer service."""
        try:
            metrics_collector = self.services.get("metrics_collector")
            alert_manager = self.services.get("alert_manager")
            data_streamer = DataStreamer(metrics_collector, alert_manager)
            await data_streamer.initialize()
            return data_streamer
        except Exception as e:
            logger.error(f"Failed to initialize data streamer: {e}")
            return None
    
    async def _initialize_grafana_client(self) -> Optional[GrafanaClient]:
        """Initialize Grafana client service."""
        try:
            grafana_client = GrafanaClient()
            await grafana_client.initialize()
            return grafana_client
        except Exception as e:
            logger.error(f"Failed to initialize Grafana client: {e}")
            return None
    
    async def _initialize_dashboard_generator(self) -> Optional[DashboardGenerator]:
        """Initialize dashboard generator service."""
        try:
            grafana_client = self.services.get("grafana_client")
            dashboard_generator = DashboardGenerator(grafana_client)
            await dashboard_generator.initialize()
            return dashboard_generator
        except Exception as e:
            logger.error(f"Failed to initialize dashboard generator: {e}")
            return None
    
    async def _initialize_twitter_client(self) -> Optional[TwitterClient]:
        """Initialize Twitter client service."""
        try:
            twitter_client = TwitterClient()
            await twitter_client.initialize()
            return twitter_client
        except Exception as e:
            logger.error(f"Failed to initialize Twitter client: {e}")
            return None
    
    async def _setup_service_dependencies(self) -> None:
        """Setup service dependencies."""
        # Define service dependencies
        dependencies = {
            "performance_analyzer": [
                ServiceDependency("metrics_collector", required=True)
            ],
            "data_streamer": [
                ServiceDependency("metrics_collector", required=False),
                ServiceDependency("alert_manager", required=False)
            ],
            "dashboard_generator": [
                ServiceDependency("grafana_client", required=True)
            ]
        }
        
        self.service_dependencies = dependencies
        logger.info("Service dependencies configured")
    
    async def _start_background_tasks(self) -> None:
        """Start background monitoring tasks."""
        try:
            self.health_check_task = asyncio.create_task(self._health_check_loop())
            
            if self.config.performance_monitoring:
                self.performance_monitor_task = asyncio.create_task(self._performance_monitor_loop())
            
            if self.config.auto_recovery:
                self.recovery_task = asyncio.create_task(self._recovery_loop())
            
            self.cleanup_task = asyncio.create_task(self._cleanup_loop())
            
            logger.info("Background monitoring tasks started")
            
        except Exception as e:
            logger.error(f"Failed to start background tasks: {e}")
            raise
    
    async def start_services(self) -> None:
        """Start all monitoring services."""
        try:
            logger.info("Starting monitoring services...")
            
            for service_name in self.startup_order:
                try:
                    await self._start_service(service_name)
                except Exception as e:
                    logger.error(f"Failed to start {service_name}: {e}")
                    if not self.config.auto_recovery:
                        raise
            
            # Start data streamer server if enabled
            if "data_streamer" in self.services:
                data_streamer = self.services["data_streamer"]
                await data_streamer.start_server()
                logger.info("Data streaming server started")
            
            logger.info("All monitoring services started")
            
        except Exception as e:
            logger.error(f"Error starting services: {e}")
            raise
    
    async def _start_service(self, service_name: str) -> bool:
        """Start a specific service."""
        try:
            service = self.services.get(service_name)
            if not service:
                logger.error(f"Service {service_name} not found")
                return False
            
            # Check dependencies
            if not await self._check_service_dependencies(service_name):
                logger.error(f"Dependencies not met for {service_name}")
                return False
            
            # Update status
            if service_name in self.service_health:
                self.service_health[service_name].status = ServiceStatus.STARTING
            
            # Start service (if it has a start method)
            if hasattr(service, 'start'):
                await service.start()
            
            # Update status to healthy
            if service_name in self.service_health:
                self.service_health[service_name].status = ServiceStatus.HEALTHY
                self.service_health[service_name].last_check = datetime.now()
            
            # Trigger event handlers
            await self._trigger_event("service_started", service_name)
            
            logger.info(f"Service {service_name} started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error starting service {service_name}: {e}")
            
            # Update status to failed
            if service_name in self.service_health:
                self.service_health[service_name].status = ServiceStatus.FAILED
                self.service_health[service_name].last_error = str(e)
                self.service_health[service_name].error_count += 1
            
            await self._trigger_event("service_failed", service_name, {"error": str(e)})
            return False
    
    async def _check_service_dependencies(self, service_name: str) -> bool:
        """Check if service dependencies are met."""
        dependencies = self.service_dependencies.get(service_name, [])
        
        for dependency in dependencies:
            dep_service = dependency.service_name
            
            if dependency.required:
                if dep_service not in self.services:
                    logger.error(f"Required dependency {dep_service} not available for {service_name}")
                    return False
                
                dep_health = self.service_health.get(dep_service)
                if not dep_health or dep_health.status not in [ServiceStatus.HEALTHY, ServiceStatus.DEGRADED]:
                    logger.error(f"Required dependency {dep_service} not healthy for {service_name}")
                    return False
        
        return True
    
    async def shutdown(self) -> None:
        """Shutdown monitoring service and all components."""
        try:
            logger.info("Shutting down FlashMM Monitoring Service...")
            
            self.running = False
            self.shutdown_event.set()
            
            # Stop background tasks
            tasks = [
                self.health_check_task,
                self.performance_monitor_task,
                self.recovery_task,
                self.cleanup_task
            ]
            
            for task in tasks:
                if task and not task.done():
                    task.cancel()
            
            await asyncio.gather(*[task for task in tasks if task], return_exceptions=True)
            
            # Shutdown services in reverse order
            for service_name in self.shutdown_order:
                try:
                    await self._stop_service(service_name)
                except Exception as e:
                    logger.error(f"Error stopping {service_name}: {e}")
            
            logger.info("FlashMM Monitoring Service shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
    
    async def _stop_service(self, service_name: str) -> bool:
        """Stop a specific service."""
        try:
            service = self.services.get(service_name)
            if not service:
                return True
            
            # Update status
            if service_name in self.service_health:
                self.service_health[service_name].status = ServiceStatus.STOPPING
            
            # Stop service
            if hasattr(service, 'shutdown'):
                await service.shutdown()
            elif hasattr(service, 'stop'):
                await service.stop()
            
            # Update status
            if service_name in self.service_health:
                self.service_health[service_name].status = ServiceStatus.STOPPED
            
            self.stats["services_stopped"] += 1
            await self._trigger_event("service_stopped", service_name)
            
            logger.info(f"Service {service_name} stopped successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping service {service_name}: {e}")
            return False
    
    # Health checking and monitoring
    
    async def _health_check_loop(self) -> None:
        """Main health check loop."""
        while self.running:
            try:
                await self._perform_health_checks()
                self.stats["last_full_health_check"] = datetime.now()
                await asyncio.sleep(self.config.health_check_interval)
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(self.config.health_check_interval)
    
    async def _perform_health_checks(self) -> None:
        """Perform health checks on all services."""
        try:
            for service_name, service in self.services.items():
                try:
                    health = await self._check_service_health(service_name, service)
                    self.service_health[service_name] = health
                    
                    # Handle status changes
                    if health.status in [ServiceStatus.FAILED, ServiceStatus.UNHEALTHY]:
                        await self._handle_unhealthy_service(service_name, health)
                    elif health.status == ServiceStatus.DEGRADED:
                        await self._handle_degraded_service(service_name, health)
                
                except Exception as e:
                    logger.error(f"Error checking health for {service_name}: {e}")
                    
                    # Mark as unhealthy on check failure
                    if service_name in self.service_health:
                        self.service_health[service_name].status = ServiceStatus.UNHEALTHY
                        self.service_health[service_name].last_error = str(e)
                        self.service_health[service_name].error_count += 1
        
        except Exception as e:
            logger.error(f"Error performing health checks: {e}")
    
    async def _check_service_health(self, service_name: str, service: Any) -> ServiceHealth:
        """Check health of a specific service."""
        try:
            start_time = time.time()
            current_health = self.service_health.get(service_name)
            
            if not current_health:
                # Create new health record
                service_type = ServiceType(service_name)
                current_health = ServiceHealth(
                    service_name=service_name,
                    service_type=service_type,
                    status=ServiceStatus.UNKNOWN,
                    last_check=datetime.now(),
                    uptime_seconds=0
                )
            
            # Perform health check
            health_status = ServiceStatus.HEALTHY
            performance_metrics = {}
            error_info = None
            
            try:
                # Use circuit breaker for health checks
                def health_check():
                    if hasattr(service, 'get_health'):
                        return service.get_health()
                    elif hasattr(service, 'get_stats'):
                        return service.get_stats()
                    else:
                        # Basic availability check
                        return {"status": "healthy", "service": service_name}
                
                circuit_breaker = self.circuit_breakers.get(service_name)
                if circuit_breaker:
                    result = circuit_breaker.call(health_check)
                else:
                    result = health_check()
                
                # Process health check result
                if isinstance(result, dict):
                    if "error" in result or result.get("status") == "error":
                        health_status = ServiceStatus.UNHEALTHY
                        error_info = result.get("error", "Unknown error")
                    elif result.get("status") == "degraded":
                        health_status = ServiceStatus.DEGRADED
                    elif result.get("warnings"):
                        health_status = ServiceStatus.DEGRADED
                    
                    performance_metrics = {
                        k: v for k, v in result.items()
                        if k not in ["status", "error", "warnings"]
                    }
            
            except Exception as e:
                health_status = ServiceStatus.FAILED
                error_info = str(e)
                logger.error(f"Health check failed for {service_name}: {e}")
            
            # Calculate uptime
            uptime = (datetime.now() - (current_health.last_check or datetime.now())).total_seconds()
            if current_health.status in [ServiceStatus.HEALTHY, ServiceStatus.DEGRADED]:
                uptime += current_health.uptime_seconds
            
            # Calculate health score
            health_score = self._calculate_health_score(health_status, current_health.error_count)
            
            # Update health record
            updated_health = ServiceHealth(
                service_name=service_name,
                service_type=current_health.service_type,
                status=health_status,
                last_check=datetime.now(),
                uptime_seconds=uptime,
                error_count=current_health.error_count + (1 if error_info else 0),
                warning_count=current_health.warning_count,
                last_error=error_info or current_health.last_error,
                performance_metrics=performance_metrics,
                dependencies=self._get_service_dependencies(service_name),
                health_score=health_score
            )
            
            return updated_health
        
        except Exception as e:
            logger.error(f"Error in health check for {service_name}: {e}")
            
            # Return failed health status
            return ServiceHealth(
                service_name=service_name,
                service_type=ServiceType(service_name),
                status=ServiceStatus.FAILED,
                last_check=datetime.now(),
                uptime_seconds=0,
                error_count=current_health.error_count + 1 if current_health else 1,
                last_error=str(e),
                health_score=0.0
            )
    
    def _calculate_health_score(self, status: ServiceStatus, error_count: int) -> float:
        """Calculate health score for a service."""
        base_score = {
            ServiceStatus.HEALTHY: 1.0,
            ServiceStatus.DEGRADED: 0.7,
            ServiceStatus.UNHEALTHY: 0.3,
            ServiceStatus.FAILED: 0.0,
            ServiceStatus.UNKNOWN: 0.5
        }.get(status, 0.5)
        
        # Reduce score based on error count
        error_penalty = min(0.5, error_count * 0.1)
        return max(0.0, base_score - error_penalty)
    
    def _get_service_dependencies(self, service_name: str) -> List[str]:
        """Get list of service dependencies."""
        dependencies = self.service_dependencies.get(service_name, [])
        return [dep.service_name for dep in dependencies]
    
    async def _handle_unhealthy_service(self, service_name: str, health: ServiceHealth) -> None:
        """Handle unhealthy service."""
        logger.warning(f"Service {service_name} is unhealthy: {health.last_error}")
        
        self.stats["total_failures"] += 1
        await self._trigger_event("service_failed", service_name, {"health": health})
        
        # Attempt restart if auto-recovery is enabled
        if self.config.auto_recovery:
            await self._attempt_service_restart(service_name)
    
    async def _handle_degraded_service(self, service_name: str, health: ServiceHealth) -> None:
        """Handle degraded service."""
        logger.warning(f"Service {service_name} is degraded")
        await self._trigger_event("health_degraded", service_name, {"health": health})
    
    async def _attempt_service_restart(self, service_name: str) -> bool:
        """Attempt to restart a failed service."""
        try:
            # Check restart limits
            restart_count = self.restart_counts.get(service_name, 0)
            if restart_count >= self.config.max_restart_attempts:
                logger.error(f"Max restart attempts exceeded for {service_name}")
                return False
            
            # Check restart delay
            last_restart = self.last_restart_time.get(service_name)
            if last_restart:
                time_since_restart = (datetime.now() - last_restart).total_seconds()
                if time_since_restart < self.config.restart_delay_seconds:
                    logger.info(f"Waiting before restart attempt for {service_name}")
                    await asyncio.sleep(self.config.restart_delay_seconds - time_since_restart)
            
            logger.info(f"Attempting to restart {service_name} (attempt {restart_count + 1})")
            
            # Stop service
            await self._stop_service(service_name)
            
            # Wait a moment
            await asyncio.sleep(2)
            
            # Start service
            success = await self._start_service(service_name)
            
            if success:
                self.restart_counts[service_name] = 0  # Reset on successful restart
                self.stats["total_recoveries"] += 1
                await self._trigger_event("service_recovered", service_name)
                logger.info(f"Successfully restarted {service_name}")
            else:
                self.restart_counts[service_name] = restart_count + 1
                self.last_restart_time[service_name] = datetime.now()
                self.stats["total_restarts"] += 1
            
            return success
        
        except Exception as e:
            logger.error(f"Error restarting {service_name}: {e}")
            self.restart_counts[service_name] = restart_count + 1
            self.last_restart_time[service_name] = datetime.now()
            return False
    
    async def _recovery_loop(self) -> None:
        """Recovery loop for failed services."""
        while self.running:
            try:
                # Check for services that need recovery
                for service_name, health in self.service_health.items():
                    if health.status == ServiceStatus.FAILED and self.config.auto_recovery:
                        restart_count = self.restart_counts.get(service_name, 0)
                        if restart_count < self.config.max_restart_attempts:
                            await self._attempt_service_restart(service_name)
                
                await asyncio.sleep(60)  # Check every minute
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in recovery loop: {e}")
                await asyncio.sleep(60)
    
    async def _performance_monitor_loop(self) -> None:
        """Performance monitoring loop."""
        while self.running:
            try:
                # Update performance statistics
                self.stats["uptime_seconds"] = (datetime.now() - self.start_time).total_seconds()
                
                # Collect performance metrics from services
                for service_name, service in self.services.items():
                    try:
                        if hasattr(service, 'get_performance_metrics'):
                            metrics = service.get_performance_metrics()
                            if service_name in self.service_health:
                                self.service_health[service_name].performance_metrics.update(metrics)
                    except Exception as e:
                        logger.debug(f"Error collecting performance metrics from {service_name}: {e}")
                
                await asyncio.sleep(30)  # Update every 30 seconds
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in performance monitor loop: {e}")
                await asyncio.sleep(30)
    
    async def _cleanup_loop(self) -> None:
        """Cleanup loop for maintenance tasks."""
        while self.running:
            try:
                # Reset restart counts after successful uptime
                current_time = datetime.now()
                for service_name in list(self.restart_counts.keys()):
                    last_restart = self.last_restart_time.get(service_name)
                    if last_restart and (current_time - last_restart).total_seconds() > 3600:  # 1 hour
                        if service_name in self.service_health:
                            health = self.service_health[service_name]
                            if health.status == ServiceStatus.HEALTHY:
                                self.restart_counts[service_name] = 0
                                logger.debug(f"Reset restart count for {service_name}")
                
                # Cleanup old performance metrics
                for service_name, health in self.service_health.items():
                    if len(health.performance_metrics) > 100:  # Keep last 100 entries
                        # This would be more sophisticated in a real implementation
                        pass
                
                await asyncio.sleep(1800)  # Run every 30 minutes
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(1800)
    
    # Event handling
    
    async def _trigger_event(self, event_type: str, service_name: str, data: Optional[Dict[str, Any]] = None) -> None:
        """Trigger event handlers."""
        try:
            handlers = self.event_handlers.get(event_type, [])
            for handler in handlers:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(service_name, data or {})
                    else:
                        handler(service_name, data or {})
                except Exception as e:
                    logger.error(f"Error in event handler for {event_type}: {e}")
        except Exception as e:
            logger.error(f"Error triggering event {event_type}: {e}")
    
    def add_event_handler(self, event_type: str, handler: Callable) -> None:
        """Add an event handler."""
        if event_type in self.event_handlers:
            self.event_handlers[event_type].append(handler)
        else:
            logger.warning(f"Unknown event type: {event_type}")
    
    def remove_event_handler(self, event_type: str, handler: Callable) -> None:
        """Remove an event handler."""
        if event_type in self.event_handlers:
            try:
                self.event_handlers[event_type].remove(handler)
            except ValueError:
                logger.warning(f"Handler not found for event type: {event_type}")
    
    # Public API methods
    
    def get_service_health(self, service_name: Optional[str] = None) -> Union[ServiceHealth, Dict[str, ServiceHealth]]:
        """Get health information for service(s)."""
        if service_name:
            return self.service_health.get(service_name)
        return self.service_health.copy()
    
    def get_overall_health(self) -> Dict[str, Any]:
        """Get overall system health."""
        healthy_count = sum(1 for h in self.service_health.values() if h.status == ServiceStatus.HEALTHY)
        degraded_count = sum(1 for h in self.service_health.values() if h.status == ServiceStatus.DEGRADED)
        unhealthy_count = sum(1 for h in self.service_health.values() if h.status in [ServiceStatus.UNHEALTHY, ServiceStatus.FAILED])
        
        total_services = len(self.service_health)
        overall_score = sum(h.health_score for h in self.service_health.values()) / total_services if total_services > 0 else 0
        
        if unhealthy_count > 0:
            overall_status = "unhealthy"
        elif degraded_count > 0:
            overall_status = "degraded"
        elif healthy_count == total_services:
            overall_status = "healthy"
        else:
            overall_status = "unknown"
        
        return {
            "overall_status": overall_status,
            "overall_score": overall_score,
            "healthy_services": healthy_count,
            "degraded_services": degraded_count,
            "unhealthy_services": unhealthy_count,
            "total_services": total_services,
            "uptime_seconds": (datetime.now() - self.start_time).total_seconds(),
            "last_health_check": self.stats.get("last_full_health_check"),
            "auto_recovery_enabled": self.config.auto_recovery
        }
    
    def get_service_statistics(self) -> Dict[str, Any]:
        """Get service statistics."""
        return {
            **self.stats,
            "services": {
                name: {
                    "type": health.service_type.value,
                    "status": health.status.value,
                    "uptime_seconds": health.uptime_seconds,
                    "error_count": health.error_count,
                    "health_score": health.health_score,
                    "restart_count": self.restart_counts.get(name, 0)
                }
                for name, health in self.service_health.items()
            },
            "configuration": {
                "enabled_services": [s.value for s in self.config.enabled_services],
                "health_check_interval": self.config.health_check_interval,
                "auto_recovery": self.config.auto_recovery,
                "max_restart_attempts": self.config.max_restart_attempts
            }
        }
    
    async def restart_service(self, service_name: str) -> bool:
        """Manually restart a service."""
        if service_name not in self.services:
            logger.error(f"Service {service_name} not found")
            return False
        
        logger.info(f"Manual restart requested for {service_name}")
        return await self._attempt_service_restart(service_name)
    
    async def stop_service(self, service_name: str) -> bool:
        """Stop a specific service."""
        if service_name not in self.services:
            logger.error(f"Service {service_name} not found")
            return False
        
        logger.info(f"Stopping service {service_name}")
        return await self._stop_service(service_name)
    
    async def start_service(self, service_name: str) -> bool:
        """Start a specific service."""
        if service_name not in self.services:
            logger.error(f"Service {service_name} not found")
            return False
        
        logger.info(f"Starting service {service_name}")
        return await self._start_service(service_name)
    
    def get_service(self, service_name: str) -> Optional[Any]:
        """Get a service instance."""
        return self.services.get(service_name)
    
    def list_services(self) -> List[str]:
        """List all registered services."""
        return list(self.services.keys())
    
    def is_service_healthy(self, service_name: str) -> bool:
        """Check if a service is healthy."""
        health = self.service_health.get(service_name)
        return health and health.status in [ServiceStatus.HEALTHY, ServiceStatus.DEGRADED]
    
    def get_service_dependencies(self, service_name: str) -> List[str]:
        """Get service dependencies."""
        return self._get_service_dependencies(service_name)
    
    async def run_health_check(self, service_name: Optional[str] = None) -> Union[ServiceHealth, Dict[str, ServiceHealth]]:
        """Run immediate health check."""
        if service_name:
            if service_name not in self.services:
                raise ValueError(f"Service {service_name} not found")
            
            service = self.services[service_name]
            health = await self._check_service_health(service_name, service)
            self.service_health[service_name] = health
            return health
        else:
            await self._perform_health_checks()
            return self.service_health.copy()
    
    # Context manager support
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        await self.start_services()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.shutdown()
    
    # Configuration management
    
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """Update monitoring configuration."""
        try:
            # Update health check interval
            if "health_check_interval" in new_config:
                self.config.health_check_interval = new_config["health_check_interval"]
            
            # Update auto recovery
            if "auto_recovery" in new_config:
                self.config.auto_recovery = new_config["auto_recovery"]
            
            # Update restart attempts
            if "max_restart_attempts" in new_config:
                self.config.max_restart_attempts = new_config["max_restart_attempts"]
            
            logger.info("Monitoring configuration updated")
            
        except Exception as e:
            logger.error(f"Error updating configuration: {e}")
            raise
    
    def get_config(self) -> Dict[str, Any]:
        """Get current monitoring configuration."""
        return {
            "enabled_services": [s.value for s in self.config.enabled_services],
            "health_check_interval": self.config.health_check_interval,
            "service_timeout": self.config.service_timeout,
            "max_restart_attempts": self.config.max_restart_attempts,
            "restart_delay_seconds": self.config.restart_delay_seconds,
            "circuit_breaker_threshold": self.config.circuit_breaker_threshold,
            "circuit_breaker_timeout": self.config.circuit_breaker_timeout,
            "auto_recovery": self.config.auto_recovery,
            "performance_monitoring": self.config.performance_monitoring
        }
    
    # Utility methods
    
    def __repr__(self) -> str:
        """String representation."""
        healthy_count = sum(1 for h in self.service_health.values() if h.status == ServiceStatus.HEALTHY)
        total_count = len(self.service_health)
        
        return (f"MonitoringService(services={total_count}, healthy={healthy_count}, "
                f"running={self.running}, uptime={self.stats['uptime_seconds']:.0f}s)")


# Factory functions and utilities

def create_monitoring_service(enabled_services: Optional[List[str]] = None,
                            auto_recovery: bool = True,
                            health_check_interval: int = 30) -> MonitoringService:
    """Create a monitoring service with specified configuration."""
    
    config = MonitoringConfig(
        enabled_services=set(ServiceType(s) for s in enabled_services or [
            "metrics_collector", "alert_manager", "performance_analyzer",
            "data_streamer", "grafana_client", "dashboard_generator", "twitter_client"
        ]),
        auto_recovery=auto_recovery,
        health_check_interval=health_check_interval
    )
    
    return MonitoringService(config)


@asynccontextmanager
async def monitoring_service_context(config: Optional[MonitoringConfig] = None):
    """Async context manager for monitoring service."""
    service = MonitoringService(config)
    try:
        await service.initialize()
        await service.start_services()
        yield service
    finally:
        await service.shutdown()


# Health check utilities

async def check_service_connectivity(service_name: str, host: str, port: int, timeout: int = 5) -> bool:
    """Check if a service is connectable."""
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port),
            timeout=timeout
        )
        writer.close()
        await writer.wait_closed()
        return True
    except Exception:
        return False


async def check_http_endpoint(url: str, timeout: int = 5, expected_status: int = 200) -> Tuple[bool, Optional[str]]:
    """Check if an HTTP endpoint is healthy."""
    try:
        import aiohttp
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
            async with session.get(url) as response:
                if response.status == expected_status:
                    return True, None
                else:
                    return False, f"HTTP {response.status}"
    except Exception as e:
        return False, str(e)


def calculate_service_availability(health_history: List[ServiceHealth], window_hours: int = 24) -> float:
    """Calculate service availability percentage."""
    if not health_history:
        return 0.0
    
    cutoff_time = datetime.now() - timedelta(hours=window_hours)
    recent_health = [h for h in health_history if h.last_check >= cutoff_time]
    
    if not recent_health:
        return 0.0
    
    healthy_count = sum(1 for h in recent_health if h.status in [ServiceStatus.HEALTHY, ServiceStatus.DEGRADED])
    return (healthy_count / len(recent_health)) * 100.0


# Global monitoring service instance
_global_monitoring_service: Optional[MonitoringService] = None


def get_monitoring_service() -> Optional[MonitoringService]:
    """Get the global monitoring service instance."""
    return _global_monitoring_service


def set_monitoring_service(service: MonitoringService) -> None:
    """Set the global monitoring service instance."""
    global _global_monitoring_service
    _global_monitoring_service = service


# CLI support functions

def print_health_status(monitoring_service: MonitoringService) -> None:
    """Print formatted health status."""
    overall_health = monitoring_service.get_overall_health()
    
    print(f"\nFlashMM Monitoring Service Health Status")
    print(f"=" * 50)
    print(f"Overall Status: {overall_health['overall_status'].upper()}")
    print(f"Overall Score: {overall_health['overall_score']:.2f}")
    print(f"Uptime: {overall_health['uptime_seconds']:.0f} seconds")
    print(f"\nService Status:")
    print(f"-" * 30)
    
    for service_name, health in monitoring_service.get_service_health().items():
        status_symbol = {
            ServiceStatus.HEALTHY: "✓",
            ServiceStatus.DEGRADED: "⚠",
            ServiceStatus.UNHEALTHY: "✗",
            ServiceStatus.FAILED: "✗",
            ServiceStatus.UNKNOWN: "?",
            ServiceStatus.STARTING: "⟳",
            ServiceStatus.STOPPING: "⟳",
            ServiceStatus.STOPPED: "○"
        }.get(health.status, "?")
        
        print(f"{status_symbol} {service_name:<20} {health.status.value:<10} (Score: {health.health_score:.2f})")
        
        if health.last_error:
            print(f"  └── Error: {health.last_error}")


async def run_monitoring_service_cli():
    """Run monitoring service from command line."""
    import argparse
    
    parser = argparse.ArgumentParser(description="FlashMM Monitoring Service")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--services", nargs="+", help="Services to enable")
    parser.add_argument("--health-check", action="store_true", help="Run health check and exit")
    parser.add_argument("--status", action="store_true", help="Show status and exit")
    
    args = parser.parse_args()
    
    # Create monitoring service
    monitoring_service = create_monitoring_service(
        enabled_services=args.services,
        auto_recovery=True,
        health_check_interval=30
    )
    
    try:
        if args.health_check or args.status:
            # Initialize and run health check
            await monitoring_service.initialize()
            await monitoring_service.start_services()
            await monitoring_service.run_health_check()
            print_health_status(monitoring_service)
            await monitoring_service.shutdown()
        else:
            # Run service continuously
            async with monitoring_service_context() as service:
                print("FlashMM Monitoring Service started. Press Ctrl+C to stop.")
                try:
                    await service.shutdown_event.wait()
                except KeyboardInterrupt:
                    print("\nShutdown requested...")
    
    except Exception as e:
        logger.error(f"Error running monitoring service: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(run_monitoring_service_cli())
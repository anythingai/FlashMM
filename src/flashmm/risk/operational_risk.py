"""
FlashMM Operational Risk Manager

Operational risk monitoring and management system with:
- System latency and connectivity monitoring
- API failure detection and degraded performance alerts
- Automatic failsafe activation
- Order rejection and fill quality monitoring
- System resource monitoring (CPU, memory, network)
- Operational risk alerts and automated responses
"""

import asyncio
import psutil
import time
from typing import Dict, Any, Optional, List, Callable, Tuple
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from dataclasses import dataclass, field
import numpy as np
from collections import deque
import aiohttp
import socket

from flashmm.config.settings import get_config
from flashmm.utils.logging import get_logger, TradingEventLogger
from flashmm.utils.exceptions import RiskError, OperationalError
from flashmm.utils.decorators import measure_latency, timeout_async

logger = get_logger(__name__)
trading_logger = TradingEventLogger()


class SystemHealthLevel(Enum):
    """System health levels."""
    EXCELLENT = "excellent"
    GOOD = "good"
    WARNING = "warning"
    CRITICAL = "critical"
    FAILURE = "failure"


class MonitorType(Enum):
    """Types of operational monitoring."""
    LATENCY = "latency"
    CONNECTIVITY = "connectivity"
    API_HEALTH = "api_health"
    SYSTEM_RESOURCES = "system_resources"
    ORDER_QUALITY = "order_quality"
    DATA_QUALITY = "data_quality"


@dataclass
class SystemMetrics:
    """System performance metrics."""
    timestamp: datetime
    
    # System resources
    cpu_percent: float
    memory_percent: float
    disk_usage_percent: float
    network_bytes_sent: int
    network_bytes_recv: int
    
    # Application metrics
    active_connections: int
    pending_requests: int
    error_rate: float
    
    # Trading metrics
    orders_per_second: float
    fill_rate: float
    rejection_rate: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'cpu_percent': self.cpu_percent,
            'memory_percent': self.memory_percent,
            'disk_usage_percent': self.disk_usage_percent,
            'network_bytes_sent': self.network_bytes_sent,
            'network_bytes_recv': self.network_bytes_recv,
            'active_connections': self.active_connections,
            'pending_requests': self.pending_requests,
            'error_rate': self.error_rate,
            'orders_per_second': self.orders_per_second,
            'fill_rate': self.fill_rate,
            'rejection_rate': self.rejection_rate
        }


@dataclass
class ApiHealthMetrics:
    """API health monitoring metrics."""
    endpoint: str
    response_time_ms: float
    status_code: int
    success: bool
    error_message: Optional[str]
    timestamp: datetime
    
    # Aggregated metrics
    avg_response_time_ms: float = 0.0
    success_rate: float = 0.0
    error_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'endpoint': self.endpoint,
            'response_time_ms': self.response_time_ms,
            'status_code': self.status_code,
            'success': self.success,
            'error_message': self.error_message,
            'timestamp': self.timestamp.isoformat(),
            'avg_response_time_ms': self.avg_response_time_ms,
            'success_rate': self.success_rate,
            'error_count': self.error_count
        }


@dataclass
class OperationalAlert:
    """Operational risk alert."""
    alert_id: str
    monitor_type: MonitorType
    severity: str  # 'low', 'medium', 'high', 'critical'
    title: str
    message: str
    timestamp: datetime
    resolved: bool = False
    resolved_time: Optional[datetime] = None
    auto_action_taken: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'alert_id': self.alert_id,
            'monitor_type': self.monitor_type.value,
            'severity': self.severity,
            'title': self.title,
            'message': self.message,
            'timestamp': self.timestamp.isoformat(),
            'resolved': self.resolved,
            'resolved_time': self.resolved_time.isoformat() if self.resolved_time else None,
            'auto_action_taken': self.auto_action_taken
        }


class SystemMonitor:
    """System resource monitoring."""
    
    def __init__(self):
        self.config = get_config()
        self.metrics_history: deque = deque(maxlen=1000)  # Keep last 1000 metrics
        
        # Thresholds
        self.cpu_warning_threshold = self.config.get("monitoring.cpu_warning_threshold", 70.0)
        self.cpu_critical_threshold = self.config.get("monitoring.cpu_critical_threshold", 90.0)
        self.memory_warning_threshold = self.config.get("monitoring.memory_warning_threshold", 80.0)
        self.memory_critical_threshold = self.config.get("monitoring.memory_critical_threshold", 95.0)
        self.disk_warning_threshold = self.config.get("monitoring.disk_warning_threshold", 85.0)
        
        # Network baseline
        self.network_baseline_sent = 0
        self.network_baseline_recv = 0
        self.last_network_check = time.time()
        
    async def collect_system_metrics(self) -> SystemMetrics:
        """Collect comprehensive system metrics."""
        try:
            current_time = datetime.now()
            
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_usage_percent = (disk.used / disk.total) * 100
            
            # Network metrics
            network = psutil.net_io_counters()
            current_time_seconds = time.time()
            time_delta = current_time_seconds - self.last_network_check
            
            if time_delta > 0:
                bytes_sent_rate = (network.bytes_sent - self.network_baseline_sent) / time_delta
                bytes_recv_rate = (network.bytes_recv - self.network_baseline_recv) / time_delta
            else:
                bytes_sent_rate = 0
                bytes_recv_rate = 0
            
            self.network_baseline_sent = network.bytes_sent
            self.network_baseline_recv = network.bytes_recv
            self.last_network_check = current_time_seconds
            
            # Process-specific metrics
            process = psutil.Process()
            connections = len(process.connections())
            
            # Trading metrics (would be populated by trading engine)
            metrics = SystemMetrics(
                timestamp=current_time,
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                disk_usage_percent=disk_usage_percent,
                network_bytes_sent=int(bytes_sent_rate),
                network_bytes_recv=int(bytes_recv_rate),
                active_connections=connections,
                pending_requests=0,  # Would be tracked by application
                error_rate=0.0,      # Would be tracked by application
                orders_per_second=0.0,  # Would be tracked by trading engine
                fill_rate=0.0,          # Would be tracked by trading engine
                rejection_rate=0.0      # Would be tracked by trading engine
            )
            
            self.metrics_history.append(metrics)
            return metrics
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            raise OperationalError(f"System metrics collection failed: {e}")
    
    def get_system_health_level(self, metrics: SystemMetrics) -> SystemHealthLevel:
        """Determine system health level based on metrics."""
        try:
            # Check critical thresholds
            if (metrics.cpu_percent >= self.cpu_critical_threshold or
                metrics.memory_percent >= self.memory_critical_threshold):
                return SystemHealthLevel.CRITICAL
            
            # Check warning thresholds
            warning_conditions = [
                metrics.cpu_percent >= self.cpu_warning_threshold,
                metrics.memory_percent >= self.memory_warning_threshold,
                metrics.disk_usage_percent >= self.disk_warning_threshold,
                metrics.error_rate > 0.05,  # 5% error rate
                metrics.rejection_rate > 0.10  # 10% rejection rate
            ]
            
            if sum(warning_conditions) >= 2:  # Multiple warning conditions
                return SystemHealthLevel.WARNING
            elif any(warning_conditions):
                return SystemHealthLevel.GOOD
            else:
                return SystemHealthLevel.EXCELLENT
                
        except Exception as e:
            logger.error(f"Error determining system health level: {e}")
            return SystemHealthLevel.FAILURE
    
    def get_performance_trends(self, lookback_minutes: int = 30) -> Dict[str, Any]:
        """Analyze performance trends over lookback period."""
        try:
            cutoff_time = datetime.now() - timedelta(minutes=lookback_minutes)
            recent_metrics = [m for m in self.metrics_history if m.timestamp > cutoff_time]
            
            if len(recent_metrics) < 2:
                return {'insufficient_data': True}
            
            # Calculate trends
            cpu_values = [m.cpu_percent for m in recent_metrics]
            memory_values = [m.memory_percent for m in recent_metrics]
            
            cpu_trend = 'stable'
            memory_trend = 'stable'
            
            if len(cpu_values) >= 10:
                cpu_slope = np.polyfit(range(len(cpu_values)), cpu_values, 1)[0]
                if cpu_slope > 0.5:
                    cpu_trend = 'increasing'
                elif cpu_slope < -0.5:
                    cpu_trend = 'decreasing'
            
            if len(memory_values) >= 10:
                memory_slope = np.polyfit(range(len(memory_values)), memory_values, 1)[0]
                if memory_slope > 0.5:
                    memory_trend = 'increasing'
                elif memory_slope < -0.5:
                    memory_trend = 'decreasing'
            
            return {
                'lookback_minutes': lookback_minutes,
                'data_points': len(recent_metrics),
                'cpu_trend': cpu_trend,
                'memory_trend': memory_trend,
                'avg_cpu_percent': np.mean(cpu_values),
                'avg_memory_percent': np.mean(memory_values),
                'max_cpu_percent': max(cpu_values),
                'max_memory_percent': max(memory_values),
                'cpu_volatility': np.std(cpu_values),
                'memory_volatility': np.std(memory_values)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing performance trends: {e}")
            return {'error': str(e)}


class ConnectivityMonitor:
    """Network connectivity and API health monitoring."""
    
    def __init__(self):
        self.config = get_config()
        self.api_health_history: Dict[str, deque] = {}
        
        # Critical endpoints to monitor
        self.critical_endpoints = {
            'sei_rpc': self.config.get("sei_rpc_url", "https://sei-testnet-rpc.polkachu.com"),
            'cambrian_api': self.config.get("cambrian_api_url", "https://api.cambrian.finance"),
            'redis': 'redis://localhost:6379',
            'influxdb': self.config.get("influxdb_url", "http://localhost:8086")
        }
        
        # Health check thresholds
        self.response_time_warning_ms = self.config.get("monitoring.response_time_warning_ms", 1000)
        self.response_time_critical_ms = self.config.get("monitoring.response_time_critical_ms", 5000)
        self.success_rate_warning = self.config.get("monitoring.success_rate_warning", 0.95)
        self.success_rate_critical = self.config.get("monitoring.success_rate_critical", 0.90)
    
    async def check_endpoint_health(self, endpoint_name: str, url: str) -> ApiHealthMetrics:
        """Check health of a specific endpoint."""
        try:
            start_time = time.perf_counter()
            
            if url.startswith('http'):
                # HTTP endpoint check
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                    async with session.get(url) as response:
                        response_time_ms = (time.perf_counter() - start_time) * 1000
                        success = response.status < 400
                        
                        return ApiHealthMetrics(
                            endpoint=endpoint_name,
                            response_time_ms=response_time_ms,
                            status_code=response.status,
                            success=success,
                            error_message=None if success else f"HTTP {response.status}",
                            timestamp=datetime.now()
                        )
            elif url.startswith('redis://'):
                # Redis connection check
                try:
                    import redis.asyncio as redis
                    redis_client = redis.from_url(url, socket_timeout=5)
                    await redis_client.ping()
                    await redis_client.close()
                    
                    response_time_ms = (time.perf_counter() - start_time) * 1000
                    return ApiHealthMetrics(
                        endpoint=endpoint_name,
                        response_time_ms=response_time_ms,
                        status_code=200,
                        success=True,
                        error_message=None,
                        timestamp=datetime.now()
                    )
                except Exception as e:
                    response_time_ms = (time.perf_counter() - start_time) * 1000
                    return ApiHealthMetrics(
                        endpoint=endpoint_name,
                        response_time_ms=response_time_ms,
                        status_code=500,
                        success=False,
                        error_message=str(e),
                        timestamp=datetime.now()
                    )
            else:
                # Generic TCP connection check
                try:
                    host, port = url.split(':')
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(5)
                    result = sock.connect_ex((host, int(port)))
                    sock.close()
                    
                    response_time_ms = (time.perf_counter() - start_time) * 1000
                    success = result == 0
                    
                    return ApiHealthMetrics(
                        endpoint=endpoint_name,
                        response_time_ms=response_time_ms,
                        status_code=200 if success else 500,
                        success=success,
                        error_message=None if success else f"Connection failed: {result}",
                        timestamp=datetime.now()
                    )
                except Exception as e:
                    response_time_ms = (time.perf_counter() - start_time) * 1000
                    return ApiHealthMetrics(
                        endpoint=endpoint_name,
                        response_time_ms=response_time_ms,
                        status_code=500,
                        success=False,
                        error_message=str(e),
                        timestamp=datetime.now()
                    )
                    
        except Exception as e:
            response_time_ms = (time.perf_counter() - start_time) * 1000
            return ApiHealthMetrics(
                endpoint=endpoint_name,
                response_time_ms=response_time_ms,
                status_code=500,
                success=False,
                error_message=str(e),
                timestamp=datetime.now()
            )
    
    async def check_all_endpoints(self) -> Dict[str, ApiHealthMetrics]:
        """Check health of all critical endpoints."""
        health_results = {}
        
        for endpoint_name, url in self.critical_endpoints.items():
            try:
                health_metric = await self.check_endpoint_health(endpoint_name, url)
                
                # Update history
                if endpoint_name not in self.api_health_history:
                    self.api_health_history[endpoint_name] = deque(maxlen=100)
                self.api_health_history[endpoint_name].append(health_metric)
                
                # Calculate aggregated metrics
                recent_metrics = list(self.api_health_history[endpoint_name])
                if recent_metrics:
                    health_metric.avg_response_time_ms = np.mean([m.response_time_ms for m in recent_metrics])
                    health_metric.success_rate = np.mean([m.success for m in recent_metrics])
                    health_metric.error_count = sum(1 for m in recent_metrics if not m.success)
                
                health_results[endpoint_name] = health_metric
                
            except Exception as e:
                logger.error(f"Error checking endpoint {endpoint_name}: {e}")
                health_results[endpoint_name] = ApiHealthMetrics(
                    endpoint=endpoint_name,
                    response_time_ms=0.0,
                    status_code=500,
                    success=False,
                    error_message=str(e),
                    timestamp=datetime.now()
                )
        
        return health_results
    
    def assess_connectivity_health(self, health_results: Dict[str, ApiHealthMetrics]) -> SystemHealthLevel:
        """Assess overall connectivity health."""
        try:
            critical_failures = 0
            warnings = 0
            
            for endpoint_name, metrics in health_results.items():
                if not metrics.success:
                    critical_failures += 1
                elif (metrics.response_time_ms > self.response_time_critical_ms or
                      metrics.success_rate < self.success_rate_critical):
                    critical_failures += 1
                elif (metrics.response_time_ms > self.response_time_warning_ms or
                      metrics.success_rate < self.success_rate_warning):
                    warnings += 1
            
            total_endpoints = len(health_results)
            
            if critical_failures >= total_endpoints * 0.5:  # 50% or more critical
                return SystemHealthLevel.CRITICAL
            elif critical_failures > 0:
                return SystemHealthLevel.WARNING
            elif warnings >= total_endpoints * 0.5:  # 50% or more warnings
                return SystemHealthLevel.WARNING
            elif warnings > 0:
                return SystemHealthLevel.GOOD
            else:
                return SystemHealthLevel.EXCELLENT
                
        except Exception as e:
            logger.error(f"Error assessing connectivity health: {e}")
            return SystemHealthLevel.FAILURE


class OperationalRiskManager:
    """Comprehensive operational risk management system."""
    
    def __init__(self):
        self.config = get_config()
        
        # Monitoring components
        self.system_monitor = SystemMonitor()
        self.connectivity_monitor = ConnectivityMonitor()
        
        # Alert management
        self.active_alerts: Dict[str, OperationalAlert] = {}
        self.alert_history: deque = deque(maxlen=1000)
        self.alert_counter = 0
        
        # Failsafe callbacks
        self.failsafe_callbacks: Dict[str, Callable] = {}
        
        # Monitoring intervals
        self.system_check_interval = self.config.get("monitoring.system_check_interval_seconds", 30)
        self.connectivity_check_interval = self.config.get("monitoring.connectivity_check_interval_seconds", 60)
        
        # Background tasks
        self._monitoring_tasks: List[asyncio.Task] = []
        self._shutdown_event = asyncio.Event()
        
        self._lock = asyncio.Lock()
    
    async def initialize(self) -> None:
        """Initialize operational risk manager."""
        try:
            # Start monitoring tasks
            await self._start_monitoring_tasks()
            
            logger.info("Operational risk manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize operational risk manager: {e}")
            raise RiskError(f"Operational risk manager initialization failed: {e}")
    
    async def _start_monitoring_tasks(self):
        """Start background monitoring tasks."""
        # System monitoring task
        system_task = asyncio.create_task(self._system_monitoring_loop())
        self._monitoring_tasks.append(system_task)
        
        # Connectivity monitoring task
        connectivity_task = asyncio.create_task(self._connectivity_monitoring_loop())
        self._monitoring_tasks.append(connectivity_task)
        
        logger.info("Started operational monitoring tasks")
    
    async def _system_monitoring_loop(self):
        """Background system monitoring loop."""
        while not self._shutdown_event.is_set():
            try:
                # Collect system metrics
                metrics = await self.system_monitor.collect_system_metrics()
                
                # Assess health level
                health_level = self.system_monitor.get_system_health_level(metrics)
                
                # Check for alerts
                await self._check_system_alerts(metrics, health_level)
                
                # Wait for next check
                await asyncio.sleep(self.system_check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in system monitoring loop: {e}")
                await asyncio.sleep(min(60, self.system_check_interval * 2))
    
    async def _connectivity_monitoring_loop(self):
        """Background connectivity monitoring loop."""
        while not self._shutdown_event.is_set():
            try:
                # Check all endpoints
                health_results = await self.connectivity_monitor.check_all_endpoints()
                
                # Assess connectivity health
                connectivity_health = self.connectivity_monitor.assess_connectivity_health(health_results)
                
                # Check for alerts
                await self._check_connectivity_alerts(health_results, connectivity_health)
                
                # Wait for next check
                await asyncio.sleep(self.connectivity_check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in connectivity monitoring loop: {e}")
                await asyncio.sleep(min(120, self.connectivity_check_interval * 2))
    
    async def _check_system_alerts(self, metrics: SystemMetrics, health_level: SystemHealthLevel):
        """Check for system-related alerts."""
        async with self._lock:
            alerts_to_create = []
            
            # CPU alerts
            if metrics.cpu_percent >= self.system_monitor.cpu_critical_threshold:
                alerts_to_create.append({
                    'monitor_type': MonitorType.SYSTEM_RESOURCES,
                    'severity': 'critical',
                    'title': 'Critical CPU Usage',
                    'message': f'CPU usage at {metrics.cpu_percent:.1f}% (threshold: {self.system_monitor.cpu_critical_threshold}%)',
                    'auto_action': 'reduce_system_load'
                })
            elif metrics.cpu_percent >= self.system_monitor.cpu_warning_threshold:
                alerts_to_create.append({
                    'monitor_type': MonitorType.SYSTEM_RESOURCES,
                    'severity': 'high',
                    'title': 'High CPU Usage',
                    'message': f'CPU usage at {metrics.cpu_percent:.1f}% (threshold: {self.system_monitor.cpu_warning_threshold}%)'
                })
            
            # Memory alerts
            if metrics.memory_percent >= self.system_monitor.memory_critical_threshold:
                alerts_to_create.append({
                    'monitor_type': MonitorType.SYSTEM_RESOURCES,
                    'severity': 'critical',
                    'title': 'Critical Memory Usage',
                    'message': f'Memory usage at {metrics.memory_percent:.1f}% (threshold: {self.system_monitor.memory_critical_threshold}%)',
                    'auto_action': 'restart_application'
                })
            elif metrics.memory_percent >= self.system_monitor.memory_warning_threshold:
                alerts_to_create.append({
                    'monitor_type': MonitorType.SYSTEM_RESOURCES,
                    'severity': 'high',
                    'title': 'High Memory Usage',
                    'message': f'Memory usage at {metrics.memory_percent:.1f}% (threshold: {self.system_monitor.memory_warning_threshold}%)'
                })
            
            # Create alerts
            for alert_data in alerts_to_create:
                await self._create_alert(**alert_data)
    
    async def _check_connectivity_alerts(self, health_results: Dict[str, ApiHealthMetrics], connectivity_health: SystemHealthLevel):
        """Check for connectivity-related alerts."""
        async with self._lock:
            for endpoint_name, metrics in health_results.items():
                if not metrics.success:
                    await self._create_alert(
                        monitor_type=MonitorType.CONNECTIVITY,
                        severity='critical',
                        title=f'{endpoint_name} Connection Failed',
                        message=f'Failed to connect to {endpoint_name}: {metrics.error_message}',
                        auto_action='activate_failsafe'
                    )
                elif metrics.response_time_ms > self.connectivity_monitor.response_time_critical_ms:
                    await self._create_alert(
                        monitor_type=MonitorType.LATENCY,
                        severity='high',
                        title=f'{endpoint_name} High Latency',
                        message=f'{endpoint_name} response time: {metrics.response_time_ms:.1f}ms (threshold: {self.connectivity_monitor.response_time_critical_ms}ms)'
                    )
    
    async def _create_alert(self, monitor_type: MonitorType, severity: str, title: str, message: str, auto_action: Optional[str] = None):
        """Create an operational alert."""
        try:
            self.alert_counter += 1
            alert_id = f"OP{self.alert_counter:06d}"
            
            alert = OperationalAlert(
                alert_id=alert_id,
                monitor_type=monitor_type,
                severity=severity,
                title=title,
                message=message,
                timestamp=datetime.now(),
                auto_action_taken=auto_action
            )
            
            # Check if similar alert already active
            similar_alert_key = f"{monitor_type.value}_{severity}_{title}"
            if similar_alert_key not in self.active_alerts:
                self.active_alerts[similar_alert_key] = alert
                self.alert_history.append(alert)
                
                logger.warning(f"🚨 OPERATIONAL ALERT: {title} - {message}")
                
                # Execute automatic action if specified
                if auto_action and auto_action in self.failsafe_callbacks:
                    try:
                        await self.failsafe_callbacks[auto_action](alert)
                        logger.info(f"Executed automatic action: {auto_action}")
                    except Exception as e:
                        logger.error(f"Failed to execute automatic action {auto_action}: {e}")
                
                # Log to trading event logger
                await trading_logger.log_pnl_event(
                    "SYSTEM",
                    0.0,
                    0.0,
                    0.0,
                    operational_alert=f"{severity}:{title}"
                )
        
        except Exception as e:
            logger.error(f"Error creating operational alert: {e}")
    
    def register_failsafe_callback(self, action_name: str, callback: Callable):
        """Register a failsafe callback function."""
        self.failsafe_callbacks[action_name] = callback
        logger.info(f"Registered failsafe callback: {action_name}")
    
    async def resolve_alert(self, alert_id: str, resolution_note: Optional[str] = None):
        """Resolve an active alert."""
        async with self._lock:
            for key, alert in self.active_alerts.items():
                if alert.alert_id == alert_id:
                    alert.resolved = True
                    alert.resolved_time = datetime.now()
                    del self.active_alerts[key]
                    
                    logger.info(f"Resolved operational alert: {alert_id}")
                    if resolution_note:
                        logger.info(f"Resolution note: {resolution_note}")
                    break
    
    async def get_operational_status(self) -> Dict[str, Any]:
        """Get comprehensive operational status."""
        try:
            # Get latest system metrics
            system_metrics = await self.system_monitor.collect_system_metrics()
            system_health = self.system_monitor.get_system_health_level(system_metrics)
            
            # Get connectivity status
            connectivity_results = await self.connectivity_monitor.check_all_endpoints()
            connectivity_health = self.connectivity_monitor.assess_connectivity_health(connectivity_results)
            
            # Get performance trends
            performance_trends = self.system_monitor.get_performance_trends()
            
            return {
                'timestamp': datetime.now().isoformat(),
                'overall_health': min(system_health, connectivity_health, key=lambda x: ['excellent', 'good', 'warning', 'critical', 'failure'].index(x.value)).value,
                'system_metrics': system_metrics.to_dict(),
                'system_health': system_health.value,
                'connectivity_health': connectivity_health.value,
                'endpoint_health': {name: metrics.to_dict() for name, metrics in connectivity_results.items()},
                'performance_trends': performance_trends,
                'active_alerts': [alert.to_dict() for alert in self.active_alerts.values()],
                'alert_summary': {
                    'total_active': len(self.active_alerts),
                    'critical': len([a for a in self.active_alerts.values() if a.severity == 'critical']),
                    'high': len([a for a in self.active_alerts.values() if a.severity == 'high']),
                    'medium': len([a for a in self.active_alerts.values() if a.severity == 'medium']),
                    'low': len([a for a in self.active_alerts.values() if a.severity == 'low'])
                },
                'monitoring_status': {
                    'system_monitoring_active': len([t for t in self._monitoring_tasks if not t.done()]) > 0,
                    'last_system_check': system_metrics.timestamp.isoformat(),
                    'registered_failsafes': list(self.failsafe_callbacks.keys())
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting operational status: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'overall_health': 'failure',
                'error': str(e)
            }
    
    async def get_alert_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent alert history."""
        try:
            recent_alerts = list(self.alert_history)[-limit:]
            return [alert.to_dict() for alert in recent_alerts]
        except Exception as e:
            logger.error(f"Error getting alert history: {e}")
            return []
    
    async def update_trading_metrics(self,
                                   orders_per_second: float,
                                   fill_rate: float,
                                   rejection_rate: float,
                                   error_rate: float):
        """Update trading-related metrics."""
        try:
            # Update the latest system metrics with trading data
            if self.system_monitor.metrics_history:
                latest_metrics = self.system_monitor.metrics_history[-1]
                latest_metrics.orders_per_second = orders_per_second
                latest_metrics.fill_rate = fill_rate
                latest_metrics.rejection_rate = rejection_rate
                latest_metrics.error_rate = error_rate
                
                # Check for trading quality alerts
                await self._check_trading_quality_alerts(latest_metrics)
                
        except Exception as e:
            logger.error(f"Error updating trading metrics: {e}")
    
    async def _check_trading_quality_alerts(self, metrics: SystemMetrics):
        """Check for trading quality alerts."""
        async with self._lock:
            # High rejection rate alert
            if metrics.rejection_rate > 0.20:  # 20% rejection rate
                await self._create_alert(
                    monitor_type=MonitorType.ORDER_QUALITY,
                    severity='critical',
                    title='High Order Rejection Rate',
                    message=f'Order rejection rate at {metrics.rejection_rate:.1%} (threshold: 20%)',
                    auto_action='review_order_parameters'
                )
            elif metrics.rejection_rate > 0.10:  # 10% rejection rate
                await self._create_alert(
                    monitor_type=MonitorType.ORDER_QUALITY,
                    severity='high',
                    title='Elevated Order Rejection Rate',
                    message=f'Order rejection rate at {metrics.rejection_rate:.1%} (threshold: 10%)'
                )
            
            # Low fill rate alert
            if metrics.fill_rate < 0.70 and metrics.orders_per_second > 1:  # 70% fill rate
                await self._create_alert(
                    monitor_type=MonitorType.ORDER_QUALITY,
                    severity='high',
                    title='Low Order Fill Rate',
                    message=f'Order fill rate at {metrics.fill_rate:.1%} (threshold: 70%)'
                )
    
    async def force_system_health_check(self) -> Dict[str, Any]:
        """Force an immediate comprehensive system health check."""
        try:
            logger.info("Performing forced system health check")
            
            # Collect fresh metrics
            system_metrics = await self.system_monitor.collect_system_metrics()
            connectivity_results = await self.connectivity_monitor.check_all_endpoints()
            
            # Assess health levels
            system_health = self.system_monitor.get_system_health_level(system_metrics)
            connectivity_health = self.connectivity_monitor.assess_connectivity_health(connectivity_results)
            
            # Check for immediate alerts
            await self._check_system_alerts(system_metrics, system_health)
            await self._check_connectivity_alerts(connectivity_results, connectivity_health)
            
            health_check_result = {
                'timestamp': datetime.now().isoformat(),
                'system_health': system_health.value,
                'connectivity_health': connectivity_health.value,
                'system_metrics': system_metrics.to_dict(),
                'endpoint_health': {name: metrics.to_dict() for name, metrics in connectivity_results.items()},
                'new_alerts': len([a for a in self.active_alerts.values() if
                                 (datetime.now() - a.timestamp).total_seconds() < 60]),
                'recommendations': self._generate_health_recommendations(system_health, connectivity_health)
            }
            
            logger.info(f"System health check completed: {system_health.value}, connectivity: {connectivity_health.value}")
            return health_check_result
            
        except Exception as e:
            logger.error(f"Error during forced health check: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'system_health': 'failure'
            }
    
    def _generate_health_recommendations(self, system_health: SystemHealthLevel, connectivity_health: SystemHealthLevel) -> List[str]:
        """Generate health-based recommendations."""
        recommendations = []
        
        if system_health == SystemHealthLevel.CRITICAL:
            recommendations.append("URGENT: System resources critically low - consider stopping non-essential processes")
            recommendations.append("Monitor system performance closely and prepare for potential restart")
        elif system_health == SystemHealthLevel.WARNING:
            recommendations.append("System resources elevated - monitor resource usage trends")
            recommendations.append("Consider reducing trading frequency to lower system load")
        
        if connectivity_health == SystemHealthLevel.CRITICAL:
            recommendations.append("CRITICAL: Multiple connectivity issues detected - activate failsafe mode")
            recommendations.append("Check network connectivity and API endpoint status")
        elif connectivity_health == SystemHealthLevel.WARNING:
            recommendations.append("Connectivity issues detected - monitor endpoint health closely")
        
        if system_health == SystemHealthLevel.EXCELLENT and connectivity_health == SystemHealthLevel.EXCELLENT:
            recommendations.append("System operating optimally - all monitoring systems healthy")
        
        return recommendations
    
    async def cleanup(self) -> None:
        """Cleanup operational risk manager."""
        try:
            logger.info("Starting operational risk manager cleanup...")
            
            # Signal shutdown to monitoring tasks
            self._shutdown_event.set()
            
            # Wait for monitoring tasks to complete
            if self._monitoring_tasks:
                await asyncio.gather(*self._monitoring_tasks, return_exceptions=True)
                self._monitoring_tasks.clear()
            
            # Resolve any remaining active alerts
            for alert in self.active_alerts.values():
                alert.resolved = True
                alert.resolved_time = datetime.now()
            self.active_alerts.clear()
            
            logger.info("Operational risk manager cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during operational risk manager cleanup: {e}")
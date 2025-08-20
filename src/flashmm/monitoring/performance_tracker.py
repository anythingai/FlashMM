"""
FlashMM Performance Tracker

Real-time performance monitoring and metrics collection for the trading system.
Tracks latency, throughput, and business metrics with alerting capabilities.
"""

import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from decimal import Decimal
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json
import time

from flashmm.config.settings import get_config
from flashmm.data.storage.redis_client import RedisClient
from flashmm.utils.logging import get_logger
from flashmm.utils.decorators import measure_latency, timeout_async

logger = get_logger(__name__)


@dataclass
class MetricPoint:
    """Single metric data point."""
    name: str
    value: float
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'value': self.value,
            'timestamp': self.timestamp.isoformat(),
            'tags': self.tags
        }


@dataclass
class MetricSummary:
    """Metric summary statistics."""
    name: str
    count: int
    min_value: float
    max_value: float
    avg_value: float
    sum_value: float
    last_value: float
    last_updated: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'count': self.count,
            'min_value': self.min_value,
            'max_value': self.max_value,
            'avg_value': round(self.avg_value, 4),
            'sum_value': self.sum_value,
            'last_value': self.last_value,
            'last_updated': self.last_updated.isoformat()
        }


class PerformanceTracker:
    """Real-time performance monitoring system."""
    
    def __init__(self):
        self.config = get_config()
        self.redis_client: Optional[RedisClient] = None
        
        # Metric storage
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.metric_summaries: Dict[str, MetricSummary] = {}
        
        # Performance thresholds
        self.latency_thresholds = {
            'trading_cycle': 200.0,          # 200ms target
            'ml_prediction': 50.0,           # 50ms target
            'quote_generation': 20.0,        # 20ms target
            'order_placement': 30.0,         # 30ms target
            'position_update': 10.0          # 10ms target
        }
        
        # Alert configuration
        self.alert_thresholds = {
            'trading.cycle_time_ms': 300.0,         # Alert if cycle > 300ms
            'trading.inventory_utilization': 0.9,   # Alert if > 90% inventory used
            'trading.error_rate': 0.05,             # Alert if > 5% error rate
            'system.memory_usage_pct': 85.0,        # Alert if > 85% memory
            'system.cpu_usage_pct': 80.0            # Alert if > 80% CPU
        }
        
        # Background tasks
        self._publisher_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        
        # Configuration
        self.enable_redis_publishing = self.config.get("monitoring.enable_redis_publishing", True)
        self.publishing_interval_seconds = self.config.get("monitoring.publishing_interval_seconds", 10)
        self.metric_retention_hours = self.config.get("monitoring.metric_retention_hours", 24)
        
        # Statistics
        self.total_metrics_recorded = 0
        self.alerts_triggered = 0
        self.start_time = datetime.now()
        
        logger.info("PerformanceTracker initialized")
    
    async def initialize(self) -> None:
        """Initialize the performance tracker."""
        try:
            # Initialize Redis client for publishing
            if self.enable_redis_publishing:
                self.redis_client = RedisClient()
                await self.redis_client.initialize()
            
            # Start background tasks
            await self._start_background_tasks()
            
            logger.info("PerformanceTracker initialization completed")
            
        except Exception as e:
            logger.error(f"Failed to initialize PerformanceTracker: {e}")
            raise
    
    async def _start_background_tasks(self) -> None:
        """Start background publishing and cleanup tasks."""
        if self.enable_redis_publishing:
            self._publisher_task = asyncio.create_task(self._publishing_loop())
        
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
    
    async def _publishing_loop(self) -> None:
        """Background metrics publishing loop."""
        while True:
            try:
                await asyncio.sleep(self.publishing_interval_seconds)
                await self._publish_metrics_to_redis()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Metrics publishing error: {e}")
                await asyncio.sleep(30)
    
    async def _cleanup_loop(self) -> None:
        """Background cleanup loop for old metrics."""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                self._cleanup_old_metrics()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Metrics cleanup error: {e}")
                await asyncio.sleep(300)
    
    @timeout_async(0.01)  # 10ms timeout for metric recording
    async def record_metric(
        self,
        name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None
    ) -> None:
        """Record a single metric point."""
        try:
            timestamp = datetime.now()
            tags = tags or {}
            
            # Create metric point
            metric_point = MetricPoint(
                name=name,
                value=value,
                timestamp=timestamp,
                tags=tags
            )
            
            # Store in memory
            self.metrics[name].append(metric_point)
            
            # Update summary statistics
            self._update_metric_summary(name, value, timestamp)
            
            # Check alert thresholds
            await self._check_alert_threshold(name, value)
            
            # Update statistics
            self.total_metrics_recorded += 1
            
            # Log high-frequency metrics at debug level
            if name.endswith('_time_ms') or name.endswith('_latency'):
                logger.debug(f"Metric recorded: {name} = {value}")
            else:
                logger.info(f"Metric recorded: {name} = {value}")
            
        except Exception as e:
            logger.error(f"Failed to record metric {name}: {e}")
    
    def _update_metric_summary(self, name: str, value: float, timestamp: datetime) -> None:
        """Update summary statistics for a metric."""
        if name not in self.metric_summaries:
            self.metric_summaries[name] = MetricSummary(
                name=name,
                count=1,
                min_value=value,
                max_value=value,
                avg_value=value,
                sum_value=value,
                last_value=value,
                last_updated=timestamp
            )
        else:
            summary = self.metric_summaries[name]
            summary.count += 1
            summary.min_value = min(summary.min_value, value)
            summary.max_value = max(summary.max_value, value)
            summary.sum_value += value
            summary.avg_value = summary.sum_value / summary.count
            summary.last_value = value
            summary.last_updated = timestamp
    
    async def _check_alert_threshold(self, name: str, value: float) -> None:
        """Check if metric value exceeds alert threshold."""
        threshold = self.alert_thresholds.get(name)
        if threshold is None:
            return
        
        if value > threshold:
            await self._trigger_alert(name, value, threshold)
    
    async def _trigger_alert(self, metric_name: str, value: float, threshold: float) -> None:
        """Trigger an alert for threshold violation."""
        alert = {
            'type': 'threshold_violation',
            'metric_name': metric_name,
            'value': value,
            'threshold': threshold,
            'severity': 'warning' if value < threshold * 1.5 else 'critical',
            'timestamp': datetime.now().isoformat(),
            'message': f"{metric_name} value {value} exceeds threshold {threshold}"
        }
        
        logger.warning(f"ALERT: {alert['message']}")
        self.alerts_triggered += 1
        
        # Publish alert to Redis
        if self.redis_client:
            try:
                await self.redis_client.publish(
                    "trading_alerts",
                    json.dumps(alert)
                )
            except Exception as e:
                logger.error(f"Failed to publish alert: {e}")
    
    async def record_latency(self, operation: str, latency_ms: float) -> None:
        """Record latency metric with threshold checking."""
        await self.record_metric(f"{operation}_latency_ms", latency_ms)
        
        # Check performance thresholds
        threshold = self.latency_thresholds.get(operation)
        if threshold and latency_ms > threshold:
            logger.warning(
                f"Performance threshold exceeded: {operation} took {latency_ms:.1f}ms "
                f"(threshold: {threshold:.1f}ms)"
            )
    
    async def record_trading_metrics(self, metrics: Dict[str, Any]) -> None:
        """Record a batch of trading metrics."""
        try:
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float, Decimal)):
                    await self.record_metric(f"trading.{metric_name}", float(value))
        except Exception as e:
            logger.error(f"Failed to record trading metrics: {e}")
    
    async def record_system_metrics(self) -> None:
        """Record system-level metrics."""
        try:
            import psutil
            
            # CPU and memory metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            
            await self.record_metric("system.cpu_usage_pct", cpu_percent)
            await self.record_metric("system.memory_usage_pct", memory.percent)
            await self.record_metric("system.memory_available_mb", memory.available / 1024 / 1024)
            
            # Process-specific metrics
            process = psutil.Process()
            process_memory = process.memory_info()
            
            await self.record_metric("process.memory_rss_mb", process_memory.rss / 1024 / 1024)
            await self.record_metric("process.memory_vms_mb", process_memory.vms / 1024 / 1024)
            await self.record_metric("process.cpu_percent", process.cpu_percent())
            
        except ImportError:
            logger.debug("psutil not available, skipping system metrics")
        except Exception as e:
            logger.error(f"Failed to record system metrics: {e}")
    
    def get_metric_summary(self, name: str) -> Optional[MetricSummary]:
        """Get summary statistics for a metric."""
        return self.metric_summaries.get(name)
    
    def get_recent_metrics(
        self,
        name: str,
        limit: Optional[int] = None,
        since: Optional[datetime] = None
    ) -> List[MetricPoint]:
        """Get recent metric points."""
        if name not in self.metrics:
            return []
        
        metrics = list(self.metrics[name])
        
        # Filter by time if specified
        if since:
            metrics = [m for m in metrics if m.timestamp >= since]
        
        # Limit results if specified
        if limit:
            metrics = metrics[-limit:]
        
        return metrics
    
    def get_all_metric_summaries(self) -> Dict[str, MetricSummary]:
        """Get all metric summaries."""
        return dict(self.metric_summaries)
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        uptime = (datetime.now() - self.start_time).total_seconds()
        
        # Calculate key performance indicators
        cycle_time_summary = self.get_metric_summary("trading_cycle_latency_ms")
        ml_prediction_summary = self.get_metric_summary("ml_prediction_latency_ms")
        
        report = {
            'uptime_seconds': round(uptime, 1),
            'total_metrics_recorded': self.total_metrics_recorded,
            'alerts_triggered': self.alerts_triggered,
            'metrics_per_second': round(self.total_metrics_recorded / uptime, 2) if uptime > 0 else 0,
            
            'performance_summary': {
                'trading_cycle': {
                    'target_ms': self.latency_thresholds.get('trading_cycle', 0),
                    'current_avg_ms': cycle_time_summary.avg_value if cycle_time_summary else 0,
                    'target_met': (cycle_time_summary.avg_value <= self.latency_thresholds.get('trading_cycle', float('inf'))) if cycle_time_summary else False
                },
                'ml_prediction': {
                    'target_ms': self.latency_thresholds.get('ml_prediction', 0),
                    'current_avg_ms': ml_prediction_summary.avg_value if ml_prediction_summary else 0,
                    'target_met': (ml_prediction_summary.avg_value <= self.latency_thresholds.get('ml_prediction', float('inf'))) if ml_prediction_summary else False
                }
            },
            
            'metric_counts': {name: len(points) for name, points in self.metrics.items()},
            'summary_count': len(self.metric_summaries),
            'start_time': self.start_time.isoformat()
        }
        
        return report
    
    async def _publish_metrics_to_redis(self) -> None:
        """Publish metrics to Redis for external consumption."""
        if not self.redis_client:
            return
        
        try:
            # Publish recent metric summaries
            summaries = {name: summary.to_dict() for name, summary in self.metric_summaries.items()}
            
            await self.redis_client.set(
                "trading_metrics_summary",
                json.dumps(summaries),
                ex=300  # 5 minutes
            )
            
            # Publish performance report
            report = self.get_performance_report()
            await self.redis_client.set(
                "trading_performance_report",
                json.dumps(report),
                ex=300  # 5 minutes
            )
            
            logger.debug(f"Published {len(summaries)} metric summaries to Redis")
            
        except Exception as e:
            logger.error(f"Failed to publish metrics to Redis: {e}")
    
    def _cleanup_old_metrics(self) -> None:
        """Remove old metric points to free memory."""
        cutoff_time = datetime.now() - timedelta(hours=self.metric_retention_hours)
        cleaned_count = 0
        
        for name, points in self.metrics.items():
            original_length = len(points)
            
            # Filter out old points
            while points and points[0].timestamp < cutoff_time:
                points.popleft()
                cleaned_count += 1
        
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} old metric points")
    
    def reset_metrics(self) -> None:
        """Reset all metrics (for testing)."""
        self.metrics.clear()
        self.metric_summaries.clear()
        self.total_metrics_recorded = 0
        self.alerts_triggered = 0
        self.start_time = datetime.now()
        
        logger.info("All metrics reset")
    
    async def cleanup(self) -> None:
        """Cleanup resources and stop background tasks."""
        # Cancel background tasks
        if self._publisher_task and not self._publisher_task.done():
            self._publisher_task.cancel()
            try:
                await self._publisher_task
            except asyncio.CancelledError:
                pass
        
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Final metrics publish
        if self.enable_redis_publishing:
            await self._publish_metrics_to_redis()
        
        # Close Redis connection
        if self.redis_client:
            await self.redis_client.close()
        
        logger.info("PerformanceTracker cleanup completed")


# Context manager for measuring operation latency
class LatencyMeasurement:
    """Context manager for measuring and recording operation latency."""
    
    def __init__(self, performance_tracker: PerformanceTracker, operation: str):
        self.performance_tracker = performance_tracker
        self.operation = operation
        self.start_time = 0.0
    
    async def __aenter__(self):
        self.start_time = time.perf_counter()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        end_time = time.perf_counter()
        latency_ms = (end_time - self.start_time) * 1000
        await self.performance_tracker.record_latency(self.operation, latency_ms)


# Global performance tracker instance
_performance_tracker: Optional[PerformanceTracker] = None


async def get_performance_tracker() -> PerformanceTracker:
    """Get global performance tracker instance."""
    global _performance_tracker
    if _performance_tracker is None:
        _performance_tracker = PerformanceTracker()
        await _performance_tracker.initialize()
    return _performance_tracker


async def cleanup_performance_tracker() -> None:
    """Cleanup global performance tracker."""
    global _performance_tracker
    if _performance_tracker:
        await _performance_tracker.cleanup()
        _performance_tracker = None
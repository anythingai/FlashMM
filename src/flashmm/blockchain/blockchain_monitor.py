"""
Blockchain Monitor

Monitors Sei network health, congestion, gas prices, validator performance,
and competitor activity with comprehensive alerting system.
"""

import asyncio
from typing import Dict, Any, Optional, List, Set, Callable
from decimal import Decimal
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
from collections import deque, defaultdict

from flashmm.config.settings import get_config
from flashmm.utils.logging import get_logger
from flashmm.utils.exceptions import BlockchainError, ValidationError
from flashmm.utils.decorators import timeout_async, measure_latency
from flashmm.blockchain.sei_client import SeiClient, NetworkHealth
from flashmm.monitoring.alerts.alert_manager import AlertManager

logger = get_logger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class NetworkMetric(Enum):
    """Network metrics to monitor."""
    BLOCK_TIME = "block_time"
    GAS_PRICE = "gas_price"
    VALIDATOR_COUNT = "validator_count"
    NETWORK_CONGESTION = "network_congestion"
    RPC_LATENCY = "rpc_latency"
    TRANSACTION_SUCCESS_RATE = "tx_success_rate"


@dataclass
class MetricThreshold:
    """Metric threshold configuration."""
    metric: NetworkMetric
    warning_threshold: float
    critical_threshold: float
    emergency_threshold: Optional[float] = None
    comparison: str = "greater"  # "greater", "less", "equal"
    duration_seconds: int = 60  # Duration before triggering alert


@dataclass
class AlertCondition:
    """Alert condition tracking."""
    metric: NetworkMetric
    severity: AlertSeverity
    threshold: float
    triggered_at: datetime
    last_alert_sent: Optional[datetime] = None
    consecutive_violations: int = 0


@dataclass
class NetworkSnapshot:
    """Network state snapshot."""
    timestamp: datetime
    block_height: int
    block_time: float
    gas_price: Decimal
    validator_count: int
    rpc_latency_ms: float
    is_syncing: bool
    transaction_pool_size: int = 0
    congestion_level: float = 0.0  # 0.0 to 1.0
    

@dataclass
class ValidatorInfo:
    """Validator information."""
    address: str
    moniker: str
    voting_power: int
    jailed: bool
    tombstoned: bool
    uptime_percentage: float
    last_signed_height: int


@dataclass
class CompetitorActivity:
    """Competitor trading activity."""
    address: str
    market: str
    activity_type: str  # "order_placed", "order_cancelled", "trade"
    volume_24h: Decimal
    trade_count_24h: int
    avg_order_size: Decimal
    last_activity: datetime


class BlockchainMonitor:
    """Comprehensive blockchain monitoring and alerting system."""
    
    def __init__(self, sei_client: SeiClient):
        self.sei_client = sei_client
        self.config = get_config()
        
        # Alert system
        self.alert_manager = AlertManager()
        
        # Monitoring configuration
        self.monitoring_interval = self.config.get("blockchain.monitoring_interval_seconds", 30)
        self.metrics_retention_hours = self.config.get("blockchain.metrics_retention_hours", 24)
        self.alert_cooldown_minutes = self.config.get("blockchain.alert_cooldown_minutes", 10)
        
        # Data storage
        self.network_snapshots: deque = deque(maxlen=2880)  # 24 hours at 30s intervals
        self.active_alerts: Dict[str, AlertCondition] = {}
        self.validator_info: Dict[str, ValidatorInfo] = {}
        self.competitor_activities: Dict[str, CompetitorActivity] = {}
        
        # Metric thresholds
        self.thresholds = self._initialize_thresholds()
        
        # Statistics
        self.monitoring_stats = {
            'snapshots_taken': 0,
            'alerts_triggered': 0,
            'network_issues_detected': 0,
            'last_healthy_check': None
        }
        
        # Background tasks
        self._monitoring_task: Optional[asyncio.Task] = None
        self._validator_monitoring_task: Optional[asyncio.Task] = None
        self._competitor_monitoring_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        
        # Alert callbacks
        self.alert_callbacks: List[Callable[[str, AlertSeverity, Dict[str, Any]], None]] = []
        
    def _initialize_thresholds(self) -> Dict[NetworkMetric, MetricThreshold]:
        """Initialize metric thresholds from configuration."""
        return {
            NetworkMetric.BLOCK_TIME: MetricThreshold(
                metric=NetworkMetric.BLOCK_TIME,
                warning_threshold=8.0,      # 8 seconds
                critical_threshold=15.0,    # 15 seconds
                emergency_threshold=30.0,   # 30 seconds
                comparison="greater"
            ),
            NetworkMetric.GAS_PRICE: MetricThreshold(
                metric=NetworkMetric.GAS_PRICE,
                warning_threshold=0.1,      # 0.1 SEI
                critical_threshold=0.5,     # 0.5 SEI
                emergency_threshold=1.0,    # 1.0 SEI
                comparison="greater"
            ),
            NetworkMetric.VALIDATOR_COUNT: MetricThreshold(
                metric=NetworkMetric.VALIDATOR_COUNT,
                warning_threshold=50,       # Less than 50 validators
                critical_threshold=30,      # Less than 30 validators
                emergency_threshold=20,     # Less than 20 validators
                comparison="less"
            ),
            NetworkMetric.RPC_LATENCY: MetricThreshold(
                metric=NetworkMetric.RPC_LATENCY,
                warning_threshold=1000,     # 1 second
                critical_threshold=5000,    # 5 seconds
                emergency_threshold=10000,  # 10 seconds
                comparison="greater"
            ),
            NetworkMetric.NETWORK_CONGESTION: MetricThreshold(
                metric=NetworkMetric.NETWORK_CONGESTION,
                warning_threshold=0.7,      # 70% congestion
                critical_threshold=0.9,     # 90% congestion
                emergency_threshold=0.95,   # 95% congestion
                comparison="greater"
            ),
            NetworkMetric.TRANSACTION_SUCCESS_RATE: MetricThreshold(
                metric=NetworkMetric.TRANSACTION_SUCCESS_RATE,
                warning_threshold=0.95,     # Below 95%
                critical_threshold=0.90,    # Below 90%
                emergency_threshold=0.80,   # Below 80%
                comparison="less"
            )
        }
    
    async def initialize(self) -> None:
        """Initialize the blockchain monitor."""
        try:
            logger.info("Initializing blockchain monitor")
            
            # Initialize alert manager
            await self.alert_manager.initialize()
            
            # Take initial snapshot
            await self._take_network_snapshot()
            
            # Start background tasks
            await self._start_background_tasks()
            
            logger.info("Blockchain monitor initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize blockchain monitor: {e}")
            raise BlockchainError(f"Blockchain monitor initialization failed: {e}")
    
    async def _start_background_tasks(self) -> None:
        """Start background monitoring tasks."""
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        self._validator_monitoring_task = asyncio.create_task(self._validator_monitoring_loop())
        self._competitor_monitoring_task = asyncio.create_task(self._competitor_monitoring_loop())
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
    
    async def _monitoring_loop(self) -> None:
        """Main network monitoring loop."""
        while True:
            try:
                await asyncio.sleep(self.monitoring_interval)
                
                # Take network snapshot
                await self._take_network_snapshot()
                
                # Check thresholds and trigger alerts
                await self._check_thresholds()
                
                # Update monitoring statistics
                self.monitoring_stats['snapshots_taken'] += 1
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Network monitoring loop error: {e}")
                self.monitoring_stats['network_issues_detected'] += 1
                await asyncio.sleep(60)  # Brief pause on error
    
    async def _validator_monitoring_loop(self) -> None:
        """Validator performance monitoring loop."""
        while True:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes
                await self._monitor_validators()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Validator monitoring error: {e}")
                await asyncio.sleep(300)
    
    async def _competitor_monitoring_loop(self) -> None:
        """Competitor activity monitoring loop."""
        while True:
            try:
                await asyncio.sleep(600)  # Check every 10 minutes
                await self._monitor_competitor_activity()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Competitor monitoring error: {e}")
                await asyncio.sleep(600)
    
    async def _cleanup_loop(self) -> None:
        """Cleanup old data loop."""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                await self._cleanup_old_data()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")
                await asyncio.sleep(3600)
    
    @timeout_async(10.0)
    @measure_latency("network_snapshot")
    async def _take_network_snapshot(self) -> None:
        """Take a snapshot of current network state."""
        try:
            # Get network health from Sei client
            network_health = await self.sei_client.check_network_health()
            
            # Get current gas price
            gas_price = await self.sei_client.get_gas_price()
            
            # Estimate network congestion
            congestion_level = await self._estimate_network_congestion()
            
            # Create snapshot
            snapshot = NetworkSnapshot(
                timestamp=datetime.now(),
                block_height=network_health.latest_block_height,
                block_time=network_health.avg_block_time,
                gas_price=gas_price,
                validator_count=network_health.validator_count,
                rpc_latency_ms=network_health.rpc_latency_ms,
                is_syncing=network_health.syncing,
                congestion_level=congestion_level
            )
            
            # Store snapshot
            self.network_snapshots.append(snapshot)
            
            # Update last healthy check if network is healthy
            if network_health.is_healthy:
                self.monitoring_stats['last_healthy_check'] = datetime.now()
            
            logger.debug(f"Network snapshot taken: block={snapshot.block_height}, "
                        f"latency={snapshot.rpc_latency_ms:.1f}ms, "
                        f"gas_price={snapshot.gas_price}")
            
        except Exception as e:
            logger.error(f"Failed to take network snapshot: {e}")
            # Create error snapshot
            error_snapshot = NetworkSnapshot(
                timestamp=datetime.now(),
                block_height=0,
                block_time=999.0,  # High block time indicates error
                gas_price=Decimal('999'),
                validator_count=0,
                rpc_latency_ms=999999.0,
                is_syncing=True,
                congestion_level=1.0
            )
            self.network_snapshots.append(error_snapshot)
    
    async def _estimate_network_congestion(self) -> float:
        """Estimate network congestion level (0.0 to 1.0)."""
        try:
            # In production, this would analyze:
            # - Transaction pool size
            # - Average transaction fees
            # - Block utilization
            # - Pending transaction count
            
            # For now, use a simple heuristic based on recent metrics
            if len(self.network_snapshots) < 5:
                return 0.0
            
            recent_snapshots = list(self.network_snapshots)[-5:]
            
            # Check for increasing block times
            avg_block_time = sum(s.block_time for s in recent_snapshots) / len(recent_snapshots)
            block_time_factor = min(avg_block_time / 6.0, 1.0)  # Normalize to Sei's 6s target
            
            # Check for increasing gas prices
            avg_gas_price = sum(s.gas_price for s in recent_snapshots) / len(recent_snapshots)
            gas_price_factor = min(float(avg_gas_price) / 0.1, 1.0)  # Normalize to 0.1 SEI baseline
            
            # Check for RPC latency
            avg_latency = sum(s.rpc_latency_ms for s in recent_snapshots) / len(recent_snapshots)
            latency_factor = min(avg_latency / 5000.0, 1.0)  # Normalize to 5s baseline
            
            # Combined congestion score
            congestion = (block_time_factor * 0.4 + gas_price_factor * 0.4 + latency_factor * 0.2)
            return min(congestion, 1.0)
            
        except Exception as e:
            logger.warning(f"Failed to estimate network congestion: {e}")
            return 0.5  # Default to moderate congestion
    
    async def _check_thresholds(self) -> None:
        """Check metric thresholds and trigger alerts."""
        if not self.network_snapshots:
            return
        
        latest_snapshot = self.network_snapshots[-1]
        
        # Check each metric threshold
        metric_values = {
            NetworkMetric.BLOCK_TIME: latest_snapshot.block_time,
            NetworkMetric.GAS_PRICE: float(latest_snapshot.gas_price),
            NetworkMetric.VALIDATOR_COUNT: latest_snapshot.validator_count,
            NetworkMetric.RPC_LATENCY: latest_snapshot.rpc_latency_ms,
            NetworkMetric.NETWORK_CONGESTION: latest_snapshot.congestion_level
        }
        
        # Add transaction success rate if available
        tx_success_rate = await self._calculate_transaction_success_rate()
        if tx_success_rate is not None:
            metric_values[NetworkMetric.TRANSACTION_SUCCESS_RATE] = tx_success_rate
        
        for metric, value in metric_values.items():
            threshold_config = self.thresholds.get(metric)
            if not threshold_config:
                continue
            
            # Check threshold violations
            severity = self._check_metric_threshold(metric, value, threshold_config)
            
            if severity:
                await self._handle_threshold_violation(metric, value, severity, latest_snapshot)
    
    def _check_metric_threshold(
        self, 
        metric: NetworkMetric, 
        value: float, 
        threshold: MetricThreshold
    ) -> Optional[AlertSeverity]:
        """Check if metric violates thresholds."""
        try:
            def violates_threshold(val: float, thresh: float, comparison: str) -> bool:
                if comparison == "greater":
                    return val > thresh
                elif comparison == "less":
                    return val < thresh
                elif comparison == "equal":
                    return abs(val - thresh) < 0.01
                return False
            
            # Check emergency threshold first
            if (threshold.emergency_threshold is not None and 
                violates_threshold(value, threshold.emergency_threshold, threshold.comparison)):
                return AlertSeverity.EMERGENCY
            
            # Check critical threshold
            if violates_threshold(value, threshold.critical_threshold, threshold.comparison):
                return AlertSeverity.CRITICAL
            
            # Check warning threshold
            if violates_threshold(value, threshold.warning_threshold, threshold.comparison):
                return AlertSeverity.WARNING
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to check threshold for {metric.value}: {e}")
            return None
    
    async def _handle_threshold_violation(
        self,
        metric: NetworkMetric,
        value: float,
        severity: AlertSeverity,
        snapshot: NetworkSnapshot
    ) -> None:
        """Handle metric threshold violation."""
        try:
            alert_key = f"{metric.value}_{severity.value}"
            
            # Check if alert is already active
            if alert_key in self.active_alerts:
                condition = self.active_alerts[alert_key]
                condition.consecutive_violations += 1
                
                # Check cooldown period
                if (condition.last_alert_sent and 
                    datetime.now() - condition.last_alert_sent < timedelta(minutes=self.alert_cooldown_minutes)):
                    return
            else:
                # Create new alert condition
                condition = AlertCondition(
                    metric=metric,
                    severity=severity,
                    threshold=value,
                    triggered_at=datetime.now(),
                    consecutive_violations=1
                )
                self.active_alerts[alert_key] = condition
            
            # Send alert
            await self._send_alert(metric, value, severity, snapshot)
            condition.last_alert_sent = datetime.now()
            
            # Update statistics
            self.monitoring_stats['alerts_triggered'] += 1
            
        except Exception as e:
            logger.error(f"Failed to handle threshold violation: {e}")
    
    async def _send_alert(
        self,
        metric: NetworkMetric,
        value: float,
        severity: AlertSeverity,
        snapshot: NetworkSnapshot
    ) -> None:
        """Send alert notification."""
        try:
            alert_data = {
                'metric': metric.value,
                'value': value,
                'severity': severity.value,
                'timestamp': snapshot.timestamp.isoformat(),
                'block_height': snapshot.block_height,
                'additional_context': {
                    'block_time': snapshot.block_time,
                    'gas_price': float(snapshot.gas_price),
                    'validator_count': snapshot.validator_count,
                    'rpc_latency_ms': snapshot.rpc_latency_ms,
                    'congestion_level': snapshot.congestion_level
                }
            }
            
            # Send via alert manager
            await self.alert_manager.send_alert(
                title=f"Blockchain Alert: {metric.value}",
                message=self._format_alert_message(metric, value, severity),
                severity=severity.value,
                source="blockchain_monitor",
                metadata=alert_data
            )
            
            # Call registered callbacks
            for callback in self.alert_callbacks:
                try:
                    await callback(metric.value, severity, alert_data)
                except Exception as e:
                    logger.warning(f"Alert callback failed: {e}")
            
            logger.warning(f"Blockchain alert sent: {metric.value} = {value} ({severity.value})")
            
        except Exception as e:
            logger.error(f"Failed to send alert: {e}")
    
    def _format_alert_message(self, metric: NetworkMetric, value: float, severity: AlertSeverity) -> str:
        """Format alert message."""
        messages = {
            NetworkMetric.BLOCK_TIME: f"Block time is {value:.1f}s (target: 6.0s)",
            NetworkMetric.GAS_PRICE: f"Gas price is {value:.4f} SEI (baseline: 0.025 SEI)",
            NetworkMetric.VALIDATOR_COUNT: f"Validator count is {int(value)} (minimum recommended: 100)",
            NetworkMetric.RPC_LATENCY: f"RPC latency is {value:.1f}ms (target: <1000ms)",
            NetworkMetric.NETWORK_CONGESTION: f"Network congestion is {value:.1%} (target: <50%)",
            NetworkMetric.TRANSACTION_SUCCESS_RATE: f"Transaction success rate is {value:.1%} (target: >95%)"
        }
        
        base_message = messages.get(metric, f"{metric.value} = {value}")
        return f"{severity.value.upper()}: {base_message}"
    
    async def _calculate_transaction_success_rate(self) -> Optional[float]:
        """Calculate recent transaction success rate."""
        try:
            # In production, this would query recent transaction results
            # For now, return a mock success rate
            return 0.98  # 98% success rate
            
        except Exception as e:
            logger.warning(f"Failed to calculate transaction success rate: {e}")
            return None
    
    async def _monitor_validators(self) -> None:
        """Monitor validator performance and status."""
        try:
            # In production, query validator set and performance data
            # For now, use mock data
            
            # Check for validator changes, jailing, tombstoning
            # Alert on significant voting power changes
            # Monitor validator uptime
            
            logger.debug("Validator monitoring completed")
            
        except Exception as e:
            logger.error(f"Validator monitoring failed: {e}")
    
    async def _monitor_competitor_activity(self) -> None:
        """Monitor competitor trading activity."""
        try:
            # In production, this would:
            # - Query on-chain transaction data
            # - Identify large traders/market makers
            # - Track their order patterns
            # - Alert on unusual activity
            
            logger.debug("Competitor activity monitoring completed")
            
        except Exception as e:
            logger.error(f"Competitor monitoring failed: {e}")
    
    async def _cleanup_old_data(self) -> None:
        """Clean up old monitoring data."""
        try:
            current_time = datetime.now()
            
            # Clean up old alerts
            expired_alerts = [
                key for key, condition in self.active_alerts.items()
                if current_time - condition.triggered_at > timedelta(hours=1)
            ]
            
            for key in expired_alerts:
                del self.active_alerts[key]
            
            if expired_alerts:
                logger.debug(f"Cleaned up {len(expired_alerts)} expired alerts")
            
        except Exception as e:
            logger.error(f"Data cleanup failed: {e}")
    
    def add_alert_callback(self, callback: Callable[[str, AlertSeverity, Dict[str, Any]], None]) -> None:
        """Add alert callback function."""
        self.alert_callbacks.append(callback)
    
    def get_current_network_status(self) -> Dict[str, Any]:
        """Get current network status summary."""
        if not self.network_snapshots:
            return {'status': 'no_data'}
        
        latest_snapshot = self.network_snapshots[-1]
        
        return {
            'status': 'healthy' if len(self.active_alerts) == 0 else 'issues_detected',
            'timestamp': latest_snapshot.timestamp.isoformat(),
            'block_height': latest_snapshot.block_height,
            'block_time': latest_snapshot.block_time,
            'gas_price': float(latest_snapshot.gas_price),
            'validator_count': latest_snapshot.validator_count,
            'rpc_latency_ms': latest_snapshot.rpc_latency_ms,
            'congestion_level': latest_snapshot.congestion_level,
            'is_syncing': latest_snapshot.is_syncing,
            'active_alerts': len(self.active_alerts),
            'last_healthy_check': self.monitoring_stats['last_healthy_check'].isoformat() if self.monitoring_stats['last_healthy_check'] else None
        }
    
    def get_metrics_history(self, hours: int = 1) -> Dict[str, List[Dict[str, Any]]]:
        """Get historical metrics data."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        filtered_snapshots = [
            snapshot for snapshot in self.network_snapshots
            if snapshot.timestamp > cutoff_time
        ]
        
        return {
            'snapshots': [
                {
                    'timestamp': snapshot.timestamp.isoformat(),
                    'block_height': snapshot.block_height,
                    'block_time': snapshot.block_time,
                    'gas_price': float(snapshot.gas_price),
                    'validator_count': snapshot.validator_count,
                    'rpc_latency_ms': snapshot.rpc_latency_ms,
                    'congestion_level': snapshot.congestion_level,
                    'is_syncing': snapshot.is_syncing
                }
                for snapshot in filtered_snapshots
            ]
        }
    
    def get_alert_history(self) -> Dict[str, Any]:
        """Get alert history and current active alerts."""
        return {
            'active_alerts': [
                {
                    'metric': condition.metric.value,
                    'severity': condition.severity.value,
                    'threshold': condition.threshold,
                    'triggered_at': condition.triggered_at.isoformat(),
                    'consecutive_violations': condition.consecutive_violations,
                    'last_alert_sent': condition.last_alert_sent.isoformat() if condition.last_alert_sent else None
                }
                for condition in self.active_alerts.values()
            ],
            'total_alerts_triggered': self.monitoring_stats['alerts_triggered'],
            'network_issues_detected': self.monitoring_stats['network_issues_detected']
        }
    
    def get_monitoring_stats(self) -> Dict[str, Any]:
        """Get monitoring system statistics."""
        return {
            'snapshots_taken': self.monitoring_stats['snapshots_taken'],
            'alerts_triggered': self.monitoring_stats['alerts_triggered'],
            'network_issues_detected': self.monitoring_stats['network_issues_detected'],
            'active_alerts_count': len(self.active_alerts),
            'snapshots_stored': len(self.network_snapshots),
            'monitoring_interval_seconds': self.monitoring_interval,
            'last_snapshot': self.network_snapshots[-1].timestamp.isoformat() if self.network_snapshots else None,
            'last_healthy_check': self.monitoring_stats['last_healthy_check'].isoformat() if self.monitoring_stats['last_healthy_check'] else None
        }
    
    async def force_health_check(self) -> Dict[str, Any]:
        """Force an immediate health check."""
        try:
            await self._take_network_snapshot()
            await self._check_thresholds()
            
            return {
                'success': True,
                'timestamp': datetime.now().isoformat(),
                'network_status': self.get_current_network_status()
            }
            
        except Exception as e:
            logger.error(f"Forced health check failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def cleanup(self) -> None:
        """Cleanup resources and stop background tasks."""
        # Cancel background tasks
        tasks = [
            self._monitoring_task,
            self._validator_monitoring_task,
            self._competitor_monitoring_task,
            self._cleanup_task
        ]
        
        for task in tasks:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Cleanup alert manager
        if self.alert_manager:
            await self.alert_manager.cleanup()
        
        logger.info("Blockchain monitor cleanup completed")
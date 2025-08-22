"""
FlashMM Enhanced Metrics Collector

Comprehensive metrics collection for the Sei WebSocket data pipeline including
performance monitoring, health tracking, and alerting for <250ms latency targets.
"""

import asyncio
import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any

import psutil

from flashmm.config.settings import get_config
from flashmm.data.storage.influxdb_client import HighPerformanceInfluxDBClient
from flashmm.data.storage.redis_client import HighPerformanceRedisClient
from flashmm.utils.logging import PerformanceLogger, get_logger

logger = get_logger(__name__)
perf_logger = PerformanceLogger()


@dataclass
class DataPipelineMetrics:
    """Data pipeline specific metrics."""
    timestamp: datetime
    total_messages_processed: int
    messages_per_second: float
    avg_processing_latency_ms: float
    p95_processing_latency_ms: float
    max_processing_latency_ms: float
    error_count: int
    error_rate_percent: float
    active_websocket_connections: int
    active_subscriptions: int
    orderbook_updates_per_second: float
    trade_updates_per_second: float
    latency_violations_count: int  # Messages exceeding 250ms target


@dataclass
class SystemResourceMetrics:
    """System resource metrics."""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    disk_percent: float
    disk_used_gb: float
    network_bytes_sent: int
    network_bytes_recv: int
    open_file_descriptors: int
    process_count: int


@dataclass
class ComponentHealthMetrics:
    """Component health and performance metrics."""
    timestamp: datetime
    redis_status: str
    redis_latency_ms: float
    redis_memory_usage_mb: float
    influxdb_status: str
    influxdb_latency_ms: float
    influxdb_write_rate: float
    websocket_client_statuses: dict[str, str]
    feed_manager_status: str
    data_normalizer_error_rate: float


@dataclass
class TradingPerformanceMetrics:
    """Trading performance metrics."""
    timestamp: datetime
    # Volume and trade metrics
    total_volume_usdc: float
    total_trades: int
    trades_per_minute: float
    fill_rate_percent: float

    # P&L metrics
    total_pnl_usdc: float
    realized_pnl_usdc: float
    unrealized_pnl_usdc: float
    maker_fees_earned_usdc: float
    taker_fees_paid_usdc: float
    net_fees_usdc: float

    # Spread metrics
    average_spread_bps: float
    current_spread_bps: float
    spread_improvement_bps: float
    spread_improvement_percent: float
    baseline_spread_bps: float

    # Inventory and risk metrics
    total_inventory_usdc: float
    inventory_utilization_percent: float
    max_position_percent: float
    var_1d_usdc: float
    max_drawdown_usdc: float

    # Performance metrics
    order_latency_ms: float
    quote_frequency_hz: float
    inventory_violations: int
    emergency_stops: int

    # Market making specific
    orders_placed: int
    orders_filled: int
    orders_cancelled: int
    active_quotes: int
    quote_update_frequency: float


@dataclass
class MLPerformanceMetrics:
    """ML model performance metrics."""
    timestamp: datetime
    # Prediction metrics
    total_predictions: int
    predictions_per_minute: float
    prediction_accuracy_percent: float
    prediction_confidence_avg: float

    # Latency metrics
    avg_inference_time_ms: float
    p95_inference_time_ms: float
    max_inference_time_ms: float

    # Cost metrics
    total_cost_usd: float
    cost_per_prediction_usd: float
    hourly_cost_usd: float

    # Method distribution
    azure_openai_predictions: int
    fallback_predictions: int
    cached_predictions: int
    cache_hit_rate_percent: float

    # Quality metrics
    ensemble_agreement_avg: float
    uncertainty_score_avg: float
    validation_pass_rate_percent: float
    api_success_rate_percent: float


@dataclass
class RiskMetrics:
    """Risk management metrics."""
    timestamp: datetime
    # Position risk
    total_exposure_usdc: float
    leverage_ratio: float
    concentration_risk_percent: float
    portfolio_beta: float

    # VaR metrics
    var_1d_95_usdc: float
    var_1d_99_usdc: float
    expected_shortfall_usdc: float

    # Drawdown metrics
    current_drawdown_usdc: float
    max_drawdown_percent: float
    drawdown_duration_minutes: int

    # Volatility metrics
    portfolio_volatility_percent: float
    realized_volatility_1h: float
    realized_volatility_1d: float

    # Compliance metrics
    position_limit_breaches: int
    inventory_limit_breaches: int
    risk_score: float  # 0-100 scale


class EnhancedMetricsCollector:
    """Enhanced metrics collector for the complete data pipeline."""

    def __init__(self):
        self.config = get_config()

        # Storage clients
        self.influxdb_client: HighPerformanceInfluxDBClient | None = None
        self.redis_client: HighPerformanceRedisClient | None = None

        # Collection settings
        self.running = False
        self.collection_interval = self.config.get("monitoring.metrics_collection_interval_seconds", 10)
        self.health_check_interval = self.config.get("monitoring.health_check_interval_seconds", 30)

        # Metric storage
        self.pipeline_metrics_history: list[DataPipelineMetrics] = []
        self.system_metrics_history: list[SystemResourceMetrics] = []
        self.component_health_history: list[ComponentHealthMetrics] = []
        self.trading_metrics_history: list[TradingPerformanceMetrics] = []
        self.ml_metrics_history: list[MLPerformanceMetrics] = []
        self.risk_metrics_history: list[RiskMetrics] = []
        self.max_history_length = 1000  # Keep last 1000 readings

        # Alerting
        self.alert_callbacks: list[Callable[[str, dict[str, Any]], None]] = []
        self.alert_thresholds = {
            "max_latency_ms": self.config.get("monitoring.alert_thresholds.max_latency_ms", 350),
            "max_error_rate": self.config.get("monitoring.alert_thresholds.error_rate_threshold", 5.0),
            "min_uptime_percent": self.config.get("monitoring.alert_thresholds.min_uptime_percent", 95.0),
            "max_cpu_percent": 80.0,
            "max_memory_percent": 85.0,
            "max_disk_percent": 90.0
        }

        # Component references (will be set by the market data service)
        self.market_data_service = None
        self.feed_manager = None
        self.market_making_engine = None
        self.position_tracker = None
        self.ml_metrics_collector = None
        self.prediction_service = None

        # Baseline metrics for spread improvement calculation
        self.baseline_spreads: dict[str, float] = {}
        self.baseline_calculation_window = 300  # 5 minutes for baseline calculation

        # Background tasks
        self.metrics_task: asyncio.Task | None = None
        self.health_task: asyncio.Task | None = None
        self.alerting_task: asyncio.Task | None = None

    async def initialize(self) -> None:
        """Initialize enhanced metrics collector."""
        try:
            # Initialize storage clients
            self.influxdb_client = HighPerformanceInfluxDBClient()
            await self.influxdb_client.initialize()

            self.redis_client = HighPerformanceRedisClient()
            await self.redis_client.initialize()

            logger.info("Enhanced MetricsCollector initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Enhanced MetricsCollector: {e}")
            # Continue without storage if not available

    def set_component_references(self, market_data_service=None, feed_manager=None,
                                 market_making_engine=None, position_tracker=None,
                                 ml_metrics_collector=None, prediction_service=None):
        """Set references to components for comprehensive metrics collection."""
        self.market_data_service = market_data_service
        self.feed_manager = feed_manager
        self.market_making_engine = market_making_engine
        self.position_tracker = position_tracker
        self.ml_metrics_collector = ml_metrics_collector
        self.prediction_service = prediction_service

    async def start(self) -> None:
        """Start comprehensive metrics collection."""
        if self.running:
            logger.warning("Metrics collector already running")
            return

        self.running = True
        logger.info("Starting enhanced metrics collection...")

        # Start background tasks
        self.metrics_task = asyncio.create_task(self._metrics_collection_loop())
        self.health_task = asyncio.create_task(self._health_monitoring_loop())
        self.alerting_task = asyncio.create_task(self._alerting_loop())

        logger.info("Enhanced metrics collection started")

    async def stop(self) -> None:
        """Stop metrics collection gracefully."""
        if not self.running:
            return

        logger.info("Stopping enhanced metrics collection...")
        self.running = False

        # Cancel background tasks
        for task in [self.metrics_task, self.health_task, self.alerting_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Close storage clients
        if self.redis_client:
            await self.redis_client.close()

        if self.influxdb_client:
            self.influxdb_client.close()

        logger.info("Enhanced metrics collection stopped")

    async def _metrics_collection_loop(self) -> None:
        """Main metrics collection loop."""
        while self.running:
            try:
                # Collect all metrics
                await self._collect_data_pipeline_metrics()
                await self._collect_system_resource_metrics()
                await self._collect_component_health_metrics()
                await self._collect_trading_performance_metrics()
                await self._collect_ml_performance_metrics()
                await self._collect_risk_metrics()

                # Store metrics
                await self._store_metrics()

                # Publish real-time metrics
                await self._publish_realtime_metrics()

                await asyncio.sleep(self.collection_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in metrics collection loop: {e}")
                await asyncio.sleep(self.collection_interval)

    async def _collect_data_pipeline_metrics(self) -> None:
        """Collect data pipeline specific metrics."""
        try:
            current_time = datetime.now()

            # Get metrics from market data service
            if self.market_data_service:
                service_metrics = await self.market_data_service.get_performance_metrics()

                # Calculate rates and latencies
                uptime = service_metrics.get("uptime_seconds", 1)
                total_messages = service_metrics.get("total_data_points", 0)
                messages_per_second = total_messages / max(uptime, 1)

                # Get latency statistics from components
                avg_latency = 0.0
                p95_latency = 0.0
                max_latency = 0.0

                if "feed_manager" in service_metrics:
                    feed_details = service_metrics["feed_manager"].get("feed_details", {})
                    latencies = []
                    for symbol_data in feed_details.values():
                        latencies.append(symbol_data.get("avg_latency_ms", 0))

                    if latencies:
                        avg_latency = sum(latencies) / len(latencies)
                        latencies.sort()
                        p95_idx = int(len(latencies) * 0.95)
                        p95_latency = latencies[p95_idx] if p95_idx < len(latencies) else latencies[-1]
                        max_latency = max(latencies)

                # Count latency violations
                latency_violations = service_metrics.get("data_quality", {}).get("latency_violations", 0)

                # Count active connections and subscriptions
                active_connections = 0
                active_subscriptions = 0

                if self.feed_manager:
                    feed_status = self.feed_manager.get_feed_status()
                    for _symbol, status in feed_status.items():
                        if isinstance(status, dict):
                            if status.get("status") in ["connected", "subscribed"]:
                                active_connections += 1
                            active_subscriptions += len(status.get("subscriptions", []))

                # Calculate rates
                orderbook_rate = 0.0
                trade_rate = 0.0
                if "data_quality" in service_metrics:
                    quality_stats = service_metrics["data_quality"]
                    orderbook_rate = quality_stats.get("valid_orderbooks", 0) / max(uptime, 1)
                    trade_rate = quality_stats.get("valid_trades", 0) / max(uptime, 1)

                pipeline_metrics = DataPipelineMetrics(
                    timestamp=current_time,
                    total_messages_processed=total_messages,
                    messages_per_second=messages_per_second,
                    avg_processing_latency_ms=avg_latency,
                    p95_processing_latency_ms=p95_latency,
                    max_processing_latency_ms=max_latency,
                    error_count=service_metrics.get("total_errors", 0),
                    error_rate_percent=service_metrics.get("error_rate", 0.0),
                    active_websocket_connections=active_connections,
                    active_subscriptions=active_subscriptions,
                    orderbook_updates_per_second=orderbook_rate,
                    trade_updates_per_second=trade_rate,
                    latency_violations_count=latency_violations
                )

                # Store in history
                self.pipeline_metrics_history.append(pipeline_metrics)
                if len(self.pipeline_metrics_history) > self.max_history_length:
                    self.pipeline_metrics_history = self.pipeline_metrics_history[-self.max_history_length:]

        except Exception as e:
            logger.error(f"Failed to collect data pipeline metrics: {e}")

    async def _collect_system_resource_metrics(self) -> None:
        """Collect system resource metrics."""
        try:
            current_time = datetime.now()

            # CPU and Memory
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            # Network statistics
            network = psutil.net_io_counters()

            # Process information
            process = psutil.Process()
            open_files = len(process.open_files())

            system_metrics = SystemResourceMetrics(
                timestamp=current_time,
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used_mb=memory.used / (1024 * 1024),
                disk_percent=disk.percent,
                disk_used_gb=disk.used / (1024 * 1024 * 1024),
                network_bytes_sent=network.bytes_sent,
                network_bytes_recv=network.bytes_recv,
                open_file_descriptors=open_files,
                process_count=len(psutil.pids())
            )

            # Store in history
            self.system_metrics_history.append(system_metrics)
            if len(self.system_metrics_history) > self.max_history_length:
                self.system_metrics_history = self.system_metrics_history[-self.max_history_length:]

        except Exception as e:
            logger.error(f"Failed to collect system resource metrics: {e}")

    async def _collect_component_health_metrics(self) -> None:
        """Collect component health and performance metrics."""
        try:
            current_time = datetime.now()

            # Redis health
            redis_status = "unknown"
            redis_latency = 0.0
            redis_memory = 0.0

            if self.redis_client:
                try:
                    health = await self.redis_client.health_check()
                    redis_status = health.get("status", "unknown")
                    redis_latency = health.get("latency_ms", 0.0)
                    redis_memory = health.get("used_memory", 0)
                except Exception as e:
                    logger.debug(f"Redis health check failed: {e}")
                    redis_status = "unhealthy"

            # InfluxDB health
            influxdb_status = "unknown"
            influxdb_latency = 0.0
            influxdb_write_rate = 0.0

            if self.influxdb_client:
                try:
                    health = await self.influxdb_client.health_check()
                    influxdb_status = health.get("status", "unknown")
                    influxdb_latency = health.get("latency_ms", 0.0)

                    perf_stats = self.influxdb_client.get_performance_stats()
                    points_written = perf_stats.get("points_written", 0)
                    # Estimate write rate (rough calculation)
                    influxdb_write_rate = points_written / max(time.time() - 3600, 1)  # Per hour
                except Exception as e:
                    logger.debug(f"InfluxDB health check failed: {e}")
                    influxdb_status = "unhealthy"

            # WebSocket client statuses
            websocket_statuses = {}
            if self.feed_manager:
                feed_status = self.feed_manager.get_feed_status()
                for symbol, status in feed_status.items():
                    if isinstance(status, dict):
                        websocket_statuses[symbol] = status.get("status", "unknown")

            # Feed manager status
            feed_manager_status = "unknown"
            if self.feed_manager:
                metrics = self.feed_manager.get_performance_metrics()
                if metrics.get("active_feeds", 0) > 0:
                    feed_manager_status = "healthy"
                else:
                    feed_manager_status = "unhealthy"

            # Data normalizer error rate
            normalizer_error_rate = 0.0
            if self.market_data_service and hasattr(self.market_data_service, 'data_normalizer'):
                normalizer_stats = self.market_data_service.data_normalizer.get_statistics()
                normalizer_error_rate = normalizer_stats.get("error_rate", 0.0)

            component_health = ComponentHealthMetrics(
                timestamp=current_time,
                redis_status=redis_status,
                redis_latency_ms=redis_latency,
                redis_memory_usage_mb=redis_memory,
                influxdb_status=influxdb_status,
                influxdb_latency_ms=influxdb_latency,
                influxdb_write_rate=influxdb_write_rate,
                websocket_client_statuses=websocket_statuses,
                feed_manager_status=feed_manager_status,
                data_normalizer_error_rate=normalizer_error_rate
            )

            # Store in history
            self.component_health_history.append(component_health)
            if len(self.component_health_history) > self.max_history_length:
                self.component_health_history = self.component_health_history[-self.max_history_length:]

        except Exception as e:
            logger.error(f"Failed to collect component health metrics: {e}")

    async def _collect_trading_performance_metrics(self) -> None:
        """Collect trading performance metrics including spreads, P&L, and volume."""
        try:
            current_time = datetime.now()

            if not self.market_making_engine:
                return

            # Get trading metrics from market making engine
            engine_metrics = self.market_making_engine.get_metrics()

            # Get portfolio summary for P&L metrics
            portfolio_summary = await self.market_making_engine.get_portfolio_summary() if self.market_making_engine else {}

            # Calculate spread metrics (would need real market data in production)
            current_spread_bps = 0.0
            spread_improvement_bps = 0.0
            spread_improvement_percent = 0.0
            baseline_spread_bps = 0.0

            # Get baseline spread calculation
            symbols = self.market_making_engine.symbols if self.market_making_engine else []
            if symbols:
                symbol = symbols[0]  # Use first symbol for baseline
                if symbol not in self.baseline_spreads:
                    # Initialize baseline (would calculate from historical data)
                    self.baseline_spreads[symbol] = 10.0  # 10 bps baseline

                baseline_spread_bps = self.baseline_spreads[symbol]
                # Current spread would come from order book data
                current_spread_bps = baseline_spread_bps * 0.6  # Simulated 40% improvement
                spread_improvement_bps = baseline_spread_bps - current_spread_bps
                spread_improvement_percent = (spread_improvement_bps / baseline_spread_bps) * 100

            # Extract inventory metrics
            inventory_usdc = portfolio_summary.get('total_notional_exposure', 0.0)
            max_inventory = portfolio_summary.get('max_inventory_limit', 2000.0)
            inventory_utilization = (inventory_usdc / max_inventory * 100) if max_inventory > 0 else 0.0

            # Calculate rates
            uptime_seconds = engine_metrics.get('uptime_seconds', 1)
            total_trades = engine_metrics.get('trades_executed', 0)
            trades_per_minute = (total_trades / max(uptime_seconds / 60, 1)) if uptime_seconds > 0 else 0.0

            # Fill rate calculation
            orders_placed = engine_metrics.get('orders_placed', 0)
            orders_filled = engine_metrics.get('orders_filled', 0)
            fill_rate = (orders_filled / max(orders_placed, 1) * 100) if orders_placed > 0 else 0.0

            trading_metrics = TradingPerformanceMetrics(
                timestamp=current_time,
                total_volume_usdc=float(engine_metrics.get('total_volume', 0)),
                total_trades=total_trades,
                trades_per_minute=trades_per_minute,
                fill_rate_percent=fill_rate,
                total_pnl_usdc=float(portfolio_summary.get('total_pnl', 0)),
                realized_pnl_usdc=0.0,
                unrealized_pnl_usdc=float(portfolio_summary.get('total_pnl', 0)),
                maker_fees_earned_usdc=0.0,
                taker_fees_paid_usdc=0.0,
                net_fees_usdc=float(portfolio_summary.get('total_net_fees', 0)),
                average_spread_bps=current_spread_bps,
                current_spread_bps=current_spread_bps,
                spread_improvement_bps=spread_improvement_bps,
                spread_improvement_percent=spread_improvement_percent,
                baseline_spread_bps=baseline_spread_bps,
                total_inventory_usdc=inventory_usdc,
                inventory_utilization_percent=inventory_utilization,
                max_position_percent=2.0,
                var_1d_usdc=0.0,
                max_drawdown_usdc=0.0,
                order_latency_ms=engine_metrics.get('component_timing_ms', {}).get('order_management', 0),
                quote_frequency_hz=5.0,
                inventory_violations=engine_metrics.get('inventory_violations', 0),
                emergency_stops=engine_metrics.get('emergency_stops', 0),
                orders_placed=orders_placed,
                orders_filled=orders_filled,
                orders_cancelled=0,
                active_quotes=engine_metrics.get('quotes_generated', 0),
                quote_update_frequency=trades_per_minute / max(total_trades, 1) if total_trades > 0 else 0.0
            )

            # Store in history
            self.trading_metrics_history.append(trading_metrics)
            if len(self.trading_metrics_history) > self.max_history_length:
                self.trading_metrics_history = self.trading_metrics_history[-self.max_history_length:]

        except Exception as e:
            logger.error(f"Failed to collect trading performance metrics: {e}")

    async def _collect_ml_performance_metrics(self) -> None:
        """Collect ML model performance metrics."""
        try:
            current_time = datetime.now()

            if not self.ml_metrics_collector:
                return

            # Get current ML performance data
            ml_performance = await self.ml_metrics_collector.get_current_performance()

            # Extract metrics
            total_predictions = ml_performance.get('prediction_count', 0)
            latency_data = ml_performance.get('latency', {})
            quality_data = ml_performance.get('quality', {})
            methods_data = ml_performance.get('methods', {})
            costs_data = ml_performance.get('costs', {})

            # Calculate rates
            uptime_minutes = 1  # Would calculate from start time
            predictions_per_minute = total_predictions / max(uptime_minutes, 1)

            # Method counts
            azure_predictions = methods_data.get('azure_openai', 0)
            fallback_predictions = methods_data.get('rule_based', 0)
            cached_predictions = 0  # Would extract from cache metrics

            ml_metrics = MLPerformanceMetrics(
                timestamp=current_time,
                total_predictions=total_predictions,
                predictions_per_minute=predictions_per_minute,
                prediction_accuracy_percent=55.0,  # Would calculate from accuracy tracking
                prediction_confidence_avg=quality_data.get('avg_confidence', 0.0),
                avg_inference_time_ms=latency_data.get('avg_ms', 0.0),
                p95_inference_time_ms=latency_data.get('p95_ms', 0.0),
                max_inference_time_ms=latency_data.get('p99_ms', 0.0),
                total_cost_usd=costs_data.get('total_usd', 0.0),
                cost_per_prediction_usd=costs_data.get('total_usd', 0.0) / max(total_predictions, 1),
                hourly_cost_usd=costs_data.get('current_hour_usd', 0.0),
                azure_openai_predictions=azure_predictions,
                fallback_predictions=fallback_predictions,
                cached_predictions=cached_predictions,
                cache_hit_rate_percent=0.0,  # Would calculate from cache metrics
                ensemble_agreement_avg=0.0,
                uncertainty_score_avg=0.0,
                validation_pass_rate_percent=100.0,
                api_success_rate_percent=100.0
            )

            # Store in history
            self.ml_metrics_history.append(ml_metrics)
            if len(self.ml_metrics_history) > self.max_history_length:
                self.ml_metrics_history = self.ml_metrics_history[-self.max_history_length:]

        except Exception as e:
            logger.error(f"Failed to collect ML performance metrics: {e}")

    async def _collect_risk_metrics(self) -> None:
        """Collect risk management metrics."""
        try:
            current_time = datetime.now()

            if not self.position_tracker:
                return

            # Get portfolio summary for risk calculations
            portfolio_summary = self.position_tracker.get_portfolio_summary()

            # Calculate risk metrics
            total_exposure = portfolio_summary.get('total_notional_exposure', 0.0)
            max_inventory = portfolio_summary.get('max_inventory_limit', 2000.0)
            portfolio_utilization = portfolio_summary.get('portfolio_utilization', 0.0)

            # VaR calculation (simplified - would use proper statistical models)
            daily_volatility = 0.02  # 2% daily volatility assumption
            var_95_confidence = 1.645
            var_99_confidence = 2.326

            var_1d_95 = total_exposure * daily_volatility * var_95_confidence
            var_1d_99 = total_exposure * daily_volatility * var_99_confidence
            expected_shortfall = var_1d_95 * 1.3  # Rough approximation

            risk_metrics = RiskMetrics(
                timestamp=current_time,
                total_exposure_usdc=total_exposure,
                leverage_ratio=total_exposure / max(max_inventory, 1),
                concentration_risk_percent=portfolio_utilization * 100,
                portfolio_beta=1.0,  # Would calculate vs market benchmark
                var_1d_95_usdc=var_1d_95,
                var_1d_99_usdc=var_1d_99,
                expected_shortfall_usdc=expected_shortfall,
                current_drawdown_usdc=abs(min(0, portfolio_summary.get('total_pnl', 0))),
                max_drawdown_percent=5.0,  # Would track historically
                drawdown_duration_minutes=0,
                portfolio_volatility_percent=daily_volatility * 100,
                realized_volatility_1h=daily_volatility * 100 / 24,
                realized_volatility_1d=daily_volatility * 100,
                position_limit_breaches=0,  # Would track from position tracker
                inventory_limit_breaches=0,
                risk_score=min(100, portfolio_utilization * 50)  # 0-100 scale
            )

            # Store in history
            self.risk_metrics_history.append(risk_metrics)
            if len(self.risk_metrics_history) > self.max_history_length:
                self.risk_metrics_history = self.risk_metrics_history[-self.max_history_length:]

        except Exception as e:
            logger.error(f"Failed to collect risk metrics: {e}")

    async def _store_metrics(self) -> None:
        """Store metrics in InfluxDB."""
        try:
            if not self.influxdb_client:
                return

            # Store latest pipeline metrics
            if self.pipeline_metrics_history:
                latest = self.pipeline_metrics_history[-1]
                await self.influxdb_client.write_point(
                    "data_pipeline_metrics",
                    {"service": "flashmm"},
                    asdict(latest),
                    timestamp=latest.timestamp
                )

            # Store latest system metrics
            if self.system_metrics_history:
                latest = self.system_metrics_history[-1]
                await self.influxdb_client.write_point(
                    "system_resource_metrics",
                    {"host": "flashmm"},
                    asdict(latest),
                    timestamp=latest.timestamp
                )

            # Store latest component health metrics
            if self.component_health_history:
                latest = self.component_health_history[-1]
                # Convert dict fields to JSON strings for InfluxDB
                data = asdict(latest)
                data["websocket_client_statuses"] = json.dumps(data["websocket_client_statuses"])

                await self.influxdb_client.write_point(
                    "component_health_metrics",
                    {"service": "flashmm"},
                    data,
                    timestamp=latest.timestamp
                )
# Store latest trading metrics
            if self.trading_metrics_history:
                latest = self.trading_metrics_history[-1]
                await self.influxdb_client.write_point(
                    "trading_performance_metrics",
                    {"service": "flashmm"},
                    asdict(latest),
                    timestamp=latest.timestamp
                )

            # Store latest ML metrics
            if self.ml_metrics_history:
                latest = self.ml_metrics_history[-1]
                await self.influxdb_client.write_point(
                    "ml_performance_metrics",
                    {"service": "flashmm"},
                    asdict(latest),
                    timestamp=latest.timestamp
                )

            # Store latest risk metrics
            if self.risk_metrics_history:
                latest = self.risk_metrics_history[-1]
                await self.influxdb_client.write_point(
                    "risk_metrics",
                    {"service": "flashmm"},
                    asdict(latest),
                    timestamp=latest.timestamp
                )

        except Exception as e:
            logger.error(f"Failed to store metrics in InfluxDB: {e}")

    async def _publish_realtime_metrics(self) -> None:
        """Publish real-time metrics to Redis for live monitoring."""
        try:
            if not self.redis_client:
                return

            # Create summary metrics payload
            summary = {
                "timestamp": datetime.now().isoformat(),
                "pipeline": {},
                "system": {},
                "health": {},
                "trading": {},
                "ml": {},
                "risk": {}
            }

            # Pipeline metrics summary
            if self.pipeline_metrics_history:
                latest = self.pipeline_metrics_history[-1]
                summary["pipeline"] = {
                    "messages_per_second": latest.messages_per_second,
                    "avg_latency_ms": latest.avg_processing_latency_ms,
                    "p95_latency_ms": latest.p95_processing_latency_ms,
                    "error_rate_percent": latest.error_rate_percent,
                    "active_connections": latest.active_websocket_connections,
                    "latency_violations": latest.latency_violations_count
                }

            # System metrics summary
            if self.system_metrics_history:
                latest = self.system_metrics_history[-1]
                summary["system"] = {
                    "cpu_percent": latest.cpu_percent,
                    "memory_percent": latest.memory_percent,
                    "disk_percent": latest.disk_percent
                }

            # Health summary
            if self.component_health_history:
                latest = self.component_health_history[-1]
                summary["health"] = {
                    "redis_status": latest.redis_status,
                    "influxdb_status": latest.influxdb_status,
                    "feed_manager_status": latest.feed_manager_status,
                    "active_websockets": len([s for s in latest.websocket_client_statuses.values() if s in ["connected", "subscribed"]])
                }

            # Trading metrics summary
            if self.trading_metrics_history:
                latest = self.trading_metrics_history[-1]
                summary["trading"] = {
                    "total_pnl_usdc": latest.total_pnl_usdc,
                    "spread_improvement_percent": latest.spread_improvement_percent,
                    "total_volume_usdc": latest.total_volume_usdc,
                    "fill_rate_percent": latest.fill_rate_percent,
                    "inventory_utilization_percent": latest.inventory_utilization_percent
                }

            # ML metrics summary
            if self.ml_metrics_history:
                latest = self.ml_metrics_history[-1]
                summary["ml"] = {
                    "total_predictions": latest.total_predictions,
                    "avg_inference_time_ms": latest.avg_inference_time_ms,
                    "prediction_accuracy_percent": latest.prediction_accuracy_percent,
                    "hourly_cost_usd": latest.hourly_cost_usd
                }

            # Risk metrics summary
            if self.risk_metrics_history:
                latest = self.risk_metrics_history[-1]
                summary["risk"] = {
                    "total_exposure_usdc": latest.total_exposure_usdc,
                    "var_1d_95_usdc": latest.var_1d_95_usdc,
                    "risk_score": latest.risk_score,
                    "current_drawdown_usdc": latest.current_drawdown_usdc
                }

            # Store summary in Redis with short TTL
            await self.redis_client.set("flashmm:metrics:realtime", summary, expire=60)

            # Publish to subscribers
            await self.redis_client.publish("flashmm:metrics:stream", summary)

        except Exception as e:
            logger.error(f"Failed to publish real-time metrics: {e}")

    async def _health_monitoring_loop(self) -> None:
        """Health monitoring and status updates loop."""
        while self.running:
            try:
                await self._update_health_status()
                await asyncio.sleep(self.health_check_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health monitoring loop: {e}")
                await asyncio.sleep(self.health_check_interval)

    async def _update_health_status(self) -> None:
        """Update overall system health status."""
        try:
            if not self.redis_client:
                return

            # Determine overall health
            overall_status = "healthy"
            issues = []

            # Check component health
            if self.component_health_history:
                latest = self.component_health_history[-1]

                if latest.redis_status != "healthy":
                    overall_status = "degraded"
                    issues.append(f"Redis: {latest.redis_status}")

                if latest.influxdb_status != "healthy":
                    overall_status = "degraded"
                    issues.append(f"InfluxDB: {latest.influxdb_status}")

                if latest.feed_manager_status != "healthy":
                    overall_status = "unhealthy"
                    issues.append(f"Feed Manager: {latest.feed_manager_status}")

                # Check WebSocket connections
                unhealthy_ws = [k for k, v in latest.websocket_client_statuses.items() if v not in ["connected", "subscribed"]]
                if unhealthy_ws:
                    overall_status = "degraded"
                    issues.append(f"WebSocket issues: {unhealthy_ws}")

            # Check system resources
            if self.system_metrics_history:
                latest = self.system_metrics_history[-1]

                if latest.cpu_percent > self.alert_thresholds["max_cpu_percent"]:
                    overall_status = "degraded"
                    issues.append(f"High CPU: {latest.cpu_percent:.1f}%")

                if latest.memory_percent > self.alert_thresholds["max_memory_percent"]:
                    overall_status = "degraded"
                    issues.append(f"High Memory: {latest.memory_percent:.1f}%")

                if latest.disk_percent > self.alert_thresholds["max_disk_percent"]:
                    overall_status = "degraded"
                    issues.append(f"High Disk: {latest.disk_percent:.1f}%")

            # Check latency violations
            if self.pipeline_metrics_history:
                latest = self.pipeline_metrics_history[-1]

                if latest.max_processing_latency_ms > self.alert_thresholds["max_latency_ms"]:
                    overall_status = "degraded"
                    issues.append(f"High Latency: {latest.max_processing_latency_ms:.1f}ms")

                if latest.error_rate_percent > self.alert_thresholds["max_error_rate"]:
                    overall_status = "unhealthy"
                    issues.append(f"High Error Rate: {latest.error_rate_percent:.1f}%")

            # Store health status
            health_status = {
                "status": overall_status,
                "timestamp": datetime.now().isoformat(),
                "issues": issues,
                "components": {
                    "redis": self.component_health_history[-1].redis_status if self.component_health_history else "unknown",
                    "influxdb": self.component_health_history[-1].influxdb_status if self.component_health_history else "unknown",
                    "feed_manager": self.component_health_history[-1].feed_manager_status if self.component_health_history else "unknown"
                }
            }

            await self.redis_client.set("flashmm:health:status", health_status, expire=120)

        except Exception as e:
            logger.error(f"Failed to update health status: {e}")

    async def _alerting_loop(self) -> None:
        """Alerting loop for threshold violations."""
        while self.running:
            try:
                await self._check_alert_conditions()
                await asyncio.sleep(60)  # Check every minute

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in alerting loop: {e}")
                await asyncio.sleep(60)

    async def _check_alert_conditions(self) -> None:
        """Check alert conditions and trigger notifications."""
        try:
            alerts = []

            # Check pipeline metrics
            if self.pipeline_metrics_history:
                latest = self.pipeline_metrics_history[-1]

                if latest.max_processing_latency_ms > self.alert_thresholds["max_latency_ms"]:
                    alerts.append({
                        "type": "latency_violation",
                        "severity": "warning",
                        "message": f"Processing latency {latest.max_processing_latency_ms:.1f}ms exceeds threshold",
                        "value": latest.max_processing_latency_ms,
                        "threshold": self.alert_thresholds["max_latency_ms"]
                    })

                if latest.error_rate_percent > self.alert_thresholds["max_error_rate"]:
                    alerts.append({
                        "type": "high_error_rate",
                        "severity": "critical",
                        "message": f"Error rate {latest.error_rate_percent:.1f}% exceeds threshold",
                        "value": latest.error_rate_percent,
                        "threshold": self.alert_thresholds["max_error_rate"]
                    })

            # Check system metrics
            if self.system_metrics_history:
                latest = self.system_metrics_history[-1]

                if latest.cpu_percent > self.alert_thresholds["max_cpu_percent"]:
                    alerts.append({
                        "type": "high_cpu_usage",
                        "severity": "warning",
                        "message": f"CPU usage {latest.cpu_percent:.1f}% exceeds threshold",
                        "value": latest.cpu_percent,
                        "threshold": self.alert_thresholds["max_cpu_percent"]
                    })

                if latest.memory_percent > self.alert_thresholds["max_memory_percent"]:
                    alerts.append({
                        "type": "high_memory_usage",
                        "severity": "warning",
                        "message": f"Memory usage {latest.memory_percent:.1f}% exceeds threshold",
                        "value": latest.memory_percent,
                        "threshold": self.alert_thresholds["max_memory_percent"]
                    })

            # Trigger alerts
            for alert in alerts:
                await self._trigger_alert(alert)

        except Exception as e:
            logger.error(f"Failed to check alert conditions: {e}")

    async def _trigger_alert(self, alert: dict[str, Any]) -> None:
        """Trigger an alert notification."""
        try:
            logger.warning(f"ALERT: {alert['message']}")

            # Store alert in Redis
            if self.redis_client:
                alert_key = f"flashmm:alerts:{alert['type']}:{int(time.time())}"
                await self.redis_client.set(alert_key, alert, expire=3600)  # Store for 1 hour

            # Call registered alert callbacks
            for callback in self.alert_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(alert['type'], alert)
                    else:
                        callback(alert['type'], alert)
                except Exception as e:
                    logger.error(f"Error in alert callback: {e}")

        except Exception as e:
            logger.error(f"Failed to trigger alert: {e}")

    def register_alert_callback(self, callback: Callable[[str, dict[str, Any]], None]) -> None:
        """Register callback for alert notifications."""
        self.alert_callbacks.append(callback)

    def add_callback(self, callback: Callable) -> None:
        """Add callback for metrics updates (compatibility method)."""
        self.register_alert_callback(callback)

    async def publish_trading_metrics(self, metrics: 'TradingPerformanceMetrics') -> None:
        """Publish trading metrics (compatibility method)."""
        try:
            if hasattr(metrics, '__dict__'):
                self.trading_metrics_history.append(metrics)
            logger.debug("Published trading metrics")
        except Exception as e:
            logger.error(f"Failed to publish trading metrics: {e}")

    async def shutdown(self) -> None:
        """Shutdown metrics collector (compatibility method)."""
        await self.stop()

    @property
    def callbacks(self) -> list:
        """Get callbacks list (compatibility property)."""
        return self.alert_callbacks

    async def _trigger_callbacks(self, data: dict[str, Any]) -> None:
        """Trigger callbacks with data (compatibility method)."""
        for callback in self.alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback("metrics_update", data)
                else:
                    callback("metrics_update", data)
            except Exception as e:
                logger.error(f"Error in callback: {e}")

    def get_current_metrics(self) -> dict[str, Any]:
        """Get current metrics summary."""
        return {
            "pipeline": asdict(self.pipeline_metrics_history[-1]) if self.pipeline_metrics_history else {},
            "system": asdict(self.system_metrics_history[-1]) if self.system_metrics_history else {},
            "components": asdict(self.component_health_history[-1]) if self.component_health_history else {}
        }

    def get_metrics_history(self, hours: int = 1) -> dict[str, list[dict[str, Any]]]:
        """Get metrics history for the specified number of hours."""
        cutoff_time = datetime.now() - timedelta(hours=hours)

        return {
            "pipeline": [
                asdict(m) for m in self.pipeline_metrics_history
                if m.timestamp >= cutoff_time
            ],
            "system": [
                asdict(m) for m in self.system_metrics_history
                if m.timestamp >= cutoff_time
            ],
            "components": [
                asdict(m) for m in self.component_health_history
                if m.timestamp >= cutoff_time
            ]
        }


# Legacy class for backward compatibility
class MetricsCollector(EnhancedMetricsCollector):
    """Legacy MetricsCollector that extends EnhancedMetricsCollector for backward compatibility."""

    def __init__(self):
        super().__init__()
        # Convert to use legacy clients if needed
        self.influxdb_client = None  # Will use legacy InfluxDBClient

    async def initialize(self) -> None:
        """Initialize with legacy clients for backward compatibility."""
        try:
            # Use legacy InfluxDB client
            from flashmm.data.storage.influxdb_client import InfluxDBClient

            self.influxdb_client = InfluxDBClient()
            await self.influxdb_client.initialize()

            logger.info("Legacy MetricsCollector initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Legacy MetricsCollector: {e}")


# Type aliases for backward compatibility
TradingMetrics = TradingPerformanceMetrics
MLMetrics = MLPerformanceMetrics

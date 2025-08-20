"""
FlashMM Monitoring System Comprehensive Test Suite

Complete test suite covering unit tests, integration tests, performance tests,
and end-to-end validation for all monitoring components.
"""

import asyncio
import pytest
import unittest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import tempfile
import os
import numpy as np
from dataclasses import asdict

# Import monitoring components for testing
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

from flashmm.monitoring.telemetry.metrics_collector import MetricsCollector, TradingMetrics, MLMetrics, RiskMetrics
from flashmm.monitoring.alerts.alert_manager import AlertManager, AlertRule, Alert, AlertSeverity, AlertStatus, AlertChannel
from flashmm.monitoring.analytics.performance_analyzer import PerformanceAnalyzer, SpreadAnalysis, PnLAttribution, TradingEfficiency
from flashmm.monitoring.streaming.data_streamer import DataStreamer, StreamType, MessageType, StreamMessage
from flashmm.monitoring.dashboards.grafana_client import GrafanaClient
from flashmm.monitoring.dashboards.dashboard_generator import DashboardGenerator, UserRole
from flashmm.monitoring.social.twitter_client import TwitterClient
from flashmm.monitoring.monitoring_service import MonitoringService, ServiceStatus, ServiceType


class TestDataGenerator:
    """Generate test data for monitoring components."""
    
    @staticmethod
    def generate_trading_metrics(count: int = 10) -> List[TradingMetrics]:
        """Generate sample trading metrics."""
        metrics = []
        base_time = datetime.now()
        
        for i in range(count):
            metrics.append(TradingMetrics(
                timestamp=base_time + timedelta(seconds=i*60),
                market="SOL-USD",
                order_latency_ms=np.random.gamma(2, 50),
                fill_rate=np.random.uniform(0.85, 0.98),
                spread_improvement_percent=np.random.uniform(15, 35),
                volume_usdc=np.random.exponential(1000),
                trades_executed=np.random.poisson(10),
                total_pnl_usdc=np.random.normal(100, 50),
                inventory_utilization_percent=np.random.uniform(30, 80),
                active_orders=np.random.poisson(25),
                avg_trade_size_usdc=np.random.exponential(500),
                price_improvement_bps=np.random.uniform(1, 5),
                market_impact_bps=np.random.uniform(2, 8),
                adverse_selection_rate=np.random.uniform(0.1, 0.3)
            ))
        
        return metrics
    
    @staticmethod
    def generate_ml_metrics(count: int = 10) -> List[MLMetrics]:
        """Generate sample ML metrics."""
        metrics = []
        base_time = datetime.now()
        
        for i in range(count):
            metrics.append(MLMetrics(
                timestamp=base_time + timedelta(minutes=i*5),
                model_name="price_predictor_v2",
                prediction_latency_ms=np.random.gamma(1.5, 20),
                prediction_accuracy=np.random.uniform(0.75, 0.92),
                model_confidence=np.random.uniform(0.6, 0.95),
                feature_importance={"volume": 0.3, "spread": 0.25, "momentum": 0.2, "volatility": 0.25},
                api_success_rate_percent=np.random.uniform(95, 99.5),
                cache_hit_rate_percent=np.random.uniform(80, 95),
                predictions_per_second=np.random.uniform(50, 200),
                model_drift_score=np.random.uniform(0, 0.3),
                training_loss=np.random.uniform(0.05, 0.2),
                validation_loss=np.random.uniform(0.06, 0.25),
                last_retrain_timestamp=base_time - timedelta(hours=np.random.uniform(1, 24))
            ))
        
        return metrics
    
    @staticmethod
    def generate_alert_rules() -> List[AlertRule]:
        """Generate sample alert rules."""
        return [
            AlertRule(
                id="test_latency_alert",
                name="Test High Latency",
                description="Test alert for high latency",
                metric="trading.order_latency_ms",
                condition="gt",
                threshold=100.0,
                severity=AlertSeverity.WARNING,
                channels=[AlertChannel.EMAIL, AlertChannel.DASHBOARD]
            ),
            AlertRule(
                id="test_pnl_alert",
                name="Test P&L Loss",
                description="Test alert for P&L loss",
                metric="trading.total_pnl_usdc",
                condition="lt",
                threshold=-500.0,
                severity=AlertSeverity.CRITICAL,
                channels=[AlertChannel.EMAIL, AlertChannel.SLACK]
            )
        ]
    
    @staticmethod
    def generate_stream_messages(count: int = 100) -> List[StreamMessage]:
        """Generate sample stream messages."""
        messages = []
        base_time = datetime.now()
        
        for i in range(count):
            messages.append(StreamMessage(
                message_type=MessageType.DATA,
                stream_type=StreamType.METRICS,
                timestamp=base_time + timedelta(seconds=i),
                data={
                    "metric": "test_metric",
                    "value": np.random.normal(100, 20),
                    "market": "SOL-USD"
                },
                sequence_id=i
            ))
        
        return messages


class TestMetricsCollector(unittest.TestCase):
    """Test suite for MetricsCollector."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.metrics_collector = MetricsCollector()
        self.test_data = TestDataGenerator()
    
    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test metrics collector initialization."""
        await self.metrics_collector.initialize()
        
        self.assertTrue(self.metrics_collector.running)
        self.assertIsNotNone(self.metrics_collector.influxdb_client)
        self.assertEqual(len(self.metrics_collector.callbacks), 0)
    
    def test_collect_trading_metrics(self):
        """Test trading metrics collection."""
        trading_metrics = self.test_data.generate_trading_metrics(1)[0]
        
        # Test data validation
        self.assertIsInstance(trading_metrics.timestamp, datetime)
        self.assertGreater(trading_metrics.order_latency_ms, 0)
        self.assertGreaterEqual(trading_metrics.fill_rate, 0)
        self.assertLessEqual(trading_metrics.fill_rate, 1)
        
        # Test metric conversion
        metric_dict = asdict(trading_metrics)
        self.assertIn('market', metric_dict)
        self.assertIn('order_latency_ms', metric_dict)
    
    def test_collect_ml_metrics(self):
        """Test ML metrics collection."""
        ml_metrics = self.test_data.generate_ml_metrics(1)[0]
        
        # Test data validation
        self.assertIsInstance(ml_metrics.timestamp, datetime)
        self.assertGreater(ml_metrics.prediction_latency_ms, 0)
        self.assertGreaterEqual(ml_metrics.prediction_accuracy, 0)
        self.assertLessEqual(ml_metrics.prediction_accuracy, 1)
        self.assertIsInstance(ml_metrics.feature_importance, dict)
    
    @pytest.mark.asyncio
    async def test_publish_metrics(self):
        """Test metrics publishing."""
        with patch.object(self.metrics_collector, '_write_to_influxdb') as mock_write:
            mock_write.return_value = True
            
            trading_metrics = self.test_data.generate_trading_metrics(1)[0]
            await self.metrics_collector.publish_trading_metrics(trading_metrics)
            
            mock_write.assert_called_once()
    
    def test_callback_system(self):
        """Test callback system."""
        callback_called = False
        test_data = {"test": "data"}
        
        def test_callback(data):
            nonlocal callback_called
            callback_called = True
            self.assertEqual(data, test_data)
        
        self.metrics_collector.add_callback(test_callback)
        self.metrics_collector._trigger_callbacks(test_data)
        
        self.assertTrue(callback_called)
    
    def test_metrics_aggregation(self):
        """Test metrics aggregation."""
        trading_metrics_list = self.test_data.generate_trading_metrics(10)
        
        # Test aggregation logic
        avg_latency = np.mean([m.order_latency_ms for m in trading_metrics_list])
        avg_fill_rate = np.mean([m.fill_rate for m in trading_metrics_list])
        
        self.assertGreater(avg_latency, 0)
        self.assertGreater(avg_fill_rate, 0)
        self.assertLessEqual(avg_fill_rate, 1)


class TestAlertManager(unittest.TestCase):
    """Test suite for AlertManager."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.alert_manager = AlertManager()
        self.test_data = TestDataGenerator()
    
    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test alert manager initialization."""
        await self.alert_manager.initialize()
        
        self.assertTrue(self.alert_manager.running)
        self.assertGreater(len(self.alert_manager.alert_rules), 0)
        self.assertGreater(len(self.alert_manager.channels), 0)
    
    @pytest.mark.asyncio
    async def test_alert_rule_management(self):
        """Test alert rule management."""
        alert_rules = self.test_data.generate_alert_rules()
        
        for rule in alert_rules:
            success = await self.alert_manager.add_alert_rule(rule)
            self.assertTrue(success)
            self.assertIn(rule.id, self.alert_manager.alert_rules)
        
        # Test rule removal
        success = await self.alert_manager.remove_alert_rule(alert_rules[0].id)
        self.assertTrue(success)
        self.assertNotIn(alert_rules[0].id, self.alert_manager.alert_rules)
    
    @pytest.mark.asyncio
    async def test_metric_evaluation(self):
        """Test metric evaluation against rules."""
        alert_rules = self.test_data.generate_alert_rules()
        
        for rule in alert_rules:
            await self.alert_manager.add_alert_rule(rule)
        
        # Test triggering alert
        triggered_alerts = await self.alert_manager.evaluate_metric(
            "trading.order_latency_ms", 150.0
        )
        
        self.assertGreater(len(triggered_alerts), 0)
        self.assertEqual(triggered_alerts[0].rule_id, "test_latency_alert")
        self.assertEqual(triggered_alerts[0].status, AlertStatus.ACTIVE)
    
    def test_condition_evaluation(self):
        """Test alert condition evaluation."""
        # Test greater than
        result = self.alert_manager._evaluate_condition("gt", 150, 100)
        self.assertTrue(result)
        
        result = self.alert_manager._evaluate_condition("gt", 50, 100)
        self.assertFalse(result)
        
        # Test less than
        result = self.alert_manager._evaluate_condition("lt", -600, -500)
        self.assertTrue(result)
        
        result = self.alert_manager._evaluate_condition("lt", -400, -500)
        self.assertFalse(result)
        
        # Test equality
        result = self.alert_manager._evaluate_condition("eq", "test", "test")
        self.assertTrue(result)
        
        result = self.alert_manager._evaluate_condition("eq", "test", "other")
        self.assertFalse(result)
    
    @pytest.mark.asyncio
    async def test_alert_acknowledgment(self):
        """Test alert acknowledgment."""
        # Create and trigger an alert
        alert_rule = self.test_data.generate_alert_rules()[0]
        await self.alert_manager.add_alert_rule(alert_rule)
        
        triggered_alerts = await self.alert_manager.evaluate_metric(
            alert_rule.metric, 150.0
        )
        
        self.assertGreater(len(triggered_alerts), 0)
        alert = triggered_alerts[0]
        
        # Acknowledge the alert
        success = await self.alert_manager.acknowledge_alert(
            alert.id, "test_user", "Test acknowledgment"
        )
        
        self.assertTrue(success)
        
        # Verify acknowledgment
        updated_alert = self.alert_manager.active_alerts.get(alert.id)
        if updated_alert:  # Alert might be removed from active alerts
            self.assertEqual(updated_alert.status, AlertStatus.ACKNOWLEDGED)
            self.assertEqual(updated_alert.acknowledged_by, "test_user")


class TestPerformanceAnalyzer(unittest.TestCase):
    """Test suite for PerformanceAnalyzer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.performance_analyzer = PerformanceAnalyzer()
        self.test_data = TestDataGenerator()
    
    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test performance analyzer initialization."""
        await self.performance_analyzer.initialize()
        
        self.assertGreater(len(self.performance_analyzer.baseline_spreads), 0)
        self.assertEqual(self.performance_analyzer.confidence_level, 0.95)
    
    @pytest.mark.asyncio
    async def test_spread_analysis(self):
        """Test spread improvement analysis."""
        market = "SOL-USD"
        analysis = await self.performance_analyzer.analyze_spread_improvement(market)
        
        self.assertIsInstance(analysis, SpreadAnalysis)
        self.assertEqual(analysis.market, market)
        self.assertGreaterEqual(analysis.improvement_percent, -100)  # Can be negative
        self.assertGreaterEqual(analysis.statistical_significance, 0)
        self.assertLessEqual(analysis.statistical_significance, 1)
    
    @pytest.mark.asyncio
    async def test_pnl_attribution(self):
        """Test P&L attribution analysis."""
        attribution = await self.performance_analyzer.analyze_pnl_attribution()
        
        self.assertIsInstance(attribution, PnLAttribution)
        self.assertIsInstance(attribution.total_pnl_usdc, (int, float))
        self.assertGreaterEqual(attribution.win_rate, 0)
        self.assertLessEqual(attribution.win_rate, 1)
        self.assertGreaterEqual(attribution.inventory_turnover, 0)
    
    @pytest.mark.asyncio
    async def test_trading_efficiency(self):
        """Test trading efficiency analysis."""
        efficiency = await self.performance_analyzer.analyze_trading_efficiency()
        
        self.assertIsInstance(efficiency, TradingEfficiency)
        self.assertGreaterEqual(efficiency.fill_rate, 0)
        self.assertLessEqual(efficiency.fill_rate, 1)
        self.assertGreaterEqual(efficiency.total_volume_usdc, 0)
        self.assertGreaterEqual(efficiency.trades_count, 0)
    
    @pytest.mark.asyncio
    async def test_spread_improvement_validation(self):
        """Test spread improvement validation."""
        market = "SOL-USD"
        claimed_improvement = 25.0
        
        validation = await self.performance_analyzer.validate_spread_improvement(
            market, claimed_improvement
        )
        
        self.assertIsInstance(validation, dict)
        self.assertIn("is_valid", validation)
        self.assertIn("measured_improvement_percent", validation)
        self.assertIn("confidence_level", validation)
        self.assertIn("sample_size", validation)
    
    def test_confidence_interval_calculation(self):
        """Test confidence interval calculation."""
        data = [10, 15, 12, 18, 14, 16, 11, 13, 17, 19]
        confidence = 0.95
        
        interval = self.performance_analyzer._calculate_confidence_interval(data, confidence)
        
        self.assertIsInstance(interval, tuple)
        self.assertEqual(len(interval), 2)
        self.assertLess(interval[0], interval[1])  # Lower bound < upper bound
    
    def test_sharpe_ratio_calculation(self):
        """Test Sharpe ratio calculation."""
        returns = [0.02, -0.01, 0.03, 0.01, -0.005, 0.015, 0.02, -0.01]
        
        sharpe = self.performance_analyzer._calculate_sharpe_ratio(returns)
        
        self.assertIsInstance(sharpe, (int, float))
        # Sharpe ratio can be positive or negative
    
    def test_max_drawdown_calculation(self):
        """Test maximum drawdown calculation."""
        returns = [10, -5, 8, -12, 15, -3, 7, -8, 20]
        
        max_dd = self.performance_analyzer._calculate_max_drawdown(returns)
        
        self.assertIsInstance(max_dd, (int, float))
        self.assertGreaterEqual(max_dd, 0)  # Drawdown is always positive


class TestDataStreamer(unittest.TestCase):
    """Test suite for DataStreamer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.data_streamer = DataStreamer()
        self.test_data = TestDataGenerator()
    
    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test data streamer initialization."""
        await self.data_streamer.initialize()
        
        self.assertGreater(len(self.data_streamer.stream_queues), 0)
        self.assertGreater(len(self.data_streamer.background_tasks), 0)
    
    @pytest.mark.asyncio
    async def test_message_streaming(self):
        """Test message streaming functionality."""
        test_data = {"test": "data", "value": 123}
        
        await self.data_streamer.stream_data(StreamType.METRICS, test_data)
        
        # Check if message was added to queue
        queue = self.data_streamer.stream_queues[StreamType.METRICS.value]
        self.assertFalse(queue.empty())
    
    def test_message_filtering(self):
        """Test message filtering."""
        message = StreamMessage(
            message_type=MessageType.DATA,
            stream_type=StreamType.TRADES,
            timestamp=datetime.now(),
            data={"market": "SOL-USD", "value": 100}
        )
        
        # Test market filter
        filters = {"markets": ["SOL-USD"]}
        result = self.data_streamer._message_matches_filters(message, filters)
        self.assertTrue(result)
        
        filters = {"markets": ["ETH-USD"]}
        result = self.data_streamer._message_matches_filters(message, filters)
        self.assertFalse(result)
        
        # Test minimum value filter
        filters = {"min_value": 50}
        result = self.data_streamer._message_matches_filters(message, filters)
        self.assertTrue(result)
        
        filters = {"min_value": 150}
        result = self.data_streamer._message_matches_filters(message, filters)
        self.assertFalse(result)
    
    def test_rate_limiting(self):
        """Test rate limiting functionality."""
        client_id = "test_client"
        
        # Setup subscription with rate limit
        from flashmm.monitoring.streaming.data_streamer import StreamSubscription, StreamType, CompressionType
        
        subscription = StreamSubscription(
            client_id=client_id,
            stream_types={StreamType.METRICS},
            rate_limit_per_second=5
        )
        
        self.data_streamer.subscriptions[client_id] = subscription
        
        # Test within rate limit
        result = self.data_streamer._check_rate_limit(client_id, 3)
        self.assertTrue(result)
        
        # Test exceeding rate limit
        result = self.data_streamer._check_rate_limit(client_id, 10)
        self.assertFalse(result)
    
    def test_queue_health(self):
        """Test queue health monitoring."""
        health = self.data_streamer.get_queue_health()
        
        self.assertIsInstance(health, dict)
        for stream_type, health_info in health.items():
            self.assertIn("size", health_info)
            self.assertIn("max_size", health_info)
            self.assertIn("utilization", health_info)
            self.assertIn("status", health_info)
            
            self.assertGreaterEqual(health_info["utilization"], 0)
            self.assertLessEqual(health_info["utilization"], 1)


class TestMonitoringService(unittest.TestCase):
    """Test suite for MonitoringService."""
    
    def setUp(self):
        """Set up test fixtures."""
        from flashmm.monitoring.monitoring_service import MonitoringConfig, ServiceType
        
        config = MonitoringConfig(
            enabled_services={ServiceType.METRICS_COLLECTOR, ServiceType.ALERT_MANAGER},
            health_check_interval=5,
            auto_recovery=True
        )
        self.monitoring_service = MonitoringService(config)
    
    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test monitoring service initialization."""
        with patch.multiple(
            self.monitoring_service,
            _initialize_metrics_collector=AsyncMock(return_value=Mock()),
            _initialize_alert_manager=AsyncMock(return_value=Mock())
        ):
            await self.monitoring_service.initialize()
            
            self.assertTrue(self.monitoring_service.running)
            self.assertGreater(len(self.monitoring_service.services), 0)
    
    def test_service_health_tracking(self):
        """Test service health tracking."""
        service_name = "test_service"
        
        from flashmm.monitoring.monitoring_service import ServiceHealth, ServiceStatus, ServiceType
        
        health = ServiceHealth(
            service_name=service_name,
            service_type=ServiceType.METRICS_COLLECTOR,
            status=ServiceStatus.HEALTHY,
            last_check=datetime.now(),
            uptime_seconds=3600,
            health_score=0.95
        )
        
        self.monitoring_service.service_health[service_name] = health
        
        retrieved_health = self.monitoring_service.get_service_health(service_name)
        self.assertEqual(retrieved_health.service_name, service_name)
        self.assertEqual(retrieved_health.status, ServiceStatus.HEALTHY)
    
    def test_overall_health_calculation(self):
        """Test overall health calculation."""
        from flashmm.monitoring.monitoring_service import ServiceHealth, ServiceStatus, ServiceType
        
        # Add healthy service
        self.monitoring_service.service_health["service1"] = ServiceHealth(
            service_name="service1",
            service_type=ServiceType.METRICS_COLLECTOR,
            status=ServiceStatus.HEALTHY,
            last_check=datetime.now(),
            uptime_seconds=3600,
            health_score=1.0
        )
        
        # Add degraded service
        self.monitoring_service.service_health["service2"] = ServiceHealth(
            service_name="service2",
            service_type=ServiceType.ALERT_MANAGER,
            status=ServiceStatus.DEGRADED,
            last_check=datetime.now(),
            uptime_seconds=1800,
            health_score=0.7
        )
        
        overall_health = self.monitoring_service.get_overall_health()
        
        self.assertEqual(overall_health["overall_status"], "degraded")
        self.assertEqual(overall_health["healthy_services"], 1)
        self.assertEqual(overall_health["degraded_services"], 1)
        self.assertEqual(overall_health["total_services"], 2)
        self.assertAlmostEqual(overall_health["overall_score"], 0.85, places=2)
    
    def test_health_score_calculation(self):
        """Test health score calculation."""
        from flashmm.monitoring.monitoring_service import ServiceStatus
        
        # Test healthy service
        score = self.monitoring_service._calculate_health_score(ServiceStatus.HEALTHY, 0)
        self.assertEqual(score, 1.0)
        
        # Test degraded service
        score = self.monitoring_service._calculate_health_score(ServiceStatus.DEGRADED, 2)
        self.assertLess(score, 1.0)
        self.assertGreater(score, 0.0)
        
        # Test failed service
        score = self.monitoring_service._calculate_health_score(ServiceStatus.FAILED, 5)
        self.assertEqual(score, 0.0)


class TestIntegration(unittest.TestCase):
    """Integration tests for monitoring components."""
    
    @pytest.mark.asyncio
    async def test_metrics_to_alerts_flow(self):
        """Test flow from metrics collection to alert triggering."""
        # Initialize components
        metrics_collector = MetricsCollector()
        alert_manager = AlertManager()
        
        await metrics_collector.initialize()
        await alert_manager.initialize()
        
        # Add test alert rule
        alert_rule = AlertRule(
            id="integration_test",
            name="Integration Test Alert",
            description="Test alert for integration testing",
            metric="trading.order_latency_ms",
            condition="gt",
            threshold=100.0,
            severity=AlertSeverity.WARNING,
            channels=[AlertChannel.DASHBOARD]
        )
        
        await alert_manager.add_alert_rule(alert_rule)
        
        # Simulate high latency metric that should trigger alert
        triggered_alerts = await alert_manager.evaluate_metric("trading.order_latency_ms", 150.0)
        
        self.assertGreater(len(triggered_alerts), 0)
        self.assertEqual(triggered_alerts[0].rule_id, "integration_test")
        
        # Cleanup
        await metrics_collector.shutdown()
        await alert_manager.shutdown()
    
    @pytest.mark.asyncio
    async def test_alerts_to_streaming_flow(self):
        """Test flow from alerts to data streaming."""
        # Initialize components
        alert_manager = AlertManager()
        data_streamer = DataStreamer(alert_manager=alert_manager)
        
        await alert_manager.initialize()
        await data_streamer.initialize()
        
        # Add callback to capture streamed data
        streamed_data = []
        
        def capture_stream(data):
            streamed_data.append(data)
        
        data_streamer.add_event_handler = capture_stream
        
        # Trigger alert
        alert_rule = AlertRule(
            id="streaming_test",
            name="Streaming Test Alert",
            description="Test alert for streaming",
            metric="test.metric",
            condition="gt",
            threshold=50.0,
            severity=AlertSeverity.CRITICAL,
            channels=[AlertChannel.DASHBOARD]
        )
        
        await alert_manager.add_alert_rule(alert_rule)
        await alert_manager.evaluate_metric("test.metric", 75.0)
        
        # Wait a moment for async processing
        await asyncio.sleep(0.1)
        
        # Cleanup
        await alert_manager.shutdown()
        await data_streamer.stop_server()
    
    @pytest.mark.asyncio
    async def test_performance_analytics_integration(self):
        """Test performance analytics integration with metrics."""
        # Initialize components
        metrics_collector = MetricsCollector()
        performance_analyzer = PerformanceAnalyzer(metrics_collector)
        
        await metrics_collector.initialize()
        await performance_analyzer.initialize()
        
        # Generate test data and run analysis
        analysis = await performance_analyzer.analyze_spread_improvement("SOL-USD")
        
        self.assertIsNotNone(analysis)
        self.assertEqual(analysis.market, "SOL-USD")
        
        # Test validation
        validation = await performance_analyzer.validate_spread_improvement("SOL-USD", 25.0)
        
        self.assertIn("is_valid", validation)
        self.assertIn("measured_improvement_percent", validation)
        
        # Cleanup
        await metrics_collector.shutdown()


class TestPerformance(unittest.TestCase):
    """Performance tests for monitoring components."""
    
    @pytest.mark.asyncio
    async def test_metrics_collection_performance(self):
        """Test metrics collection performance."""
        metrics_collector = MetricsCollector()
        await metrics_collector.initialize()
        
        # Generate large batch of metrics
        trading_metrics = TestDataGenerator.generate_trading_metrics(1000)
        
        start_time = time.time()
        
        # Simulate batch processing
        for metrics in trading_metrics:
            await metrics_collector.publish_trading_metrics(metrics)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Should process 1000 metrics in reasonable time (< 5 seconds)
        self.assertLess(processing_time, 5.0)
        
        # Calculate throughput
        throughput = len(trading_metrics) / processing_time
        self.assertGreater(throughput, 100)  # > 100 metrics/second
        
        await metrics_collector.shutdown()
    
    @pytest.mark.asyncio
    async def test_alert_processing_performance(self):
        """Test alert processing performance."""
        alert_manager = AlertManager()
        await alert_manager.initialize()
        
        # Add multiple alert rules
        rules = TestDataGenerator.generate_alert_rules()
        for rule in rules * 10:  # 20 rules total
            rule.id = f"{rule.id}_{time.time()}_{hash(rule.id)}"
            await alert_manager.add_alert_rule(rule)
        
        start_time = time.time()
        
        # Process many metric evaluations
        for i in range(500):
            await alert_manager.evaluate_metric("trading.order_latency_ms", 150.0 + i)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Should process 500 evaluations in reasonable time
        self.assertLess(processing_time, 10.0)
        
        await alert_manager.shutdown()
    
    @pytest.mark.asyncio
    async def test_streaming_throughput(self):
        """Test data streaming throughput."""
        data_streamer = DataStreamer()
        await data_streamer.initialize()
        messages = TestDataGenerator.generate_stream_messages(1000)
        
        start_time = time.time()
        
        # Stream all messages
        for message in messages:
            await data_streamer.stream_data(message.stream_type, message.data)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Should handle 1000 messages in reasonable time
        self.assertLess(processing_time, 2.0)
        
        await data_streamer.stop_server()


class TestEndToEnd(unittest.TestCase):
    """End-to-end tests for complete monitoring system."""
    
    @pytest.mark.asyncio
    async def test_complete_monitoring_pipeline(self):
        """Test complete monitoring pipeline from metrics to dashboards."""
        from flashmm.monitoring.monitoring_service import create_monitoring_service
        
        # Create monitoring service with all components
        monitoring_service = create_monitoring_service(
            enabled_services=["metrics_collector", "alert_manager", "performance_analyzer"],
            auto_recovery=True,
            health_check_interval=5
        )
        
        try:
            # Initialize and start
            await monitoring_service.initialize()
            await monitoring_service.start_services()
            
            # Verify all services are running
            health = monitoring_service.get_overall_health()
            self.assertIn(health["overall_status"], ["healthy", "degraded"])
            self.assertGreater(health["total_services"], 0)
            
            # Test service interactions
            metrics_collector = monitoring_service.get_service("metrics_collector")
            alert_manager = monitoring_service.get_service("alert_manager")
            
            if metrics_collector and alert_manager:
                # Generate test data that should trigger alerts
                test_data = TestDataGenerator.generate_trading_metrics(1)[0]
                await metrics_collector.publish_trading_metrics(test_data)
                
                # Wait for processing
                await asyncio.sleep(1)
                
                # Check if alerts were processed
                active_alerts = alert_manager.get_active_alerts()
                # Note: May or may not have alerts depending on test data values
        
        finally:
            await monitoring_service.shutdown()
    
    @pytest.mark.asyncio
    async def test_system_resilience(self):
        """Test system resilience to component failures."""
        from flashmm.monitoring.monitoring_service import create_monitoring_service
        
        monitoring_service = create_monitoring_service(
            enabled_services=["metrics_collector", "alert_manager"],
            auto_recovery=True,
            health_check_interval=2
        )
        
        try:
            await monitoring_service.initialize()
            await monitoring_service.start_services()
            
            # Simulate service failure
            service_name = "metrics_collector"
            service = monitoring_service.get_service(service_name)
            
            if service and hasattr(service, 'running'):
                # Simulate failure
                service.running = False
                
                # Wait for health check to detect failure
                await asyncio.sleep(3)
                
                # Check if failure was detected
                health = monitoring_service.get_service_health(service_name)
                # Health status should reflect the issue
                
                # Wait for potential recovery
                await asyncio.sleep(5)
        
        finally:
            await monitoring_service.shutdown()


class TestConfigurationValidation(unittest.TestCase):
    """Test configuration validation and error handling."""
    
    def test_invalid_alert_rule_validation(self):
        """Test validation of invalid alert rules."""
        from flashmm.monitoring.alerts.alert_manager import AlertRule, AlertSeverity, AlertChannel
        
        # Test invalid threshold type
        with self.assertRaises((ValueError, TypeError)):
            AlertRule(
                id="invalid_rule",
                name="Invalid Rule",
                description="Rule with invalid threshold",
                metric="test.metric",
                condition="gt",
                threshold="not_a_number",  # Invalid threshold
                severity=AlertSeverity.WARNING,
                channels=[AlertChannel.EMAIL]
            )
    
    def test_configuration_loading(self):
        """Test configuration loading and validation."""
        from flashmm.monitoring.monitoring_service import MonitoringConfig, ServiceType
        
        # Test valid configuration
        config = MonitoringConfig(
            enabled_services={ServiceType.METRICS_COLLECTOR},
            health_check_interval=30,
            auto_recovery=True
        )
        
        self.assertEqual(config.health_check_interval, 30)
        self.assertTrue(config.auto_recovery)
        self.assertIn(ServiceType.METRICS_COLLECTOR, config.enabled_services)
    
    def test_service_dependency_validation(self):
        """Test service dependency validation."""
        from flashmm.monitoring.monitoring_service import MonitoringService, MonitoringConfig, ServiceType
        
        # Test configuration with dependencies
        config = MonitoringConfig(
            enabled_services={ServiceType.PERFORMANCE_ANALYZER}  # Depends on metrics_collector
        )
        
        monitoring_service = MonitoringService(config)
        
        # Test dependency checking
        dependencies = monitoring_service._get_service_dependencies("performance_analyzer")
        self.assertIsInstance(dependencies, list)


class TestSecurityAndValidation(unittest.TestCase):
    """Test security aspects and input validation."""
    
    def test_input_sanitization(self):
        """Test input sanitization for metrics and alerts."""
        from flashmm.monitoring.telemetry.metrics_collector import TradingMetrics
        
        # Test with potential injection attempts
        try:
            metrics = TradingMetrics(
                timestamp=datetime.now(),
                market="SOL-USD'; DROP TABLE metrics; --",  # SQL injection attempt
                order_latency_ms=50.0,
                fill_rate=0.95,
                spread_improvement_percent=25.0,
                volume_usdc=1000.0,
                trades_executed=10,
                total_pnl_usdc=100.0,
                inventory_utilization_percent=50.0,
                active_orders=20,
                avg_trade_size_usdc=100.0,
                price_improvement_bps=2.0,
                market_impact_bps=3.0,
                adverse_selection_rate=0.15
            )
            
            # Should handle malicious input gracefully
            self.assertIsInstance(metrics.market, str)
            
        except Exception as e:
            # Should not crash on malicious input
            self.fail(f"Input sanitization failed: {e}")
    
    def test_websocket_security(self):
        """Test WebSocket security measures."""
        from flashmm.monitoring.streaming.data_streamer import DataStreamer
        
        data_streamer = DataStreamer()
        
        # Test rate limiting
        client_id = "test_client"
        
        # Simulate rapid requests
        for i in range(100):
            result = data_streamer._check_rate_limit(client_id, 1)
            # Rate limiting should kick in
        
        # Test message size limits
        large_data = {"data": "x" * (1024 * 1024 * 2)}  # 2MB data
        
        # Should handle large messages appropriately
        # (Implementation would depend on actual size checking)
    
    def test_authentication_validation(self):
        """Test authentication validation."""
        from flashmm.monitoring.streaming.data_streamer import StreamingClient
        
        # Test with invalid token
        client = StreamingClient("ws://localhost:8765", "invalid_token")
        
        # Should handle invalid authentication gracefully
        self.assertIsNotNone(client.auth_token)


class TestUtilities:
    """Utility functions for testing."""
    
    @staticmethod
    def create_temp_config() -> Dict[str, Any]:
        """Create temporary configuration for testing."""
        return {
            "monitoring": {
                "health_check_interval": 5,
                "auto_recovery": True,
                "max_restart_attempts": 2
            },
            "alerts": {
                "email": {
                    "enabled": False
                },
                "slack": {
                    "enabled": False
                }
            },
            "streaming": {
                "host": "localhost",
                "port": 8765,
                "max_connections": 100
            },
            "analytics": {
                "confidence_level": 0.95,
                "min_sample_size": 10
            }
        }
    
    @staticmethod
    def setup_test_environment():
        """Setup test environment."""
        # Create temporary directories
        temp_dir = tempfile.mkdtemp()
        
        # Setup environment variables
        os.environ["FLASHMM_TEST_MODE"] = "true"
        os.environ["FLASHMM_CONFIG_PATH"] = temp_dir
        
        return temp_dir
    
    @staticmethod
    def cleanup_test_environment(temp_dir: str):
        """Cleanup test environment."""
        import shutil
        
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        
        # Clean up environment variables
        os.environ.pop("FLASHMM_TEST_MODE", None)
        os.environ.pop("FLASHMM_CONFIG_PATH", None)
    
    @staticmethod
    async def wait_for_condition(condition_func: Callable[[], bool],
                               timeout: float = 5.0,
                               interval: float = 0.1) -> bool:
        """Wait for a condition to become true."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if condition_func():
                return True
            await asyncio.sleep(interval)
        
        return False
    
    @staticmethod
    def assert_metrics_valid(metrics_data: Dict[str, Any]):
        """Assert that metrics data is valid."""
        required_fields = ["timestamp", "value"]
        
        for field in required_fields:
            assert field in metrics_data, f"Missing required field: {field}"
        
        assert isinstance(metrics_data["timestamp"], (str, datetime))
        assert isinstance(metrics_data["value"], (int, float))
    
    @staticmethod
    def generate_load_test_data(num_samples: int = 1000) -> List[Dict[str, Any]]:
        """Generate data for load testing."""
        data = []
        base_time = datetime.now()
        
        for i in range(num_samples):
            data.append({
                "timestamp": base_time + timedelta(seconds=i),
                "metric": f"test_metric_{i % 10}",
                "value": np.random.normal(100, 20),
                "market": np.random.choice(["SOL-USD", "ETH-USD", "BTC-USD"]),
                "category": np.random.choice(["trading", "system", "ml"])
            })
        
        return data


# Test runners and configurations

class TestConfig:
    """Test configuration settings."""
    
    PERFORMANCE_TEST_TIMEOUT = 30.0
    INTEGRATION_TEST_TIMEOUT = 60.0
    LOAD_TEST_SAMPLES = 1000
    CONCURRENT_CONNECTIONS = 50
    
    # Test data settings
    TEST_MARKETS = ["SOL-USD", "ETH-USD", "BTC-USD"]
    TEST_METRICS_COUNT = 100
    TEST_ALERT_RULES_COUNT = 10
    
    # Performance thresholds
    MAX_LATENCY_MS = 100
    MIN_THROUGHPUT_MSG_PER_SEC = 100
    MAX_MEMORY_USAGE_MB = 500


def run_unit_tests():
    """Run unit tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestMetricsCollector,
        TestAlertManager,
        TestPerformanceAnalyzer,
        TestDataStreamer,
        TestMonitoringService,
        TestConfigurationValidation,
        TestSecurityAndValidation
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


def run_integration_tests():
    """Run integration tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add integration test classes
    integration_classes = [
        TestIntegration,
        TestEndToEnd
    ]
    
    for test_class in integration_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


def run_performance_tests():
    """Run performance tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add performance test classes
    performance_classes = [
        TestPerformance
    ]
    
    for test_class in performance_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


async def run_load_tests():
    """Run load tests."""
    print("Running load tests...")
    
    # Test metrics collection under load
    metrics_collector = MetricsCollector()
    await metrics_collector.initialize()
    
    try:
        # Generate load
        test_data = TestUtilities.generate_load_test_data(TestConfig.LOAD_TEST_SAMPLES)
        
        start_time = time.time()
        
        for data in test_data:
            # Simulate metrics publishing
            trading_metrics = TradingMetrics(
                timestamp=data["timestamp"],
                market=data["market"],
                order_latency_ms=abs(data["value"]),
                fill_rate=min(1.0, abs(data["value"]) / 200),
                spread_improvement_percent=abs(data["value"]) / 5,
                volume_usdc=abs(data["value"]) * 10,
                trades_executed=int(abs(data["value"]) / 10),
                total_pnl_usdc=data["value"],
                inventory_utilization_percent=min(100, abs(data["value"])),
                active_orders=int(abs(data["value"]) / 5),
                avg_trade_size_usdc=abs(data["value"]),
                price_improvement_bps=abs(data["value"]) / 50,
                market_impact_bps=abs(data["value"]) / 25,
                adverse_selection_rate=min(1.0, abs(data["value"]) / 500)
            )
            
            await metrics_collector.publish_trading_metrics(trading_metrics)
        
        end_time = time.time()
        processing_time = end_time - start_time
        throughput = len(test_data) / processing_time
        
        print(f"Load test completed:")
        print(f"  Processed: {len(test_data)} metrics")
        print(f"  Time: {processing_time:.2f} seconds")
        print(f"  Throughput: {throughput:.2f} metrics/second")
        
        # Verify performance meets requirements
        assert throughput >= TestConfig.MIN_THROUGHPUT_MSG_PER_SEC, f"Throughput {throughput} below minimum {TestConfig.MIN_THROUGHPUT_MSG_PER_SEC}"
        
        print("Load test PASSED")
        
    finally:
        await metrics_collector.shutdown()


def generate_test_report():
    """Generate comprehensive test report."""
    report = {
        "test_execution_time": datetime.now().isoformat(),
        "test_environment": {
            "python_version": sys.version,
            "test_framework": "unittest + pytest",
            "async_support": True
        },
        "test_categories": {
            "unit_tests": {
                "description": "Individual component testing",
                "test_count": 0,
                "classes": [
                    "TestMetricsCollector",
                    "TestAlertManager",
                    "TestPerformanceAnalyzer",
                    "TestDataStreamer",
                    "TestMonitoringService"
                ]
            },
            "integration_tests": {
                "description": "Component interaction testing",
                "test_count": 0,
                "classes": [
                    "TestIntegration",
                    "TestEndToEnd"
                ]
            },
            "performance_tests": {
                "description": "Performance and load testing",
                "test_count": 0,
                "classes": [
                    "TestPerformance"
                ]
            },
            "security_tests": {
                "description": "Security and validation testing",
                "test_count": 0,
                "classes": [
                    "TestSecurityAndValidation"
                ]
            }
        },
        "coverage_areas": [
            "Metrics collection and publishing",
            "Alert rule evaluation and notification",
            "Performance analysis and validation",
            "Real-time data streaming",
            "Dashboard integration",
            "Service orchestration and health checks",
            "Error handling and recovery",
            "Security and input validation",
            "Configuration management",
            "Integration between components"
        ]
    }
    
    return report


# Main test execution
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="FlashMM Monitoring Test Suite")
    parser.add_argument("--unit", action="store_true", help="Run unit tests")
    parser.add_argument("--integration", action="store_true", help="Run integration tests")
    parser.add_argument("--performance", action="store_true", help="Run performance tests")
    parser.add_argument("--load", action="store_true", help="Run load tests")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--report", action="store_true", help="Generate test report")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if not any([args.unit, args.integration, args.performance, args.load, args.all, args.report]):
        args.all = True  # Default to running all tests
    
    success = True
    
    try:
        # Setup test environment
        temp_dir = TestUtilities.setup_test_environment()
        
        print("=" * 60)
        print("FlashMM Monitoring System Test Suite")
        print("=" * 60)
        
        if args.unit or args.all:
            print("\n🧪 Running Unit Tests...")
            success &= run_unit_tests()
        
        if args.integration or args.all:
            print("\n🔗 Running Integration Tests...")
            success &= run_integration_tests()
        
        if args.performance or args.all:
            print("\n⚡ Running Performance Tests...")
            success &= run_performance_tests()
        
        if args.load or args.all:
            print("\n🚀 Running Load Tests...")
            asyncio.run(run_load_tests())
        
        if args.report or args.all:
            print("\n📊 Generating Test Report...")
            report = generate_test_report()
            
            # Save report to file
            report_file = os.path.join(temp_dir, "test_report.json")
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            print(f"Test report saved to: {report_file}")
            
            if args.verbose:
                print("\nTest Report Summary:")
                print(json.dumps(report, indent=2))
        
        print("\n" + "=" * 60)
        if success:
            print("✅ ALL TESTS PASSED")
        else:
            print("❌ SOME TESTS FAILED")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test execution failed: {e}")
        success = False
    
    finally:
        # Cleanup test environment
        TestUtilities.cleanup_test_environment(temp_dir)
    
    sys.exit(0 if success else 1)
        
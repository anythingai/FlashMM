"""
FlashMM Monitoring System Integration Testing and Validation

Comprehensive integration tests that validate the entire monitoring system
working together as a cohesive unit, including end-to-end workflows,
data flow validation, and system performance under realistic conditions.
"""

import asyncio
import json
import os

# Import all monitoring components
import sys
import time
from datetime import datetime
from typing import Any

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

from flashmm.monitoring.alerts.alert_manager import AlertChannel, AlertRule, AlertSeverity
from flashmm.monitoring.analytics.performance_analyzer import ReportType
from flashmm.monitoring.dashboards.dashboard_generator import UserRole
from flashmm.monitoring.monitoring_service import create_monitoring_service
from flashmm.monitoring.streaming.data_streamer import StreamType
from flashmm.monitoring.telemetry.metrics_collector import MLMetrics, TradingMetrics


class IntegrationTestScenario:
    """Integration test scenario configuration."""

    def __init__(self, name: str, description: str, duration_seconds: int = 60):
        self.name = name
        self.description = description
        self.duration_seconds = duration_seconds
        self.start_time: datetime | None = None
        self.end_time: datetime | None = None
        self.results = {}
        self.success = False

    def start(self):
        """Start the test scenario."""
        self.start_time = datetime.now()
        print(f"\n🚀 Starting scenario: {self.name}")
        print(f"   Description: {self.description}")
        print(f"   Duration: {self.duration_seconds}s")

    def finish(self, success: bool = True):
        """Finish the test scenario."""
        self.end_time = datetime.now()
        self.success = success
        duration = (self.end_time - self.start_time).total_seconds() if self.end_time and self.start_time else 0

        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"   {status} in {duration:.2f}s")

    def add_result(self, key: str, value: Any):
        """Add a result to the scenario."""
        self.results[key] = value


class MonitoringSystemValidator:
    """Validates the complete monitoring system functionality."""

    def __init__(self):
        self.monitoring_service = None
        self.test_data = []
        self.validation_results = {}
        self.websocket_clients = []

    async def setup_system(self) -> bool:
        """Setup the complete monitoring system."""
        try:
            print("\n🔧 Setting up FlashMM Monitoring System...")

            # Create monitoring service with all components
            self.monitoring_service = create_monitoring_service(
                enabled_services=[
                    "metrics_collector",
                    "alert_manager",
                    "performance_analyzer",
                    "data_streamer",
                    "grafana_client",
                    "dashboard_generator",
                    "twitter_client"
                ],
                auto_recovery=True,
                health_check_interval=10
            )

            # Initialize and start services
            await self.monitoring_service.initialize()
            await self.monitoring_service.start_services()

            # Wait for services to stabilize
            await asyncio.sleep(5)

            # Verify system health
            health = self.monitoring_service.get_overall_health() if self.monitoring_service else {'overall_status': 'unknown', 'healthy_services': 0, 'total_services': 0}
            print(f"   System Health: {health['overall_status']}")
            print(f"   Services: {health['healthy_services']}/{health['total_services']} healthy")

            if health['overall_status'] in ['healthy', 'degraded']:
                print("✅ Monitoring system setup complete")
                return True
            else:
                print("❌ Monitoring system setup failed")
                return False

        except Exception as e:
            print(f"❌ Setup failed: {e}")
            return False

    async def teardown_system(self):
        """Teardown the monitoring system."""
        try:
            print("\n🔄 Tearing down monitoring system...")

            # Close WebSocket clients
            for client in self.websocket_clients:
                if hasattr(client, 'disconnect'):
                    await client.disconnect()

            # Shutdown monitoring service
            if self.monitoring_service:
                await self.monitoring_service.shutdown()

            print("✅ System teardown complete")

        except Exception as e:
            print(f"⚠️ Teardown error: {e}")

    async def run_end_to_end_test(self) -> IntegrationTestScenario:
        """Run complete end-to-end test."""
        scenario = IntegrationTestScenario(
            "End-to-End Data Flow",
            "Test complete data flow from metrics ingestion to dashboard display",
            duration_seconds=120
        )

        scenario.start()

        try:
            # Step 1: Generate and publish metrics
            print("   📊 Step 1: Publishing metrics...")
            metrics_published = await self._publish_test_metrics()
            scenario.add_result("metrics_published", metrics_published)

            # Wait for processing
            await asyncio.sleep(5)

            # Step 2: Verify alert processing
            print("   🚨 Step 2: Verifying alert processing...")
            alerts_triggered = await self._verify_alert_processing()
            scenario.add_result("alerts_triggered", alerts_triggered)

            # Step 3: Test performance analytics
            print("   📈 Step 3: Running performance analytics...")
            analytics_results = await self._run_performance_analytics()
            scenario.add_result("analytics_completed", analytics_results is not None)

            # Step 4: Test real-time streaming
            print("   🌊 Step 4: Testing real-time streaming...")
            streaming_test = await self._test_real_time_streaming()
            scenario.add_result("streaming_functional", streaming_test)

            # Step 5: Generate dashboards
            print("   📋 Step 5: Generating dashboards...")
            dashboards_created = await self._generate_test_dashboards()
            scenario.add_result("dashboards_created", dashboards_created)

            # Step 6: Test social media integration
            print("   🐦 Step 6: Testing social media integration...")
            social_test = await self._test_social_integration()
            scenario.add_result("social_integration", social_test)

            # Validate overall system health
            final_health = self.monitoring_service.get_overall_health() if self.monitoring_service else {'overall_status': 'unknown'}
            scenario.add_result("final_system_health", final_health['overall_status'])

            # Determine success
            success = (
                metrics_published > 0 and
                alerts_triggered >= 0 and  # Alerts may or may not trigger
                analytics_results is not None and
                streaming_test and
                dashboards_created > 0 and
                final_health['overall_status'] in ['healthy', 'degraded']
            )

            scenario.finish(success)
            return scenario

        except Exception as e:
            print(f"   ❌ End-to-end test failed: {e}")
            scenario.finish(False)
            return scenario

    async def _publish_test_metrics(self) -> int:
        """Publish test metrics to the system."""
        try:
            metrics_collector = self.monitoring_service.get_service("metrics_collector") if self.monitoring_service else None
            if not metrics_collector:
                return 0

            published_count = 0

            # Generate realistic trading metrics
            for _ in range(50):
                trading_metrics = TradingMetrics(
                    timestamp=datetime.now(),
                    total_volume_usdc=np.random.exponential(1000),
                    total_trades=np.random.poisson(10),
                    trades_per_minute=np.random.uniform(5, 20),
                    fill_rate_percent=np.random.uniform(85, 98),
                    total_pnl_usdc=np.random.normal(100, 50),
                    realized_pnl_usdc=np.random.normal(50, 25),
                    unrealized_pnl_usdc=np.random.normal(50, 25),
                    maker_fees_earned_usdc=np.random.uniform(10, 50),
                    taker_fees_paid_usdc=np.random.uniform(5, 25),
                    net_fees_usdc=np.random.uniform(5, 25),
                    average_spread_bps=np.random.uniform(8, 15),
                    current_spread_bps=np.random.uniform(6, 12),
                    spread_improvement_bps=np.random.uniform(1, 5),
                    spread_improvement_percent=np.random.uniform(15, 35),
                    baseline_spread_bps=np.random.uniform(10, 18),
                    total_inventory_usdc=np.random.uniform(500, 2000),
                    inventory_utilization_percent=np.random.uniform(30, 80),
                    max_position_percent=np.random.uniform(15, 25),
                    var_1d_usdc=np.random.uniform(50, 200),
                    max_drawdown_usdc=np.random.uniform(20, 100),
                    order_latency_ms=np.random.gamma(2, 50),
                    quote_frequency_hz=np.random.uniform(3, 8),
                    inventory_violations=np.random.poisson(2),
                    emergency_stops=np.random.poisson(0.5),
                    orders_placed=np.random.poisson(50),
                    orders_filled=np.random.poisson(25),
                    orders_cancelled=np.random.poisson(10),
                    active_quotes=np.random.poisson(25),
                    quote_update_frequency=np.random.uniform(2, 6)
                )

                await metrics_collector.publish_trading_metrics(trading_metrics)
                published_count += 1

                # Add some delay to simulate realistic timing
                await asyncio.sleep(0.1)

            # Generate ML metrics
            for _i in range(20):
                ml_metrics = MLMetrics(
                    timestamp=datetime.now(),
                    total_predictions=np.random.poisson(100),
                    predictions_per_minute=np.random.uniform(50, 200),
                    prediction_accuracy_percent=np.random.uniform(75, 92),
                    prediction_confidence_avg=np.random.uniform(0.6, 0.95),
                    avg_inference_time_ms=np.random.gamma(1.5, 20),
                    p95_inference_time_ms=np.random.gamma(2, 30),
                    max_inference_time_ms=np.random.gamma(3, 40),
                    total_cost_usd=np.random.uniform(1, 20),
                    cost_per_prediction_usd=np.random.uniform(0.01, 0.2),
                    hourly_cost_usd=np.random.uniform(1, 10),
                    azure_openai_predictions=np.random.poisson(80),
                    fallback_predictions=np.random.poisson(20),
                    cached_predictions=np.random.poisson(40),
                    cache_hit_rate_percent=np.random.uniform(80, 95),
                    ensemble_agreement_avg=np.random.uniform(0.7, 0.95),
                    uncertainty_score_avg=np.random.uniform(0.1, 0.4),
                    validation_pass_rate_percent=np.random.uniform(95, 99.5),
                    api_success_rate_percent=np.random.uniform(95, 99.5)
                )

                await metrics_collector.publish_ml_metrics(ml_metrics)
                published_count += 1

                await asyncio.sleep(0.05)

            return published_count

        except Exception as e:
            print(f"      Error publishing metrics: {e}")
            return 0

    async def _verify_alert_processing(self) -> int:
        """Verify alert processing functionality."""
        try:
            alert_manager = self.monitoring_service.get_service("alert_manager") if self.monitoring_service else None
            if not alert_manager:
                return 0

            # Add test alert rule
            test_rule = AlertRule(
                id="integration_test_alert",
                name="Integration Test Alert",
                description="Test alert for integration testing",
                metric="trading.order_latency_ms",
                condition="gt",
                threshold=200.0,  # High threshold to potentially trigger
                severity=AlertSeverity.WARNING,
                channels=[AlertChannel.DASHBOARD, AlertChannel.EMAIL]
            )

            await alert_manager.add_alert_rule(test_rule)

            # Trigger test conditions
            await alert_manager.evaluate_metric("trading.order_latency_ms", 250.0)
            await alert_manager.evaluate_metric("trading.order_latency_ms", 150.0)
            await alert_manager.evaluate_metric("trading.order_latency_ms", 300.0)

            # Wait for processing
            await asyncio.sleep(2)

            # Check active alerts
            active_alerts = alert_manager.get_active_alerts()
            return len(active_alerts)

        except Exception as e:
            print(f"      Error in alert processing: {e}")
            return -1

    async def _run_performance_analytics(self) -> dict[str, Any] | None:
        """Run performance analytics."""
        try:
            performance_analyzer = self.monitoring_service.get_service("performance_analyzer") if self.monitoring_service else None
            if not performance_analyzer:
                return None

            # Generate performance report
            report = await performance_analyzer.generate_performance_report(
                ReportType.REAL_TIME,
                markets=["SOL-USD", "ETH-USD"]
            )

            if report:
                return {
                    "report_id": report.report_id,
                    "spread_analyses": len(report.spread_analysis),
                    "summary": report.summary,
                    "data_quality_score": report.data_quality_score
                }

            return None

        except Exception as e:
            print(f"      Error in performance analytics: {e}")
            return None

    async def _test_real_time_streaming(self) -> bool:
        """Test real-time streaming functionality."""
        try:
            data_streamer = self.monitoring_service.get_service("data_streamer") if self.monitoring_service else None
            if not data_streamer:
                return False

            # Test streaming metrics
            test_data = {
                "metric": "test_integration_metric",
                "value": 123.45,
                "timestamp": datetime.now().isoformat(),
                "market": "SOL-USD"
            }

            await data_streamer.stream_metrics(test_data)
            await data_streamer.stream_pnl({"pnl": 100.0, "currency": "USDC"})
            await data_streamer.stream_spreads({"spread": 5.2, "market": "ETH-USD"})

            # Test queue health
            queue_health = data_streamer.get_queue_health()
            healthy_queues = sum(1 for q in queue_health.values() if q["status"] == "healthy")

            return healthy_queues > 0

        except Exception as e:
            print(f"      Error in streaming test: {e}")
            return False

    async def _generate_test_dashboards(self) -> int:
        """Generate test dashboards."""
        try:
            dashboard_generator = self.monitoring_service.get_service("dashboard_generator") if self.monitoring_service else None
            grafana_client = self.monitoring_service.get_service("grafana_client") if self.monitoring_service else None

            if not dashboard_generator or not grafana_client:
                return 0

            dashboards_created = 0

            # Test different user roles
            for role in [UserRole.ADMIN, UserRole.TRADER, UserRole.PUBLIC]:
                try:
                    dashboard = await dashboard_generator.generate_user_dashboard(
                        role,
                        f"integration_test_{role.value}"
                    )

                    if dashboard:
                        dashboards_created += 1

                except Exception as e:
                    print(f"      Dashboard creation failed for {role.value}: {e}")

            return dashboards_created

        except Exception as e:
            print(f"      Error generating dashboards: {e}")
            return 0

    async def _test_social_integration(self) -> bool:
        """Test social media integration."""
        try:
            twitter_client = self.monitoring_service.get_service("twitter_client") if self.monitoring_service else None
            if not twitter_client:
                return False

            # Test performance summary generation (without actually posting)
            test_metrics = {
                "total_pnl_usdc": 150.75,
                "spread_improvement_percent": 28.3,
                "trades_executed": 47,
                "fill_rate": 0.96
            }

            # This would normally post to Twitter, but we'll just test the formatting
            summary_text = await twitter_client._format_performance_summary(test_metrics)

            return len(summary_text) > 0 and len(summary_text) <= 280  # Twitter limit

        except Exception as e:
            print(f"      Error in social integration test: {e}")
            return False

    async def run_load_test(self) -> IntegrationTestScenario:
        """Run load test on the integrated system."""
        scenario = IntegrationTestScenario(
            "Load Test",
            "Test system performance under load",
            duration_seconds=300  # 5 minutes
        )

        scenario.start()

        try:
            print("   📈 Generating load...")

            # Metrics load
            metrics_task = asyncio.create_task(self._generate_metrics_load())

            # Alert evaluation load
            alerts_task = asyncio.create_task(self._generate_alerts_load())

            # Streaming load
            streaming_task = asyncio.create_task(self._generate_streaming_load())

            # Monitor system during load
            monitoring_task = asyncio.create_task(self._monitor_system_under_load(scenario))

            # Wait for all tasks to complete
            await asyncio.gather(
                metrics_task,
                alerts_task,
                streaming_task,
                monitoring_task,
                return_exceptions=True
            )

            # Final health check
            final_health = self.monitoring_service.get_overall_health() if self.monitoring_service else {'overall_status': 'unknown'}
            scenario.add_result("final_health", final_health)

            success = final_health['overall_status'] in ['healthy', 'degraded']
            scenario.finish(success)

            return scenario

        except Exception as e:
            print(f"   ❌ Load test failed: {e}")
            scenario.finish(False)
            return scenario

    async def _generate_metrics_load(self):
        """Generate metrics load."""
        metrics_collector = self.monitoring_service.get_service("metrics_collector") if self.monitoring_service else None
        if not metrics_collector:
            return

        for i in range(1000):  # Generate 1000 metrics
            trading_metrics = TradingMetrics(
                timestamp=datetime.now(),
                total_volume_usdc=np.random.exponential(1000),
                total_trades=np.random.poisson(10),
                trades_per_minute=np.random.uniform(5, 20),
                fill_rate_percent=np.random.uniform(85, 98),
                total_pnl_usdc=np.random.normal(100, 50),
                realized_pnl_usdc=np.random.normal(50, 25),
                unrealized_pnl_usdc=np.random.normal(50, 25),
                maker_fees_earned_usdc=np.random.uniform(10, 50),
                taker_fees_paid_usdc=np.random.uniform(5, 25),
                net_fees_usdc=np.random.uniform(5, 25),
                average_spread_bps=np.random.uniform(8, 15),
                current_spread_bps=np.random.uniform(6, 12),
                spread_improvement_bps=np.random.uniform(1, 5),
                spread_improvement_percent=np.random.uniform(15, 35),
                baseline_spread_bps=np.random.uniform(10, 18),
                total_inventory_usdc=np.random.uniform(500, 2000),
                inventory_utilization_percent=np.random.uniform(30, 80),
                max_position_percent=np.random.uniform(15, 25),
                var_1d_usdc=np.random.uniform(50, 200),
                max_drawdown_usdc=np.random.uniform(20, 100),
                order_latency_ms=np.random.gamma(2, 50),
                quote_frequency_hz=np.random.uniform(3, 8),
                inventory_violations=np.random.poisson(2),
                emergency_stops=np.random.poisson(0.5),
                orders_placed=np.random.poisson(50),
                orders_filled=np.random.poisson(25),
                orders_cancelled=np.random.poisson(10),
                active_quotes=np.random.poisson(25),
                quote_update_frequency=np.random.uniform(2, 6)
            )

            await metrics_collector.publish_trading_metrics(trading_metrics)

            if i % 100 == 0:
                await asyncio.sleep(0.1)  # Brief pause every 100 metrics

    async def _generate_alerts_load(self):
        """Generate alerts evaluation load."""
        alert_manager = self.monitoring_service.get_service("alert_manager") if self.monitoring_service else None
        if not alert_manager:
            return

        metrics = [
            "trading.order_latency_ms",
            "trading.total_pnl_usdc",
            "trading.fill_rate",
            "system.cpu_percent",
            "ml.prediction_accuracy"
        ]

        for i in range(500):  # 500 evaluations
            metric = np.random.choice(metrics)
            value = np.random.normal(100, 30)

            await alert_manager.evaluate_metric(metric, value)

            if i % 50 == 0:
                await asyncio.sleep(0.05)

    async def _generate_streaming_load(self):
        """Generate streaming load."""
        data_streamer = self.monitoring_service.get_service("data_streamer") if self.monitoring_service else None
        if not data_streamer:
            return

        for i in range(2000):  # 2000 stream messages
            stream_type_options = [StreamType.METRICS, StreamType.TRADES, StreamType.PNL]
            stream_type = stream_type_options[np.random.randint(0, len(stream_type_options))]

            data = {
                "message_id": i,
                "value": np.random.normal(100, 20),
                "timestamp": datetime.now().isoformat(),
                "market": np.random.choice(["SOL-USD", "ETH-USD", "BTC-USD"])
            }

            await data_streamer.stream_data(stream_type, data)

            if i % 200 == 0:
                await asyncio.sleep(0.01)

    async def _monitor_system_under_load(self, scenario: IntegrationTestScenario):
        """Monitor system performance under load."""
        monitoring_duration = 60  # Monitor for 60 seconds
        check_interval = 5  # Check every 5 seconds

        health_samples = []

        for _ in range(monitoring_duration // check_interval):
            health = self.monitoring_service.get_overall_health() if self.monitoring_service else {'overall_status': 'unknown', 'overall_score': 0, 'healthy_services': 0, 'total_services': 0}
            health_samples.append({
                "timestamp": datetime.now().isoformat(),
                "status": health['overall_status'],
                "score": health['overall_score'],
                "healthy_services": health['healthy_services'],
                "total_services": health['total_services']
            })

            await asyncio.sleep(check_interval)

        # Calculate health metrics
        avg_score = np.mean([h['score'] for h in health_samples])
        healthy_percentage = sum(1 for h in health_samples if h['status'] in ['healthy', 'degraded']) / len(health_samples)

        scenario.add_result("load_test_health_samples", health_samples)
        scenario.add_result("average_health_score", avg_score)
        scenario.add_result("healthy_percentage", healthy_percentage)

    async def run_failure_recovery_test(self) -> IntegrationTestScenario:
        """Test system failure recovery capabilities."""
        scenario = IntegrationTestScenario(
            "Failure Recovery Test",
            "Test system recovery from component failures",
            duration_seconds=180
        )

        scenario.start()

        try:
            print("   🔧 Testing failure recovery...")

            # Record initial health
            initial_health = self.monitoring_service.get_overall_health() if self.monitoring_service else {'overall_status': 'unknown', 'healthy_services': 0, 'total_services': 0}
            scenario.add_result("initial_health", initial_health)

            # Simulate service failure
            print("      Simulating service failure...")
            service_to_fail = "metrics_collector"
            service = self.monitoring_service.get_service(service_to_fail) if self.monitoring_service else None

            if service and hasattr(service, 'running'):
                # Simulate failure
                original_state = service.running
                service.running = False

                # Wait for detection
                await asyncio.sleep(15)

                # Check health after failure
                failed_health = self.monitoring_service.get_overall_health() if self.monitoring_service else {'overall_status': 'unknown', 'healthy_services': 0, 'total_services': 0}
                scenario.add_result("health_after_failure", failed_health)

                # Restore service
                print("      Restoring service...")
                service.running = original_state

                # Wait for recovery
                await asyncio.sleep(30)

                # Check recovery
                recovered_health = self.monitoring_service.get_overall_health() if self.monitoring_service else {'overall_status': 'unknown', 'healthy_services': 0, 'total_services': 0}
                scenario.add_result("health_after_recovery", recovered_health)

                # Determine success
                recovery_successful = (
                    recovered_health['overall_status'] in ['healthy', 'degraded'] and
                    recovered_health['healthy_services'] >= initial_health['healthy_services']
                )

                scenario.finish(recovery_successful)
            else:
                print("      Service failure simulation not supported")
                scenario.finish(False)

            return scenario

        except Exception as e:
            print(f"   ❌ Failure recovery test failed: {e}")
            scenario.finish(False)
            return scenario

    def generate_integration_report(self, scenarios: list[IntegrationTestScenario]) -> dict[str, Any]:
        """Generate comprehensive integration test report."""
        passed_scenarios = [s for s in scenarios if s.success]
        failed_scenarios = [s for s in scenarios if not s.success]

        total_duration = sum((s.end_time - s.start_time).total_seconds() for s in scenarios if s.end_time and s.start_time)

        # System health summary
        final_health = self.monitoring_service.get_overall_health() if self.monitoring_service else {}

        report = {
            "integration_test_report": {
                "timestamp": datetime.now().isoformat(),
                "summary": {
                    "total_scenarios": len(scenarios),
                    "passed_scenarios": len(passed_scenarios),
                    "failed_scenarios": len(failed_scenarios),
                    "success_rate": len(passed_scenarios) / len(scenarios) if scenarios else 0,
                    "total_duration_seconds": total_duration
                },
                "system_status": {
                    "final_health": final_health,
                    "services_tested": [
                        "metrics_collector",
                        "alert_manager",
                        "performance_analyzer",
                        "data_streamer",
                        "grafana_client",
                        "dashboard_generator",
                        "twitter_client"
                    ]
                },
                "scenarios": []
            }
        }

        # Add scenario details
        for scenario in scenarios:
            scenario_data = {
                "name": scenario.name,
                "description": scenario.description,
                "success": scenario.success,
                "duration_seconds": (scenario.end_time - scenario.start_time).total_seconds() if scenario.end_time and scenario.start_time else 0,
                "results": scenario.results
            }
            report["integration_test_report"]["scenarios"].append(scenario_data)

        return report


async def run_comprehensive_integration_test():
    """Run comprehensive integration test suite."""
    print("=" * 80)
    print("FlashMM Monitoring System - Comprehensive Integration Test")
    print("=" * 80)

    validator = MonitoringSystemValidator()
    scenarios = []

    try:
        # Setup system
        if not await validator.setup_system():
            print("❌ System setup failed - aborting tests")
            return False

        # Run test scenarios
        print("\n🧪 Running Integration Test Scenarios...")

        # End-to-end test
        e2e_scenario = await validator.run_end_to_end_test()
        scenarios.append(e2e_scenario)

        # Load test
        load_scenario = await validator.run_load_test()
        scenarios.append(load_scenario)

        # Failure recovery test
        recovery_scenario = await validator.run_failure_recovery_test()
        scenarios.append(recovery_scenario)

        # Generate report
        print("\n📊 Generating Integration Test Report...")
        report = validator.generate_integration_report(scenarios)

        # Save report
        report_file = f"integration_test_report_{int(time.time())}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"📄 Report saved to: {report_file}")

        # Print summary
        summary = report["integration_test_report"]["summary"]
        print("\n📋 Test Summary:")
        print(f"   Total Scenarios: {summary['total_scenarios']}")
        print(f"   Passed: {summary['passed_scenarios']}")
        print(f"   Failed: {summary['failed_scenarios']}")
        print(f"   Success Rate: {summary['success_rate']:.1%}")
        print(f"   Total Duration: {summary['total_duration_seconds']:.2f}s")

        success = summary['success_rate'] >= 0.8  # 80% success rate required

        if success:
            print("\n✅ INTEGRATION TESTS PASSED")
        else:
            print("\n❌ INTEGRATION TESTS FAILED")

        return success

    except Exception as e:
        print(f"\n❌ Integration test failed with error: {e}")
        return False

    finally:
        # Cleanup
        await validator.teardown_system()


# Validation utilities

def validate_monitoring_deployment(config_file: str = "monitoring_config.json") -> bool:
    """Validate monitoring system deployment configuration."""
    print("\n🔍 Validating Monitoring System Deployment...")

    try:
        # Check configuration file
        if os.path.exists(config_file):
            with open(config_file) as f:
                config = json.load(f)

            required_sections = [
                "monitoring", "alerts", "streaming",
                "analytics", "grafana", "social"
            ]

            missing_sections = [s for s in required_sections if s not in config]

            if missing_sections:
                print(f"❌ Missing configuration sections: {missing_sections}")
                return False

            print("✅ Configuration file valid")
        else:
            print(f"⚠️ Configuration file {config_file} not found")

        # Check required dependencies
        required_packages = [
            "asyncio", "aiohttp", "websockets", "numpy", "pandas",
            "influxdb", "grafana-api", "tweepy"
        ]

        missing_packages = []
        for package in required_packages:
            try:
                __import__(package.replace("-", "_"))
            except ImportError:
                missing_packages.append(package)

        if missing_packages:
            print(f"❌ Missing required packages: {missing_packages}")
            print("   Install with: pip install " + " ".join(missing_packages))
            return False

        print("✅ All required packages available")

        # Check environment variables
        required_env_vars = [
            "INFLUXDB_URL", "INFLUXDB_TOKEN", "GRAFANA_URL", "GRAFANA_API_KEY"
        ]

        missing_env_vars = [v for v in required_env_vars if not os.getenv(v)]

        if missing_env_vars:
            print(f"⚠️ Missing optional environment variables: {missing_env_vars}")

        print("✅ Deployment validation complete")
        return True

    except Exception as e:
        print(f"❌ Deployment validation failed: {e}")
        return False


def create_deployment_checklist() -> dict[str, Any]:
    """Create deployment checklist for monitoring system."""
    return {
        "deployment_checklist": {
            "pre_deployment": [
                {"task": "Install required Python packages", "completed": False},
                {"task": "Configure InfluxDB connection", "completed": False},
                {"task": "Setup Grafana instance and API key", "completed": False},
                {"task": "Configure alert notification channels", "completed": False},
                {"task": "Setup Twitter API credentials (optional)", "completed": False},
                {"task": "Configure environment variables", "completed": False},
                {"task": "Run unit tests", "completed": False},
                {"task": "Run integration tests", "completed": False}
            ],
            "deployment": [
                {"task": "Deploy monitoring service", "completed": False},
                {"task": "Start all monitoring components", "completed": False},
                {"task": "Verify system health", "completed": False},
                {"task": "Test alert notifications", "completed": False},
                {"task": "Validate dashboard access", "completed": False},
                {"task": "Test real-time streaming", "completed": False}
            ],
            "post_deployment": [
                {"task": "Monitor system for 24 hours", "completed": False},
                {"task": "Validate performance metrics", "completed": False},
                {"task": "Test failover scenarios", "completed": False},
                {"task": "Document operational procedures", "completed": False},
                {"task": "Train operations team", "completed": False},
                {"task": "Setup monitoring alerts for the monitoring system", "completed": False}
            ],
            "validation": [
                {"task": "Verify spread improvement tracking", "completed": False},
                {"task": "Validate P&L attribution accuracy", "completed": False},
                {"task": "Test public dashboard access", "completed": False},
                {"task": "Verify social media integration", "completed": False},
                {"task": "Validate alert escalation workflows", "completed": False}
            ]
        },
        "success_criteria": {
            "system_uptime": "> 99.5%",
            "alert_response_time": "< 60 seconds",
            "dashboard_load_time": "< 5 seconds",
            "metrics_processing_latency": "< 100ms",
            "spread_improvement_accuracy": "> 95%"
        }
    }


# Performance benchmarks for validation

PERFORMANCE_BENCHMARKS = {
    "metrics_collection": {
        "throughput_min": 1000,  # metrics per second
        "latency_max": 100,      # milliseconds
        "memory_usage_max": 512   # MB
    },
    "alert_processing": {
        "evaluation_time_max": 50,  # milliseconds
        "notification_delay_max": 30,  # seconds
        "concurrent_alerts_max": 1000
    },
    "streaming": {
        "connections_max": 1000,
        "messages_per_second_min": 10000,
        "message_latency_max": 5  # milliseconds
    },
    "analytics": {
        "report_generation_time_max": 60,  # seconds
        "spread_analysis_time_max": 10,    # seconds
        "data_quality_min": 0.95
    }
}


# Main execution
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FlashMM Monitoring Integration Tests")
    parser.add_argument("--validate-deployment", action="store_true", help="Validate deployment configuration")
    parser.add_argument("--run-integration", action="store_true", help="Run full integration tests")
    parser.add_argument("--checklist", action="store_true", help="Generate deployment checklist")
    parser.add_argument("--config", default="monitoring_config.json", help="Configuration file path")

    args = parser.parse_args()

    if args.checklist:
        checklist = create_deployment_checklist()
        print(json.dumps(checklist, indent=2))

    elif args.validate_deployment:
        success = validate_monitoring_deployment(args.config)
        exit(0 if success else 1)

    elif args.run_integration:
        success = asyncio.run(run_comprehensive_integration_test())
        exit(0 if success else 1)

    else:
        # Run all validations by default
        print("Running all validation steps...")

        deployment_valid = validate_monitoring_deployment(args.config)
        if deployment_valid:
            integration_success = asyncio.run(run_comprehensive_integration_test())
            exit(0 if integration_success else 1)
        else:
            exit(1)

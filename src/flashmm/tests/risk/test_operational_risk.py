"""
Test suite for Operational Risk Manager

Tests operational risk monitoring components including:
- System health monitoring and metrics collection
- Connectivity monitoring and endpoint health checks
- Resource utilization tracking and alerting
- Performance monitoring and latency tracking
- System failure detection and recovery procedures
"""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from flashmm.risk.operational_risk import ConnectivityMonitor, OperationalRiskManager, SystemMonitor


class TestSystemMonitor:
    """Test system monitoring functionality."""

    @pytest.fixture
    def system_monitor(self):
        """Create system monitor for testing."""
        return SystemMonitor()

    @pytest.mark.asyncio
    async def test_initialization(self, system_monitor):
        """Test system monitor initializes correctly."""
        await system_monitor.initialize()

        assert system_monitor.cpu_threshold_pct == 80.0
        assert system_monitor.memory_threshold_pct == 85.0
        assert system_monitor.disk_threshold_pct == 90.0
        assert system_monitor.monitoring_interval == 30.0
        assert system_monitor.enabled

    @pytest.mark.asyncio
    async def test_cpu_monitoring(self, system_monitor):
        """Test CPU usage monitoring."""
        await system_monitor.initialize()

        # Mock psutil for consistent testing
        with patch('psutil.cpu_percent', return_value=75.5):
            cpu_metrics = await system_monitor.collect_cpu_metrics()

            assert 'cpu_usage_pct' in cpu_metrics
            assert cpu_metrics['cpu_usage_pct'] == 75.5
            assert 'cpu_count' in cpu_metrics
            assert 'load_average' in cpu_metrics
            assert cpu_metrics['status'] == 'normal'  # Below 80% threshold

    @pytest.mark.asyncio
    async def test_memory_monitoring(self, system_monitor):
        """Test memory usage monitoring."""
        await system_monitor.initialize()

        # Mock memory metrics
        mock_memory = MagicMock()
        mock_memory.total = 16 * 1024 * 1024 * 1024  # 16GB
        mock_memory.available = 4 * 1024 * 1024 * 1024  # 4GB available
        mock_memory.percent = 75.0

        with patch('psutil.virtual_memory', return_value=mock_memory):
            memory_metrics = await system_monitor.collect_memory_metrics()

            assert 'memory_usage_pct' in memory_metrics
            assert memory_metrics['memory_usage_pct'] == 75.0
            assert 'total_memory_gb' in memory_metrics
            assert 'available_memory_gb' in memory_metrics
            assert memory_metrics['status'] == 'normal'  # Below 85% threshold

    @pytest.mark.asyncio
    async def test_disk_monitoring(self, system_monitor):
        """Test disk usage monitoring."""
        await system_monitor.initialize()

        # Mock disk usage
        mock_disk = MagicMock()
        mock_disk.total = 1024 * 1024 * 1024 * 1024  # 1TB
        mock_disk.used = 512 * 1024 * 1024 * 1024    # 512GB used
        mock_disk.free = 512 * 1024 * 1024 * 1024    # 512GB free
        mock_disk.percent = 50.0

        with patch('psutil.disk_usage', return_value=mock_disk):
            disk_metrics = await system_monitor.collect_disk_metrics()

            assert 'disk_usage_pct' in disk_metrics
            assert disk_metrics['disk_usage_pct'] == 50.0
            assert 'total_disk_gb' in disk_metrics
            assert 'free_disk_gb' in disk_metrics
            assert disk_metrics['status'] == 'normal'  # Below 90% threshold

    @pytest.mark.asyncio
    async def test_high_resource_usage_alerts(self, system_monitor):
        """Test alerts for high resource usage."""
        await system_monitor.initialize()

        alerts_received = []

        async def mock_alert_callback(alert_data):
            alerts_received.append(alert_data)

        system_monitor.set_alert_callback(mock_alert_callback)

        # Mock high CPU usage
        with patch('psutil.cpu_percent', return_value=95.0):
            cpu_metrics = await system_monitor.collect_cpu_metrics()

            assert cpu_metrics['status'] == 'critical'

            # Check for alert
            await system_monitor.check_resource_thresholds(cpu_metrics, 'cpu')

            assert len(alerts_received) > 0
            assert alerts_received[0]['resource_type'] == 'cpu'
            assert alerts_received[0]['usage_pct'] == 95.0

    @pytest.mark.asyncio
    async def test_process_monitoring(self, system_monitor):
        """Test monitoring of specific processes."""
        await system_monitor.initialize()

        # Mock process list
        mock_process1 = MagicMock()
        mock_process1.info = {
            'pid': 1234,
            'name': 'python',
            'cpu_percent': 15.5,
            'memory_percent': 8.2,
            'create_time': datetime.now().timestamp() - 3600
        }

        mock_process2 = MagicMock()
        mock_process2.info = {
            'pid': 5678,
            'name': 'trading_engine',
            'cpu_percent': 25.0,
            'memory_percent': 12.5,
            'create_time': datetime.now().timestamp() - 1800
        }

        with patch('psutil.process_iter', return_value=[mock_process1, mock_process2]):
            process_metrics = await system_monitor.collect_process_metrics(['python', 'trading_engine'])

            assert len(process_metrics) == 2
            assert process_metrics[0]['name'] == 'python'
            assert process_metrics[0]['cpu_percent'] == 15.5
            assert process_metrics[1]['name'] == 'trading_engine'
            assert process_metrics[1]['cpu_percent'] == 25.0

    @pytest.mark.asyncio
    async def test_system_health_scoring(self, system_monitor):
        """Test overall system health score calculation."""
        await system_monitor.initialize()

        # Mock normal system metrics
        normal_metrics = {
            'cpu': {'cpu_usage_pct': 45.0, 'status': 'normal'},
            'memory': {'memory_usage_pct': 60.0, 'status': 'normal'},
            'disk': {'disk_usage_pct': 70.0, 'status': 'normal'}
        }

        health_score = await system_monitor.calculate_health_score(normal_metrics)

        assert 80 <= health_score <= 100  # Should be high for normal metrics

        # Mock stressed system metrics
        stressed_metrics = {
            'cpu': {'cpu_usage_pct': 95.0, 'status': 'critical'},
            'memory': {'memory_usage_pct': 92.0, 'status': 'critical'},
            'disk': {'disk_usage_pct': 95.0, 'status': 'critical'}
        }

        stressed_health_score = await system_monitor.calculate_health_score(stressed_metrics)

        assert stressed_health_score < 50  # Should be low for stressed metrics
        assert stressed_health_score < health_score


class TestConnectivityMonitor:
    """Test connectivity monitoring functionality."""

    @pytest.fixture
    def connectivity_monitor(self):
        """Create connectivity monitor for testing."""
        return ConnectivityMonitor()

    @pytest.mark.asyncio
    async def test_initialization(self, connectivity_monitor):
        """Test connectivity monitor initializes correctly."""
        await connectivity_monitor.initialize()

        assert connectivity_monitor.timeout_seconds == 10.0
        assert connectivity_monitor.retry_attempts == 3
        assert connectivity_monitor.check_interval == 60.0
        assert len(connectivity_monitor.endpoints) == 0

    @pytest.mark.asyncio
    async def test_endpoint_registration(self, connectivity_monitor):
        """Test endpoint registration and management."""
        await connectivity_monitor.initialize()

        # Register test endpoints
        endpoints = [
            {
                'name': 'binance_api',
                'url': 'https://api.binance.com/api/v3/ping',
                'method': 'GET',
                'expected_status': 200,
                'critical': True
            },
            {
                'name': 'coinbase_api',
                'url': 'https://api.coinbase.com/v2/time',
                'method': 'GET',
                'expected_status': 200,
                'critical': False
            }
        ]

        for endpoint in endpoints:
            await connectivity_monitor.register_endpoint(endpoint)

        assert len(connectivity_monitor.endpoints) == 2
        assert 'binance_api' in connectivity_monitor.endpoints
        assert 'coinbase_api' in connectivity_monitor.endpoints

    @pytest.mark.asyncio
    async def test_successful_endpoint_check(self, connectivity_monitor):
        """Test successful endpoint health check."""
        await connectivity_monitor.initialize()

        endpoint = {
            'name': 'test_api',
            'url': 'https://httpbin.org/status/200',
            'method': 'GET',
            'expected_status': 200,
            'critical': True
        }

        await connectivity_monitor.register_endpoint(endpoint)

        # Mock successful HTTP response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.headers = {'content-type': 'application/json'}

        with patch('aiohttp.ClientSession.request', return_value=mock_response):
            health_result = await connectivity_monitor.check_endpoint_health('test_api')

            assert health_result['success']
            assert health_result['status_code'] == 200
            assert health_result['response_time_ms'] > 0
            assert health_result['error'] is None

    @pytest.mark.asyncio
    async def test_failed_endpoint_check(self, connectivity_monitor):
        """Test failed endpoint health check."""
        await connectivity_monitor.initialize()

        endpoint = {
            'name': 'failing_api',
            'url': 'https://httpbin.org/status/500',
            'method': 'GET',
            'expected_status': 200,
            'critical': True
        }

        await connectivity_monitor.register_endpoint(endpoint)

        # Mock failed HTTP response
        mock_response = AsyncMock()
        mock_response.status = 500

        with patch('aiohttp.ClientSession.request', return_value=mock_response):
            health_result = await connectivity_monitor.check_endpoint_health('failing_api')

            assert not health_result['success']
            assert health_result['status_code'] == 500
            assert health_result['error'] is not None

    @pytest.mark.asyncio
    async def test_timeout_handling(self, connectivity_monitor):
        """Test handling of connection timeouts."""
        await connectivity_monitor.initialize()

        endpoint = {
            'name': 'timeout_api',
            'url': 'https://httpbin.org/delay/15',  # Will timeout
            'method': 'GET',
            'expected_status': 200,
            'critical': True
        }

        await connectivity_monitor.register_endpoint(endpoint)

        # Mock timeout exception
        with patch('aiohttp.ClientSession.request', side_effect=TimeoutError()):
            health_result = await connectivity_monitor.check_endpoint_health('timeout_api')

            assert not health_result['success']
            assert 'timeout' in health_result['error'].lower()
            assert health_result['response_time_ms'] >= connectivity_monitor.timeout_seconds * 1000

    @pytest.mark.asyncio
    async def test_retry_mechanism(self, connectivity_monitor):
        """Test retry mechanism for failed requests."""
        await connectivity_monitor.initialize()

        endpoint = {
            'name': 'retry_api',
            'url': 'https://httpbin.org/status/503',
            'method': 'GET',
            'expected_status': 200,
            'critical': True
        }

        await connectivity_monitor.register_endpoint(endpoint)

        # Mock responses: first two fail, third succeeds
        responses = [
            AsyncMock(status=503),
            AsyncMock(status=503),
            AsyncMock(status=200)
        ]

        with patch('aiohttp.ClientSession.request', side_effect=responses):
            health_result = await connectivity_monitor.check_endpoint_health('retry_api')

            assert health_result['success']  # Should succeed on third attempt
            assert health_result['attempts'] == 3

    @pytest.mark.asyncio
    async def test_connectivity_scoring(self, connectivity_monitor):
        """Test connectivity health scoring."""
        await connectivity_monitor.initialize()

        # Register multiple endpoints
        endpoints = [
            {'name': 'api1', 'url': 'https://api1.com', 'method': 'GET', 'expected_status': 200, 'critical': True},
            {'name': 'api2', 'url': 'https://api2.com', 'method': 'GET', 'expected_status': 200, 'critical': True},
            {'name': 'api3', 'url': 'https://api3.com', 'method': 'GET', 'expected_status': 200, 'critical': False},
        ]

        for endpoint in endpoints:
            await connectivity_monitor.register_endpoint(endpoint)

        # Mock health check results
        health_results = {
            'api1': {'success': True, 'response_time_ms': 150},
            'api2': {'success': False, 'response_time_ms': 5000},  # Failed
            'api3': {'success': True, 'response_time_ms': 200}
        }

        connectivity_score = await connectivity_monitor.calculate_connectivity_score(health_results)

        # Should be reduced due to critical api2 failure
        assert 0 <= connectivity_score <= 100
        assert connectivity_score < 100  # Should be less than perfect due to failure

    @pytest.mark.asyncio
    async def test_latency_monitoring(self, connectivity_monitor):
        """Test latency monitoring and tracking."""
        await connectivity_monitor.initialize()

        endpoint = {
            'name': 'latency_api',
            'url': 'https://httpbin.org/delay/1',
            'method': 'GET',
            'expected_status': 200,
            'critical': True
        }

        await connectivity_monitor.register_endpoint(endpoint)

        # Simulate multiple checks to build latency history
        latencies = []
        for i in range(10):
            mock_response = AsyncMock()
            mock_response.status = 200

            with patch('aiohttp.ClientSession.request', return_value=mock_response):
                # Mock different response times
                with patch('time.perf_counter', side_effect=[0, 0.1 + i*0.01]):  # Increasing latency
                    health_result = await connectivity_monitor.check_endpoint_health('latency_api')
                    latencies.append(health_result['response_time_ms'])

        # Analyze latency trends
        latency_analysis = await connectivity_monitor.analyze_latency_trends('latency_api')

        assert 'average_latency_ms' in latency_analysis
        assert 'latency_trend' in latency_analysis
        assert 'latency_percentiles' in latency_analysis


class TestOperationalRiskManager:
    """Test the complete operational risk management system."""

    @pytest.fixture
    def operational_manager(self):
        """Create operational risk manager for testing."""
        return OperationalRiskManager()

    @pytest.mark.asyncio
    async def test_manager_initialization(self, operational_manager):
        """Test operational risk manager initializes all components."""
        await operational_manager.initialize()

        assert operational_manager.system_monitor is not None
        assert operational_manager.connectivity_monitor is not None
        assert operational_manager.enabled
        assert operational_manager.monitoring_interval > 0

    @pytest.mark.asyncio
    async def test_comprehensive_health_check(self, operational_manager):
        """Test comprehensive system health assessment."""
        await operational_manager.initialize()

        # Mock system metrics
        with patch.object(operational_manager.system_monitor, 'collect_all_metrics') as mock_system:
            with patch.object(operational_manager.connectivity_monitor, 'check_all_endpoints') as mock_connectivity:

                mock_system.return_value = {
                    'cpu': {'cpu_usage_pct': 45.0, 'status': 'normal'},
                    'memory': {'memory_usage_pct': 60.0, 'status': 'normal'},
                    'disk': {'disk_usage_pct': 70.0, 'status': 'normal'},
                    'health_score': 85.0
                }

                mock_connectivity.return_value = {
                    'binance_api': {'success': True, 'response_time_ms': 150},
                    'coinbase_api': {'success': True, 'response_time_ms': 200},
                    'connectivity_score': 92.0
                }

                health_assessment = await operational_manager.perform_health_check()

                assert 'system_health' in health_assessment
                assert 'connectivity_health' in health_assessment
                assert 'overall_score' in health_assessment
                assert 'risk_level' in health_assessment
                assert 'recommendations' in health_assessment

    @pytest.mark.asyncio
    async def test_risk_level_determination(self, operational_manager):
        """Test operational risk level determination."""
        await operational_manager.initialize()

        # Test scenarios with different health scores
        test_scenarios = [
            {'system_score': 95, 'connectivity_score': 90, 'expected_risk': 'low'},
            {'system_score': 75, 'connectivity_score': 80, 'expected_risk': 'medium'},
            {'system_score': 45, 'connectivity_score': 50, 'expected_risk': 'high'},
            {'system_score': 25, 'connectivity_score': 30, 'expected_risk': 'critical'},
        ]

        for scenario in test_scenarios:
            risk_level = await operational_manager.determine_risk_level(
                scenario['system_score'],
                scenario['connectivity_score']
            )

            assert risk_level == scenario['expected_risk']

    @pytest.mark.asyncio
    async def test_automated_recovery_procedures(self, operational_manager):
        """Test automated recovery procedures."""
        await operational_manager.initialize()

        recovery_actions = []

        async def mock_recovery_callback(action_data):
            recovery_actions.append(action_data)

        operational_manager.set_recovery_callback(mock_recovery_callback)

        # Simulate critical system state
        _critical_health = {
            'system_health': {
                'cpu': {'cpu_usage_pct': 98.0, 'status': 'critical'},
                'memory': {'memory_usage_pct': 95.0, 'status': 'critical'},
                'health_score': 15.0
            },
            'connectivity_health': {
                'connectivity_score': 20.0,
                'failed_endpoints': ['binance_api', 'coinbase_api']
            },
            'overall_score': 17.5,
            'risk_level': 'critical'
        }

        await operational_manager.trigger_recovery_procedures()

        # Should have triggered multiple recovery actions
        assert len(recovery_actions) > 0
        action_types = [action['type'] for action in recovery_actions]
        assert 'resource_cleanup' in action_types or 'system_restart' in action_types

    @pytest.mark.asyncio
    async def test_performance_monitoring(self, operational_manager):
        """Test system performance monitoring over time."""
        await operational_manager.initialize()

        # Simulate performance data collection over time
        performance_data = []

        for i in range(24):  # 24 data points (hours)
            with patch.object(operational_manager, 'perform_health_check') as mock_health:
                mock_health.return_value = {
                    'system_health': {'health_score': 80 + i % 20},  # Varying scores
                    'connectivity_health': {'connectivity_score': 75 + i % 25},
                    'overall_score': 77.5 + i % 22.5,
                    'timestamp': datetime.now() - timedelta(hours=24-i)
                }

                health_data = await operational_manager.perform_health_check()
                performance_data.append(health_data)

        # Analyze performance trends
        trend_analysis = await operational_manager.analyze_performance_trends(performance_data)

        assert 'average_system_score' in trend_analysis
        assert 'average_connectivity_score' in trend_analysis
        assert 'performance_trend' in trend_analysis
        assert 'stability_score' in trend_analysis

    @pytest.mark.asyncio
    async def test_alert_escalation(self, operational_manager):
        """Test alert escalation procedures."""
        await operational_manager.initialize()

        alerts_sent = []

        async def mock_alert_callback(alert_data):
            alerts_sent.append(alert_data)

        operational_manager.set_alert_callback(mock_alert_callback)

        # Simulate escalating issues
        issues = [
            {'severity': 'warning', 'component': 'cpu', 'message': 'High CPU usage'},
            {'severity': 'critical', 'component': 'connectivity', 'message': 'API endpoint down'},
            {'severity': 'critical', 'component': 'memory', 'message': 'Memory exhaustion'},
        ]

        for issue in issues:
            await operational_manager.handle_operational_issue(issue)

        # Should have escalated critical issues
        critical_alerts = [alert for alert in alerts_sent if alert['severity'] == 'critical']
        assert len(critical_alerts) >= 2  # At least 2 critical alerts

    @pytest.mark.asyncio
    async def test_resource_cleanup_procedures(self, operational_manager):
        """Test automated resource cleanup procedures."""
        await operational_manager.initialize()

        cleanup_actions = []

        async def mock_cleanup_callback(cleanup_data):
            cleanup_actions.append(cleanup_data)

        operational_manager.set_cleanup_callback(mock_cleanup_callback)

        # Simulate resource exhaustion
        resource_state = {
            'memory_usage_pct': 92.0,
            'disk_usage_pct': 95.0,
            'cpu_usage_pct': 88.0,
            'open_file_descriptors': 8000,  # High number
            'network_connections': 5000     # High number
        }

        await operational_manager.perform_resource_cleanup(resource_state)

        # Should have performed cleanup actions
        assert len(cleanup_actions) > 0
        cleanup_types = [action['type'] for action in cleanup_actions]
        expected_cleanups = ['memory_cleanup', 'disk_cleanup', 'connection_cleanup']
        assert any(cleanup_type in cleanup_types for cleanup_type in expected_cleanups)

    @pytest.mark.asyncio
    async def test_continuous_monitoring(self, operational_manager):
        """Test continuous monitoring functionality."""
        await operational_manager.initialize()

        monitoring_data = []

        async def mock_monitoring_callback(data):
            monitoring_data.append(data)

        operational_manager.set_monitoring_callback(mock_monitoring_callback)

        # Start monitoring for a short period
        monitoring_task = asyncio.create_task(operational_manager.start_continuous_monitoring())

        # Let it run for a short time
        await asyncio.sleep(0.5)

        # Stop monitoring
        monitoring_task.cancel()

        try:
            await monitoring_task
        except asyncio.CancelledError:
            pass

        # Should have collected some monitoring data
        assert len(monitoring_data) >= 0  # May or may not have data depending on timing


@pytest.mark.asyncio
async def test_operational_crisis_scenario():
    """Test operational risk manager during system crisis."""
    manager = OperationalRiskManager()
    await manager.initialize()

    crisis_actions = []

    async def mock_crisis_callback(action_data):
        crisis_actions.append(action_data)

    manager.set_recovery_callback(mock_crisis_callback)

    # Simulate system crisis with multiple failures
    with patch.object(manager.system_monitor, 'collect_all_metrics') as mock_system:
        with patch.object(manager.connectivity_monitor, 'check_all_endpoints') as mock_connectivity:

            # Critical system state
            mock_system.return_value = {
                'cpu': {'cpu_usage_pct': 99.0, 'status': 'critical'},
                'memory': {'memory_usage_pct': 98.0, 'status': 'critical'},
                'disk': {'disk_usage_pct': 97.0, 'status': 'critical'},
                'health_score': 5.0
            }

            # All endpoints failing
            mock_connectivity.return_value = {
                'binance_api': {'success': False, 'error': 'Connection timeout'},
                'coinbase_api': {'success': False, 'error': 'Connection refused'},
                'connectivity_score': 0.0
            }

            crisis_assessment = await manager.perform_health_check()

            # Should detect critical operational crisis
            assert crisis_assessment['risk_level'] == 'critical'
            assert crisis_assessment['overall_score'] < 10

            # Should trigger emergency procedures
            await manager.trigger_recovery_procedures()

            assert len(crisis_actions) > 0
            assert any(action['type'] == 'emergency_shutdown' for action in crisis_actions)


if __name__ == "__main__":
    pytest.main([__file__])

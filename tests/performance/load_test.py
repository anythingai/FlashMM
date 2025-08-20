#!/usr/bin/env python3
"""
FlashMM Load Testing Suite
Performance validation and load testing for production deployment
"""

import asyncio
import aiohttp
import time
import json
import statistics
import argparse
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LoadTestConfig:
    """Load test configuration"""
    base_url: str = "http://localhost:8000"
    concurrent_users: int = 100
    test_duration: int = 300  # 5 minutes
    ramp_up_time: int = 60    # 1 minute
    endpoints: List[str] = None
    expected_rps: int = 1000
    max_latency_p95: float = 0.1  # 100ms
    max_error_rate: float = 0.01  # 1%

    def __post_init__(self):
        if self.endpoints is None:
            self.endpoints = [
                "/health",
                "/health/detailed", 
                "/metrics",
                "/api/v1/trading/status",
                "/api/v1/market/data",
                "/api/v1/positions",
                "/api/v1/orders"
            ]

@dataclass 
class TestResult:
    """Individual test result"""
    endpoint: str
    method: str
    status_code: int
    response_time: float
    success: bool
    error_message: Optional[str] = None
    timestamp: float = 0

@dataclass
class LoadTestReport:
    """Load test report summary"""
    config: LoadTestConfig
    start_time: float
    end_time: float
    total_requests: int
    successful_requests: int
    failed_requests: int
    average_rps: float
    peak_rps: float
    latency_p50: float
    latency_p95: float
    latency_p99: float
    error_rate: float
    errors_by_type: Dict[str, int]
    performance_metrics: Dict[str, float]

class FlashMMLoadTester:
    """FlashMM load testing framework"""
    
    def __init__(self, config: LoadTestConfig):
        self.config = config
        self.results: List[TestResult] = []
        self.start_time = 0
        self.end_time = 0
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def __aenter__(self):
        connector = aiohttp.TCPConnector(
            limit=self.config.concurrent_users * 2,
            limit_per_host=self.config.concurrent_users,
            keepalive_timeout=30,
            enable_cleanup_closed=True
        )
        
        timeout = aiohttp.ClientTimeout(total=30, connect=10)
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={
                "User-Agent": "FlashMM-LoadTester/1.0",
                "Accept": "application/json"
            }
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def make_request(self, endpoint: str, method: str = "GET") -> TestResult:
        """Make a single HTTP request and record results"""
        start_time = time.time()
        
        try:
            url = f"{self.config.base_url}{endpoint}"
            
            async with self.session.request(method, url) as response:
                response_time = time.time() - start_time
                
                # Read response to ensure full request completion
                await response.text()
                
                return TestResult(
                    endpoint=endpoint,
                    method=method,
                    status_code=response.status,
                    response_time=response_time,
                    success=200 <= response.status < 400,
                    timestamp=start_time
                )
                
        except Exception as e:
            response_time = time.time() - start_time
            return TestResult(
                endpoint=endpoint,
                method=method,
                status_code=0,
                response_time=response_time,
                success=False,
                error_message=str(e),
                timestamp=start_time
            )

    async def user_simulation(self, user_id: int) -> List[TestResult]:
        """Simulate a single user's load pattern"""
        user_results = []
        
        # Calculate when this user should start (for ramp-up)
        start_delay = (user_id * self.config.ramp_up_time) / self.config.concurrent_users
        await asyncio.sleep(start_delay)
        
        user_start_time = time.time()
        
        # Run load test for the specified duration
        while time.time() - user_start_time < self.config.test_duration:
            # Select endpoint (could be weighted based on real usage patterns)
            endpoint = self.config.endpoints[user_id % len(self.config.endpoints)]
            
            result = await self.make_request(endpoint)
            user_results.append(result)
            
            # Add some realistic think time between requests
            await asyncio.sleep(0.1 + (user_id % 10) * 0.01)
        
        return user_results

    async def run_load_test(self) -> LoadTestReport:
        """Execute the complete load test"""
        logger.info(f"Starting load test with {self.config.concurrent_users} users for {self.config.test_duration}s")
        
        self.start_time = time.time()
        
        # Create user simulation tasks
        tasks = [
            self.user_simulation(user_id) 
            for user_id in range(self.config.concurrent_users)
        ]
        
        # Run all user simulations concurrently
        user_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        self.end_time = time.time()
        
        # Flatten results
        self.results = []
        for result_list in user_results:
            if isinstance(result_list, list):
                self.results.extend(result_list)
            else:
                logger.error(f"User simulation failed: {result_list}")
        
        return self.generate_report()

    def generate_report(self) -> LoadTestReport:
        """Generate comprehensive load test report"""
        if not self.results:
            raise ValueError("No test results available")
        
        # Basic statistics
        total_requests = len(self.results)
        successful_results = [r for r in self.results if r.success]
        failed_results = [r for r in self.results if not r.success]
        
        successful_requests = len(successful_results)
        failed_requests = len(failed_results)
        
        # Calculate RPS
        test_duration = self.end_time - self.start_time
        average_rps = total_requests / test_duration
        
        # Calculate peak RPS (using 10-second windows)
        rps_samples = []
        for i in range(0, int(test_duration), 10):
            window_start = self.start_time + i
            window_end = window_start + 10
            window_requests = [
                r for r in self.results 
                if window_start <= r.timestamp <= window_end
            ]
            rps_samples.append(len(window_requests) / 10)
        
        peak_rps = max(rps_samples) if rps_samples else 0
        
        # Calculate latency percentiles
        response_times = [r.response_time for r in successful_results]
        if response_times:
            latency_p50 = statistics.quantiles(response_times, n=2)[0]  # median
            latency_p95 = statistics.quantiles(response_times, n=20)[18]  # 95th percentile
            latency_p99 = statistics.quantiles(response_times, n=100)[98]  # 99th percentile
        else:
            latency_p50 = latency_p95 = latency_p99 = 0
        
        # Error analysis
        error_rate = failed_requests / total_requests if total_requests > 0 else 0
        errors_by_type = {}
        
        for result in failed_results:
            error_key = f"HTTP_{result.status_code}" if result.status_code > 0 else "NETWORK_ERROR"
            errors_by_type[error_key] = errors_by_type.get(error_key, 0) + 1
        
        # Performance metrics
        performance_metrics = {
            "avg_response_time": statistics.mean(response_times) if response_times else 0,
            "min_response_time": min(response_times) if response_times else 0,
            "max_response_time": max(response_times) if response_times else 0,
            "std_dev_response_time": statistics.stdev(response_times) if len(response_times) > 1 else 0,
            "throughput_mbps": self._calculate_throughput(),
            "concurrent_users": self.config.concurrent_users,
            "test_duration": test_duration
        }
        
        return LoadTestReport(
            config=self.config,
            start_time=self.start_time,
            end_time=self.end_time,
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            average_rps=average_rps,
            peak_rps=peak_rps,
            latency_p50=latency_p50,
            latency_p95=latency_p95,
            latency_p99=latency_p99,
            error_rate=error_rate,
            errors_by_type=errors_by_type,
            performance_metrics=performance_metrics
        )
    
    def _calculate_throughput(self) -> float:
        """Calculate approximate throughput in Mbps"""
        # Estimate average response size (would be more accurate with actual measurements)
        avg_response_size = 1024  # 1KB average
        total_bytes = len(self.results) * avg_response_size
        duration = self.end_time - self.start_time
        return (total_bytes * 8) / (duration * 1_000_000)  # Convert to Mbps

    def validate_performance_requirements(self, report: LoadTestReport) -> Tuple[bool, List[str]]:
        """Validate performance against requirements"""
        issues = []
        
        # Check RPS requirement
        if report.average_rps < self.config.expected_rps:
            issues.append(f"Average RPS {report.average_rps:.1f} below target {self.config.expected_rps}")
        
        # Check latency requirement
        if report.latency_p95 > self.config.max_latency_p95:
            issues.append(f"P95 latency {report.latency_p95*1000:.1f}ms above target {self.config.max_latency_p95*1000:.1f}ms")
        
        # Check error rate requirement
        if report.error_rate > self.config.max_error_rate:
            issues.append(f"Error rate {report.error_rate*100:.2f}% above target {self.config.max_error_rate*100:.2f}%")
        
        return len(issues) == 0, issues

def print_load_test_report(report: LoadTestReport):
    """Print formatted load test report"""
    print("\n" + "="*60)
    print("FlashMM Load Test Report")
    print("="*60)
    
    print(f"Test Configuration:")
    print(f"  Base URL: {report.config.base_url}")
    print(f"  Concurrent Users: {report.config.concurrent_users}")
    print(f"  Test Duration: {report.config.test_duration}s")
    print(f"  Ramp-up Time: {report.config.ramp_up_time}s")
    print()
    
    print(f"Test Summary:")
    print(f"  Total Requests: {report.total_requests:,}")
    print(f"  Successful: {report.successful_requests:,} ({report.successful_requests/report.total_requests*100:.1f}%)")
    print(f"  Failed: {report.failed_requests:,} ({report.error_rate*100:.2f}%)")
    print(f"  Average RPS: {report.average_rps:.1f}")
    print(f"  Peak RPS: {report.peak_rps:.1f}")
    print()
    
    print(f"Latency Statistics:")
    print(f"  P50 (Median): {report.latency_p50*1000:.1f}ms")
    print(f"  P95: {report.latency_p95*1000:.1f}ms")
    print(f"  P99: {report.latency_p99*1000:.1f}ms")
    print(f"  Average: {report.performance_metrics['avg_response_time']*1000:.1f}ms")
    print(f"  Min: {report.performance_metrics['min_response_time']*1000:.1f}ms")
    print(f"  Max: {report.performance_metrics['max_response_time']*1000:.1f}ms")
    print()
    
    if report.errors_by_type:
        print(f"Error Breakdown:")
        for error_type, count in sorted(report.errors_by_type.items()):
            percentage = (count / report.total_requests) * 100
            print(f"  {error_type}: {count} ({percentage:.2f}%)")
        print()
    
    print(f"Performance Metrics:")
    print(f"  Throughput: {report.performance_metrics['throughput_mbps']:.2f} Mbps")
    print(f"  Std Dev: {report.performance_metrics['std_dev_response_time']*1000:.1f}ms")

async def run_health_check_load_test(base_url: str, duration: int = 60) -> Dict:
    """Quick load test focused on health endpoints"""
    config = LoadTestConfig(
        base_url=base_url,
        concurrent_users=50,
        test_duration=duration,
        endpoints=["/health", "/health/detailed"],
        expected_rps=500
    )
    
    async with FlashMMLoadTester(config) as tester:
        return await tester.run_load_test()

async def run_api_load_test(base_url: str, duration: int = 300) -> Dict:
    """Comprehensive API load test"""
    config = LoadTestConfig(
        base_url=base_url,
        concurrent_users=100,
        test_duration=duration,
        endpoints=[
            "/health",
            "/metrics", 
            "/api/v1/trading/status",
            "/api/v1/market/data",
            "/api/v1/positions"
        ],
        expected_rps=1000
    )
    
    async with FlashMMLoadTester(config) as tester:
        return await tester.run_load_test()

async def run_stress_test(base_url: str, duration: int = 180) -> Dict:
    """Stress test to find breaking points"""
    config = LoadTestConfig(
        base_url=base_url,
        concurrent_users=500,  # High concurrency
        test_duration=duration,
        ramp_up_time=30,       # Quick ramp-up
        endpoints=["/health"],
        expected_rps=2000,     # High RPS target
        max_latency_p95=0.5,   # More lenient latency
        max_error_rate=0.05    # More lenient error rate
    )
    
    async with FlashMMLoadTester(config) as tester:
        return await tester.run_load_test()

class KubernetesLoadTester:
    """Kubernetes-specific load testing"""
    
    def __init__(self, namespace: str = "flashmm"):
        self.namespace = namespace
    
    def get_service_url(self, service_name: str = "flashmm-app") -> Optional[str]:
        """Get internal service URL for load testing"""
        try:
            import subprocess
            result = subprocess.run([
                "kubectl", "get", "service", service_name, "-n", self.namespace,
                "-o", "jsonpath={.spec.clusterIP}:{.spec.ports[0].port}"
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                cluster_ip_port = result.stdout.strip()
                return f"http://{cluster_ip_port}"
            
        except Exception as e:
            logger.error(f"Failed to get service URL: {e}")
        
        return None
    
    async def run_internal_load_test(self, test_type: str = "api") -> Optional[LoadTestReport]:
        """Run load test against internal Kubernetes service"""
        service_url = self.get_service_url()
        if not service_url:
            logger.error("Could not determine service URL")
            return None
        
        if test_type == "health":
            return await run_health_check_load_test(service_url)
        elif test_type == "stress":
            return await run_stress_test(service_url)
        else:
            return await run_api_load_test(service_url)

def run_performance_benchmarks():
    """Run comprehensive performance benchmarks"""
    benchmarks = {
        "container_startup": benchmark_container_startup,
        "scaling_performance": benchmark_scaling_performance,
        "database_performance": benchmark_database_performance,
        "cache_performance": benchmark_cache_performance
    }
    
    results = {}
    
    for benchmark_name, benchmark_func in benchmarks.items():
        logger.info(f"Running {benchmark_name} benchmark...")
        try:
            result = benchmark_func()
            results[benchmark_name] = result
            logger.info(f"{benchmark_name} benchmark completed")
        except Exception as e:
            logger.error(f"{benchmark_name} benchmark failed: {e}")
            results[benchmark_name] = {"error": str(e)}
    
    return results

def benchmark_container_startup():
    """Benchmark container startup time"""
    import subprocess
    
    start_time = time.time()
    
    # Create a test pod and measure startup time
    result = subprocess.run([
        "kubectl", "run", f"startup-benchmark-{int(time.time())}",
        "--image=ghcr.io/flashmm/flashmm:latest",
        "--restart=Never",
        "--rm", "-i",
        "--", "echo", "startup-complete"
    ], capture_output=True, text=True, timeout=120)
    
    startup_time = time.time() - start_time
    
    return {
        "startup_time_seconds": startup_time,
        "success": result.returncode == 0,
        "target_startup_time": 30.0,  # 30 second target
        "passes_requirement": startup_time < 30.0
    }

def benchmark_scaling_performance():
    """Benchmark scaling performance"""
    import subprocess
    
    namespace = "flashmm"
    deployment = "flashmm-app"
    
    # Get current replica count
    result = subprocess.run([
        "kubectl", "get", "deployment", deployment, "-n", namespace,
        "-o", "jsonpath={.status.replicas}"
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        return {"error": "Could not get current replica count"}
    
    current_replicas = int(result.stdout.strip())
    target_replicas = current_replicas + 2
    
    # Scale up and measure time
    start_time = time.time()
    
    subprocess.run([
        "kubectl", "scale", "deployment", deployment, "-n", namespace,
        "--replicas", str(target_replicas)
    ], capture_output=True)
    
    # Wait for rollout to complete
    subprocess.run([
        "kubectl", "rollout", "status", f"deployment/{deployment}",
        "-n", namespace, "--timeout=300s"
    ], capture_output=True)
    
    scale_up_time = time.time() - start_time
    
    # Scale back to original
    subprocess.run([
        "kubectl", "scale", "deployment", deployment, "-n", namespace,
        "--replicas", str(current_replicas)
    ], capture_output=True)
    
    return {
        "scale_up_time_seconds": scale_up_time,
        "original_replicas": current_replicas,
        "target_replicas": target_replicas,
        "target_scale_time": 60.0,  # 1 minute target
        "passes_requirement": scale_up_time < 60.0
    }

def benchmark_database_performance():
    """Benchmark database connection and query performance"""
    import subprocess
    
    namespace = "flashmm"
    
    # Test PostgreSQL performance
    start_time = time.time()
    
    result = subprocess.run([
        "kubectl", "exec", "-n", namespace, "deployment/postgres", "--",
        "psql", "-U", "flashmm", "-d", f"flashmm_{namespace}",
        "-c", "SELECT COUNT(*) FROM information_schema.tables;"
    ], capture_output=True, text=True, timeout=30)
    
    query_time = time.time() - start_time
    
    return {
        "postgres_query_time_seconds": query_time,
        "success": result.returncode == 0,
        "target_query_time": 1.0,  # 1 second target
        "passes_requirement": query_time < 1.0 and result.returncode == 0
    }

def benchmark_cache_performance():
    """Benchmark Redis cache performance"""
    import subprocess
    
    namespace = "flashmm"
    
    # Test Redis performance
    start_time = time.time()
    
    result = subprocess.run([
        "kubectl", "exec", "-n", namespace, "deployment/redis", "--",
        "redis-cli", "ping"
    ], capture_output=True, text=True, timeout=10)
    
    ping_time = time.time() - start_time
    
    return {
        "redis_ping_time_seconds": ping_time,
        "success": result.returncode == 0 and "PONG" in result.stdout,
        "target_ping_time": 0.01,  # 10ms target
        "passes_requirement": ping_time < 0.01 and "PONG" in result.stdout
    }

async def main():
    """Main load testing execution"""
    parser = argparse.ArgumentParser(description="FlashMM Load Testing Suite")
    parser.add_argument("--base-url", default="http://localhost:8000", help="Base URL for testing")
    parser.add_argument("--test-type", choices=["health", "api", "stress", "benchmark"], 
                       default="api", help="Type of load test")
    parser.add_argument("--users", type=int, default=100, help="Number of concurrent users")
    parser.add_argument("--duration", type=int, default=300, help="Test duration in seconds")
    parser.add_argument("--namespace", default="flashmm", help="Kubernetes namespace")
    parser.add_argument("--output", help="Output file for results (JSON)")
    parser.add_argument("--kubernetes", action="store_true", help="Use Kubernetes internal service")
    
    args = parser.parse_args()
    
    if args.kubernetes:
        k8s_tester = KubernetesLoadTester(args.namespace)
        report = await k8s_tester.run_internal_load_test(args.test_type)
        if not report:
            logger.error("Kubernetes load test failed")
            return 1
    elif args.test_type == "benchmark":
        benchmark_results = run_performance_benchmarks()
        print("\nPerformance Benchmark Results:")
        print("=" * 40)
        for name, result in benchmark_results.items():
            print(f"\n{name}:")
            for key, value in result.items():
                print(f"  {key}: {value}")
        return 0
    else:
        config = LoadTestConfig(
            base_url=args.base_url,
            concurrent_users=args.users,
            test_duration=args.duration
        )
        
        async with FlashMMLoadTester(config) as tester:
            report = await tester.run_load_test()
    
    # Print report
    print_load_test_report(report)
    
    # Validate performance requirements
    passed, issues = FlashMMLoadTester(report.config).validate_performance_requirements(report)
    
    if not passed:
        print(f"\nPerformance Requirements Failed:")
        for issue in issues:
            print(f"  ❌ {issue}")
    else:
        print(f"\n✅ All performance requirements passed!")
    
    # Save results if output file specified
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(asdict(report), f, indent=2, default=str)
        print(f"\nResults saved to {args.output}")
    
    return 0 if passed else 1

if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))
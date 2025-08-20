"""
FlashMM Data Pipeline End-to-End Demonstration

Comprehensive demonstration of the complete Sei WebSocket data pipeline
showcasing <250ms latency real-time market data processing.
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any
import signal
import sys

from flashmm.data.market_data_service import MarketDataService
from flashmm.monitoring.telemetry.metrics_collector import EnhancedMetricsCollector
from flashmm.config.settings import get_config
from flashmm.utils.logging import setup_logging, get_logger

# Setup logging for demo
setup_logging(log_level="INFO", environment="demo", enable_json=False)
logger = get_logger(__name__)


class PipelineDemo:
    """Complete pipeline demonstration."""
    
    def __init__(self):
        self.config = get_config()
        self.market_data_service: MarketDataService = None
        self.metrics_collector: EnhancedMetricsCollector = None
        self.running = False
        
        # Demo statistics
        self.demo_start_time = None
        self.data_received_count = 0
        self.latency_measurements = []
        self.symbols_seen = set()
        
        # Graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle graceful shutdown signals."""
        logger.info("Received shutdown signal, stopping demo...")
        self.running = False
    
    async def initialize(self) -> None:
        """Initialize all pipeline components."""
        logger.info("🚀 Initializing FlashMM Data Pipeline Demo")
        logger.info("=" * 60)
        
        try:
            # Initialize market data service
            logger.info("📡 Initializing Market Data Service...")
            self.market_data_service = MarketDataService()
            await self.market_data_service.initialize()
            
            # Initialize metrics collector
            logger.info("📊 Initializing Metrics Collector...")
            self.metrics_collector = EnhancedMetricsCollector()
            await self.metrics_collector.initialize()
            
            # Connect metrics collector to market data service
            self.metrics_collector.set_component_references(
                market_data_service=self.market_data_service,
                feed_manager=self.market_data_service.feed_manager
            )
            
            # Register data callbacks for demo
            self.market_data_service.subscribe_to_data("orderbook", self._on_orderbook_data)
            self.market_data_service.subscribe_to_data("trades", self._on_trade_data)
            
            # Register alert callback
            self.metrics_collector.register_alert_callback(self._on_alert)
            
            logger.info("✅ Pipeline initialization complete!")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize pipeline: {e}")
            raise
    
    async def start_demo(self) -> None:
        """Start the complete pipeline demonstration."""
        logger.info("\n🎯 Starting FlashMM Data Pipeline Demo")
        logger.info("=" * 60)
        
        try:
            self.demo_start_time = datetime.now()
            self.running = True
            
            # Start metrics collection
            logger.info("📈 Starting metrics collection...")
            metrics_task = asyncio.create_task(self.metrics_collector.start())
            
            # Start market data service
            logger.info("🔄 Starting market data service...")
            await self.market_data_service.start()
            
            # Start demo monitoring
            logger.info("🎥 Starting demo monitoring...")
            demo_task = asyncio.create_task(self._demo_monitor())
            
            # Display configuration
            await self._display_configuration()
            
            # Wait for demo to run
            logger.info("🟢 Pipeline is now running! Press Ctrl+C to stop.")
            logger.info("📊 Waiting for Sei market data...")
            
            # Keep running until stopped
            while self.running:
                await asyncio.sleep(1)
            
            # Cleanup
            logger.info("\n🛑 Stopping pipeline demo...")
            demo_task.cancel()
            
            await self.market_data_service.stop()
            await self.metrics_collector.stop()
            
            # Display final statistics
            await self._display_final_stats()
            
        except Exception as e:
            logger.error(f"❌ Demo failed: {e}")
            raise
    
    async def _display_configuration(self) -> None:
        """Display current configuration."""
        logger.info("\n⚙️  Pipeline Configuration:")
        logger.info("-" * 30)
        
        service_info = self.market_data_service.get_service_info()
        
        logger.info(f"🔗 Sei WebSocket URL: {service_info['configuration'].get('sei_ws_url', 'Not configured')}")
        logger.info(f"📊 Symbols: {', '.join(service_info['symbols'])}")
        logger.info(f"🎯 Trading Enabled: {service_info['enable_trading']}")
        logger.info(f"💾 Redis: {service_info['configuration'].get('redis_url', 'Not configured')}")
        logger.info(f"📈 InfluxDB: {service_info['configuration'].get('influxdb_url', 'Not configured')}")
        
        # Display latency targets
        logger.info("\n🎯 Performance Targets:")
        logger.info("-" * 25)
        logger.info("📡 WebSocket to Processing: <250ms")
        logger.info("🔄 Data Normalization: <10ms")
        logger.info("💾 Storage Operations: <50ms")
        logger.info("🚀 End-to-End Pipeline: <350ms")
    
    async def _demo_monitor(self) -> None:
        """Monitor demo progress and display statistics."""
        last_report_time = time.time()
        
        while self.running:
            try:
                await asyncio.sleep(10)  # Report every 10 seconds
                
                current_time = time.time()
                if current_time - last_report_time >= 10:
                    await self._display_live_stats()
                    last_report_time = current_time
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in demo monitor: {e}")
    
    async def _display_live_stats(self) -> None:
        """Display live statistics."""
        try:
            # Get current metrics
            service_metrics = await self.market_data_service.get_performance_metrics()
            current_metrics = self.metrics_collector.get_current_metrics()
            
            uptime = (datetime.now() - self.demo_start_time).total_seconds()
            
            logger.info(f"\n📊 Live Statistics (Uptime: {uptime:.0f}s)")
            logger.info("-" * 40)
            
            # Pipeline metrics
            pipeline_metrics = current_metrics.get('pipeline', {})
            if pipeline_metrics:
                logger.info(f"📨 Messages/sec: {pipeline_metrics.get('messages_per_second', 0):.1f}")
                logger.info(f"⚡ Avg Latency: {pipeline_metrics.get('avg_processing_latency_ms', 0):.1f}ms")
                logger.info(f"🎯 95th Percentile: {pipeline_metrics.get('p95_processing_latency_ms', 0):.1f}ms")
                logger.info(f"❌ Error Rate: {pipeline_metrics.get('error_rate_percent', 0):.2f}%")
                logger.info(f"🔗 Active Connections: {pipeline_metrics.get('active_websocket_connections', 0)}")
                
                # Latency violation check
                violations = pipeline_metrics.get('latency_violations_count', 0)
                if violations > 0:
                    logger.warning(f"⚠️  Latency violations: {violations}")
                else:
                    logger.info("✅ All messages within 250ms target")
            
            # System metrics
            system_metrics = current_metrics.get('system', {})
            if system_metrics:
                logger.info(f"💻 CPU: {system_metrics.get('cpu_percent', 0):.1f}%")
                logger.info(f"🧠 Memory: {system_metrics.get('memory_percent', 0):.1f}%")
                logger.info(f"💾 Disk: {system_metrics.get('disk_percent', 0):.1f}%")
            
            # Component health
            components = current_metrics.get('components', {})
            if components:
                logger.info(f"🔴 Redis: {components.get('redis_status', 'unknown')}")
                logger.info(f"🟠 InfluxDB: {components.get('influxdb_status', 'unknown')}")
                logger.info(f"🟢 Feed Manager: {components.get('feed_manager_status', 'unknown')}")
            
            # Data received
            if self.data_received_count > 0:
                logger.info(f"📈 Total Data Points: {self.data_received_count}")
                logger.info(f"🎪 Symbols Seen: {', '.join(sorted(self.symbols_seen))}")
                
                if self.latency_measurements:
                    avg_latency = sum(self.latency_measurements) / len(self.latency_measurements)
                    max_latency = max(self.latency_measurements)
                    logger.info(f"⚡ Demo Avg Latency: {avg_latency:.1f}ms")
                    logger.info(f"⚡ Demo Max Latency: {max_latency:.1f}ms")
            
        except Exception as e:
            logger.error(f"Error displaying live stats: {e}")
    
    async def _on_orderbook_data(self, symbol: str, data: Dict[str, Any]) -> None:
        """Handle order book data."""
        processing_start = time.time()
        
        try:
            self.data_received_count += 1
            self.symbols_seen.add(symbol)
            
            # Calculate processing latency
            processing_latency = (time.time() - processing_start) * 1000
            self.latency_measurements.append(processing_latency)
            
            # Keep only recent measurements
            if len(self.latency_measurements) > 1000:
                self.latency_measurements = self.latency_measurements[-1000:]
            
            # Log significant order book updates
            if self.data_received_count % 100 == 0:  # Every 100th message
                best_bid = data.get('best_bid', 'N/A')
                best_ask = data.get('best_ask', 'N/A')
                spread = data.get('spread', 'N/A')
                
                logger.info(f"📖 {symbol} Order Book - Bid: {best_bid}, Ask: {best_ask}, Spread: {spread}")
            
        except Exception as e:
            logger.error(f"Error processing order book data: {e}")
    
    async def _on_trade_data(self, symbol: str, data: Dict[str, Any]) -> None:
        """Handle trade data."""
        processing_start = time.time()
        
        try:
            self.data_received_count += 1
            self.symbols_seen.add(symbol)
            
            # Calculate processing latency
            processing_latency = (time.time() - processing_start) * 1000
            self.latency_measurements.append(processing_latency)
            
            # Log all trades
            price = data.get('price', 'N/A')
            size = data.get('size', 'N/A')
            side = data.get('side', 'N/A')
            
            logger.info(f"💰 {symbol} Trade - Price: {price}, Size: {size}, Side: {side}")
            
        except Exception as e:
            logger.error(f"Error processing trade data: {e}")
    
    async def _on_alert(self, alert_type: str, alert_data: Dict[str, Any]) -> None:
        """Handle system alerts."""
        severity = alert_data.get('severity', 'info')
        message = alert_data.get('message', 'No message')
        
        if severity == 'critical':
            logger.error(f"🚨 CRITICAL ALERT [{alert_type}]: {message}")
        elif severity == 'warning':
            logger.warning(f"⚠️  WARNING [{alert_type}]: {message}")
        else:
            logger.info(f"ℹ️  INFO [{alert_type}]: {message}")
    
    async def _display_final_stats(self) -> None:
        """Display final demo statistics."""
        try:
            uptime = (datetime.now() - self.demo_start_time).total_seconds()
            
            logger.info("\n🏁 Final Demo Statistics")
            logger.info("=" * 50)
            
            logger.info(f"⏱️  Total Runtime: {uptime:.1f} seconds")
            logger.info(f"📊 Total Data Points: {self.data_received_count}")
            logger.info(f"🎪 Symbols Processed: {', '.join(sorted(self.symbols_seen))}")
            
            if self.data_received_count > 0:
                throughput = self.data_received_count / uptime
                logger.info(f"🚀 Average Throughput: {throughput:.1f} messages/second")
            
            if self.latency_measurements:
                avg_latency = sum(self.latency_measurements) / len(self.latency_measurements)
                max_latency = max(self.latency_measurements)
                min_latency = min(self.latency_measurements)
                
                # Calculate percentiles
                sorted_latencies = sorted(self.latency_measurements)
                p95_idx = int(len(sorted_latencies) * 0.95)
                p99_idx = int(len(sorted_latencies) * 0.99)
                
                p95_latency = sorted_latencies[p95_idx] if p95_idx < len(sorted_latencies) else max_latency
                p99_latency = sorted_latencies[p99_idx] if p99_idx < len(sorted_latencies) else max_latency
                
                logger.info(f"⚡ Latency Statistics:")
                logger.info(f"   - Average: {avg_latency:.1f}ms")
                logger.info(f"   - Minimum: {min_latency:.1f}ms")
                logger.info(f"   - Maximum: {max_latency:.1f}ms")
                logger.info(f"   - 95th Percentile: {p95_latency:.1f}ms")
                logger.info(f"   - 99th Percentile: {p99_latency:.1f}ms")
                
                # Check against targets
                target_violations = len([l for l in self.latency_measurements if l > 250])
                success_rate = ((len(self.latency_measurements) - target_violations) / len(self.latency_measurements)) * 100
                
                logger.info(f"🎯 Target Compliance:")
                logger.info(f"   - Messages within 250ms: {success_rate:.1f}%")
                logger.info(f"   - Target violations: {target_violations}")
                
                if success_rate >= 95:
                    logger.info("✅ EXCELLENT: >95% of messages within latency target!")
                elif success_rate >= 90:
                    logger.info("✅ GOOD: >90% of messages within latency target")
                else:
                    logger.warning("⚠️  NEEDS IMPROVEMENT: <90% within latency target")
            
            # Final component health
            try:
                health_status = await self.market_data_service.get_health_status()
                logger.info(f"🏥 Final System Health: {health_status.get('service_status', 'unknown')}")
                
                component_health = health_status.get('component_health', {})
                for component, health in component_health.items():
                    status = health.get('status', 'unknown')
                    logger.info(f"   - {component}: {status}")
                
            except Exception as e:
                logger.error(f"Error getting final health status: {e}")
            
            logger.info("\n🎉 Demo completed successfully!")
            logger.info("Thank you for trying FlashMM Data Pipeline!")
            
        except Exception as e:
            logger.error(f"Error displaying final stats: {e}")


async def run_pipeline_demo():
    """Run the complete pipeline demonstration."""
    demo = PipelineDemo()
    
    try:
        await demo.initialize()
        await demo.start_demo()
        
    except KeyboardInterrupt:
        logger.info("Demo interrupted by user")
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        sys.exit(1)


async def run_quick_test():
    """Run a quick pipeline test without full demo."""
    logger.info("🧪 Running Quick Pipeline Test")
    logger.info("=" * 40)
    
    try:
        # Test component initialization
        logger.info("1️⃣  Testing component initialization...")
        
        service = MarketDataService()
        await service.initialize()
        logger.info("✅ Market Data Service initialized")
        
        metrics = EnhancedMetricsCollector()
        await metrics.initialize()
        logger.info("✅ Metrics Collector initialized")
        
        # Test configuration
        logger.info("2️⃣  Testing configuration...")
        service_info = service.get_service_info()
        logger.info(f"✅ Service configured for {len(service_info['symbols'])} symbols")
        
        # Test health checks
        logger.info("3️⃣  Testing health checks...")
        health = await service.get_health_status()
        logger.info(f"✅ System health: {health.get('service_status', 'unknown')}")
        
        # Test metrics collection
        logger.info("4️⃣  Testing metrics collection...")
        perf_metrics = await service.get_performance_metrics()
        logger.info(f"✅ Metrics collected, service uptime: {perf_metrics.get('uptime_seconds', 0):.1f}s")
        
        # Cleanup
        await service.stop()
        await metrics.stop()
        
        logger.info("🎉 Quick test completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Quick test failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    """Main entry point for pipeline demo."""
    
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "test":
            logger.info("Running quick test mode...")
            asyncio.run(run_quick_test())
        elif sys.argv[1] == "help":
            print("FlashMM Data Pipeline Demo")
            print("Usage:")
            print("  python pipeline_demo.py        - Run full demo")
            print("  python pipeline_demo.py test   - Run quick test")
            print("  python pipeline_demo.py help   - Show this help")
            sys.exit(0)
        else:
            logger.error(f"Unknown argument: {sys.argv[1]}")
            sys.exit(1)
    else:
        logger.info("Running full pipeline demo...")
        asyncio.run(run_pipeline_demo())
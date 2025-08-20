"""
Performance benchmarks for FlashMM data pipeline.

Tests to ensure the pipeline meets the <250ms latency target from Sei WebSocket
to internal processing completion.
"""

import pytest
import asyncio
import time
import statistics
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from flashmm.data.ingestion.data_normalizer import SeiDataNormalizer
from flashmm.data.ingestion.websocket_client import SeiWebSocketClient
from flashmm.data.market_data_service import MarketDataService


class TestLatencyBenchmarks:
    """Test latency performance of the data pipeline."""
    
    @pytest.fixture
    def sample_sei_orderbook_message(self):
        """Large order book message for realistic testing."""
        return {
            "jsonrpc": "2.0",
            "id": "orderbook_sub",
            "result": {
                "events": [
                    {
                        "type": "OrderBookUpdate",
                        "attributes": [
                            {"key": "market_id", "value": "SEI/USDC"},
                            {"key": "sequence", "value": "12345"},
                            {"key": "timestamp", "value": "2024-01-01T00:00:00Z"}
                        ],
                        "data": {
                            "bids": [[f"0.{42-i:03d}", f"{1000+i*100}"] for i in range(20)],
                            "asks": [[f"0.{43+i:03d}", f"{1500+i*50}"] for i in range(20)]
                        }
                    }
                ]
            }
        }
    
    @pytest.fixture
    def sample_sei_trade_message(self):
        """Trade message for testing."""
        return {
            "jsonrpc": "2.0",
            "id": "trade_sub",
            "result": {
                "events": [
                    {
                        "type": "Trade",
                        "attributes": [
                            {"key": "market_id", "value": "SEI/USDC"},
                            {"key": "price", "value": "0.0425"},
                            {"key": "size", "value": "1000"},
                            {"key": "side", "value": "buy"},
                            {"key": "trade_id", "value": "67890"},
                            {"key": "timestamp", "value": "2024-01-01T00:01:00Z"}
                        ]
                    }
                ]
            }
        }
    
    @pytest.mark.asyncio
    async def test_data_normalization_latency(self, sample_sei_orderbook_message):
        """Test data normalization latency is under 10ms target."""
        normalizer = SeiDataNormalizer()
        await normalizer.initialize()
        
        latencies = []
        
        # Run multiple iterations to get statistical data
        for _ in range(100):
            start_time = time.time()
            
            result = await normalizer.normalize_sei_data(sample_sei_orderbook_message)
            
            end_time = time.time()
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
            
            assert result is not None
        
        # Statistical analysis
        avg_latency = statistics.mean(latencies)
        p95_latency = statistics.quantiles(latencies, n=20)[18]  # 95th percentile
        max_latency = max(latencies)
        
        print(f"Normalization Latency Stats:")
        print(f"  Average: {avg_latency:.2f}ms")
        print(f"  95th percentile: {p95_latency:.2f}ms")
        print(f"  Maximum: {max_latency:.2f}ms")
        
        # Assert latency targets
        assert avg_latency < 10.0, f"Average normalization latency {avg_latency:.2f}ms exceeds 10ms target"
        assert p95_latency < 20.0, f"95th percentile latency {p95_latency:.2f}ms exceeds 20ms target"
        assert max_latency < 50.0, f"Maximum latency {max_latency:.2f}ms exceeds 50ms target"
    
    @pytest.mark.asyncio
    async def test_websocket_message_processing_latency(self, sample_sei_orderbook_message):
        """Test WebSocket message processing latency."""
        message_handler = AsyncMock()
        
        with patch('flashmm.data.ingestion.websocket_client.get_config') as mock_config:
            mock_config.return_value.get.return_value = None
            
            client = SeiWebSocketClient(
                name="benchmark_client",
                url="wss://test.com/websocket",
                message_handler=message_handler
            )
        
        latencies = []
        
        # Benchmark message handling
        for _ in range(50):
            start_time = time.time()
            
            await client._handle_message(sample_sei_orderbook_message, 0.0)
            
            end_time = time.time()
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
        
        # Statistical analysis
        avg_latency = statistics.mean(latencies)
        p95_latency = statistics.quantiles(latencies, n=20)[18]
        max_latency = max(latencies)
        
        print(f"WebSocket Message Processing Latency Stats:")
        print(f"  Average: {avg_latency:.2f}ms")
        print(f"  95th percentile: {p95_latency:.2f}ms")
        print(f"  Maximum: {max_latency:.2f}ms")
        
        # Assert reasonable latency for message processing
        assert avg_latency < 5.0, f"Average message processing latency {avg_latency:.2f}ms exceeds 5ms target"
        assert p95_latency < 10.0, f"95th percentile latency {p95_latency:.2f}ms exceeds 10ms target"
    
    @pytest.mark.asyncio
    async def test_end_to_end_pipeline_latency(self, sample_sei_orderbook_message, sample_sei_trade_message):
        """Test end-to-end pipeline latency from WebSocket to storage."""
        # Mock storage clients for speed
        mock_redis_client = AsyncMock()
        mock_redis_client.set_orderbook = AsyncMock()
        mock_redis_client.set = AsyncMock()
        mock_redis_client.publish = AsyncMock()
        
        mock_influxdb_client = AsyncMock()
        mock_influxdb_client.write_orderbook_snapshot = AsyncMock()
        mock_influxdb_client.write_trade = AsyncMock()
        
        with patch('flashmm.data.market_data_service.HighPerformanceRedisClient', return_value=mock_redis_client), \
             patch('flashmm.data.market_data_service.HighPerformanceInfluxDBClient', return_value=mock_influxdb_client), \
             patch('flashmm.data.market_data_service.EnhancedFeedManager') as mock_feed_manager_class, \
             patch('flashmm.data.market_data_service.get_config') as mock_config:
            
            mock_feed_manager = AsyncMock()
            mock_feed_manager_class.return_value = mock_feed_manager
            
            # Mock configuration
            mock_config.return_value.get.side_effect = lambda key, default=None: {
                "sei.ws_url": "wss://test.sei.com/websocket",
                "redis_url": "redis://localhost:6379",
                "influxdb_url": "http://localhost:8086",
                "trading.symbols": ["SEI/USDC"]
            }.get(key, default)
            
            service = MarketDataService()
            await service.initialize()
            
            # Test order book processing latency
            orderbook_latencies = []
            normalizer = SeiDataNormalizer()
            await normalizer.initialize()
            
            for _ in range(50):
                start_time = time.time()
                
                # Simulate complete pipeline: normalize + process + store
                normalized_data = await normalizer.normalize_sei_data(sample_sei_orderbook_message)
                await service._process_market_data("SEI/USDC", normalized_data)
                
                end_time = time.time()
                latency_ms = (end_time - start_time) * 1000
                orderbook_latencies.append(latency_ms)
            
            # Test trade processing latency
            trade_latencies = []
            
            for _ in range(50):
                start_time = time.time()
                
                normalized_data = await normalizer.normalize_sei_data(sample_sei_trade_message)
                await service._process_market_data("SEI/USDC", normalized_data)
                
                end_time = time.time()
                latency_ms = (end_time - start_time) * 1000
                trade_latencies.append(latency_ms)
            
            # Analyze order book latencies
            ob_avg = statistics.mean(orderbook_latencies)
            ob_p95 = statistics.quantiles(orderbook_latencies, n=20)[18]
            ob_max = max(orderbook_latencies)
            
            print(f"End-to-End OrderBook Processing Latency:")
            print(f"  Average: {ob_avg:.2f}ms")
            print(f"  95th percentile: {ob_p95:.2f}ms")
            print(f"  Maximum: {ob_max:.2f}ms")
            
            # Analyze trade latencies
            trade_avg = statistics.mean(trade_latencies)
            trade_p95 = statistics.quantiles(trade_latencies, n=20)[18]
            trade_max = max(trade_latencies)
            
            print(f"End-to-End Trade Processing Latency:")
            print(f"  Average: {trade_avg:.2f}ms")
            print(f"  95th percentile: {trade_p95:.2f}ms")
            print(f"  Maximum: {trade_max:.2f}ms")
            
            # Assert latency targets (250ms end-to-end target)
            assert ob_avg < 100.0, f"Average orderbook processing {ob_avg:.2f}ms exceeds 100ms target"
            assert ob_p95 < 200.0, f"95th percentile orderbook processing {ob_p95:.2f}ms exceeds 200ms target"
            assert ob_max < 250.0, f"Maximum orderbook processing {ob_max:.2f}ms exceeds 250ms target"
            
            assert trade_avg < 50.0, f"Average trade processing {trade_avg:.2f}ms exceeds 50ms target"
            assert trade_p95 < 100.0, f"95th percentile trade processing {trade_p95:.2f}ms exceeds 100ms target"
            assert trade_max < 150.0, f"Maximum trade processing {trade_max:.2f}ms exceeds 150ms target"
    
    @pytest.mark.asyncio
    async def test_throughput_performance(self, sample_sei_orderbook_message):
        """Test system throughput under load."""
        normalizer = SeiDataNormalizer()
        await normalizer.initialize()
        
        # Test message throughput
        message_count = 1000
        start_time = time.time()
        
        # Process messages concurrently
        tasks = []
        for _ in range(message_count):
            task = asyncio.create_task(normalizer.normalize_sei_data(sample_sei_orderbook_message))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        end_time = time.time()
        duration = end_time - start_time
        throughput = message_count / duration
        
        print(f"Throughput Performance:")
        print(f"  Processed {message_count} messages in {duration:.2f}s")
        print(f"  Throughput: {throughput:.0f} messages/second")
        
        # Verify all messages were processed
        successful_results = [r for r in results if r is not None]
        success_rate = len(successful_results) / len(results) * 100
        
        print(f"  Success rate: {success_rate:.1f}%")
        
        # Assert performance targets
        assert throughput >= 1000, f"Throughput {throughput:.0f} msg/s is below 1000 msg/s target"
        assert success_rate >= 99.0, f"Success rate {success_rate:.1f}% is below 99% target"
    
    @pytest.mark.asyncio
    async def test_memory_efficiency(self, sample_sei_orderbook_message):
        """Test memory usage doesn't grow excessively under load."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        normalizer = SeiDataNormalizer()
        await normalizer.initialize()
        
        # Measure initial memory
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process many messages
        for batch in range(10):  # 10 batches of 100 messages each
            tasks = []
            for _ in range(100):
                task = asyncio.create_task(normalizer.normalize_sei_data(sample_sei_orderbook_message))
                tasks.append(task)
            
            await asyncio.gather(*tasks)
            
            # Force garbage collection
            import gc
            gc.collect()
        
        # Measure final memory
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_growth = final_memory - initial_memory
        
        print(f"Memory Usage:")
        print(f"  Initial: {initial_memory:.1f} MB")
        print(f"  Final: {final_memory:.1f} MB")
        print(f"  Growth: {memory_growth:.1f} MB")
        
        # Assert reasonable memory growth (should be minimal for stateless processing)
        assert memory_growth < 50.0, f"Memory growth {memory_growth:.1f} MB exceeds 50 MB limit"
    
    @pytest.mark.asyncio
    async def test_concurrent_symbol_processing(self):
        """Test processing multiple symbols concurrently."""
        symbols = ["SEI/USDC", "ETH/USDC", "BTC/USDC", "ATOM/USDC", "OSMO/USDC"]
        
        normalizer = SeiDataNormalizer()
        await normalizer.initialize()
        
        # Create messages for different symbols
        messages = []
        for symbol in symbols:
            message = {
                "jsonrpc": "2.0",
                "id": f"sub_{symbol.replace('/', '_')}",
                "result": {
                    "events": [
                        {
                            "type": "OrderBookUpdate",
                            "attributes": [
                                {"key": "market_id", "value": symbol},
                                {"key": "sequence", "value": "12345"},
                                {"key": "timestamp", "value": "2024-01-01T00:00:00Z"}
                            ],
                            "data": {
                                "bids": [["0.042", "1000"]],
                                "asks": [["0.043", "1000"]]
                            }
                        }
                    ]
                }
            }
            messages.append(message)
        
        # Test concurrent processing
        iterations = 100
        latencies = []
        
        for _ in range(iterations):
            start_time = time.time()
            
            # Process all symbols concurrently
            tasks = [normalizer.normalize_sei_data(msg) for msg in messages]
            results = await asyncio.gather(*tasks)
            
            end_time = time.time()
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
            
            # Verify all processed successfully
            assert all(r is not None for r in results)
            assert len(set(r["symbol"] for r in results)) == len(symbols)
        
        # Analyze performance
        avg_latency = statistics.mean(latencies)
        p95_latency = statistics.quantiles(latencies, n=20)[18]
        
        print(f"Concurrent Symbol Processing ({len(symbols)} symbols):")
        print(f"  Average latency: {avg_latency:.2f}ms")
        print(f"  95th percentile: {p95_latency:.2f}ms")
        
        # Should be able to process multiple symbols within latency budget
        assert avg_latency < 50.0, f"Average concurrent processing {avg_latency:.2f}ms exceeds 50ms target"
        assert p95_latency < 100.0, f"95th percentile concurrent processing {p95_latency:.2f}ms exceeds 100ms target"
    
    @pytest.mark.asyncio
    async def test_error_handling_performance_impact(self, sample_sei_orderbook_message):
        """Test that error handling doesn't significantly impact performance."""
        normalizer = SeiDataNormalizer()
        await normalizer.initialize()
        
        # Mix of valid and invalid messages
        valid_message = sample_sei_orderbook_message
        invalid_messages = [
            {"invalid": "format"},
            {"jsonrpc": "2.0", "error": {"code": -1, "message": "Error"}},
            None,
            "string_message"
        ]
        
        # Test with mixed valid/invalid messages
        message_count = 500
        latencies = []
        
        for i in range(message_count):
            start_time = time.time()
            
            # Alternate between valid and invalid messages
            if i % 5 == 0:  # 20% invalid messages
                message = invalid_messages[i % len(invalid_messages)]
            else:
                message = valid_message
            
            try:
                result = await normalizer.normalize_sei_data(message)
            except:
                result = None
            
            end_time = time.time()
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
        
        # Analyze performance with error handling
        avg_latency = statistics.mean(latencies)
        p95_latency = statistics.quantiles(latencies, n=20)[18]
        
        print(f"Error Handling Performance Impact:")
        print(f"  Average latency with 20% errors: {avg_latency:.2f}ms")
        print(f"  95th percentile: {p95_latency:.2f}ms")
        
        # Error handling shouldn't significantly degrade performance
        assert avg_latency < 15.0, f"Average latency with errors {avg_latency:.2f}ms exceeds 15ms target"
        assert p95_latency < 30.0, f"95th percentile with errors {p95_latency:.2f}ms exceeds 30ms target"


class TestScalabilityBenchmarks:
    """Test scalability of the data pipeline."""
    
    @pytest.mark.asyncio
    async def test_connection_scaling(self):
        """Test performance with multiple WebSocket connections."""
        connection_counts = [1, 5, 10, 20]
        
        for conn_count in connection_counts:
            message_handler = AsyncMock()
            
            clients = []
            with patch('flashmm.data.ingestion.websocket_client.get_config') as mock_config:
                mock_config.return_value.get.return_value = None
                
                for i in range(conn_count):
                    client = SeiWebSocketClient(
                        name=f"client_{i}",
                        url=f"wss://test{i}.com/websocket",
                        message_handler=message_handler
                    )
                    clients.append(client)
            
            # Test message processing across all clients
            test_message = {"test": "message", "client_id": "benchmark"}
            
            start_time = time.time()
            
            # Simulate concurrent message processing
            tasks = []
            for client in clients:
                for _ in range(10):  # 10 messages per client
                    task = asyncio.create_task(client._handle_message(test_message, 0.0))
                    tasks.append(task)
            
            await asyncio.gather(*tasks)
            
            end_time = time.time()
            duration = end_time - start_time
            total_messages = conn_count * 10
            throughput = total_messages / duration
            
            print(f"Connection Scaling Test ({conn_count} connections):")
            print(f"  Processed {total_messages} messages in {duration:.3f}s")
            print(f"  Throughput: {throughput:.0f} messages/second")
            
            # Throughput should remain reasonable even with many connections
            assert throughput >= 500, f"Throughput {throughput:.0f} with {conn_count} connections is too low"
    
    @pytest.mark.asyncio
    async def test_subscription_scaling(self):
        """Test performance with many subscriptions per client."""
        subscription_counts = [1, 10, 50, 100]
        
        for sub_count in subscription_counts:
            message_handler = AsyncMock()
            
            with patch('flashmm.data.ingestion.websocket_client.get_config') as mock_config:
                mock_config.return_value.get.return_value = None
                
                client = SeiWebSocketClient(
                    name="scaling_test_client",
                    url="wss://test.com/websocket",
                    message_handler=message_handler
                )
            
            # Simulate many subscriptions
            mock_websocket = AsyncMock()
            client.websocket = mock_websocket
            client.connected = True
            
            start_time = time.time()
            
            # Create subscriptions
            subscription_tasks = []
            for i in range(sub_count):
                if i % 2 == 0:
                    task = client.subscribe_orderbook(f"SYMBOL{i}/USDC")
                else:
                    task = client.subscribe_trades(f"SYMBOL{i}/USDC")
                subscription_tasks.append(task)
            
            subscription_ids = await asyncio.gather(*subscription_tasks)
            
            end_time = time.time()
            duration = end_time - start_time
            
            print(f"Subscription Scaling Test ({sub_count} subscriptions):")
            print(f"  Created {len(subscription_ids)} subscriptions in {duration:.3f}s")
            print(f"  Rate: {len(subscription_ids) / duration:.0f} subscriptions/second")
            
            # Should be able to handle many subscriptions efficiently
            assert len(subscription_ids) == sub_count
            assert duration < sub_count * 0.01, f"Subscription creation too slow: {duration:.3f}s for {sub_count} subs"


if __name__ == "__main__":
    # Run specific benchmark
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "latency":
        pytest.main([__file__ + "::TestLatencyBenchmarks::test_end_to_end_pipeline_latency", "-v", "-s"])
    else:
        pytest.main([__file__, "-v", "-s"])
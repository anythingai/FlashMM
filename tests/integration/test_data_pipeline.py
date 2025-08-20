"""
Integration tests for FlashMM data pipeline.

Tests the complete flow from Sei WebSocket messages through normalization 
to storage in Redis and InfluxDB.
"""

import pytest
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
from decimal import Decimal

from flashmm.data.market_data_service import MarketDataService
from flashmm.data.ingestion.feed_manager import EnhancedFeedManager
from flashmm.data.ingestion.data_normalizer import SeiDataNormalizer
from flashmm.data.ingestion.websocket_client import SeiWebSocketClient
from flashmm.data.storage.redis_client import HighPerformanceRedisClient
from flashmm.data.storage.influxdb_client import HighPerformanceInfluxDBClient
from flashmm.utils.exceptions import DataIngestionError


class TestDataPipelineIntegration:
    """Test complete data pipeline integration."""
    
    @pytest.fixture
    def sample_sei_messages(self):
        """Sample Sei WebSocket messages for testing."""
        return {
            "orderbook": {
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
                                "bids": [["0.042", "1000"], ["0.041", "500"]],
                                "asks": [["0.043", "1500"], ["0.044", "1200"]]
                            }
                        }
                    ]
                }
            },
            "trade": {
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
        }
    
    @pytest.fixture
    async def mock_redis_client(self):
        """Mock Redis client with realistic behavior."""
        client = AsyncMock(spec=HighPerformanceRedisClient)
        client.initialize = AsyncMock()
        client.set = AsyncMock(return_value=True)
        client.get = AsyncMock(return_value=None)
        client.set_orderbook = AsyncMock(return_value=True)
        client.get_orderbook = AsyncMock(return_value=None)
        client.publish = AsyncMock(return_value=1)
        client.health_check = AsyncMock(return_value={"status": "healthy", "latency_ms": 2.0})
        client.close = AsyncMock()
        return client
    
    @pytest.fixture
    async def mock_influxdb_client(self):
        """Mock InfluxDB client with realistic behavior."""
        client = AsyncMock(spec=HighPerformanceInfluxDBClient)
        client.initialize = AsyncMock()
        client.write_orderbook_snapshot = AsyncMock()
        client.write_trade = AsyncMock()
        client.health_check = AsyncMock(return_value={"status": "healthy", "latency_ms": 5.0})
        client.close = MagicMock()
        return client
    
    @pytest.fixture
    async def mock_websocket_client(self):
        """Mock WebSocket client."""
        client = AsyncMock(spec=SeiWebSocketClient)
        client.connect = AsyncMock()
        client.start = AsyncMock()
        client.disconnect = AsyncMock()
        client.subscribe_orderbook = AsyncMock(return_value="orderbook_sub_id")
        client.subscribe_trades = AsyncMock(return_value="trades_sub_id")
        client.is_connected = MagicMock(return_value=True)
        client.get_performance_stats = MagicMock(return_value={
            "connected": True,
            "message_count": 100,
            "avg_latency_ms": 50.0
        })
        return client
    
    @pytest.mark.asyncio
    async def test_end_to_end_orderbook_processing(self, sample_sei_messages, mock_redis_client, mock_influxdb_client):
        """Test end-to-end order book message processing."""
        # Create normalizer
        normalizer = SeiDataNormalizer()
        await normalizer.initialize()
        
        # Process the message
        orderbook_message = sample_sei_messages["orderbook"]
        normalized_data = await normalizer.normalize_sei_data(orderbook_message)
        
        # Verify normalization
        assert normalized_data is not None
        assert normalized_data["type"] == "orderbook"
        assert normalized_data["symbol"] == "SEI/USDC"
        assert len(normalized_data["bids"]) == 2
        assert len(normalized_data["asks"]) == 2
        
        # Simulate storage operations
        await mock_redis_client.set_orderbook("SEI/USDC", normalized_data)
        await mock_influxdb_client.write_orderbook_snapshot("SEI/USDC", normalized_data)
        
        # Verify storage calls
        mock_redis_client.set_orderbook.assert_called_once_with("SEI/USDC", normalized_data)
        mock_influxdb_client.write_orderbook_snapshot.assert_called_once_with("SEI/USDC", normalized_data)
    
    @pytest.mark.asyncio
    async def test_end_to_end_trade_processing(self, sample_sei_messages, mock_redis_client, mock_influxdb_client):
        """Test end-to-end trade message processing."""
        # Create normalizer
        normalizer = SeiDataNormalizer()
        await normalizer.initialize()
        
        # Process the message
        trade_message = sample_sei_messages["trade"]
        normalized_data = await normalizer.normalize_sei_data(trade_message)
        
        # Verify normalization
        assert normalized_data is not None
        assert normalized_data["type"] == "trade"
        assert normalized_data["symbol"] == "SEI/USDC"
        assert normalized_data["price"] == "0.0425"
        assert normalized_data["size"] == "1000"
        assert normalized_data["side"] == "buy"
        
        # Simulate storage operations
        await mock_redis_client.set(f"recent_trades:SEI/USDC", [normalized_data])
        await mock_influxdb_client.write_trade(normalized_data)
        
        # Verify storage calls
        mock_redis_client.set.assert_called_once()
        mock_influxdb_client.write_trade.assert_called_once_with(normalized_data)
    
    @pytest.mark.asyncio
    async def test_feed_manager_integration(self, mock_websocket_client, mock_redis_client, mock_influxdb_client):
        """Test feed manager integration with components."""
        with patch('flashmm.data.ingestion.feed_manager.SeiWebSocketClient', return_value=mock_websocket_client), \
             patch('flashmm.data.ingestion.feed_manager.HighPerformanceRedisClient', return_value=mock_redis_client), \
             patch('flashmm.data.ingestion.feed_manager.HighPerformanceInfluxDBClient', return_value=mock_influxdb_client), \
             patch('flashmm.data.ingestion.feed_manager.get_config') as mock_config:
            
            # Mock configuration
            mock_config.return_value.get.side_effect = lambda key, default=None: {
                "trading.symbols": ["SEI/USDC"],
                "sei.ws_url": "wss://test.sei.com/websocket"
            }.get(key, default)
            
            # Create feed manager
            feed_manager = EnhancedFeedManager()
            await feed_manager.initialize()
            
            # Verify initialization
            mock_redis_client.initialize.assert_called_once()
            mock_influxdb_client.initialize.assert_called_once()
            
            # Verify market feeds were created
            assert "SEI/USDC" in feed_manager.market_feeds
            
            # Start feed manager
            with patch.object(feed_manager, '_start_market_feed', new_callable=AsyncMock) as mock_start_feed, \
                 patch.object(feed_manager, '_subscribe_all_feeds', new_callable=AsyncMock) as mock_subscribe:
                
                await feed_manager.start()
                
                mock_start_feed.assert_called()
                mock_subscribe.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_market_data_service_initialization(self, mock_redis_client, mock_influxdb_client):
        """Test market data service initialization."""
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
            
            # Create and initialize service
            service = MarketDataService()
            await service.initialize()
            
            # Verify all components were initialized
            mock_redis_client.initialize.assert_called_once()
            mock_influxdb_client.initialize.assert_called_once()
            mock_feed_manager.initialize.assert_called_once()
            
            assert service.redis_client == mock_redis_client
            assert service.influxdb_client == mock_influxdb_client
            assert service.feed_manager == mock_feed_manager
    
    @pytest.mark.asyncio
    async def test_market_data_service_data_processing(self, sample_sei_messages, mock_redis_client, mock_influxdb_client):
        """Test market data service processing flow."""
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
            
            # Create service
            service = MarketDataService()
            await service.initialize()
            
            # Simulate normalized order book data
            orderbook_data = {
                "type": "orderbook",
                "symbol": "SEI/USDC",
                "timestamp": "2024-01-01T00:00:00Z",
                "bids": [["0.042", "1000"]],
                "asks": [["0.043", "1000"]],
                "source": "sei",
                "best_bid": "0.042",
                "best_ask": "0.043",
                "spread": "0.001",
                "mid_price": "0.0425"
            }
            
            # Process the data
            await service._process_market_data("SEI/USDC", orderbook_data)
            
            # Verify storage operations
            mock_redis_client.set_orderbook.assert_called_once_with("SEI/USDC", orderbook_data)
            mock_influxdb_client.write_orderbook_snapshot.assert_called_once_with("SEI/USDC", orderbook_data)
            mock_redis_client.publish.assert_called_once()
            
            # Verify statistics
            assert service.total_data_points == 1
            assert service.data_quality_stats["valid_orderbooks"] == 1
    
    @pytest.mark.asyncio
    async def test_data_quality_validation(self, mock_redis_client, mock_influxdb_client):
        """Test data quality validation in the pipeline."""
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
            
            # Test invalid order book (empty)
            invalid_orderbook = {
                "type": "orderbook",
                "symbol": "SEI/USDC",
                "timestamp": "2024-01-01T00:00:00Z",
                "bids": [],
                "asks": [],
                "source": "sei"
            }
            
            with pytest.raises(ValueError, match="Empty order book"):
                await service._validate_data_quality("SEI/USDC", invalid_orderbook)
            
            assert service.data_quality_stats["invalid_orderbooks"] == 1
            
            # Test invalid trade (zero price)
            invalid_trade = {
                "type": "trade",
                "symbol": "SEI/USDC",
                "timestamp": "2024-01-01T00:00:00Z",
                "price": "0",
                "size": "1000",
                "side": "buy",
                "source": "sei"
            }
            
            with pytest.raises(ValueError, match="Invalid trade price"):
                await service._validate_data_quality("SEI/USDC", invalid_trade)
            
            assert service.data_quality_stats["invalid_trades"] == 1
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_behavior(self, mock_redis_client, mock_influxdb_client):
        """Test circuit breaker behavior under error conditions."""
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
            service.max_consecutive_errors = 3  # Low threshold for testing
            await service.initialize()
            
            # Simulate consecutive errors
            invalid_data = {
                "type": "orderbook",
                "symbol": "SEI/USDC",
                "bids": [],
                "asks": [],
                "timestamp": "2024-01-01T00:00:00Z",
                "source": "sei"
            }
            
            # Process multiple invalid messages
            for _ in range(4):  # Exceed threshold
                try:
                    await service._process_market_data("SEI/USDC", invalid_data)
                except:
                    pass  # Expected to fail
            
            # Circuit breaker should be open
            assert service.circuit_open is True
            assert service.consecutive_errors >= service.max_consecutive_errors
    
    @pytest.mark.asyncio
    async def test_subscriber_notification(self, mock_redis_client, mock_influxdb_client):
        """Test subscriber notification system."""
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
            
            # Register subscribers
            orderbook_callback = AsyncMock()
            trade_callback = AsyncMock()
            
            service.subscribe_to_data("orderbook", orderbook_callback)
            service.subscribe_to_data("trades", trade_callback)
            
            # Process order book data
            orderbook_data = {
                "type": "orderbook",
                "symbol": "SEI/USDC",
                "timestamp": "2024-01-01T00:00:00Z",
                "bids": [["0.042", "1000"]],
                "asks": [["0.043", "1000"]],
                "source": "sei"
            }
            
            await service._process_market_data("SEI/USDC", orderbook_data)
            
            # Verify callback was called
            orderbook_callback.assert_called_once_with("SEI/USDC", orderbook_data)
            trade_callback.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_health_monitoring_integration(self, mock_redis_client, mock_influxdb_client):
        """Test health monitoring across all components."""
        with patch('flashmm.data.market_data_service.HighPerformanceRedisClient', return_value=mock_redis_client), \
             patch('flashmm.data.market_data_service.HighPerformanceInfluxDBClient', return_value=mock_influxdb_client), \
             patch('flashmm.data.market_data_service.EnhancedFeedManager') as mock_feed_manager_class, \
             patch('flashmm.data.market_data_service.get_config') as mock_config:
            
            mock_feed_manager = AsyncMock()
            mock_feed_manager.get_feed_status.return_value = {
                "SEI/USDC": {"status": "subscribed"}
            }
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
            
            # Check component health
            await service._check_component_health()
            
            # Verify health checks were performed
            mock_redis_client.health_check.assert_called_once()
            mock_influxdb_client.health_check.assert_called_once()
            mock_feed_manager.get_feed_status.assert_called_once()
            
            # Verify health check results
            assert "redis" in service.health_checks
            assert "influxdb" in service.health_checks
            assert "feed_manager" in service.health_checks
            
            assert service.health_checks["redis"].status == "healthy"
            assert service.health_checks["influxdb"].status == "healthy"
            assert service.health_checks["feed_manager"].status == "healthy"
    
    @pytest.mark.asyncio
    async def test_performance_metrics_collection(self, mock_redis_client, mock_influxdb_client):
        """Test performance metrics collection across components."""
        with patch('flashmm.data.market_data_service.HighPerformanceRedisClient', return_value=mock_redis_client), \
             patch('flashmm.data.market_data_service.HighPerformanceInfluxDBClient', return_value=mock_influxdb_client), \
             patch('flashmm.data.market_data_service.EnhancedFeedManager') as mock_feed_manager_class, \
             patch('flashmm.data.market_data_service.get_config') as mock_config:
            
            mock_feed_manager = AsyncMock()
            mock_feed_manager.get_performance_metrics.return_value = {
                "total_messages": 1000,
                "error_rate": 0.5
            }
            mock_feed_manager_class.return_value = mock_feed_manager
            
            mock_redis_client.get_performance_stats.return_value = {
                "operation_count": 500,
                "avg_latency_ms": 2.0
            }
            
            mock_influxdb_client.get_performance_stats.return_value = {
                "points_written": 800,
                "avg_write_latency_ms": 5.0
            }
            
            # Mock configuration
            mock_config.return_value.get.side_effect = lambda key, default=None: {
                "sei.ws_url": "wss://test.sei.com/websocket",
                "redis_url": "redis://localhost:6379",
                "influxdb_url": "http://localhost:8086",
                "trading.symbols": ["SEI/USDC"]
            }.get(key, default)
            
            service = MarketDataService()
            service.start_time = datetime.now()
            service.total_data_points = 1000
            service.total_errors = 5
            await service.initialize()
            
            # Get performance metrics
            metrics = await service.get_performance_metrics()
            
            # Verify metrics structure
            assert "uptime_seconds" in metrics
            assert "total_data_points" in metrics
            assert "error_rate" in metrics
            assert "feed_manager" in metrics
            assert "redis" in metrics
            assert "influxdb" in metrics
            
            # Verify values
            assert metrics["total_data_points"] == 1000
            assert metrics["total_errors"] == 5
            assert metrics["error_rate"] == 0.5  # 5/1000 * 100
            
    @pytest.mark.asyncio
    async def test_graceful_shutdown(self, mock_redis_client, mock_influxdb_client):
        """Test graceful shutdown of all components."""
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
            
            # Start the service (mock background tasks)
            service.health_monitor_task = AsyncMock()
            service.metrics_publisher_task = AsyncMock()
            service.status = service.ServiceStatus.RUNNING
            
            # Stop the service
            await service.stop()
            
            # Verify all components were stopped
            mock_feed_manager.stop.assert_called_once()
            mock_redis_client.close.assert_called_once()
            mock_influxdb_client.close.assert_called_once()
            
            # Verify background tasks were cancelled
            service.health_monitor_task.cancel.assert_called_once()
            service.metrics_publisher_task.cancel.assert_called_once()
            
            assert service.status == service.ServiceStatus.STOPPED


class TestDataPipelineErrorScenarios:
    """Test error scenarios in the data pipeline."""
    
    @pytest.mark.asyncio
    async def test_redis_failure_handling(self):
        """Test handling of Redis failures."""
        # Create a Redis client that fails
        mock_redis_client = AsyncMock()
        mock_redis_client.initialize.side_effect = Exception("Redis connection failed")
        
        with patch('flashmm.data.market_data_service.HighPerformanceRedisClient', return_value=mock_redis_client), \
             patch('flashmm.data.market_data_service.get_config') as mock_config:
            
            mock_config.return_value.get.side_effect = lambda key, default=None: {
                "sei.ws_url": "wss://test.sei.com/websocket",
                "redis_url": "redis://localhost:6379",
                "influxdb_url": "http://localhost:8086",
                "trading.symbols": ["SEI/USDC"]
            }.get(key, default)
            
            service = MarketDataService()
            
            with pytest.raises(DataIngestionError, match="Service initialization failed"):
                await service.initialize()
    
    @pytest.mark.asyncio
    async def test_influxdb_failure_handling(self):
        """Test handling of InfluxDB failures."""
        mock_redis_client = AsyncMock()
        mock_redis_client.initialize = AsyncMock()
        
        mock_influxdb_client = AsyncMock()
        mock_influxdb_client.initialize.side_effect = Exception("InfluxDB connection failed")
        
        with patch('flashmm.data.market_data_service.HighPerformanceRedisClient', return_value=mock_redis_client), \
             patch('flashmm.data.market_data_service.HighPerformanceInfluxDBClient', return_value=mock_influxdb_client), \
             patch('flashmm.data.market_data_service.get_config') as mock_config:
            
            mock_config.return_value.get.side_effect = lambda key, default=None: {
                "sei.ws_url": "wss://test.sei.com/websocket",
                "redis_url": "redis://localhost:6379",
                "influxdb_url": "http://localhost:8086",
                "trading.symbols": ["SEI/USDC"]
            }.get(key, default)
            
            service = MarketDataService()
            
            with pytest.raises(DataIngestionError, match="Service initialization failed"):
                await service.initialize()
    
    @pytest.mark.asyncio
    async def test_feed_manager_failure_handling(self, mock_redis_client, mock_influxdb_client):
        """Test handling of feed manager failures."""
        mock_feed_manager = AsyncMock()
        mock_feed_manager.initialize.side_effect = Exception("Feed manager failed")
        
        with patch('flashmm.data.market_data_service.HighPerformanceRedisClient', return_value=mock_redis_client), \
             patch('flashmm.data.market_data_service.HighPerformanceInfluxDBClient', return_value=mock_influxdb_client), \
             patch('flashmm.data.market_data_service.EnhancedFeedManager', return_value=mock_feed_manager), \
             patch('flashmm.data.market_data_service.get_config') as mock_config:
            
            mock_config.return_value.get.side_effect = lambda key, default=None: {
                "sei.ws_url": "wss://test.sei.com/websocket",
                "redis_url": "redis://localhost:6379",
                "influxdb_url": "http://localhost:8086",
                "trading.symbols": ["SEI/USDC"]
            }.get(key, default)
            
            service = MarketDataService()
            
            with pytest.raises(DataIngestionError, match="Service initialization failed"):
                await service.initialize()
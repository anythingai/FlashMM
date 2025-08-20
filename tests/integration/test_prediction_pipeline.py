"""
Integration tests for FlashMM Prediction Pipeline

Tests the complete end-to-end prediction workflow including
market data integration, Azure OpenAI predictions, and fallback mechanisms.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta
from decimal import Decimal

from flashmm.ml.prediction_service import PredictionService, PredictionServiceConfig
from flashmm.ml.inference.inference_engine import InferenceEngine
from flashmm.data.market_data_service import MarketDataService
from flashmm.data.storage.data_models import OrderBookSnapshot, Trade, MarketStats, OrderBookLevel, Side
from flashmm.data.storage.redis_client import RedisClient


@pytest.fixture
async def mock_market_data_service():
    """Create mock market data service."""
    service = Mock(spec=MarketDataService)
    
    # Create sample market data
    order_book = OrderBookSnapshot(
        symbol="SEI/USDC",
        timestamp=datetime.utcnow(),
        sequence=12345,
        bids=[
            OrderBookLevel(price=Decimal("1.2500"), size=Decimal("1000")),
            OrderBookLevel(price=Decimal("1.2490"), size=Decimal("800")),
        ],
        asks=[
            OrderBookLevel(price=Decimal("1.2510"), size=Decimal("900")),
            OrderBookLevel(price=Decimal("1.2520"), size=Decimal("700")),
        ],
        source="test"
    )
    
    trades = [
        Trade(
            symbol="SEI/USDC",
            timestamp=datetime.utcnow() - timedelta(seconds=i),
            price=Decimal("1.2505"),
            size=Decimal("100"),
            side=Side.BUY if i % 2 == 0 else Side.SELL,
            trade_id=f"trade_{i}",
            source="test"
        )
        for i in range(10)
    ]
    
    market_stats = MarketStats(
        symbol="SEI/USDC",
        timestamp=datetime.utcnow(),
        window_seconds=60,
        volume=Decimal("50000"),
        trade_count=100,
        vwap=Decimal("1.2505")
    )
    
    service.get_latest_orderbook = AsyncMock(return_value=order_book)
    service.get_recent_trades = AsyncMock(return_value=trades)
    service.get_market_stats = AsyncMock(return_value=market_stats)
    
    return service


@pytest.fixture
async def mock_redis_client():
    """Create mock Redis client."""
    client = Mock(spec=RedisClient)
    client.initialize = AsyncMock()
    client.cleanup = AsyncMock()
    client.set = AsyncMock()
    client.get = AsyncMock(return_value=None)
    client.publish = AsyncMock()
    client.ping = AsyncMock()
    return client


class TestPredictionPipeline:
    """Test complete prediction pipeline integration."""
    
    @pytest.mark.asyncio
    async def test_inference_engine_initialization(self):
        """Test inference engine initialization with prediction service."""
        engine = InferenceEngine()
        
        # Mock dependencies to avoid external calls
        with patch('flashmm.ml.prediction_service.get_prediction_service') as mock_get_service:
            mock_service = Mock()
            mock_service.initialize = AsyncMock()
            mock_service.ensemble_engine = Mock()
            mock_get_service.return_value = mock_service
            
            await engine.initialize()
            
            assert engine._initialized == True
            assert engine.prediction_service is not None
    
    @pytest.mark.asyncio
    async def test_end_to_end_rule_based_prediction(self, mock_market_data_service):
        """Test complete prediction pipeline using rule-based engine."""
        # Configure prediction service without Azure OpenAI
        config = PredictionServiceConfig(
            prediction_frequency_hz=1.0,  # Slow for testing
            publish_to_redis=False,  # Disable Redis for test
            enable_caching=False     # Disable caching for test
        )
        
        service = PredictionService(config)
        
        # Mock Azure OpenAI as unavailable
        with patch('flashmm.ml.prediction_service.PredictionService._azure_openai_configured', return_value=False):
            await service.initialize()
            
            # Get market data
            order_book = await mock_market_data_service.get_latest_orderbook("SEI/USDC")
            trades = await mock_market_data_service.get_recent_trades("SEI/USDC")
            market_stats = await mock_market_data_service.get_market_stats("SEI/USDC")
            
            # Update market data in service
            await service.update_market_data(
                order_book=order_book,
                trades={"SEI/USDC": trades},
                market_stats=market_stats
            )
            
            # Generate prediction
            prediction = await service.ensemble_engine.predict(
                order_book=order_book,
                recent_trades=trades,
                market_stats=market_stats
            )
            
            # Validate prediction
            assert prediction is not None
            assert prediction.direction in ['bullish', 'bearish', 'neutral']
            assert 0 <= prediction.confidence <= 1
            assert prediction.validation_passed == True
            assert prediction.symbol == "SEI/USDC"
            assert prediction.response_time_ms > 0
            
            # Clean up
            await service.stop()
    
    @pytest.mark.asyncio
    async def test_inference_engine_legacy_compatibility(self, mock_market_data_service):
        """Test inference engine legacy interface compatibility."""
        engine = InferenceEngine()
        
        # Mock prediction service
        mock_service = Mock()
        mock_service.initialize = AsyncMock()
        mock_ensemble = Mock()
        
        # Mock prediction result
        from flashmm.ml.models.prediction_models import PredictionResult, PredictionMethod, PredictionConfidence
        mock_prediction = PredictionResult(
            direction='bullish',
            confidence=0.75,
            price_change_bps=12.5,
            magnitude='medium',
            reasoning='Test prediction',
            method=PredictionMethod.RULE_BASED,
            symbol='SEI/USDC'
        )
        
        mock_ensemble.predict = AsyncMock(return_value=mock_prediction)
        mock_service.ensemble_engine = mock_ensemble
        
        with patch('flashmm.ml.inference.inference_engine.get_prediction_service', return_value=mock_service):
            await engine.initialize()
            
            # Test legacy prediction interface
            market_data = {
                'order_book': {
                    'symbol': 'SEI/USDC',
                    'timestamp': datetime.utcnow().isoformat(),
                    'bids': [['1.2500', '1000'], ['1.2490', '800']],
                    'asks': [['1.2510', '900'], ['1.2520', '700']],
                    'source': 'test'
                },
                'recent_trades': [
                    {
                        'symbol': 'SEI/USDC',
                        'timestamp': datetime.utcnow().isoformat(),
                        'price': '1.2505',
                        'size': '100',
                        'side': 'buy',
                        'trade_id': 'trade_1'
                    }
                ],
                'symbol': 'SEI/USDC'
            }
            
            legacy_prediction = await engine.predict(market_data)
            
            # Validate legacy format
            assert legacy_prediction is not None
            assert 'price_prediction' in legacy_prediction
            assert 'confidence' in legacy_prediction
            assert 'signal_strength' in legacy_prediction
            assert 'direction' in legacy_prediction
            assert legacy_prediction['confidence'] == 0.75
            assert legacy_prediction['direction'] == 'bullish'
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_integration(self):
        """Test circuit breaker integration in prediction pipeline."""
        config = PredictionServiceConfig(
            prediction_frequency_hz=10.0,  # Fast for testing failures
            publish_to_redis=False,
            enable_caching=False
        )
        
        service = PredictionService(config)
        
        # Mock failing Azure OpenAI client
        mock_azure_client = Mock()
        mock_azure_client.initialize = AsyncMock()
        mock_azure_client.predict_market_direction = AsyncMock(side_effect=Exception("API Error"))
        
        with patch('flashmm.ml.clients.azure_openai_client.AzureOpenAIClient', return_value=mock_azure_client):
            with patch('flashmm.ml.prediction_service.PredictionService._azure_openai_configured', return_value=True):
                await service.initialize()
                
                # Create test market data
                order_book = OrderBookSnapshot(
                    symbol="SEI/USDC",
                    timestamp=datetime.utcnow(),
                    sequence=1,
                    bids=[OrderBookLevel(price=Decimal("1.25"), size=Decimal("1000"))],
                    asks=[OrderBookLevel(price=Decimal("1.26"), size=Decimal("1000"))],
                    source="test"
                )
                
                trades = [
                    Trade(
                        symbol="SEI/USDC",
                        timestamp=datetime.utcnow(),
                        price=Decimal("1.255"),
                        size=Decimal("100"),
                        side=Side.BUY,
                        trade_id="test_trade",
                        source="test"
                    )
                ]
                
                # Generate multiple predictions to trigger circuit breaker
                predictions = []
                for i in range(10):
                    try:
                        prediction = await service.ensemble_engine.predict(
                            order_book=order_book,
                            recent_trades=trades
                        )
                        predictions.append(prediction)
                    except Exception:
                        continue
                
                # Should have fallback predictions
                assert len(predictions) > 0
                
                # Check circuit breaker state
                cb_status = await service.ensemble_engine.circuit_breaker.get_status()
                assert cb_status['state'] in ['open', 'half_open']
                
                # Later predictions should use fallback
                final_prediction = predictions[-1]
                assert final_prediction.api_success == False
                
                await service.stop()
    
    @pytest.mark.asyncio
    async def test_prediction_caching(self):
        """Test prediction caching mechanism."""
        config = PredictionServiceConfig(
            enable_caching=True,
            cache_ttl_seconds=60
        )
        
        service = PredictionService(config)
        
        # Mock components to avoid external dependencies
        with patch('flashmm.ml.prediction_service.PredictionService._azure_openai_configured', return_value=False):
            await service.initialize()
            
            # Create test data
            order_book = OrderBookSnapshot(
                symbol="SEI/USDC",
                timestamp=datetime.utcnow(),
                sequence=1,
                bids=[OrderBookLevel(price=Decimal("1.25"), size=Decimal("1000"))],
                asks=[OrderBookLevel(price=Decimal("1.26"), size=Decimal("1000"))],
                source="test"
            )
            
            trades = []
            
            # First prediction (should miss cache)
            prediction1 = await service.ensemble_engine.predict(
                order_book=order_book,
                recent_trades=trades
            )
            
            assert prediction1.cache_hit == False
            
            # Second prediction with same data (should hit cache)
            prediction2 = await service.ensemble_engine.predict(
                order_book=order_book,
                recent_trades=trades
            )
            
            assert prediction2.cache_hit == True
            
            await service.stop()
    
    @pytest.mark.asyncio
    async def test_performance_monitoring(self):
        """Test performance monitoring and statistics collection."""
        config = PredictionServiceConfig(
            prediction_frequency_hz=5.0,
            publish_to_redis=False
        )
        
        service = PredictionService(config)
        
        with patch('flashmm.ml.prediction_service.PredictionService._azure_openai_configured', return_value=False):
            await service.initialize()
            
            # Generate some predictions
            order_book = OrderBookSnapshot(
                symbol="SEI/USDC",
                timestamp=datetime.utcnow(),
                sequence=1,
                bids=[OrderBookLevel(price=Decimal("1.25"), size=Decimal("1000"))],
                asks=[OrderBookLevel(price=Decimal("1.26"), size=Decimal("1000"))],
                source="test"
            )
            
            for i in range(5):
                await service.ensemble_engine.predict(
                    order_book=order_book,
                    recent_trades=[]
                )
            
            # Check performance stats
            stats = await service.get_performance_stats()
            
            assert 'service' in stats
            assert 'timing' in stats
            assert 'ensemble_engine' in stats
            assert stats['service']['total_predictions'] == 5
            assert stats['service']['is_running'] == False  # Not started
            
            await service.stop()
    
    @pytest.mark.asyncio
    async def test_health_monitoring(self):
        """Test health monitoring functionality."""
        config = PredictionServiceConfig(
            health_check_interval_seconds=1,
            publish_to_redis=False
        )
        
        service = PredictionService(config)
        
        with patch('flashmm.ml.prediction_service.PredictionService._azure_openai_configured', return_value=False):
            await service.initialize()
            
            # Check initial health
            health = await service.get_health_status()
            
            assert 'status' in health
            assert 'components' in health
            assert 'service' in health['components']
            assert 'ensemble_engine' in health['components']
            assert 'timing' in health['components']
            
            # Start service briefly to test running state
            await service.start()
            await asyncio.sleep(0.1)  # Brief run
            
            health_running = await service.get_health_status()
            assert health_running['components']['service']['status'] == 'running'
            
            await service.stop()
    
    @pytest.mark.asyncio
    async def test_market_data_integration(self, mock_market_data_service, mock_redis_client):
        """Test integration with market data service."""
        config = PredictionServiceConfig(
            prediction_frequency_hz=2.0,
            publish_to_redis=True,
            market_data_timeout_ms=500.0
        )
        
        # Create prediction service with mocked dependencies
        service = PredictionService(config)
        
        with patch('flashmm.ml.prediction_service.PredictionService._azure_openai_configured', return_value=False):
            with patch.object(service, 'redis_client', mock_redis_client):
                await service.initialize()
                
                # Get market data from mock service
                order_book = await mock_market_data_service.get_latest_orderbook("SEI/USDC")
                trades = await mock_market_data_service.get_recent_trades("SEI/USDC") 
                market_stats = await mock_market_data_service.get_market_stats("SEI/USDC")
                
                # Update service with market data
                await service.update_market_data(
                    order_book=order_book,
                    trades={"SEI/USDC": trades},
                    market_stats=market_stats
                )
                
                # Check data age
                data_age = await service.market_data_buffer.get_data_age_ms("SEI/USDC")
                assert data_age < config.market_data_timeout_ms
                
                # Generate prediction with updated data
                prediction = await service.ensemble_engine.predict(
                    order_book=order_book,
                    recent_trades=trades,
                    market_stats=market_stats
                )
                
                assert prediction is not None
                assert prediction.symbol == "SEI/USDC"
                
                await service.stop()
    
    @pytest.mark.asyncio
    async def test_prediction_distribution(self, mock_redis_client):
        """Test prediction result distribution."""
        config = PredictionServiceConfig(
            publish_to_redis=True,
            redis_prediction_key="test:predictions",
            redis_channel="test:updates"
        )
        
        service = PredictionService(config)
        
        with patch('flashmm.ml.prediction_service.PredictionService._azure_openai_configured', return_value=False):
            with patch.object(service, 'redis_client', mock_redis_client):
                await service.initialize()
                
                # Add a test subscriber
                received_predictions = []
                
                def test_subscriber(prediction):
                    received_predictions.append(prediction)
                
                service.add_prediction_subscriber(test_subscriber)
                
                # Generate prediction
                order_book = OrderBookSnapshot(
                    symbol="SEI/USDC",
                    timestamp=datetime.utcnow(),
                    sequence=1,
                    bids=[OrderBookLevel(price=Decimal("1.25"), size=Decimal("1000"))],
                    asks=[OrderBookLevel(price=Decimal("1.26"), size=Decimal("1000"))],
                    source="test"
                )
                
                prediction = await service.ensemble_engine.predict(
                    order_book=order_book,
                    recent_trades=[]
                )
                
                # Distribute prediction
                await service.distributor.distribute(prediction)
                
                # Check distribution
                assert len(received_predictions) == 1
                assert received_predictions[0].symbol == "SEI/USDC"
                
                # Verify Redis calls
                mock_redis_client.set.assert_called()
                mock_redis_client.publish.assert_called()
                
                await service.stop()


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    @pytest.mark.asyncio
    async def test_stale_market_data(self):
        """Test handling of stale market data."""
        config = PredictionServiceConfig(
            market_data_timeout_ms=100.0,  # Very short timeout
            publish_to_redis=False
        )
        
        service = PredictionService(config)
        
        with patch('flashmm.ml.prediction_service.PredictionService._azure_openai_configured', return_value=False):
            await service.initialize()
            
            # Create stale order book (old timestamp)
            stale_order_book = OrderBookSnapshot(
                symbol="SEI/USDC",
                timestamp=datetime.utcnow() - timedelta(seconds=1),  # 1 second old
                sequence=1,
                bids=[OrderBookLevel(price=Decimal("1.25"), size=Decimal("1000"))],
                asks=[OrderBookLevel(price=Decimal("1.26"), size=Decimal("1000"))],
                source="test"
            )
            
            # Update with stale data
            await service.update_market_data(order_book=stale_order_book)
            
            # Wait for data to become stale
            await asyncio.sleep(0.2)  # 200ms > 100ms timeout
            
            # Try to get market data (should return None due to staleness)
            market_data = await service.market_data_buffer.get_market_data("SEI/USDC")
            assert market_data is None
            
            await service.stop()
    
    @pytest.mark.asyncio
    async def test_empty_market_data(self):
        """Test handling of empty or invalid market data."""
        service = PredictionService()
        
        with patch('flashmm.ml.prediction_service.PredictionService._azure_openai_configured', return_value=False):
            await service.initialize()
            
            # Try prediction with empty order book
            empty_order_book = OrderBookSnapshot(
                symbol="SEI/USDC",
                timestamp=datetime.utcnow(),
                sequence=1,
                bids=[],  # Empty bids
                asks=[],  # Empty asks
                source="test"
            )
            
            prediction = await service.ensemble_engine.predict(
                order_book=empty_order_book,
                recent_trades=[]
            )
            
            # Should still return a prediction (fallback)
            assert prediction is not None
            assert prediction.direction == 'neutral'  # Likely neutral for empty data
            
            await service.stop()
    
    @pytest.mark.asyncio
    async def test_prediction_validation_failure(self):
        """Test handling of prediction validation failures."""
        service = PredictionService()
        
        with patch('flashmm.ml.prediction_service.PredictionService._azure_openai_configured', return_value=False):
            await service.initialize()
            
            # Mock rule engine to return invalid prediction
            invalid_prediction = {
                'direction': 'invalid_direction',  # Invalid direction
                'confidence': 1.5,  # Invalid confidence > 1.0
                'price_change_bps': 10000,  # Unrealistic price change
                'reasoning': 'Test invalid prediction'
            }
            
            with patch.object(service.ensemble_engine.rule_engine, 'predict', return_value=invalid_prediction):
                order_book = OrderBookSnapshot(
                    symbol="SEI/USDC",
                    timestamp=datetime.utcnow(),
                    sequence=1,
                    bids=[OrderBookLevel(price=Decimal("1.25"), size=Decimal("1000"))],
                    asks=[OrderBookLevel(price=Decimal("1.26"), size=Decimal("1000"))],
                    source="test"
                )
                
                prediction = await service.ensemble_engine.predict(
                    order_book=order_book,
                    recent_trades=[]
                )
                
                # Should return safe fallback prediction
                assert prediction is not None
                assert prediction.validation_passed == False
                assert prediction.direction == 'neutral'  # Safe fallback
            
            await service.stop()


if __name__ == "__main__":
    pytest.main([__file__])
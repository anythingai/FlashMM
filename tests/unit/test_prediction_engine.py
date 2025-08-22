"""
Unit tests for FlashMM Prediction Engine components.

Tests Azure OpenAI integration, ensemble logic, circuit breaker,
and rule-based fallback functionality.
"""

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, Mock, patch

import pytest

from flashmm.data.storage.data_models import (
    MarketStats,
    OrderBookLevel,
    OrderBookSnapshot,
    Side,
    Trade,
)
from flashmm.ml.clients.azure_openai_client import AzureOpenAIClient, AzureOpenAIConfig
from flashmm.ml.fallback.rule_based_engine import RuleBasedEngine
from flashmm.ml.features.feature_extractor import FeatureExtractor
from flashmm.ml.models.prediction_models import (
    EnsemblePredictionEngine,
    PredictionCache,
    PredictionMethod,
    PredictionResult,
    PredictionValidator,
)
from flashmm.ml.prompts.market_prompts import MarketPredictionPrompt, PredictionResponseParser
from flashmm.ml.reliability.circuit_breaker import AzureOpenAICircuitBreaker, CircuitState


# Test fixtures
@pytest.fixture
def sample_order_book():
    """Create sample order book snapshot."""
    return OrderBookSnapshot(
        symbol="SEI/USDC",
        timestamp=datetime.utcnow(),
        sequence=12345,
        bids=[
            OrderBookLevel(price=Decimal("1.2500"), size=Decimal("1000")),
            OrderBookLevel(price=Decimal("1.2490"), size=Decimal("800")),
            OrderBookLevel(price=Decimal("1.2480"), size=Decimal("600"))
        ],
        asks=[
            OrderBookLevel(price=Decimal("1.2510"), size=Decimal("900")),
            OrderBookLevel(price=Decimal("1.2520"), size=Decimal("700")),
            OrderBookLevel(price=Decimal("1.2530"), size=Decimal("500"))
        ],
        source="test"
    )


@pytest.fixture
def sample_trades():
    """Create sample trades list."""
    base_time = datetime.utcnow()
    return [
        Trade(
            symbol="SEI/USDC",
            timestamp=base_time - timedelta(seconds=i),
            price=Decimal("1.2505") + Decimal(str(i * 0.0001)),
            size=Decimal("100") + Decimal(str(i * 10)),
            side=Side.BUY if i % 2 == 0 else Side.SELL,
            trade_id=f"trade_{i}",
            sequence=i,
            source="test"
        )
        for i in range(10)
    ]


@pytest.fixture
def sample_market_stats():
    """Create sample market statistics."""
    return MarketStats(
        symbol="SEI/USDC",
        timestamp=datetime.utcnow(),
        window_seconds=60,
        open_price=Decimal("1.2500"),
        high_price=Decimal("1.2520"),
        low_price=Decimal("1.2490"),
        close_price=Decimal("1.2505"),
        volume=Decimal("50000"),
        trade_count=100,
        vwap=Decimal("1.2505"),
        avg_spread=Decimal("0.0010"),
        avg_spread_bps=Decimal("8.0")
    )


class TestAzureOpenAIClient:
    """Test Azure OpenAI client functionality."""

    def test_config_initialization(self):
        """Test Azure OpenAI configuration."""
        config = AzureOpenAIConfig()
        assert config.max_tokens == 150
        assert config.temperature == 0.1
        assert config.timeout_seconds == 10
        assert config.max_retries == 3

    @pytest.mark.asyncio
    async def test_client_initialization(self):
        """Test client initialization without actual API calls."""
        config = AzureOpenAIConfig()
        config.api_key = "test-key"
        config.endpoint = "https://test.openai.azure.com/"

        client = AzureOpenAIClient(config)
        assert not client._initialized
        assert client.request_count == 0
        assert client.total_latency == 0.0

    @pytest.mark.asyncio
    async def test_prediction_with_mock_response(self, sample_order_book, sample_trades):
        """Test prediction with mocked Azure OpenAI response."""
        config = AzureOpenAIConfig()
        config.api_key = "test-key"

        client = AzureOpenAIClient(config)

        # Mock the OpenAI client
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '''
        {
            "direction": "bullish",
            "confidence": 0.75,
            "price_change_bps": 15.5,
            "magnitude": "medium",
            "reasoning": "Strong bid volume dominance"
        }
        '''

        with patch.object(client, 'client') as mock_client:
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            client._initialized = True

            prediction = await client.predict_market_direction(
                order_book=sample_order_book,
                recent_trades=sample_trades
            )

            assert prediction['direction'] == 'bullish'
            assert prediction['confidence'] == 0.75
            assert prediction['price_change_bps'] == 15.5
            assert prediction['api_success']

    def test_performance_stats(self):
        """Test performance statistics tracking."""
        client = AzureOpenAIClient()

        # Simulate some metrics
        client.request_count = 100
        client.success_count = 95
        client.error_count = 5
        client.total_latency = 15000  # 15 seconds total

        stats = client.get_performance_stats()

        assert stats['request_count'] == 100
        assert stats['success_rate'] == 95.0
        assert stats['error_rate'] == 5.0
        assert stats['average_latency_ms'] == 150.0


class TestRuleBasedEngine:
    """Test rule-based prediction engine."""

    @pytest.mark.asyncio
    async def test_basic_prediction(self, sample_order_book, sample_trades):
        """Test basic rule-based prediction."""
        engine = RuleBasedEngine()

        prediction = await engine.predict(
            order_book=sample_order_book,
            recent_trades=sample_trades
        )

        assert 'direction' in prediction
        assert 'confidence' in prediction
        assert 'price_change_bps' in prediction
        assert 'reasoning' in prediction
        assert prediction['direction'] in ['bullish', 'bearish', 'neutral']
        assert 0 <= prediction['confidence'] <= 1

    @pytest.mark.asyncio
    async def test_order_flow_analysis(self, sample_order_book):
        """Test order flow imbalance analysis."""
        engine = RuleBasedEngine()

        # Test with bid-heavy order book
        signal = await engine._analyze_order_flow(sample_order_book)

        assert 'direction' in signal
        assert 'strength' in signal
        assert 'confidence' in signal
        assert 'imbalance_ratio' in signal

    @pytest.mark.asyncio
    async def test_momentum_analysis(self, sample_trades):
        """Test price momentum analysis."""
        engine = RuleBasedEngine()

        signal = await engine._analyze_momentum(sample_trades)

        assert 'direction' in signal
        assert 'strength' in signal
        assert 'confidence' in signal

    @pytest.mark.asyncio
    async def test_volume_pressure_analysis(self, sample_trades):
        """Test volume pressure analysis."""
        engine = RuleBasedEngine()

        signal = await engine._analyze_volume_pressure(sample_trades)

        assert 'direction' in signal
        assert 'buy_ratio' in signal
        assert 'buy_volume' in signal
        assert 'sell_volume' in signal

    def test_performance_stats(self):
        """Test performance statistics."""
        engine = RuleBasedEngine()
        engine.prediction_count = 50

        stats = engine.get_performance_stats()

        assert stats['prediction_count'] == 50
        assert 'engine_version' in stats
        assert 'signal_weights' in stats


class TestCircuitBreaker:
    """Test circuit breaker functionality."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_initialization(self):
        """Test circuit breaker initialization."""
        cb = AzureOpenAICircuitBreaker()

        assert cb.state == CircuitState.CLOSED
        assert cb.name == "azure-openai"
        assert cb.consecutive_failures == 0

    @pytest.mark.asyncio
    async def test_successful_call(self):
        """Test successful function call through circuit breaker."""
        cb = AzureOpenAICircuitBreaker()

        async def mock_success():
            return {"status": "success"}

        result = await cb.call(mock_success)
        assert result["status"] == "success"
        assert cb.consecutive_failures == 0

    @pytest.mark.asyncio
    async def test_circuit_opening_on_failures(self):
        """Test circuit opening after repeated failures."""
        cb = AzureOpenAICircuitBreaker()

        async def mock_failure():
            raise Exception("API failure")

        # Trigger multiple failures to open circuit
        for _ in range(6):  # More than failure threshold
            try:
                await cb.call(mock_failure)
            except Exception as e:
                # Expected circuit breaker failures
                print(f"Circuit breaker failure: {e}")

        assert cb.state == CircuitState.OPEN
        assert cb.consecutive_failures >= cb.config.failure_threshold

    @pytest.mark.asyncio
    async def test_fallback_when_circuit_open(self, sample_order_book, sample_trades):
        """Test fallback prediction when circuit is open."""
        mock_rule_engine = Mock()
        mock_rule_engine.predict = AsyncMock(return_value={
            'direction': 'neutral',
            'confidence': 0.6,
            'price_change_bps': 0.0,
            'reasoning': 'Rule-based fallback'
        })

        cb = AzureOpenAICircuitBreaker(mock_rule_engine)

        # Force circuit open
        await cb.force_open("Test")

        # Test fallback
        result = await cb._fallback_prediction(
            sample_order_book, sample_trades
        )

        assert result['direction'] == 'neutral'
        assert not result['api_success']
        assert result['circuit_breaker_state'] == 'open'

    @pytest.mark.asyncio
    async def test_circuit_recovery(self):
        """Test circuit recovery after timeout."""
        cb = AzureOpenAICircuitBreaker()

        # Force circuit open
        await cb.force_open("Test failure")
        assert cb.state == CircuitState.OPEN

        # Simulate timeout passage
        cb.last_failure_time = datetime.utcnow() - timedelta(seconds=31)

        # Test successful call should move to half-open
        async def mock_success():
            return {"status": "success"}

        await cb.call(mock_success)
        # After successful calls, circuit should close
        for _ in range(cb.config.success_threshold):
            await cb.call(mock_success)

        assert cb.state == CircuitState.CLOSED


class TestPredictionValidator:
    """Test prediction validation."""

    @pytest.mark.asyncio
    async def test_valid_prediction(self):
        """Test validation of valid prediction."""
        validator = PredictionValidator()

        valid_prediction = {
            'direction': 'bullish',
            'confidence': 0.75,
            'price_change_bps': 12.5,
            'magnitude': 'medium'
        }

        is_valid, errors = await validator.validate(valid_prediction)

        assert is_valid
        assert len(errors) == 0

    @pytest.mark.asyncio
    async def test_invalid_direction(self):
        """Test validation with invalid direction."""
        validator = PredictionValidator()

        invalid_prediction = {
            'direction': 'invalid_direction',
            'confidence': 0.75,
            'price_change_bps': 12.5,
            'magnitude': 'medium'
        }

        is_valid, errors = await validator.validate(invalid_prediction)

        assert not is_valid
        assert any('direction' in error for error in errors)

    @pytest.mark.asyncio
    async def test_confidence_out_of_range(self):
        """Test validation with confidence out of range."""
        validator = PredictionValidator()

        invalid_prediction = {
            'direction': 'bullish',
            'confidence': 1.5,  # Invalid - above 1.0
            'price_change_bps': 12.5,
            'magnitude': 'medium'
        }

        is_valid, errors = await validator.validate(invalid_prediction)

        assert not is_valid
        assert any('confidence' in error.lower() for error in errors)

    @pytest.mark.asyncio
    async def test_consistency_validation(self):
        """Test prediction consistency validation."""
        validator = PredictionValidator()

        inconsistent_prediction = {
            'direction': 'bullish',
            'confidence': 0.75,
            'price_change_bps': -25.0,  # Inconsistent with bullish direction
            'magnitude': 'medium'
        }

        is_valid, errors = await validator.validate(inconsistent_prediction)

        assert not is_valid
        assert any('inconsistent' in error.lower() for error in errors)


class TestPredictionCache:
    """Test prediction caching functionality."""

    @pytest.mark.asyncio
    async def test_cache_set_and_get(self, sample_order_book, sample_trades):
        """Test setting and getting cached predictions."""
        cache = PredictionCache(ttl_seconds=60)

        prediction = {
            'direction': 'bullish',
            'confidence': 0.75,
            'price_change_bps': 12.5
        }

        # Set cache
        await cache.set(sample_order_book, sample_trades, prediction)

        # Get cache
        cached_prediction = await cache.get(sample_order_book, sample_trades)

        assert cached_prediction is not None
        assert cached_prediction['direction'] == 'bullish'
        assert cached_prediction['confidence'] == 0.75

    @pytest.mark.asyncio
    async def test_cache_expiration(self, sample_order_book, sample_trades):
        """Test cache expiration."""
        cache = PredictionCache(ttl_seconds=1)  # Very short TTL

        prediction = {
            'direction': 'bullish',
            'confidence': 0.75,
            'price_change_bps': 12.5
        }

        # Set cache
        await cache.set(sample_order_book, sample_trades, prediction)

        # Wait for expiration
        await asyncio.sleep(1.1)

        # Should return None (expired)
        cached_prediction = await cache.get(sample_order_book, sample_trades)
        assert cached_prediction is None

    @pytest.mark.asyncio
    async def test_cache_stats(self):
        """Test cache statistics."""
        cache = PredictionCache(ttl_seconds=60, max_size=100)

        stats = await cache.get_stats()

        assert stats['size'] == 0
        assert stats['max_size'] == 100
        assert stats['ttl_seconds'] == 60


class TestEnsemblePredictionEngine:
    """Test ensemble prediction engine."""

    @pytest.mark.asyncio
    async def test_engine_initialization(self):
        """Test ensemble engine initialization."""
        engine = EnsemblePredictionEngine()

        assert engine.rule_engine is not None
        assert engine.validator is not None
        assert engine.circuit_breaker is not None

    @pytest.mark.asyncio
    async def test_rule_based_prediction(self, sample_order_book, sample_trades):
        """Test prediction using only rule-based engine."""
        engine = EnsemblePredictionEngine(azure_client=None)

        prediction = await engine.predict(
            order_book=sample_order_book,
            recent_trades=sample_trades,
            force_method=PredictionMethod.RULE_BASED
        )

        assert isinstance(prediction, PredictionResult)
        assert prediction.method == PredictionMethod.RULE_BASED
        assert prediction.validation_passed

    @pytest.mark.asyncio
    async def test_ensemble_with_mock_azure(self, sample_order_book, sample_trades):
        """Test ensemble prediction with mocked Azure client."""
        # Mock Azure client
        mock_azure_client = Mock()
        mock_azure_client.predict_market_direction = AsyncMock(return_value={
            'direction': 'bullish',
            'confidence': 0.8,
            'price_change_bps': 20.0,
            'reasoning': 'Azure OpenAI prediction',
            'api_success': True
        })

        engine = EnsemblePredictionEngine(azure_client=mock_azure_client)

        prediction = await engine.predict(
            order_book=sample_order_book,
            recent_trades=sample_trades
        )

        assert isinstance(prediction, PredictionResult)
        assert prediction.validation_passed
        assert prediction.ensemble_agreement > 0

    @pytest.mark.asyncio
    async def test_prediction_with_caching(self, sample_order_book, sample_trades):
        """Test prediction with caching enabled."""
        engine = EnsemblePredictionEngine(enable_caching=True, cache_ttl=60)

        # First prediction
        _prediction1 = await engine.predict(
            order_book=sample_order_book,
            recent_trades=sample_trades,
            force_method=PredictionMethod.RULE_BASED
        )

        # Second prediction (should be cached)
        prediction2 = await engine.predict(
            order_book=sample_order_book,
            recent_trades=sample_trades,
            force_method=PredictionMethod.RULE_BASED
        )

        assert prediction2.cache_hit

    @pytest.mark.asyncio
    async def test_performance_stats(self):
        """Test performance statistics."""
        engine = EnsemblePredictionEngine()
        engine.prediction_count = 10
        engine.validation_failures = 1

        stats = await engine.get_performance_stats()

        assert 'prediction_count' in stats
        assert 'method_counts' in stats
        assert 'validation_failures' in stats
        assert stats['prediction_count'] == 10
        assert stats['validation_failures'] == 1

    @pytest.mark.asyncio
    async def test_health_check(self):
        """Test health check functionality."""
        engine = EnsemblePredictionEngine()

        health = await engine.health_check()

        assert 'status' in health
        assert 'components' in health
        assert 'rule_engine' in health['components']


class TestFeatureExtractor:
    """Test feature extraction functionality."""

    @pytest.mark.asyncio
    async def test_book_features(self, sample_order_book):
        """Test order book feature extraction."""
        extractor = FeatureExtractor()

        features = await extractor._extract_book_features(sample_order_book)

        assert 'book_best_bid' in features
        assert 'book_best_ask' in features
        assert 'book_mid_price' in features
        assert 'book_spread_bps' in features
        assert 'book_depth_imbalance' in features

        # Validate feature values
        assert features['book_best_bid'] > 0
        assert features['book_best_ask'] > features['book_best_bid']
        assert -1 <= features['book_depth_imbalance'] <= 1

    @pytest.mark.asyncio
    async def test_trade_features(self, sample_trades):
        """Test trade-based feature extraction."""
        extractor = FeatureExtractor()

        features = await extractor._extract_trade_features(sample_trades)

        assert 'trade_total_volume' in features
        assert 'trade_buy_ratio' in features
        assert 'trade_sell_ratio' in features
        assert 'trade_vwap' in features

        # Validate feature values
        assert features['trade_total_volume'] > 0
        assert 0 <= features['trade_buy_ratio'] <= 1
        assert 0 <= features['trade_sell_ratio'] <= 1
        assert abs(features['trade_buy_ratio'] + features['trade_sell_ratio'] - 1.0) < 0.01

    @pytest.mark.asyncio
    async def test_comprehensive_feature_extraction(self, sample_order_book, sample_trades, sample_market_stats):
        """Test comprehensive feature extraction."""
        extractor = FeatureExtractor()

        features = await extractor.extract_features(
            current_book=sample_order_book,
            recent_trades=sample_trades,
            market_stats=sample_market_stats
        )

        assert 'feature_timestamp' in features
        assert 'symbol' in features
        assert 'feature_count' in features
        assert features['feature_count'] > 0

        # Check that we have features from different categories
        book_features = [k for k in features.keys() if k.startswith('book_')]
        trade_features = [k for k in features.keys() if k.startswith('trade_')]
        regime_features = [k for k in features.keys() if k.startswith('regime_')]

        assert len(book_features) > 0
        assert len(trade_features) > 0
        assert len(regime_features) > 0

    def test_feature_names(self):
        """Test feature name enumeration."""
        extractor = FeatureExtractor()

        feature_names = extractor.get_feature_names()

        assert len(feature_names) > 0
        assert 'book_mid_price' in feature_names
        assert 'trade_total_volume' in feature_names
        assert 'micro_level_1_imbalance' in feature_names


class TestPromptEngineering:
    """Test prompt engineering components."""

    def test_market_prompt_building(self, sample_order_book, sample_trades):
        """Test market prompt building."""
        prompt_builder = MarketPredictionPrompt()

        prompt = prompt_builder.build_market_prompt(
            order_book=sample_order_book,
            recent_trades=sample_trades,
            prediction_horizon_ms=200
        )

        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert "SEI/USDC" in prompt
        assert "200ms" in prompt
        assert "ORDER BOOK" in prompt
        assert "RECENT TRADES" in prompt

    def test_system_prompt(self):
        """Test system prompt content."""
        prompt_builder = MarketPredictionPrompt()

        system_prompt = prompt_builder.SYSTEM_PROMPT

        assert "JSON" in system_prompt
        assert "direction" in system_prompt
        assert "confidence" in system_prompt
        assert "price_change_bps" in system_prompt

    @pytest.mark.asyncio
    async def test_response_parsing_valid(self):
        """Test parsing valid response."""
        parser = PredictionResponseParser()

        valid_response = '''
        {
            "direction": "bullish",
            "confidence": 0.75,
            "price_change_bps": 15.5,
            "magnitude": "medium",
            "reasoning": "Strong buying pressure"
        }
        '''

        prediction = await parser.parse_prediction(valid_response)

        assert prediction['direction'] == 'bullish'
        assert prediction['confidence'] == 0.75
        assert prediction['price_change_bps'] == 15.5

    @pytest.mark.asyncio
    async def test_response_parsing_invalid(self):
        """Test parsing invalid response."""
        parser = PredictionResponseParser()

        invalid_response = "This is not valid JSON"

        prediction = await parser.parse_prediction(invalid_response)

        # Should return fallback prediction
        assert prediction['direction'] == 'neutral'
        assert prediction['confidence'] == 0.5
        assert 'fallback' in prediction['model_version']


# Integration test
class TestIntegration:
    """Integration tests for the complete prediction pipeline."""

    @pytest.mark.asyncio
    async def test_end_to_end_rule_based_prediction(self, sample_order_book, sample_trades, sample_market_stats):
        """Test complete end-to-end prediction using rule-based engine."""
        # Initialize ensemble engine with rule-based only
        engine = EnsemblePredictionEngine(azure_client=None)

        # Generate prediction
        prediction = await engine.predict(
            order_book=sample_order_book,
            recent_trades=sample_trades,
            market_stats=sample_market_stats
        )

        # Validate prediction result
        assert isinstance(prediction, PredictionResult)
        assert prediction.direction in ['bullish', 'bearish', 'neutral']
        assert 0 <= prediction.confidence <= 1
        assert prediction.validation_passed
        assert prediction.symbol == "SEI/USDC"
        assert prediction.response_time_ms > 0

        # Test actionability
        if prediction.confidence >= 0.6 and prediction.direction != 'neutral':
            assert prediction.is_actionable()

    @pytest.mark.asyncio
    async def test_circuit_breaker_integration(self, sample_order_book, sample_trades):
        """Test circuit breaker integration with ensemble engine."""
        # Create engine with mocked failing Azure client
        mock_azure_client = Mock()
        mock_azure_client.predict_market_direction = AsyncMock(side_effect=Exception("API Error"))

        engine = EnsemblePredictionEngine(azure_client=mock_azure_client)

        # Multiple predictions should trigger circuit breaker
        predictions = []
        for _ in range(10):
            prediction = await engine.predict(
                order_book=sample_order_book,
                recent_trades=sample_trades
            )
            predictions.append(prediction)

        # Should have fallback predictions
        assert all(isinstance(p, PredictionResult) for p in predictions)

        # Later predictions should show circuit breaker was triggered
        circuit_status = await engine.circuit_breaker.get_status()
        assert circuit_status['state'] in ['open', 'half_open']


if __name__ == "__main__":
    pytest.main([__file__])

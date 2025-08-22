"""
FlashMM Prediction Models

Ensemble prediction system combining Azure OpenAI with rule-based fallback,
including prediction validation, confidence scoring, and uncertainty quantification.
"""

import asyncio
import hashlib
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np

from flashmm.data.storage.data_models import MarketStats, OrderBookSnapshot, Trade
from flashmm.ml.clients.azure_openai_client import AzureOpenAIClient
from flashmm.ml.fallback.rule_based_engine import RuleBasedEngine
from flashmm.ml.reliability.circuit_breaker import AzureOpenAICircuitBreaker
from flashmm.utils.logging import get_logger

logger = get_logger(__name__)


class PredictionMethod(Enum):
    """Prediction method enumeration."""
    AZURE_OPENAI = "azure_openai"
    RULE_BASED = "rule_based"
    ENSEMBLE = "ensemble"
    FALLBACK = "fallback"


class PredictionConfidence(Enum):
    """Prediction confidence levels."""
    HIGH = "high"       # > 0.75
    MEDIUM = "medium"   # 0.6 - 0.75
    LOW = "low"         # 0.45 - 0.6
    VERY_LOW = "very_low"  # < 0.45


class PredictionResult:
    """Structured prediction result with metadata."""

    def __init__(self,
                 direction: str,
                 confidence: float,
                 price_change_bps: float,
                 magnitude: str = "medium",
                 reasoning: str = "",
                 key_factors: list[str] | None = None,
                 method: PredictionMethod = PredictionMethod.ENSEMBLE,
                 model_version: str = "ensemble-v1.0",
                 symbol: str = "",
                 response_time_ms: float = 0.0,
                 api_success: bool = True,
                 ensemble_agreement: float = 1.0,
                 uncertainty_score: float = 0.0,
                 validation_passed: bool = True,
                 cache_hit: bool = False):
        """Initialize prediction result."""
        self.direction = direction
        self.confidence = confidence
        self.price_change_bps = price_change_bps
        self.magnitude = magnitude
        self.reasoning = reasoning
        self.key_factors = key_factors or []
        self.method = method
        self.model_version = model_version
        self.symbol = symbol
        self.response_time_ms = response_time_ms
        self.api_success = api_success
        self.ensemble_agreement = ensemble_agreement
        self.uncertainty_score = uncertainty_score
        self.validation_passed = validation_passed
        self.cache_hit = cache_hit
        self.timestamp = datetime.utcnow()

        # Calculate confidence level
        self.confidence_level = self._calculate_confidence_level()

    def _calculate_confidence_level(self) -> PredictionConfidence:
        """Calculate confidence level enum."""
        if self.confidence > 0.75:
            return PredictionConfidence.HIGH
        elif self.confidence > 0.6:
            return PredictionConfidence.MEDIUM
        elif self.confidence > 0.45:
            return PredictionConfidence.LOW
        else:
            return PredictionConfidence.VERY_LOW

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'direction': self.direction,
            'confidence': self.confidence,
            'confidence_level': self.confidence_level.value,
            'price_change_bps': self.price_change_bps,
            'magnitude': self.magnitude,
            'reasoning': self.reasoning,
            'key_factors': self.key_factors,
            'method': self.method.value,
            'model_version': self.model_version,
            'symbol': self.symbol,
            'response_time_ms': self.response_time_ms,
            'api_success': self.api_success,
            'ensemble_agreement': self.ensemble_agreement,
            'uncertainty_score': self.uncertainty_score,
            'validation_passed': self.validation_passed,
            'cache_hit': self.cache_hit,
            'timestamp': self.timestamp.isoformat()
        }

    def is_actionable(self, min_confidence: float = 0.6) -> bool:
        """Check if prediction is actionable based on confidence."""
        return (self.confidence >= min_confidence and
                self.validation_passed and
                self.direction != 'neutral')


class PredictionValidator:
    """Validator for prediction results."""

    def __init__(self):
        """Initialize prediction validator."""
        self.validation_rules = [
            self._validate_direction,
            self._validate_confidence,
            self._validate_price_change,
            self._validate_consistency,
            self._validate_magnitude
        ]

    async def validate(self, prediction: dict[str, Any]) -> tuple[bool, list[str]]:
        """Validate prediction result.

        Args:
            prediction: Prediction dictionary to validate

        Returns:
            Tuple of (is_valid, validation_errors)
        """
        errors = []

        try:
            for rule in self.validation_rules:
                try:
                    error = await rule(prediction)
                    if error:
                        errors.append(error)
                except Exception as e:
                    errors.append(f"Validation rule error: {e}")

            return len(errors) == 0, errors

        except Exception as e:
            logger.error(f"Prediction validation failed: {e}")
            return False, [f"Validation exception: {e}"]

    async def _validate_direction(self, prediction: dict[str, Any]) -> str | None:
        """Validate direction field."""
        direction = prediction.get('direction', '').lower()
        valid_directions = ['bullish', 'bearish', 'neutral']

        if direction not in valid_directions:
            return f"Invalid direction: {direction}"
        return None

    async def _validate_confidence(self, prediction: dict[str, Any]) -> str | None:
        """Validate confidence field."""
        confidence = prediction.get('confidence')

        if not isinstance(confidence, int | float):
            return f"Invalid confidence type: {type(confidence)}"

        if not 0 <= confidence <= 1:
            return f"Confidence out of range [0,1]: {confidence}"

        return None

    async def _validate_price_change(self, prediction: dict[str, Any]) -> str | None:
        """Validate price change field."""
        price_change_bps = prediction.get('price_change_bps')

        if not isinstance(price_change_bps, int | float):
            return f"Invalid price_change_bps type: {type(price_change_bps)}"

        if abs(price_change_bps) > 1000:  # Sanity check
            return f"Price change too large: {price_change_bps} bps"

        return None

    async def _validate_consistency(self, prediction: dict[str, Any]) -> str | None:
        """Validate prediction consistency."""
        direction = prediction.get('direction', '').lower()
        price_change_bps = prediction.get('price_change_bps', 0)

        # Check direction-price change consistency
        if direction == 'bullish' and price_change_bps < -5:
            return f"Inconsistent: bullish direction with negative price change ({price_change_bps} bps)"

        if direction == 'bearish' and price_change_bps > 5:
            return f"Inconsistent: bearish direction with positive price change ({price_change_bps} bps)"

        if direction == 'neutral' and abs(price_change_bps) > 20:
            return f"Inconsistent: neutral direction with large price change ({price_change_bps} bps)"

        return None

    async def _validate_magnitude(self, prediction: dict[str, Any]) -> str | None:
        """Validate magnitude field."""
        magnitude = prediction.get('magnitude', '').lower()
        valid_magnitudes = ['low', 'medium', 'high']

        if magnitude not in valid_magnitudes:
            return f"Invalid magnitude: {magnitude}"

        # Check magnitude-price change consistency
        price_change_bps = abs(prediction.get('price_change_bps', 0))

        if magnitude == 'high' and price_change_bps < 10:
            return "Inconsistent: high magnitude with small price change"

        if magnitude == 'low' and price_change_bps > 30:
            return "Inconsistent: low magnitude with large price change"

        return None


class PredictionCache:
    """Cache for prediction results to avoid redundant API calls."""

    def __init__(self, ttl_seconds: int = 30, max_size: int = 1000):
        """Initialize prediction cache.

        Args:
            ttl_seconds: Time-to-live for cached predictions
            max_size: Maximum cache size
        """
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self.cache: dict[str, dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    def _generate_cache_key(self,
                          order_book: OrderBookSnapshot,
                          trades: list[Trade]) -> str:
        """Generate cache key from market data."""
        # Create hash from key market data points
        key_data = {
            'symbol': order_book.symbol,
            'best_bid': float(order_book.best_bid or 0),
            'best_ask': float(order_book.best_ask or 0),
            'bid_size': float(order_book.bids[0].size) if order_book.bids else 0,
            'ask_size': float(order_book.asks[0].size) if order_book.asks else 0,
            'trade_count': len(trades),
            'recent_volume': sum(float(t.size) for t in trades[-5:]) if trades else 0
        }

        # Create hash
        key_str = str(sorted(key_data.items()))
        return hashlib.sha256(key_str.encode()).hexdigest()[:32]  # Take first 32 chars to match MD5 length

    async def get(self,
                  order_book: OrderBookSnapshot,
                  trades: list[Trade]) -> dict[str, Any] | None:
        """Get cached prediction if available and valid."""
        async with self._lock:
            cache_key = self._generate_cache_key(order_book, trades)

            if cache_key not in self.cache:
                return None

            cached_item = self.cache[cache_key]

            # Check if expired
            cache_time = datetime.fromisoformat(cached_item['cached_at'])
            if (datetime.utcnow() - cache_time).total_seconds() > self.ttl_seconds:
                del self.cache[cache_key]
                return None

            return cached_item['prediction']

    async def set(self,
                  order_book: OrderBookSnapshot,
                  trades: list[Trade],
                  prediction: dict[str, Any]) -> None:
        """Cache prediction result."""
        async with self._lock:
            cache_key = self._generate_cache_key(order_book, trades)

            self.cache[cache_key] = {
                'prediction': prediction,
                'cached_at': datetime.utcnow().isoformat()
            }

            # Cleanup old entries if cache is full
            if len(self.cache) > self.max_size:
                await self._cleanup_old_entries()

    async def _cleanup_old_entries(self) -> None:
        """Remove old cache entries."""
        # Remove oldest 25% of entries
        sorted_items = sorted(
            self.cache.items(),
            key=lambda x: x[1]['cached_at']
        )

        items_to_remove = len(sorted_items) // 4
        for i in range(items_to_remove):
            del self.cache[sorted_items[i][0]]

    async def clear(self) -> None:
        """Clear all cached predictions."""
        async with self._lock:
            self.cache.clear()

    async def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        async with self._lock:
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'ttl_seconds': self.ttl_seconds
            }


class EnsemblePredictionEngine:
    """Ensemble prediction engine combining multiple prediction methods."""

    def __init__(self,
                 azure_client: AzureOpenAIClient | None = None,
                 rule_engine: RuleBasedEngine | None = None,
                 enable_caching: bool = True,
                 cache_ttl: int = 30):
        """Initialize ensemble prediction engine.

        Args:
            azure_client: Azure OpenAI client
            rule_engine: Rule-based prediction engine
            enable_caching: Whether to enable prediction caching
            cache_ttl: Cache time-to-live in seconds
        """
        self.azure_client = azure_client
        self.rule_engine = rule_engine or RuleBasedEngine()

        # Circuit breaker for Azure OpenAI
        self.circuit_breaker = AzureOpenAICircuitBreaker(self.rule_engine)

        # Prediction validator
        self.validator = PredictionValidator()

        # Prediction cache
        self.cache = PredictionCache(ttl_seconds=cache_ttl) if enable_caching else None

        # Performance tracking
        self.prediction_count = 0
        self.method_counts = {method.value: 0 for method in PredictionMethod}
        self.validation_failures = 0
        self.cache_hits = 0
        self.last_prediction_time = None

        # Ensemble weights
        self.ensemble_weights = {
            'azure_openai': 0.7,
            'rule_based': 0.3
        }

    async def initialize(self) -> None:
        """Initialize the ensemble engine."""
        try:
            if self.azure_client:
                await self.azure_client.initialize()

            logger.info("Ensemble prediction engine initialized")

        except Exception as e:
            logger.error(f"Failed to initialize ensemble engine: {e}")
            raise

    async def predict(self,
                     order_book: OrderBookSnapshot,
                     recent_trades: list[Trade],
                     market_stats: MarketStats | None = None,
                     prediction_horizon_ms: int = 200,
                     force_method: PredictionMethod | None = None) -> PredictionResult:
        """Generate ensemble prediction.

        Args:
            order_book: Current order book snapshot
            recent_trades: Recent trades list
            market_stats: Market statistics (optional)
            prediction_horizon_ms: Prediction horizon in milliseconds
            force_method: Force specific prediction method (for testing)

        Returns:
            Structured prediction result
        """
        start_time = datetime.utcnow()

        try:
            # Check cache first
            if self.cache and not force_method:
                cached_prediction = await self.cache.get(order_book, recent_trades)
                if cached_prediction:
                    self.cache_hits += 1
                    result = PredictionResult(**cached_prediction, cache_hit=True)
                    result.response_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
                    return result

            # Generate prediction based on method
            if force_method:
                prediction_data = await self._predict_with_method(
                    force_method, order_book, recent_trades, market_stats, prediction_horizon_ms
                )
            else:
                prediction_data = await self._ensemble_predict(
                    order_book, recent_trades, market_stats, prediction_horizon_ms
                )

            # Validate prediction
            is_valid, validation_errors = await self.validator.validate(prediction_data)
            if not is_valid:
                logger.warning(f"Prediction validation failed: {validation_errors}")
                self.validation_failures += 1
                prediction_data = await self._create_safe_fallback_prediction(order_book.symbol)

            # Create prediction result
            prediction_data['validation_passed'] = is_valid
            prediction_data['response_time_ms'] = (datetime.utcnow() - start_time).total_seconds() * 1000

            result = PredictionResult(**prediction_data)

            # Cache result if enabled
            if self.cache and not force_method:
                await self.cache.set(order_book, recent_trades, prediction_data)

            # Update stats
            self.prediction_count += 1
            self.method_counts[result.method.value] += 1
            self.last_prediction_time = datetime.utcnow()

            return result

        except Exception as e:
            logger.error(f"Ensemble prediction failed: {e}")

            # Return safe fallback
            fallback_data = await self._create_safe_fallback_prediction(order_book.symbol)
            fallback_data['response_time_ms'] = (datetime.utcnow() - start_time).total_seconds() * 1000

            return PredictionResult(**fallback_data)

    async def _ensemble_predict(self,
                               order_book: OrderBookSnapshot,
                               recent_trades: list[Trade],
                               market_stats: MarketStats | None,
                               prediction_horizon_ms: int) -> dict[str, Any]:
        """Generate ensemble prediction combining multiple methods."""
        predictions = {}

        # Get Azure OpenAI prediction (with circuit breaker)
        if self.azure_client:
            try:
                azure_prediction = await self.circuit_breaker.predict_with_fallback(
                    self.azure_client.predict_market_direction,
                    order_book=order_book,
                    recent_trades=recent_trades,
                    market_stats=market_stats,
                    prediction_horizon_ms=prediction_horizon_ms
                )
                predictions['azure_openai'] = azure_prediction

            except Exception as e:
                logger.warning(f"Azure OpenAI prediction failed in ensemble: {e}")

        # Get rule-based prediction
        try:
            rule_prediction = await self.rule_engine.predict(
                order_book=order_book,
                recent_trades=recent_trades,
                market_stats=market_stats
            )
            predictions['rule_based'] = rule_prediction

        except Exception as e:
            logger.warning(f"Rule-based prediction failed in ensemble: {e}")

        # If we have multiple predictions, combine them
        if len(predictions) > 1:
            return await self._combine_predictions(predictions, order_book.symbol)
        elif predictions:
            # Single prediction available
            method_name = list(predictions.keys())[0]
            single_prediction = list(predictions.values())[0]
            single_prediction['method'] = method_name
            single_prediction['ensemble_agreement'] = 1.0
            return single_prediction
        else:
            # No predictions available - fallback
            return await self._create_safe_fallback_prediction(order_book.symbol)

    async def _predict_with_method(self,
                                  method: PredictionMethod,
                                  order_book: OrderBookSnapshot,
                                  recent_trades: list[Trade],
                                  market_stats: MarketStats | None,
                                  prediction_horizon_ms: int) -> dict[str, Any]:
        """Generate prediction with specific method."""
        if method == PredictionMethod.AZURE_OPENAI and self.azure_client:
            prediction = await self.azure_client.predict_market_direction(
                order_book=order_book,
                recent_trades=recent_trades,
                market_stats=market_stats,
                prediction_horizon_ms=prediction_horizon_ms
            )
            prediction['method'] = method.value
            return prediction

        elif method == PredictionMethod.RULE_BASED:
            prediction = await self.rule_engine.predict(
                order_book=order_book,
                recent_trades=recent_trades,
                market_stats=market_stats
            )
            prediction['method'] = method.value
            return prediction

        else:
            return await self._create_safe_fallback_prediction(order_book.symbol)

    async def _combine_predictions(self,
                                  predictions: dict[str, dict[str, Any]],
                                  symbol: str) -> dict[str, Any]:
        """Combine multiple predictions using ensemble logic."""
        try:
            # Extract prediction components
            directions = []
            confidences = []
            price_changes = []
            weights = []

            for method, prediction in predictions.items():
                if prediction and 'direction' in prediction:
                    directions.append(prediction['direction'])
                    confidences.append(prediction.get('confidence', 0.5))
                    price_changes.append(prediction.get('price_change_bps', 0.0))
                    weights.append(self.ensemble_weights.get(method, 0.1))

            if not directions:
                return await self._create_safe_fallback_prediction(symbol)

            # Normalize weights
            total_weight = sum(weights)
            if total_weight > 0:
                weights = [w / total_weight for w in weights]

            # Combine predictions
            # Direction: weighted voting
            direction_scores = {'bullish': 0.0, 'bearish': 0.0, 'neutral': 0.0}

            for i, direction in enumerate(directions):
                direction_scores[direction] += weights[i] * confidences[i]

            final_direction = max(direction_scores.items(), key=lambda x: x[1])[0]

            # Confidence: weighted average, adjusted for agreement
            final_confidence = sum(c * w for c, w in zip(confidences, weights, strict=False))

            # Calculate agreement score
            agreement = self._calculate_agreement(directions, confidences)
            final_confidence *= agreement  # Reduce confidence if predictions disagree

            # Price change: weighted average
            final_price_change = sum(p * w for p, w in zip(price_changes, weights, strict=False))

            # Determine magnitude
            magnitude = "low"
            if abs(final_price_change) > 15:
                magnitude = "high"
            elif abs(final_price_change) > 5:
                magnitude = "medium"

            # Build reasoning
            reasoning_parts = []
            for method, prediction in predictions.items():
                if prediction.get('reasoning'):
                    reasoning_parts.append(f"{method}: {prediction['reasoning'][:50]}")

            reasoning = f"Ensemble: {'; '.join(reasoning_parts[:2])}"

            return {
                'direction': final_direction,
                'confidence': min(0.95, max(0.05, final_confidence)),
                'price_change_bps': round(final_price_change, 1),
                'magnitude': magnitude,
                'reasoning': reasoning,
                'key_factors': ['ensemble_prediction'],
                'method': PredictionMethod.ENSEMBLE.value,
                'model_version': 'ensemble-v1.0',
                'symbol': symbol,
                'ensemble_agreement': agreement,
                'uncertainty_score': 1.0 - agreement,
                'api_success': any(p.get('api_success', False) for p in predictions.values())
            }

        except Exception as e:
            logger.error(f"Prediction combination failed: {e}")
            return await self._create_safe_fallback_prediction(symbol)

    def _calculate_agreement(self, directions: list[str], confidences: list[float]) -> float:
        """Calculate agreement score between predictions."""
        if len(directions) <= 1:
            return 1.0

        # Count direction agreement
        direction_counts = {}
        for direction in directions:
            direction_counts[direction] = direction_counts.get(direction, 0) + 1

        max_count = max(direction_counts.values())
        direction_agreement = max_count / len(directions)

        # Consider confidence spread
        confidence_std = np.std(confidences) if len(confidences) > 1 else 0.0
        confidence_agreement = max(0.0, 1.0 - confidence_std * 2)  # Lower std = higher agreement

        # Combined agreement score
        return (direction_agreement * 0.7 + confidence_agreement * 0.3)

    async def _create_safe_fallback_prediction(self, symbol: str) -> dict[str, Any]:
        """Create safe fallback prediction when all methods fail."""
        return {
            'direction': 'neutral',
            'confidence': 0.5,
            'price_change_bps': 0.0,
            'magnitude': 'low',
            'reasoning': 'Ensemble fallback: All prediction methods failed',
            'key_factors': ['ensemble_fallback'],
            'method': PredictionMethod.FALLBACK.value,
            'model_version': 'ensemble-fallback-v1.0',
            'symbol': symbol,
            'ensemble_agreement': 1.0,
            'uncertainty_score': 0.5,
            'api_success': False
        }

    async def get_performance_stats(self) -> dict[str, Any]:
        """Get comprehensive performance statistics."""
        stats = {
            'prediction_count': self.prediction_count,
            'method_counts': self.method_counts.copy(),
            'validation_failures': self.validation_failures,
            'cache_hits': self.cache_hits,
            'last_prediction_time': self.last_prediction_time.isoformat() if self.last_prediction_time else None,
            'ensemble_weights': self.ensemble_weights.copy(),
            'circuit_breaker_status': None,
            'cache_stats': None
        }

        # Add circuit breaker stats
        if self.circuit_breaker:
            stats['circuit_breaker_status'] = await self.circuit_breaker.get_status()

        # Add cache stats
        if self.cache:
            stats['cache_stats'] = await self.cache.get_stats()

        # Add Azure client stats
        if self.azure_client:
            stats['azure_client_stats'] = self.azure_client.get_performance_stats()

        # Add rule engine stats
        if self.rule_engine:
            stats['rule_engine_stats'] = self.rule_engine.get_performance_stats()

        return stats

    async def health_check(self) -> dict[str, Any]:
        """Perform health check on prediction engine."""
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'components': {}
        }

        try:
            # Check Azure OpenAI client
            if self.azure_client:
                azure_health = await self.azure_client.health_check()
                health_status['components']['azure_openai'] = azure_health

                if azure_health['status'] != 'healthy':
                    health_status['status'] = 'degraded'

            # Check circuit breaker
            if self.circuit_breaker:
                cb_status = await self.circuit_breaker.get_status()
                health_status['components']['circuit_breaker'] = {
                    'status': 'healthy' if cb_status['state'] != 'open' else 'unhealthy',
                    'state': cb_status['state']
                }

                if cb_status['state'] == 'open':
                    health_status['status'] = 'degraded'

            # Rule engine is always available
            health_status['components']['rule_engine'] = {'status': 'healthy'}

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            health_status['status'] = 'unhealthy'
            health_status['error'] = str(e)

        return health_status

    async def cleanup(self) -> None:
        """Clean up resources."""
        try:
            if self.azure_client:
                await self.azure_client.cleanup()

            if self.cache:
                await self.cache.clear()

            logger.info("Ensemble prediction engine cleanup completed")

        except Exception as e:
            logger.error(f"Cleanup error: {e}")

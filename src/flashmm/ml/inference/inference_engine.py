"""
FlashMM ML Inference Engine

Enhanced inference engine with Azure OpenAI o4-mini integration,
ensemble predictions, and seamless integration with market data pipeline.
"""

import asyncio
from datetime import datetime
from typing import Any

from flashmm.config.settings import get_config
from flashmm.data.storage.data_models import OrderBookSnapshot, Trade
from flashmm.ml.models.prediction_models import PredictionMethod, PredictionResult
from flashmm.ml.prediction_service import PredictionService, get_prediction_service
from flashmm.utils.decorators import measure_latency, timeout_async
from flashmm.utils.exceptions import InferenceError, ModelError
from flashmm.utils.logging import get_logger

logger = get_logger(__name__)


class InferenceEngine:
    """Enhanced ML inference engine with Azure OpenAI integration."""

    def __init__(self):
        """Initialize inference engine."""
        self.config = get_config()
        self.prediction_service: PredictionService | None = None
        self.confidence_threshold = self.config.get("ml.confidence_threshold", 0.6)
        self.inference_timeout_ms = self.config.get("ml.inference_timeout_ms", 450)  # Updated for Azure OpenAI
        self._initialized = False

        # Performance tracking
        self.prediction_count = 0
        self.last_prediction_time = None
        self.successful_predictions = 0
        self.failed_predictions = 0

    async def initialize(self) -> None:
        """Initialize the inference engine."""
        if self._initialized:
            return

        logger.info("Initializing enhanced ML InferenceEngine with Azure OpenAI...")

        try:
            # Get or create prediction service
            self.prediction_service = get_prediction_service()
            await self.prediction_service.initialize()

            self._initialized = True
            logger.info("Enhanced ML InferenceEngine initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize InferenceEngine: {e}")
            raise ModelError(f"InferenceEngine initialization failed: {e}") from e

    @timeout_async(0.45)  # 450ms timeout for Azure OpenAI integration
    @measure_latency("ml_inference")
    async def predict(self, market_data: dict[str, Any]) -> dict[str, Any] | None:
        """Generate prediction from market data using Azure OpenAI ensemble.

        Args:
            market_data: Market data dictionary containing order_book, trades, etc.

        Returns:
            Prediction dictionary or None if prediction fails/low confidence
        """
        if not self._initialized:
            await self.initialize()

        if not self.prediction_service:
            raise InferenceError("Prediction service not available")

        try:
            # Extract market data components
            order_book = market_data.get('order_book')
            recent_trades = market_data.get('recent_trades', [])
            market_stats = market_data.get('market_stats')
            symbol = market_data.get('symbol', 'SEI/USDC')

            # Validate required data
            if not order_book:
                logger.warning("No order book data available for prediction")
                return None

            # Convert to proper data models if needed
            if isinstance(order_book, dict):
                order_book = self._convert_to_order_book_snapshot(order_book, symbol)

            if recent_trades and isinstance(recent_trades[0], dict):
                recent_trades = self._convert_to_trades(recent_trades, symbol)

            # Generate prediction using ensemble engine
            if not hasattr(self.prediction_service, 'ensemble_engine') or not self.prediction_service.ensemble_engine:
                raise InferenceError("Ensemble engine not available")

            prediction_result = await self.prediction_service.ensemble_engine.predict(
                order_book=order_book,
                recent_trades=recent_trades,
                market_stats=market_stats,
                prediction_horizon_ms=200  # Short-term prediction for market making
            )

            # Update statistics
            self.prediction_count += 1
            self.last_prediction_time = datetime.utcnow()

            if prediction_result.validation_passed:
                self.successful_predictions += 1
            else:
                self.failed_predictions += 1

            # Filter by confidence threshold
            if prediction_result.confidence < self.confidence_threshold:
                logger.debug(f"Prediction confidence {prediction_result.confidence:.3f} below threshold {self.confidence_threshold}")
                return None

            # Convert to legacy format for backward compatibility
            legacy_prediction = self._convert_to_legacy_format(prediction_result)

            return legacy_prediction

        except Exception as e:
            logger.error(f"Inference failed: {e}")
            self.failed_predictions += 1
            raise InferenceError(f"Prediction failed: {e}") from e

    def _convert_to_order_book_snapshot(self, order_book_dict: dict[str, Any], symbol: str) -> OrderBookSnapshot:
        """Convert dictionary to OrderBookSnapshot."""
        try:
            from flashmm.data.storage.data_models import OrderBookLevel

            # Extract bids and asks
            bids = []
            asks = []

            if 'bids' in order_book_dict:
                for bid_data in order_book_dict['bids'][:10]:  # Top 10 levels
                    if isinstance(bid_data, list) and len(bid_data) >= 2:
                        bids.append(OrderBookLevel(price=bid_data[0], size=bid_data[1]))

            if 'asks' in order_book_dict:
                for ask_data in order_book_dict['asks'][:10]:  # Top 10 levels
                    if isinstance(ask_data, list) and len(ask_data) >= 2:
                        asks.append(OrderBookLevel(price=ask_data[0], size=ask_data[1]))

            return OrderBookSnapshot(
                symbol=symbol,
                timestamp=datetime.fromisoformat(order_book_dict.get('timestamp', datetime.utcnow().isoformat())),
                sequence=order_book_dict.get('sequence'),
                bids=bids,
                asks=asks,
                source=order_book_dict.get('source', 'market_data_service')
            )

        except Exception as e:
            logger.error(f"Failed to convert order book: {e}")
            raise InferenceError(f"Order book conversion failed: {e}") from e

    def _convert_to_trades(self, trades_list: list[dict[str, Any]], symbol: str) -> list[Trade]:
        """Convert trade dictionaries to Trade objects."""
        try:
            from flashmm.data.storage.data_models import Side

            trades = []
            for trade_data in trades_list:
                trades.append(Trade(
                    symbol=symbol,
                    timestamp=datetime.fromisoformat(trade_data.get('timestamp', datetime.utcnow().isoformat())),
                    price=trade_data['price'],
                    size=trade_data['size'],
                    side=Side(trade_data.get('side', 'buy')),
                    trade_id=trade_data.get('trade_id'),
                    source=trade_data.get('source', 'market_data_service'),
                    sequence=trade_data.get('sequence')
                ))

            return trades

        except Exception as e:
            logger.error(f"Failed to convert trades: {e}")
            raise InferenceError(f"Trade conversion failed: {e}") from e

    def _convert_to_legacy_format(self, prediction_result: PredictionResult) -> dict[str, Any]:
        """Convert PredictionResult to legacy format for backward compatibility."""
        # Map direction to signal strength
        if prediction_result.direction == 'bullish':
            signal_strength = prediction_result.confidence
        elif prediction_result.direction == 'bearish':
            signal_strength = -prediction_result.confidence
        else:
            signal_strength = 0.0

        # Estimate price prediction from price change
        price_prediction = prediction_result.price_change_bps / 10000.0  # Convert bps to decimal

        return {
            "price_prediction": price_prediction,
            "confidence": prediction_result.confidence,
            "signal_strength": signal_strength,
            "timestamp": prediction_result.timestamp.isoformat(),
            "symbol": prediction_result.symbol,

            # Enhanced fields
            "direction": prediction_result.direction,
            "price_change_bps": prediction_result.price_change_bps,
            "magnitude": prediction_result.magnitude,
            "reasoning": prediction_result.reasoning,
            "method": prediction_result.method.value,
            "model_version": prediction_result.model_version,
            "response_time_ms": prediction_result.response_time_ms,
            "api_success": prediction_result.api_success,
            "validation_passed": prediction_result.validation_passed,
            "confidence_level": prediction_result.confidence_level.value,
            "ensemble_agreement": prediction_result.ensemble_agreement,
            "uncertainty_score": prediction_result.uncertainty_score
        }

    async def get_prediction(self, market_data: dict[str, Any]) -> dict[str, Any] | None:
        """Alias for predict method for backward compatibility."""
        return await self.predict(market_data)

    async def predict_with_method(self,
                                 market_data: dict[str, Any],
                                 method: str = "ensemble") -> dict[str, Any] | None:
        """Generate prediction using specific method.

        Args:
            market_data: Market data dictionary
            method: Prediction method ('azure_openai', 'rule_based', 'ensemble')

        Returns:
            Prediction dictionary or None
        """
        if not self._initialized:
            await self.initialize()

        try:
            # Map method string to enum
            method_map = {
                'azure_openai': PredictionMethod.AZURE_OPENAI,
                'rule_based': PredictionMethod.RULE_BASED,
                'ensemble': PredictionMethod.ENSEMBLE,
                'fallback': PredictionMethod.FALLBACK
            }

            prediction_method = method_map.get(method.lower(), PredictionMethod.ENSEMBLE)

            # Extract and convert market data
            order_book = market_data.get('order_book')
            recent_trades = market_data.get('recent_trades', [])
            market_stats = market_data.get('market_stats')
            symbol = market_data.get('symbol', 'SEI/USDC')

            if not order_book:
                return None

            if isinstance(order_book, dict):
                order_book = self._convert_to_order_book_snapshot(order_book, symbol)

            if recent_trades and isinstance(recent_trades[0], dict):
                recent_trades = self._convert_to_trades(recent_trades, symbol)

            # Generate prediction with specific method
            if not self.prediction_service:
                raise InferenceError("Prediction service not available")
            if not hasattr(self.prediction_service, 'ensemble_engine') or not self.prediction_service.ensemble_engine:
                raise InferenceError("Ensemble engine not available")

            prediction_result = await self.prediction_service.ensemble_engine.predict(
                order_book=order_book,
                recent_trades=recent_trades,
                market_stats=market_stats,
                force_method=prediction_method
            )

            # Convert to legacy format
            return self._convert_to_legacy_format(prediction_result)

        except Exception as e:
            logger.error(f"Method-specific prediction failed: {e}")
            raise InferenceError(f"Prediction with method {method} failed: {e}") from e

    async def get_prediction_health(self) -> dict[str, Any]:
        """Get prediction engine health status."""
        if not self._initialized or not self.prediction_service:
            return {
                'status': 'unhealthy',
                'reason': 'not_initialized',
                'timestamp': datetime.utcnow().isoformat()
            }

        try:
            # Get prediction service health
            service_health = await self.prediction_service.get_health_status()

            # Add inference engine specific metrics
            success_rate = (self.successful_predictions / max(self.prediction_count, 1)) * 100

            return {
                'status': service_health['status'],
                'timestamp': datetime.utcnow().isoformat(),
                'inference_engine': {
                    'prediction_count': self.prediction_count,
                    'successful_predictions': self.successful_predictions,
                    'failed_predictions': self.failed_predictions,
                    'success_rate': success_rate,
                    'confidence_threshold': self.confidence_threshold,
                    'last_prediction_time': self.last_prediction_time.isoformat() if self.last_prediction_time else None
                },
                'prediction_service': service_health
            }

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                'status': 'unhealthy',
                'reason': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the prediction models."""
        model_info = {
            "model_loaded": self._initialized,
            "model_type": "ensemble_azure_openai",
            "confidence_threshold": self.confidence_threshold,
            "inference_timeout_ms": self.inference_timeout_ms,
            "prediction_count": self.prediction_count,
            "success_rate": (self.successful_predictions / max(self.prediction_count, 1)) * 100
        }

        # Add prediction service info if available
        if self.prediction_service and self._initialized:
            try:
                asyncio.create_task(self.prediction_service.get_performance_stats())
                # Note: This is a simplified sync version - full async stats available via get_prediction_health
                model_info["service_available"] = True
                model_info["service_running"] = self.prediction_service.is_running
            except Exception:
                model_info["service_available"] = False

        return model_info

    async def start_prediction_service(self) -> None:
        """Start the prediction service if not already running."""
        if not self._initialized:
            await self.initialize()

        if self.prediction_service and not self.prediction_service.is_running:
            await self.prediction_service.start()
            logger.info("Prediction service started from inference engine")

    async def stop_prediction_service(self) -> None:
        """Stop the prediction service."""
        if self.prediction_service and self.prediction_service.is_running:
            await self.prediction_service.stop()
            logger.info("Prediction service stopped from inference engine")

    async def cleanup(self) -> None:
        """Clean up resources."""
        try:
            if self.prediction_service:
                await self.prediction_service.stop()

            self._initialized = False
            logger.info("Inference engine cleanup completed")

        except Exception as e:
            logger.error(f"Cleanup error: {e}")

    # Legacy method compatibility
    async def _prepare_features(self, market_data: dict[str, Any]) -> None:
        """Legacy feature preparation - now handled by prediction service."""
        # This method is kept for backward compatibility but is no longer used
        # Feature extraction is now handled by the FeatureExtractor in the prediction service
        pass

    async def _process_output(self, output: Any) -> dict[str, Any]:
        """Legacy output processing - now handled by prediction service."""
        # This method is kept for backward compatibility but is no longer used
        # Output processing is now handled by the ensemble prediction engine
        return {}

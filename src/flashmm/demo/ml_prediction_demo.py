"""
FlashMM ML Prediction Engine Demo

Comprehensive demonstration of the Azure OpenAI-powered prediction engine
including ensemble predictions, fallback mechanisms, and performance monitoring.
"""

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any

from flashmm.config.settings import get_config
from flashmm.data.storage.data_models import (
    MarketStats,
    OrderBookLevel,
    OrderBookSnapshot,
    Side,
    Trade,
)
from flashmm.ml.inference.inference_engine import InferenceEngine
from flashmm.ml.models.prediction_models import PredictionMethod
from flashmm.ml.prediction_service import PredictionService, PredictionServiceConfig
from flashmm.monitoring.ml_metrics import MLPerformanceDashboard, get_ml_metrics_collector
from flashmm.utils.logging import get_logger

logger = get_logger(__name__)


class MLPredictionDemo:
    """Demo class for ML prediction engine capabilities."""

    def __init__(self):
        """Initialize demo."""
        self.config = get_config()
        self.prediction_service: PredictionService | None = None
        self.inference_engine: InferenceEngine | None = None
        self.ml_metrics = get_ml_metrics_collector()
        self.dashboard = MLPerformanceDashboard(self.ml_metrics)

    async def initialize(self) -> None:
        """Initialize demo components."""
        try:
            logger.info("🚀 Initializing FlashMM ML Prediction Demo...")

            # Initialize prediction service
            config = PredictionServiceConfig(
                prediction_frequency_hz=2.0,  # 2Hz for demo
                publish_to_redis=False,       # Disable Redis for demo
                enable_caching=True,          # Enable caching
                cache_ttl_seconds=30
            )

            self.prediction_service = PredictionService(config)
            await self.prediction_service.initialize()

            # Initialize inference engine
            self.inference_engine = InferenceEngine()
            await self.inference_engine.initialize()

            logger.info("✅ Demo initialization complete")

        except Exception as e:
            logger.error(f"❌ Demo initialization failed: {e}")
            raise

    def _create_sample_market_data(self, symbol: str = "SEI/USDC") -> dict[str, Any]:
        """Create realistic sample market data."""
        base_price = Decimal("1.2500")
        spread_bps = 8  # 8 basis points spread
        spread = base_price * Decimal(str(spread_bps / 10000))

        # Create order book with realistic depth
        bids = [
            OrderBookLevel(price=base_price - spread/2, size=Decimal("1500")),
            OrderBookLevel(price=base_price - spread/2 - Decimal("0.0010"), size=Decimal("1200")),
            OrderBookLevel(price=base_price - spread/2 - Decimal("0.0020"), size=Decimal("800")),
            OrderBookLevel(price=base_price - spread/2 - Decimal("0.0030"), size=Decimal("600")),
            OrderBookLevel(price=base_price - spread/2 - Decimal("0.0040"), size=Decimal("400")),
        ]

        asks = [
            OrderBookLevel(price=base_price + spread/2, size=Decimal("1400")),
            OrderBookLevel(price=base_price + spread/2 + Decimal("0.0010"), size=Decimal("1100")),
            OrderBookLevel(price=base_price + spread/2 + Decimal("0.0020"), size=Decimal("750")),
            OrderBookLevel(price=base_price + spread/2 + Decimal("0.0030"), size=Decimal("550")),
            OrderBookLevel(price=base_price + spread/2 + Decimal("0.0040"), size=Decimal("350")),
        ]

        order_book = OrderBookSnapshot(
            symbol=symbol,
            timestamp=datetime.utcnow(),
            sequence=12345,
            bids=bids,
            asks=asks,
            source="demo"
        )

        # Create recent trades with some momentum
        base_time = datetime.utcnow()
        trades = []

        for i in range(20):
            # Add slight upward momentum
            price_offset = Decimal(str((i - 10) * 0.0001))
            trade_price = base_price + price_offset

            trades.append(Trade(
                symbol=symbol,
                timestamp=base_time - timedelta(seconds=i * 2),
                price=trade_price,
                size=Decimal("50") + Decimal(str(i * 5)),
                side=Side.BUY if i % 3 != 0 else Side.SELL,  # 2/3 buy, 1/3 sell
                trade_id=f"demo_trade_{i}",
                sequence=i,  # Add required sequence parameter
                source="demo"
            ))

        # Create market stats
        market_stats = MarketStats(
            symbol=symbol,
            timestamp=datetime.utcnow(),
            window_seconds=300,  # 5 minute window
            open_price=base_price,
            high_price=base_price + Decimal("0.0020"),
            low_price=base_price - Decimal("0.0015"),
            close_price=base_price,
            volume=Decimal("25000"),
            trade_count=150,
            vwap=base_price,
            avg_spread=base_price * Decimal(str(spread_bps / 10000)),
            avg_spread_bps=Decimal(str(spread_bps))
        )

        return {
            'order_book': order_book,
            'recent_trades': trades,
            'market_stats': market_stats
        }

    async def demo_basic_prediction(self) -> Any:
        """Demonstrate basic prediction functionality."""
        logger.info("\n📈 === BASIC PREDICTION DEMO ===")

        # Create sample market data
        market_data = self._create_sample_market_data()

        # Test rule-based prediction
        logger.info("🔧 Testing rule-based prediction...")
        if not self.prediction_service or not self.prediction_service.ensemble_engine:
            raise RuntimeError("Prediction service not initialized")

        rule_prediction = await self.prediction_service.ensemble_engine.predict(
            order_book=market_data['order_book'],
            recent_trades=market_data['recent_trades'],
            market_stats=market_data['market_stats'],
            force_method=PredictionMethod.RULE_BASED
        )

        logger.info(f"✅ Rule-based result: {rule_prediction.direction} "
                   f"(confidence: {rule_prediction.confidence:.3f}, "
                   f"change: {rule_prediction.price_change_bps:.1f} bps)")

        # Test ensemble prediction (will use rule-based if Azure OpenAI not configured)
        logger.info("🎯 Testing ensemble prediction...")
        if not self.prediction_service or not self.prediction_service.ensemble_engine:
            raise RuntimeError("Prediction service not initialized")

        ensemble_prediction = await self.prediction_service.ensemble_engine.predict(
            order_book=market_data['order_book'],
            recent_trades=market_data['recent_trades'],
            market_stats=market_data['market_stats']
        )

        logger.info(f"✅ Ensemble result: {ensemble_prediction.direction} "
                   f"(confidence: {ensemble_prediction.confidence:.3f}, "
                   f"agreement: {ensemble_prediction.ensemble_agreement:.3f})")

        # Record metrics
        await self.ml_metrics.record_prediction(rule_prediction)
        await self.ml_metrics.record_prediction(ensemble_prediction)

        return ensemble_prediction

    async def demo_inference_engine_compatibility(self) -> None:
        """Demonstrate inference engine backward compatibility."""
        logger.info("\n🔄 === INFERENCE ENGINE COMPATIBILITY DEMO ===")

        # Create legacy market data format
        legacy_market_data = {
            'order_book': {
                'symbol': 'SEI/USDC',
                'timestamp': datetime.utcnow().isoformat(),
                'bids': [['1.2496', '1500'], ['1.2486', '1200'], ['1.2476', '800']],
                'asks': [['1.2504', '1400'], ['1.2514', '1100'], ['1.2524', '750']],
                'source': 'demo'
            },
            'recent_trades': [
                {
                    'symbol': 'SEI/USDC',
                    'timestamp': (datetime.utcnow() - timedelta(seconds=i)).isoformat(),
                    'price': str(1.2500 + i * 0.0001),
                    'size': str(100 + i * 10),
                    'side': 'buy' if i % 2 == 0 else 'sell',
                    'trade_id': f'legacy_trade_{i}'
                }
                for i in range(10)
            ],
            'symbol': 'SEI/USDC'
        }

        # Test legacy interface
        logger.info("🔧 Testing legacy predict() interface...")
        if not self.inference_engine:
            raise RuntimeError("Inference engine not initialized")

        legacy_prediction = await self.inference_engine.predict(legacy_market_data)

        if legacy_prediction:
            logger.info("✅ Legacy prediction successful:")
            logger.info(f"   • Direction: {legacy_prediction['direction']}")
            logger.info(f"   • Confidence: {legacy_prediction['confidence']:.3f}")
            logger.info(f"   • Signal Strength: {legacy_prediction['signal_strength']:.3f}")
            logger.info(f"   • Price Prediction: {legacy_prediction['price_prediction']:.6f}")
            logger.info(f"   • Method: {legacy_prediction['method']}")
            logger.info(f"   • Validation: {legacy_prediction['validation_passed']}")
        else:
            logger.warning("⚠️ Legacy prediction returned None (low confidence)")

        # Test method-specific prediction
        logger.info("🎛️ Testing method-specific prediction...")
        method_prediction = await self.inference_engine.predict_with_method(
            legacy_market_data,
            method="rule_based"
        )

        if method_prediction:
            logger.info(f"✅ Method-specific prediction: {method_prediction['direction']} "
                       f"(confidence: {method_prediction['confidence']:.3f})")

    async def demo_circuit_breaker(self) -> None:
        """Demonstrate circuit breaker functionality."""
        logger.info("\n⚡ === CIRCUIT BREAKER DEMO ===")

        # Get circuit breaker status
        if not self.prediction_service or not self.prediction_service.ensemble_engine or not self.prediction_service.ensemble_engine.circuit_breaker:
            raise RuntimeError("Prediction service not initialized")

        cb_status = await self.prediction_service.ensemble_engine.circuit_breaker.get_status()
        logger.info(f"🔧 Circuit breaker state: {cb_status['state']}")
        logger.info(f"   • Failures: {cb_status['consecutive_failures']}")
        logger.info(f"   • Successes: {cb_status['consecutive_successes']}")

        # Test manual circuit opening
        logger.info("🔴 Manually opening circuit breaker...")
        await self.prediction_service.ensemble_engine.circuit_breaker.force_open("Demo test")

        # Try prediction with open circuit
        market_data = self._create_sample_market_data()

        logger.info("🔧 Testing prediction with open circuit...")
        if not self.prediction_service or not self.prediction_service.ensemble_engine:
            raise RuntimeError("Prediction service not initialized")

        fallback_prediction = await self.prediction_service.ensemble_engine.predict(
            order_book=market_data['order_book'],
            recent_trades=market_data['recent_trades']
        )

        logger.info(f"✅ Fallback prediction: {fallback_prediction.direction} "
                   f"(API success: {fallback_prediction.api_success})")

        # Reset circuit breaker
        logger.info("🔵 Resetting circuit breaker...")
        if not self.prediction_service or not self.prediction_service.ensemble_engine or not self.prediction_service.ensemble_engine.circuit_breaker:
            raise RuntimeError("Prediction service not initialized")

        await self.prediction_service.ensemble_engine.circuit_breaker.reset()

        updated_status = await self.prediction_service.ensemble_engine.circuit_breaker.get_status()
        logger.info(f"✅ Circuit breaker reset: {updated_status['state']}")

    async def demo_caching(self) -> None:
        """Demonstrate prediction caching."""
        logger.info("\n💾 === CACHING DEMO ===")

        market_data = self._create_sample_market_data()

        # First prediction (cache miss)
        logger.info("🔧 First prediction (should miss cache)...")
        start_time = datetime.utcnow()
        if not self.prediction_service or not self.prediction_service.ensemble_engine:
            raise RuntimeError("Prediction service not initialized")

        prediction1 = await self.prediction_service.ensemble_engine.predict(
            order_book=market_data['order_book'],
            recent_trades=market_data['recent_trades']
        )
        first_duration = (datetime.utcnow() - start_time).total_seconds() * 1000

        logger.info(f"✅ First prediction: cache_hit={prediction1.cache_hit}, "
                   f"duration={first_duration:.1f}ms")

        # Second prediction with same data (cache hit)
        logger.info("🔧 Second prediction (should hit cache)...")
        start_time = datetime.utcnow()
        if not self.prediction_service or not self.prediction_service.ensemble_engine:
            raise RuntimeError("Prediction service not initialized")

        prediction2 = await self.prediction_service.ensemble_engine.predict(
            order_book=market_data['order_book'],
            recent_trades=market_data['recent_trades']
        )
        second_duration = (datetime.utcnow() - start_time).total_seconds() * 1000

        logger.info(f"✅ Second prediction: cache_hit={prediction2.cache_hit}, "
                   f"duration={second_duration:.1f}ms")

        if prediction2.cache_hit:
            speedup = first_duration / max(second_duration, 0.1)
            logger.info(f"🚀 Cache speedup: {speedup:.1f}x faster")

    async def demo_performance_monitoring(self) -> None:
        """Demonstrate performance monitoring and metrics."""
        logger.info("\n📊 === PERFORMANCE MONITORING DEMO ===")

        # Generate multiple predictions for metrics
        logger.info("🔧 Generating predictions for metrics...")
        market_data = self._create_sample_market_data()

        predictions = []
        for i in range(10):
            # Slight variations in market data
            market_data['order_book'].sequence = 12345 + i
            market_data['order_book'].timestamp = datetime.utcnow()

            if not self.prediction_service or not self.prediction_service.ensemble_engine:
                raise RuntimeError("Prediction service not initialized")

            prediction = await self.prediction_service.ensemble_engine.predict(
                order_book=market_data['order_book'],
                recent_trades=market_data['recent_trades'],
                market_stats=market_data['market_stats']
            )

            predictions.append(prediction)

            # Record with mock cost
            await self.ml_metrics.record_prediction(prediction, cost_usd=0.001)

            # Small delay
            await asyncio.sleep(0.1)

        # Get current performance
        logger.info("📈 Current performance metrics:")
        current_perf = await self.ml_metrics.get_current_performance()

        logger.info(f"   • Total Predictions: {current_perf['prediction_count']}")
        logger.info(f"   • Avg Latency: {current_perf['latency']['avg_ms']:.1f}ms")
        logger.info(f"   • P95 Latency: {current_perf['latency']['p95_ms']:.1f}ms")
        logger.info(f"   • Total Cost: ${current_perf['costs']['total_usd']:.6f}")

        # Generate performance report
        logger.info("📋 Generating performance report...")
        report = await self.dashboard.generate_performance_report(hours=1)

        logger.info("\n" + "="*50)
        logger.info(report)
        logger.info("="*50)

    async def demo_health_monitoring(self) -> None:
        """Demonstrate health monitoring."""
        logger.info("\n🏥 === HEALTH MONITORING DEMO ===")

        # Get service health
        if not self.prediction_service:
            raise RuntimeError("Prediction service not initialized")

        service_health = await self.prediction_service.get_health_status()
        logger.info(f"🔧 Service health: {service_health['status']}")

        for component, status in service_health['components'].items():
            logger.info(f"   • {component}: {status.get('status', 'unknown')}")

        # Get inference engine health
        if not self.inference_engine:
            raise RuntimeError("Inference engine not initialized")

        engine_health = await self.inference_engine.get_prediction_health()
        logger.info(f"🔧 Inference engine health: {engine_health['status']}")

        if 'inference_engine' in engine_health:
            ie_stats = engine_health['inference_engine']
            logger.info(f"   • Predictions: {ie_stats['prediction_count']}")
            logger.info(f"   • Success Rate: {ie_stats['success_rate']:.1f}%")
            logger.info(f"   • Confidence Threshold: {ie_stats['confidence_threshold']}")

    async def demo_feature_extraction(self) -> None:
        """Demonstrate feature extraction capabilities."""
        logger.info("\n🧮 === FEATURE EXTRACTION DEMO ===")

        from flashmm.ml.features.feature_extractor import FeatureExtractor

        extractor = FeatureExtractor()
        market_data = self._create_sample_market_data()

        logger.info("🔧 Extracting features from market data...")
        features = await extractor.extract_features(
            current_book=market_data['order_book'],
            recent_trades=market_data['recent_trades'],
            market_stats=market_data['market_stats']
        )

        logger.info(f"✅ Extracted {features['feature_count']} features:")

        # Show key features
        key_features = [
            'book_mid_price', 'book_spread_bps', 'book_depth_imbalance',
            'trade_total_volume', 'trade_buy_ratio', 'trade_vwap',
            'micro_level_1_imbalance', 'regime_avg_spread'
        ]

        for feature in key_features:
            if feature in features:
                logger.info(f"   • {feature}: {features[feature]:.6f}")

    async def run_complete_demo(self) -> None:
        """Run complete demonstration of all capabilities."""
        try:
            logger.info("🎬 Starting FlashMM ML Prediction Engine Demo")
            logger.info("="*60)

            await self.initialize()

            # Run all demo sections
            await self.demo_basic_prediction()
            await self.demo_inference_engine_compatibility()
            await self.demo_circuit_breaker()
            await self.demo_caching()
            await self.demo_feature_extraction()
            await self.demo_performance_monitoring()
            await self.demo_health_monitoring()

            logger.info("\n🎉 === DEMO COMPLETE ===")
            logger.info("✅ All ML prediction engine components demonstrated successfully!")

            # Final summary
            logger.info("\n📋 DEMO SUMMARY:")
            logger.info("✅ Basic predictions (rule-based & ensemble)")
            logger.info("✅ Legacy inference engine compatibility")
            logger.info("✅ Circuit breaker patterns")
            logger.info("✅ Prediction caching")
            logger.info("✅ Feature extraction")
            logger.info("✅ Performance monitoring")
            logger.info("✅ Health monitoring")

        except Exception as e:
            logger.error(f"❌ Demo failed: {e}")
            raise
        finally:
            # Cleanup
            if self.prediction_service:
                await self.prediction_service.stop()

            if self.inference_engine:
                await self.inference_engine.cleanup()


async def main():
    """Main demo runner."""
    demo = MLPredictionDemo()
    await demo.run_complete_demo()


if __name__ == "__main__":
    # Configure logging for demo
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Run demo
    asyncio.run(main())

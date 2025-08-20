"""
FlashMM Prediction Service Orchestrator

Main service that coordinates ML predictions, manages timing,
distributes results, and provides health monitoring.
"""

import asyncio
import time
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
import json

from flashmm.config.settings import get_config
from flashmm.utils.logging import get_logger
from flashmm.data.storage.data_models import OrderBookSnapshot, Trade, MarketStats
from flashmm.data.storage.redis_client import RedisClient
from flashmm.ml.models.prediction_models import EnsemblePredictionEngine, PredictionResult, PredictionMethod
from flashmm.ml.clients.azure_openai_client import AzureOpenAIClient
from flashmm.ml.fallback.rule_based_engine import RuleBasedEngine
from flashmm.ml.features.feature_extractor import FeatureExtractor
from flashmm.monitoring.telemetry.metrics_collector import MetricsCollector

logger = get_logger(__name__)


@dataclass
class PredictionServiceConfig:
    """Configuration for prediction service."""
    prediction_frequency_hz: float = 3.5          # Target prediction frequency
    max_prediction_latency_ms: float = 450.0      # Max end-to-end latency
    market_data_timeout_ms: float = 100.0         # Market data staleness timeout
    enable_feature_extraction: bool = True        # Enable feature engineering
    enable_caching: bool = True                   # Enable prediction caching
    cache_ttl_seconds: int = 30                   # Cache time-to-live
    
    # Distribution settings
    publish_to_redis: bool = True                 # Publish results to Redis
    redis_prediction_key: str = "flashmm:predictions"
    redis_channel: str = "flashmm:prediction_updates"
    
    # Health monitoring
    health_check_interval_seconds: int = 60       # Health check frequency
    performance_log_interval_seconds: int = 300   # Performance logging frequency


class PredictionTiming:
    """Manages prediction timing and frequency control."""
    
    def __init__(self, target_frequency_hz: float):
        """Initialize prediction timing.
        
        Args:
            target_frequency_hz: Target prediction frequency in Hz
        """
        self.target_frequency_hz = target_frequency_hz
        self.target_interval_ms = 1000.0 / target_frequency_hz
        self.last_prediction_time = None
        self.prediction_times: List[float] = []
        self.actual_frequency_hz = 0.0
        
    async def wait_for_next_prediction(self) -> None:
        """Wait until it's time for the next prediction."""
        if self.last_prediction_time is None:
            self.last_prediction_time = time.time()
            return
        
        elapsed_ms = (time.time() - self.last_prediction_time) * 1000
        wait_time_ms = max(0, self.target_interval_ms - elapsed_ms)
        
        if wait_time_ms > 0:
            await asyncio.sleep(wait_time_ms / 1000.0)
        
        self.last_prediction_time = time.time()
        
        # Update frequency tracking
        self.prediction_times.append(self.last_prediction_time)
        
        # Keep only recent predictions for frequency calculation
        cutoff_time = self.last_prediction_time - 60  # Last minute
        self.prediction_times = [t for t in self.prediction_times if t > cutoff_time]
        
        # Calculate actual frequency
        if len(self.prediction_times) > 1:
            time_span = self.prediction_times[-1] - self.prediction_times[0]
            if time_span > 0:
                self.actual_frequency_hz = (len(self.prediction_times) - 1) / time_span
    
    def get_frequency_stats(self) -> Dict[str, float]:
        """Get frequency statistics."""
        return {
            'target_frequency_hz': self.target_frequency_hz,
            'actual_frequency_hz': self.actual_frequency_hz,
            'frequency_deviation': abs(self.actual_frequency_hz - self.target_frequency_hz),
            'predictions_last_minute': len(self.prediction_times)
        }


class MarketDataBuffer:
    """Buffer for market data with staleness detection."""
    
    def __init__(self, max_age_ms: float = 100.0):
        """Initialize market data buffer.
        
        Args:
            max_age_ms: Maximum age of market data before considering stale
        """
        self.max_age_ms = max_age_ms
        self.order_books: Dict[str, OrderBookSnapshot] = {}
        self.trades: Dict[str, List[Trade]] = {}
        self.market_stats: Dict[str, MarketStats] = {}
        self._lock = asyncio.Lock()
    
    async def update_order_book(self, order_book: OrderBookSnapshot) -> None:
        """Update order book data."""
        async with self._lock:
            self.order_books[order_book.symbol] = order_book
    
    async def update_trades(self, symbol: str, trades: List[Trade]) -> None:
        """Update trades data."""
        async with self._lock:
            self.trades[symbol] = trades[-50:]  # Keep last 50 trades
    
    async def update_market_stats(self, stats: MarketStats) -> None:
        """Update market statistics."""
        async with self._lock:
            self.market_stats[stats.symbol] = stats
    
    async def get_market_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get current market data for symbol.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Market data dictionary or None if stale/missing
        """
        async with self._lock:
            order_book = self.order_books.get(symbol)
            if not order_book:
                return None
            
            # Check staleness
            age_ms = (datetime.utcnow() - order_book.timestamp).total_seconds() * 1000
            if age_ms > self.max_age_ms:
                logger.warning(f"Market data for {symbol} is stale ({age_ms:.1f}ms old)")
                return None
            
            return {
                'order_book': order_book,
                'trades': self.trades.get(symbol, []),
                'market_stats': self.market_stats.get(symbol)
            }
    
    async def get_data_age_ms(self, symbol: str) -> float:
        """Get age of market data in milliseconds."""
        async with self._lock:
            order_book = self.order_books.get(symbol)
            if not order_book:
                return float('inf')
            
            return (datetime.utcnow() - order_book.timestamp).total_seconds() * 1000


class PredictionDistributor:
    """Distributes prediction results to various consumers."""
    
    def __init__(self, config: PredictionServiceConfig, redis_client: Optional[RedisClient] = None):
        """Initialize prediction distributor.
        
        Args:
            config: Prediction service configuration
            redis_client: Redis client for publishing results
        """
        self.config = config
        self.redis_client = redis_client
        self.subscribers: List[Callable] = []
        self.distribution_count = 0
        self.last_distribution_time = None
    
    def add_subscriber(self, callback: Callable[[PredictionResult], None]) -> None:
        """Add prediction result subscriber.
        
        Args:
            callback: Callback function to receive prediction results
        """
        self.subscribers.append(callback)
    
    async def distribute(self, prediction: PredictionResult) -> None:
        """Distribute prediction result to all consumers.
        
        Args:
            prediction: Prediction result to distribute
        """
        try:
            start_time = time.time()
            
            # Convert to dictionary for serialization
            prediction_dict = prediction.to_dict()
            
            # Publish to Redis if enabled
            if self.config.publish_to_redis and self.redis_client:
                await self._publish_to_redis(prediction_dict)
            
            # Notify subscribers
            for subscriber in self.subscribers:
                try:
                    if asyncio.iscoroutinefunction(subscriber):
                        await subscriber(prediction)
                    else:
                        subscriber(prediction)
                except Exception as e:
                    logger.error(f"Subscriber notification failed: {e}")
            
            self.distribution_count += 1
            self.last_distribution_time = datetime.utcnow()
            
            distribution_time = (time.time() - start_time) * 1000
            if distribution_time > 10:  # Log if distribution takes > 10ms
                logger.warning(f"Slow prediction distribution: {distribution_time:.1f}ms")
                
        except Exception as e:
            logger.error(f"Prediction distribution failed: {e}")
    
    async def _publish_to_redis(self, prediction_dict: Dict[str, Any]) -> None:
        """Publish prediction to Redis."""
        try:
            # Store latest prediction
            prediction_key = f"{self.config.redis_prediction_key}:{prediction_dict['symbol']}"
            await self.redis_client.set(
                prediction_key,
                json.dumps(prediction_dict),
                ex=self.config.cache_ttl_seconds
            )
            
            # Publish update notification
            await self.redis_client.publish(
                self.config.redis_channel,
                json.dumps({
                    'type': 'prediction_update',
                    'symbol': prediction_dict['symbol'],
                    'timestamp': prediction_dict['timestamp'],
                    'direction': prediction_dict['direction'],
                    'confidence': prediction_dict['confidence']
                })
            )
            
        except Exception as e:
            logger.error(f"Redis publication failed: {e}")
    
    def get_distribution_stats(self) -> Dict[str, Any]:
        """Get distribution statistics."""
        return {
            'distribution_count': self.distribution_count,
            'subscriber_count': len(self.subscribers),
            'last_distribution_time': self.last_distribution_time.isoformat() if self.last_distribution_time else None,
            'redis_enabled': self.config.publish_to_redis and self.redis_client is not None
        }


class PredictionService:
    """Main prediction service orchestrator."""
    
    def __init__(self, config: Optional[PredictionServiceConfig] = None):
        """Initialize prediction service.
        
        Args:
            config: Service configuration
        """
        self.config = config or PredictionServiceConfig()
        self.config_manager = get_config()
        
        # Core components
        self.ensemble_engine: Optional[EnsemblePredictionEngine] = None
        self.feature_extractor: Optional[FeatureExtractor] = None
        self.redis_client: Optional[RedisClient] = None
        self.metrics_collector: Optional[MetricsCollector] = None
        
        # Service components
        self.timing = PredictionTiming(self.config.prediction_frequency_hz)
        self.market_data_buffer = MarketDataBuffer(self.config.market_data_timeout_ms)
        self.distributor: Optional[PredictionDistributor] = None
        
        # Service state
        self.is_running = False
        self.prediction_task: Optional[asyncio.Task] = None
        self.health_check_task: Optional[asyncio.Task] = None
        self.performance_log_task: Optional[asyncio.Task] = None
        
        # Statistics
        self.service_start_time = None
        self.total_predictions = 0
        self.successful_predictions = 0
        self.failed_predictions = 0
        self.average_prediction_latency_ms = 0.0
        self.last_prediction_result: Optional[PredictionResult] = None
        
        # Active symbols
        self.active_symbols: List[str] = ['SEI/USDC']  # Default symbol
    
    async def initialize(self) -> None:
        """Initialize prediction service."""
        try:
            logger.info("Initializing prediction service...")
            
            # Initialize Redis client
            redis_url = self.config_manager.get("redis_url")
            if redis_url and self.config.publish_to_redis:
                self.redis_client = RedisClient()
                await self.redis_client.initialize()
            
            # Initialize metrics collector
            self.metrics_collector = MetricsCollector()
            await self.metrics_collector.initialize()
            
            # Initialize feature extractor
            if self.config.enable_feature_extraction:
                self.feature_extractor = FeatureExtractor()
            
            # Initialize ensemble prediction engine
            azure_client = AzureOpenAIClient() if self._azure_openai_configured() else None
            rule_engine = RuleBasedEngine()
            
            self.ensemble_engine = EnsemblePredictionEngine(
                azure_client=azure_client,
                rule_engine=rule_engine,
                enable_caching=self.config.enable_caching,
                cache_ttl=self.config.cache_ttl_seconds
            )
            
            await self.ensemble_engine.initialize()
            
            # Initialize prediction distributor
            self.distributor = PredictionDistributor(self.config, self.redis_client)
            
            self.service_start_time = datetime.utcnow()
            
            logger.info("Prediction service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize prediction service: {e}")
            raise
    
    def _azure_openai_configured(self) -> bool:
        """Check if Azure OpenAI is properly configured."""
        endpoint = self.config_manager.get("azure_openai.endpoint")
        api_key = self.config_manager.get("azure_openai.api_key")
        deployment = self.config_manager.get("azure_openai.model_deployment")
        
        return all([endpoint, api_key, deployment])
    
    async def start(self) -> None:
        """Start the prediction service."""
        if self.is_running:
            logger.warning("Prediction service is already running")
            return
        
        try:
            if not self.ensemble_engine:
                await self.initialize()
            
            self.is_running = True
            
            # Start prediction loop
            self.prediction_task = asyncio.create_task(self._prediction_loop())
            
            # Start health monitoring
            if self.config.health_check_interval_seconds > 0:
                self.health_check_task = asyncio.create_task(self._health_check_loop())
            
            # Start performance logging
            if self.config.performance_log_interval_seconds > 0:
                self.performance_log_task = asyncio.create_task(self._performance_log_loop())
            
            logger.info(f"Prediction service started with {self.config.prediction_frequency_hz}Hz frequency")
            
        except Exception as e:
            logger.error(f"Failed to start prediction service: {e}")
            self.is_running = False
            raise
    
    async def stop(self) -> None:
        """Stop the prediction service."""
        if not self.is_running:
            return
        
        logger.info("Stopping prediction service...")
        
        self.is_running = False
        
        # Cancel tasks
        for task in [self.prediction_task, self.health_check_task, self.performance_log_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Cleanup resources
        if self.ensemble_engine:
            await self.ensemble_engine.cleanup()
        
        if self.redis_client:
            await self.redis_client.cleanup()
        
        logger.info("Prediction service stopped")
    
    async def _prediction_loop(self) -> None:
        """Main prediction loop."""
        logger.info("Starting prediction loop...")
        
        while self.is_running:
            try:
                # Wait for next prediction timing
                await self.timing.wait_for_next_prediction()
                
                # Process predictions for all active symbols
                await self._process_predictions()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Prediction loop error: {e}")
                await asyncio.sleep(1)  # Brief pause before retrying
    
    async def _process_predictions(self) -> None:
        """Process predictions for all active symbols."""
        prediction_tasks = []
        
        for symbol in self.active_symbols:
            task = asyncio.create_task(self._predict_for_symbol(symbol))
            prediction_tasks.append(task)
        
        # Wait for all predictions to complete (with timeout)
        try:
            await asyncio.wait_for(
                asyncio.gather(*prediction_tasks, return_exceptions=True),
                timeout=self.config.max_prediction_latency_ms / 1000.0
            )
        except asyncio.TimeoutError:
            logger.warning("Prediction timeout exceeded")
            # Cancel pending tasks
            for task in prediction_tasks:
                if not task.done():
                    task.cancel()
    
    async def _predict_for_symbol(self, symbol: str) -> None:
        """Generate prediction for specific symbol."""
        start_time = time.time()
        
        try:
            # Get market data
            market_data = await self.market_data_buffer.get_market_data(symbol)
            if not market_data:
                logger.warning(f"No market data available for {symbol}")
                return
            
            # Generate prediction
            prediction = await self.ensemble_engine.predict(
                order_book=market_data['order_book'],
                recent_trades=market_data['trades'],
                market_stats=market_data.get('market_stats'),
                prediction_horizon_ms=int(1000 / self.config.prediction_frequency_hz)
            )
            
            # Update statistics
            self.total_predictions += 1
            if prediction.validation_passed:
                self.successful_predictions += 1
            else:
                self.failed_predictions += 1
            
            # Update average latency
            prediction_latency = (time.time() - start_time) * 1000
            self.average_prediction_latency_ms = (
                (self.average_prediction_latency_ms * (self.total_predictions - 1) + prediction_latency) /
                self.total_predictions
            )
            
            self.last_prediction_result = prediction
            
            # Distribute prediction
            if self.distributor:
                await self.distributor.distribute(prediction)
            
            # Record metrics
            if self.metrics_collector:
                await self._record_prediction_metrics(prediction, prediction_latency)
            
            # Log performance warnings
            if prediction_latency > self.config.max_prediction_latency_ms:
                logger.warning(f"Prediction latency exceeded threshold: {prediction_latency:.1f}ms")
            
        except Exception as e:
            logger.error(f"Prediction failed for {symbol}: {e}")
            self.failed_predictions += 1
    
    async def _record_prediction_metrics(self, prediction: PredictionResult, latency_ms: float) -> None:
        """Record prediction metrics."""
        try:
            # Basic metrics
            await self.metrics_collector.record_counter(
                'prediction_total',
                tags={'symbol': prediction.symbol, 'method': prediction.method.value}
            )
            
            await self.metrics_collector.record_histogram(
                'prediction_latency_ms',
                latency_ms,
                tags={'symbol': prediction.symbol}
            )
            
            await self.metrics_collector.record_gauge(
                'prediction_confidence',
                prediction.confidence,
                tags={'symbol': prediction.symbol, 'direction': prediction.direction}
            )
            
            # Success/failure metrics
            if prediction.validation_passed:
                await self.metrics_collector.record_counter(
                    'prediction_success_total',
                    tags={'symbol': prediction.symbol}
                )
            else:
                await self.metrics_collector.record_counter(
                    'prediction_failure_total',
                    tags={'symbol': prediction.symbol}
                )
            
            # API success metrics
            if prediction.api_success:
                await self.metrics_collector.record_counter(
                    'azure_openai_success_total',
                    tags={'symbol': prediction.symbol}
                )
            else:
                await self.metrics_collector.record_counter(
                    'azure_openai_failure_total',
                    tags={'symbol': prediction.symbol}
                )
            
        except Exception as e:
            logger.error(f"Failed to record metrics: {e}")
    
    async def _health_check_loop(self) -> None:
        """Health check monitoring loop."""
        while self.is_running:
            try:
                await asyncio.sleep(self.config.health_check_interval_seconds)
                
                if not self.is_running:
                    break
                
                # Perform health checks
                health_status = await self.get_health_status()
                
                # Log health issues
                if health_status['status'] != 'healthy':
                    logger.warning(f"Health check failed: {health_status}")
                
                # Record health metrics
                if self.metrics_collector:
                    await self.metrics_collector.record_gauge(
                        'service_healthy',
                        1.0 if health_status['status'] == 'healthy' else 0.0
                    )
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")
    
    async def _performance_log_loop(self) -> None:
        """Performance logging loop."""
        while self.is_running:
            try:
                await asyncio.sleep(self.config.performance_log_interval_seconds)
                
                if not self.is_running:
                    break
                
                # Log performance statistics
                stats = await self.get_performance_stats()
                logger.info(f"Performance stats: {json.dumps(stats, indent=2)}")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Performance logging error: {e}")
    
    async def update_market_data(self, 
                                order_book: Optional[OrderBookSnapshot] = None,
                                trades: Optional[Dict[str, List[Trade]]] = None,
                                market_stats: Optional[MarketStats] = None) -> None:
        """Update market data in buffer.
        
        Args:
            order_book: Order book snapshot
            trades: Trades by symbol
            market_stats: Market statistics
        """
        try:
            if order_book:
                await self.market_data_buffer.update_order_book(order_book)
            
            if trades:
                for symbol, trade_list in trades.items():
                    await self.market_data_buffer.update_trades(symbol, trade_list)
            
            if market_stats:
                await self.market_data_buffer.update_market_stats(market_stats)
                
        except Exception as e:
            logger.error(f"Failed to update market data: {e}")
    
    def add_prediction_subscriber(self, callback: Callable[[PredictionResult], None]) -> None:
        """Add prediction result subscriber."""
        if self.distributor:
            self.distributor.add_subscriber(callback)
    
    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        stats = {
            'service': {
                'is_running': self.is_running,
                'start_time': self.service_start_time.isoformat() if self.service_start_time else None,
                'uptime_seconds': (datetime.utcnow() - self.service_start_time).total_seconds() if self.service_start_time else 0,
                'active_symbols': self.active_symbols,
                'total_predictions': self.total_predictions,
                'successful_predictions': self.successful_predictions,
                'failed_predictions': self.failed_predictions,
                'success_rate': self.successful_predictions / max(self.total_predictions, 1),
                'average_prediction_latency_ms': self.average_prediction_latency_ms
            },
            'timing': self.timing.get_frequency_stats(),
            'distribution': self.distributor.get_distribution_stats() if self.distributor else {},
            'last_prediction': self.last_prediction_result.to_dict() if self.last_prediction_result else None
        }
        
        # Add ensemble engine stats
        if self.ensemble_engine:
            stats['ensemble_engine'] = await self.ensemble_engine.get_performance_stats()
        
        return stats
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get service health status."""
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'components': {}
        }
        
        try:
            # Check if service is running
            if not self.is_running:
                health_status['status'] = 'unhealthy'
                health_status['components']['service'] = {'status': 'stopped'}
                return health_status
            
            health_status['components']['service'] = {'status': 'running'}
            
            # Check ensemble engine health
            if self.ensemble_engine:
                engine_health = await self.ensemble_engine.health_check()
                health_status['components']['ensemble_engine'] = engine_health
                
                if engine_health['status'] != 'healthy':
                    health_status['status'] = 'degraded'
            
            # Check market data freshness
            data_ages = {}
            for symbol in self.active_symbols:
                age_ms = await self.market_data_buffer.get_data_age_ms(symbol)
                data_ages[symbol] = age_ms
                
                if age_ms > self.config.market_data_timeout_ms * 2:  # 2x timeout threshold
                    health_status['status'] = 'degraded'
            
            health_status['components']['market_data'] = {
                'status': 'healthy' if health_status['status'] != 'degraded' else 'stale',
                'data_ages_ms': data_ages
            }
            
            # Check Redis connection
            if self.redis_client:
                try:
                    await self.redis_client.ping()
                    health_status['components']['redis'] = {'status': 'healthy'}
                except Exception:
                    health_status['components']['redis'] = {'status': 'unhealthy'}
                    health_status['status'] = 'degraded'
            
            # Check prediction frequency
            frequency_stats = self.timing.get_frequency_stats()
            frequency_deviation = frequency_stats['frequency_deviation']
            
            if frequency_deviation > self.config.prediction_frequency_hz * 0.2:  # 20% deviation
                health_status['status'] = 'degraded'
            
            health_status['components']['timing'] = {
                'status': 'healthy' if frequency_deviation <= self.config.prediction_frequency_hz * 0.2 else 'degraded',
                'frequency_deviation_hz': frequency_deviation
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            health_status['status'] = 'unhealthy'
            health_status['error'] = str(e)
        
        return health_status
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()


# Global prediction service instance
_prediction_service: Optional[PredictionService] = None


def get_prediction_service() -> PredictionService:
    """Get global prediction service instance."""
    global _prediction_service
    if _prediction_service is None:
        _prediction_service = PredictionService()
    return _prediction_service
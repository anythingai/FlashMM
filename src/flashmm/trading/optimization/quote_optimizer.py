"""
FlashMM Quote Optimizer

Advanced quote optimization system with adaptive spread sizing, competition analysis,
and ML-driven optimization to achieve ≥40% spread improvement vs baseline.
"""

import asyncio
import json
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any

import numpy as np

from flashmm.config.settings import get_config
from flashmm.data.storage.redis_client import RedisClient
from flashmm.utils.decorators import measure_latency, timeout_async
from flashmm.utils.exceptions import TradingError
from flashmm.utils.logging import get_logger

logger = get_logger(__name__)


class MarketCondition(Enum):
    """Market condition states."""
    CALM = "calm"                    # Low volatility, tight spreads
    VOLATILE = "volatile"            # High volatility, wider spreads
    TRENDING = "trending"            # Strong directional movement
    CHOPPY = "choppy"               # High frequency but low direction
    ILLIQUID = "illiquid"           # Low volume, wide spreads
    COMPETITIVE = "competitive"      # Many market makers, tight competition


@dataclass
class MarketMetrics:
    """Real-time market metrics for optimization."""
    symbol: str
    mid_price: Decimal
    best_bid: Decimal
    best_ask: Decimal
    current_spread_bps: float

    # Volatility metrics
    price_volatility_1m: float
    price_volatility_5m: float
    volume_volatility: float

    # Competition metrics
    bid_depth: Decimal
    ask_depth: Decimal
    order_book_imbalance: float
    num_market_makers: int

    # Historical performance
    recent_fills: int
    recent_volume: Decimal
    recent_pnl: Decimal

    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            'symbol': self.symbol,
            'mid_price': float(self.mid_price),
            'best_bid': float(self.best_bid),
            'best_ask': float(self.best_ask),
            'current_spread_bps': self.current_spread_bps,
            'price_volatility_1m': self.price_volatility_1m,
            'price_volatility_5m': self.price_volatility_5m,
            'volume_volatility': self.volume_volatility,
            'bid_depth': float(self.bid_depth),
            'ask_depth': float(self.ask_depth),
            'order_book_imbalance': self.order_book_imbalance,
            'num_market_makers': self.num_market_makers,
            'recent_fills': self.recent_fills,
            'recent_volume': float(self.recent_volume),
            'recent_pnl': float(self.recent_pnl),
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class OptimizationResult:
    """Quote optimization result."""
    symbol: str
    original_bid_spread_bps: float
    original_ask_spread_bps: float
    optimized_bid_spread_bps: float
    optimized_ask_spread_bps: float

    improvement_pct: float
    confidence_score: float
    market_condition: MarketCondition
    optimization_reason: str

    expected_fill_rate: float
    expected_pnl_improvement: float
    risk_adjustment: float

    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            'symbol': self.symbol,
            'original_bid_spread_bps': self.original_bid_spread_bps,
            'original_ask_spread_bps': self.original_ask_spread_bps,
            'optimized_bid_spread_bps': self.optimized_bid_spread_bps,
            'optimized_ask_spread_bps': self.optimized_ask_spread_bps,
            'improvement_pct': self.improvement_pct,
            'confidence_score': self.confidence_score,
            'market_condition': self.market_condition.value,
            'optimization_reason': self.optimization_reason,
            'expected_fill_rate': self.expected_fill_rate,
            'expected_pnl_improvement': self.expected_pnl_improvement,
            'risk_adjustment': self.risk_adjustment,
            'timestamp': self.timestamp.isoformat()
        }


class MarketConditionAnalyzer:
    """Analyzes real-time market conditions."""

    def __init__(self):
        self.price_history: dict[str, deque] = {}
        self.volume_history: dict[str, deque] = {}
        self.spread_history: dict[str, deque] = {}

        # Configuration
        self.volatility_window_1m = 60  # 1 minute in seconds
        self.volatility_window_5m = 300  # 5 minutes in seconds
        self.history_max_length = 1000

    def update_market_data(
        self,
        symbol: str,
        price: Decimal,
        volume: Decimal,
        spread_bps: float
    ) -> None:
        """Update market data for analysis."""
        timestamp = datetime.now()

        # Initialize histories if needed
        if symbol not in self.price_history:
            self.price_history[symbol] = deque(maxlen=self.history_max_length)
            self.volume_history[symbol] = deque(maxlen=self.history_max_length)
            self.spread_history[symbol] = deque(maxlen=self.history_max_length)

        # Add new data points
        self.price_history[symbol].append((timestamp, float(price)))
        self.volume_history[symbol].append((timestamp, float(volume)))
        self.spread_history[symbol].append((timestamp, spread_bps))

    def calculate_volatility(self, symbol: str, window_seconds: int) -> float:
        """Calculate price volatility over specified window."""
        if symbol not in self.price_history:
            return 0.0

        cutoff_time = datetime.now() - timedelta(seconds=window_seconds)
        recent_prices = [
            price for timestamp, price in self.price_history[symbol]
            if timestamp >= cutoff_time
        ]

        if len(recent_prices) < 2:
            return 0.0

        # Calculate returns
        returns = []
        for i in range(1, len(recent_prices)):
            ret = (recent_prices[i] - recent_prices[i-1]) / recent_prices[i-1]
            returns.append(ret)

        if not returns:
            return 0.0

        # Return annualized volatility
        volatility = np.std(returns) * np.sqrt(86400)  # Annualize to daily
        return float(volatility)

    def detect_market_condition(self, metrics: MarketMetrics) -> MarketCondition:
        """Detect current market condition."""
        # Volatility thresholds
        low_vol_threshold = 0.02   # 2% daily
        high_vol_threshold = 0.08  # 8% daily

        # Spread thresholds
        tight_spread_threshold = 5.0   # 5 bps
        wide_spread_threshold = 20.0   # 20 bps

        vol_1m = metrics.price_volatility_1m
        vol_5m = metrics.price_volatility_5m
        spread_bps = metrics.current_spread_bps

        # Determine market condition
        if vol_1m < low_vol_threshold and spread_bps < tight_spread_threshold:
            return MarketCondition.CALM
        elif vol_1m > high_vol_threshold:
            if vol_5m > vol_1m * 0.8:  # Sustained volatility
                return MarketCondition.VOLATILE
            else:
                return MarketCondition.CHOPPY
        elif abs(metrics.order_book_imbalance) > 0.3:  # Strong directional bias
            return MarketCondition.TRENDING
        elif spread_bps > wide_spread_threshold or metrics.bid_depth + metrics.ask_depth < 1000:
            return MarketCondition.ILLIQUID
        elif metrics.num_market_makers > 3 and spread_bps < tight_spread_threshold:
            return MarketCondition.COMPETITIVE
        else:
            return MarketCondition.CALM


class CompetitionAnalyzer:
    """Analyzes competition and market making landscape."""

    def __init__(self):
        self.competitor_tracking: dict[str, dict[str, Any]] = {}
        self.spread_benchmarks: dict[str, deque] = {}

    def analyze_competition(self, symbol: str, order_book: dict[str, Any]) -> dict[str, Any]:
        """Analyze competitive landscape."""
        try:
            bids = order_book.get('bids', [])
            asks = order_book.get('asks', [])

            if not bids or not asks:
                return {'num_market_makers': 0, 'avg_spread_bps': 0, 'our_rank': 0}

            _best_bid = Decimal(str(bids[0]['price']))
            _best_ask = Decimal(str(asks[0]['price']))

            # Analyze spread distribution
            spreads = []
            for i in range(min(5, len(bids), len(asks))):
                bid_price = Decimal(str(bids[i]['price']))
                ask_price = Decimal(str(asks[i]['price']))
                spread_bps = float((ask_price - bid_price) / ((ask_price + bid_price) / 2) * 10000)
                spreads.append(spread_bps)

            # Count unique market makers (simplified)
            unique_sizes = set()
            for bid in bids[:10]:
                unique_sizes.add(bid['size'])
            for ask in asks[:10]:
                unique_sizes.add(ask['size'])

            estimated_mm_count = min(len(unique_sizes), 10)

            return {
                'num_market_makers': estimated_mm_count,
                'avg_spread_bps': np.mean(spreads) if spreads else 0,
                'min_spread_bps': min(spreads) if spreads else 0,
                'spread_std_bps': np.std(spreads) if spreads else 0,
                'competitive_pressure': estimated_mm_count / 10.0  # Normalized
            }

        except Exception as e:
            logger.error(f"Competition analysis error: {e}")
            return {'num_market_makers': 0, 'avg_spread_bps': 0, 'competitive_pressure': 0}


class MLSpreadPredictor:
    """ML-based spread optimization predictor."""

    def __init__(self):
        self.feature_history: dict[str, list[dict[str, float]]] = {}
        self.performance_history: dict[str, list[dict[str, float]]] = {}
        self.model_weights = {
            'volatility_factor': 0.3,
            'competition_factor': 0.2,
            'inventory_factor': 0.2,
            'momentum_factor': 0.15,
            'volume_factor': 0.15
        }

    def predict_optimal_spreads(
        self,
        metrics: MarketMetrics,
        competition_data: dict[str, Any],
        inventory_position: float,
        ml_prediction: dict[str, Any] | None = None
    ) -> tuple[float, float]:
        """Predict optimal bid and ask spreads."""
        try:
            # Extract features
            features = self._extract_features(metrics, competition_data, inventory_position, ml_prediction)

            # Base spread calculation
            base_spread_bps = max(2.0, metrics.current_spread_bps * 0.8)  # Start 20% tighter

            # Apply ML-based adjustments
            adjustments = self._calculate_spread_adjustments(features)

            bid_spread_bps = base_spread_bps * (1 + adjustments['bid_adjustment'])
            ask_spread_bps = base_spread_bps * (1 + adjustments['ask_adjustment'])

            # Apply bounds
            min_spread_bps = 1.0
            max_spread_bps = 50.0

            bid_spread_bps = max(min_spread_bps, min(max_spread_bps, bid_spread_bps))
            ask_spread_bps = max(min_spread_bps, min(max_spread_bps, ask_spread_bps))

            return bid_spread_bps, ask_spread_bps

        except Exception as e:
            logger.error(f"ML spread prediction error: {e}")
            # Fallback to conservative spreads
            return 5.0, 5.0

    def _extract_features(
        self,
        metrics: MarketMetrics,
        competition_data: dict[str, Any],
        inventory_position: float,
        ml_prediction: dict[str, Any] | None
    ) -> dict[str, float]:
        """Extract features for ML prediction."""
        features = {
            # Volatility features
            'volatility_1m': metrics.price_volatility_1m,
            'volatility_5m': metrics.price_volatility_5m,
            'volatility_ratio': metrics.price_volatility_1m / max(metrics.price_volatility_5m, 0.001),

            # Competition features
            'num_competitors': competition_data.get('num_market_makers', 0),
            'competitive_pressure': competition_data.get('competitive_pressure', 0),
            'avg_competitor_spread': competition_data.get('avg_spread_bps', 0),

            # Inventory features
            'inventory_position': inventory_position,
            'inventory_skew': abs(inventory_position),

            # Market microstructure
            'order_book_imbalance': metrics.order_book_imbalance,
            'bid_ask_ratio': float(metrics.bid_depth / max(metrics.ask_depth, 1)),
            'volume_ratio': metrics.volume_volatility,

            # Recent performance
            'recent_fill_rate': metrics.recent_fills / 100.0,  # Normalize
            'recent_pnl_norm': float(metrics.recent_pnl) / 1000.0  # Normalize
        }

        # Add ML prediction features if available
        if ml_prediction:
            features.update({
                'ml_confidence': ml_prediction.get('confidence', 0.5),
                'ml_direction': ml_prediction.get('predicted_direction', 0),
                'ml_magnitude': ml_prediction.get('predicted_magnitude', 0)
            })

        return features

    def _calculate_spread_adjustments(self, features: dict[str, float]) -> dict[str, float]:
        """Calculate spread adjustments based on features."""
        # Volatility adjustment
        vol_adj = features['volatility_1m'] * self.model_weights['volatility_factor']

        # Competition adjustment (tighter spreads in competitive markets)
        comp_adj = -features['competitive_pressure'] * self.model_weights['competition_factor']

        # Inventory adjustment (skew spreads based on position)
        inv_adj = features['inventory_position'] * self.model_weights['inventory_factor']

        # Volume adjustment
        vol_factor_adj = features['volume_ratio'] * self.model_weights['volume_factor']

        # Apply directional skewing
        bid_adjustment = vol_adj + comp_adj - inv_adj + vol_factor_adj
        ask_adjustment = vol_adj + comp_adj + inv_adj + vol_factor_adj

        # Apply bounds to adjustments
        bid_adjustment = max(-0.5, min(1.0, bid_adjustment))
        ask_adjustment = max(-0.5, min(1.0, ask_adjustment))

        return {
            'bid_adjustment': bid_adjustment,
            'ask_adjustment': ask_adjustment
        }


class QuoteOptimizer:
    """Advanced quote optimization system."""

    def __init__(self):
        self.config = get_config()
        self.redis_client: RedisClient | None = None

        # Components
        self.market_analyzer = MarketConditionAnalyzer()
        self.competition_analyzer = CompetitionAnalyzer()
        self.ml_predictor = MLSpreadPredictor()

        # Optimization history
        self.optimization_history: dict[str, deque] = {}
        self.performance_tracking: dict[str, dict[str, Any]] = {}

        # Configuration
        self.min_improvement_threshold = self.config.get("optimization.min_improvement_pct", 20.0)  # 20%
        self.target_improvement = self.config.get("optimization.target_improvement_pct", 40.0)  # 40%
        self.max_spread_bps = self.config.get("optimization.max_spread_bps", 30.0)
        self.min_spread_bps = self.config.get("optimization.min_spread_bps", 1.0)

        # Optimization modes
        self.optimization_mode = self.config.get("optimization.mode", "adaptive")  # aggressive, conservative, adaptive

        # Background tasks
        self._optimization_task: asyncio.Task | None = None

        # Statistics
        self.total_optimizations = 0
        self.successful_optimizations = 0
        self.average_improvement_pct = 0.0

        logger.info("QuoteOptimizer initialized")

    async def initialize(self) -> None:
        """Initialize the quote optimizer."""
        try:
            # Initialize Redis client
            self.redis_client = RedisClient()
            await self.redis_client.initialize()

            # Load historical performance data
            await self._load_performance_history()

            # Start background optimization monitoring
            self._optimization_task = asyncio.create_task(self._optimization_monitoring_loop())

            logger.info("QuoteOptimizer initialization completed")

        except Exception as e:
            logger.error(f"Failed to initialize QuoteOptimizer: {e}")
            raise TradingError(f"QuoteOptimizer initialization failed: {e}") from e

    async def _optimization_monitoring_loop(self) -> None:
        """Background optimization monitoring loop."""
        while True:
            try:
                await asyncio.sleep(60)  # Run every minute
                await self._update_performance_metrics()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Optimization monitoring error: {e}")
                await asyncio.sleep(30)

    @timeout_async(0.05)  # 50ms timeout for optimization
    @measure_latency("quote_optimization")
    async def optimize_quotes(
        self,
        symbol: str,
        current_quotes: list[dict[str, Any]],
        market_data: dict[str, Any],
        position_data: dict[str, Any],
        ml_prediction: dict[str, Any] | None = None
    ) -> OptimizationResult:
        """Optimize quotes based on market conditions and ML predictions."""
        try:
            # Build market metrics
            metrics = await self._build_market_metrics(symbol, market_data, position_data)

            # Analyze competition
            competition_data = self.competition_analyzer.analyze_competition(
                symbol, market_data.get('order_book', {})
            )

            # Update market data for analysis
            mid_price = (metrics.best_bid + metrics.best_ask) / 2
            self.market_analyzer.update_market_data(
                symbol, mid_price, metrics.recent_volume, metrics.current_spread_bps
            )

            # Detect market condition
            market_condition = self.market_analyzer.detect_market_condition(metrics)

            # Calculate inventory position
            inventory_position = float(position_data.get('base_balance', 0))

            # Get ML-based optimal spreads
            optimal_bid_spread, optimal_ask_spread = self.ml_predictor.predict_optimal_spreads(
                metrics, competition_data, inventory_position, ml_prediction
            )

            # Apply optimization mode adjustments
            optimal_bid_spread, optimal_ask_spread = self._apply_optimization_mode(
                optimal_bid_spread, optimal_ask_spread, market_condition
            )

            # Calculate original spreads
            original_bid_spread, original_ask_spread = self._extract_current_spreads(current_quotes, metrics)

            # Calculate improvement
            improvement_pct = self._calculate_improvement(
                original_bid_spread, original_ask_spread,
                optimal_bid_spread, optimal_ask_spread
            )

            # Build optimization result
            result = OptimizationResult(
                symbol=symbol,
                original_bid_spread_bps=original_bid_spread,
                original_ask_spread_bps=original_ask_spread,
                optimized_bid_spread_bps=optimal_bid_spread,
                optimized_ask_spread_bps=optimal_ask_spread,
                improvement_pct=improvement_pct,
                confidence_score=self._calculate_confidence_score(metrics, competition_data, ml_prediction),
                market_condition=market_condition,
                optimization_reason=self._generate_optimization_reason(market_condition, improvement_pct),
                expected_fill_rate=self._estimate_fill_rate(optimal_bid_spread, optimal_ask_spread, metrics),
                expected_pnl_improvement=improvement_pct / 100.0 * float(metrics.recent_pnl),
                risk_adjustment=self._calculate_risk_adjustment(metrics, inventory_position)
            )

            # Store optimization result
            await self._store_optimization_result(result)

            # Update statistics
            self.total_optimizations += 1
            if improvement_pct >= self.min_improvement_threshold:
                self.successful_optimizations += 1

            # Update average improvement
            alpha = 0.05  # Exponential moving average
            self.average_improvement_pct = (
                alpha * improvement_pct + (1 - alpha) * self.average_improvement_pct
            )

            logger.debug(
                f"Quote optimization for {symbol}: {improvement_pct:.1f}% improvement "
                f"(target: {self.target_improvement:.1f}%)"
            )

            return result

        except Exception as e:
            logger.error(f"Quote optimization failed for {symbol}: {e}")
            raise TradingError(f"Quote optimization failed: {e}") from e

    async def _build_market_metrics(
        self,
        symbol: str,
        market_data: dict[str, Any],
        position_data: dict[str, Any]
    ) -> MarketMetrics:
        """Build market metrics from available data."""
        order_book = market_data.get('order_book', {})
        bids = order_book.get('bids', [])
        asks = order_book.get('asks', [])

        if not bids or not asks:
            raise TradingError(f"No order book data for {symbol}")

        best_bid = Decimal(str(bids[0]['price']))
        best_ask = Decimal(str(asks[0]['price']))
        mid_price = (best_bid + best_ask) / 2

        current_spread_bps = float((best_ask - best_bid) / mid_price * 10000)

        # Calculate order book metrics
        bid_depth = sum(Decimal(str(bid['size'])) for bid in bids[:5]) or Decimal('0')
        ask_depth = sum(Decimal(str(ask['size'])) for ask in asks[:5]) or Decimal('0')

        imbalance = float((bid_depth - ask_depth) / (bid_depth + ask_depth)) if (bid_depth + ask_depth) > 0 else 0

        # Get volatility from analyzer
        vol_1m = self.market_analyzer.calculate_volatility(symbol, 60)
        vol_5m = self.market_analyzer.calculate_volatility(symbol, 300)

        return MarketMetrics(
            symbol=symbol,
            mid_price=mid_price,
            best_bid=best_bid,
            best_ask=best_ask,
            current_spread_bps=current_spread_bps,
            price_volatility_1m=vol_1m,
            price_volatility_5m=vol_5m,
            volume_volatility=market_data.get('volume_volatility', 0.0),
            bid_depth=bid_depth,
            ask_depth=ask_depth,
            order_book_imbalance=imbalance,
            num_market_makers=len({bid['size'] for bid in bids[:10]}),
            recent_fills=position_data.get('recent_fills', 0),
            recent_volume=Decimal(str(position_data.get('recent_volume', 0))),
            recent_pnl=Decimal(str(position_data.get('recent_pnl', 0)))
        )

    def _apply_optimization_mode(
        self,
        bid_spread: float,
        ask_spread: float,
        market_condition: MarketCondition
    ) -> tuple[float, float]:
        """Apply optimization mode adjustments."""
        if self.optimization_mode == "aggressive":
            # More aggressive spread tightening
            bid_spread *= 0.8
            ask_spread *= 0.8
        elif self.optimization_mode == "conservative":
            # More conservative approach
            bid_spread *= 1.1
            ask_spread *= 1.1
        else:  # adaptive
            # Adjust based on market condition
            if market_condition == MarketCondition.COMPETITIVE:
                bid_spread *= 0.9
                ask_spread *= 0.9
            elif market_condition == MarketCondition.VOLATILE:
                bid_spread *= 1.2
                ask_spread *= 1.2
            elif market_condition == MarketCondition.ILLIQUID:
                bid_spread *= 1.3
                ask_spread *= 1.3

        # Apply bounds
        bid_spread = max(self.min_spread_bps, min(self.max_spread_bps, bid_spread))
        ask_spread = max(self.min_spread_bps, min(self.max_spread_bps, ask_spread))

        return bid_spread, ask_spread

    def _extract_current_spreads(
        self,
        current_quotes: list[dict[str, Any]],
        metrics: MarketMetrics
    ) -> tuple[float, float]:
        """Extract current bid and ask spreads from quotes."""
        if not current_quotes:
            return metrics.current_spread_bps / 2, metrics.current_spread_bps / 2

        bid_quotes = [q for q in current_quotes if q.get('side') == 'buy']
        ask_quotes = [q for q in current_quotes if q.get('side') == 'sell']

        if not bid_quotes or not ask_quotes:
            return metrics.current_spread_bps / 2, metrics.current_spread_bps / 2

        # Get best quotes
        best_bid_quote = max(bid_quotes, key=lambda x: x.get('price', 0))
        best_ask_quote = min(ask_quotes, key=lambda x: x.get('price', float('inf')))

        bid_price = Decimal(str(best_bid_quote['price']))
        ask_price = Decimal(str(best_ask_quote['price']))

        # Calculate spreads from mid
        bid_spread_bps = float((metrics.mid_price - bid_price) / metrics.mid_price * 10000)
        ask_spread_bps = float((ask_price - metrics.mid_price) / metrics.mid_price * 10000)

        return bid_spread_bps, ask_spread_bps

    def _calculate_improvement(
        self,
        orig_bid: float,
        orig_ask: float,
        opt_bid: float,
        opt_ask: float
    ) -> float:
        """Calculate percentage improvement in spreads."""
        if orig_bid + orig_ask <= 0:
            return 0.0

        original_total_spread = orig_bid + orig_ask
        optimized_total_spread = opt_bid + opt_ask

        # Improvement is reduction in spread
        improvement_pct = (original_total_spread - optimized_total_spread) / original_total_spread * 100

        return max(0.0, improvement_pct)

    def _calculate_confidence_score(
        self,
        metrics: MarketMetrics,
        competition_data: dict[str, Any],
        ml_prediction: dict[str, Any] | None
    ) -> float:
        """Calculate confidence score for optimization."""
        score = 0.5  # Base score

        # Higher confidence with more market data
        if metrics.recent_fills > 10:
            score += 0.2

        # Higher confidence with ML prediction
        if ml_prediction and ml_prediction.get('confidence', 0) > 0.7:
            score += 0.2

        # Lower confidence in highly volatile markets
        if metrics.price_volatility_1m > 0.05:
            score -= 0.1

        # Higher confidence with good competition data
        if competition_data.get('num_market_makers', 0) > 2:
            score += 0.1

        return max(0.1, min(1.0, score))

    def _generate_optimization_reason(self, market_condition: MarketCondition, improvement_pct: float) -> str:
        """Generate human-readable optimization reason."""
        if improvement_pct >= self.target_improvement:
            return f"Achieved {improvement_pct:.1f}% spread improvement in {market_condition.value} market"
        elif improvement_pct >= self.min_improvement_threshold:
            return f"Moderate {improvement_pct:.1f}% improvement due to {market_condition.value} conditions"
        else:
            return f"Limited optimization potential in {market_condition.value} market"

    def _estimate_fill_rate(self, bid_spread: float, ask_spread: float, metrics: MarketMetrics) -> float:
        """Estimate expected fill rate based on spread tightness."""
        avg_spread = (bid_spread + ask_spread) / 2

        # Empirical model: tighter spreads = higher fill rate
        base_fill_rate = 0.3  # 30% base fill rate

        # Adjust based on spread tightness relative to market
        if avg_spread < metrics.current_spread_bps * 0.8:
            fill_rate_multiplier = 1.5  # Higher fill rate for tight spreads
        elif avg_spread > metrics.current_spread_bps * 1.2:
            fill_rate_multiplier = 0.7  # Lower fill rate for wide spreads
        else:
            fill_rate_multiplier = 1.0

        # Adjust for market conditions
        if metrics.price_volatility_1m > 0.05:  # High volatility
            fill_rate_multiplier *= 1.2

        if metrics.recent_fills > 20:  # Good recent performance
            fill_rate_multiplier *= 1.1

        estimated_fill_rate = base_fill_rate * fill_rate_multiplier
        return max(0.1, min(0.9, estimated_fill_rate))

    def _calculate_risk_adjustment(self, metrics: MarketMetrics, inventory_position: float) -> float:
        """Calculate risk adjustment factor."""
        risk_factor = 0.0

        # Volatility risk
        risk_factor += metrics.price_volatility_1m * 0.5

        # Inventory risk
        risk_factor += abs(inventory_position) * 0.3

        # Market condition risk
        if metrics.order_book_imbalance > 0.5:
            risk_factor += 0.2

        return max(0.0, min(1.0, risk_factor))

    async def _store_optimization_result(self, result: OptimizationResult) -> None:
        """Store optimization result for tracking."""
        try:
            # Store in memory
            if result.symbol not in self.optimization_history:
                self.optimization_history[result.symbol] = deque(maxlen=100)

            self.optimization_history[result.symbol].append(result)

            # Store in Redis
            if self.redis_client:
                key = f"optimization_result:{result.symbol}:{int(result.timestamp.timestamp())}"
                await self.redis_client.set(
                    key,
                    json.dumps(result.to_dict()),
                    expire=3600  # 1 hour
                )

        except Exception as e:
            logger.error(f"Failed to store optimization result: {e}")

    async def _load_performance_history(self) -> None:
        """Load historical performance data."""
        try:
            if not self.redis_client:
                return

            # Load recent optimization results
            keys = await self.redis_client.keys("optimization_result:*")
            for key in keys[-100:]:  # Last 100 results
                data = await self.redis_client.get(key)
                if data:
                    result_dict = json.loads(data)
                    symbol = result_dict['symbol']

                    if symbol not in self.optimization_history:
                        self.optimization_history[symbol] = deque(maxlen=100)

                    # Reconstruct optimization result
                    result = OptimizationResult(
                        symbol=result_dict['symbol'],
                        original_bid_spread_bps=result_dict['original_bid_spread_bps'],
                        original_ask_spread_bps=result_dict['original_ask_spread_bps'],
                        optimized_bid_spread_bps=result_dict['optimized_bid_spread_bps'],
                        optimized_ask_spread_bps=result_dict['optimized_ask_spread_bps'],
                        improvement_pct=result_dict['improvement_pct'],
                        confidence_score=result_dict['confidence_score'],
                        market_condition=MarketCondition(result_dict['market_condition']),
                        optimization_reason=result_dict['optimization_reason'],
                        expected_fill_rate=result_dict['expected_fill_rate'],
                        expected_pnl_improvement=result_dict['expected_pnl_improvement'],
                        risk_adjustment=result_dict['risk_adjustment'],
                        timestamp=datetime.fromisoformat(result_dict['timestamp'])
                    )

                    self.optimization_history[symbol].append(result)

            logger.info(f"Loaded optimization history for {len(self.optimization_history)} symbols")

        except Exception as e:
            logger.warning(f"Failed to load performance history: {e}")

    async def _update_performance_metrics(self) -> None:
        """Update performance tracking metrics."""
        try:
            for symbol, results in self.optimization_history.items():
                if not results:
                    continue

                recent_results = [r for r in results if r.timestamp > datetime.now() - timedelta(hours=1)]
                if not recent_results:
                    continue

                # Calculate performance metrics
                avg_improvement = sum(r.improvement_pct for r in recent_results) / len(recent_results)
                avg_confidence = sum(r.confidence_score for r in recent_results) / len(recent_results)

                # Update tracking
                self.performance_tracking[symbol] = {
                    'recent_optimizations': len(recent_results),
                    'average_improvement_pct': avg_improvement,
                    'average_confidence': avg_confidence,
                    'target_achievement_rate': len([r for r in recent_results if r.improvement_pct >= self.target_improvement]) / len(recent_results),
                    'last_updated': datetime.now().isoformat()
                }

        except Exception as e:
            logger.error(f"Performance metrics update error: {e}")

    # Public interface methods
    def get_optimization_performance(self, symbol: str | None = None) -> dict[str, Any]:
        """Get optimization performance summary."""
        if symbol:
            # Symbol-specific performance
            symbol_data = self.performance_tracking.get(symbol, {})
            history = self.optimization_history.get(symbol, [])

            return {
                'symbol': symbol,
                'total_optimizations': len(history),
                'recent_performance': symbol_data,
                'recent_results': [r.to_dict() for r in list(history)[-10:]]  # Last 10 results
            }
        else:
            # Overall performance
            return {
                'total_optimizations': self.total_optimizations,
                'successful_optimizations': self.successful_optimizations,
                'success_rate': self.successful_optimizations / max(self.total_optimizations, 1),
                'average_improvement_pct': round(self.average_improvement_pct, 2),
                'target_improvement_pct': self.target_improvement,
                'symbols_tracked': list(self.optimization_history.keys()),
                'optimization_mode': self.optimization_mode,
                'performance_by_symbol': self.performance_tracking
            }

    def get_market_condition_analysis(self, symbol: str) -> dict[str, Any]:
        """Get current market condition analysis for symbol."""
        try:
            if symbol not in self.optimization_history or not self.optimization_history[symbol]:
                return {'error': 'No optimization history for symbol'}

            recent_result = self.optimization_history[symbol][-1]

            return {
                'symbol': symbol,
                'current_market_condition': recent_result.market_condition.value,
                'confidence_score': recent_result.confidence_score,
                'last_optimization': recent_result.timestamp.isoformat(),
                'recent_improvement_pct': recent_result.improvement_pct,
                'optimization_reason': recent_result.optimization_reason
            }

        except Exception as e:
            logger.error(f"Market condition analysis error for {symbol}: {e}")
            return {'error': str(e)}

    async def calibrate_optimization_parameters(self, symbol: str) -> None:
        """Calibrate optimization parameters based on recent performance."""
        try:
            if symbol not in self.optimization_history:
                return

            recent_results = [
                r for r in self.optimization_history[symbol]
                if r.timestamp > datetime.now() - timedelta(hours=24)
            ]

            if len(recent_results) < 10:  # Need sufficient data
                return

            # Analyze recent performance
            improvements = [r.improvement_pct for r in recent_results]
            avg_improvement = sum(improvements) / len(improvements)

            # Adjust optimization mode if performance is poor
            if avg_improvement < self.min_improvement_threshold:
                if self.optimization_mode == "aggressive":
                    self.optimization_mode = "adaptive"
                    logger.info(f"Switched to adaptive mode for {symbol} due to poor performance")
                elif self.optimization_mode == "adaptive":
                    self.optimization_mode = "conservative"
                    logger.info(f"Switched to conservative mode for {symbol} due to poor performance")
            elif avg_improvement > self.target_improvement:
                if self.optimization_mode == "conservative":
                    self.optimization_mode = "adaptive"
                    logger.info(f"Switched to adaptive mode for {symbol} due to good performance")
                elif self.optimization_mode == "adaptive":
                    self.optimization_mode = "aggressive"
                    logger.info(f"Switched to aggressive mode for {symbol} due to excellent performance")

        except Exception as e:
            logger.error(f"Parameter calibration error for {symbol}: {e}")

    async def cleanup(self) -> None:
        """Cleanup resources and stop background tasks."""
        # Cancel background tasks
        if self._optimization_task and not self._optimization_task.done():
            self._optimization_task.cancel()
            try:
                await self._optimization_task
            except asyncio.CancelledError:
                pass

        # Save final performance data
        if self.redis_client:
            try:
                performance_data = {
                    'total_optimizations': self.total_optimizations,
                    'successful_optimizations': self.successful_optimizations,
                    'average_improvement_pct': self.average_improvement_pct,
                    'performance_tracking': self.performance_tracking
                }

                await self.redis_client.set(
                    "quote_optimizer_performance",
                    json.dumps(performance_data),
                    expire=86400  # 24 hours
                )
            except Exception as e:
                logger.error(f"Failed to save performance data: {e}")

        # Close Redis connection
        if self.redis_client:
            await self.redis_client.close()

        logger.info("QuoteOptimizer cleanup completed")


# Global optimizer instance
_quote_optimizer: QuoteOptimizer | None = None


async def get_quote_optimizer() -> QuoteOptimizer:
    """Get global quote optimizer instance."""
    global _quote_optimizer
    if _quote_optimizer is None:
        _quote_optimizer = QuoteOptimizer()
        await _quote_optimizer.initialize()
    return _quote_optimizer


async def cleanup_quote_optimizer() -> None:
    """Cleanup global quote optimizer."""
    global _quote_optimizer
    if _quote_optimizer:
        await _quote_optimizer.cleanup()
        _quote_optimizer = None

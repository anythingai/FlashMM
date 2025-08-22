"""
FlashMM Market Risk Monitor

Comprehensive market risk monitoring system with:
- Real-time volatility detection and analysis
- Market regime change detection
- Liquidity risk assessment
- Correlation breakdown detection
- Market stress condition monitoring
"""

import asyncio
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np

from flashmm.config.settings import get_config
from flashmm.utils.decorators import measure_latency, timeout_async
from flashmm.utils.exceptions import RiskError
from flashmm.utils.logging import get_logger

logger = get_logger(__name__)


class MarketRegime(Enum):
    """Market regime classifications."""
    CALM = "calm"
    TRENDING = "trending"
    VOLATILE = "volatile"
    CRISIS = "crisis"
    RECOVERY = "recovery"
    UNKNOWN = "unknown"


class VolatilityLevel(Enum):
    """Volatility level classifications."""
    VERY_LOW = "very_low"
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    EXTREME = "extreme"


class LiquidityCondition(Enum):
    """Liquidity condition classifications."""
    EXCELLENT = "excellent"
    GOOD = "good"
    NORMAL = "normal"
    POOR = "poor"
    CRITICAL = "critical"


@dataclass
class MarketSnapshot:
    """Market data snapshot for analysis."""
    timestamp: datetime
    symbol: str
    price: float
    volume: float
    bid_price: float
    ask_price: float
    bid_size: float
    ask_size: float

    @property
    def spread_bps(self) -> float:
        """Calculate bid-ask spread in basis points."""
        if self.ask_price > 0 and self.bid_price > 0:
            mid_price = (self.bid_price + self.ask_price) / 2
            if mid_price > 0:
                return ((self.ask_price - self.bid_price) / mid_price) * 10000
        return 0.0

    @property
    def mid_price(self) -> float:
        """Calculate mid price."""
        return (self.bid_price + self.ask_price) / 2 if (self.bid_price > 0 and self.ask_price > 0) else self.price


@dataclass
class VolatilityMetrics:
    """Volatility analysis metrics."""
    current_volatility: float
    volatility_percentile: float
    volatility_level: VolatilityLevel
    volatility_trend: str  # 'increasing', 'decreasing', 'stable'
    garch_forecast: float
    realized_vol_1h: float
    realized_vol_4h: float
    realized_vol_24h: float
    vol_of_vol: float  # Volatility of volatility
    volatility_clustering: bool


@dataclass
class LiquidityMetrics:
    """Liquidity analysis metrics."""
    bid_ask_spread_bps: float
    spread_percentile: float
    order_book_depth: float
    depth_ratio: float  # bid depth / ask depth
    price_impact: float
    turnover_rate: float
    liquidity_score: float  # 0-1 composite score
    liquidity_condition: LiquidityCondition
    market_impact_cost: float


@dataclass
class MarketStressIndicators:
    """Market stress indicators."""
    volatility_spike: bool
    volume_spike: bool
    spread_widening: bool
    correlation_breakdown: bool
    liquidity_evaporation: bool
    price_gaps: bool
    stress_score: float  # 0-1 composite stress score
    stress_level: str  # 'low', 'medium', 'high', 'extreme'


class VolatilityDetector:
    """Advanced volatility detection and analysis."""

    def __init__(self, lookback_periods: int = 288):  # 24 hours of 5-min data
        self.lookback_periods = lookback_periods
        self.price_history: dict[str, deque] = {}
        self.return_history: dict[str, deque] = {}
        self.volatility_history: dict[str, deque] = {}

        # GARCH parameters (simplified)
        self.garch_alpha = 0.1
        self.garch_beta = 0.85
        self.garch_omega = 0.00001

    def update_price_data(self, symbol: str, price: float, timestamp: datetime):
        """Update price data for volatility calculation."""
        if symbol not in self.price_history:
            self.price_history[symbol] = deque(maxlen=self.lookback_periods)
            self.return_history[symbol] = deque(maxlen=self.lookback_periods)
            self.volatility_history[symbol] = deque(maxlen=self.lookback_periods)

        self.price_history[symbol].append((timestamp, price))

        # Calculate return if we have previous price
        if len(self.price_history[symbol]) >= 2:
            prev_price = self.price_history[symbol][-2][1]
            if prev_price > 0:
                return_pct = (price - prev_price) / prev_price
                self.return_history[symbol].append(return_pct)

    async def calculate_volatility_metrics(self, symbol: str) -> VolatilityMetrics | None:
        """Calculate comprehensive volatility metrics."""
        try:
            returns = list(self.return_history.get(symbol, []))
            if len(returns) < 20:  # Need minimum data
                return None

            # Calculate different timeframe volatilities
            current_vol = self._calculate_realized_volatility(returns[-12:])  # 1 hour
            vol_1h = self._calculate_realized_volatility(returns[-12:])
            vol_4h = self._calculate_realized_volatility(returns[-48:])
            vol_24h = self._calculate_realized_volatility(returns)

            # Calculate volatility percentile
            vol_history = list(self.volatility_history.get(symbol, []))
            if len(vol_history) >= 20:
                vol_percentile = self._calculate_percentile(current_vol, vol_history)
            else:
                vol_percentile = 50.0

            # Classify volatility level
            vol_level = self._classify_volatility_level(vol_percentile)

            # Detect volatility trend
            vol_trend = self._detect_volatility_trend(vol_history[-10:] if vol_history else [])

            # GARCH forecast
            garch_forecast = self._calculate_garch_forecast(returns)

            # Volatility of volatility
            vol_of_vol = np.std(vol_history[-20:]) if len(vol_history) >= 20 else 0.0

            # Volatility clustering detection
            clustering = self._detect_volatility_clustering(returns)

            # Store current volatility
            self.volatility_history[symbol].append(current_vol)

            return VolatilityMetrics(
                current_volatility=current_vol,
                volatility_percentile=vol_percentile,
                volatility_level=vol_level,
                volatility_trend=vol_trend,
                garch_forecast=garch_forecast,
                realized_vol_1h=vol_1h,
                realized_vol_4h=vol_4h,
                realized_vol_24h=vol_24h,
                vol_of_vol=float(vol_of_vol),
                volatility_clustering=clustering
            )

        except Exception as e:
            logger.error(f"Error calculating volatility metrics for {symbol}: {e}")
            return None

    def _calculate_realized_volatility(self, returns: list[float]) -> float:
        """Calculate realized volatility from returns."""
        if len(returns) < 2:
            return 0.0

        # Annualized volatility (assuming 5-minute intervals)
        return np.std(returns) * np.sqrt(288 * 365)  # 288 5-min periods per day

    def _calculate_percentile(self, value: float, history: list[float]) -> float:
        """Calculate percentile of value in historical distribution."""
        if not history:
            return 50.0

        sorted_history = sorted(history)
        position = 0
        for i, hist_val in enumerate(sorted_history):
            if value <= hist_val:
                position = i
                break
        else:
            position = len(sorted_history)

        return (position / len(sorted_history)) * 100

    def _classify_volatility_level(self, percentile: float) -> VolatilityLevel:
        """Classify volatility level based on percentile."""
        if percentile >= 95:
            return VolatilityLevel.EXTREME
        elif percentile >= 80:
            return VolatilityLevel.HIGH
        elif percentile >= 20:
            return VolatilityLevel.NORMAL
        elif percentile >= 5:
            return VolatilityLevel.LOW
        else:
            return VolatilityLevel.VERY_LOW

    def _detect_volatility_trend(self, recent_vols: list[float]) -> str:
        """Detect volatility trend."""
        if len(recent_vols) < 5:
            return 'stable'

        # Simple trend detection using linear regression slope
        x = list(range(len(recent_vols)))
        if len(recent_vols) >= 2:
            slope = np.polyfit(x, recent_vols, 1)[0]
            if slope > 0.01:
                return 'increasing'
            elif slope < -0.01:
                return 'decreasing'

        return 'stable'

    def _calculate_garch_forecast(self, returns: list[float]) -> float:
        """Simple GARCH(1,1) volatility forecast."""
        if len(returns) < 10:
            return 0.0

        # Simplified GARCH calculation
        variance_forecast = self.garch_omega
        if len(returns) >= 2:
            last_return = returns[-1]
            last_variance = np.var(returns[-10:])  # Use recent variance as proxy

            variance_forecast = (self.garch_omega +
                               self.garch_alpha * (last_return ** 2) +
                               self.garch_beta * last_variance)

        return np.sqrt(variance_forecast * 288 * 365)  # Annualized

    def _detect_volatility_clustering(self, returns: list[float]) -> bool:
        """Detect volatility clustering."""
        if len(returns) < 20:
            return False

        # Calculate squared returns as proxy for volatility
        squared_returns = [r ** 2 for r in returns]

        # Check for autocorrelation in squared returns
        if len(squared_returns) >= 10:
            recent_vol = np.mean(squared_returns[-5:])
            historical_vol = np.mean(squared_returns[:-5])

            return bool(recent_vol > 2 * historical_vol)  # Simple clustering detection

        return False


class RegimeChangeDetector:
    """Market regime change detection system."""

    def __init__(self):
        self.regime_history: dict[str, list[tuple[datetime, MarketRegime]]] = {}
        self.regime_indicators: dict[str, deque] = {}

        # Regime detection parameters
        self.volatility_threshold_high = 0.3  # 30% annualized
        self.volatility_threshold_extreme = 0.6  # 60% annualized
        self.trend_strength_threshold = 0.1  # 10% directional move
        self.crisis_correlation_threshold = 0.8  # High correlation indicates crisis

    async def detect_regime_change(self,
                                 symbol: str,
                                 volatility_metrics: VolatilityMetrics,
                                 price_data: list[float],
                                 volume_data: list[float]) -> MarketRegime:
        """Detect current market regime."""
        try:
            if symbol not in self.regime_indicators:
                self.regime_indicators[symbol] = deque(maxlen=100)

            # Calculate regime indicators
            vol_indicator = self._calculate_volatility_regime_indicator(volatility_metrics)
            trend_indicator = self._calculate_trend_indicator(price_data)
            stress_indicator = self._calculate_stress_indicator(volatility_metrics, volume_data)

            # Determine regime based on indicators
            regime = self._classify_regime(vol_indicator, trend_indicator, stress_indicator)

            # Update regime history
            if symbol not in self.regime_history:
                self.regime_history[symbol] = []

            # Check for regime change
            current_time = datetime.now()
            if (not self.regime_history[symbol] or
                self.regime_history[symbol][-1][1] != regime):

                self.regime_history[symbol].append((current_time, regime))
                logger.info(f"Market regime change detected for {symbol}: {regime.value}")

                # Keep only recent history
                if len(self.regime_history[symbol]) > 50:
                    self.regime_history[symbol] = self.regime_history[symbol][-25:]

            return regime

        except Exception as e:
            logger.error(f"Error detecting regime change for {symbol}: {e}")
            return MarketRegime.UNKNOWN

    def _calculate_volatility_regime_indicator(self, vol_metrics: VolatilityMetrics) -> float:
        """Calculate volatility-based regime indicator."""
        current_vol = vol_metrics.current_volatility

        if current_vol > self.volatility_threshold_extreme:
            return 1.0  # Crisis regime
        elif current_vol > self.volatility_threshold_high:
            return 0.7  # High volatility regime
        else:
            return current_vol / self.volatility_threshold_high  # Normalized

    def _calculate_trend_indicator(self, price_data: list[float]) -> float:
        """Calculate trend strength indicator."""
        if len(price_data) < 10:
            return 0.0

        # Calculate directional movement
        recent_prices = price_data[-20:] if len(price_data) >= 20 else price_data
        if len(recent_prices) < 2:
            return 0.0

        price_change = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
        return abs(price_change)  # Absolute trend strength

    def _calculate_stress_indicator(self, vol_metrics: VolatilityMetrics, volume_data: list[float]) -> float:
        """Calculate market stress indicator."""
        stress_score = 0.0

        # Volatility component
        if vol_metrics.volatility_level in [VolatilityLevel.HIGH, VolatilityLevel.EXTREME]:
            stress_score += 0.3

        # Volatility clustering component
        if vol_metrics.volatility_clustering:
            stress_score += 0.2

        # Volume spike component
        if len(volume_data) >= 10:
            recent_volume = np.mean(volume_data[-5:])
            historical_volume = np.mean(volume_data[:-5])
            if historical_volume > 0 and recent_volume > 2 * historical_volume:
                stress_score += 0.2

        # Vol-of-vol component
        if vol_metrics.vol_of_vol > 0.1:  # High vol-of-vol indicates instability
            stress_score += 0.3

        return min(1.0, stress_score)

    def _classify_regime(self, vol_indicator: float, trend_indicator: float, stress_indicator: float) -> MarketRegime:
        """Classify market regime based on indicators."""
        # Crisis regime
        if stress_indicator > 0.8 or vol_indicator > 0.9:
            return MarketRegime.CRISIS

        # Volatile regime
        if vol_indicator > 0.6 or stress_indicator > 0.6:
            return MarketRegime.VOLATILE

        # Trending regime
        if trend_indicator > self.trend_strength_threshold and vol_indicator < 0.4:
            return MarketRegime.TRENDING

        # Recovery regime (declining volatility after crisis)
        if vol_indicator > 0.3 and stress_indicator < 0.3:
            return MarketRegime.RECOVERY

        # Calm regime
        if vol_indicator < 0.3 and stress_indicator < 0.2:
            return MarketRegime.CALM

        return MarketRegime.UNKNOWN


class LiquidityRiskAssessor:
    """Liquidity risk assessment system."""

    def __init__(self):
        self.spread_history: dict[str, deque] = {}
        self.depth_history: dict[str, deque] = {}
        self.volume_history: dict[str, deque] = {}

    async def assess_liquidity_risk(self, snapshot: MarketSnapshot) -> LiquidityMetrics:
        """Assess liquidity risk from market snapshot."""
        try:
            symbol = snapshot.symbol

            # Initialize history if needed
            if symbol not in self.spread_history:
                self.spread_history[symbol] = deque(maxlen=288)  # 24 hours
                self.depth_history[symbol] = deque(maxlen=288)
                self.volume_history[symbol] = deque(maxlen=288)

            # Update history
            self.spread_history[symbol].append(snapshot.spread_bps)
            depth = snapshot.bid_size + snapshot.ask_size
            self.depth_history[symbol].append(depth)
            self.volume_history[symbol].append(snapshot.volume)

            # Calculate spread percentile
            spread_history = list(self.spread_history[symbol])
            spread_percentile = self._calculate_percentile(snapshot.spread_bps, spread_history)

            # Calculate depth ratio
            depth_ratio = (snapshot.bid_size / max(snapshot.ask_size, 0.001)) if snapshot.ask_size > 0 else 1.0

            # Estimate price impact
            price_impact = self._estimate_price_impact(snapshot)

            # Calculate turnover rate
            volume_history = list(self.volume_history[symbol])
            turnover_rate = self._calculate_turnover_rate(volume_history)

            # Composite liquidity score
            liquidity_score = self._calculate_liquidity_score(
                snapshot.spread_bps, depth, snapshot.volume, price_impact
            )

            # Classify liquidity condition
            liquidity_condition = self._classify_liquidity_condition(liquidity_score)

            # Market impact cost
            market_impact_cost = self._estimate_market_impact_cost(snapshot)

            return LiquidityMetrics(
                bid_ask_spread_bps=snapshot.spread_bps,
                spread_percentile=spread_percentile,
                order_book_depth=depth,
                depth_ratio=depth_ratio,
                price_impact=price_impact,
                turnover_rate=turnover_rate,
                liquidity_score=liquidity_score,
                liquidity_condition=liquidity_condition,
                market_impact_cost=market_impact_cost
            )

        except Exception as e:
            logger.error(f"Error assessing liquidity risk for {snapshot.symbol}: {e}")
            return self._default_liquidity_metrics()

    def _calculate_percentile(self, value: float, history: list[float]) -> float:
        """Calculate percentile of value in historical distribution."""
        if not history or len(history) < 5:
            return 50.0

        position = sum(1 for h in history if h <= value)
        return (position / len(history)) * 100

    def _estimate_price_impact(self, snapshot: MarketSnapshot) -> float:
        """Estimate price impact as percentage."""
        if snapshot.mid_price > 0 and (snapshot.bid_size > 0 or snapshot.ask_size > 0):
            # Simple price impact model: spread / 2
            return snapshot.spread_bps / 2 / 10000  # Convert to percentage
        return 0.0

    def _calculate_turnover_rate(self, volume_history: list[float]) -> float:
        """Calculate volume turnover rate."""
        if not volume_history:
            return 0.0

        recent_volume = np.mean(volume_history[-12:]) if len(volume_history) >= 12 else np.mean(volume_history)
        return float(recent_volume / 1000.0)  # Normalized turnover rate

    def _calculate_liquidity_score(self, spread: float, depth: float, volume: float, price_impact: float) -> float:
        """Calculate composite liquidity score (0-1, higher is better)."""
        # Spread component (lower is better)
        spread_score = max(0.0, 1.0 - (spread / 100.0))  # Normalize spread

        # Depth component (higher is better)
        depth_score = min(1.0, depth / 10000.0)  # Normalize depth

        # Volume component (higher is better)
        volume_score = min(1.0, volume / 100000.0)  # Normalize volume

        # Price impact component (lower is better)
        impact_score = max(0.0, 1.0 - (price_impact * 10))

        # Weighted composite score
        weights = [0.3, 0.3, 0.2, 0.2]
        scores = [spread_score, depth_score, volume_score, impact_score]

        return sum(w * s for w, s in zip(weights, scores, strict=False))

    def _classify_liquidity_condition(self, liquidity_score: float) -> LiquidityCondition:
        """Classify liquidity condition based on score."""
        if liquidity_score >= 0.8:
            return LiquidityCondition.EXCELLENT
        elif liquidity_score >= 0.6:
            return LiquidityCondition.GOOD
        elif liquidity_score >= 0.4:
            return LiquidityCondition.NORMAL
        elif liquidity_score >= 0.2:
            return LiquidityCondition.POOR
        else:
            return LiquidityCondition.CRITICAL

    def _estimate_market_impact_cost(self, snapshot: MarketSnapshot) -> float:
        """Estimate market impact cost for a standard trade size."""
        # Simple model: cost increases with spread and decreases with depth
        standard_trade_size = 1000.0  # Standard trade size in USDC

        if snapshot.bid_size > 0 and snapshot.ask_size > 0:
            avg_depth = (snapshot.bid_size + snapshot.ask_size) / 2
            impact_factor = standard_trade_size / max(avg_depth, 100.0)
            base_cost = snapshot.spread_bps / 10000  # Base spread cost

            return base_cost * (1 + impact_factor)

        return snapshot.spread_bps / 10000

    def _default_liquidity_metrics(self) -> LiquidityMetrics:
        """Return default liquidity metrics."""
        return LiquidityMetrics(
            bid_ask_spread_bps=0.0,
            spread_percentile=50.0,
            order_book_depth=0.0,
            depth_ratio=1.0,
            price_impact=0.0,
            turnover_rate=0.0,
            liquidity_score=0.5,
            liquidity_condition=LiquidityCondition.NORMAL,
            market_impact_cost=0.0
        )


class MarketRiskMonitor:
    """Comprehensive market risk monitoring system."""

    def __init__(self):
        self.config = get_config()
        self.volatility_detector = VolatilityDetector()
        self.regime_detector = RegimeChangeDetector()
        self.liquidity_assessor = LiquidityRiskAssessor()

        # Risk thresholds
        self.volatility_alert_threshold = self.config.get("risk.volatility_alert_threshold", 0.4)
        self.liquidity_alert_threshold = self.config.get("risk.liquidity_alert_threshold", 0.3)
        self.stress_alert_threshold = self.config.get("risk.stress_alert_threshold", 0.7)

        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Initialize market risk monitor."""
        try:
            logger.info("Market risk monitor initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize market risk monitor: {e}")
            raise RiskError(f"Market risk monitor initialization failed: {e}") from e

    @timeout_async(0.1)  # 100ms timeout for market risk analysis
    @measure_latency("market_risk_analysis")
    async def analyze_market_risk(self, market_data: dict[str, Any]) -> dict[str, Any]:
        """Analyze comprehensive market risk.

        Args:
            market_data: Dictionary containing market snapshots by symbol

        Returns:
            Comprehensive market risk analysis
        """
        async with self._lock:
            try:
                risk_analysis = {
                    'timestamp': datetime.now().isoformat(),
                    'symbols': {},
                    'global_indicators': {},
                    'alerts': [],
                    'recommendations': []
                }

                # Analyze each symbol
                for symbol, data in market_data.items():
                    symbol_analysis = await self._analyze_symbol_risk(symbol, data)
                    risk_analysis['symbols'][symbol] = symbol_analysis

                    # Collect alerts
                    if symbol_analysis.get('alerts'):
                        risk_analysis['alerts'].extend(symbol_analysis['alerts'])

                # Calculate global indicators
                risk_analysis['global_indicators'] = await self._calculate_global_indicators(
                    risk_analysis['symbols']
                )

                # Generate recommendations
                risk_analysis['recommendations'] = self._generate_risk_recommendations(
                    risk_analysis['symbols'], risk_analysis['global_indicators']
                )

                return risk_analysis

            except Exception as e:
                logger.error(f"Error analyzing market risk: {e}")
                raise RiskError(f"Market risk analysis failed: {e}") from e

    async def _analyze_symbol_risk(self, symbol: str, data: dict[str, Any]) -> dict[str, Any]:
        """Analyze risk for a specific symbol."""
        try:
            # Create market snapshot
            snapshot = MarketSnapshot(
                timestamp=datetime.now(),
                symbol=symbol,
                price=data.get('price', 0.0),
                volume=data.get('volume', 0.0),
                bid_price=data.get('bid_price', 0.0),
                ask_price=data.get('ask_price', 0.0),
                bid_size=data.get('bid_size', 0.0),
                ask_size=data.get('ask_size', 0.0)
            )

            # Update price data
            self.volatility_detector.update_price_data(symbol, snapshot.price, snapshot.timestamp)

            # Calculate volatility metrics
            volatility_metrics = await self.volatility_detector.calculate_volatility_metrics(symbol)

            # Detect regime change
            price_history = data.get('price_history', [snapshot.price])
            volume_history = data.get('volume_history', [snapshot.volume])

            current_regime = MarketRegime.UNKNOWN
            if volatility_metrics:
                current_regime = await self.regime_detector.detect_regime_change(
                    symbol, volatility_metrics, price_history, volume_history
                )

            # Assess liquidity risk
            liquidity_metrics = await self.liquidity_assessor.assess_liquidity_risk(snapshot)

            # Calculate stress indicators
            stress_indicators = self._calculate_stress_indicators(
                volatility_metrics, liquidity_metrics, current_regime
            )

            # Generate alerts
            alerts = self._generate_symbol_alerts(
                symbol, volatility_metrics, liquidity_metrics, stress_indicators
            )

            return {
                'symbol': symbol,
                'market_snapshot': {
                    'price': snapshot.price,
                    'volume': snapshot.volume,
                    'spread_bps': snapshot.spread_bps,
                    'mid_price': snapshot.mid_price
                },
                'volatility_metrics': volatility_metrics.__dict__ if volatility_metrics else {},
                'liquidity_metrics': liquidity_metrics.__dict__,
                'market_regime': current_regime.value,
                'stress_indicators': stress_indicators.__dict__,
                'alerts': alerts
            }

        except Exception as e:
            logger.error(f"Error analyzing symbol risk for {symbol}: {e}")
            return {'symbol': symbol, 'error': str(e)}

    def _calculate_stress_indicators(self,
                                   volatility_metrics: VolatilityMetrics | None,
                                   liquidity_metrics: LiquidityMetrics,
                                   regime: MarketRegime) -> MarketStressIndicators:
        """Calculate market stress indicators."""
        try:
            # Volatility spike
            vol_spike = False
            if volatility_metrics:
                vol_spike = (volatility_metrics.volatility_level in [VolatilityLevel.HIGH, VolatilityLevel.EXTREME] or
                           volatility_metrics.volatility_percentile > 90)

            # Volume spike (would need historical volume data)
            volume_spike = False  # Placeholder

            # Spread widening
            spread_widening = liquidity_metrics.spread_percentile > 80

            # Correlation breakdown (would need multi-asset data)
            correlation_breakdown = False  # Placeholder

            # Liquidity evaporation
            liquidity_evaporation = liquidity_metrics.liquidity_condition in [
                LiquidityCondition.POOR, LiquidityCondition.CRITICAL
            ]

            # Price gaps (would need tick data)
            price_gaps = False  # Placeholder

            # Calculate composite stress score
            stress_components = [
                0.3 if vol_spike else 0.0,
                0.2 if volume_spike else 0.0,
                0.2 if spread_widening else 0.0,
                0.1 if correlation_breakdown else 0.0,
                0.15 if liquidity_evaporation else 0.0,
                0.05 if price_gaps else 0.0
            ]

            stress_score = sum(stress_components)

            # Regime adjustment
            if regime in [MarketRegime.CRISIS, MarketRegime.VOLATILE]:
                stress_score = min(1.0, stress_score + 0.2)

            # Classify stress level
            if stress_score >= 0.8:
                stress_level = 'extreme'
            elif stress_score >= 0.6:
                stress_level = 'high'
            elif stress_score >= 0.4:
                stress_level = 'medium'
            else:
                stress_level = 'low'

            return MarketStressIndicators(
                volatility_spike=vol_spike,
                volume_spike=volume_spike,
                spread_widening=spread_widening,
                correlation_breakdown=correlation_breakdown,
                liquidity_evaporation=liquidity_evaporation,
                price_gaps=price_gaps,
                stress_score=stress_score,
                stress_level=stress_level
            )

        except Exception as e:
            logger.error(f"Error calculating stress indicators: {e}")
            return MarketStressIndicators(
                volatility_spike=False,
                volume_spike=False,
                spread_widening=False,
                correlation_breakdown=False,
                liquidity_evaporation=False,
                price_gaps=False,
                stress_score=0.0,
                stress_level='low'
            )

    def _generate_symbol_alerts(self,
                               symbol: str,
                               volatility_metrics: VolatilityMetrics | None,
                               liquidity_metrics: LiquidityMetrics,
                               stress_indicators: MarketStressIndicators) -> list[dict[str, Any]]:
        """Generate alerts for symbol-specific risks."""
        alerts = []

        # Volatility alerts
        if volatility_metrics and volatility_metrics.current_volatility > self.volatility_alert_threshold:
            alerts.append({
                'type': 'volatility_alert',
                'symbol': symbol,
                'severity': 'high' if volatility_metrics.volatility_level == VolatilityLevel.EXTREME else 'medium',
                'message': f"High volatility detected: {volatility_metrics.current_volatility:.1%} (threshold: {self.volatility_alert_threshold:.1%})",
                'timestamp': datetime.now().isoformat()
            })

        # Liquidity alerts
        if liquidity_metrics.liquidity_score < self.liquidity_alert_threshold:
            alerts.append({
                'type': 'liquidity_alert',
                'symbol': symbol,
                'severity': 'high' if liquidity_metrics.liquidity_condition == LiquidityCondition.CRITICAL else 'medium',
                'message': f"Poor liquidity detected: score {liquidity_metrics.liquidity_score:.2f} (threshold: {self.liquidity_alert_threshold:.2f})",
                'timestamp': datetime.now().isoformat()
            })

        # Stress alerts
        if stress_indicators.stress_score > self.stress_alert_threshold:
            alerts.append({
                'type': 'market_stress_alert',
                'symbol': symbol,
                'severity': 'critical' if stress_indicators.stress_level == 'extreme' else 'high',
                'message': f"Market stress detected: {stress_indicators.stress_level} level (score: {stress_indicators.stress_score:.2f})",
                'timestamp': datetime.now().isoformat()
            })

        return alerts

    async def _calculate_global_indicators(self, symbols_analysis: dict[str, dict[str, Any]]) -> dict[str, Any]:
        """Calculate global market risk indicators."""
        try:
            if not symbols_analysis:
                return {}

            # Aggregate volatility metrics
            volatilities = []
            stress_scores = []
            liquidity_scores = []
            regimes = []

            for symbol_data in symbols_analysis.values():
                if 'error' in symbol_data:
                    continue

                vol_metrics = symbol_data.get('volatility_metrics', {})
                if vol_metrics and 'current_volatility' in vol_metrics:
                    volatilities.append(vol_metrics['current_volatility'])

                stress_indicators = symbol_data.get('stress_indicators', {})
                if stress_indicators and 'stress_score' in stress_indicators:
                    stress_scores.append(stress_indicators['stress_score'])

                liquidity_metrics = symbol_data.get('liquidity_metrics', {})
                if liquidity_metrics and 'liquidity_score' in liquidity_metrics:
                    liquidity_scores.append(liquidity_metrics['liquidity_score'])

                regime = symbol_data.get('market_regime')
                if regime:
                    regimes.append(regime)

            # Calculate global metrics
            global_indicators = {
                'average_volatility': np.mean(volatilities) if volatilities else 0.0,
                'max_volatility': max(volatilities) if volatilities else 0.0,
                'average_stress_score': np.mean(stress_scores) if stress_scores else 0.0,
                'max_stress_score': max(stress_scores) if stress_scores else 0.0,
                'average_liquidity_score': np.mean(liquidity_scores) if liquidity_scores else 1.0,
                'min_liquidity_score': min(liquidity_scores) if liquidity_scores else 1.0,
                'regime_distribution': {regime: regimes.count(regime) for regime in set(regimes)} if regimes else {}
            }

            # Overall market health score (0-1, higher is better)
            health_components = []
            if volatilities:
                health_components.append(1.0 - min(1.0, np.mean(volatilities) / 0.5))  # Normalize to 50% vol
            if stress_scores:
                health_components.append(1.0 - np.mean(stress_scores))
            if liquidity_scores:
                health_components.append(np.mean(liquidity_scores))

            global_indicators['market_health_score'] = np.mean(health_components) if health_components else 0.5

            return global_indicators

        except Exception as e:
            logger.error(f"Error calculating global indicators: {e}")
            return {}

    def _generate_risk_recommendations(self,
                                     symbols_analysis: dict[str, dict[str, Any]],
                                     global_indicators: dict[str, Any]) -> list[str]:
        """Generate risk management recommendations."""
        recommendations = []

        # Global recommendations
        market_health = global_indicators.get('market_health_score', 0.5)
        if market_health < 0.3:
            recommendations.append("CRITICAL: Consider reducing overall position sizes due to poor market conditions")
        elif market_health < 0.5:
            recommendations.append("WARNING: Monitor positions closely - market conditions are deteriorating")

        avg_stress = global_indicators.get('average_stress_score', 0.0)
        if avg_stress > 0.7:
            recommendations.append("High market stress detected - consider widening spreads and reducing quote frequency")

        # Symbol-specific recommendations
        for symbol, data in symbols_analysis.items():
            if 'error' in data:
                continue

            alerts = data.get('alerts', [])
            if any(alert['severity'] in ['critical', 'high'] for alert in alerts):
                recommendations.append(f"Review risk controls for {symbol} due to elevated risk levels")

            regime = data.get('market_regime')
            if regime == 'crisis':
                recommendations.append(f"Consider halting trading in {symbol} during crisis regime")
            elif regime == 'volatile':
                recommendations.append(f"Reduce position sizes and widen spreads for {symbol} during volatile regime")

        return recommendations

    async def add_market_data(self, symbol: str, market_data: dict[str, Any]) -> None:
        """Add market data for analysis (compatibility method)."""
        try:
            # Update price data for volatility calculation
            price = market_data.get('price', 0.0)
            timestamp = datetime.now()
            self.volatility_detector.update_price_data(symbol, price, timestamp)
            logger.debug(f"Added market data for {symbol}")
        except Exception as e:
            logger.error(f"Error adding market data for {symbol}: {e}")

    async def analyze_market_conditions(self, market_data: dict[str, Any]) -> dict[str, Any]:
        """Analyze market conditions (compatibility method)."""
        try:
            return await self.analyze_market_risk(market_data)
        except Exception as e:
            logger.error(f"Error analyzing market conditions: {e}")
            return {"error": str(e)}

    def set_alert_callback(self, callback: Callable) -> None:
        """Set alert callback (compatibility method)."""
        # Store callback for future use
        self._alert_callback = callback
        logger.info("Alert callback set for market risk monitor")

    async def cleanup(self) -> None:
        """Cleanup market risk monitor."""
        logger.info("Market risk monitor cleanup completed")

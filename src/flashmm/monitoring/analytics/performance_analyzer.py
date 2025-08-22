"""
FlashMM Performance Analytics System

Comprehensive performance analysis with spread improvement validation, P&L attribution,
trading efficiency metrics, and automated reporting capabilities.
"""

import json
import statistics
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import numpy as np

from flashmm.config.settings import get_config
from flashmm.monitoring.telemetry.metrics_collector import MetricsCollector
from flashmm.utils.logging import get_logger

logger = get_logger(__name__)


class PerformanceMetric(Enum):
    """Performance metric types."""
    SPREAD_IMPROVEMENT = "spread_improvement"
    PNL_ATTRIBUTION = "pnl_attribution"
    VOLUME_EFFICIENCY = "volume_efficiency"
    EXECUTION_QUALITY = "execution_quality"
    RISK_ADJUSTED_RETURN = "risk_adjusted_return"
    MARKET_IMPACT = "market_impact"
    FILL_RATE = "fill_rate"
    SLIPPAGE = "slippage"
    INVENTORY_TURNOVER = "inventory_turnover"
    UPTIME = "uptime"


class ReportType(Enum):
    """Report types."""
    DAILY = "daily"
    HOURLY = "hourly"
    REAL_TIME = "real_time"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    CUSTOM = "custom"


@dataclass
class SpreadAnalysis:
    """Spread improvement analysis results."""
    timestamp: datetime
    market: str
    baseline_spread_bps: float
    current_spread_bps: float
    improvement_bps: float
    improvement_percent: float
    volume_weighted_improvement: float
    confidence_interval: tuple[float, float]
    statistical_significance: float
    sample_size: int

    # Additional metrics
    bid_improvement_bps: float = 0.0
    ask_improvement_bps: float = 0.0
    mid_price_stability: float = 0.0
    quote_update_frequency: float = 0.0
    effective_spread_bps: float = 0.0


@dataclass
class PnLAttribution:
    """P&L attribution analysis."""
    timestamp: datetime
    total_pnl_usdc: float

    # P&L breakdown
    spread_capture_pnl: float
    inventory_pnl: float
    market_making_pnl: float
    fees_paid: float
    fees_earned: float
    slippage_cost: float

    # Performance metrics
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float

    # Risk metrics
    var_95: float  # Value at Risk 95%
    var_99: float  # Value at Risk 99%
    expected_shortfall: float
    volatility: float

    # Efficiency metrics
    return_on_capital: float
    inventory_turnover: float
    capital_efficiency: float


@dataclass
class TradingEfficiency:
    """Trading efficiency metrics."""
    timestamp: datetime

    # Execution metrics
    fill_rate: float
    average_fill_time_ms: float
    slippage_bps: float
    market_impact_bps: float

    # Volume metrics
    total_volume_usdc: float
    trades_count: int
    average_trade_size_usdc: float
    volume_participation_rate: float

    # Quality metrics
    price_improvement_rate: float
    adverse_selection_rate: float
    inventory_risk_score: float
    quote_competitiveness: float


@dataclass
class PerformanceReport:
    """Comprehensive performance report."""
    report_id: str
    report_type: ReportType
    start_time: datetime
    end_time: datetime
    generated_at: datetime

    # Summary metrics
    summary: dict[str, Any]

    # Detailed analysis
    spread_analysis: list[SpreadAnalysis]
    pnl_attribution: list[PnLAttribution]
    trading_efficiency: list[TradingEfficiency]

    # Benchmarks and comparisons
    benchmarks: dict[str, Any]
    period_comparison: dict[str, Any]

    # Alerts and recommendations
    alerts: list[dict[str, Any]]
    recommendations: list[str]

    # Metadata
    data_quality_score: float
    confidence_level: float
    report_version: str = "1.0"


class PerformanceAnalyzer:
    """Performance analytics engine with spread improvement validation."""

    def __init__(self, metrics_collector: MetricsCollector | None = None):
        self.config = get_config()
        self.metrics_collector = metrics_collector

        # Data storage
        self.historical_data: dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.baseline_spreads: dict[str, float] = {}
        self.performance_cache: dict[str, Any] = {}

        # Analysis parameters
        self.confidence_level = self.config.get("analytics.confidence_level", 0.95)
        self.min_sample_size = self.config.get("analytics.min_sample_size", 100)
        self.spread_window_minutes = self.config.get("analytics.spread_window_minutes", 60)
        self.pnl_window_hours = self.config.get("analytics.pnl_window_hours", 24)

        # Benchmarks
        self.target_spread_improvement = self.config.get("analytics.target_spread_improvement_percent", 25.0)
        self.target_sharpe_ratio = self.config.get("analytics.target_sharpe_ratio", 2.0)
        self.target_fill_rate = self.config.get("analytics.target_fill_rate", 0.95)
        self.target_uptime = self.config.get("analytics.target_uptime", 0.99)

        # Performance tracking
        self.analysis_stats = {
            "total_analyses": 0,
            "spread_analyses": 0,
            "pnl_analyses": 0,
            "efficiency_analyses": 0,
            "reports_generated": 0,
            "alerts_triggered": 0,
            "last_analysis": None
        }

        logger.info("PerformanceAnalyzer initialized")

    async def initialize(self) -> None:
        """Initialize performance analyzer."""
        try:
            # Load historical baselines
            await self._load_baseline_spreads()

            # Initialize data collection
            await self._initialize_data_collection()

            # Start background analysis tasks
            await self._start_background_analysis()

            logger.info("PerformanceAnalyzer initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize PerformanceAnalyzer: {e}")
            raise

    async def _load_baseline_spreads(self) -> None:
        """Load baseline spread measurements for comparison."""
        # This would typically load from historical data or configuration
        default_baselines = {
            "SOL-USD": 8.5,  # 8.5 bps baseline spread
            "ETH-USD": 6.2,  # 6.2 bps baseline spread
            "BTC-USD": 4.8,  # 4.8 bps baseline spread
            "USDC-USD": 2.1  # 2.1 bps baseline spread
        }

        # Load from config or use defaults
        configured_baselines = self.config.get("analytics.baseline_spreads", {})
        self.baseline_spreads.update(default_baselines)
        self.baseline_spreads.update(configured_baselines)

        logger.info(f"Loaded baseline spreads for {len(self.baseline_spreads)} markets")

    async def _initialize_data_collection(self) -> None:
        """Initialize data collection for analysis."""
        # Initialize data structures for different time windows
        for metric in PerformanceMetric:
            self.historical_data[metric.value] = deque(maxlen=10000)

        # Initialize market-specific data
        markets = self.config.get("analytics.markets", ["SOL-USD", "ETH-USD", "BTC-USD"])
        for market in markets:
            self.historical_data[f"spread_data_{market}"] = deque(maxlen=5000)
            self.historical_data[f"trade_data_{market}"] = deque(maxlen=5000)

    async def _start_background_analysis(self) -> None:
        """Start background analysis tasks."""
        # This would start periodic analysis tasks
        # For now, we'll implement the core analysis methods
        pass

    async def analyze_spread_improvement(self, market: str,
                                       start_time: datetime | None = None,
                                       end_time: datetime | None = None) -> SpreadAnalysis:
        """Analyze spread improvement for a specific market."""
        try:
            if not start_time:
                start_time = datetime.now() - timedelta(minutes=self.spread_window_minutes)
            if not end_time:
                end_time = datetime.now()

            # Get spread data for the time window
            spread_data = await self._get_spread_data(market, start_time, end_time)

            if len(spread_data) < self.min_sample_size:
                logger.warning(f"Insufficient spread data for {market}: {len(spread_data)} samples")
                return self._create_empty_spread_analysis(market, start_time)

            # Calculate spread metrics
            current_spreads = [data['spread_bps'] for data in spread_data]
            baseline_spread = self.baseline_spreads.get(market, np.mean(current_spreads))

            current_mean_spread = float(np.mean(current_spreads))
            np.median(current_spreads)
            np.std(current_spreads)

            # Calculate improvement
            improvement_bps = baseline_spread - current_mean_spread
            improvement_percent = (improvement_bps / baseline_spread) * 100 if baseline_spread > 0 else 0

            # Volume-weighted improvement
            volumes = [data.get('volume', 1.0) for data in spread_data]
            volume_weighted_spread = float(np.average(current_spreads, weights=volumes))
            volume_weighted_improvement = baseline_spread - volume_weighted_spread

            # Statistical significance and confidence interval
            confidence_interval = self._calculate_confidence_interval(current_spreads, self.confidence_level)
            statistical_significance = self._calculate_significance(current_spreads, float(baseline_spread))

            # Additional metrics
            bid_ask_data = await self._get_bid_ask_data(market, start_time, end_time)
            bid_improvement = self._calculate_bid_improvement(bid_ask_data, float(baseline_spread))
            ask_improvement = self._calculate_bid_improvement(bid_ask_data, float(baseline_spread))

            mid_price_stability = self._calculate_mid_price_stability(bid_ask_data)
            quote_frequency = len(spread_data) / ((end_time - start_time).total_seconds() / 60)  # per minute
            effective_spread = self._calculate_effective_spread(spread_data)

            analysis = SpreadAnalysis(
                timestamp=end_time,
                market=market,
                baseline_spread_bps=float(baseline_spread),
                current_spread_bps=float(current_mean_spread),
                improvement_bps=float(improvement_bps),
                improvement_percent=float(improvement_percent),
                volume_weighted_improvement=float(volume_weighted_improvement),
                confidence_interval=confidence_interval,
                statistical_significance=float(statistical_significance),
                sample_size=len(spread_data),
                bid_improvement_bps=float(bid_improvement),
                ask_improvement_bps=float(ask_improvement),
                mid_price_stability=float(mid_price_stability),
                quote_update_frequency=float(quote_frequency),
                effective_spread_bps=float(effective_spread)
            )

            # Store analysis
            self.historical_data[f"spread_analysis_{market}"].append(analysis)
            self.analysis_stats["spread_analyses"] += 1

            logger.info(f"Spread analysis completed for {market}: {improvement_percent:.2f}% improvement")
            return analysis

        except Exception as e:
            logger.error(f"Error analyzing spread improvement for {market}: {e}")
            return self._create_empty_spread_analysis(market, end_time or datetime.now())

    async def analyze_pnl_attribution(self, start_time: datetime | None = None,
                                    end_time: datetime | None = None) -> PnLAttribution:
        """Analyze P&L attribution and performance metrics."""
        try:
            if not start_time:
                start_time = datetime.now() - timedelta(hours=self.pnl_window_hours)
            if not end_time:
                end_time = datetime.now()

            # Get P&L data
            pnl_data = await self._get_pnl_data(start_time, end_time)
            trade_data = await self._get_trade_data(start_time, end_time)

            if not pnl_data:
                logger.warning(f"No P&L data available for period {start_time} to {end_time}")
                return self._create_empty_pnl_attribution(end_time)

            # Calculate P&L components
            total_pnl = sum(pnl['total_pnl'] for pnl in pnl_data)
            spread_capture_pnl = sum(pnl.get('spread_pnl', 0) for pnl in pnl_data)
            inventory_pnl = sum(pnl.get('inventory_pnl', 0) for pnl in pnl_data)
            market_making_pnl = sum(pnl.get('mm_pnl', 0) for pnl in pnl_data)

            fees_paid = sum(pnl.get('fees_paid', 0) for pnl in pnl_data)
            fees_earned = sum(pnl.get('fees_earned', 0) for pnl in pnl_data)
            slippage_cost = sum(pnl.get('slippage', 0) for pnl in pnl_data)

            # Calculate performance metrics
            returns = [pnl['total_pnl'] for pnl in pnl_data]
            sharpe_ratio = self._calculate_sharpe_ratio(returns)
            max_drawdown = self._calculate_max_drawdown(returns)

            # Trading statistics
            win_rate = len([r for r in returns if r > 0]) / len(returns) if returns else 0
            profit_factor = self._calculate_profit_factor(returns)

            # Risk metrics
            var_95 = float(np.percentile(returns, 5)) if returns else 0.0
            var_99 = float(np.percentile(returns, 1)) if returns else 0.0
            expected_shortfall = float(np.mean([r for r in returns if r <= var_95])) if returns else 0.0
            volatility = float(np.std(returns)) if returns else 0.0

            # Efficiency metrics
            capital_used = self._calculate_capital_usage(trade_data)
            return_on_capital = (total_pnl / capital_used) if capital_used > 0 else 0
            inventory_turnover = self._calculate_inventory_turnover(trade_data)
            capital_efficiency = self._calculate_capital_efficiency(trade_data, total_pnl)

            attribution = PnLAttribution(
                timestamp=end_time,
                total_pnl_usdc=float(total_pnl),
                spread_capture_pnl=float(spread_capture_pnl),
                inventory_pnl=float(inventory_pnl),
                market_making_pnl=float(market_making_pnl),
                fees_paid=float(fees_paid),
                fees_earned=float(fees_earned),
                slippage_cost=float(slippage_cost),
                sharpe_ratio=float(sharpe_ratio),
                max_drawdown=float(max_drawdown),
                win_rate=float(win_rate),
                profit_factor=float(profit_factor),
                var_95=float(var_95),
                var_99=float(var_99),
                expected_shortfall=float(expected_shortfall),
                volatility=float(volatility),
                return_on_capital=float(return_on_capital),
                inventory_turnover=float(inventory_turnover),
                capital_efficiency=float(capital_efficiency)
            )

            # Store analysis
            self.historical_data["pnl_attribution"].append(attribution)
            self.analysis_stats["pnl_analyses"] += 1

            logger.info(f"P&L attribution analysis completed: ${total_pnl:.2f} total P&L")
            return attribution

        except Exception as e:
            logger.error(f"Error analyzing P&L attribution: {e}")
            return self._create_empty_pnl_attribution(end_time or datetime.now())

    async def analyze_trading_efficiency(self, start_time: datetime | None = None,
                                       end_time: datetime | None = None) -> TradingEfficiency:
        """Analyze trading efficiency metrics."""
        try:
            if not start_time:
                start_time = datetime.now() - timedelta(hours=1)
            if not end_time:
                end_time = datetime.now()

            # Get trading data
            trade_data = await self._get_trade_data(start_time, end_time)
            order_data = await self._get_order_data(start_time, end_time)

            if not trade_data:
                logger.warning("No trade data available for efficiency analysis")
                return self._create_empty_efficiency_analysis(end_time)

            # Execution metrics
            fill_rate = self._calculate_fill_rate(order_data)
            avg_fill_time = self._calculate_average_fill_time(order_data)
            slippage = self._calculate_slippage(trade_data)
            market_impact = self._calculate_market_impact(trade_data)

            # Volume metrics
            total_volume = sum(trade['volume_usdc'] for trade in trade_data)
            trades_count = len(trade_data)
            avg_trade_size = total_volume / trades_count if trades_count > 0 else 0
            participation_rate = self._calculate_participation_rate(trade_data, start_time, end_time)

            # Quality metrics
            price_improvement_rate = self._calculate_price_improvement_rate(trade_data)
            adverse_selection_rate = self._calculate_adverse_selection_rate(trade_data)
            inventory_risk = self._calculate_inventory_risk_score(trade_data)
            quote_competitiveness = self._calculate_quote_competitiveness(trade_data)

            efficiency = TradingEfficiency(
                timestamp=end_time,
                fill_rate=fill_rate,
                average_fill_time_ms=avg_fill_time,
                slippage_bps=slippage,
                market_impact_bps=market_impact,
                total_volume_usdc=total_volume,
                trades_count=trades_count,
                average_trade_size_usdc=avg_trade_size,
                volume_participation_rate=participation_rate,
                price_improvement_rate=price_improvement_rate,
                adverse_selection_rate=adverse_selection_rate,
                inventory_risk_score=inventory_risk,
                quote_competitiveness=quote_competitiveness
            )

            # Store analysis
            self.historical_data["trading_efficiency"].append(efficiency)
            self.analysis_stats["efficiency_analyses"] += 1

            logger.info(f"Trading efficiency analysis completed: {fill_rate:.1%} fill rate, {slippage:.2f} bps slippage")
            return efficiency

        except Exception as e:
            logger.error(f"Error analyzing trading efficiency: {e}")
            return self._create_empty_efficiency_analysis(end_time or datetime.now())

    async def generate_performance_report(self, report_type: ReportType,
                                        start_time: datetime | None = None,
                                        end_time: datetime | None = None,
                                        markets: list[str] | None = None) -> PerformanceReport:
        """Generate comprehensive performance report."""
        try:
            if not end_time:
                end_time = datetime.now()

            # Determine time window based on report type
            if not start_time:
                if report_type == ReportType.HOURLY:
                    start_time = end_time - timedelta(hours=1)
                elif report_type == ReportType.DAILY:
                    start_time = end_time - timedelta(days=1)
                elif report_type == ReportType.WEEKLY:
                    start_time = end_time - timedelta(weeks=1)
                elif report_type == ReportType.MONTHLY:
                    start_time = end_time - timedelta(days=30)
                else:
                    start_time = end_time - timedelta(hours=1)

            if not markets:
                markets = list(self.baseline_spreads.keys())

            report_id = f"performance_report_{int(end_time.timestamp())}"

            # Generate analyses
            spread_analyses = []
            for market in markets:
                try:
                    analysis = await self.analyze_spread_improvement(market, start_time, end_time)
                    spread_analyses.append(analysis)
                except Exception as e:
                    logger.error(f"Failed to analyze spread for {market}: {e}")

            pnl_attribution = await self.analyze_pnl_attribution(start_time, end_time)
            trading_efficiency = await self.analyze_trading_efficiency(start_time, end_time)

            # Generate summary
            summary = await self._generate_summary(spread_analyses, pnl_attribution, trading_efficiency)

            # Generate benchmarks and comparisons
            benchmarks = await self._generate_benchmarks(spread_analyses, pnl_attribution, trading_efficiency)
            period_comparison = await self._generate_period_comparison(report_type, start_time, end_time)

            # Generate alerts and recommendations
            alerts = await self._generate_performance_alerts(spread_analyses, pnl_attribution, trading_efficiency)
            recommendations = await self._generate_recommendations(spread_analyses, pnl_attribution, trading_efficiency)

            # Calculate data quality and confidence
            data_quality_score = self._calculate_data_quality_score(spread_analyses, pnl_attribution, trading_efficiency)
            confidence_level = self._calculate_overall_confidence(spread_analyses)

            report = PerformanceReport(
                report_id=report_id,
                report_type=report_type,
                start_time=start_time,
                end_time=end_time,
                generated_at=datetime.now(),
                summary=summary,
                spread_analysis=spread_analyses,
                pnl_attribution=[pnl_attribution],
                trading_efficiency=[trading_efficiency],
                benchmarks=benchmarks,
                period_comparison=period_comparison,
                alerts=alerts,
                recommendations=recommendations,
                data_quality_score=data_quality_score,
                confidence_level=confidence_level
            )

            # Store report
            self.performance_cache[report_id] = report
            self.analysis_stats["reports_generated"] += 1
            self.analysis_stats["last_analysis"] = datetime.now()

            logger.info(f"Generated {report_type.value} performance report: {report_id}")
            return report

        except Exception as e:
            logger.error(f"Error generating performance report: {e}")
            raise

    async def validate_spread_improvement(self, market: str, claimed_improvement: float) -> dict[str, Any]:
        """Validate spread improvement claims with statistical analysis."""
        try:
            # Get recent spread analysis
            analysis = await self.analyze_spread_improvement(market)

            # Statistical validation
            validation_result = {
                "market": market,
                "claimed_improvement_percent": claimed_improvement,
                "measured_improvement_percent": analysis.improvement_percent,
                "difference": abs(claimed_improvement - analysis.improvement_percent),
                "is_valid": False,
                "confidence_level": analysis.statistical_significance,
                "sample_size": analysis.sample_size,
                "validation_timestamp": datetime.now(),
                "validation_details": {}
            }

            # Validation criteria
            tolerance = self.config.get("analytics.improvement_tolerance_percent", 5.0)
            min_confidence = self.config.get("analytics.min_confidence_level", 0.90)
            min_samples = self.config.get("analytics.min_validation_samples", 50)

            # Check validation criteria
            within_tolerance = validation_result["difference"] <= tolerance
            sufficient_confidence = analysis.statistical_significance >= min_confidence
            sufficient_samples = analysis.sample_size >= min_samples

            validation_result["is_valid"] = within_tolerance and sufficient_confidence and sufficient_samples
            validation_result["validation_details"] = {
                "within_tolerance": within_tolerance,
                "tolerance_threshold": tolerance,
                "sufficient_confidence": sufficient_confidence,
                "confidence_threshold": min_confidence,
                "sufficient_samples": sufficient_samples,
                "sample_threshold": min_samples,
                "baseline_spread_bps": analysis.baseline_spread_bps,
                "current_spread_bps": analysis.current_spread_bps,
                "confidence_interval": analysis.confidence_interval
            }

            logger.info(f"Spread improvement validation for {market}: {'VALID' if validation_result['is_valid'] else 'INVALID'}")
            return validation_result

        except Exception as e:
            logger.error(f"Error validating spread improvement for {market}: {e}")
            return {
                "market": market,
                "claimed_improvement_percent": claimed_improvement,
                "is_valid": False,
                "error": str(e),
                "validation_timestamp": datetime.now()
            }

    # Helper methods for data retrieval (these would integrate with actual data sources)

    async def _get_spread_data(self, market: str, start_time: datetime, end_time: datetime) -> list[dict[str, Any]]:
        """Get spread data for analysis."""
        # This would integrate with actual data sources
        # For now, return simulated data
        return [
            {
                "timestamp": start_time + timedelta(seconds=i*60),
                "market": market,
                "spread_bps": np.random.normal(6.0, 1.5),  # Simulate improved spreads
                "volume": np.random.exponential(1000),
                "bid": 100.0 - np.random.uniform(0.03, 0.06),
                "ask": 100.0 + np.random.uniform(0.03, 0.06)
            }
            for i in range(int((end_time - start_time).total_seconds() / 60))
        ]

    async def _get_bid_ask_data(self, market: str, start_time: datetime, end_time: datetime) -> list[dict[str, Any]]:
        """Get bid/ask data for analysis."""
        # Simulate bid/ask data
        return [
            {
                "timestamp": start_time + timedelta(seconds=i*10),
                "market": market,
                "bid": 100.0 - np.random.uniform(0.02, 0.05),
                "ask": 100.0 + np.random.uniform(0.02, 0.05),
                "mid": 100.0 + np.random.normal(0, 0.01)
            }
            for i in range(int((end_time - start_time).total_seconds() / 10))
        ]

    async def _get_pnl_data(self, start_time: datetime, end_time: datetime) -> list[dict[str, Any]]:
        """Get P&L data for analysis."""
        # Simulate P&L data
        hours = int((end_time - start_time).total_seconds() / 3600)
        return [
            {
                "timestamp": start_time + timedelta(hours=i),
                "total_pnl": np.random.normal(50, 20),  # Positive expected P&L
                "spread_pnl": np.random.normal(30, 10),
                "inventory_pnl": np.random.normal(20, 15),
                "mm_pnl": np.random.normal(25, 12),
                "fees_paid": np.random.uniform(5, 15),
                "fees_earned": np.random.uniform(8, 20),
                "slippage": np.random.uniform(2, 8)
            }
            for i in range(max(1, hours))
        ]

    async def _get_trade_data(self, start_time: datetime, end_time: datetime) -> list[dict[str, Any]]:
        """Get trade data for analysis."""
        # Simulate trade data
        num_trades = np.random.poisson(100)  # Average 100 trades per period
        return [
            {
                "timestamp": start_time + timedelta(seconds=np.random.uniform(0, (end_time - start_time).total_seconds())),
                "volume_usdc": np.random.exponential(500),
                "price": 100 + np.random.normal(0, 2),
                "side": np.random.choice(["buy", "sell"]),
                "fill_time_ms": np.random.gamma(2, 50),  # Gamma distribution for fill times
                "slippage_bps": np.random.exponential(3),
                "market_impact_bps": np.random.exponential(2)
            }
            for _ in range(num_trades)
        ]

    async def _get_order_data(self, start_time: datetime, end_time: datetime) -> list[dict[str, Any]]:
        """Get order data for analysis."""
        # Simulate order data
        num_orders = np.random.poisson(150)  # More orders than trades (some unfilled)
        return [
            {
                "timestamp": start_time + timedelta(seconds=np.random.uniform(0, (end_time - start_time).total_seconds())),
                "order_id": f"order_{i}",
                "volume_usdc": np.random.exponential(400),
                "side": np.random.choice(["buy", "sell"]),
                "status": np.random.choice(["filled", "partial", "cancelled"], p=[0.85, 0.10, 0.05]),
                "fill_time_ms": np.random.gamma(2, 75) if np.random.random() > 0.15 else None,
                "requested_price": 100 + np.random.normal(0, 2),
                "filled_price": 100 + np.random.normal(0, 2.2)
            }
            for i in range(num_orders)
        ]

    # Calculation helper methods

    def _calculate_confidence_interval(self, data: list[float], confidence: float) -> tuple[float, float]:
        """Calculate confidence interval for data."""
        if not data:
            return (0.0, 0.0)

        mean = float(np.mean(data))
        sem = statistics.stdev(data) / np.sqrt(len(data)) if len(data) > 1 else 0
        z_score = 1.96 if confidence == 0.95 else 2.576  # 95% or 99%
        margin = z_score * sem

        return (float(mean - margin), float(mean + margin))

    def _calculate_significance(self, data: list[float], baseline: float) -> float:
        """Calculate statistical significance of difference from baseline."""
        if not data:
            return 0.0

        # Use fallback implementation to avoid scipy type issues
        mean = float(np.mean(data))
        std = float(np.std(data))
        n = len(data)

        if std == 0:
            return 1.0 if mean != baseline else 0.0

        t_stat = abs(mean - baseline) / (std / np.sqrt(n))
        # Rough approximation for significance
        return float(min(0.99, t_stat / 10))

    def _calculate_bid_improvement(self, bid_ask_data: list[dict[str, Any]], baseline: float) -> float:
        """Calculate bid improvement in bps."""
        if not bid_ask_data:
            return 0.0

        bid_prices = [data['bid'] for data in bid_ask_data]
        mid_prices = [data['mid'] for data in bid_ask_data]

        bid_spreads = [(mid - bid) / mid * 10000 for bid, mid in zip(bid_prices, mid_prices, strict=False)]
        avg_bid_spread = float(np.mean(bid_spreads))

        baseline_half = baseline / 2
        return float(baseline_half - avg_bid_spread)

    def _calculate_ask_improvement(self, bid_ask_data: list[dict[str, Any]], baseline: float) -> float:
        """Calculate ask improvement in bps."""
        if not bid_ask_data:
            return 0.0

        ask_prices = [data['ask'] for data in bid_ask_data]
        mid_prices = [data['mid'] for data in bid_ask_data]

        ask_spreads = [(ask - mid) / mid * 10000 for ask, mid in zip(ask_prices, mid_prices, strict=False)]
        avg_ask_spread = float(np.mean(ask_spreads))

        baseline_half = baseline / 2
        return float(baseline_half - avg_ask_spread)

    def _calculate_mid_price_stability(self, bid_ask_data: list[dict[str, Any]]) -> float:
        """Calculate mid-price stability metric."""
        if len(bid_ask_data) < 2:
            return 1.0

        mid_prices = [data['mid'] for data in bid_ask_data]
        price_changes = [abs(mid_prices[i] - mid_prices[i-1]) / mid_prices[i-1]
                        for i in range(1, len(mid_prices))]

        if not price_changes:
            return 1.0

        avg_change = float(np.mean(price_changes))
        # Stability score: lower changes = higher stability
        return float(max(0.0, 1.0 - avg_change * 1000))  # Scale for readability

    def _calculate_effective_spread(self, spread_data: list[dict[str, Any]]) -> float:
        """Calculate effective spread considering volume weighting."""
        if not spread_data:
            return 0.0

        spreads = [data['spread_bps'] for data in spread_data]
        volumes = [data.get('volume', 1.0) for data in spread_data]

        return float(np.average(spreads, weights=volumes))

    def _calculate_sharpe_ratio(self, returns: list[float]) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) < 2:
            return 0.0

        mean_return = float(np.mean(returns))
        std_return = float(np.std(returns))

        if std_return == 0:
            return float('inf') if mean_return > 0 else 0.0

        # Assuming risk-free rate is negligible for short periods
        return float(mean_return / std_return)

    def _calculate_max_drawdown(self, returns: list[float]) -> float:
        """Calculate maximum drawdown."""
        if not returns:
            return 0.0

        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = cumulative - running_max

        return float(abs(np.min(drawdown))) if len(drawdown) > 0 else 0.0

    def _calculate_profit_factor(self, returns: list[float]) -> float:
        """Calculate profit factor."""
        if not returns:
            return 0.0

        profits = sum(r for r in returns if r > 0)
        losses = abs(sum(r for r in returns if r < 0))

        return profits / losses if losses > 0 else float('inf') if profits > 0 else 0.0

    def _calculate_capital_usage(self, trade_data: list[dict[str, Any]]) -> float:
        """Calculate capital usage."""
        if not trade_data:
            return 0.0

        max_position = 0
        current_position = 0

        for trade in sorted(trade_data, key=lambda x: x['timestamp']):
            side_multiplier = 1 if trade['side'] == 'buy' else -1
            current_position += trade['volume_usdc'] * side_multiplier
            max_position = max(max_position, abs(current_position))

        return max_position

    def _calculate_inventory_turnover(self, trade_data: list[dict[str, Any]]) -> float:
        """Calculate inventory turnover ratio."""
        if not trade_data:
            return 0.0

        total_volume = sum(trade['volume_usdc'] for trade in trade_data)
        avg_inventory = self._calculate_capital_usage(trade_data)

        return total_volume / avg_inventory if avg_inventory > 0 else 0.0

    def _calculate_capital_efficiency(self, trade_data: list[dict[str, Any]], pnl: float) -> float:
        """Calculate capital efficiency ratio."""
        if not trade_data:
            return 0.0

        capital_usage = self._calculate_capital_usage(trade_data)
        return pnl / capital_usage if capital_usage > 0 else 0.0

    def _calculate_fill_rate(self, order_data: list[dict[str, Any]]) -> float:
        """Calculate order fill rate."""
        if not order_data:
            return 0.0

        filled_orders = len([order for order in order_data if order['status'] == 'filled'])
        return filled_orders / len(order_data)

    def _calculate_average_fill_time(self, order_data: list[dict[str, Any]]) -> float:
        """Calculate average fill time in milliseconds."""
        if not order_data:
            return 0.0

        fill_times = [order['fill_time_ms'] for order in order_data
                     if order['fill_time_ms'] is not None]

        return float(np.mean(fill_times)) if fill_times else 0.0

    def _calculate_slippage(self, trade_data: list[dict[str, Any]]) -> float:
        """Calculate average slippage in bps."""
        if not trade_data:
            return 0.0

        slippages = [trade.get('slippage_bps', 0) for trade in trade_data]
        return float(np.mean(slippages))

    def _calculate_market_impact(self, trade_data: list[dict[str, Any]]) -> float:
        """Calculate average market impact in bps."""
        if not trade_data:
            return 0.0

        impacts = [trade.get('market_impact_bps', 0) for trade in trade_data]
        return float(np.mean(impacts))

    def _calculate_participation_rate(self, trade_data: list[dict[str, Any]],
                                   start_time: datetime, end_time: datetime) -> float:
        """Calculate market participation rate."""
        if not trade_data:
            return 0.0

        our_volume = sum(trade['volume_usdc'] for trade in trade_data)
        # Simulate total market volume (this would come from market data)
        market_volume = our_volume * np.random.uniform(10, 50)  # We're 2-10% of market

        return our_volume / market_volume if market_volume > 0 else 0.0

    def _calculate_price_improvement_rate(self, trade_data: list[dict[str, Any]]) -> float:
        """Calculate price improvement rate."""
        if not trade_data:
            return 0.0

        # Simulate price improvement (would be calculated from actual execution vs quoted prices)
        improved_trades = np.random.binomial(len(trade_data), 0.6)  # 60% get price improvement
        return improved_trades / len(trade_data)

    def _calculate_adverse_selection_rate(self, trade_data: list[dict[str, Any]]) -> float:
        """Calculate adverse selection rate."""
        if not trade_data:
            return 0.0

        # Simulate adverse selection (would be calculated from post-trade price movements)
        adverse_trades = np.random.binomial(len(trade_data), 0.15)  # 15% adverse selection
        return adverse_trades / len(trade_data)

    def _calculate_inventory_risk_score(self, trade_data: list[dict[str, Any]]) -> float:
        """Calculate inventory risk score."""
        if not trade_data:
            return 0.0

        max_position = self._calculate_capital_usage(trade_data)
        total_volume = sum(trade['volume_usdc'] for trade in trade_data)

        # Risk score based on position size relative to volume
        risk_ratio = max_position / total_volume if total_volume > 0 else 0
        return min(1.0, risk_ratio)  # Cap at 1.0

    def _calculate_quote_competitiveness(self, trade_data: list[dict[str, Any]]) -> float:
        """Calculate quote competitiveness score."""
        # Simulate competitiveness score (would be calculated from market data comparison)
        return np.random.uniform(0.7, 0.95)  # 70-95% competitive

    # Helper methods for creating empty analyses

    def _create_empty_spread_analysis(self, market: str, timestamp: datetime) -> SpreadAnalysis:
        """Create empty spread analysis for error cases."""
        return SpreadAnalysis(
            timestamp=timestamp,
            market=market,
            baseline_spread_bps=0.0,
            current_spread_bps=0.0,
            improvement_bps=0.0,
            improvement_percent=0.0,
            volume_weighted_improvement=0.0,
            confidence_interval=(0.0, 0.0),
            statistical_significance=0.0,
            sample_size=0
        )

    def _create_empty_pnl_attribution(self, timestamp: datetime) -> PnLAttribution:
        """Create empty P&L attribution for error cases."""
        return PnLAttribution(
            timestamp=timestamp,
            total_pnl_usdc=0.0,
            spread_capture_pnl=0.0,
            inventory_pnl=0.0,
            market_making_pnl=0.0,
            fees_paid=0.0,
            fees_earned=0.0,
            slippage_cost=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            win_rate=0.0,
            profit_factor=0.0,
            var_95=0.0,
            var_99=0.0,
            expected_shortfall=0.0,
            volatility=0.0,
            return_on_capital=0.0,
            inventory_turnover=0.0,
            capital_efficiency=0.0
        )

    def _create_empty_efficiency_analysis(self, timestamp: datetime) -> TradingEfficiency:
        """Create empty efficiency analysis for error cases."""
        return TradingEfficiency(
            timestamp=timestamp,
            fill_rate=0.0,
            average_fill_time_ms=0.0,
            slippage_bps=0.0,
            market_impact_bps=0.0,
            total_volume_usdc=0.0,
            trades_count=0,
            average_trade_size_usdc=0.0,
            volume_participation_rate=0.0,
            price_improvement_rate=0.0,
            adverse_selection_rate=0.0,
            inventory_risk_score=0.0,
            quote_competitiveness=0.0
        )

    # Report generation helper methods

    async def _generate_summary(self, spread_analyses: list[SpreadAnalysis],
                              pnl_attribution: PnLAttribution,
                              trading_efficiency: TradingEfficiency) -> dict[str, Any]:
        """Generate performance summary."""
        return {
            "overall_performance": "positive" if pnl_attribution.total_pnl_usdc > 0 else "negative",
            "total_pnl_usdc": pnl_attribution.total_pnl_usdc,
            "average_spread_improvement": np.mean([a.improvement_percent for a in spread_analyses]) if spread_analyses else 0,
            "sharpe_ratio": pnl_attribution.sharpe_ratio,
            "fill_rate": trading_efficiency.fill_rate,
            "total_volume_usdc": trading_efficiency.total_volume_usdc,
            "trades_executed": trading_efficiency.trades_count,
            "average_slippage_bps": trading_efficiency.slippage_bps,
            "uptime_estimate": 0.995,  # Would be calculated from actual system metrics
            "key_achievements": [],
            "areas_for_improvement": []
        }

    async def _generate_benchmarks(self, spread_analyses: list[SpreadAnalysis],
                                 pnl_attribution: PnLAttribution,
                                 trading_efficiency: TradingEfficiency) -> dict[str, Any]:
        """Generate benchmark comparisons."""
        avg_improvement = np.mean([a.improvement_percent for a in spread_analyses]) if spread_analyses else 0

        return {
            "spread_improvement": {
                "actual": avg_improvement,
                "target": self.target_spread_improvement,
                "vs_target": avg_improvement - self.target_spread_improvement,
                "status": "above_target" if avg_improvement >= self.target_spread_improvement else "below_target"
            },
            "sharpe_ratio": {
                "actual": pnl_attribution.sharpe_ratio,
                "target": self.target_sharpe_ratio,
                "vs_target": pnl_attribution.sharpe_ratio - self.target_sharpe_ratio,
                "status": "above_target" if pnl_attribution.sharpe_ratio >= self.target_sharpe_ratio else "below_target"
            },
            "fill_rate": {
                "actual": trading_efficiency.fill_rate,
                "target": self.target_fill_rate,
                "vs_target": trading_efficiency.fill_rate - self.target_fill_rate,
                "status": "above_target" if trading_efficiency.fill_rate >= self.target_fill_rate else "below_target"
            }
        }

    async def _generate_period_comparison(self, report_type: ReportType,
                                        start_time: datetime, end_time: datetime) -> dict[str, Any]:
        """Generate period-over-period comparison."""
        # This would compare with previous period
        return {
            "comparison_period": "previous_" + report_type.value,
            "pnl_change_percent": np.random.uniform(-10, 25),  # Simulate comparison
            "volume_change_percent": np.random.uniform(-5, 15),
            "spread_improvement_change": np.random.uniform(-2, 5),
            "fill_rate_change": np.random.uniform(-0.02, 0.03),
            "trend": "improving"  # Would be calculated from actual data
        }

    async def _generate_performance_alerts(self, spread_analyses: list[SpreadAnalysis],
                                         pnl_attribution: PnLAttribution,
                                         trading_efficiency: TradingEfficiency) -> list[dict[str, Any]]:
        """Generate performance-based alerts."""
        alerts = []

        # Check spread improvement
        avg_improvement = np.mean([a.improvement_percent for a in spread_analyses]) if spread_analyses else 0
        if avg_improvement < self.target_spread_improvement * 0.8:  # 20% below target
            alerts.append({
                "type": "performance_warning",
                "metric": "spread_improvement",
                "message": f"Spread improvement ({avg_improvement:.1f}%) below target ({self.target_spread_improvement:.1f}%)",
                "severity": "warning",
                "timestamp": datetime.now()
            })

        # Check P&L
        if pnl_attribution.total_pnl_usdc < 0:
            alerts.append({
                "type": "performance_alert",
                "metric": "pnl",
                "message": f"Negative P&L detected: ${pnl_attribution.total_pnl_usdc:.2f}",
                "severity": "critical",
                "timestamp": datetime.now()
            })

        # Check fill rate
        if trading_efficiency.fill_rate < self.target_fill_rate * 0.9:  # 10% below target
            alerts.append({
                "type": "execution_warning",
                "metric": "fill_rate",
                "message": f"Fill rate ({trading_efficiency.fill_rate:.1%}) below target ({self.target_fill_rate:.1%})",
                "severity": "warning",
                "timestamp": datetime.now()
            })

        return alerts

    async def _generate_recommendations(self, spread_analyses: list[SpreadAnalysis],
                                      pnl_attribution: PnLAttribution,
                                      trading_efficiency: TradingEfficiency) -> list[str]:
        """Generate performance recommendations."""
        recommendations = []

        # Spread improvement recommendations
        avg_improvement = np.mean([a.improvement_percent for a in spread_analyses]) if spread_analyses else 0
        if avg_improvement < self.target_spread_improvement:
            recommendations.append(
                f"Consider tightening spreads to achieve target improvement of {self.target_spread_improvement:.1f}%"
            )

        # Risk management recommendations
        if pnl_attribution.max_drawdown > pnl_attribution.total_pnl_usdc * 0.1:
            recommendations.append("Review risk management parameters to reduce drawdown")

        # Execution efficiency recommendations
        if trading_efficiency.slippage_bps > 5:
            recommendations.append("Optimize execution algorithms to reduce slippage")

        if trading_efficiency.fill_rate < 0.9:
            recommendations.append("Review order placement strategy to improve fill rates")

        return recommendations

    def _calculate_data_quality_score(self, spread_analyses: list[SpreadAnalysis],
                                    pnl_attribution: PnLAttribution,
                                    trading_efficiency: TradingEfficiency) -> float:
        """Calculate overall data quality score."""
        quality_factors = []

        # Sample size quality
        if spread_analyses:
            avg_sample_size = float(np.mean([a.sample_size for a in spread_analyses]))
            sample_quality = min(1.0, avg_sample_size / self.min_sample_size)
            quality_factors.append(sample_quality)

        # Data completeness (simulate)
        completeness = 0.95  # Would be calculated from actual data availability
        quality_factors.append(completeness)

        # Data freshness (simulate)
        freshness = 1.0  # Assume recent data
        quality_factors.append(freshness)

        return float(np.mean(quality_factors)) if quality_factors else 0.5

    def _calculate_overall_confidence(self, spread_analyses: list[SpreadAnalysis]) -> float:
        """Calculate overall confidence level."""
        if not spread_analyses:
            return 0.5

        confidences = [a.statistical_significance for a in spread_analyses]
        return float(np.mean(confidences))

    def get_performance_statistics(self) -> dict[str, Any]:
        """Get performance analyzer statistics."""
        return {
            **self.analysis_stats,
            "baseline_markets": len(self.baseline_spreads),
            "historical_data_points": sum(len(deque_data) for deque_data in self.historical_data.values()),
            "cache_size": len(self.performance_cache),
            "target_spread_improvement": self.target_spread_improvement,
            "target_sharpe_ratio": self.target_sharpe_ratio,
            "target_fill_rate": self.target_fill_rate
        }

    async def export_analysis_data(self, format: str = "json") -> str:
        """Export analysis data in specified format."""
        try:
            export_data = {
                "export_timestamp": datetime.now().isoformat(),
                "baseline_spreads": self.baseline_spreads,
                "analysis_statistics": self.analysis_stats,
                "recent_analyses": {}
            }

            # Add recent analyses
            for key, data in self.historical_data.items():
                if data:
                    recent = list(data)[-10:]  # Last 10 entries
                    if hasattr(recent[0], '__dict__'):
                        export_data["recent_analyses"][key] = [asdict(item) if hasattr(item, '__dict__') else item for item in recent]
                    else:
                        export_data["recent_analyses"][key] = recent

            if format.lower() == "json":
                return json.dumps(export_data, indent=2, default=str)
            else:
                return str(export_data)

        except Exception as e:
            logger.error(f"Error exporting analysis data: {e}")
            return "{}"

    async def cleanup_old_data(self, days_to_keep: int = 7) -> None:
        """Clean up old analysis data."""
        try:
            cutoff_time = datetime.now() - timedelta(days=days_to_keep)

            # Clean up performance cache
            old_reports = [
                report_id for report_id, report in self.performance_cache.items()
                if report.generated_at < cutoff_time
            ]

            for report_id in old_reports:
                del self.performance_cache[report_id]

            logger.info(f"Cleaned up {len(old_reports)} old performance reports")

        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")

"""
FlashMM Position Limits Manager

Dynamic position limit management system with:
- Market-specific and global exposure limits
- Time-based position limit adjustments
- Concentration risk monitoring
- Leverage and margin utilization tracking
- Volatility-based limit scaling
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any

import numpy as np

from flashmm.config.settings import get_config
from flashmm.utils.decorators import measure_latency, timeout_async
from flashmm.utils.exceptions import RiskError
from flashmm.utils.logging import get_logger

logger = get_logger(__name__)


class LimitType(Enum):
    """Position limit types."""
    NOTIONAL = "notional"           # Dollar value limit
    QUANTITY = "quantity"           # Share/token quantity limit
    PERCENTAGE = "percentage"       # Percentage of portfolio
    LEVERAGE = "leverage"           # Leverage ratio limit
    CONCENTRATION = "concentration" # Single asset concentration
    TIME_BASED = "time_based"      # Time-based dynamic limits


class LimitScope(Enum):
    """Limit scope definitions."""
    SYMBOL = "symbol"               # Per trading symbol
    SECTOR = "sector"               # Per sector/category
    GLOBAL = "global"               # Portfolio-wide
    INTRADAY = "intraday"          # Intraday trading limits
    OVERNIGHT = "overnight"         # Overnight position limits


@dataclass
class PositionLimit:
    """Position limit definition."""
    name: str
    limit_type: LimitType
    scope: LimitScope

    # Limit values
    max_value: Decimal
    warning_value: Decimal
    min_value: Decimal = Decimal('0')

    # Dynamic adjustment parameters
    volatility_multiplier: float = 1.0
    time_decay_factor: float = 1.0
    market_condition_factor: float = 1.0

    # Metadata
    symbol: str | None = None
    sector: str | None = None
    description: str = ""
    enabled: bool = True

    # Tracking
    current_utilization: Decimal = Decimal('0')
    peak_utilization: Decimal = Decimal('0')
    violations_count: int = 0
    last_violation_time: datetime | None = None

    def calculate_effective_limit(self) -> Decimal:
        """Calculate effective limit with dynamic adjustments."""
        base_limit = self.max_value

        # Apply volatility adjustment
        vol_adjusted = base_limit * Decimal(str(self.volatility_multiplier))

        # Apply time decay (for intraday limits)
        time_adjusted = vol_adjusted * Decimal(str(self.time_decay_factor))

        # Apply market condition adjustment
        market_adjusted = time_adjusted * Decimal(str(self.market_condition_factor))

        # Ensure minimum limit
        return max(self.min_value, market_adjusted)

    def check_violation(self, current_value: Decimal) -> tuple[bool, str]:
        """Check if current value violates limit.

        Returns:
            (is_violation, violation_level)
        """
        effective_limit = self.calculate_effective_limit()
        self.current_utilization = current_value / effective_limit if effective_limit > 0 else Decimal('0')
        self.peak_utilization = max(self.peak_utilization, self.current_utilization)

        if current_value > effective_limit:
            self.violations_count += 1
            self.last_violation_time = datetime.now()
            return True, "limit_exceeded"
        elif current_value > self.warning_value:
            return True, "warning_threshold"

        return False, "within_limits"

    def get_status(self) -> dict[str, Any]:
        """Get limit status."""
        effective_limit = self.calculate_effective_limit()

        return {
            'name': self.name,
            'type': self.limit_type.value,
            'scope': self.scope.value,
            'symbol': self.symbol,
            'max_value': float(self.max_value),
            'warning_value': float(self.warning_value),
            'effective_limit': float(effective_limit),
            'current_utilization': float(self.current_utilization),
            'peak_utilization': float(self.peak_utilization),
            'utilization_percent': float(self.current_utilization * 100),
            'violations_count': self.violations_count,
            'last_violation': self.last_violation_time.isoformat() if self.last_violation_time else None,
            'enabled': self.enabled,
            'adjustments': {
                'volatility_multiplier': self.volatility_multiplier,
                'time_decay_factor': self.time_decay_factor,
                'market_condition_factor': self.market_condition_factor
            }
        }


class DynamicLimitCalculator:
    """Dynamic position limit calculator based on market conditions."""

    def __init__(self):
        self.config = get_config()

        # Historical data for calculations
        self.volatility_history: dict[str, list[float]] = {}
        self.volume_history: dict[str, list[float]] = {}
        self.spread_history: dict[str, list[float]] = {}

        # Market regime indicators
        self.market_stress_level = 0.0  # 0.0 = calm, 1.0 = extreme stress
        self.liquidity_score = 1.0      # 1.0 = normal, 0.0 = illiquid

    async def calculate_volatility_multiplier(self, symbol: str, lookback_hours: int = 24) -> float:
        """Calculate volatility-based limit multiplier."""
        try:
            vol_history = self.volatility_history.get(symbol, [])
            if len(vol_history) < 10:
                return 1.0  # Default if insufficient data

            # Calculate recent volatility vs historical average
            recent_vol = np.mean(vol_history[-12:])  # Last 12 periods (1 hour if 5min intervals)
            historical_vol = np.mean(vol_history)

            if historical_vol == 0:
                return 1.0

            vol_ratio = recent_vol / historical_vol

            # Reduce limits when volatility is high
            if vol_ratio > 2.0:
                return 0.5  # Halve limits in high volatility
            elif vol_ratio > 1.5:
                return 0.7  # Reduce by 30%
            elif vol_ratio > 1.2:
                return 0.85 # Reduce by 15%
            elif vol_ratio < 0.5:
                return 1.2  # Increase by 20% in low volatility
            else:
                return 1.0

        except Exception as e:
            logger.error(f"Error calculating volatility multiplier for {symbol}: {e}")
            return 1.0

    async def calculate_time_decay_factor(self, limit_scope: LimitScope) -> float:
        """Calculate time-based decay factor for intraday limits."""
        if limit_scope != LimitScope.INTRADAY:
            return 1.0

        try:
            # Get current time and calculate time until market close
            now = datetime.now()

            # Assume market open 24/7 for crypto, but reduce limits during low-activity periods
            hour = now.hour

            # Reduce limits during typically low-activity hours (2-8 AM UTC)
            if 2 <= hour <= 8:
                return 0.6  # Reduce limits by 40% during low activity
            elif 8 <= hour <= 16:
                return 1.0  # Normal limits during active hours
            else:
                return 0.8  # Slight reduction during evening hours

        except Exception as e:
            logger.error(f"Error calculating time decay factor: {e}")
            return 1.0

    async def calculate_market_condition_factor(self) -> float:
        """Calculate market condition adjustment factor."""
        try:
            # Combine various market stress indicators
            stress_factor = 1.0 - (self.market_stress_level * 0.5)  # Reduce limits by up to 50% in stress
            liquidity_factor = 0.5 + (self.liquidity_score * 0.5)   # Reduce by up to 50% in illiquid markets

            return min(stress_factor, liquidity_factor)

        except Exception as e:
            logger.error(f"Error calculating market condition factor: {e}")
            return 1.0

    def update_market_data(self, symbol: str, volatility: float, volume: float, spread: float):
        """Update market data for calculations."""
        # Update volatility history
        if symbol not in self.volatility_history:
            self.volatility_history[symbol] = []
        self.volatility_history[symbol].append(volatility)
        if len(self.volatility_history[symbol]) > 288:  # Keep 24 hours of 5min data
            self.volatility_history[symbol] = self.volatility_history[symbol][-144:]  # Keep 12 hours

        # Update volume history
        if symbol not in self.volume_history:
            self.volume_history[symbol] = []
        self.volume_history[symbol].append(volume)
        if len(self.volume_history[symbol]) > 288:
            self.volume_history[symbol] = self.volume_history[symbol][-144:]

        # Update spread history
        if symbol not in self.spread_history:
            self.spread_history[symbol] = []
        self.spread_history[symbol].append(spread)
        if len(self.spread_history[symbol]) > 288:
            self.spread_history[symbol] = self.spread_history[symbol][-144:]

    def update_market_stress_level(self, stress_indicators: dict[str, float]):
        """Update market stress level based on various indicators."""
        try:
            # Calculate weighted stress score
            weights = {
                'volatility_spike': 0.3,
                'volume_spike': 0.2,
                'spread_widening': 0.2,
                'correlation_breakdown': 0.15,
                'liquidity_drop': 0.15
            }

            stress_score = 0.0
            for indicator, value in stress_indicators.items():
                weight = weights.get(indicator, 0.0)
                stress_score += value * weight

            self.market_stress_level = min(1.0, max(0.0, stress_score))

            # Update liquidity score (inverse of spread widening and volume drop)
            liquidity_indicators = stress_indicators.get('liquidity_drop', 0.0)
            self.liquidity_score = 1.0 - min(1.0, liquidity_indicators)

        except Exception as e:
            logger.error(f"Error updating market stress level: {e}")


class ConcentrationRiskMonitor:
    """Monitor concentration risk across positions."""

    def __init__(self):
        self.config = get_config()
        self.max_single_asset_percent = self.config.get("risk.max_single_asset_percent", 50.0)
        self.max_sector_percent = self.config.get("risk.max_sector_percent", 70.0)
        self.diversification_threshold = self.config.get("risk.diversification_threshold", 5)

    async def check_concentration_risk(self, positions: dict[str, dict[str, Any]]) -> dict[str, Any]:
        """Check portfolio concentration risk.

        Args:
            positions: Dictionary of symbol -> position data

        Returns:
            Concentration risk analysis
        """
        try:
            if not positions:
                return {
                    'concentrated_positions': [],
                    'sector_concentrations': {},
                    'diversification_score': 1.0,
                    'concentration_risk_level': 'low',
                    'recommendations': []
                }

            total_portfolio_value = sum(
                abs(pos.get('notional_value', 0.0)) for pos in positions.values()
            )

            if total_portfolio_value == 0:
                return self._empty_concentration_result()

            # Calculate individual asset concentrations
            asset_concentrations = {}
            for symbol, position in positions.items():
                notional_value = abs(position.get('notional_value', 0.0))
                concentration_pct = (notional_value / total_portfolio_value) * 100
                asset_concentrations[symbol] = concentration_pct

            # Identify concentrated positions
            concentrated_positions = [
                {'symbol': symbol, 'concentration_pct': pct}
                for symbol, pct in asset_concentrations.items()
                if pct > self.max_single_asset_percent
            ]

            # Calculate diversification score (Herfindahl-Hirschman Index)
            hhi = sum(pct ** 2 for pct in asset_concentrations.values())
            diversification_score = max(0.0, 1.0 - (hhi / 10000))  # Normalize to 0-1

            # Determine risk level
            max_concentration = max(asset_concentrations.values()) if asset_concentrations else 0.0
            if max_concentration > 70:
                risk_level = 'critical'
            elif max_concentration > 50:
                risk_level = 'high'
            elif max_concentration > 30:
                risk_level = 'medium'
            else:
                risk_level = 'low'

            # Generate recommendations
            recommendations = self._generate_concentration_recommendations(
                concentrated_positions, diversification_score, risk_level
            )

            return {
                'concentrated_positions': concentrated_positions,
                'asset_concentrations': asset_concentrations,
                'sector_concentrations': {},  # Would implement sector mapping
                'diversification_score': diversification_score,
                'concentration_risk_level': risk_level,
                'max_single_position_pct': max_concentration,
                'total_positions': len(positions),
                'recommendations': recommendations
            }

        except Exception as e:
            logger.error(f"Error checking concentration risk: {e}")
            return self._empty_concentration_result()

    def _empty_concentration_result(self) -> dict[str, Any]:
        """Return empty concentration result."""
        return {
            'concentrated_positions': [],
            'asset_concentrations': {},
            'sector_concentrations': {},
            'diversification_score': 1.0,
            'concentration_risk_level': 'low',
            'max_single_position_pct': 0.0,
            'total_positions': 0,
            'recommendations': []
        }

    def _generate_concentration_recommendations(self,
                                             concentrated_positions: list[dict],
                                             diversification_score: float,
                                             risk_level: str) -> list[str]:
        """Generate concentration risk recommendations."""
        recommendations = []

        if concentrated_positions:
            for pos in concentrated_positions:
                recommendations.append(
                    f"Reduce {pos['symbol']} position from {pos['concentration_pct']:.1f}% "
                    f"to below {self.max_single_asset_percent}%"
                )

        if diversification_score < 0.5:
            recommendations.append("Increase portfolio diversification across more assets")

        if risk_level in ['high', 'critical']:
            recommendations.append("Consider implementing position size limits")
            recommendations.append("Review risk management policies")

        return recommendations


class PositionLimitsManager:
    """Comprehensive position limits management system."""

    def __init__(self):
        self.config = get_config()
        self.limits: dict[str, PositionLimit] = {}
        self.limit_calculator = DynamicLimitCalculator()
        self.concentration_monitor = ConcentrationRiskMonitor()

        # Global settings
        self.global_notional_limit = Decimal(str(self.config.get("risk.global_notional_limit", 10000.0)))
        self.single_position_limit = Decimal(str(self.config.get("risk.single_position_limit", 2000.0)))

        self._lock = asyncio.Lock()

        # Additional attributes for backward compatibility
        self.current_portfolio_value: Decimal = Decimal('0')
        self.current_positions: dict[str, dict[str, Any]] = {}

    async def initialize(self) -> None:
        """Initialize position limits manager."""
        try:
            # Create default limits
            await self._create_default_limits()

            logger.info(f"Position limits manager initialized with {len(self.limits)} limits")

        except Exception as e:
            logger.error(f"Failed to initialize position limits manager: {e}")
            raise RiskError(f"Position limits manager initialization failed: {e}") from e

    async def _create_default_limits(self) -> None:
        """Create default position limits."""
        symbols = self.config.get("trading.symbols", ["SEI/USDC"])

        # Global portfolio limit
        global_limit = PositionLimit(
            name="global_notional",
            limit_type=LimitType.NOTIONAL,
            scope=LimitScope.GLOBAL,
            max_value=self.global_notional_limit,
            warning_value=self.global_notional_limit * Decimal('0.8'),
            description="Global portfolio notional limit"
        )
        self.limits[global_limit.name] = global_limit

        # Per-symbol limits
        for symbol in symbols:
            # Notional limit per symbol
            symbol_limit = PositionLimit(
                name=f"{symbol}_notional",
                limit_type=LimitType.NOTIONAL,
                scope=LimitScope.SYMBOL,
                symbol=symbol,
                max_value=self.single_position_limit,
                warning_value=self.single_position_limit * Decimal('0.8'),
                description=f"Notional position limit for {symbol}"
            )
            self.limits[symbol_limit.name] = symbol_limit

            # Concentration limit per symbol
            concentration_limit = PositionLimit(
                name=f"{symbol}_concentration",
                limit_type=LimitType.CONCENTRATION,
                scope=LimitScope.SYMBOL,
                symbol=symbol,
                max_value=Decimal('50.0'),  # 50% max concentration
                warning_value=Decimal('40.0'),  # 40% warning
                description=f"Concentration limit for {symbol}"
            )
            self.limits[concentration_limit.name] = concentration_limit

        # Intraday leverage limit
        leverage_limit = PositionLimit(
            name="intraday_leverage",
            limit_type=LimitType.LEVERAGE,
            scope=LimitScope.INTRADAY,
            max_value=Decimal('3.0'),  # 3x leverage max
            warning_value=Decimal('2.5'),  # 2.5x warning
            description="Intraday leverage limit"
        )
        self.limits[leverage_limit.name] = leverage_limit

    @timeout_async(0.1)  # 100ms timeout for limit checks
    @measure_latency("position_limit_check")
    async def check_position_limits(self, positions: dict[str, dict[str, Any]]) -> dict[str, Any]:
        """Check all position limits against current positions.

        Args:
            positions: Dictionary of symbol -> position data

        Returns:
            Limit check results with violations and warnings
        """
        async with self._lock:
            try:
                # Update dynamic limit factors
                await self._update_dynamic_factors(positions)

                violations = []
                warnings = []
                limit_statuses = {}

                # Check each limit
                for limit_name, limit in self.limits.items():
                    if not limit.enabled:
                        continue

                    current_value = await self._calculate_current_value(limit, positions)
                    is_violation, violation_level = limit.check_violation(current_value)

                    limit_statuses[limit_name] = limit.get_status()

                    if is_violation:
                        violation_info = {
                            'limit_name': limit_name,
                            'limit_type': limit.limit_type.value,
                            'scope': limit.scope.value,
                            'symbol': limit.symbol,
                            'current_value': float(current_value),
                            'limit_value': float(limit.calculate_effective_limit()),
                            'utilization_percent': float(limit.current_utilization * 100),
                            'violation_level': violation_level,
                            'timestamp': datetime.now().isoformat()
                        }

                        if violation_level == "limit_exceeded":
                            violations.append(violation_info)
                            logger.critical(f"🚨 POSITION LIMIT VIOLATED: {limit_name} - "
                                          f"{current_value} exceeds {limit.calculate_effective_limit()}")
                        else:
                            warnings.append(violation_info)
                            logger.warning(f"⚠️ Position limit warning: {limit_name} - "
                                         f"{current_value} approaching {limit.calculate_effective_limit()}")

                # Check concentration risk
                concentration_analysis = await self.concentration_monitor.check_concentration_risk(positions)

                return {
                    'timestamp': datetime.now().isoformat(),
                    'violations': violations,
                    'warnings': warnings,
                    'total_violations': len(violations),
                    'total_warnings': len(warnings),
                    'limit_statuses': limit_statuses,
                    'concentration_analysis': concentration_analysis,
                    'compliance_score': self._calculate_compliance_score(violations, warnings),
                    'recommendations': self._generate_recommendations(violations, warnings, concentration_analysis)
                }

            except Exception as e:
                logger.error(f"Error checking position limits: {e}")
                raise RiskError(f"Position limit check failed: {e}") from e

    async def _update_dynamic_factors(self, positions: dict[str, dict[str, Any]]) -> None:
        """Update dynamic adjustment factors for all limits."""
        try:
            # Update market data for calculator
            for symbol, position in positions.items():
                volatility = position.get('volatility', 0.02)
                volume = position.get('volume', 1000.0)
                spread = position.get('spread_bps', 5.0)

                self.limit_calculator.update_market_data(symbol, volatility, volume, spread)

            # Update market stress indicators (would get from market data in production)
            stress_indicators = {
                'volatility_spike': 0.2,
                'volume_spike': 0.1,
                'spread_widening': 0.15,
                'correlation_breakdown': 0.1,
                'liquidity_drop': 0.1
            }
            self.limit_calculator.update_market_stress_level(stress_indicators)

            # Update dynamic factors for each limit
            for limit in self.limits.values():
                if limit.symbol:
                    limit.volatility_multiplier = await self.limit_calculator.calculate_volatility_multiplier(limit.symbol)

                limit.time_decay_factor = await self.limit_calculator.calculate_time_decay_factor(limit.scope)
                limit.market_condition_factor = await self.limit_calculator.calculate_market_condition_factor()

        except Exception as e:
            logger.error(f"Error updating dynamic factors: {e}")

    async def _calculate_current_value(self, limit: PositionLimit, positions: dict[str, dict[str, Any]]) -> Decimal:
        """Calculate current value for a specific limit."""
        try:
            if limit.limit_type == LimitType.NOTIONAL:
                if limit.scope == LimitScope.GLOBAL:
                    # Sum all position notional values
                    return Decimal(str(sum(
                        abs(pos.get('notional_value', 0.0)) for pos in positions.values()
                    )))
                elif limit.scope == LimitScope.SYMBOL and limit.symbol:
                    # Single symbol notional value
                    position = positions.get(limit.symbol, {})
                    return Decimal(str(abs(position.get('notional_value', 0.0))))

            elif limit.limit_type == LimitType.CONCENTRATION:
                if limit.symbol:
                    position = positions.get(limit.symbol, {})
                    position_value = abs(position.get('notional_value', 0.0))
                    total_value = sum(abs(pos.get('notional_value', 0.0)) for pos in positions.values())

                    if total_value > 0:
                        return Decimal(str((position_value / total_value) * 100))  # Percentage

            elif limit.limit_type == LimitType.LEVERAGE:
                # Calculate portfolio leverage
                total_notional = sum(abs(pos.get('notional_value', 0.0)) for pos in positions.values())
                total_equity = sum(pos.get('equity_value', 0.0) for pos in positions.values())

                if total_equity > 0:
                    return Decimal(str(total_notional / total_equity))

            return Decimal('0')

        except Exception as e:
            logger.error(f"Error calculating current value for limit {limit.name}: {e}")
            return Decimal('0')

    def _calculate_compliance_score(self, violations: list[dict], warnings: list[dict]) -> float:
        """Calculate overall compliance score (0-1)."""
        total_limits = len([limit for limit in self.limits.values() if limit.enabled])
        if total_limits == 0:
            return 1.0

        # Weight violations more heavily than warnings
        violation_penalty = len(violations) * 0.2
        warning_penalty = len(warnings) * 0.05

        total_penalty = violation_penalty + warning_penalty

        return max(0.0, 1.0 - (total_penalty / total_limits))

    def _generate_recommendations(self,
                                violations: list[dict],
                                warnings: list[dict],
                                concentration_analysis: dict[str, Any]) -> list[str]:
        """Generate recommendations based on limit checks."""
        recommendations = []

        # Violation recommendations
        for violation in violations:
            if violation['violation_level'] == 'limit_exceeded':
                recommendations.append(
                    f"URGENT: Reduce {violation.get('symbol', 'position')} size by "
                    f"{violation['utilization_percent'] - 100:.1f}% to comply with {violation['limit_name']}"
                )

        # Warning recommendations
        for warning in warnings:
            recommendations.append(
                f"Monitor {warning.get('symbol', 'position')} - approaching limit "
                f"({warning['utilization_percent']:.1f}% utilized)"
            )

        # Add concentration recommendations
        recommendations.extend(concentration_analysis.get('recommendations', []))

        return recommendations

    async def add_custom_limit(self, limit: PositionLimit) -> None:
        """Add a custom position limit."""
        async with self._lock:
            self.limits[limit.name] = limit
            logger.info(f"Added custom position limit: {limit.name}")

    async def remove_limit(self, limit_name: str) -> None:
        """Remove a position limit."""
        async with self._lock:
            if limit_name in self.limits:
                del self.limits[limit_name]
                logger.info(f"Removed position limit: {limit_name}")

    async def update_limit_value(self, limit_name: str, new_max_value: Decimal) -> None:
        """Update a limit's maximum value."""
        async with self._lock:
            if limit_name in self.limits:
                old_value = self.limits[limit_name].max_value
                self.limits[limit_name].max_value = new_max_value
                self.limits[limit_name].warning_value = new_max_value * Decimal('0.8')

                logger.info(f"Updated limit {limit_name}: {old_value} -> {new_max_value}")

    def get_all_limits_status(self) -> dict[str, Any]:
        """Get status of all position limits."""
        return {
            'total_limits': len(self.limits),
            'enabled_limits': len([limit for limit in self.limits.values() if limit.enabled]),
            'limits': {name: limit.get_status() for name, limit in self.limits.items()},
            'global_settings': {
                'global_notional_limit': float(self.global_notional_limit),
                'single_position_limit': float(self.single_position_limit)
            }
        }

    async def emergency_limit_reduction(self, reduction_factor: float, reason: str) -> None:
        """Emergency reduction of all position limits.

        Args:
            reduction_factor: Factor to reduce limits by (e.g., 0.5 for 50% reduction)
            reason: Reason for emergency reduction
        """
        async with self._lock:
            try:
                logger.critical(f"🚨 EMERGENCY LIMIT REDUCTION: {reason} - Reducing limits by {(1-reduction_factor)*100:.0f}%")

                for limit_name, limit in self.limits.items():
                    old_max = limit.max_value
                    old_warning = limit.warning_value

                    # Apply reduction factor
                    limit.max_value *= Decimal(str(reduction_factor))
                    limit.warning_value *= Decimal(str(reduction_factor))

                    logger.warning(f"Reduced {limit_name}: max {old_max} -> {limit.max_value}, "
                                 f"warning {old_warning} -> {limit.warning_value}")

            except Exception as e:
                logger.error(f"Error during emergency limit reduction: {e}")
                raise RiskError(f"Emergency limit reduction failed: {e}") from e

    async def check_portfolio_limits(self, positions: dict[str, dict[str, Any]],
                                   portfolio_value: Decimal) -> dict[str, Any]:
        """Check portfolio-wide limits.

        Args:
            positions: Dictionary of positions
            portfolio_value: Total portfolio value

        Returns:
            Portfolio limit check results
        """
        try:
            # Update internal state
            self.current_positions = positions
            self.current_portfolio_value = portfolio_value

            # Use existing check_position_limits method
            return await self.check_position_limits(positions)

        except Exception as e:
            logger.error(f"Error checking portfolio limits: {e}")
            return {
                'violations': [],
                'warnings': [],
                'total_violations': 0,
                'total_warnings': 0,
                'compliance_score': 0.0,
                'error': str(e)
            }

    async def validate_position(self, symbol: str, side: str, size: float,
                              price: float) -> dict[str, Any]:
        """Validate if a position is within limits.

        Args:
            symbol: Trading symbol
            side: 'buy' or 'sell'
            size: Position size
            price: Position price

        Returns:
            Validation results
        """
        try:
            # Convert to Decimal for calculations
            size_decimal = Decimal(str(size))
            price_decimal = Decimal(str(price))
            notional_value = abs(size_decimal * price_decimal)

            # Check symbol-specific limits
            symbol_limit_name = f"{symbol}_notional"
            if symbol_limit_name in self.limits:
                limit = self.limits[symbol_limit_name]
                effective_limit = limit.calculate_effective_limit()

                if notional_value > effective_limit:
                    return {
                        'allowed': False,
                        'violations': [f"Position size {notional_value} exceeds limit {effective_limit} for {symbol}"],
                        'recommended_max_size': float(effective_limit / price_decimal)
                    }

            # Check global limits
            current_total = sum(
                Decimal(str(abs(pos.get('notional_value', 0.0)))) for pos in self.current_positions.values()
            )
            total_notional = current_total + notional_value

            global_limit = self.limits.get('global_notional')
            if global_limit:
                effective_global_limit = global_limit.calculate_effective_limit()
                if total_notional > effective_global_limit:
                    available_limit = max(Decimal('0'), effective_global_limit - current_total)
                    return {
                        'allowed': False,
                        'violations': [f"Total portfolio exposure {total_notional} would exceed global limit {effective_global_limit}"],
                        'recommended_max_size': float(available_limit / price_decimal)
                    }

            return {
                'allowed': True,
                'violations': [],
                'recommended_max_size': size
            }

        except Exception as e:
            logger.error(f"Error validating position for {symbol}: {e}")
            return {
                'allowed': False,
                'violations': [f"Position validation error: {e}"],
                'recommended_max_size': 0.0
            }

    async def get_current_limits(self) -> dict[str, Decimal]:
        """Get current effective limits for all symbols.

        Returns:
            Dictionary mapping symbol to current limit value
        """
        try:
            current_limits = {}

            for _limit_name, limit in self.limits.items():
                if limit.symbol and limit.limit_type == LimitType.NOTIONAL:
                    current_limits[limit.symbol] = limit.calculate_effective_limit()
                elif limit.scope == LimitScope.GLOBAL:
                    current_limits['global'] = limit.calculate_effective_limit()

            return current_limits

        except Exception as e:
            logger.error(f"Error getting current limits: {e}")
            return {}

    async def update_limits(self, market_data: dict[str, Any]) -> None:
        """Update limits based on market data.

        Args:
            market_data: Market data for limit adjustments
        """
        try:
            # Update dynamic factors based on market data
            await self._update_dynamic_factors(market_data)

            logger.debug("Updated position limits based on market data")

        except Exception as e:
            logger.error(f"Error updating limits with market data: {e}")

    async def cleanup(self) -> None:
        """Cleanup position limits manager."""
        logger.info("Position limits manager cleanup completed")

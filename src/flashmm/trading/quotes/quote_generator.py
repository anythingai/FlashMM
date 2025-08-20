"""
FlashMM Quote Generator

Advanced quote generation with ML-driven pricing, spread optimization,
and sophisticated risk controls for market making operations.
"""

import asyncio
import math
from typing import Dict, List, Optional, Tuple, Any
from decimal import Decimal, ROUND_HALF_UP
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum

from flashmm.config.settings import get_config
from flashmm.utils.logging import get_logger
from flashmm.utils.exceptions import QuotingError, ValidationError
from flashmm.utils.decorators import measure_latency, timeout_async

logger = get_logger(__name__)


class QuoteValidationLevel(Enum):
    """Quote validation strictness levels."""
    STRICT = "strict"
    NORMAL = "normal"
    RELAXED = "relaxed"


class SpreadOptimizationMode(Enum):
    """Spread optimization strategies."""
    AGGRESSIVE = "aggressive"  # Tight spreads for high volume
    CONSERVATIVE = "conservative"  # Wide spreads for stability
    ADAPTIVE = "adaptive"  # Dynamic based on conditions
    BALANCED = "balanced"  # Balanced approach


@dataclass
class QuoteParameters:
    """Quote generation parameters."""
    symbol: str
    base_spread_bps: float
    min_spread_bps: float
    max_spread_bps: float
    base_size: Decimal
    min_size: Decimal
    max_size: Decimal
    max_levels: int
    prediction_weight: float
    inventory_skew_factor: float
    volatility_multiplier: float
    liquidity_threshold: float


@dataclass
class MarketConditions:
    """Market conditions for quote optimization."""
    mid_price: Decimal
    best_bid: Decimal
    best_ask: Decimal
    spread_bps: float
    volatility: float
    liquidity_score: float
    order_book_imbalance: float
    recent_volume: float
    competition_tightness: float
    market_stress_level: float


@dataclass
class PredictionContext:
    """ML prediction context for quote generation."""
    direction: str
    confidence: float
    price_change_bps: float
    signal_strength: float
    uncertainty_score: float
    time_horizon_ms: int
    model_version: str
    ensemble_agreement: float


@dataclass
class Quote:
    """Individual quote with comprehensive metadata."""
    symbol: str
    side: str  # 'buy' or 'sell'
    price: Decimal
    size: Decimal
    level: int
    spread_bps: float
    confidence_score: float
    expected_fill_probability: float
    adverse_selection_score: float
    profit_expectation: float
    skew_adjustment: float
    volatility_adjustment: float
    timestamp: datetime
    quote_id: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert quote to dictionary."""
        return {
            'symbol': self.symbol,
            'side': self.side,
            'price': float(self.price),
            'size': float(self.size),
            'level': self.level,
            'spread_bps': self.spread_bps,
            'confidence_score': self.confidence_score,
            'expected_fill_probability': self.expected_fill_probability,
            'adverse_selection_score': self.adverse_selection_score,
            'profit_expectation': self.profit_expectation,
            'skew_adjustment': self.skew_adjustment,
            'volatility_adjustment': self.volatility_adjustment,
            'timestamp': self.timestamp.isoformat(),
            'quote_id': self.quote_id
        }


@dataclass
class QuoteSet:
    """Set of quotes with optimization metadata."""
    symbol: str
    quotes: List[Quote]
    bid_quotes: List[Quote]
    ask_quotes: List[Quote]
    total_bid_size: Decimal
    total_ask_size: Decimal
    weighted_bid_price: Decimal
    weighted_ask_price: Decimal
    effective_spread_bps: float
    expected_profitability: float
    risk_score: float
    optimization_mode: SpreadOptimizationMode
    generation_latency_ms: float
    timestamp: datetime


class SpreadOptimizer:
    """Advanced spread optimization algorithms."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or get_config()
        self.optimization_cache: Dict[str, Dict] = {}
        self.performance_history: List[Dict] = []
        
    @measure_latency("spread_optimization")
    async def optimize_spreads(
        self,
        market_conditions: MarketConditions,
        prediction: PredictionContext,
        quote_params: QuoteParameters,
        mode: SpreadOptimizationMode = SpreadOptimizationMode.ADAPTIVE
    ) -> Dict[str, float]:
        """Optimize spreads based on market conditions and predictions."""
        try:
            if mode == SpreadOptimizationMode.AGGRESSIVE:
                return await self._aggressive_optimization(market_conditions, prediction, quote_params)
            elif mode == SpreadOptimizationMode.CONSERVATIVE:
                return await self._conservative_optimization(market_conditions, prediction, quote_params)
            elif mode == SpreadOptimizationMode.BALANCED:
                return await self._balanced_optimization(market_conditions, prediction, quote_params)
            else:  # ADAPTIVE
                return await self._adaptive_optimization(market_conditions, prediction, quote_params)
                
        except Exception as e:
            logger.error(f"Spread optimization failed: {e}")
            # Fallback to base spreads
            return {
                'base_spread': quote_params.base_spread_bps,
                'bid_adjustment': 0.0,
                'ask_adjustment': 0.0,
                'volatility_factor': 1.0,
                'competition_factor': 1.0,
                'prediction_factor': 1.0
            }
    
    async def _aggressive_optimization(
        self, 
        market_conditions: MarketConditions,
        prediction: PredictionContext,
        quote_params: QuoteParameters
    ) -> Dict[str, float]:
        """Aggressive spread optimization for high-frequency trading."""
        base_spread = quote_params.base_spread_bps * 0.7  # Tighter base spread
        
        # Competition factor - very responsive to market tightness
        competition_factor = 0.5 if market_conditions.competition_tightness > 0.8 else 0.8
        
        # Prediction factor - higher weight on ML signals
        prediction_factor = 1.0 - (prediction.confidence * 0.4)
        
        # Volatility factor - moderate adjustment
        volatility_factor = 1.0 + (market_conditions.volatility * quote_params.volatility_multiplier * 0.8)
        
        return {
            'base_spread': base_spread,
            'bid_adjustment': -prediction.signal_strength * 0.3,
            'ask_adjustment': prediction.signal_strength * 0.3,
            'volatility_factor': volatility_factor,
            'competition_factor': competition_factor,
            'prediction_factor': prediction_factor
        }
    
    async def _conservative_optimization(
        self,
        market_conditions: MarketConditions,
        prediction: PredictionContext,
        quote_params: QuoteParameters
    ) -> Dict[str, float]:
        """Conservative spread optimization for stable conditions."""
        base_spread = quote_params.base_spread_bps * 1.3  # Wider base spread
        
        # Competition factor - less responsive
        competition_factor = 0.9 if market_conditions.competition_tightness > 0.9 else 1.0
        
        # Prediction factor - lower weight on ML signals
        prediction_factor = 1.0 - (prediction.confidence * 0.2)
        
        # Volatility factor - higher adjustment for safety
        volatility_factor = 1.0 + (market_conditions.volatility * quote_params.volatility_multiplier * 1.5)
        
        return {
            'base_spread': base_spread,
            'bid_adjustment': -prediction.signal_strength * 0.1,
            'ask_adjustment': prediction.signal_strength * 0.1,
            'volatility_factor': volatility_factor,
            'competition_factor': competition_factor,
            'prediction_factor': prediction_factor
        }
    
    async def _balanced_optimization(
        self,
        market_conditions: MarketConditions,
        prediction: PredictionContext,
        quote_params: QuoteParameters
    ) -> Dict[str, float]:
        """Balanced spread optimization for general market making."""
        base_spread = quote_params.base_spread_bps  # Standard base spread
        
        # Competition factor - balanced response
        competition_factor = 0.8 if market_conditions.competition_tightness > 0.7 else 0.9
        
        # Prediction factor - moderate weight on ML signals
        prediction_factor = 1.0 - (prediction.confidence * 0.3)
        
        # Volatility factor - standard adjustment
        volatility_factor = 1.0 + (market_conditions.volatility * quote_params.volatility_multiplier)
        
        return {
            'base_spread': base_spread,
            'bid_adjustment': -prediction.signal_strength * 0.2,
            'ask_adjustment': prediction.signal_strength * 0.2,
            'volatility_factor': volatility_factor,
            'competition_factor': competition_factor,
            'prediction_factor': prediction_factor
        }
    
    async def _adaptive_optimization(
        self,
        market_conditions: MarketConditions,
        prediction: PredictionContext,
        quote_params: QuoteParameters
    ) -> Dict[str, float]:
        """Adaptive spread optimization based on real-time conditions."""
        # Dynamic base spread based on market stress
        stress_multiplier = 1.0 + market_conditions.market_stress_level
        base_spread = quote_params.base_spread_bps * stress_multiplier
        
        # Adaptive competition factor
        if market_conditions.competition_tightness > 0.9:
            competition_factor = 0.6  # Very aggressive in tight markets
        elif market_conditions.competition_tightness > 0.7:
            competition_factor = 0.8  # Moderately aggressive
        else:
            competition_factor = 1.0   # Standard in loose markets
        
        # Dynamic prediction factor based on uncertainty
        uncertainty_penalty = prediction.uncertainty_score * 0.5
        prediction_factor = 1.0 - (prediction.confidence * 0.3) + uncertainty_penalty
        
        # Adaptive volatility factor
        vol_threshold = 0.03  # 3% volatility threshold
        if market_conditions.volatility > vol_threshold:
            volatility_factor = 1.0 + (market_conditions.volatility * quote_params.volatility_multiplier * 2.0)
        else:
            volatility_factor = 1.0 + (market_conditions.volatility * quote_params.volatility_multiplier)
        
        # Liquidity adjustment
        liquidity_factor = 1.0
        if market_conditions.liquidity_score < quote_params.liquidity_threshold:
            liquidity_factor = 1.0 + (quote_params.liquidity_threshold - market_conditions.liquidity_score)
        
        return {
            'base_spread': base_spread,
            'bid_adjustment': -prediction.signal_strength * 0.25,
            'ask_adjustment': prediction.signal_strength * 0.25,
            'volatility_factor': volatility_factor,
            'competition_factor': competition_factor,
            'prediction_factor': prediction_factor,
            'liquidity_factor': liquidity_factor
        }


class QuoteValidator:
    """Comprehensive quote validation and sanity checks."""
    
    def __init__(self, validation_level: QuoteValidationLevel = QuoteValidationLevel.NORMAL):
        self.validation_level = validation_level
        self.config = get_config()
        
    async def validate_quote_set(
        self,
        quote_set: QuoteSet,
        market_conditions: MarketConditions,
        quote_params: QuoteParameters
    ) -> Tuple[bool, List[str]]:
        """Validate a complete quote set."""
        errors = []
        
        try:
            # Basic validation
            basic_errors = await self._validate_basic_constraints(quote_set, quote_params)
            errors.extend(basic_errors)
            
            # Price validation
            price_errors = await self._validate_prices(quote_set, market_conditions)
            errors.extend(price_errors)
            
            # Size validation
            size_errors = await self._validate_sizes(quote_set, quote_params)
            errors.extend(size_errors)
            
            # Risk validation
            risk_errors = await self._validate_risk_limits(quote_set, market_conditions)
            errors.extend(risk_errors)
            
            # Market structure validation
            if self.validation_level != QuoteValidationLevel.RELAXED:
                structure_errors = await self._validate_market_structure(quote_set, market_conditions)
                errors.extend(structure_errors)
            
            # Advanced validation for strict mode
            if self.validation_level == QuoteValidationLevel.STRICT:
                advanced_errors = await self._validate_advanced_constraints(quote_set, market_conditions)
                errors.extend(advanced_errors)
            
            return len(errors) == 0, errors
            
        except Exception as e:
            logger.error(f"Quote validation failed: {e}")
            return False, [f"Validation error: {e}"]
    
    async def _validate_basic_constraints(
        self, quote_set: QuoteSet, quote_params: QuoteParameters
    ) -> List[str]:
        """Validate basic quote constraints."""
        errors = []
        
        if not quote_set.quotes:
            errors.append("No quotes generated")
            return errors
        
        for quote in quote_set.quotes:
            # Price validation
            if quote.price <= 0:
                errors.append(f"Invalid price: {quote.price}")
            
            # Size validation
            if quote.size <= 0:
                errors.append(f"Invalid size: {quote.size}")
            
            if quote.size < quote_params.min_size:
                errors.append(f"Size below minimum: {quote.size} < {quote_params.min_size}")
            
            if quote.size > quote_params.max_size:
                errors.append(f"Size above maximum: {quote.size} > {quote_params.max_size}")
            
            # Spread validation
            if quote.spread_bps < quote_params.min_spread_bps:
                errors.append(f"Spread below minimum: {quote.spread_bps} < {quote_params.min_spread_bps}")
            
            if quote.spread_bps > quote_params.max_spread_bps:
                errors.append(f"Spread above maximum: {quote.spread_bps} > {quote_params.max_spread_bps}")
        
        return errors
    
    async def _validate_prices(
        self, quote_set: QuoteSet, market_conditions: MarketConditions
    ) -> List[str]:
        """Validate quote prices against market conditions."""
        errors = []
        
        # Get best quotes
        best_bid = None
        best_ask = None
        
        for quote in quote_set.quotes:
            if quote.side == 'buy' and (not best_bid or quote.price > best_bid.price):
                best_bid = quote
            elif quote.side == 'sell' and (not best_ask or quote.price < best_ask.price):
                best_ask = quote
        
        if best_bid and best_ask:
            # Check bid-ask spread
            if best_bid.price >= best_ask.price:
                errors.append(f"Invalid bid-ask spread: bid {best_bid.price} >= ask {best_ask.price}")
            
            # Check against market prices (fat finger protection)
            mid_price = market_conditions.mid_price
            price_deviation_threshold = float(mid_price) * 0.05  # 5% deviation threshold
            
            if abs(float(best_bid.price) - float(mid_price)) > price_deviation_threshold:
                errors.append(f"Bid price too far from mid: {best_bid.price} vs {mid_price}")
            
            if abs(float(best_ask.price) - float(mid_price)) > price_deviation_threshold:
                errors.append(f"Ask price too far from mid: {best_ask.price} vs {mid_price}")
        
        return errors
    
    async def _validate_sizes(
        self, quote_set: QuoteSet, quote_params: QuoteParameters
    ) -> List[str]:
        """Validate quote sizes."""
        errors = []
        
        total_bid_size = sum(q.size for q in quote_set.quotes if q.side == 'buy')
        total_ask_size = sum(q.size for q in quote_set.quotes if q.side == 'sell')
        
        # Check size balance
        max_total_size = quote_params.max_size * quote_params.max_levels
        if total_bid_size > max_total_size:
            errors.append(f"Total bid size too large: {total_bid_size} > {max_total_size}")
        
        if total_ask_size > max_total_size:
            errors.append(f"Total ask size too large: {total_ask_size} > {max_total_size}")
        
        return errors
    
    async def _validate_risk_limits(
        self, quote_set: QuoteSet, market_conditions: MarketConditions
    ) -> List[str]:
        """Validate risk limits."""
        errors = []
        
        # Check market stress conditions
        if market_conditions.market_stress_level > 0.8:
            errors.append("Market stress level too high for quoting")
        
        # Check volatility limits
        if market_conditions.volatility > 0.1:  # 10% volatility threshold
            errors.append(f"Volatility too high: {market_conditions.volatility}")
        
        # Check liquidity conditions
        if market_conditions.liquidity_score < 0.1:  # Very low liquidity
            errors.append(f"Liquidity too low: {market_conditions.liquidity_score}")
        
        return errors
    
    async def _validate_market_structure(
        self, quote_set: QuoteSet, market_conditions: MarketConditions
    ) -> List[str]:
        """Validate quotes against market microstructure."""
        errors = []
        
        # Check if we're improving the market
        if quote_set.effective_spread_bps >= market_conditions.spread_bps:
            errors.append(f"Not improving market spread: {quote_set.effective_spread_bps} >= {market_conditions.spread_bps}")
        
        # Check for adverse selection risk
        high_risk_quotes = [q for q in quote_set.quotes if q.adverse_selection_score > 0.8]
        if len(high_risk_quotes) > len(quote_set.quotes) * 0.3:  # >30% high risk
            errors.append("Too many quotes with high adverse selection risk")
        
        return errors
    
    async def _validate_advanced_constraints(
        self, quote_set: QuoteSet, market_conditions: MarketConditions
    ) -> List[str]:
        """Advanced validation for strict mode."""
        errors = []
        
        # Check profitability expectations
        unprofitable_quotes = [q for q in quote_set.quotes if q.profit_expectation <= 0]
        if len(unprofitable_quotes) > 0:
            errors.append(f"{len(unprofitable_quotes)} quotes have negative profit expectation")
        
        # Check fill probability distribution
        low_fill_prob_quotes = [q for q in quote_set.quotes if q.expected_fill_probability < 0.1]
        if len(low_fill_prob_quotes) > len(quote_set.quotes) * 0.5:
            errors.append("Too many quotes with low fill probability")
        
        return errors


class QuoteGenerator:
    """Advanced quote generator with ML integration and optimization."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or get_config()
        self.spread_optimizer = SpreadOptimizer(config)
        self.validator = QuoteValidator(QuoteValidationLevel.NORMAL)
        
        # Performance tracking
        self.quotes_generated = 0
        self.validation_failures = 0
        self.optimization_time_ms = 0.0
        
    @timeout_async(0.15)  # 150ms timeout for quote generation
    @measure_latency("quote_generation")
    async def generate_quotes(
        self,
        symbol: str,
        market_conditions: MarketConditions,
        prediction: PredictionContext,
        quote_params: QuoteParameters,
        optimization_mode: SpreadOptimizationMode = SpreadOptimizationMode.ADAPTIVE
    ) -> Optional[QuoteSet]:
        """Generate optimized quotes based on ML predictions and market conditions."""
        start_time = datetime.now()
        
        try:
            # Handle edge cases first
            if not await self._check_quote_generation_conditions(market_conditions, quote_params):
                logger.warning(f"Quote generation conditions not met for {symbol}")
                return None
            
            # Optimize spreads
            spread_optimization = await self.spread_optimizer.optimize_spreads(
                market_conditions, prediction, quote_params, optimization_mode
            )
            
            # Generate quotes
            quotes = await self._generate_quote_levels(
                symbol, market_conditions, prediction, quote_params, spread_optimization
            )
            
            if not quotes:
                logger.warning(f"No quotes generated for {symbol}")
                return None
            
            # Create quote set
            quote_set = await self._create_quote_set(
                symbol, quotes, optimization_mode, start_time
            )
            
            # Validate quotes
            is_valid, validation_errors = await self.validator.validate_quote_set(
                quote_set, market_conditions, quote_params
            )
            
            if not is_valid:
                logger.warning(f"Quote validation failed for {symbol}: {validation_errors}")
                self.validation_failures += 1
                return None
            
            self.quotes_generated += len(quotes)
            
            logger.debug(f"Generated {len(quotes)} valid quotes for {symbol} "
                        f"in {(datetime.now() - start_time).total_seconds() * 1000:.1f}ms")
            
            return quote_set
            
        except Exception as e:
            logger.error(f"Quote generation failed for {symbol}: {e}")
            return None
    
    async def _check_quote_generation_conditions(
        self, market_conditions: MarketConditions, quote_params: QuoteParameters
    ) -> bool:
        """Check if conditions are suitable for quote generation."""
        # Market stress check
        if market_conditions.market_stress_level > 0.9:
            return False
        
        # Volatility spike check
        if market_conditions.volatility > 0.15:  # 15% volatility threshold
            return False
        
        # Liquidity check
        if market_conditions.liquidity_score < quote_params.liquidity_threshold:
            return False
        
        # Spread check - don't quote in extremely wide markets
        if market_conditions.spread_bps > quote_params.max_spread_bps * 2:
            return False
        
        return True
    
    async def _generate_quote_levels(
        self,
        symbol: str,
        market_conditions: MarketConditions,
        prediction: PredictionContext,
        quote_params: QuoteParameters,
        spread_optimization: Dict[str, float]
    ) -> List[Quote]:
        """Generate individual quote levels."""
        quotes = []
        
        try:
            for level in range(quote_params.max_levels):
                # Calculate level-specific parameters
                level_spreads = await self._calculate_level_spreads(
                    level, spread_optimization, quote_params
                )
                
                # Calculate prices
                bid_price, ask_price = await self._calculate_level_prices(
                    market_conditions.mid_price, level_spreads, prediction
                )
                
                # Calculate sizes
                bid_size, ask_size = await self._calculate_level_sizes(
                    level, market_conditions, prediction, quote_params
                )
                
                # Calculate quote metadata
                bid_metadata = await self._calculate_quote_metadata(
                    'buy', bid_price, bid_size, level, market_conditions, prediction
                )
                ask_metadata = await self._calculate_quote_metadata(
                    'sell', ask_price, ask_size, level, market_conditions, prediction
                )
                
                # Create quotes
                if bid_size >= quote_params.min_size:
                    quotes.append(Quote(
                        symbol=symbol,
                        side='buy',
                        price=bid_price,
                        size=bid_size,
                        level=level,
                        spread_bps=level_spreads['bid_spread'],
                        confidence_score=bid_metadata['confidence'],
                        expected_fill_probability=bid_metadata['fill_probability'],
                        adverse_selection_score=bid_metadata['adverse_selection'],
                        profit_expectation=bid_metadata['profit_expectation'],
                        skew_adjustment=spread_optimization.get('bid_adjustment', 0.0),
                        volatility_adjustment=spread_optimization.get('volatility_factor', 1.0),
                        timestamp=datetime.now(),
                        quote_id=f"{symbol}_buy_{level}_{datetime.now().strftime('%H%M%S%f')}"
                    ))
                
                if ask_size >= quote_params.min_size:
                    quotes.append(Quote(
                        symbol=symbol,
                        side='sell',
                        price=ask_price,
                        size=ask_size,
                        level=level,
                        spread_bps=level_spreads['ask_spread'],
                        confidence_score=ask_metadata['confidence'],
                        expected_fill_probability=ask_metadata['fill_probability'],
                        adverse_selection_score=ask_metadata['adverse_selection'],
                        profit_expectation=ask_metadata['profit_expectation'],
                        skew_adjustment=spread_optimization.get('ask_adjustment', 0.0),
                        volatility_adjustment=spread_optimization.get('volatility_factor', 1.0),
                        timestamp=datetime.now(),
                        quote_id=f"{symbol}_sell_{level}_{datetime.now().strftime('%H%M%S%f')}"
                    ))
            
            return quotes
            
        except Exception as e:
            logger.error(f"Error generating quote levels: {e}")
            return []
    
    async def _calculate_level_spreads(
        self, level: int, spread_optimization: Dict[str, float], quote_params: QuoteParameters
    ) -> Dict[str, float]:
        """Calculate spreads for a specific level."""
        base_spread = spread_optimization['base_spread']
        level_multiplier = (level + 1)  # Level 0 = 1x, Level 1 = 2x, etc.
        
        # Apply all optimization factors
        adjusted_spread = base_spread * level_multiplier
        adjusted_spread *= spread_optimization.get('volatility_factor', 1.0)
        adjusted_spread *= spread_optimization.get('competition_factor', 1.0)
        adjusted_spread *= spread_optimization.get('prediction_factor', 1.0)
        adjusted_spread *= spread_optimization.get('liquidity_factor', 1.0)
        
        # Calculate bid/ask spreads with adjustments
        bid_spread = adjusted_spread + spread_optimization.get('bid_adjustment', 0.0)
        ask_spread = adjusted_spread + spread_optimization.get('ask_adjustment', 0.0)
        
        # Ensure within bounds
        bid_spread = max(quote_params.min_spread_bps, min(quote_params.max_spread_bps, bid_spread))
        ask_spread = max(quote_params.min_spread_bps, min(quote_params.max_spread_bps, ask_spread))
        
        return {
            'bid_spread': bid_spread,
            'ask_spread': ask_spread,
            'level_multiplier': level_multiplier
        }
    
    async def _calculate_level_prices(
        self, mid_price: Decimal, level_spreads: Dict[str, float], prediction: PredictionContext
    ) -> Tuple[Decimal, Decimal]:
        """Calculate bid and ask prices for a level."""
        # Base prices from spreads
        bid_price = mid_price * (1 - Decimal(str(level_spreads['bid_spread'])) / 10000)
        ask_price = mid_price * (1 + Decimal(str(level_spreads['ask_spread'])) / 10000)
        
        # Apply prediction skew (small adjustment for expected price movement)
        if prediction.confidence > 0.7:
            price_skew = Decimal(str(prediction.price_change_bps)) / 10000 * Decimal('0.1')  # 10% of predicted move
            bid_price += mid_price * price_skew
            ask_price += mid_price * price_skew
        
        # Round to appropriate precision
        bid_price = bid_price.quantize(Decimal('0.000001'), rounding=ROUND_HALF_UP)
        ask_price = ask_price.quantize(Decimal('0.000001'), rounding=ROUND_HALF_UP)
        
        return bid_price, ask_price
    
    async def _calculate_level_sizes(
        self,
        level: int,
        market_conditions: MarketConditions,
        prediction: PredictionContext,
        quote_params: QuoteParameters
    ) -> Tuple[Decimal, Decimal]:
        """Calculate bid and ask sizes for a level."""
        # Base size decreases with level
        level_factor = 1.0 / (level + 1)
        base_size = quote_params.base_size * Decimal(str(level_factor))
        
        # Prediction confidence adjustment
        confidence_factor = Decimal(str(0.5 + prediction.confidence * 0.5))  # 0.5 to 1.0 range
        adjusted_size = base_size * confidence_factor
        
        # Liquidity adjustment
        liquidity_factor = Decimal(str(min(1.0, market_conditions.liquidity_score + 0.5)))
        final_size = adjusted_size * liquidity_factor
        
        # Inventory skew (would come from position tracker in practice)
        inventory_skew = quote_params.inventory_skew_factor
        
        bid_size = final_size * Decimal(str(1.0 - inventory_skew * 0.2))
        ask_size = final_size * Decimal(str(1.0 + inventory_skew * 0.2))
        
        # Ensure within bounds
        bid_size = max(quote_params.min_size, min(quote_params.max_size, bid_size))
        ask_size = max(quote_params.min_size, min(quote_params.max_size, ask_size))
        
        # Round to appropriate precision
        bid_size = bid_size.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
        ask_size = ask_size.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
        
        return bid_size, ask_size
    
    async def _calculate_quote_metadata(
        self,
        side: str,
        price: Decimal,
        size: Decimal,
        level: int,
        market_conditions: MarketConditions,
        prediction: PredictionContext
    ) -> Dict[str, float]:
        """Calculate comprehensive quote metadata."""
        try:
            # Confidence score based on prediction and market conditions
            confidence = prediction.confidence * (1.0 - level * 0.1)  # Decreases with level
            confidence *= market_conditions.liquidity_score  # Adjusted by liquidity
            confidence = max(0.1, min(1.0, confidence))
            
            # Fill probability estimation
            distance_from_mid = abs(float(price) - float(market_conditions.mid_price)) / float(market_conditions.mid_price)
            fill_probability = max(0.05, 1.0 - (distance_from_mid * 10))  # Decreases with distance
            
            # Adverse selection score
            if side == 'buy':
                # Higher risk if we're bidding when prediction is bearish
                adverse_selection = 0.5
                if prediction.direction == 'bearish':
                    adverse_selection += prediction.signal_strength * 0.3
            else:  # sell
                # Higher risk if we're asking when prediction is bullish
                adverse_selection = 0.5
                if prediction.direction == 'bullish':
                    adverse_selection += prediction.signal_strength * 0.3
            
            adverse_selection = max(0.0, min(1.0, adverse_selection))
            
            # Profit expectation (simplified)
            spread_contribution = market_conditions.spread_bps / 10000  # Convert bps to decimal
            volatility_risk = market_conditions.volatility * 0.5
            profit_expectation = spread_contribution - volatility_risk - (adverse_selection * 0.1)
            
            return {
                'confidence': confidence,
                'fill_probability': fill_probability,
                'adverse_selection': adverse_selection,
                'profit_expectation': profit_expectation
            }
            
        except Exception as e:
            logger.error(f"Error calculating quote metadata: {e}")
            return {
                'confidence': 0.5,
                'fill_probability': 0.3,
                'adverse_selection': 0.5,
                'profit_expectation': 0.01
            }
    
    async def _create_quote_set(
        self,
        symbol: str,
        quotes: List[Quote],
        optimization_mode: SpreadOptimizationMode,
        start_time: datetime
    ) -> QuoteSet:
        """Create a comprehensive quote set from individual quotes."""
        try:
            # Separate bid and ask quotes
            bid_quotes = [q for q in quotes if q.side == 'buy']
            ask_quotes = [q for q in quotes if q.side == 'sell']
            
            # Calculate totals
            total_bid_size = sum(q.size for q in bid_quotes)
            total_ask_size = sum(q.size for q in ask_quotes)
            
            # Calculate weighted prices
            if bid_quotes:
                weighted_bid_price = sum(q.price * q.size for q in bid_quotes) / total_bid_size
            else:
                weighted_bid_price = Decimal('0')
            
            if ask_quotes:
                weighted_ask_price = sum(q.price * q.size for q in ask_quotes) / total_ask_size
            else:
                weighted_ask_price = Decimal('0')
            
            # Calculate effective spread
            if bid_quotes and ask_quotes:
                best_bid = max(bid_quotes, key=lambda q: q.price)
                best_ask = min(ask_quotes, key=lambda q: q.price)
                mid_price = (best_bid.price + best_ask.price) / 2
                effective_spread_bps = float((best_ask.price - best_bid.price) / mid_price * 10000)
            else:
                effective_spread_bps = 0.0
            
            # Calculate expected profitability
            expected_profitability = sum(q.profit_expectation * float(q.size) for q in quotes) / float(total_bid_size + total_ask_size) if quotes else 0.0
            
            # Calculate risk score
            risk_score = sum(q.adverse_selection_score for q in quotes) / len(quotes) if quotes else 0.0
            
            # Calculate generation latency
            generation_latency_ms = (datetime.now() - start_time).total_seconds() * 1000
            
            return QuoteSet(
                symbol=symbol,
                quotes=quotes,
                bid_quotes=bid_quotes,
                ask_quotes=ask_quotes,
                total_bid_size=total_bid_size,
                total_ask_size=total_ask_size,
                weighted_bid_price=weighted_bid_price,
                weighted_ask_price=weighted_ask_price,
                effective_spread_bps=effective_spread_bps,
                expected_profitability=expected_profitability,
                risk_score=risk_score,
                optimization_mode=optimization_mode,
                generation_latency_ms=generation_latency_ms,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error creating quote set: {e}")
            # Return minimal quote set
            return QuoteSet(
                symbol=symbol,
                quotes=quotes,
                bid_quotes=[q for q in quotes if q.side == 'buy'],
                ask_quotes=[q for q in quotes if q.side == 'sell'],
                total_bid_size=sum(q.size for q in quotes if q.side == 'buy'),
                total_ask_size=sum(q.size for q in quotes if q.side == 'sell'),
                weighted_bid_price=Decimal('0'),
                weighted_ask_price=Decimal('0'),
                effective_spread_bps=0.0,
                expected_profitability=0.0,
                risk_score=0.5,
                optimization_mode=optimization_mode,
                generation_latency_ms=0.0,
                timestamp=datetime.now()
            )
    
    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get quote generator performance statistics."""
        return {
            'quotes_generated': self.quotes_generated,
            'validation_failures': self.validation_failures,
            'validation_success_rate': (self.quotes_generated - self.validation_failures) / max(self.quotes_generated, 1),
            'avg_optimization_time_ms': self.optimization_time_ms / max(self.quotes_generated, 1),
            'spread_optimizer_stats': getattr(self.spread_optimizer, 'performance_history', []),
            'validator_level': self.validator.validation_level.value
        }
    
    async def update_optimization_mode(self, mode: SpreadOptimizationMode) -> None:
        """Update the default optimization mode."""
        self.default_optimization_mode = mode
        logger.info(f"Updated optimization mode to {mode.value}")
    
    async def update_validation_level(self, level: QuoteValidationLevel) -> None:
        """Update the validation strictness level."""
        self.validator = QuoteValidator(level)
        logger.info(f"Updated validation level to {level.value}")


# Utility functions for quote generation
async def create_quote_parameters_from_config(config: Dict, symbol: str) -> QuoteParameters:
    """Create quote parameters from configuration."""
    return QuoteParameters(
        symbol=symbol,
        base_spread_bps=config.get("trading.base_spread_bps", 10.0),
        min_spread_bps=config.get("trading.min_spread_bps", 3.0),
        max_spread_bps=config.get("trading.max_spread_bps", 100.0),
        base_size=Decimal(str(config.get("trading.base_quote_size", 100.0))),
        min_size=Decimal(str(config.get("trading.min_quote_size", 10.0))),
        max_size=Decimal(str(config.get("trading.max_quote_size", 1000.0))),
        max_levels=config.get("trading.max_quote_levels", 3),
        prediction_weight=config.get("trading.prediction_weight", 0.7),
        inventory_skew_factor=config.get("trading.inventory_skew_multiplier", 2.0),
        volatility_multiplier=config.get("trading.volatility_multiplier", 1.5),
        liquidity_threshold=config.get("trading.liquidity_threshold", 0.3)
    )


async def create_market_conditions_from_data(market_data: Dict) -> MarketConditions:
    """Create market conditions from market data."""
    order_book = market_data.get("order_book", {})
    bids = order_book.get("bids", [])
    asks = order_book.get("asks", [])
    
    if not bids or not asks:
        raise ValueError("Invalid market data: no bids or asks")
    
    best_bid = Decimal(str(bids[0][0]))
    best_ask = Decimal(str(asks[0][0]))
    mid_price = (best_bid + best_ask) / 2
    
    return MarketConditions(
        mid_price=mid_price,
        best_bid=best_bid,
        best_ask=best_ask,
        spread_bps=float((best_ask - best_bid) / mid_price * 10000),
        volatility=market_data.get("volatility", 0.02),
        liquidity_score=market_data.get("liquidity_score", 0.5),
        order_book_imbalance=market_data.get("imbalance", 0.0),
        recent_volume=market_data.get("volume_24h", 0.0),
        competition_tightness=market_data.get("competition_tightness", 0.5),
        market_stress_level=market_data.get("market_stress_level", 0.0)
    )


async def create_prediction_context_from_signal(prediction: Dict) -> PredictionContext:
    """Create prediction context from ML prediction signal."""
    return PredictionContext(
        direction=prediction.get("direction", "neutral"),
        confidence=prediction.get("confidence", 0.0),
        price_change_bps=prediction.get("price_change_bps", 0.0),
        signal_strength=prediction.get("signal_strength", 0.0),
        uncertainty_score=prediction.get("uncertainty_score", 1.0),
        time_horizon_ms=prediction.get("time_horizon_ms", 200),
        model_version=prediction.get("model_version", "unknown"),
        ensemble_agreement=prediction.get("ensemble_agreement", 0.5)
    )
"""
FlashMM ML-Driven Quoting Strategy

Advanced market-making strategy that generates optimal quotes based on ML predictions,
market microstructure analysis, and sophisticated risk management.
"""

import asyncio
import math
from typing import Dict, List, Optional, Tuple, Any
from decimal import Decimal, ROUND_HALF_UP
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from flashmm.config.settings import get_config
from flashmm.ml.inference.inference_engine import InferenceEngine
from flashmm.trading.execution.order_router import OrderRouter
from flashmm.trading.risk.risk_manager import RiskManager
from flashmm.trading.risk.position_tracker import PositionTracker
from flashmm.data.market_data_service import MarketDataService
from flashmm.utils.logging import get_logger, TradingEventLogger
from flashmm.utils.exceptions import TradingError, QuotingError
from flashmm.utils.decorators import require_trading_enabled, measure_latency

logger = get_logger(__name__)
trading_logger = TradingEventLogger()


class QuoteConfidence(Enum):
    """Quote confidence levels based on prediction quality."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class QuoteLevel:
    """Individual quote level with pricing and sizing."""
    level: int
    side: str  # 'buy' or 'sell'
    price: Decimal
    size: Decimal
    spread_bps: float
    confidence: QuoteConfidence
    skew_factor: float = 0.0  # Inventory skew adjustment
    volatility_factor: float = 1.0  # Volatility adjustment


@dataclass
class MarketState:
    """Current market state for quote generation."""
    symbol: str
    mid_price: Decimal
    best_bid: Decimal
    best_ask: Decimal
    bid_size: Decimal
    ask_size: Decimal
    spread_bps: float
    volatility: float
    liquidity_score: float
    order_book_imbalance: float
    recent_volume: float
    timestamp: datetime


@dataclass
class PredictionSignal:
    """Processed prediction signal for quote generation."""
    direction: str  # 'bullish', 'bearish', 'neutral'
    confidence: float
    price_change_bps: float
    magnitude: str
    signal_strength: float
    uncertainty_score: float
    timestamp: datetime


class QuotingStrategy:
    """Advanced ML-driven market-making strategy for FlashMM."""
    
    def __init__(self):
        self.config = get_config()
        self.running = False
        
        # Core components
        self.inference_engine: Optional[InferenceEngine] = None
        self.order_router: Optional[OrderRouter] = None
        self.risk_manager: Optional[RiskManager] = None
        self.position_tracker: Optional[PositionTracker] = None
        self.market_data_service: Optional[MarketDataService] = None
        
        # Strategy parameters
        self.quote_frequency = self.config.get("trading.quote_frequency_hz", 5.0)
        self.max_quote_levels = self.config.get("trading.max_quote_levels", 3)
        self.base_spread_bps = self.config.get("trading.base_spread_bps", 10.0)
        self.min_spread_bps = self.config.get("trading.min_spread_bps", 3.0)
        self.max_spread_bps = self.config.get("trading.max_spread_bps", 100.0)
        
        # ML-driven parameters
        self.prediction_weight = self.config.get("trading.prediction_weight", 0.7)
        self.min_prediction_confidence = self.config.get("trading.min_prediction_confidence", 0.6)
        self.volatility_multiplier = self.config.get("trading.volatility_multiplier", 1.5)
        self.liquidity_threshold = self.config.get("trading.liquidity_threshold", 0.3)
        
        # Inventory management
        self.max_inventory_ratio = self.config.get("trading.max_inventory_ratio", 0.02)  # 2%
        self.inventory_skew_multiplier = self.config.get("trading.inventory_skew_multiplier", 2.0)
        
        # Quote sizing
        self.base_quote_size = Decimal(str(self.config.get("trading.base_quote_size", 100.0)))
        self.min_quote_size = Decimal(str(self.config.get("trading.min_quote_size", 10.0)))
        self.max_quote_size = Decimal(str(self.config.get("trading.max_quote_size", 1000.0)))
        
        # Current state
        self.current_quotes: Dict[str, List[QuoteLevel]] = {}
        self.last_quote_time = datetime.now()
        self.last_prediction: Optional[PredictionSignal] = None
        self.market_states: Dict[str, MarketState] = {}
        
        # Performance tracking
        self.quotes_generated = 0
        self.quotes_filled = 0
        self.total_spread_improvement = 0.0
        self.inventory_tracking: Dict[str, float] = {}
    
    async def initialize(self) -> None:
        """Initialize the strategy and all dependencies."""
        logger.info("Initializing ML-driven QuotingStrategy...")
        
        try:
            # Initialize ML inference engine
            self.inference_engine = InferenceEngine()
            await self.inference_engine.initialize()
            
            # Initialize order router
            self.order_router = OrderRouter()
            await self.order_router.initialize()
            
            # Initialize risk manager
            self.risk_manager = RiskManager()
            await self.risk_manager.initialize()
            
            # Initialize position tracker
            self.position_tracker = PositionTracker()
            await self.position_tracker.initialize()
            
            # Initialize market data service
            self.market_data_service = MarketDataService()
            await self.market_data_service.initialize()
            
            # Initialize strategy-specific state
            await self._initialize_strategy_state()
            
            logger.info("ML-driven QuotingStrategy initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize QuotingStrategy: {e}")
            raise TradingError(f"Strategy initialization failed: {e}")
    
    async def _initialize_strategy_state(self) -> None:
        """Initialize strategy-specific state and parameters."""
        # Load symbols from config
        symbols = self.config.get("trading.symbols", ["SEI/USDC"])
        
        # Initialize state for each symbol
        for symbol in symbols:
            self.current_quotes[symbol] = []
            self.market_states[symbol] = None
            self.inventory_tracking[symbol] = 0.0
        
        # Start ML inference engine if not running
        if self.inference_engine:
            await self.inference_engine.start_prediction_service()
    
    async def start(self) -> None:
        """Start the quoting strategy."""
        if not await self._validate_preconditions():
            raise TradingError("Strategy preconditions not met")
        
        logger.info("Starting QuotingStrategy...")
        self.running = True
        
        # Start main strategy loop
        asyncio.create_task(self._strategy_loop())
    
    async def stop(self) -> None:
        """Stop the strategy gracefully."""
        logger.info("Stopping QuotingStrategy...")
        self.running = False
        
        # Cancel all open orders
        if self.order_router:
            await self.order_router.cancel_all_orders()
        
        logger.info("QuotingStrategy stopped")
    
    async def _validate_preconditions(self) -> bool:
        """Validate that all preconditions are met for trading."""
        if not self.config.get("trading.enable_trading", False):
            logger.warning("Trading is disabled in configuration")
            return False
        
        if not self.inference_engine:
            logger.error("ML inference engine not initialized")
            return False
        
        if not self.order_router:
            logger.error("Order router not initialized")
            return False
        
        if not self.risk_manager:
            logger.error("Risk manager not initialized")
            return False
        
        return True
    
    async def _strategy_loop(self) -> None:
        """Main strategy execution loop."""
        while self.running:
            try:
                await self._execute_strategy_cycle()
                
                # Wait for next cycle
                sleep_time = 1.0 / self.quote_frequency
                await asyncio.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Error in strategy loop: {e}")
                await asyncio.sleep(1.0)  # Brief pause on error
    
    @require_trading_enabled
    @measure_latency("strategy_cycle")
    async def _execute_strategy_cycle(self) -> None:
        """Execute one complete strategy cycle."""
        symbols = self.config.get("trading.symbols", ["SEI/USDC"])
        
        for symbol in symbols:
            try:
                await self._process_symbol(symbol)
            except Exception as e:
                logger.error(f"Error processing symbol {symbol}: {e}")
    
    async def _process_symbol(self, symbol: str) -> None:
        """Process a single trading symbol."""
        # Get market data
        market_data = await self._get_market_data(symbol)
        if not market_data:
            return
        
        # Get ML prediction
        prediction = await self._get_prediction(symbol, market_data)
        if not prediction:
            return
        
        # Check risk limits
        if not await self._check_risk_limits(symbol):
            return
        
        # Generate quotes
        quotes = await self._generate_quotes(symbol, market_data, prediction)
        
        # Submit quotes
        if quotes:
            await self._submit_quotes(symbol, quotes)
    
    async def _get_market_data(self, symbol: str) -> Optional[Dict]:
        """Get comprehensive market data for symbol."""
        if not self.market_data_service:
            return None
        
        try:
            # Get order book data from market data service
            market_data = await self.market_data_service.get_market_snapshot(symbol)
            
            if not market_data:
                logger.warning(f"No market data available for {symbol}")
                return None
            
            # Enhance with additional analysis
            enhanced_data = await self._enhance_market_data(market_data, symbol)
            return enhanced_data
            
        except Exception as e:
            logger.error(f"Failed to get market data for {symbol}: {e}")
            return None
    
    async def _enhance_market_data(self, market_data: Dict, symbol: str) -> Dict:
        """Enhance market data with additional analysis."""
        try:
            # Extract order book
            order_book = market_data.get("order_book", {})
            bids = order_book.get("bids", [])
            asks = order_book.get("asks", [])
            
            if not bids or not asks:
                return market_data
            
            best_bid = Decimal(str(bids[0][0]))
            best_ask = Decimal(str(asks[0][0]))
            mid_price = (best_bid + best_ask) / Decimal('2')
            
            # Calculate market microstructure metrics
            spread_bps = float((best_ask - best_bid) / mid_price * 10000)
            
            # Order book imbalance
            bid_size = sum(Decimal(str(level[1])) for level in bids[:5])
            ask_size = sum(Decimal(str(level[1])) for level in asks[:5])
            total_size = bid_size + ask_size
            imbalance = float((bid_size - ask_size) / total_size) if total_size > 0 else 0.0
            
            # Liquidity score based on depth
            liquidity_score = min(float(total_size) / 1000.0, 1.0)  # Normalize to 0-1
            
            # Volatility estimate from recent prices
            volatility = await self._estimate_volatility(symbol)
            
            # Create market state
            market_state = MarketState(
                symbol=symbol,
                mid_price=mid_price,
                best_bid=best_bid,
                best_ask=best_ask,
                bid_size=bid_size,
                ask_size=ask_size,
                spread_bps=spread_bps,
                volatility=volatility,
                liquidity_score=liquidity_score,
                order_book_imbalance=imbalance,
                recent_volume=market_data.get("volume_24h", 0.0),
                timestamp=datetime.now()
            )
            
            # Store market state
            self.market_states[symbol] = market_state
            
            # Return enhanced data
            return {
                **market_data,
                "market_state": market_state,
                "mid_price": float(mid_price),
                "spread_bps": spread_bps,
                "imbalance": imbalance,
                "liquidity_score": liquidity_score,
                "volatility": volatility
            }
            
        except Exception as e:
            logger.error(f"Failed to enhance market data: {e}")
            return market_data
    
    async def _estimate_volatility(self, symbol: str) -> float:
        """Estimate volatility from recent price changes."""
        try:
            # Simple volatility estimate - in production would use historical data
            market_state = self.market_states.get(symbol)
            if not market_state:
                return 0.02  # Default 2% volatility
            
            # Use spread as volatility proxy for now
            return min(market_state.spread_bps / 10000.0 * 2, 0.1)  # Cap at 10%
            
        except Exception:
            return 0.02  # Default fallback
    
    async def _get_prediction(self, symbol: str, market_data: Dict) -> Optional[PredictionSignal]:
        """Get and process ML prediction for the symbol."""
        if not self.inference_engine:
            return None
        
        try:
            # Get prediction from ML engine
            prediction_dict = await self.inference_engine.predict(market_data)
            
            if not prediction_dict:
                return None
            
            # Convert to PredictionSignal
            prediction_signal = PredictionSignal(
                direction=prediction_dict.get("direction", "neutral"),
                confidence=prediction_dict.get("confidence", 0.0),
                price_change_bps=prediction_dict.get("price_change_bps", 0.0),
                magnitude=prediction_dict.get("magnitude", "low"),
                signal_strength=prediction_dict.get("signal_strength", 0.0),
                uncertainty_score=prediction_dict.get("uncertainty_score", 1.0),
                timestamp=datetime.now()
            )
            
            # Store latest prediction
            self.last_prediction = prediction_signal
            
            # Validate prediction confidence
            if prediction_signal.confidence < self.min_prediction_confidence:
                logger.debug(f"Prediction confidence {prediction_signal.confidence:.3f} below threshold")
                return None
            
            return prediction_signal
            
        except Exception as e:
            logger.error(f"Prediction failed for {symbol}: {e}")
            return None
    
    async def _check_risk_limits(self, symbol: str) -> bool:
        """Check if trading is allowed based on risk limits."""
        if not self.risk_manager:
            return False
        
        try:
            # Check risk manager
            risk_allowed = await self.risk_manager.check_trading_allowed(symbol)
            if not risk_allowed:
                return False
            
            # Check inventory limits
            inventory_allowed = await self._check_inventory_limits(symbol)
            if not inventory_allowed:
                return False
            
            # Check market conditions
            market_conditions_ok = await self._check_market_conditions(symbol)
            if not market_conditions_ok:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Risk check failed for {symbol}: {e}")
            return False
    
    async def _check_inventory_limits(self, symbol: str) -> bool:
        """Check inventory limits for symbol."""
        if not self.position_tracker:
            return True  # Allow if no position tracker
        
        try:
            position = await self.position_tracker.get_position(symbol)
            inventory_value = abs(position.get("value_usdc", 0.0))
            
            # Get notional traded limit
            max_inventory = self.config.get("trading.max_inventory_usdc", 2000.0)
            
            if inventory_value > max_inventory:
                logger.warning(f"Inventory limit exceeded for {symbol}: {inventory_value} > {max_inventory}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Inventory check failed for {symbol}: {e}")
            return False
    
    async def _check_market_conditions(self, symbol: str) -> bool:
        """Check if market conditions are suitable for market making."""
        market_state = self.market_states.get(symbol)
        if not market_state:
            return False
        
        # Check if spread is too wide (market stress)
        if market_state.spread_bps > self.max_spread_bps:
            logger.warning(f"Market spread too wide for {symbol}: {market_state.spread_bps:.2f} bps")
            return False
        
        # Check liquidity
        if market_state.liquidity_score < self.liquidity_threshold:
            logger.warning(f"Insufficient liquidity for {symbol}: {market_state.liquidity_score:.3f}")
            return False
        
        # Check volatility
        if market_state.volatility > 0.05:  # 5% volatility threshold
            logger.warning(f"High volatility for {symbol}: {market_state.volatility:.3f}")
            return False
        
        return True
    
    @measure_latency("quote_generation")
    async def _generate_quotes(
        self,
        symbol: str,
        market_data: Dict,
        prediction: PredictionSignal
    ) -> List[QuoteLevel]:
        """Generate sophisticated ML-driven quotes."""
        quotes = []
        
        try:
            market_state = market_data.get("market_state")
            if not market_state:
                return quotes
            
            # Calculate base spread adjustments
            spread_adjustments = await self._calculate_spread_adjustments(
                symbol, market_state, prediction
            )
            
            # Calculate inventory skew
            inventory_skew = await self._calculate_inventory_skew(symbol, market_state)
            
            # Generate quote levels
            for level in range(self.max_quote_levels):
                # Calculate quote confidence
                quote_confidence = self._determine_quote_confidence(prediction, level)
                
                # Calculate spreads for this level
                bid_spread, ask_spread = self._calculate_level_spreads(
                    level, spread_adjustments, inventory_skew, prediction
                )
                
                # Calculate prices
                bid_price = market_state.mid_price * (1 - Decimal(str(bid_spread)) / 10000)
                ask_price = market_state.mid_price * (1 + Decimal(str(ask_spread)) / 10000)
                
                # Calculate sizes
                bid_size, ask_size = await self._calculate_quote_sizes(
                    symbol, level, market_state, prediction, quote_confidence
                )
                
                # Create quote levels
                if bid_size > self.min_quote_size:
                    quotes.append(QuoteLevel(
                        level=level,
                        side="buy",
                        price=bid_price.quantize(Decimal('0.000001'), rounding=ROUND_HALF_UP),
                        size=bid_size.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP),
                        spread_bps=bid_spread,
                        confidence=quote_confidence,
                        skew_factor=inventory_skew,
                        volatility_factor=spread_adjustments.get("volatility_factor", 1.0)
                    ))
                
                if ask_size > self.min_quote_size:
                    quotes.append(QuoteLevel(
                        level=level,
                        side="sell",
                        price=ask_price.quantize(Decimal('0.000001'), rounding=ROUND_HALF_UP),
                        size=ask_size.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP),
                        spread_bps=ask_spread,
                        confidence=quote_confidence,
                        skew_factor=inventory_skew,
                        volatility_factor=spread_adjustments.get("volatility_factor", 1.0)
                    ))
            
            # Update performance tracking
            self.quotes_generated += len(quotes)
            
            # Log quote generation
            if quotes:
                logger.debug(f"Generated {len(quotes)} quotes for {symbol} "
                           f"(prediction: {prediction.direction}, confidence: {prediction.confidence:.3f})")
            
            return quotes
            
        except Exception as e:
            logger.error(f"Error generating quotes for {symbol}: {e}")
            return []
    
    async def _calculate_spread_adjustments(
        self, symbol: str, market_state: MarketState, prediction: PredictionSignal
    ) -> Dict[str, float]:
        """Calculate spread adjustments based on prediction and market conditions."""
        adjustments = {
            "base_spread": self.base_spread_bps,
            "prediction_factor": 1.0,
            "volatility_factor": 1.0,
            "liquidity_factor": 1.0,
            "competition_factor": 1.0
        }
        
        try:
            # Prediction-based adjustment
            if prediction.confidence > 0.8:
                # High confidence predictions allow tighter spreads
                adjustments["prediction_factor"] = 0.7
            elif prediction.confidence < 0.6:
                # Low confidence predictions require wider spreads
                adjustments["prediction_factor"] = 1.3
            
            # Volatility adjustment
            if market_state.volatility > 0.03:  # 3% volatility
                adjustments["volatility_factor"] = 1.0 + (market_state.volatility * self.volatility_multiplier)
            
            # Liquidity adjustment
            if market_state.liquidity_score < 0.5:
                adjustments["liquidity_factor"] = 1.0 + (0.5 - market_state.liquidity_score)
            
            # Competition adjustment (based on current spread)
            if market_state.spread_bps < self.base_spread_bps:
                # Market is tight, we can be more aggressive
                adjustments["competition_factor"] = 0.8
            
            return adjustments
            
        except Exception as e:
            logger.error(f"Error calculating spread adjustments: {e}")
            return adjustments
    
    async def _calculate_inventory_skew(self, symbol: str, market_state: MarketState) -> float:
        """Calculate inventory skew factor to manage position risk."""
        if not self.position_tracker:
            return 0.0
        
        try:
            position = await self.position_tracker.get_position(symbol)
            inventory_value = position.get("value_usdc", 0.0)
            
            # Calculate skew factor based on inventory deviation
            max_inventory = self.config.get("trading.max_inventory_usdc", 2000.0)
            inventory_ratio = inventory_value / max_inventory
            
            # Skew factor ranges from -1 to 1
            skew_factor = inventory_ratio * self.inventory_skew_multiplier
            
            # Clamp to reasonable limits
            return max(-1.0, min(1.0, skew_factor))
            
        except Exception as e:
            logger.error(f"Error calculating inventory skew: {e}")
            return 0.0
    
    def _determine_quote_confidence(self, prediction: PredictionSignal, level: int) -> QuoteConfidence:
        """Determine quote confidence based on prediction and level."""
        try:
            # Base confidence decreases with level distance
            base_confidence = prediction.confidence * (1.0 - level * 0.1)
            
            if base_confidence > 0.9:
                return QuoteConfidence.VERY_HIGH
            elif base_confidence > 0.75:
                return QuoteConfidence.HIGH
            elif base_confidence > 0.6:
                return QuoteConfidence.MEDIUM
            else:
                return QuoteConfidence.LOW
                
        except Exception:
            return QuoteConfidence.LOW
    
    def _calculate_level_spreads(
        self, 
        level: int, 
        spread_adjustments: Dict[str, float], 
        inventory_skew: float,
        prediction: PredictionSignal
    ) -> Tuple[float, float]:
        """Calculate bid and ask spreads for a specific level."""
        try:
            # Base spread for this level
            base_spread = spread_adjustments["base_spread"] * (level + 1)
            
            # Apply all adjustment factors
            adjusted_spread = base_spread
            for factor_name, factor_value in spread_adjustments.items():
                if factor_name != "base_spread":
                    adjusted_spread *= factor_value
            
            # Apply prediction skew
            prediction_skew = 0.0
            if prediction.direction == "bullish":
                prediction_skew = -prediction.signal_strength * 0.2  # Tighter ask
            elif prediction.direction == "bearish":
                prediction_skew = prediction.signal_strength * 0.2   # Tighter bid
            
            # Calculate final spreads
            bid_spread = adjusted_spread - prediction_skew + (inventory_skew * 0.5)
            ask_spread = adjusted_spread + prediction_skew - (inventory_skew * 0.5)
            
            # Ensure minimum spreads
            bid_spread = max(self.min_spread_bps, bid_spread)
            ask_spread = max(self.min_spread_bps, ask_spread)
            
            # Ensure maximum spreads
            bid_spread = min(self.max_spread_bps, bid_spread)
            ask_spread = min(self.max_spread_bps, ask_spread)
            
            return bid_spread, ask_spread
            
        except Exception as e:
            logger.error(f"Error calculating level spreads: {e}")
            base = self.base_spread_bps * (level + 1)
            return base, base
    
    async def _calculate_quote_sizes(
        self,
        symbol: str,
        level: int,
        market_state: MarketState,
        prediction: PredictionSignal,
        confidence: QuoteConfidence
    ) -> Tuple[Decimal, Decimal]:
        """Calculate bid and ask sizes for quotes."""
        try:
            # Base size decreases with level
            level_multiplier = 1.0 / (level + 1)
            base_size = self.base_quote_size * Decimal(str(level_multiplier))
            
            # Confidence adjustment
            confidence_multipliers = {
                QuoteConfidence.VERY_HIGH: 1.2,
                QuoteConfidence.HIGH: 1.0,
                QuoteConfidence.MEDIUM: 0.8,
                QuoteConfidence.LOW: 0.6
            }
            
            confidence_multiplier = Decimal(str(confidence_multipliers.get(confidence, 0.8)))
            adjusted_size = base_size * confidence_multiplier
            
            # Liquidity adjustment
            liquidity_multiplier = Decimal(str(min(1.0, market_state.liquidity_score + 0.5)))
            final_size = adjusted_size * liquidity_multiplier
            
            # Apply inventory skew to sizes
            inventory_skew = await self._calculate_inventory_skew(symbol, market_state)
            
            # Skew sizes based on inventory
            bid_size = final_size * Decimal(str(1.0 - inventory_skew * 0.3))
            ask_size = final_size * Decimal(str(1.0 + inventory_skew * 0.3))
            
            # Ensure minimum and maximum sizes
            bid_size = max(self.min_quote_size, min(self.max_quote_size, bid_size))
            ask_size = max(self.min_quote_size, min(self.max_quote_size, ask_size))
            
            return bid_size, ask_size
            
        except Exception as e:
            logger.error(f"Error calculating quote sizes: {e}")
            default_size = self.base_quote_size / Decimal(str(level + 1))
            return default_size, default_size
    
    async def _submit_quotes(self, symbol: str, quotes: List[QuoteLevel]) -> None:
        """Submit quotes to the market."""
        if not self.order_router or not quotes:
            return
        
        try:
            # Cancel existing quotes for this symbol
            await self._cancel_existing_quotes(symbol)
            
            successful_quotes = []
            
            # Submit new quotes
            for quote in quotes:
                try:
                    order_result = await self.order_router.place_order(
                        symbol=symbol,
                        side=quote.side,
                        price=float(quote.price),
                        size=float(quote.size),
                        order_type="limit",
                    )
                    
                    if order_result:
                        successful_quotes.append(quote)
                        
                        await trading_logger.log_order_event(
                            "quote_submitted",
                            order_result.get("order_id", ""),
                            symbol,
                            quote.side,
                            float(quote.price),
                            float(quote.size),
                            level=quote.level,
                            spread_bps=quote.spread_bps,
                            confidence=quote.confidence.value
                        )
                        
                except Exception as e:
                    logger.error(f"Failed to submit quote {quote.side} level {quote.level}: {e}")
            
            # Update current quotes
            self.current_quotes[symbol] = successful_quotes
            self.last_quote_time = datetime.now()
            
            # Calculate spread improvement
            if successful_quotes:
                await self._track_spread_improvement(symbol, successful_quotes)
            
            logger.debug(f"Successfully submitted {len(successful_quotes)}/{len(quotes)} quotes for {symbol}")
            
        except Exception as e:
            logger.error(f"Error submitting quotes for {symbol}: {e}")
    
    async def _track_spread_improvement(self, symbol: str, quotes: List[QuoteLevel]) -> None:
        """Track spread improvement metrics."""
        try:
            market_state = self.market_states.get(symbol)
            if not market_state:
                return
            
            # Find best quotes
            best_bid_quote = None
            best_ask_quote = None
            
            for quote in quotes:
                if quote.side == "buy" and (not best_bid_quote or quote.price > best_bid_quote.price):
                    best_bid_quote = quote
                elif quote.side == "sell" and (not best_ask_quote or quote.price < best_ask_quote.price):
                    best_ask_quote = quote
            
            if best_bid_quote and best_ask_quote:
                # Calculate our spread
                our_spread = float(best_ask_quote.price - best_bid_quote.price)
                our_spread_bps = (our_spread / float(market_state.mid_price)) * 10000
                
                # Calculate improvement vs market spread
                improvement_bps = market_state.spread_bps - our_spread_bps
                self.total_spread_improvement += improvement_bps
                
                logger.debug(f"Spread improvement for {symbol}: {improvement_bps:.2f} bps "
                           f"(market: {market_state.spread_bps:.2f}, ours: {our_spread_bps:.2f})")
                
        except Exception as e:
            logger.error(f"Error tracking spread improvement: {e}")
    
    async def _cancel_existing_quotes(self, symbol: str) -> None:
        """Cancel existing quotes for a symbol."""
        if not self.order_router:
            return
        
        try:
            await self.order_router.cancel_orders_for_symbol(symbol)
        except Exception as e:
            logger.error(f"Error canceling quotes for {symbol}: {e}")
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        try:
            metrics = {
                "quotes_generated": self.quotes_generated,
                "quotes_filled": self.quotes_filled,
                "fill_rate": self.quotes_filled / max(self.quotes_generated, 1),
                "total_spread_improvement_bps": self.total_spread_improvement,
                "avg_spread_improvement_bps": self.total_spread_improvement / max(self.quotes_generated, 1),
                "active_symbols": len(self.current_quotes),
                "last_prediction": self.last_prediction.direction if self.last_prediction else None,
                "last_prediction_confidence": self.last_prediction.confidence if self.last_prediction else 0.0,
                "inventory_tracking": self.inventory_tracking.copy(),
                "uptime_seconds": (datetime.now() - self.last_quote_time).total_seconds() if self.running else 0
            }
            
            # Add per-symbol metrics
            symbol_metrics = {}
            for symbol, quotes in self.current_quotes.items():
                symbol_metrics[symbol] = {
                    "active_quotes": len(quotes),
                    "bid_quotes": len([q for q in quotes if q.side == "buy"]),
                    "ask_quotes": len([q for q in quotes if q.side == "sell"]),
                    "avg_confidence": sum(q.confidence.value for q in quotes) / len(quotes) if quotes else 0.0,
                    "market_state": self.market_states.get(symbol).__dict__ if self.market_states.get(symbol) else None
                }
            
            metrics["symbol_metrics"] = symbol_metrics
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return {"error": str(e)}
    
    def get_status(self) -> Dict:
        """Get current strategy status."""
        return {
            "running": self.running,
            "last_quote_time": self.last_quote_time.isoformat(),
            "active_symbols": list(self.current_quotes.keys()),
            "quote_frequency_hz": self.quote_frequency,
            "max_quote_levels": self.max_quote_levels,
            "total_quotes_generated": self.quotes_generated,
            "prediction_confidence": self.last_prediction.confidence if self.last_prediction else 0.0,
            "spread_improvement_bps": self.total_spread_improvement
        }
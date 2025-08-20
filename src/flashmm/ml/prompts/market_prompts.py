"""
FlashMM Market Prediction Prompt Engineering

Structured prompt engineering for Azure OpenAI o4-mini market predictions
with optimized prompts for short-term price movement forecasting.
"""

import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from decimal import Decimal

from flashmm.data.storage.data_models import OrderBookSnapshot, Trade, MarketStats
from flashmm.utils.logging import get_logger

logger = get_logger(__name__)


class MarketPredictionPrompt:
    """Structured prompt builder for market prediction using Azure OpenAI o4-mini."""
    
    SYSTEM_PROMPT = """You are an expert quantitative trading assistant specializing in ultra-short-term cryptocurrency price prediction for automated market making.

Your task is to analyze real-time market data and predict price direction for the next 100-500ms timeframe with high precision.

RESPONSE FORMAT (JSON only, no additional text):
{
    "direction": "bullish|bearish|neutral",
    "confidence": 0.0-1.0,
    "price_change_bps": -500 to +500,
    "magnitude": "low|medium|high",
    "reasoning": "brief technical explanation",
    "key_factors": ["factor1", "factor2", "factor3"]
}

ANALYSIS PRIORITIES:
1. Order book imbalance and depth analysis
2. Recent trade momentum and volume patterns
3. Bid-ask spread dynamics and liquidity
4. Microstructure signals (order flow, pressure)
5. Short-term volatility and pattern recognition

CONSTRAINTS:
- Focus on 100-500ms prediction horizon
- Be deterministic and precise
- Confidence must reflect actual conviction
- Consider market microstructure over macro factors
- Prioritize order book signals over price history"""

    def __init__(self):
        """Initialize prompt builder."""
        self.prompt_cache = {}
        
    def build_market_prompt(
        self, 
        order_book: OrderBookSnapshot,
        recent_trades: List[Trade],
        market_stats: Optional[MarketStats] = None,
        prediction_horizon_ms: int = 200
    ) -> str:
        """Build comprehensive market analysis prompt.
        
        Args:
            order_book: Current order book snapshot
            recent_trades: Recent trades (last 10-20)
            market_stats: Market statistics (optional)
            prediction_horizon_ms: Prediction horizon in milliseconds
            
        Returns:
            Formatted prompt string
        """
        try:
            # Calculate market metrics
            book_metrics = self._calculate_book_metrics(order_book)
            trade_metrics = self._calculate_trade_metrics(recent_trades)
            
            prompt = f"""MARKET DATA ANALYSIS for {order_book.symbol}
Timestamp: {order_book.timestamp.isoformat()}
Prediction Horizon: {prediction_horizon_ms}ms

ORDER BOOK SNAPSHOT:
{self._format_order_book(order_book, book_metrics)}

RECENT TRADES ({len(recent_trades)} trades):
{self._format_recent_trades(recent_trades, trade_metrics)}

MARKET MICROSTRUCTURE:
{self._format_microstructure_signals(order_book, recent_trades, book_metrics, trade_metrics)}

{self._format_market_stats(market_stats) if market_stats else ""}

PREDICT: Price direction for next {prediction_horizon_ms}ms with confidence and magnitude."""

            return prompt
            
        except Exception as e:
            logger.error(f"Failed to build market prompt: {e}")
            return self._build_fallback_prompt(order_book.symbol)
    
    def _calculate_book_metrics(self, order_book: OrderBookSnapshot) -> Dict[str, Any]:
        """Calculate order book derived metrics."""
        metrics = {}
        
        try:
            if order_book.bids and order_book.asks:
                # Basic metrics
                metrics['best_bid'] = order_book.best_bid
                metrics['best_ask'] = order_book.best_ask
                metrics['mid_price'] = order_book.mid_price
                metrics['spread'] = order_book.spread
                metrics['spread_bps'] = order_book.spread_bps
                
                # Size metrics
                metrics['bid_size'] = order_book.bids[0].size if order_book.bids else Decimal('0')
                metrics['ask_size'] = order_book.asks[0].size if order_book.asks else Decimal('0')
                
                # Imbalance metrics
                total_bid_size = sum(level.size for level in order_book.bids[:5])
                total_ask_size = sum(level.size for level in order_book.asks[:5])
                
                if total_bid_size + total_ask_size > 0:
                    metrics['book_imbalance'] = float((total_bid_size - total_ask_size) / (total_bid_size + total_ask_size))
                else:
                    metrics['book_imbalance'] = 0.0
                
                # Depth metrics
                metrics['bid_depth_5'] = float(total_bid_size)
                metrics['ask_depth_5'] = float(total_ask_size)
                
                # Price level metrics
                if len(order_book.bids) >= 3 and len(order_book.asks) >= 3:
                    bid_levels = [level.price for level in order_book.bids[:3]]
                    ask_levels = [level.price for level in order_book.asks[:3]]
                    
                    metrics['bid_price_variance'] = float(max(bid_levels) - min(bid_levels))
                    metrics['ask_price_variance'] = float(max(ask_levels) - min(ask_levels))
                
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating book metrics: {e}")
            return {}
    
    def _calculate_trade_metrics(self, trades: List[Trade]) -> Dict[str, Any]:
        """Calculate trade-based metrics."""
        metrics = {}
        
        try:
            if not trades:
                return metrics
            
            # Volume metrics
            total_volume = sum(trade.size for trade in trades)
            buy_volume = sum(trade.size for trade in trades if trade.side.value in ['buy', 'bid'])
            sell_volume = sum(trade.size for trade in trades if trade.side.value in ['sell', 'ask'])
            
            metrics['total_volume'] = float(total_volume)
            metrics['buy_volume'] = float(buy_volume)
            metrics['sell_volume'] = float(sell_volume)
            
            if total_volume > 0:
                metrics['buy_ratio'] = float(buy_volume / total_volume)
                metrics['sell_ratio'] = float(sell_volume / total_volume)
            else:
                metrics['buy_ratio'] = 0.5
                metrics['sell_ratio'] = 0.5
            
            # Price momentum
            if len(trades) >= 2:
                recent_prices = [float(trade.price) for trade in trades[-5:]]
                if len(recent_prices) >= 2:
                    price_change = recent_prices[-1] - recent_prices[0]
                    metrics['price_momentum'] = price_change
                    metrics['price_momentum_bps'] = (price_change / recent_prices[0]) * 10000 if recent_prices[0] > 0 else 0
            
            # Trade frequency
            if len(trades) >= 2:
                time_span = (trades[-1].timestamp - trades[0].timestamp).total_seconds()
                if time_span > 0:
                    metrics['trade_frequency'] = len(trades) / time_span
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating trade metrics: {e}")
            return {}
    
    def _format_order_book(self, order_book: OrderBookSnapshot, metrics: Dict[str, Any]) -> str:
        """Format order book for prompt."""
        try:
            lines = []
            
            # Best bid/ask
            lines.append(f"Best Bid: ${metrics.get('best_bid', 'N/A')} (Size: {metrics.get('bid_size', 'N/A')})")
            lines.append(f"Best Ask: ${metrics.get('best_ask', 'N/A')} (Size: {metrics.get('ask_size', 'N/A')})")
            lines.append(f"Mid Price: ${metrics.get('mid_price', 'N/A')}")
            lines.append(f"Spread: {metrics.get('spread_bps', 'N/A'):.1f} bps")
            
            # Order book depth (top 5 levels)
            lines.append("\nBid Levels (Top 5):")
            for i, level in enumerate(order_book.bids[:5]):
                lines.append(f"  {i+1}. ${level.price} @ {level.size}")
            
            lines.append("Ask Levels (Top 5):")
            for i, level in enumerate(order_book.asks[:5]):
                lines.append(f"  {i+1}. ${level.price} @ {level.size}")
            
            # Imbalance
            imbalance = metrics.get('book_imbalance', 0)
            lines.append(f"\nBook Imbalance: {imbalance:.3f} ({'Bid Heavy' if imbalance > 0 else 'Ask Heavy' if imbalance < 0 else 'Balanced'})")
            
            return "\n".join(lines)
            
        except Exception as e:
            logger.error(f"Error formatting order book: {e}")
            return "Order book formatting error"
    
    def _format_recent_trades(self, trades: List[Trade], metrics: Dict[str, Any]) -> str:
        """Format recent trades for prompt."""
        try:
            if not trades:
                return "No recent trades"
            
            lines = []
            
            # Trade summary
            lines.append(f"Total Volume: {metrics.get('total_volume', 0):.4f}")
            lines.append(f"Buy/Sell Ratio: {metrics.get('buy_ratio', 0.5):.3f}/{metrics.get('sell_ratio', 0.5):.3f}")
            
            if 'price_momentum_bps' in metrics:
                lines.append(f"Price Momentum: {metrics['price_momentum_bps']:.1f} bps")
            
            # Recent trades (last 5)
            lines.append("\nLast 5 Trades:")
            for i, trade in enumerate(trades[-5:]):
                side_symbol = "🟢" if trade.side.value in ['buy', 'bid'] else "🔴"
                lines.append(f"  {side_symbol} ${trade.price} @ {trade.size} ({trade.side.value})")
            
            return "\n".join(lines)
            
        except Exception as e:
            logger.error(f"Error formatting trades: {e}")
            return "Trade formatting error"
    
    def _format_microstructure_signals(
        self, 
        order_book: OrderBookSnapshot, 
        trades: List[Trade],
        book_metrics: Dict[str, Any],
        trade_metrics: Dict[str, Any]
    ) -> str:
        """Format microstructure analysis."""
        try:
            lines = []
            
            # Order flow imbalance
            imbalance = book_metrics.get('book_imbalance', 0)
            if abs(imbalance) > 0.2:
                signal_strength = "Strong" if abs(imbalance) > 0.4 else "Moderate"
                direction = "Bullish" if imbalance > 0 else "Bearish"
                lines.append(f"Order Flow: {signal_strength} {direction} ({imbalance:.3f})")
            else:
                lines.append("Order Flow: Balanced")
            
            # Spread analysis
            spread_bps = book_metrics.get('spread_bps', 0)
            if spread_bps:
                if spread_bps < 10:
                    lines.append("Spread: Tight (High Liquidity)")
                elif spread_bps > 50:
                    lines.append("Spread: Wide (Low Liquidity)")
                else:
                    lines.append("Spread: Normal")
            
            # Volume pressure
            buy_ratio = trade_metrics.get('buy_ratio', 0.5)
            if buy_ratio > 0.7:
                lines.append("Volume Pressure: Strong Buying")
            elif buy_ratio < 0.3:
                lines.append("Volume Pressure: Strong Selling")
            else:
                lines.append("Volume Pressure: Balanced")
            
            # Momentum signal
            momentum_bps = trade_metrics.get('price_momentum_bps', 0)
            if abs(momentum_bps) > 10:
                direction = "Upward" if momentum_bps > 0 else "Downward"
                strength = "Strong" if abs(momentum_bps) > 25 else "Moderate"
                lines.append(f"Price Momentum: {strength} {direction} ({momentum_bps:.1f} bps)")
            
            return "\n".join(lines)
            
        except Exception as e:
            logger.error(f"Error formatting microstructure: {e}")
            return "Microstructure analysis error"
    
    def _format_market_stats(self, stats: MarketStats) -> str:
        """Format market statistics."""
        try:
            lines = ["\nMARKET STATISTICS:"]
            
            if stats.volume > 0:
                lines.append(f"Volume ({stats.window_seconds}s): {stats.volume}")
            
            if stats.trade_count > 0:
                lines.append(f"Trade Count: {stats.trade_count}")
            
            if stats.vwap:
                lines.append(f"VWAP: ${stats.vwap}")
            
            if stats.avg_spread_bps:
                lines.append(f"Avg Spread: {stats.avg_spread_bps:.1f} bps")
            
            return "\n".join(lines)
            
        except Exception as e:
            logger.error(f"Error formatting market stats: {e}")
            return ""
    
    def _build_fallback_prompt(self, symbol: str) -> str:
        """Build a minimal fallback prompt when data is insufficient."""
        return f"""MARKET DATA ANALYSIS for {symbol}
Timestamp: {datetime.utcnow().isoformat()}

LIMITED DATA AVAILABLE
Please provide a conservative neutral prediction with low confidence.

PREDICT: Price direction for next 200ms with confidence and magnitude."""

    def validate_prompt_size(self, prompt: str, max_tokens: int = 1000) -> bool:
        """Validate prompt size doesn't exceed token limits."""
        # Rough estimation: 1 token ≈ 4 characters
        estimated_tokens = len(prompt) / 4
        return estimated_tokens <= max_tokens


class PredictionResponseParser:
    """Parse and validate Azure OpenAI prediction responses."""
    
    def __init__(self):
        """Initialize response parser."""
        self.validation_errors = []
    
    async def parse_prediction(self, response_content: str) -> Dict[str, Any]:
        """Parse AI response into structured prediction.
        
        Args:
            response_content: Raw response from Azure OpenAI
            
        Returns:
            Structured prediction dictionary
            
        Raises:
            ValueError: If response cannot be parsed or validated
        """
        try:
            # Clean and extract JSON
            cleaned_response = self._clean_response(response_content)
            prediction_data = json.loads(cleaned_response)
            
            # Validate and normalize
            validated_prediction = self._validate_prediction(prediction_data)
            
            return validated_prediction
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}")
            return self._create_fallback_prediction(f"JSON parse error: {e}")
        
        except Exception as e:
            logger.error(f"Prediction parsing failed: {e}")
            return self._create_fallback_prediction(f"Parse error: {e}")
    
    def _clean_response(self, response: str) -> str:
        """Clean response content to extract valid JSON."""
        try:
            # Remove common prefixes/suffixes
            response = response.strip()
            
            # Find JSON boundaries
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                return response[start_idx:end_idx]
            
            return response
            
        except Exception:
            return response
    
    def _validate_prediction(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and normalize prediction data."""
        validated = {}
        
        # Required fields
        required_fields = ['direction', 'confidence', 'price_change_bps']
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")
        
        # Validate direction
        direction = str(data['direction']).lower()
        if direction not in ['bullish', 'bearish', 'neutral']:
            raise ValueError(f"Invalid direction: {direction}")
        validated['direction'] = direction
        
        # Validate confidence
        confidence = float(data['confidence'])
        if not 0 <= confidence <= 1:
            raise ValueError(f"Invalid confidence: {confidence}")
        validated['confidence'] = confidence
        
        # Validate price change
        price_change_bps = float(data['price_change_bps'])
        if abs(price_change_bps) > 500:
            logger.warning(f"Large price change prediction: {price_change_bps} bps")
            # Clamp to reasonable range
            price_change_bps = max(-500, min(500, price_change_bps))
        validated['price_change_bps'] = price_change_bps
        
        # Optional fields
        validated['magnitude'] = str(data.get('magnitude', 'medium')).lower()
        validated['reasoning'] = str(data.get('reasoning', 'No reasoning provided'))[:200]  # Limit length
        validated['key_factors'] = data.get('key_factors', [])[:5]  # Limit to 5 factors
        
        # Add metadata
        validated['timestamp'] = datetime.utcnow().isoformat()
        validated['model_version'] = 'azure-openai-o4-mini'
        
        return validated
    
    def _create_fallback_prediction(self, error_msg: str) -> Dict[str, Any]:
        """Create a safe fallback prediction when parsing fails."""
        return {
            'direction': 'neutral',
            'confidence': 0.5,
            'price_change_bps': 0.0,
            'magnitude': 'low',
            'reasoning': f'Fallback prediction: {error_msg}',
            'key_factors': ['parsing_error'],
            'timestamp': datetime.utcnow().isoformat(),
            'model_version': 'azure-openai-fallback'
        }
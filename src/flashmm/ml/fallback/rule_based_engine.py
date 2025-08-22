"""
FlashMM Rule-Based Prediction Engine

Fallback prediction engine using traditional market-making signals
when Azure OpenAI is unavailable. Provides reliable predictions based
on order book analysis and technical indicators.
"""

from datetime import datetime
from typing import Any

import numpy as np

from flashmm.data.storage.data_models import MarketStats, OrderBookSnapshot, Trade
from flashmm.utils.logging import get_logger

logger = get_logger(__name__)


class RuleBasedEngine:
    """Rule-based prediction engine for fallback scenarios."""

    def __init__(self):
        """Initialize rule-based engine."""
        self.prediction_count = 0
        self.last_prediction_time = None

        # Confidence thresholds
        self.high_confidence_threshold = 0.75
        self.medium_confidence_threshold = 0.60
        self.low_confidence_threshold = 0.45

        # Signal strength weights
        self.weights = {
            'order_flow_imbalance': 0.30,
            'spread_analysis': 0.20,
            'momentum_signal': 0.25,
            'volume_pressure': 0.15,
            'microstructure': 0.10
        }

    async def predict(
        self,
        order_book: OrderBookSnapshot,
        recent_trades: list[Trade],
        market_stats: MarketStats | None = None
    ) -> dict[str, Any]:
        """Generate rule-based prediction.

        Args:
            order_book: Current order book snapshot
            recent_trades: Recent trades list
            market_stats: Market statistics (optional)

        Returns:
            Structured prediction dictionary
        """
        try:
            start_time = datetime.utcnow()

            # Extract signals
            signals = await self._extract_signals(order_book, recent_trades, market_stats)

            # Generate prediction
            prediction = await self._generate_prediction(signals, order_book.symbol)

            # Add metadata
            prediction['response_time_ms'] = (datetime.utcnow() - start_time).total_seconds() * 1000
            prediction['signal_count'] = len(signals)
            prediction['engine_version'] = 'rule-based-v1.0'

            self.prediction_count += 1
            self.last_prediction_time = datetime.utcnow()

            return prediction

        except Exception as e:
            logger.error(f"Rule-based prediction failed: {e}")
            return self._create_neutral_prediction(order_book.symbol)

    async def _extract_signals(
        self,
        order_book: OrderBookSnapshot,
        trades: list[Trade],
        stats: MarketStats | None
    ) -> dict[str, dict[str, Any]]:
        """Extract trading signals from market data."""
        signals = {}

        # Order flow imbalance signal
        signals['order_flow'] = await self._analyze_order_flow(order_book)

        # Spread analysis signal
        signals['spread'] = await self._analyze_spread(order_book)

        # Momentum signal
        if trades:
            signals['momentum'] = await self._analyze_momentum(trades)

        # Volume pressure signal
        if trades:
            signals['volume'] = await self._analyze_volume_pressure(trades)

        # Microstructure signal
        signals['microstructure'] = await self._analyze_microstructure(order_book, trades)

        # Market regime signal
        if stats:
            signals['regime'] = await self._analyze_market_regime(stats)

        return signals

    async def _analyze_order_flow(self, book: OrderBookSnapshot) -> dict[str, Any]:
        """Analyze order flow imbalance."""
        signal = {
            'direction': 'neutral',
            'strength': 0.0,
            'confidence': 0.5,
            'reasoning': 'No clear order flow signal'
        }

        try:
            if not book.bids or not book.asks:
                return signal

            # Calculate imbalance at multiple levels
            levels_to_check = min(5, len(book.bids), len(book.asks))

            bid_volume = sum(float(level.size) for level in book.bids[:levels_to_check])
            ask_volume = sum(float(level.size) for level in book.asks[:levels_to_check])
            total_volume = bid_volume + ask_volume

            if total_volume == 0:
                return signal

            imbalance = (bid_volume - ask_volume) / total_volume

            # Classify imbalance
            if abs(imbalance) > 0.4:
                # Strong imbalance
                signal['direction'] = 'bullish' if imbalance > 0 else 'bearish'
                signal['strength'] = abs(imbalance)
                signal['confidence'] = min(0.8, 0.5 + abs(imbalance))
                signal['reasoning'] = f"Strong {'bid' if imbalance > 0 else 'ask'} volume dominance"

            elif abs(imbalance) > 0.2:
                # Moderate imbalance
                signal['direction'] = 'bullish' if imbalance > 0 else 'bearish'
                signal['strength'] = abs(imbalance)
                signal['confidence'] = 0.5 + abs(imbalance) * 0.5
                signal['reasoning'] = f"Moderate {'bid' if imbalance > 0 else 'ask'} volume advantage"

            else:
                # Balanced
                signal['direction'] = 'neutral'
                signal['strength'] = 0.0
                signal['confidence'] = 0.6  # High confidence in neutrality
                signal['reasoning'] = 'Balanced order book'

            # Add additional metrics
            signal['imbalance_ratio'] = imbalance
            signal['bid_volume'] = bid_volume
            signal['ask_volume'] = ask_volume

            return signal

        except Exception as e:
            logger.error(f"Order flow analysis failed: {e}")
            return signal

    async def _analyze_spread(self, book: OrderBookSnapshot) -> dict[str, Any]:
        """Analyze spread dynamics."""
        signal = {
            'direction': 'neutral',
            'strength': 0.0,
            'confidence': 0.5,
            'reasoning': 'No spread signal'
        }

        try:
            if not book.spread_bps or not book.mid_price:
                return signal

            spread_bps = float(book.spread_bps)

            # Spread regime classification
            if spread_bps < 5:
                # Extremely tight spread - high liquidity, potential mean reversion
                signal['direction'] = 'neutral'  # Mean reversion bias
                signal['strength'] = 0.3
                signal['confidence'] = 0.7
                signal['reasoning'] = 'Extremely tight spread suggests mean reversion'

            elif spread_bps < 15:
                # Tight spread - normal market conditions
                signal['direction'] = 'neutral'
                signal['strength'] = 0.1
                signal['confidence'] = 0.6
                signal['reasoning'] = 'Tight spread indicates normal liquidity'

            elif spread_bps < 50:
                # Normal spread
                signal['direction'] = 'neutral'
                signal['strength'] = 0.0
                signal['confidence'] = 0.5
                signal['reasoning'] = 'Normal spread conditions'

            elif spread_bps < 100:
                # Wide spread - reduced liquidity
                signal['direction'] = 'neutral'
                signal['strength'] = 0.2
                signal['confidence'] = 0.4  # Lower confidence due to uncertainty
                signal['reasoning'] = 'Wide spread indicates reduced liquidity'

            else:
                # Very wide spread - illiquid market
                signal['direction'] = 'neutral'
                signal['strength'] = 0.0
                signal['confidence'] = 0.3  # Very low confidence
                signal['reasoning'] = 'Very wide spread indicates illiquid conditions'

            signal['spread_bps'] = spread_bps

            return signal

        except Exception as e:
            logger.error(f"Spread analysis failed: {e}")
            return signal

    async def _analyze_momentum(self, trades: list[Trade]) -> dict[str, Any]:
        """Analyze price momentum from recent trades."""
        signal = {
            'direction': 'neutral',
            'strength': 0.0,
            'confidence': 0.5,
            'reasoning': 'No momentum signal'
        }

        try:
            if len(trades) < 3:
                return signal

            # Calculate price momentum
            prices = [float(trade.price) for trade in trades]

            # Short-term momentum (last 3 vs previous 3)
            if len(prices) >= 6:
                recent_avg = np.mean(prices[-3:])
                older_avg = np.mean(prices[-6:-3])

                if older_avg > 0:
                    momentum = (recent_avg - older_avg) / older_avg
                    momentum_bps = momentum * 10000

                    # Classify momentum
                    if abs(momentum_bps) > 25:
                        # Strong momentum
                        signal['direction'] = 'bullish' if momentum > 0 else 'bearish'
                        signal['strength'] = min(1.0, abs(momentum_bps) / 50)
                        signal['confidence'] = min(0.8, 0.5 + abs(momentum_bps) / 100)
                        signal['reasoning'] = f"Strong {'upward' if momentum > 0 else 'downward'} momentum"

                    elif abs(momentum_bps) > 10:
                        # Moderate momentum
                        signal['direction'] = 'bullish' if momentum > 0 else 'bearish'
                        signal['strength'] = abs(momentum_bps) / 50
                        signal['confidence'] = 0.5 + abs(momentum_bps) / 200
                        signal['reasoning'] = f"Moderate {'upward' if momentum > 0 else 'downward'} momentum"

                    signal['momentum_bps'] = momentum_bps

            # Also check simple price change
            if len(prices) >= 2:
                price_change = prices[-1] - prices[0]
                if prices[0] > 0:
                    change_bps = (price_change / prices[0]) * 10000
                    signal['price_change_bps'] = change_bps

            return signal

        except Exception as e:
            logger.error(f"Momentum analysis failed: {e}")
            return signal

    async def _analyze_volume_pressure(self, trades: list[Trade]) -> dict[str, Any]:
        """Analyze volume pressure from trade sides."""
        signal = {
            'direction': 'neutral',
            'strength': 0.0,
            'confidence': 0.5,
            'reasoning': 'No volume pressure signal'
        }

        try:
            if not trades:
                return signal

            # Calculate buy/sell volume
            buy_volume = sum(float(trade.size) for trade in trades
                           if trade.side.value in ['buy', 'bid'])
            sell_volume = sum(float(trade.size) for trade in trades
                            if trade.side.value in ['sell', 'ask'])
            total_volume = buy_volume + sell_volume

            if total_volume == 0:
                return signal

            buy_ratio = buy_volume / total_volume

            # Classify volume pressure
            if buy_ratio > 0.75:
                # Strong buying pressure
                signal['direction'] = 'bullish'
                signal['strength'] = (buy_ratio - 0.5) * 2  # Scale to 0-1
                signal['confidence'] = min(0.8, 0.5 + (buy_ratio - 0.5))
                signal['reasoning'] = 'Strong buying volume pressure'

            elif buy_ratio > 0.65:
                # Moderate buying pressure
                signal['direction'] = 'bullish'
                signal['strength'] = (buy_ratio - 0.5) * 2
                signal['confidence'] = 0.5 + (buy_ratio - 0.5) * 0.5
                signal['reasoning'] = 'Moderate buying volume pressure'

            elif buy_ratio < 0.25:
                # Strong selling pressure
                signal['direction'] = 'bearish'
                signal['strength'] = (0.5 - buy_ratio) * 2
                signal['confidence'] = min(0.8, 0.5 + (0.5 - buy_ratio))
                signal['reasoning'] = 'Strong selling volume pressure'

            elif buy_ratio < 0.35:
                # Moderate selling pressure
                signal['direction'] = 'bearish'
                signal['strength'] = (0.5 - buy_ratio) * 2
                signal['confidence'] = 0.5 + (0.5 - buy_ratio) * 0.5
                signal['reasoning'] = 'Moderate selling volume pressure'

            else:
                # Balanced volume
                signal['direction'] = 'neutral'
                signal['strength'] = 0.0
                signal['confidence'] = 0.6
                signal['reasoning'] = 'Balanced buy/sell volume'

            signal['buy_ratio'] = buy_ratio
            signal['buy_volume'] = buy_volume
            signal['sell_volume'] = sell_volume

            return signal

        except Exception as e:
            logger.error(f"Volume pressure analysis failed: {e}")
            return signal

    async def _analyze_microstructure(
        self,
        book: OrderBookSnapshot,
        trades: list[Trade]
    ) -> dict[str, Any]:
        """Analyze market microstructure signals."""
        signal = {
            'direction': 'neutral',
            'strength': 0.0,
            'confidence': 0.5,
            'reasoning': 'No microstructure signal'
        }

        try:
            # Size clustering analysis
            if len(book.bids) >= 3 and len(book.asks) >= 3:
                bid_sizes = [float(level.size) for level in book.bids[:3]]
                ask_sizes = [float(level.size) for level in book.asks[:3]]

                # Check for size dominance at best levels
                if bid_sizes[0] > ask_sizes[0] * 2:
                    signal['direction'] = 'bullish'
                    signal['strength'] = 0.3
                    signal['confidence'] = 0.6
                    signal['reasoning'] = 'Large bid at best level'

                elif ask_sizes[0] > bid_sizes[0] * 2:
                    signal['direction'] = 'bearish'
                    signal['strength'] = 0.3
                    signal['confidence'] = 0.6
                    signal['reasoning'] = 'Large ask at best level'

            # Trade frequency analysis
            if trades and len(trades) >= 5:
                recent_trades = trades[-5:]
                time_diffs = []
                for i in range(len(recent_trades) - 1):
                    diff = (recent_trades[i+1].timestamp - recent_trades[i].timestamp).total_seconds()
                    time_diffs.append(diff)

                if time_diffs:
                    avg_interval = np.mean(time_diffs)
                    if avg_interval < 1.0:  # Very frequent trading
                        signal['strength'] = max(signal['strength'], 0.2)
                        signal['reasoning'] += ' | High trade frequency'

            return signal

        except Exception as e:
            logger.error(f"Microstructure analysis failed: {e}")
            return signal

    async def _analyze_market_regime(self, stats: MarketStats) -> dict[str, Any]:
        """Analyze current market regime."""
        signal = {
            'direction': 'neutral',
            'strength': 0.0,
            'confidence': 0.5,
            'reasoning': 'No regime signal'
        }

        try:
            # Volume regime
            if stats.volume and stats.trade_count:
                avg_trade_size = float(stats.volume) / stats.trade_count

                # Classify by average trade size
                if avg_trade_size > 1000:  # Large trades
                    signal['strength'] = max(signal['strength'], 0.1)
                    signal['reasoning'] = 'Large average trade size regime'
                elif avg_trade_size < 100:  # Small trades
                    signal['confidence'] = max(0.3, signal['confidence'] - 0.1)  # Lower confidence
                    signal['reasoning'] = 'Small trade size regime'

            # Volatility regime
            if stats.high_price and stats.low_price and stats.vwap:
                price_range = float(stats.high_price - stats.low_price)
                vwap = float(stats.vwap)

                if vwap > 0:
                    volatility = (price_range / vwap) * 10000  # bps

                    if volatility > 100:  # High volatility
                        signal['confidence'] = max(0.3, signal['confidence'] - 0.2)
                        signal['reasoning'] += ' | High volatility regime'
                    elif volatility < 20:  # Low volatility
                        signal['confidence'] = min(0.8, signal['confidence'] + 0.1)
                        signal['reasoning'] += ' | Low volatility regime'

            return signal

        except Exception as e:
            logger.error(f"Market regime analysis failed: {e}")
            return signal

    async def _generate_prediction(self, signals: dict[str, dict[str, Any]], symbol: str) -> dict[str, Any]:
        """Generate final prediction from signals."""
        try:
            # Initialize prediction
            prediction = {
                'direction': 'neutral',
                'confidence': 0.5,
                'price_change_bps': 0.0,
                'magnitude': 'low',
                'reasoning': 'Rule-based analysis',
                'key_factors': [],
                'timestamp': datetime.utcnow().isoformat(),
                'model_version': 'rule-based-v1.0',
                'symbol': symbol
            }

            # Collect signal directions and strengths
            signal_scores = {'bullish': 0.0, 'bearish': 0.0, 'neutral': 0.0}
            confidence_sum = 0.0
            key_factors = []

            for signal_name, signal_data in signals.items():
                if not signal_data:
                    continue

                direction = signal_data.get('direction', 'neutral')
                strength = signal_data.get('strength', 0.0)
                confidence = signal_data.get('confidence', 0.5)
                reasoning = signal_data.get('reasoning', '')

                # Weight the signal
                weight = self.weights.get(signal_name, 0.1)
                weighted_strength = strength * weight * confidence

                signal_scores[direction] += weighted_strength
                confidence_sum += confidence * weight

                # Add to key factors if significant
                if strength > 0.2 and confidence > 0.6:
                    key_factors.append(f"{signal_name}: {reasoning}")

            # Determine final direction
            if signal_scores['bullish'] > signal_scores['bearish'] + 0.1:
                prediction['direction'] = 'bullish'
                net_strength = signal_scores['bullish'] - signal_scores['bearish']
            elif signal_scores['bearish'] > signal_scores['bullish'] + 0.1:
                prediction['direction'] = 'bearish'
                net_strength = signal_scores['bearish'] - signal_scores['bullish']
            else:
                prediction['direction'] = 'neutral'
                net_strength = 0.0

            # Calculate confidence
            if confidence_sum > 0:
                prediction['confidence'] = min(0.9, max(0.3, confidence_sum))

            # Estimate price change
            if net_strength > 0:
                # Scale strength to basis points (conservative estimates)
                base_change = net_strength * 20  # Max ~20 bps for strong signals

                if prediction['direction'] == 'bearish':
                    base_change = -base_change

                prediction['price_change_bps'] = round(base_change, 1)

            # Determine magnitude
            if abs(prediction['price_change_bps']) > 15:
                prediction['magnitude'] = 'high'
            elif abs(prediction['price_change_bps']) > 5:
                prediction['magnitude'] = 'medium'
            else:
                prediction['magnitude'] = 'low'

            # Add key factors
            prediction['key_factors'] = key_factors[:5]  # Limit to top 5

            # Build reasoning
            if key_factors:
                prediction['reasoning'] = f"Rule-based: {'; '.join(key_factors[:2])}"
            else:
                prediction['reasoning'] = "Rule-based: No strong signals detected"

            return prediction

        except Exception as e:
            logger.error(f"Prediction generation failed: {e}")
            return self._create_neutral_prediction(symbol)

    def _create_neutral_prediction(self, symbol: str) -> dict[str, Any]:
        """Create a safe neutral prediction."""
        return {
            'direction': 'neutral',
            'confidence': 0.5,
            'price_change_bps': 0.0,
            'magnitude': 'low',
            'reasoning': 'Rule-based fallback: Neutral stance due to analysis error',
            'key_factors': ['analysis_error'],
            'timestamp': datetime.utcnow().isoformat(),
            'model_version': 'rule-based-fallback',
            'symbol': symbol,
            'response_time_ms': 1.0,
            'signal_count': 0,
            'engine_version': 'rule-based-v1.0'
        }

    def get_performance_stats(self) -> dict[str, Any]:
        """Get performance statistics."""
        return {
            'prediction_count': self.prediction_count,
            'last_prediction_time': self.last_prediction_time.isoformat() if self.last_prediction_time else None,
            'engine_version': 'rule-based-v1.0',
            'signal_weights': self.weights,
            'confidence_thresholds': {
                'high': self.high_confidence_threshold,
                'medium': self.medium_confidence_threshold,
                'low': self.low_confidence_threshold
            }
        }

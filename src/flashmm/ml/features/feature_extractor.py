"""
FlashMM Feature Engineering Pipeline

Extracts meaningful features from market data for AI-driven predictions
with focus on microstructure signals and short-term patterns.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from decimal import Decimal
import asyncio

from flashmm.data.storage.data_models import OrderBookSnapshot, Trade, MarketStats
from flashmm.utils.logging import get_logger

logger = get_logger(__name__)


class FeatureExtractor:
    """Extract features from market data for ML predictions."""
    
    def __init__(self, lookback_seconds: int = 30):
        """Initialize feature extractor.
        
        Args:
            lookback_seconds: Historical data lookback window
        """
        self.lookback_seconds = lookback_seconds
        self.feature_cache = {}
        self.last_extraction_time = None
        
    async def extract_features(
        self,
        current_book: OrderBookSnapshot,
        recent_trades: List[Trade],
        historical_books: List[OrderBookSnapshot] = None,
        market_stats: Optional[MarketStats] = None
    ) -> Dict[str, Any]:
        """Extract comprehensive feature set from market data.
        
        Args:
            current_book: Current order book snapshot
            recent_trades: Recent trades list
            historical_books: Historical order book snapshots
            market_stats: Market statistics (optional)
            
        Returns:
            Dictionary of extracted features
        """
        try:
            features = {}
            
            # Order book features
            book_features = await self._extract_book_features(current_book)
            features.update(book_features)
            
            # Trade-based features
            if recent_trades:
                trade_features = await self._extract_trade_features(recent_trades)
                features.update(trade_features)
            
            # Time series features
            if historical_books:
                time_series_features = await self._extract_time_series_features(historical_books)
                features.update(time_series_features)
            
            # Market microstructure features
            micro_features = await self._extract_microstructure_features(
                current_book, recent_trades
            )
            features.update(micro_features)
            
            # Technical indicators
            if recent_trades and len(recent_trades) >= 10:
                technical_features = await self._extract_technical_features(recent_trades)
                features.update(technical_features)
            
            # Market regime features
            if market_stats:
                regime_features = await self._extract_regime_features(market_stats)
                features.update(regime_features)
            
            # Add metadata
            features['feature_timestamp'] = datetime.utcnow().isoformat()
            features['symbol'] = current_book.symbol
            features['feature_count'] = len([k for k in features.keys() if k.startswith(('book_', 'trade_', 'ts_', 'micro_', 'tech_', 'regime_'))])
            
            self.last_extraction_time = datetime.utcnow()
            
            return features
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return await self._create_fallback_features(current_book.symbol)
    
    async def _extract_book_features(self, book: OrderBookSnapshot) -> Dict[str, float]:
        """Extract order book-based features."""
        features = {}
        
        try:
            if not book.bids or not book.asks:
                return features
            
            # Basic price features
            features['book_best_bid'] = float(book.best_bid or 0)
            features['book_best_ask'] = float(book.best_ask or 0)
            features['book_mid_price'] = float(book.mid_price or 0)
            features['book_spread'] = float(book.spread or 0)
            features['book_spread_bps'] = float(book.spread_bps or 0)
            
            # Size features
            features['book_bid_size'] = float(book.bids[0].size) if book.bids else 0.0
            features['book_ask_size'] = float(book.asks[0].size) if book.asks else 0.0
            features['book_size_ratio'] = (features['book_bid_size'] / max(features['book_ask_size'], 0.001))
            
            # Depth features (top 5 levels)
            bid_depths = [float(level.size) for level in book.bids[:5]]
            ask_depths = [float(level.size) for level in book.asks[:5]]
            
            features['book_bid_depth_5'] = sum(bid_depths)
            features['book_ask_depth_5'] = sum(ask_depths)
            features['book_total_depth_5'] = features['book_bid_depth_5'] + features['book_ask_depth_5']
            
            # Depth distribution
            if features['book_total_depth_5'] > 0:
                features['book_depth_imbalance'] = (features['book_bid_depth_5'] - features['book_ask_depth_5']) / features['book_total_depth_5']
            else:
                features['book_depth_imbalance'] = 0.0
            
            # Price level analysis
            if len(book.bids) >= 3 and len(book.asks) >= 3:
                bid_prices = [float(level.price) for level in book.bids[:3]]
                ask_prices = [float(level.price) for level in book.asks[:3]]
                
                # Price clustering
                features['book_bid_price_std'] = np.std(bid_prices) if len(bid_prices) > 1 else 0.0
                features['book_ask_price_std'] = np.std(ask_prices) if len(ask_prices) > 1 else 0.0
                
                # Relative spreads
                if features['book_mid_price'] > 0:
                    features['book_bid_spread_1'] = (features['book_mid_price'] - bid_prices[0]) / features['book_mid_price'] * 10000
                    features['book_ask_spread_1'] = (ask_prices[0] - features['book_mid_price']) / features['book_mid_price'] * 10000
                    
                    if len(bid_prices) > 1 and len(ask_prices) > 1:
                        features['book_bid_spread_2'] = (features['book_mid_price'] - bid_prices[1]) / features['book_mid_price'] * 10000
                        features['book_ask_spread_2'] = (ask_prices[1] - features['book_mid_price']) / features['book_mid_price'] * 10000
            
            # Size-weighted mid price
            if features['book_bid_size'] + features['book_ask_size'] > 0:
                features['book_weighted_mid'] = (
                    features['book_best_bid'] * features['book_ask_size'] + 
                    features['book_best_ask'] * features['book_bid_size']
                ) / (features['book_bid_size'] + features['book_ask_size'])
            else:
                features['book_weighted_mid'] = features['book_mid_price']
            
            return features
            
        except Exception as e:
            logger.error(f"Book feature extraction failed: {e}")
            return {}
    
    async def _extract_trade_features(self, trades: List[Trade]) -> Dict[str, float]:
        """Extract trade-based features."""
        features = {}
        
        try:
            if not trades:
                return features
            
            # Convert to pandas for easier analysis
            trade_data = []
            for trade in trades:
                trade_data.append({
                    'timestamp': trade.timestamp,
                    'price': float(trade.price),
                    'size': float(trade.size),
                    'side': 1 if trade.side.value in ['buy', 'bid'] else -1,
                    'notional': float(trade.price * trade.size)
                })
            
            df = pd.DataFrame(trade_data)
            
            # Volume features
            features['trade_total_volume'] = df['size'].sum()
            features['trade_total_notional'] = df['notional'].sum()
            features['trade_count'] = len(df)
            
            # Side analysis
            buy_trades = df[df['side'] == 1]
            sell_trades = df[df['side'] == -1]
            
            features['trade_buy_volume'] = buy_trades['size'].sum()
            features['trade_sell_volume'] = sell_trades['size'].sum()
            features['trade_buy_count'] = len(buy_trades)
            features['trade_sell_count'] = len(sell_trades)
            
            if features['trade_total_volume'] > 0:
                features['trade_buy_ratio'] = features['trade_buy_volume'] / features['trade_total_volume']
                features['trade_sell_ratio'] = features['trade_sell_volume'] / features['trade_total_volume']
            else:
                features['trade_buy_ratio'] = 0.5
                features['trade_sell_ratio'] = 0.5
            
            # Price momentum
            if len(df) >= 2:
                features['trade_price_change'] = df['price'].iloc[-1] - df['price'].iloc[0]
                if df['price'].iloc[0] > 0:
                    features['trade_price_change_bps'] = (features['trade_price_change'] / df['price'].iloc[0]) * 10000
                else:
                    features['trade_price_change_bps'] = 0.0
                
                # Recent vs older momentum
                if len(df) >= 4:
                    recent_avg = df['price'].iloc[-2:].mean()
                    older_avg = df['price'].iloc[:2].mean()
                    if older_avg > 0:
                        features['trade_momentum_ratio'] = recent_avg / older_avg
                    else:
                        features['trade_momentum_ratio'] = 1.0
            
            # Volume-weighted average price
            if features['trade_total_volume'] > 0:
                features['trade_vwap'] = (df['price'] * df['size']).sum() / features['trade_total_volume']
            else:
                features['trade_vwap'] = df['price'].mean() if len(df) > 0 else 0.0
            
            # Trade size distribution
            features['trade_avg_size'] = df['size'].mean()
            features['trade_med_size'] = df['size'].median()
            features['trade_max_size'] = df['size'].max()
            features['trade_size_std'] = df['size'].std()
            
            # Time-based features
            if len(df) >= 2:
                time_diffs = df['timestamp'].diff().dt.total_seconds().dropna()
                if len(time_diffs) > 0:
                    features['trade_avg_interval'] = time_diffs.mean()
                    features['trade_frequency'] = 1.0 / features['trade_avg_interval'] if features['trade_avg_interval'] > 0 else 0.0
            
            # Order flow imbalance
            df['signed_volume'] = df['size'] * df['side']
            features['trade_order_flow'] = df['signed_volume'].sum()
            
            # Recent trade intensity (last 5 trades)
            if len(df) >= 5:
                recent_df = df.tail(5)
                features['trade_recent_buy_ratio'] = (recent_df['side'] == 1).sum() / len(recent_df)
                features['trade_recent_avg_size'] = recent_df['size'].mean()
                features['trade_recent_volume'] = recent_df['size'].sum()
            
            return features
            
        except Exception as e:
            logger.error(f"Trade feature extraction failed: {e}")
            return {}
    
    async def _extract_time_series_features(self, historical_books: List[OrderBookSnapshot]) -> Dict[str, float]:
        """Extract time series features from historical order books."""
        features = {}
        
        try:
            if len(historical_books) < 3:
                return features
            
            # Extract price series
            timestamps = []
            mid_prices = []
            spreads = []
            depths = []
            
            for book in historical_books:
                if book.mid_price:
                    timestamps.append(book.timestamp)
                    mid_prices.append(float(book.mid_price))
                    spreads.append(float(book.spread_bps or 0))
                    
                    # Calculate total depth (top 3 levels)
                    bid_depth = sum(float(level.size) for level in book.bids[:3])
                    ask_depth = sum(float(level.size) for level in book.asks[:3])
                    depths.append(bid_depth + ask_depth)
            
            if len(mid_prices) < 3:
                return features
            
            # Price volatility
            price_returns = np.diff(mid_prices) / np.array(mid_prices[:-1])
            features['ts_price_volatility'] = np.std(price_returns) * 10000  # in bps
            features['ts_price_trend'] = np.polyfit(range(len(mid_prices)), mid_prices, 1)[0]
            
            # Spread stability
            features['ts_spread_mean'] = np.mean(spreads)
            features['ts_spread_std'] = np.std(spreads)
            features['ts_spread_trend'] = np.polyfit(range(len(spreads)), spreads, 1)[0]
            
            # Depth analysis
            if depths:
                features['ts_depth_mean'] = np.mean(depths)
                features['ts_depth_std'] = np.std(depths)
                features['ts_depth_trend'] = np.polyfit(range(len(depths)), depths, 1)[0]
            
            # Moving averages
            if len(mid_prices) >= 5:
                ma_short = np.mean(mid_prices[-3:])
                ma_long = np.mean(mid_prices[-5:])
                features['ts_ma_ratio'] = ma_short / ma_long if ma_long > 0 else 1.0
            
            # Price acceleration
            if len(mid_prices) >= 3:
                recent_change = mid_prices[-1] - mid_prices[-2]
                prev_change = mid_prices[-2] - mid_prices[-3]
                features['ts_price_acceleration'] = recent_change - prev_change
            
            return features
            
        except Exception as e:
            logger.error(f"Time series feature extraction failed: {e}")
            return {}
    
    async def _extract_microstructure_features(
        self, 
        book: OrderBookSnapshot, 
        trades: List[Trade]
    ) -> Dict[str, float]:
        """Extract market microstructure features."""
        features = {}
        
        try:
            # Order book pressure
            if book.bids and book.asks:
                # Aggregate size at different levels
                bid_sizes = [float(level.size) for level in book.bids[:10]]
                ask_sizes = [float(level.size) for level in book.asks[:10]]
                
                # Level-by-level analysis
                for i in range(min(3, len(bid_sizes), len(ask_sizes))):
                    level_imbalance = (bid_sizes[i] - ask_sizes[i]) / (bid_sizes[i] + ask_sizes[i])
                    features[f'micro_level_{i+1}_imbalance'] = level_imbalance
                
                # Cumulative depth imbalance
                cum_bid_depth = np.cumsum(bid_sizes[:5])
                cum_ask_depth = np.cumsum(ask_sizes[:5])
                
                for i in range(min(3, len(cum_bid_depth), len(cum_ask_depth))):
                    total_depth = cum_bid_depth[i] + cum_ask_depth[i]
                    if total_depth > 0:
                        features[f'micro_cum_imbalance_{i+1}'] = (cum_bid_depth[i] - cum_ask_depth[i]) / total_depth
            
            # Trade impact analysis
            if trades and len(trades) >= 3:
                # Price impact of trades
                recent_trades = trades[-3:]
                prices = [float(t.price) for t in recent_trades]
                sizes = [float(t.size) for t in recent_trades]
                
                if len(prices) >= 2:
                    price_changes = np.diff(prices)
                    avg_size = np.mean(sizes[:-1])  # Size before price change
                    
                    if avg_size > 0:
                        features['micro_price_impact'] = np.mean(np.abs(price_changes)) / avg_size
                
                # Trade clustering
                timestamps = [t.timestamp for t in recent_trades]
                if len(timestamps) >= 2:
                    intervals = [(timestamps[i+1] - timestamps[i]).total_seconds() 
                               for i in range(len(timestamps)-1)]
                    features['micro_trade_clustering'] = 1.0 / (np.mean(intervals) + 0.001)
            
            # Liquidity estimation
            if book.spread and book.mid_price:
                spread_cost = float(book.spread) / float(book.mid_price) * 10000  # bps
                bid_depth = sum(float(level.size) for level in book.bids[:3])
                ask_depth = sum(float(level.size) for level in book.asks[:3])
                avg_depth = (bid_depth + ask_depth) / 2
                
                if avg_depth > 0:
                    features['micro_liquidity_score'] = avg_depth / (spread_cost + 1)
                else:
                    features['micro_liquidity_score'] = 0.0
            
            return features
            
        except Exception as e:
            logger.error(f"Microstructure feature extraction failed: {e}")
            return {}
    
    async def _extract_technical_features(self, trades: List[Trade]) -> Dict[str, float]:
        """Extract technical analysis features."""
        features = {}
        
        try:
            if len(trades) < 10:
                return features
            
            # Create price series
            prices = [float(trade.price) for trade in trades]
            volumes = [float(trade.size) for trade in trades]
            
            # Simple moving averages
            if len(prices) >= 5:
                sma_5 = np.mean(prices[-5:])
                sma_10 = np.mean(prices[-10:]) if len(prices) >= 10 else sma_5
                
                features['tech_sma_5'] = sma_5
                features['tech_sma_10'] = sma_10
                features['tech_sma_ratio'] = sma_5 / sma_10 if sma_10 > 0 else 1.0
            
            # RSI approximation (simplified)
            if len(prices) >= 10:
                price_changes = np.diff(prices)
                gains = np.where(price_changes > 0, price_changes, 0)
                losses = np.where(price_changes < 0, -price_changes, 0)
                
                avg_gain = np.mean(gains)
                avg_loss = np.mean(losses)
                
                if avg_loss > 0:
                    rs = avg_gain / avg_loss
                    features['tech_rsi'] = 100 - (100 / (1 + rs))
                else:
                    features['tech_rsi'] = 100 if avg_gain > 0 else 50
            
            # Volume trend
            if len(volumes) >= 5:
                recent_vol = np.mean(volumes[-3:])
                older_vol = np.mean(volumes[-5:-2])
                features['tech_volume_trend'] = recent_vol / older_vol if older_vol > 0 else 1.0
            
            # Price momentum
            if len(prices) >= 3:
                features['tech_momentum_3'] = (prices[-1] - prices[-3]) / prices[-3] * 10000 if prices[-3] > 0 else 0.0
            
            if len(prices) >= 5:
                features['tech_momentum_5'] = (prices[-1] - prices[-5]) / prices[-5] * 10000 if prices[-5] > 0 else 0.0
            
            # Bollinger Bands approximation
            if len(prices) >= 10:
                mean_price = np.mean(prices[-10:])
                std_price = np.std(prices[-10:])
                current_price = prices[-1]
                
                if std_price > 0:
                    features['tech_bb_position'] = (current_price - mean_price) / (2 * std_price)
                else:
                    features['tech_bb_position'] = 0.0
            
            return features
            
        except Exception as e:
            logger.error(f"Technical feature extraction failed: {e}")
            return {}
    
    async def _extract_regime_features(self, stats: MarketStats) -> Dict[str, float]:
        """Extract market regime features."""
        features = {}
        
        try:
            # Volume regime
            if stats.volume > 0:
                features['regime_volume'] = float(stats.volume)
                features['regime_trade_count'] = stats.trade_count
                
                if stats.trade_count > 0:
                    features['regime_avg_trade_size'] = float(stats.volume) / stats.trade_count
                else:
                    features['regime_avg_trade_size'] = 0.0
            
            # Volatility regime (from price range)
            if stats.high_price and stats.low_price and stats.vwap:
                price_range = float(stats.high_price - stats.low_price)
                features['regime_price_range'] = price_range
                features['regime_volatility'] = price_range / float(stats.vwap) * 10000 if stats.vwap > 0 else 0.0
            
            # Spread regime
            if stats.avg_spread_bps:
                features['regime_avg_spread'] = float(stats.avg_spread_bps)
                
                # Classify spread regime
                if features['regime_avg_spread'] < 10:
                    features['regime_spread_tight'] = 1.0
                    features['regime_spread_normal'] = 0.0
                    features['regime_spread_wide'] = 0.0
                elif features['regime_avg_spread'] > 50:
                    features['regime_spread_tight'] = 0.0
                    features['regime_spread_normal'] = 0.0
                    features['regime_spread_wide'] = 1.0
                else:
                    features['regime_spread_tight'] = 0.0
                    features['regime_spread_normal'] = 1.0
                    features['regime_spread_wide'] = 0.0
            
            return features
            
        except Exception as e:
            logger.error(f"Regime feature extraction failed: {e}")
            return {}
    
    async def _create_fallback_features(self, symbol: str) -> Dict[str, Any]:
        """Create minimal fallback features when extraction fails."""
        return {
            'symbol': symbol,
            'feature_timestamp': datetime.utcnow().isoformat(),
            'feature_count': 0,
            'book_mid_price': 0.0,
            'book_spread_bps': 100.0,
            'trade_total_volume': 0.0,
            'fallback_mode': True
        }
    
    def get_feature_names(self) -> List[str]:
        """Get list of all possible feature names."""
        return [
            # Order book features
            'book_best_bid', 'book_best_ask', 'book_mid_price', 'book_spread', 'book_spread_bps',
            'book_bid_size', 'book_ask_size', 'book_size_ratio',
            'book_bid_depth_5', 'book_ask_depth_5', 'book_total_depth_5', 'book_depth_imbalance',
            'book_bid_price_std', 'book_ask_price_std',
            'book_bid_spread_1', 'book_ask_spread_1', 'book_bid_spread_2', 'book_ask_spread_2',
            'book_weighted_mid',
            
            # Trade features
            'trade_total_volume', 'trade_total_notional', 'trade_count',
            'trade_buy_volume', 'trade_sell_volume', 'trade_buy_count', 'trade_sell_count',
            'trade_buy_ratio', 'trade_sell_ratio',
            'trade_price_change', 'trade_price_change_bps', 'trade_momentum_ratio',
            'trade_vwap', 'trade_avg_size', 'trade_med_size', 'trade_max_size', 'trade_size_std',
            'trade_avg_interval', 'trade_frequency', 'trade_order_flow',
            'trade_recent_buy_ratio', 'trade_recent_avg_size', 'trade_recent_volume',
            
            # Time series features
            'ts_price_volatility', 'ts_price_trend', 'ts_spread_mean', 'ts_spread_std', 'ts_spread_trend',
            'ts_depth_mean', 'ts_depth_std', 'ts_depth_trend', 'ts_ma_ratio', 'ts_price_acceleration',
            
            # Microstructure features
            'micro_level_1_imbalance', 'micro_level_2_imbalance', 'micro_level_3_imbalance',
            'micro_cum_imbalance_1', 'micro_cum_imbalance_2', 'micro_cum_imbalance_3',
            'micro_price_impact', 'micro_trade_clustering', 'micro_liquidity_score',
            
            # Technical features
            'tech_sma_5', 'tech_sma_10', 'tech_sma_ratio', 'tech_rsi',
            'tech_volume_trend', 'tech_momentum_3', 'tech_momentum_5', 'tech_bb_position',
            
            # Regime features
            'regime_volume', 'regime_trade_count', 'regime_avg_trade_size',
            'regime_price_range', 'regime_volatility', 'regime_avg_spread',
            'regime_spread_tight', 'regime_spread_normal', 'regime_spread_wide'
        ]
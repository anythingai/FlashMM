"""
Unit tests for FlashMM data models.
"""

import pytest
from decimal import Decimal
from datetime import datetime
from pydantic import ValidationError

from flashmm.data.storage.data_models import (
    OrderBookLevel, OrderBookSnapshot, Trade, Side, OrderType, OrderStatus,
    MarketStats, Position, OrderRequest, OrderResponse, MLPrediction,
    HealthCheck, PerformanceMetric
)


class TestOrderBookLevel:
    """Test OrderBookLevel model."""
    
    def test_valid_orderbook_level(self):
        """Test valid order book level creation."""
        level = OrderBookLevel(price="0.042", size="1000.0")
        assert level.price == Decimal("0.042")
        assert level.size == Decimal("1000.0")
    
    def test_decimal_conversion(self):
        """Test automatic decimal conversion."""
        level = OrderBookLevel(price=0.042, size=1000)
        assert isinstance(level.price, Decimal)
        assert isinstance(level.size, Decimal)
    
    def test_invalid_price(self):
        """Test invalid price validation."""
        with pytest.raises(ValidationError):
            OrderBookLevel(price=0, size=1000)
        
        with pytest.raises(ValidationError):
            OrderBookLevel(price=-1, size=1000)
    
    def test_invalid_size(self):
        """Test invalid size validation."""
        with pytest.raises(ValidationError):
            OrderBookLevel(price=0.042, size=-100)
    
    def test_string_representation(self):
        """Test string representation."""
        level = OrderBookLevel(price="0.042", size="1000.0")
        assert str(level) == "0.042@1000.0"


class TestOrderBookSnapshot:
    """Test OrderBookSnapshot model."""
    
    def test_valid_orderbook_snapshot(self, sample_orderbook_data):
        """Test valid order book snapshot creation."""
        bids = [OrderBookLevel(price=bid[0], size=bid[1]) for bid in sample_orderbook_data["bids"]]
        asks = [OrderBookLevel(price=ask[0], size=ask[1]) for ask in sample_orderbook_data["asks"]]
        
        snapshot = OrderBookSnapshot(
            symbol=sample_orderbook_data["symbol"],
            timestamp=datetime.fromisoformat(sample_orderbook_data["timestamp"].replace("Z", "+00:00")),
            bids=bids,
            asks=asks,
            source="test"
        )
        
        assert snapshot.symbol == "SEI/USDC"
        assert len(snapshot.bids) == 2
        assert len(snapshot.asks) == 2
        assert snapshot.source == "test"
    
    def test_best_prices(self):
        """Test best bid/ask price calculation."""
        bids = [
            OrderBookLevel(price="0.042", size="1000"),
            OrderBookLevel(price="0.041", size="500")
        ]
        asks = [
            OrderBookLevel(price="0.043", size="1000"),
            OrderBookLevel(price="0.044", size="500")
        ]
        
        snapshot = OrderBookSnapshot(
            symbol="SEI/USDC",
            timestamp=datetime.now(),
            bids=bids,
            asks=asks
        )
        
        assert snapshot.best_bid == Decimal("0.042")
        assert snapshot.best_ask == Decimal("0.043")
        assert snapshot.spread == Decimal("0.001")
        assert snapshot.mid_price == Decimal("0.0425")
    
    def test_spread_bps_calculation(self):
        """Test spread in basis points calculation."""
        bids = [OrderBookLevel(price="100", size="1000")]
        asks = [OrderBookLevel(price="101", size="1000")]
        
        snapshot = OrderBookSnapshot(
            symbol="TEST/USDC",
            timestamp=datetime.now(),
            bids=bids,
            asks=asks
        )
        
        # Spread = 1, best_bid = 100, spread_bps = (1/100) * 10000 = 100 bps
        assert snapshot.spread_bps == Decimal("100")
    
    def test_empty_orderbook(self):
        """Test empty order book handling."""
        snapshot = OrderBookSnapshot(
            symbol="SEI/USDC",
            timestamp=datetime.now(),
            bids=[],
            asks=[]
        )
        
        assert snapshot.best_bid is None
        assert snapshot.best_ask is None
        assert snapshot.spread is None
        assert snapshot.mid_price is None
    
    def test_crossed_book_validation(self):
        """Test crossed book validation."""
        bids = [OrderBookLevel(price="0.045", size="1000")]  # Higher than ask
        asks = [OrderBookLevel(price="0.043", size="1000")]
        
        with pytest.raises(ValidationError, match="Crossed book detected"):
            OrderBookSnapshot(
                symbol="SEI/USDC",
                timestamp=datetime.now(),
                bids=bids,
                asks=asks
            )
    
    def test_bid_ordering_validation(self):
        """Test bid ordering validation."""
        bids = [
            OrderBookLevel(price="0.041", size="1000"),  # Wrong order
            OrderBookLevel(price="0.042", size="500")
        ]
        asks = [OrderBookLevel(price="0.043", size="1000")]
        
        with pytest.raises(ValidationError, match="Bids must be sorted in descending price order"):
            OrderBookSnapshot(
                symbol="SEI/USDC",
                timestamp=datetime.now(),
                bids=bids,
                asks=asks
            )
    
    def test_ask_ordering_validation(self):
        """Test ask ordering validation."""
        bids = [OrderBookLevel(price="0.042", size="1000")]
        asks = [
            OrderBookLevel(price="0.044", size="1000"),  # Wrong order
            OrderBookLevel(price="0.043", size="500")
        ]
        
        with pytest.raises(ValidationError, match="Asks must be sorted in ascending price order"):
            OrderBookSnapshot(
                symbol="SEI/USDC",
                timestamp=datetime.now(),
                bids=bids,
                asks=asks
            )
    
    def test_to_dict_conversion(self):
        """Test conversion to dictionary."""
        bids = [OrderBookLevel(price="0.042", size="1000")]
        asks = [OrderBookLevel(price="0.043", size="1000")]
        
        snapshot = OrderBookSnapshot(
            symbol="SEI/USDC",
            timestamp=datetime.now(),
            sequence=12345,
            bids=bids,
            asks=asks,
            source="test"
        )
        
        data = snapshot.to_dict()
        
        assert data["symbol"] == "SEI/USDC"
        assert data["sequence"] == 12345
        assert data["bids"] == [["0.042", "1000"]]
        assert data["asks"] == [["0.043", "1000"]]
        assert data["source"] == "test"
    
    def test_from_dict_creation(self):
        """Test creation from dictionary."""
        data = {
            "symbol": "SEI/USDC",
            "timestamp": "2024-01-01T00:00:00+00:00",
            "sequence": 12345,
            "bids": [["0.042", "1000"]],
            "asks": [["0.043", "1000"]],
            "source": "test"
        }
        
        snapshot = OrderBookSnapshot.from_dict(data)
        
        assert snapshot.symbol == "SEI/USDC"
        assert snapshot.sequence == 12345
        assert len(snapshot.bids) == 1
        assert len(snapshot.asks) == 1
        assert snapshot.source == "test"


class TestTrade:
    """Test Trade model."""
    
    def test_valid_trade(self, sample_trade_data):
        """Test valid trade creation."""
        trade = Trade(
            symbol=sample_trade_data["symbol"],
            timestamp=datetime.fromisoformat(sample_trade_data["timestamp"].replace("Z", "+00:00")),
            price=sample_trade_data["price"],
            size=sample_trade_data["size"],
            side=Side.BUY,
            source="test"
        )
        
        assert trade.symbol == "SEI/USDC"
        assert trade.price == Decimal("0.0421")
        assert trade.size == Decimal("100.0")
        assert trade.side == Side.BUY
    
    def test_notional_value_calculation(self):
        """Test notional value calculation."""
        trade = Trade(
            symbol="SEI/USDC",
            timestamp=datetime.now(),
            price="0.042",
            size="1000",
            side=Side.BUY
        )
        
        assert trade.notional_value == Decimal("42.0")
    
    def test_invalid_price(self):
        """Test invalid price validation."""
        with pytest.raises(ValidationError):
            Trade(
                symbol="SEI/USDC",
                timestamp=datetime.now(),
                price=0,
                size="1000",
                side=Side.BUY
            )
    
    def test_invalid_size(self):
        """Test invalid size validation."""
        with pytest.raises(ValidationError):
            Trade(
                symbol="SEI/USDC",
                timestamp=datetime.now(),
                price="0.042",
                size=0,
                side=Side.BUY
            )
    
    def test_to_dict_conversion(self):
        """Test conversion to dictionary."""
        trade = Trade(
            symbol="SEI/USDC",
            timestamp=datetime.now(),
            price="0.042",
            size="1000",
            side=Side.BUY,
            trade_id="12345",
            sequence=100
        )
        
        data = trade.to_dict()
        
        assert data["symbol"] == "SEI/USDC"
        assert data["price"] == "0.042"
        assert data["size"] == "1000"
        assert data["side"] == "buy"
        assert data["trade_id"] == "12345"
        assert data["sequence"] == 100


class TestMarketStats:
    """Test MarketStats model."""
    
    def test_valid_market_stats(self):
        """Test valid market stats creation."""
        stats = MarketStats(
            symbol="SEI/USDC",
            timestamp=datetime.now(),
            window_seconds=3600,
            open_price="0.040",
            high_price="0.045",
            low_price="0.038",
            close_price="0.042",
            volume="100000",
            trade_count=150
        )
        
        assert stats.symbol == "SEI/USDC"
        assert stats.window_seconds == 3600
        assert stats.open_price == Decimal("0.040")
        assert stats.trade_count == 150
    
    def test_decimal_conversion(self):
        """Test automatic decimal conversion."""
        stats = MarketStats(
            symbol="SEI/USDC",
            timestamp=datetime.now(),
            window_seconds=3600,
            volume=100000.5
        )
        
        assert isinstance(stats.volume, Decimal)
        assert stats.volume == Decimal("100000.5")


class TestPosition:
    """Test Position model."""
    
    def test_position_properties(self):
        """Test position property calculations."""
        position = Position(
            symbol="SEI/USDC",
            base_balance="1000",
            quote_balance="-42",
            unrealized_pnl="5.0",
            realized_pnl="2.0",
            last_updated=datetime.now()
        )
        
        assert position.net_position == Decimal("1000")
        assert position.total_pnl == Decimal("7.0")
        assert position.is_long is True
        assert position.is_short is False
        assert position.is_flat is False
    
    def test_short_position(self):
        """Test short position properties."""
        position = Position(
            symbol="SEI/USDC",
            base_balance="-500",
            quote_balance="21",
            last_updated=datetime.now()
        )
        
        assert position.is_long is False
        assert position.is_short is True
        assert position.is_flat is False
    
    def test_flat_position(self):
        """Test flat position properties."""
        position = Position(
            symbol="SEI/USDC",
            base_balance="0",
            quote_balance="0",
            last_updated=datetime.now()
        )
        
        assert position.is_long is False
        assert position.is_short is False
        assert position.is_flat is True


class TestOrderRequest:
    """Test OrderRequest model."""
    
    def test_valid_order_request(self):
        """Test valid order request creation."""
        order = OrderRequest(
            symbol="SEI/USDC",
            side=Side.BUY,
            price="0.042",
            size="1000",
            order_type=OrderType.LIMIT
        )
        
        assert order.symbol == "SEI/USDC"
        assert order.side == Side.BUY
        assert order.price == Decimal("0.042")
        assert order.size == Decimal("1000")
        assert order.order_type == OrderType.LIMIT
    
    def test_invalid_price(self):
        """Test invalid price validation."""
        with pytest.raises(ValidationError):
            OrderRequest(
                symbol="SEI/USDC",
                side=Side.BUY,
                price=0,
                size="1000"
            )
    
    def test_invalid_size(self):
        """Test invalid size validation."""
        with pytest.raises(ValidationError):
            OrderRequest(
                symbol="SEI/USDC",
                side=Side.BUY,
                price="0.042",
                size=0
            )


class TestOrderResponse:
    """Test OrderResponse model."""
    
    def test_fill_percentage_calculation(self):
        """Test fill percentage calculation."""
        order = OrderResponse(
            order_id="12345",
            symbol="SEI/USDC",
            side=Side.BUY,
            price="0.042",
            size="1000",
            status=OrderStatus.PARTIALLY_FILLED,
            timestamp=datetime.now(),
            filled_size="250"
        )
        
        assert order.fill_percentage == Decimal("25")
        assert order.is_fully_filled is False
    
    def test_fully_filled_order(self):
        """Test fully filled order properties."""
        order = OrderResponse(
            order_id="12345",
            symbol="SEI/USDC",
            side=Side.BUY,
            price="0.042",
            size="1000",
            status=OrderStatus.FILLED,
            timestamp=datetime.now(),
            filled_size="1000"
        )
        
        assert order.fill_percentage == Decimal("100")
        assert order.is_fully_filled is True


class TestMLPrediction:
    """Test MLPrediction model."""
    
    def test_valid_prediction(self):
        """Test valid ML prediction creation."""
        prediction = MLPrediction(
            symbol="SEI/USDC",
            timestamp=datetime.now(),
            prediction_type="price_direction",
            value=0.043,
            confidence=0.85,
            signal_strength=0.7,
            model_version="v1.0.0"
        )
        
        assert prediction.symbol == "SEI/USDC"
        assert prediction.confidence == 0.85
        assert prediction.signal_strength == 0.7
        assert prediction.is_bullish is True
        assert prediction.is_bearish is False
        assert prediction.is_high_confidence is True
    
    def test_bearish_prediction(self):
        """Test bearish prediction properties."""
        prediction = MLPrediction(
            symbol="SEI/USDC",
            timestamp=datetime.now(),
            prediction_type="price_direction",
            value=0.040,
            confidence=0.6,
            signal_strength=-0.5
        )
        
        assert prediction.is_bullish is False
        assert prediction.is_bearish is True
        assert prediction.is_high_confidence is False
    
    def test_confidence_validation(self):
        """Test confidence range validation."""
        with pytest.raises(ValidationError):
            MLPrediction(
                symbol="SEI/USDC",
                timestamp=datetime.now(),
                prediction_type="price",
                value=0.042,
                confidence=1.5,  # Invalid
                signal_strength=0.5
            )
    
    def test_signal_strength_validation(self):
        """Test signal strength range validation."""
        with pytest.raises(ValidationError):
            MLPrediction(
                symbol="SEI/USDC",
                timestamp=datetime.now(),
                prediction_type="price",
                value=0.042,
                confidence=0.8,
                signal_strength=1.5  # Invalid
            )


class TestHealthCheck:
    """Test HealthCheck model."""
    
    def test_valid_health_check(self):
        """Test valid health check creation."""
        health = HealthCheck(
            service="redis",
            status="healthy",
            timestamp=datetime.now(),
            latency_ms=2.5,
            details={"connected_clients": 10}
        )
        
        assert health.service == "redis"
        assert health.status == "healthy"
        assert health.is_healthy is True
        assert health.details["connected_clients"] == 10
    
    def test_unhealthy_status(self):
        """Test unhealthy status detection."""
        health = HealthCheck(
            service="influxdb",
            status="error",
            timestamp=datetime.now(),
            latency_ms=1000.0
        )
        
        assert health.is_healthy is False


class TestPerformanceMetric:
    """Test PerformanceMetric model."""
    
    def test_valid_metric(self):
        """Test valid performance metric creation."""
        metric = PerformanceMetric(
            metric_name="latency",
            value=25.5,
            timestamp=datetime.now(),
            tags={"service": "websocket", "operation": "message_processing"},
            unit="ms"
        )
        
        assert metric.metric_name == "latency"
        assert metric.value == 25.5
        assert metric.unit == "ms"
        assert metric.tags["service"] == "websocket"
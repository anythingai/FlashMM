"""
Unit tests for FlashMM data normalizer.
"""

from datetime import datetime
from decimal import Decimal
from unittest.mock import AsyncMock, patch

import pytest

from flashmm.data.ingestion.data_normalizer import DataNormalizer, SeiDataNormalizer
from flashmm.data.storage.data_models import Side
from flashmm.utils.exceptions import DataValidationError


class TestSeiDataNormalizer:
    """Test SeiDataNormalizer functionality."""

    @pytest.fixture
    async def normalizer(self):
        """Create normalizer instance."""
        normalizer = SeiDataNormalizer()
        await normalizer.initialize()
        return normalizer

    @pytest.fixture
    def sei_orderbook_message(self):
        """Sample Sei order book message."""
        return {
            "jsonrpc": "2.0",
            "id": "subscription_id",
            "result": {
                "events": [
                    {
                        "type": "OrderBookUpdate",
                        "attributes": [
                            {"key": "market_id", "value": "SEI/USDC"},
                            {"key": "sequence", "value": "12345"},
                            {"key": "timestamp", "value": "2024-01-01T00:00:00Z"}
                        ],
                        "data": {
                            "bids": [["0.042", "1000"], ["0.041", "500"]],
                            "asks": [["0.043", "1500"], ["0.044", "1200"]]
                        }
                    }
                ]
            }
        }

    @pytest.fixture
    def sei_trade_message(self):
        """Sample Sei trade message."""
        return {
            "jsonrpc": "2.0",
            "id": "trade_subscription",
            "result": {
                "events": [
                    {
                        "type": "Trade",
                        "attributes": [
                            {"key": "market_id", "value": "SEI/USDC"},
                            {"key": "price", "value": "0.0425"},
                            {"key": "size", "value": "1000"},
                            {"key": "side", "value": "buy"},
                            {"key": "trade_id", "value": "67890"},
                            {"key": "timestamp", "value": "2024-01-01T00:01:00Z"}
                        ]
                    }
                ]
            }
        }

    @pytest.fixture
    def sei_error_message(self):
        """Sample Sei error message."""
        return {
            "jsonrpc": "2.0",
            "id": "subscription_id",
            "error": {
                "code": -32000,
                "message": "Subscription failed"
            }
        }

    @pytest.mark.asyncio
    async def test_normalization_initialization(self, normalizer):
        """Test normalizer initialization."""
        assert normalizer is not None
        assert "sei" in normalizer.supported_sources
        assert normalizer.message_stats["processed"] == 0

    @pytest.mark.asyncio
    async def test_normalize_orderbook_message(self, normalizer, sei_orderbook_message):
        """Test normalizing Sei order book message."""
        result = await normalizer.normalize_sei_data(sei_orderbook_message)

        assert result is not None
        assert result["type"] == "orderbook"
        assert result["symbol"] == "SEI/USDC"
        assert result["source"] == "sei"
        assert "sequence" in result
        assert result["sequence"] == 12345

        # Check bids and asks
        assert len(result["bids"]) == 2
        assert len(result["asks"]) == 2
        assert result["bids"][0] == ["0.042", "1000"]
        assert result["asks"][0] == ["0.043", "1500"]

        # Check derived metrics
        assert "spread" in result
        assert "mid_price" in result
        assert result["spread"] == "0.001"
        assert result["mid_price"] == "0.0425"

    @pytest.mark.asyncio
    async def test_normalize_trade_message(self, normalizer, sei_trade_message):
        """Test normalizing Sei trade message."""
        result = await normalizer.normalize_sei_data(sei_trade_message)

        assert result is not None
        assert result["type"] == "trade"
        assert result["symbol"] == "SEI/USDC"
        assert result["source"] == "sei"
        assert result["price"] == "0.0425"
        assert result["size"] == "1000"
        assert result["side"] == "buy"
        assert result["trade_id"] == "67890"
        assert "notional_value" in result
        assert result["notional_value"] == "42.5"

    @pytest.mark.asyncio
    async def test_normalize_error_message(self, normalizer, sei_error_message):
        """Test handling Sei error message."""
        result = await normalizer.normalize_sei_data(sei_error_message)
        assert result is None

    @pytest.mark.asyncio
    async def test_normalize_invalid_message(self, normalizer):
        """Test handling invalid message format."""
        invalid_message = {"invalid": "format"}
        result = await normalizer.normalize_sei_data(invalid_message)
        assert result is None

    @pytest.mark.asyncio
    async def test_normalize_non_dict_message(self, normalizer):
        """Test handling non-dictionary message."""
        result = await normalizer.normalize_sei_data("invalid string")
        assert result is None

        result = await normalizer.normalize_sei_data(None)
        assert result is None

    @pytest.mark.asyncio
    async def test_symbol_normalization(self, normalizer):
        """Test symbol normalization."""
        # Test mapping
        assert normalizer._normalize_symbol("sei/usdc") == "SEI/USDC"
        assert normalizer._normalize_symbol("eth/usdc") == "ETH/USDC"

        # Test separator conversion
        assert normalizer._normalize_symbol("SEI-USDC") == "SEI/USDC"
        assert normalizer._normalize_symbol("SEI_USDC") == "SEI/USDC"
        assert normalizer._normalize_symbol("SEI.USDC") == "SEI/USDC"

        # Test case conversion
        assert normalizer._normalize_symbol("btc/usdc") == "BTC/USDC"

        # Test unknown symbol
        assert normalizer._normalize_symbol("UNKNOWN_TOKEN") == "UNKNOWN_TOKEN"

    @pytest.mark.asyncio
    async def test_side_normalization(self, normalizer):
        """Test side normalization."""
        assert normalizer._normalize_side("buy") == Side.BUY
        assert normalizer._normalize_side("BUY") == Side.BUY
        assert normalizer._normalize_side("bid") == Side.BUY
        assert normalizer._normalize_side("b") == Side.BUY
        assert normalizer._normalize_side("long") == Side.BUY

        assert normalizer._normalize_side("sell") == Side.SELL
        assert normalizer._normalize_side("SELL") == Side.SELL
        assert normalizer._normalize_side("ask") == Side.SELL
        assert normalizer._normalize_side("s") == Side.SELL
        assert normalizer._normalize_side("short") == Side.SELL

        # Test unknown side (defaults to BUY)
        assert normalizer._normalize_side("unknown") == Side.BUY
        assert normalizer._normalize_side("") == Side.BUY

    @pytest.mark.asyncio
    async def test_timestamp_parsing(self, normalizer):
        """Test timestamp parsing."""
        # Test ISO format
        iso_timestamp = "2024-01-01T00:00:00Z"
        parsed = normalizer._parse_timestamp(iso_timestamp)
        assert isinstance(parsed, datetime)

        # Test ISO format with timezone
        iso_tz_timestamp = "2024-01-01T00:00:00+00:00"
        parsed = normalizer._parse_timestamp(iso_tz_timestamp)
        assert isinstance(parsed, datetime)

        # Test Unix timestamp (seconds)
        unix_timestamp = 1704067200  # 2024-01-01T00:00:00Z
        parsed = normalizer._parse_timestamp(unix_timestamp)
        assert isinstance(parsed, datetime)

        # Test Unix timestamp (milliseconds)
        unix_ms_timestamp = 1704067200000
        parsed = normalizer._parse_timestamp(unix_ms_timestamp)
        assert isinstance(parsed, datetime)

        # Test string Unix timestamp
        parsed = normalizer._parse_timestamp("1704067200")
        assert isinstance(parsed, datetime)

        # Test invalid timestamp (should return current time)
        parsed = normalizer._parse_timestamp("invalid")
        assert isinstance(parsed, datetime)

        # Test None timestamp
        parsed = normalizer._parse_timestamp(None)
        assert isinstance(parsed, datetime)

    @pytest.mark.asyncio
    async def test_orderbook_level_parsing(self, normalizer):
        """Test order book level parsing."""
        # Valid levels
        levels = [["0.042", "1000"], ["0.041", "500"]]
        parsed = await normalizer._parse_order_book_levels(levels, "bid")

        assert len(parsed) == 2
        assert parsed[0].price == Decimal("0.042")
        assert parsed[0].size == Decimal("1000")

        # Invalid level format
        invalid_levels = [["0.042"], ["invalid", "data"], [0, -100]]
        parsed = await normalizer._parse_order_book_levels(invalid_levels, "ask")

        # Should skip invalid levels
        assert len(parsed) == 0

    @pytest.mark.asyncio
    async def test_attributes_extraction(self, normalizer):
        """Test Sei attributes extraction."""
        attributes = [
            {"key": "market_id", "value": "SEI/USDC"},
            {"key": "price", "value": "0.042"},
            {"key": "sequence", "value": "12345"}
        ]

        extracted = normalizer._extract_sei_attributes(attributes)

        assert extracted["market_id"] == "SEI/USDC"
        assert extracted["price"] == "0.042"
        assert extracted["sequence"] == "12345"

        # Test malformed attributes
        malformed = [
            {"invalid": "format"},
            {"key": "valid", "value": "data"}
        ]

        extracted = normalizer._extract_sei_attributes(malformed)
        assert len(extracted) == 1
        assert extracted["valid"] == "data"

    @pytest.mark.asyncio
    async def test_direct_event_message(self, normalizer):
        """Test handling direct event messages (not wrapped in result)."""
        direct_event = {
            "type": "OrderBookUpdate",
            "attributes": [
                {"key": "market_id", "value": "SEI/USDC"},
                {"key": "sequence", "value": "12345"}
            ],
            "data": {
                "bids": [["0.042", "1000"]],
                "asks": [["0.043", "1000"]]
            }
        }

        result = await normalizer.normalize_sei_data(direct_event)

        assert result is not None
        assert result["type"] == "orderbook"
        assert result["symbol"] == "SEI/USDC"

    @pytest.mark.asyncio
    async def test_statistics_tracking(self, normalizer, sei_orderbook_message, sei_trade_message):
        """Test statistics tracking."""
        initial_stats = normalizer.get_statistics()
        assert initial_stats["messages_processed"] == 0

        # Process messages
        await normalizer.normalize_sei_data(sei_orderbook_message)
        await normalizer.normalize_sei_data(sei_trade_message)

        stats = normalizer.get_statistics()
        assert stats["messages_processed"] == 2
        assert stats["orderbook_updates"] == 1
        assert stats["trade_updates"] == 1

    @pytest.mark.asyncio
    async def test_data_validation(self, normalizer):
        """Test data validation functionality."""
        # Valid orderbook data
        valid_orderbook = {
            "type": "orderbook",
            "symbol": "SEI/USDC",
            "timestamp": "2024-01-01T00:00:00Z",
            "bids": [["0.042", "1000"]],
            "asks": [["0.043", "1000"]],
            "source": "sei"
        }

        is_valid = await normalizer.validate_normalized_data(valid_orderbook)
        assert is_valid is True

        # Invalid orderbook data - missing required field
        invalid_orderbook = {
            "type": "orderbook",
            "symbol": "SEI/USDC",
            # Missing timestamp
            "bids": [],
            "asks": [],
            "source": "sei"
        }

        with pytest.raises(DataValidationError, match="Missing required field"):
            await normalizer.validate_normalized_data(invalid_orderbook)

        # Valid trade data
        valid_trade = {
            "type": "trade",
            "symbol": "SEI/USDC",
            "timestamp": "2024-01-01T00:00:00Z",
            "price": "0.042",
            "size": "1000",
            "side": "buy",
            "source": "sei"
        }

        is_valid = await normalizer.validate_normalized_data(valid_trade)
        assert is_valid is True

        # Invalid trade data - invalid price
        invalid_trade = {
            "type": "trade",
            "symbol": "SEI/USDC",
            "timestamp": "2024-01-01T00:00:00Z",
            "price": "0",
            "size": "1000",
            "side": "buy",
            "source": "sei"
        }

        with pytest.raises(DataValidationError, match="Price and size must be positive"):
            await normalizer.validate_normalized_data(invalid_trade)

    @pytest.mark.asyncio
    async def test_market_data_normalization(self, normalizer):
        """Test market data normalization."""
        market_data_event = {
            "type": "MarketData",
            "attributes": [
                {"key": "market_id", "value": "SEI/USDC"},
                {"key": "timestamp", "value": "2024-01-01T00:00:00Z"}
            ],
            "data": {
                "volume": "100000",
                "high": "0.045",
                "low": "0.040",
                "vwap": "0.0425"
            }
        }

        result = await normalizer._process_sei_event(market_data_event)

        assert result is not None
        assert result["type"] == "market_stats"
        assert result["symbol"] == "SEI/USDC"
        assert result["volume"] == "100000"
        assert result["high"] == "0.045"


class TestDataNormalizer:
    """Test legacy DataNormalizer functionality."""

    @pytest.fixture
    async def normalizer(self):
        """Create legacy normalizer instance."""
        normalizer = DataNormalizer()
        await normalizer.initialize()
        return normalizer

    @pytest.mark.asyncio
    async def test_legacy_compatibility(self, normalizer):
        """Test that legacy normalizer extends SeiDataNormalizer."""
        assert isinstance(normalizer, SeiDataNormalizer)
        assert "external_feed" in normalizer.supported_sources

    @pytest.mark.asyncio
    async def test_external_feed_normalization(self, normalizer):
        """Test external feed normalization (placeholder)."""
        result = await normalizer.normalize_external_feed_data({})
        assert result is None  # Not implemented yet


class TestNormalizerErrorHandling:
    """Test normalizer error handling and edge cases."""

    @pytest.fixture
    async def normalizer(self):
        """Create normalizer instance."""
        normalizer = SeiDataNormalizer()
        await normalizer.initialize()
        return normalizer

    @pytest.mark.asyncio
    async def test_malformed_orderbook_data(self, normalizer):
        """Test handling malformed order book data."""
        malformed_message = {
            "result": {
                "events": [
                    {
                        "type": "OrderBookUpdate",
                        "attributes": [
                            {"key": "market_id", "value": "SEI/USDC"}
                        ],
                        "data": {
                            "bids": "invalid_format",  # Should be array
                            "asks": []
                        }
                    }
                ]
            }
        }

        # Should handle gracefully and return None
        result = await normalizer.normalize_sei_data(malformed_message)
        assert result is None

    @pytest.mark.asyncio
    async def test_missing_event_data(self, normalizer):
        """Test handling messages with missing event data."""
        message_no_data = {
            "result": {
                "events": [
                    {
                        "type": "OrderBookUpdate",
                        "attributes": [
                            {"key": "market_id", "value": "SEI/USDC"}
                        ]
                        # Missing "data" field
                    }
                ]
            }
        }

        result = await normalizer.normalize_sei_data(message_no_data)
        # Should still process but with empty order book
        assert result is not None
        assert len(result["bids"]) == 0
        assert len(result["asks"]) == 0

    @pytest.mark.asyncio
    async def test_circuit_breaker_trigger(self, normalizer):
        """Test circuit breaker functionality."""
        # Process multiple invalid messages to trigger circuit breaker
        invalid_message = {"invalid": "data"}

        # This should eventually trigger the circuit breaker
        # (implementation depends on the circuit breaker configuration)
        for _ in range(15):  # More than the failure threshold
            result = await normalizer.normalize_sei_data(invalid_message)
            assert result is None

        # Check error statistics
        stats = normalizer.get_statistics()
        assert stats["messages_errors"] > 0

    @pytest.mark.asyncio
    async def test_performance_tracking(self, normalizer, sei_orderbook_message):
        """Test performance tracking with measure_latency decorator."""
        # The measure_latency decorator should track processing time
        # This is more of an integration test

        with patch('flashmm.utils.logging.PerformanceLogger') as mock_perf_logger:
            mock_instance = AsyncMock()
            mock_perf_logger.return_value = mock_instance

            result = await normalizer.normalize_sei_data(sei_orderbook_message)

            assert result is not None
            # The decorator should have logged latency metrics
            # (actual verification would depend on the logging implementation)

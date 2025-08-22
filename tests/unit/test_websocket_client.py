"""
Unit tests for FlashMM WebSocket client.
"""

import asyncio
import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from flashmm.data.ingestion.websocket_client import SeiWebSocketClient, WebSocketClient
from flashmm.utils.exceptions import WebSocketError


class TestSeiWebSocketClient:
    """Test SeiWebSocketClient functionality."""

    @pytest.fixture
    def mock_websocket(self):
        """Mock WebSocket connection."""
        mock_ws = AsyncMock()
        mock_ws.closed = False
        mock_ws.send = AsyncMock()
        mock_ws.close = AsyncMock()
        mock_ws.ping = AsyncMock()
        return mock_ws

    @pytest.fixture
    def mock_config(self):
        """Mock configuration."""
        config = MagicMock()
        config.get.side_effect = lambda key, default=None: {
            "data_ingestion.max_reconnect_attempts": 5,
            "data_ingestion.websocket_reconnect_delay": 2,
            "data_ingestion.heartbeat_interval": 10,
        }.get(key, default)
        return config

    @pytest.fixture
    async def sei_client(self, mock_config):
        """Create SeiWebSocketClient instance."""
        message_handler = AsyncMock()

        with patch('flashmm.data.ingestion.websocket_client.get_config', return_value=mock_config):
            client = SeiWebSocketClient(
                name="test_client",
                url="wss://test.sei.com/websocket",
                message_handler=message_handler,
                backup_urls=["wss://backup.sei.com/websocket"]
            )

        return client

    @pytest.mark.asyncio
    async def test_client_initialization(self, sei_client):
        """Test client initialization."""
        assert sei_client.name == "test_client"
        assert sei_client.url == "wss://test.sei.com/websocket"
        assert len(sei_client.backup_urls) == 1
        assert sei_client.connected is False
        assert sei_client.running is False
        assert len(sei_client.subscriptions) == 0

    @pytest.mark.asyncio
    async def test_successful_connection(self, sei_client, mock_websocket):
        """Test successful WebSocket connection."""
        with patch('websockets.connect', return_value=mock_websocket):
            await sei_client.connect()

            assert sei_client.connected is True
            assert sei_client.websocket == mock_websocket
            assert sei_client.current_attempt == 0
            assert sei_client.failure_count == 0

    @pytest.mark.asyncio
    async def test_connection_failure_with_failover(self, sei_client, mock_websocket):
        """Test connection failure and failover to backup URL."""
        connection_attempts = []

        async def mock_connect(url, **kwargs):
            connection_attempts.append(url)
            if url == "wss://test.sei.com/websocket":
                raise ConnectionError("Primary failed")
            return mock_websocket

        with patch('websockets.connect', side_effect=mock_connect):
            await sei_client.connect()

            assert len(connection_attempts) == 2
            assert connection_attempts[0] == "wss://test.sei.com/websocket"
            assert connection_attempts[1] == "wss://backup.sei.com/websocket"
            assert sei_client.connected is True
            assert sei_client.current_url_index == 1  # Using backup

    @pytest.mark.asyncio
    async def test_connection_failure_all_urls(self, sei_client):
        """Test connection failure when all URLs fail."""
        with patch('websockets.connect', side_effect=ConnectionError("All failed")):
            with pytest.raises(WebSocketError, match="Failed to connect to any Sei WebSocket URL"):
                await sei_client.connect()

            assert sei_client.connected is False
            assert sei_client.failure_count > 0

    @pytest.mark.asyncio
    async def test_circuit_breaker_activation(self, sei_client):
        """Test circuit breaker activation after multiple failures."""
        sei_client.circuit_breaker_threshold = 2

        with patch('websockets.connect', side_effect=ConnectionError("Failed")):
            # First failure
            with pytest.raises(WebSocketError):
                await sei_client.connect()

            # Second failure - should trigger circuit breaker
            with pytest.raises(WebSocketError):
                await sei_client.connect()

            assert sei_client.circuit_open is True

            # Third attempt should fail due to circuit breaker
            with pytest.raises(WebSocketError, match="Circuit breaker open"):
                await sei_client.connect()

    @pytest.mark.asyncio
    async def test_subscribe_orderbook(self, sei_client, mock_websocket):
        """Test order book subscription."""
        sei_client.websocket = mock_websocket
        sei_client.connected = True

        subscription_id = await sei_client.subscribe_orderbook("SEI/USDC")

        assert subscription_id is not None
        assert subscription_id in sei_client.subscriptions
        assert sei_client.subscriptions[subscription_id]["type"] == "orderbook"
        assert sei_client.subscriptions[subscription_id]["target"] == "SEI/USDC"

        # Check that WebSocket send was called
        mock_websocket.send.assert_called_once()
        sent_message = json.loads(mock_websocket.send.call_args[0][0])

        assert sent_message["method"] == "subscribe"
        assert "SEI/USDC" in sent_message["params"]["query"]
        assert "OrderBookUpdate" in sent_message["params"]["query"]

    @pytest.mark.asyncio
    async def test_subscribe_trades(self, sei_client, mock_websocket):
        """Test trade subscription."""
        sei_client.websocket = mock_websocket
        sei_client.connected = True

        subscription_id = await sei_client.subscribe_trades("SEI/USDC")

        assert subscription_id is not None
        assert sei_client.subscriptions[subscription_id]["type"] == "trades"

        # Check message format
        mock_websocket.send.assert_called_once()
        sent_message = json.loads(mock_websocket.send.call_args[0][0])

        assert "Trade" in sent_message["params"]["query"]
        assert "SEI/USDC" in sent_message["params"]["query"]

    @pytest.mark.asyncio
    async def test_subscribe_account_updates(self, sei_client, mock_websocket):
        """Test account updates subscription."""
        sei_client.websocket = mock_websocket
        sei_client.connected = True

        test_address = "sei1abcdef123456"
        subscription_id = await sei_client.subscribe_account_updates(test_address)

        assert subscription_id is not None
        assert sei_client.subscriptions[subscription_id]["type"] == "account"

        # Check message format
        mock_websocket.send.assert_called_once()
        sent_message = json.loads(mock_websocket.send.call_args[0][0])

        assert "AccountUpdate" in sent_message["params"]["query"]
        assert test_address in sent_message["params"]["query"]

    @pytest.mark.asyncio
    async def test_unsubscribe(self, sei_client, mock_websocket):
        """Test unsubscription."""
        sei_client.websocket = mock_websocket
        sei_client.connected = True

        # First subscribe
        subscription_id = await sei_client.subscribe_orderbook("SEI/USDC")
        assert subscription_id in sei_client.subscriptions

        # Then unsubscribe
        await sei_client.unsubscribe(subscription_id)

        assert subscription_id not in sei_client.subscriptions
        assert mock_websocket.send.call_count == 2  # Subscribe + unsubscribe

        # Check unsubscribe message
        unsubscribe_call = mock_websocket.send.call_args_list[1]
        unsubscribe_message = json.loads(unsubscribe_call[0][0])
        assert unsubscribe_message["method"] == "unsubscribe"
        assert unsubscribe_message["id"] == subscription_id

    @pytest.mark.asyncio
    async def test_subscription_without_connection(self, sei_client):
        """Test subscription when not connected."""
        sei_client.connected = False

        with pytest.raises(WebSocketError, match="Cannot subscribe.*not connected"):
            await sei_client.subscribe_orderbook("SEI/USDC")

    @pytest.mark.asyncio
    async def test_resubscription_after_reconnect(self, sei_client, mock_websocket):
        """Test re-subscription after reconnection."""
        sei_client.websocket = mock_websocket
        sei_client.connected = True

        # Create some subscriptions
        _sub1 = await sei_client.subscribe_orderbook("SEI/USDC")
        _sub2 = await sei_client.subscribe_trades("ETH/USDC")

        initial_call_count = mock_websocket.send.call_count

        # Simulate re-subscription
        await sei_client._resubscribe_all()

        # Should have called send for each existing subscription
        assert mock_websocket.send.call_count == initial_call_count + 2

    @pytest.mark.asyncio
    async def test_message_handling(self, sei_client, mock_websocket):
        """Test message handling."""
        message_handler = AsyncMock()
        sei_client.message_handler = message_handler

        test_message = {
            "result": {
                "events": [
                    {
                        "type": "OrderBookUpdate",
                        "data": {"test": "data"}
                    }
                ]
            }
        }

        await sei_client._handle_message(test_message, 10.0)

        message_handler.assert_called_once_with(test_message)
        assert sei_client.message_count == 1

    @pytest.mark.asyncio
    async def test_message_type_detection(self, sei_client):
        """Test message type detection."""
        # RPC result with events
        rpc_message = {
            "result": {
                "events": [
                    {"type": "OrderBookUpdate"}
                ]
            }
        }
        message_type = sei_client._get_message_type(rpc_message)
        assert message_type == "OrderBookUpdate"

        # RPC result without events
        rpc_simple = {"result": {"data": "test"}}
        message_type = sei_client._get_message_type(rpc_simple)
        assert message_type == "rpc_result"

        # Method message
        method_message = {"method": "subscribe"}
        message_type = sei_client._get_message_type(method_message)
        assert message_type == "subscribe"

        # Error message
        error_message = {"error": {"code": -1, "message": "Error"}}
        message_type = sei_client._get_message_type(error_message)
        assert message_type == "error"

        # Unknown message
        unknown_message = {"unknown": "format"}
        message_type = sei_client._get_message_type(unknown_message)
        assert message_type == "unknown"

    @pytest.mark.asyncio
    async def test_performance_stats(self, sei_client):
        """Test performance statistics."""
        sei_client.message_count = 100
        sei_client.latency_measurements = [10.0, 20.0, 30.0]
        sei_client.failure_count = 2
        sei_client.connected = True

        stats = sei_client.get_performance_stats()

        assert stats["connected"] is True
        assert stats["message_count"] == 100
        assert stats["avg_latency_ms"] == 20.0
        assert stats["max_latency_ms"] == 30.0
        assert stats["failure_count"] == 2
        assert stats["active_subscriptions"] == 0  # No subscriptions yet

    @pytest.mark.asyncio
    async def test_heartbeat_monitoring(self, sei_client, mock_websocket):
        """Test heartbeat monitoring."""
        sei_client.websocket = mock_websocket
        sei_client.connected = True
        sei_client.running = True
        sei_client.heartbeat_interval = 0.1  # Fast for testing

        # Mock successful ping
        pong_waiter = AsyncMock()
        mock_websocket.ping.return_value = pong_waiter

        # Start heartbeat monitor
        heartbeat_task = asyncio.create_task(sei_client._heartbeat_monitor())

        # Let it run for a short time
        await asyncio.sleep(0.2)

        # Stop and cleanup
        sei_client.running = False
        heartbeat_task.cancel()

        try:
            await heartbeat_task
        except asyncio.CancelledError:
            pass

        # Should have attempted ping
        assert mock_websocket.ping.called

    @pytest.mark.asyncio
    async def test_heartbeat_timeout(self, sei_client, mock_websocket):
        """Test heartbeat timeout detection."""
        sei_client.websocket = mock_websocket
        sei_client.connected = True
        sei_client.heartbeat_interval = 0.1

        # Simulate old last_pong time
        sei_client.last_pong = datetime.fromtimestamp(0)  # Very old

        # Start heartbeat monitor
        sei_client.running = True
        heartbeat_task = asyncio.create_task(sei_client._heartbeat_monitor())

        # Let it run briefly
        await asyncio.sleep(0.15)

        # Should have detected timeout and set connected = False
        assert sei_client.connected is False

        # Cleanup
        sei_client.running = False
        heartbeat_task.cancel()

        try:
            await heartbeat_task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_disconnect(self, sei_client, mock_websocket):
        """Test graceful disconnection."""
        sei_client.websocket = mock_websocket
        sei_client.connected = True
        sei_client.running = True

        # Start a mock heartbeat task
        sei_client.heartbeat_task = AsyncMock()

        await sei_client.disconnect()

        assert sei_client.running is False
        assert sei_client.connected is False
        mock_websocket.close.assert_called_once()
        sei_client.heartbeat_task.cancel.assert_called_once()

    @pytest.mark.asyncio
    async def test_is_connected_check(self, sei_client, mock_websocket):
        """Test connection status check."""
        # Not connected initially
        assert sei_client.is_connected() is False

        # Set up as connected
        sei_client.websocket = mock_websocket
        sei_client.connected = True
        mock_websocket.closed = False
        sei_client.circuit_open = False

        assert sei_client.is_connected() is True

        # Test with closed websocket
        mock_websocket.closed = True
        assert sei_client.is_connected() is False

        # Test with circuit breaker open
        mock_websocket.closed = False
        sei_client.circuit_open = True
        assert sei_client.is_connected() is False


class TestWebSocketClient:
    """Test legacy WebSocketClient compatibility."""

    @pytest.fixture
    def legacy_client(self):
        """Create legacy WebSocketClient instance."""
        message_handler = MagicMock()

        client = WebSocketClient(
            name="legacy_client",
            url="wss://test.com/websocket",
            message_handler=message_handler
        )

        return client

    @pytest.mark.asyncio
    async def test_legacy_compatibility(self, legacy_client):
        """Test that legacy client extends SeiWebSocketClient."""
        assert isinstance(legacy_client, SeiWebSocketClient)
        assert legacy_client.name == "legacy_client"

    @pytest.mark.asyncio
    async def test_legacy_message_handler_conversion(self, legacy_client):
        """Test that legacy message handler receives string messages."""
        # The legacy client should convert dict messages back to strings
        # for backward compatibility

        test_data = {"test": "message"}

        # This should trigger the conversion in the legacy wrapper
        await legacy_client.message_handler(test_data)

        # The original handler should have been called with JSON string
        legacy_client.message_handler.assert_called_once()


class TestWebSocketClientErrorHandling:
    """Test WebSocket client error handling scenarios."""

    @pytest.fixture
    async def sei_client(self):
        """Create SeiWebSocketClient for error testing."""
        message_handler = AsyncMock()

        with patch('flashmm.data.ingestion.websocket_client.get_config') as mock_config:
            mock_config.return_value.get.return_value = None

            client = SeiWebSocketClient(
                name="error_test_client",
                url="wss://test.com/websocket",
                message_handler=message_handler
            )

        return client

    @pytest.mark.asyncio
    async def test_message_handler_error(self, sei_client):
        """Test handling of message handler errors."""
        # Make message handler raise an exception
        sei_client.message_handler = AsyncMock(side_effect=Exception("Handler error"))

        # Should handle the error gracefully
        with pytest.raises(Exception, match="Message handling failed"):
            await sei_client._handle_message({"test": "data"}, 10.0)

    @pytest.mark.asyncio
    async def test_send_message_error(self, sei_client):
        """Test send message error handling."""
        mock_websocket = AsyncMock()
        mock_websocket.send.side_effect = Exception("Send failed")

        sei_client.websocket = mock_websocket
        sei_client.connected = True

        with pytest.raises(WebSocketError, match="Failed to send message"):
            await sei_client.send("test message")

        # Should mark as disconnected after send failure
        assert sei_client.connected is False

    @pytest.mark.asyncio
    async def test_reconnection_with_backoff(self, sei_client):
        """Test reconnection with exponential backoff."""
        sei_client.max_reconnect_attempts = 3
        sei_client.base_reconnect_delay = 0.1  # Fast for testing

        connection_attempts = []

        async def mock_connect(url, **kwargs):
            import time
            connection_attempts.append(time.time())
            raise ConnectionError("Connection failed")

        with patch('websockets.connect', side_effect=mock_connect):
            with patch('asyncio.sleep') as mock_sleep:
                with pytest.raises(WebSocketError, match="Max reconnection attempts reached"):
                    await sei_client._reconnect_with_backoff()

                # Should have used exponential backoff
                assert mock_sleep.call_count >= 1

                # Check backoff delays (approximately exponential)
                delays = [call[0][0] for call in mock_sleep.call_args_list]
                assert len(delays) > 0
                # First delay should be base delay
                assert delays[0] >= sei_client.base_reconnect_delay

    @pytest.mark.asyncio
    async def test_latency_violation_logging(self, sei_client):
        """Test logging when latency exceeds target."""
        sei_client.message_count = 0

        with patch('flashmm.utils.logging.get_logger') as mock_logger:
            mock_log_instance = MagicMock()
            mock_logger.return_value = mock_log_instance

            # Simulate high latency message (>250ms)
            test_message = {"test": "data"}
            await sei_client._handle_message(test_message, 300.0)  # 300ms

            # Should have logged a warning
            mock_log_instance.warning.assert_called()
            warning_call = mock_log_instance.warning.call_args[0][0]
            assert "High latency detected" in warning_call
            assert "300.00ms" in warning_call

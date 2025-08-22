"""
FlashMM Real-Time Data Streaming System

WebSocket-based real-time data streaming for dashboard updates with support for
multiple data streams, connection management, and high-performance delivery.
"""

import asyncio
import gzip
import json
import time
from collections import defaultdict, deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import websockets

from flashmm.config.settings import get_config
from flashmm.monitoring.alerts.alert_manager import AlertManager
from flashmm.monitoring.telemetry.metrics_collector import MetricsCollector
from flashmm.utils.logging import get_logger

logger = get_logger(__name__)


class StreamType(Enum):
    """Types of data streams."""
    METRICS = "metrics"
    ALERTS = "alerts"
    TRADES = "trades"
    ORDERS = "orders"
    PNL = "pnl"
    SPREADS = "spreads"
    SYSTEM = "system"
    HEARTBEAT = "heartbeat"


class MessageType(Enum):
    """WebSocket message types."""
    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"
    DATA = "data"
    BATCH = "batch"
    HEARTBEAT = "heartbeat"
    ERROR = "error"
    STATUS = "status"
    AUTH = "auth"


class CompressionType(Enum):
    """Data compression types."""
    NONE = "none"
    GZIP = "gzip"
    JSON = "json"


@dataclass
class StreamSubscription:
    """Stream subscription configuration."""
    client_id: str
    stream_types: set[StreamType]
    filters: dict[str, Any] = field(default_factory=dict)
    rate_limit_per_second: int = 100
    compression: CompressionType = CompressionType.NONE
    batch_size: int = 10
    batch_timeout_ms: int = 100
    last_activity: datetime = field(default_factory=datetime.now)
    bytes_sent: int = 0
    messages_sent: int = 0


@dataclass
class StreamMessage:
    """Stream message data structure."""
    message_type: MessageType
    stream_type: StreamType
    timestamp: datetime
    data: dict[str, Any]
    sequence_id: int = 0
    client_id: str | None = None
    compressed: bool = False


@dataclass
class ConnectionStats:
    """Connection statistics."""
    client_id: str
    connected_at: datetime
    last_seen: datetime
    messages_sent: int = 0
    bytes_sent: int = 0
    messages_received: int = 0
    bytes_received: int = 0
    subscriptions: int = 0
    errors: int = 0
    reconnections: int = 0


class DataStreamer:
    """High-performance real-time data streaming system."""

    def __init__(self, metrics_collector: MetricsCollector | None = None,
                 alert_manager: AlertManager | None = None):
        self.config = get_config()
        self.metrics_collector = metrics_collector
        self.alert_manager = alert_manager

        # Server configuration
        self.host = self.config.get("streaming.host", "localhost")
        self.port = self.config.get("streaming.port", 8765)
        self.max_connections = self.config.get("streaming.max_connections", 1000)
        self.max_message_size = self.config.get("streaming.max_message_size", 1024 * 1024)  # 1MB

        # Connection management
        self.connections: dict[str, websockets.WebSocketServerProtocol] = {}
        self.subscriptions: dict[str, StreamSubscription] = {}
        self.connection_stats: dict[str, ConnectionStats] = {}

        # Data streams and queues
        self.stream_queues: dict[str, asyncio.Queue] = {}
        self.batch_buffers: dict[str, list[StreamMessage]] = defaultdict(list)
        self.sequence_counter = 0

        # Performance tracking
        self.stats = {
            "server_start_time": datetime.now(),
            "total_connections": 0,
            "active_connections": 0,
            "messages_sent": 0,
            "bytes_sent": 0,
            "messages_processed": 0,
            "errors": 0,
            "average_latency_ms": 0.0,
            "throughput_messages_per_second": 0.0,
            "compression_ratio": 1.0
        }

        # Server state
        self.server = None
        self.running = False
        self.background_tasks: list[asyncio.Task] = []

        # Thread pool for CPU-intensive operations
        self.thread_pool = ThreadPoolExecutor(max_workers=4)

        # Rate limiting
        self.rate_limiter = defaultdict(lambda: deque(maxlen=1000))

        logger.info("DataStreamer initialized")

    async def initialize(self) -> None:
        """Initialize the data streaming server."""
        try:
            # Initialize stream queues
            for stream_type in StreamType:
                self.stream_queues[stream_type.value] = asyncio.Queue(maxsize=10000)

            # Start background tasks
            self.background_tasks.extend([
                asyncio.create_task(self._message_processor()),
                asyncio.create_task(self._batch_processor()),
                asyncio.create_task(self._heartbeat_processor()),
                asyncio.create_task(self._stats_processor()),
                asyncio.create_task(self._cleanup_processor())
            ])

            # Integration with other systems
            if self.metrics_collector:
                await self._setup_metrics_integration()

            if self.alert_manager:
                await self._setup_alerts_integration()

            logger.info("DataStreamer initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize DataStreamer: {e}")
            raise

    async def start_server(self) -> None:
        """Start the WebSocket server."""
        try:
            self.server = await websockets.serve(
                self._handle_connection,
                self.host,
                self.port,
                max_size=self.max_message_size,
                ping_interval=30,
                ping_timeout=10,
                close_timeout=10
            )

            self.running = True
            self.stats["server_start_time"] = datetime.now()

            logger.info(f"DataStreamer server started on {self.host}:{self.port}")

        except Exception as e:
            logger.error(f"Failed to start DataStreamer server: {e}")
            raise

    async def stop_server(self) -> None:
        """Stop the WebSocket server."""
        try:
            logger.info("Stopping DataStreamer server...")

            self.running = False

            # Close all connections
            for client_id, websocket in list(self.connections.items()):
                try:
                    await websocket.close()
                except Exception as e:
                    # Connection errors expected during shutdown
                    logger.debug(f"Expected connection close error for {client_id}: {e}")

            # Stop server
            if self.server:
                self.server.close()
                await self.server.wait_closed()

            # Cancel background tasks
            for task in self.background_tasks:
                task.cancel()

            await asyncio.gather(*self.background_tasks, return_exceptions=True)

            # Shutdown thread pool
            self.thread_pool.shutdown(wait=True)

            logger.info("DataStreamer server stopped")

        except Exception as e:
            logger.error(f"Error stopping DataStreamer server: {e}")

    async def _handle_connection(self, websocket, path):
        """Handle new WebSocket connection."""
        client_id = f"client_{int(time.time() * 1000)}_{id(websocket)}"

        try:
            # Register connection
            self.connections[client_id] = websocket
            self.connection_stats[client_id] = ConnectionStats(
                client_id=client_id,
                connected_at=datetime.now(),
                last_seen=datetime.now()
            )

            self.stats["total_connections"] += 1
            self.stats["active_connections"] += 1

            logger.info(f"New connection: {client_id} from {websocket.remote_address}")

            # Send welcome message
            await self._send_message(websocket, StreamMessage(
                message_type=MessageType.STATUS,
                stream_type=StreamType.SYSTEM,
                timestamp=datetime.now(),
                data={
                    "status": "connected",
                    "client_id": client_id,
                    "server_time": datetime.now().isoformat(),
                    "available_streams": [stream.value for stream in StreamType],
                    "compression_types": [comp.value for comp in CompressionType]
                },
                client_id=client_id
            ))

            # Handle messages
            async for message in websocket:
                try:
                    await self._handle_message(client_id, websocket, message)

                    # Update stats
                    stats = self.connection_stats[client_id]
                    stats.messages_received += 1
                    stats.bytes_received += len(message)
                    stats.last_seen = datetime.now()

                except Exception as e:
                    logger.error(f"Error handling message from {client_id}: {e}")
                    self.connection_stats[client_id].errors += 1

                    await self._send_error(websocket, f"Message processing error: {str(e)}")

        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Connection closed: {client_id}")

        except Exception as e:
            logger.error(f"Connection error for {client_id}: {e}")
            self.stats["errors"] += 1

        finally:
            # Cleanup connection
            await self._cleanup_connection(client_id)

    async def _handle_message(self, client_id: str, websocket, raw_message: str) -> None:
        """Handle incoming WebSocket message."""
        try:
            message = json.loads(raw_message)
            message_type = MessageType(message.get("type", ""))

            if message_type == MessageType.SUBSCRIBE:
                await self._handle_subscribe(client_id, websocket, message)

            elif message_type == MessageType.UNSUBSCRIBE:
                await self._handle_unsubscribe(client_id, websocket, message)

            elif message_type == MessageType.AUTH:
                await self._handle_auth(client_id, websocket, message)

            elif message_type == MessageType.HEARTBEAT:
                await self._handle_heartbeat(client_id, websocket, message)

            else:
                await self._send_error(websocket, f"Unknown message type: {message_type}")

        except (json.JSONDecodeError, ValueError) as e:
            await self._send_error(websocket, f"Invalid message format: {str(e)}")

        except Exception as e:
            logger.error(f"Error handling message from {client_id}: {e}")
            await self._send_error(websocket, f"Internal error: {str(e)}")

    async def _handle_subscribe(self, client_id: str, websocket, message: dict[str, Any]) -> None:
        """Handle stream subscription request."""
        try:
            stream_types = set()
            for stream_name in message.get("streams", []):
                try:
                    stream_types.add(StreamType(stream_name))
                except ValueError:
                    logger.warning(f"Unknown stream type: {stream_name}")

            if not stream_types:
                await self._send_error(websocket, "No valid stream types specified")
                return

            # Create or update subscription
            subscription = StreamSubscription(
                client_id=client_id,
                stream_types=stream_types,
                filters=message.get("filters", {}),
                rate_limit_per_second=message.get("rate_limit", 100),
                compression=CompressionType(message.get("compression", "none")),
                batch_size=message.get("batch_size", 10),
                batch_timeout_ms=message.get("batch_timeout", 100)
            )

            self.subscriptions[client_id] = subscription
            self.connection_stats[client_id].subscriptions = len(stream_types)

            # Send confirmation
            await self._send_message(websocket, StreamMessage(
                message_type=MessageType.STATUS,
                stream_type=StreamType.SYSTEM,
                timestamp=datetime.now(),
                data={
                    "status": "subscribed",
                    "streams": [stream.value for stream in stream_types],
                    "filters": subscription.filters,
                    "compression": subscription.compression.value,
                    "batch_size": subscription.batch_size
                },
                client_id=client_id
            ))

            logger.info(f"Client {client_id} subscribed to {len(stream_types)} streams")

        except Exception as e:
            logger.error(f"Error handling subscription for {client_id}: {e}")
            await self._send_error(websocket, f"Subscription error: {str(e)}")

    async def _handle_unsubscribe(self, client_id: str, websocket, message: dict[str, Any]) -> None:
        """Handle stream unsubscription request."""
        try:
            if client_id in self.subscriptions:
                stream_names = message.get("streams", [])

                if not stream_names:
                    # Unsubscribe from all
                    del self.subscriptions[client_id]
                    self.connection_stats[client_id].subscriptions = 0
                else:
                    # Unsubscribe from specific streams
                    subscription = self.subscriptions[client_id]
                    for stream_name in stream_names:
                        try:
                            stream_type = StreamType(stream_name)
                            subscription.stream_types.discard(stream_type)
                        except ValueError:
                            logger.warning(f"Unknown stream type: {stream_name}")

                    if not subscription.stream_types:
                        del self.subscriptions[client_id]

                    self.connection_stats[client_id].subscriptions = len(subscription.stream_types)

                # Send confirmation
                await self._send_message(websocket, StreamMessage(
                    message_type=MessageType.STATUS,
                    stream_type=StreamType.SYSTEM,
                    timestamp=datetime.now(),
                    data={
                        "status": "unsubscribed",
                        "streams": stream_names or "all"
                    },
                    client_id=client_id
                ))

                logger.info(f"Client {client_id} unsubscribed from streams")

        except Exception as e:
            logger.error(f"Error handling unsubscription for {client_id}: {e}")
            await self._send_error(websocket, f"Unsubscription error: {str(e)}")

    async def _handle_auth(self, client_id: str, websocket, message: dict[str, Any]) -> None:
        """Handle authentication request."""
        # Simplified authentication - in production, implement proper auth
        token = message.get("token", "")
        valid_tokens = self.config.get("streaming.auth_tokens", ["demo_token"])

        if token in valid_tokens:
            await self._send_message(websocket, StreamMessage(
                message_type=MessageType.STATUS,
                stream_type=StreamType.SYSTEM,
                timestamp=datetime.now(),
                data={"status": "authenticated", "level": "full_access"},
                client_id=client_id
            ))
        else:
            await self._send_error(websocket, "Invalid authentication token")

    async def _handle_heartbeat(self, client_id: str, websocket, message: dict[str, Any]) -> None:
        """Handle heartbeat message."""
        await self._send_message(websocket, StreamMessage(
            message_type=MessageType.HEARTBEAT,
            stream_type=StreamType.HEARTBEAT,
            timestamp=datetime.now(),
            data={"pong": message.get("ping", "")},
            client_id=client_id
        ))

    async def _cleanup_connection(self, client_id: str) -> None:
        """Clean up connection resources."""
        try:
            # Remove from active connections
            self.connections.pop(client_id, None)
            self.subscriptions.pop(client_id, None)

            # Clear batch buffer
            self.batch_buffers.pop(client_id, None)

            # Update stats
            self.stats["active_connections"] = len(self.connections)

            # Keep connection stats for a while for analysis
            if client_id in self.connection_stats:
                self.connection_stats[client_id].last_seen = datetime.now()

            logger.debug(f"Cleaned up connection: {client_id}")

        except Exception as e:
            logger.error(f"Error cleaning up connection {client_id}: {e}")

    # Data streaming methods

    async def stream_data(self, stream_type: StreamType, data: dict[str, Any],
                         filters: dict[str, Any] | None = None) -> None:
        """Stream data to subscribed clients."""
        try:
            message = StreamMessage(
                message_type=MessageType.DATA,
                stream_type=stream_type,
                timestamp=datetime.now(),
                data=data,
                sequence_id=self._get_next_sequence()
            )

            # Add to queue for processing
            queue = self.stream_queues.get(stream_type.value)
            if queue:
                try:
                    await queue.put(message)
                except asyncio.QueueFull:
                    logger.warning(f"Stream queue full for {stream_type.value}, dropping message")
                    self.stats["errors"] += 1

        except Exception as e:
            logger.error(f"Error streaming {stream_type.value} data: {e}")
            self.stats["errors"] += 1

    async def stream_metrics(self, metrics: dict[str, Any]) -> None:
        """Stream metrics data."""
        await self.stream_data(StreamType.METRICS, {
            "metrics": metrics,
            "timestamp": datetime.now().isoformat(),
            "source": "metrics_collector"
        })

    async def stream_alert(self, alert_data: dict[str, Any]) -> None:
        """Stream alert data."""
        await self.stream_data(StreamType.ALERTS, {
            "alert": alert_data,
            "timestamp": datetime.now().isoformat(),
            "source": "alert_manager"
        })

    async def stream_trade(self, trade_data: dict[str, Any]) -> None:
        """Stream trade data."""
        await self.stream_data(StreamType.TRADES, {
            "trade": trade_data,
            "timestamp": datetime.now().isoformat(),
            "source": "trading_engine"
        })

    async def stream_pnl(self, pnl_data: dict[str, Any]) -> None:
        """Stream P&L data."""
        await self.stream_data(StreamType.PNL, {
            "pnl": pnl_data,
            "timestamp": datetime.now().isoformat(),
            "source": "pnl_calculator"
        })

    async def stream_spreads(self, spread_data: dict[str, Any]) -> None:
        """Stream spread data."""
        await self.stream_data(StreamType.SPREADS, {
            "spreads": spread_data,
            "timestamp": datetime.now().isoformat(),
            "source": "spread_analyzer"
        })

    # Background processors

    async def _message_processor(self) -> None:
        """Process messages from stream queues."""
        while self.running:
            try:
                # Process messages from all stream queues
                for stream_type_name, queue in self.stream_queues.items():
                    try:
                        # Process multiple messages at once for efficiency
                        messages = []
                        for _ in range(50):  # Process up to 50 messages at once
                            try:
                                message = queue.get_nowait()
                                messages.append(message)
                            except asyncio.QueueEmpty:
                                break

                        if messages:
                            await self._distribute_messages(messages)
                            self.stats["messages_processed"] += len(messages)

                    except Exception as e:
                        logger.error(f"Error processing {stream_type_name} queue: {e}")

                # Small delay to prevent CPU spinning
                await asyncio.sleep(0.001)  # 1ms

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in message processor: {e}")
                await asyncio.sleep(1)

    async def _distribute_messages(self, messages: list[StreamMessage]) -> None:
        """Distribute messages to subscribed clients."""
        try:
            for client_id, subscription in list(self.subscriptions.items()):
                if client_id not in self.connections:
                    continue

                websocket = self.connections[client_id]

                # Filter messages for this subscription
                relevant_messages = [
                    msg for msg in messages
                    if msg.stream_type in subscription.stream_types
                    and self._message_matches_filters(msg, subscription.filters)
                ]

                if not relevant_messages:
                    continue

                # Check rate limiting
                if not self._check_rate_limit(client_id, len(relevant_messages)):
                    continue

                try:
                    if subscription.batch_size > 1:
                        # Add to batch buffer
                        self.batch_buffers[client_id].extend(relevant_messages)
                    else:
                        # Send immediately
                        for message in relevant_messages:
                            await self._send_message(websocket, message, subscription.compression)

                except websockets.exceptions.ConnectionClosed:
                    await self._cleanup_connection(client_id)
                except Exception as e:
                    logger.error(f"Error sending messages to {client_id}: {e}")
                    self.connection_stats[client_id].errors += 1

        except Exception as e:
            logger.error(f"Error distributing messages: {e}")

    async def _batch_processor(self) -> None:
        """Process batched messages."""
        while self.running:
            try:
                current_time = datetime.now()

                for client_id in list(self.batch_buffers.keys()):
                    if client_id not in self.subscriptions or client_id not in self.connections:
                        self.batch_buffers.pop(client_id, None)
                        continue

                    subscription = self.subscriptions[client_id]
                    websocket = self.connections[client_id]
                    buffer = self.batch_buffers[client_id]

                    if not buffer:
                        continue

                    # Check if batch should be sent
                    should_send = (
                        len(buffer) >= subscription.batch_size or
                        (buffer and (current_time - buffer[0].timestamp).total_seconds() * 1000 >= subscription.batch_timeout_ms)
                    )

                    if should_send:
                        try:
                            # Create batch message
                            batch_message = StreamMessage(
                                message_type=MessageType.BATCH,
                                stream_type=StreamType.SYSTEM,
                                timestamp=current_time,
                                data={
                                    "messages": [asdict(msg) for msg in buffer],
                                    "count": len(buffer)
                                },
                                client_id=client_id
                            )

                            await self._send_message(websocket, batch_message, subscription.compression)

                            # Clear buffer
                            self.batch_buffers[client_id] = []

                        except websockets.exceptions.ConnectionClosed:
                            await self._cleanup_connection(client_id)
                        except Exception as e:
                            logger.error(f"Error sending batch to {client_id}: {e}")
                            self.connection_stats[client_id].errors += 1

                await asyncio.sleep(0.01)  # 10ms

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in batch processor: {e}")
                await asyncio.sleep(1)

    async def _heartbeat_processor(self) -> None:
        """Send periodic heartbeats to clients."""
        while self.running:
            try:
                heartbeat_message = StreamMessage(
                    message_type=MessageType.HEARTBEAT,
                    stream_type=StreamType.HEARTBEAT,
                    timestamp=datetime.now(),
                    data={
                        "server_time": datetime.now().isoformat(),
                        "active_connections": len(self.connections),
                        "server_uptime": (datetime.now() - self.stats["server_start_time"]).total_seconds()
                    }
                )

                # Send to all connected clients
                for client_id, websocket in list(self.connections.items()):
                    try:
                        await self._send_message(websocket, heartbeat_message)
                    except websockets.exceptions.ConnectionClosed:
                        await self._cleanup_connection(client_id)
                    except Exception as e:
                        logger.debug(f"Heartbeat failed for {client_id}: {e}")

                await asyncio.sleep(30)  # Send heartbeat every 30 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in heartbeat processor: {e}")
                await asyncio.sleep(30)

    async def _stats_processor(self) -> None:
        """Update streaming statistics."""
        while self.running:
            try:
                # Calculate throughput
                current_time = datetime.now()
                if hasattr(self, '_last_stats_time'):
                    time_diff = (current_time - self._last_stats_time).total_seconds()
                    if time_diff > 0:
                        messages_diff = self.stats["messages_sent"] - getattr(self, '_last_message_count', 0)
                        self.stats["throughput_messages_per_second"] = messages_diff / time_diff

                self._last_stats_time = current_time
                self._last_message_count = self.stats["messages_sent"]

                # Update active connections
                self.stats["active_connections"] = len(self.connections)

                # Log stats periodically
                if int(current_time.timestamp()) % 60 == 0:  # Every minute
                    logger.info(f"Streaming stats: {self.stats['active_connections']} connections, "
                              f"{self.stats['throughput_messages_per_second']:.1f} msg/s, "
                              f"{self.stats['messages_sent']} total messages")

                await asyncio.sleep(1)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in stats processor: {e}")
                await asyncio.sleep(5)

    async def _cleanup_processor(self) -> None:
        """Clean up old connection stats and data."""
        while self.running:
            try:
                current_time = datetime.now()
                cutoff_time = current_time - timedelta(hours=1)

                # Clean up old connection stats
                old_connections = [
                    client_id for client_id, stats in self.connection_stats.items()
                    if client_id not in self.connections and stats.last_seen < cutoff_time
                ]

                for client_id in old_connections:
                    del self.connection_stats[client_id]

                # Clean up orphaned batch buffers
                orphaned_buffers = [
                    client_id for client_id in self.batch_buffers.keys()
                    if client_id not in self.connections
                ]

                for client_id in orphaned_buffers:
                    del self.batch_buffers[client_id]

                if old_connections or orphaned_buffers:
                    logger.debug(f"Cleaned up {len(old_connections)} old connections and {len(orphaned_buffers)} orphaned buffers")

                await asyncio.sleep(300)  # Clean up every 5 minutes

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup processor: {e}")
                await asyncio.sleep(300)

    # Helper methods

    async def _send_message(self, websocket, message: StreamMessage,
                          compression: CompressionType = CompressionType.NONE) -> None:
        """Send message to WebSocket client."""
        try:
            start_time = time.time()

            # Serialize message
            data = {
                "type": message.message_type.value,
                "stream": message.stream_type.value,
                "timestamp": message.timestamp.isoformat(),
                "data": message.data,
                "sequence": message.sequence_id
            }

            if message.client_id:
                data["client_id"] = message.client_id

            # Convert to JSON
            json_data = json.dumps(data, separators=(',', ':'))

            # Apply compression if requested
            if compression == CompressionType.GZIP:
                compressed_data = await asyncio.get_event_loop().run_in_executor(
                    self.thread_pool, gzip.compress, json_data.encode('utf-8')
                )
                # Send binary data with compression flag
                await websocket.send(b'\x01' + compressed_data)  # \x01 indicates gzip
                message.compressed = True
                self.stats["compression_ratio"] = len(json_data) / len(compressed_data)
            else:
                # Send as text
                await websocket.send(json_data)

            # Update statistics
            message_size = len(json_data)
            self.stats["messages_sent"] += 1
            self.stats["bytes_sent"] += message_size

            # Update connection stats
            if message.client_id and message.client_id in self.connection_stats:
                stats = self.connection_stats[message.client_id]
                stats.messages_sent += 1
                stats.bytes_sent += message_size

            # Update latency stats
            latency_ms = (time.time() - start_time) * 1000
            current_avg = self.stats["average_latency_ms"]
            total_messages = self.stats["messages_sent"]
            self.stats["average_latency_ms"] = ((current_avg * (total_messages - 1)) + latency_ms) / total_messages

        except Exception as e:
            logger.error(f"Error sending message: {e}")
            self.stats["errors"] += 1
            raise

    async def _send_error(self, websocket, error_message: str) -> None:
        """Send error message to client."""
        try:
            error_msg = StreamMessage(
                message_type=MessageType.ERROR,
                stream_type=StreamType.SYSTEM,
                timestamp=datetime.now(),
                data={"error": error_message}
            )
            await self._send_message(websocket, error_msg)
        except Exception as e:
            logger.error(f"Error sending error message: {e}")

    def _message_matches_filters(self, message: StreamMessage, filters: dict[str, Any]) -> bool:
        """Check if message matches subscription filters."""
        if not filters:
            return True

        try:
            # Check market filter
            if "markets" in filters:
                market = message.data.get("market") or message.data.get("symbol")
                if market and market not in filters["markets"]:
                    return False

            # Check severity filter (for alerts)
            if "severity" in filters and message.stream_type == StreamType.ALERTS:
                alert_severity = message.data.get("alert", {}).get("severity")
                if alert_severity and alert_severity not in filters["severity"]:
                    return False

            # Check minimum value filter
            if "min_value" in filters:
                value = message.data.get("value") or message.data.get("amount")
                if value is not None and value < filters["min_value"]:
                    return False

            return True

        except Exception as e:
            logger.error(f"Error applying filters: {e}")
            return True  # Default to allowing message on filter error

    def _check_rate_limit(self, client_id: str, message_count: int) -> bool:
        """Check if client is within rate limits."""
        try:
            subscription = self.subscriptions.get(client_id)
            if not subscription:
                return False

            current_time = datetime.now()
            rate_window = self.rate_limiter[client_id]

            # Remove old entries outside the 1-second window
            cutoff_time = current_time - timedelta(seconds=1)
            while rate_window and rate_window[0] < cutoff_time:
                rate_window.popleft()

            # Check if adding new messages would exceed rate limit
            if len(rate_window) + message_count > subscription.rate_limit_per_second:
                return False

            # Add timestamps for new messages
            for _ in range(message_count):
                rate_window.append(current_time)

            return True

        except Exception as e:
            logger.error(f"Error checking rate limit for {client_id}: {e}")
            return True  # Default to allowing on error

    def _get_next_sequence(self) -> int:
        """Get next sequence number."""
        self.sequence_counter += 1
        return self.sequence_counter

    # Integration methods

    async def _setup_metrics_integration(self) -> None:
        """Setup integration with metrics collector."""
        try:
            if not self.metrics_collector:
                logger.warning("No metrics collector provided for integration")
                return

            # Add callback to metrics collector to stream metrics
            def metrics_callback(metrics_data):
                asyncio.create_task(self.stream_metrics(metrics_data))

            if hasattr(self.metrics_collector, 'add_callback'):
                self.metrics_collector.add_callback(metrics_callback)
            else:
                logger.warning("Metrics collector does not support callbacks")

            logger.info("Metrics integration setup complete")

        except Exception as e:
            logger.error(f"Error setting up metrics integration: {e}")

    async def _setup_alerts_integration(self) -> None:
        """Setup integration with alert manager."""
        try:
            if not self.alert_manager:
                logger.warning("No alert manager provided for integration")
                return

            # Add callback to alert manager to stream alerts
            def alert_callback(alert):
                alert_data = asdict(alert) if hasattr(alert, '__dict__') else alert
                asyncio.create_task(self.stream_alert(alert_data))

            if hasattr(self.alert_manager, 'add_alert_handler'):
                self.alert_manager.add_alert_handler(alert_callback)
            else:
                logger.warning("Alert manager does not support alert handlers")

            logger.info("Alerts integration setup complete")

        except Exception as e:
            logger.error(f"Error setting up alerts integration: {e}")

    # Public API methods

    def get_connection_stats(self) -> dict[str, Any]:
        """Get connection statistics."""
        return {
            "active_connections": len(self.connections),
            "total_connections": self.stats["total_connections"],
            "active_subscriptions": len(self.subscriptions),
            "connection_details": {
                client_id: {
                    "connected_at": stats.connected_at.isoformat(),
                    "last_seen": stats.last_seen.isoformat(),
                    "messages_sent": stats.messages_sent,
                    "bytes_sent": stats.bytes_sent,
                    "subscriptions": stats.subscriptions,
                    "errors": stats.errors
                }
                for client_id, stats in self.connection_stats.items()
                if client_id in self.connections
            }
        }

    def get_streaming_stats(self) -> dict[str, Any]:
        """Get streaming performance statistics."""
        return {
            **self.stats,
            "server_uptime_seconds": (datetime.now() - self.stats["server_start_time"]).total_seconds(),
            "queue_sizes": {
                stream_type: queue.qsize()
                for stream_type, queue in self.stream_queues.items()
            },
            "batch_buffer_sizes": {
                client_id: len(buffer)
                for client_id, buffer in self.batch_buffers.items()
            }
        }

    def get_client_info(self, client_id: str) -> dict[str, Any] | None:
        """Get information about a specific client."""
        if client_id not in self.connection_stats:
            return None

        stats = self.connection_stats[client_id]
        subscription = self.subscriptions.get(client_id)

        return {
            "client_id": client_id,
            "connected": client_id in self.connections,
            "connection_stats": asdict(stats),
            "subscription": {
                "streams": [stream.value for stream in subscription.stream_types] if subscription else [],
                "filters": subscription.filters if subscription else {},
                "rate_limit": subscription.rate_limit_per_second if subscription else 0,
                "compression": subscription.compression.value if subscription else "none",
                "batch_size": subscription.batch_size if subscription else 1
            } if subscription else None,
            "batch_buffer_size": len(self.batch_buffers.get(client_id, [])),
            "rate_limit_status": {
                "current_rate": len(self.rate_limiter.get(client_id, [])),
                "limit": subscription.rate_limit_per_second if subscription else 0
            }
        }

    async def disconnect_client(self, client_id: str, reason: str = "Server disconnect") -> bool:
        """Disconnect a specific client."""
        try:
            if client_id in self.connections:
                websocket = self.connections[client_id]

                # Send disconnect message
                await self._send_message(websocket, StreamMessage(
                    message_type=MessageType.STATUS,
                    stream_type=StreamType.SYSTEM,
                    timestamp=datetime.now(),
                    data={"status": "disconnecting", "reason": reason},
                    client_id=client_id
                ))

                # Close connection
                await websocket.close()
                return True

            return False

        except Exception as e:
            logger.error(f"Error disconnecting client {client_id}: {e}")
            return False

    async def broadcast_system_message(self, message: str, message_type: str = "info") -> int:
        """Broadcast system message to all connected clients."""
        try:
            system_message = StreamMessage(
                message_type=MessageType.STATUS,
                stream_type=StreamType.SYSTEM,
                timestamp=datetime.now(),
                data={
                    "status": "system_message",
                    "message": message,
                    "type": message_type
                }
            )

            sent_count = 0
            for client_id, websocket in list(self.connections.items()):
                try:
                    await self._send_message(websocket, system_message)
                    sent_count += 1
                except Exception as e:
                    logger.debug(f"Failed to send system message to {client_id}: {e}")

            return sent_count

        except Exception as e:
            logger.error(f"Error broadcasting system message: {e}")
            return 0

    def set_rate_limit(self, client_id: str, rate_limit: int) -> bool:
        """Set rate limit for a specific client."""
        try:
            if client_id in self.subscriptions:
                self.subscriptions[client_id].rate_limit_per_second = rate_limit
                return True
            return False
        except Exception as e:
            logger.error(f"Error setting rate limit for {client_id}: {e}")
            return False

    def get_queue_health(self) -> dict[str, Any]:
        """Get health status of streaming queues."""
        queue_health = {}

        for stream_type, queue in self.stream_queues.items():
            queue_size = queue.qsize()
            max_size = queue.maxsize

            queue_health[stream_type] = {
                "size": queue_size,
                "max_size": max_size,
                "utilization": queue_size / max_size if max_size > 0 else 0,
                "status": "healthy" if queue_size < max_size * 0.8 else "warning" if queue_size < max_size * 0.95 else "critical"
            }

        return queue_health

    async def flush_all_queues(self) -> dict[str, int]:
        """Flush all pending messages in queues."""
        flushed_counts = {}

        try:
            for stream_type, queue in self.stream_queues.items():
                count = 0
                while not queue.empty():
                    try:
                        queue.get_nowait()
                        count += 1
                    except asyncio.QueueEmpty:
                        break
                flushed_counts[stream_type] = count

            # Also flush batch buffers
            for client_id, buffer in self.batch_buffers.items():
                if buffer:
                    flushed_counts[f"batch_buffer_{client_id}"] = len(buffer)
                    buffer.clear()

            logger.info(f"Flushed queues: {flushed_counts}")
            return flushed_counts

        except Exception as e:
            logger.error(f"Error flushing queues: {e}")
            return {}

    def __repr__(self) -> str:
        """String representation of DataStreamer."""
        return (f"DataStreamer(active_connections={len(self.connections)}, "
                f"subscriptions={len(self.subscriptions)}, "
                f"running={self.running}, "
                f"host={self.host}:{self.port})")


# Utility functions for client integration

def create_websocket_client(server_url: str, auth_token: str | None = None) -> 'StreamingClient':
    """Create a WebSocket client for connecting to the data streamer."""
    return StreamingClient(server_url, auth_token)


class StreamingClient:
    """Client for connecting to FlashMM data streaming server."""

    def __init__(self, server_url: str, auth_token: str | None = None):
        self.server_url = server_url
        self.auth_token = auth_token
        self.websocket = None
        self.connected = False
        self.subscriptions: set[str] = set()
        self.message_handlers: dict[str, list[Callable]] = defaultdict(list)

    async def connect(self) -> bool:
        """Connect to the streaming server."""
        try:
            self.websocket = await websockets.connect(self.server_url)
            self.connected = True

            # Authenticate if token provided
            if self.auth_token:
                await self._send_message({
                    "type": "auth",
                    "token": self.auth_token
                })

            # Start message handler
            asyncio.create_task(self._message_handler())

            return True

        except Exception as e:
            logger.error(f"Failed to connect to streaming server: {e}")
            return False

    async def disconnect(self) -> None:
        """Disconnect from the streaming server."""
        try:
            if self.websocket:
                await self.websocket.close()
            self.connected = False
        except Exception as e:
            logger.error(f"Error disconnecting: {e}")

    async def subscribe(self, streams: list[str], filters: dict[str, Any] | None = None,
                       compression: str = "none", batch_size: int = 1) -> bool:
        """Subscribe to data streams."""
        try:
            await self._send_message({
                "type": "subscribe",
                "streams": streams,
                "filters": filters or {},
                "compression": compression,
                "batch_size": batch_size
            })

            self.subscriptions.update(streams)
            return True

        except Exception as e:
            logger.error(f"Error subscribing to streams: {e}")
            return False

    async def unsubscribe(self, streams: list[str] | None = None) -> bool:
        """Unsubscribe from data streams."""
        try:
            await self._send_message({
                "type": "unsubscribe",
                "streams": streams or []
            })

            if streams:
                self.subscriptions.difference_update(streams)
            else:
                self.subscriptions.clear()

            return True

        except Exception as e:
            logger.error(f"Error unsubscribing from streams: {e}")
            return False

    def add_message_handler(self, stream_type: str, handler: Callable) -> None:
        """Add a message handler for a specific stream type."""
        self.message_handlers[stream_type].append(handler)

    def add_event_handler(self, handler: Callable) -> None:
        """Add an event handler (compatibility method)."""
        self.add_message_handler("all", handler)

    async def _send_message(self, message: dict[str, Any]) -> None:
        """Send message to server."""
        if self.websocket and self.connected:
            await self.websocket.send(json.dumps(message))

    async def _message_handler(self) -> None:
        """Handle incoming messages from server."""
        try:
            if not self.websocket:
                logger.error("WebSocket not initialized for message handling")
                return

            async for message in self.websocket:
                try:
                    if isinstance(message, bytes):
                        # Handle compressed messages
                        if message.startswith(b'\x01'):
                            message = gzip.decompress(message[1:]).decode('utf-8')
                        else:
                            message = message.decode('utf-8')

                    data = json.loads(message)
                    stream_type = data.get("stream", "")

                    # Call registered handlers
                    for handler in self.message_handlers.get(stream_type, []):
                        try:
                            if asyncio.iscoroutinefunction(handler):
                                await handler(data)
                            else:
                                handler(data)
                        except Exception as e:
                            logger.error(f"Error in message handler: {e}")

                except Exception as e:
                    logger.error(f"Error processing message: {e}")

        except websockets.exceptions.ConnectionClosed:
            self.connected = False
            logger.info("Connection to streaming server closed")
        except Exception as e:
            logger.error(f"Error in message handler: {e}")
            self.connected = False

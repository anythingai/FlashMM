# FlashMM External Service Integrations

## Overview
FlashMM integrates with several external services to provide market-making functionality, data ingestion, monitoring, and social media updates. This document details the integration architecture, API specifications, error handling, and failover strategies for each external service.

## Sei Blockchain Integration

### Connection Architecture
```
FlashMM → [WebSocket + REST] → Sei RPC Node → Sei Network
         ↓
    [Backup Nodes for Failover]
```

### WebSocket Integration
**Purpose**: Real-time order book and trade data ingestion  
**Protocol**: WebSocket over TLS  
**Target Latency**: < 250ms round-trip

#### Connection Configuration
```python
class SeiWebSocketConfig:
    primary_url: str = "wss://rpc.sei-apis.com/websocket"
    backup_urls: List[str] = [
        "wss://sei-testnet-rpc.polkachu.com/websocket",
        "wss://sei-rpc.lavenderfive.com/websocket"
    ]
    
    # Connection settings
    ping_interval: int = 30  # seconds
    ping_timeout: int = 10   # seconds
    max_reconnect_attempts: int = 5
    reconnect_delay: int = 2  # seconds, exponential backoff
    
    # Subscription settings
    max_subscriptions: int = 10
    buffer_size: int = 8192
```

#### Subscription Management
```python
# Order book subscription
subscription_request = {
    "jsonrpc": "2.0",
    "method": "subscribe",
    "id": 1,
    "params": {
        "query": "tm.event='OrderBookUpdate' AND market_id='SEI/USDC'"
    }
}

# Trade subscription
trade_subscription = {
    "jsonrpc": "2.0", 
    "method": "subscribe",
    "id": 2,
    "params": {
        "query": "tm.event='Trade' AND market_id='SEI/USDC'"
    }
}
```

#### Message Handling
```python
class SeiWebSocketHandler:
    async def handle_message(self, message: Dict) -> None:
        """Process incoming WebSocket messages"""
        
        if message.get("result", {}).get("events"):
            events = message["result"]["events"]
            
            for event in events:
                event_type = event.get("type")
                
                if event_type == "OrderBookUpdate":
                    await self._handle_orderbook_update(event)
                elif event_type == "Trade":
                    await self._handle_trade_event(event)
                elif event_type == "MarketStatus":
                    await self._handle_market_status(event)
    
    async def _handle_orderbook_update(self, event: Dict) -> None:
        """Process order book updates"""
        try:
            orderbook = OrderBook(
                symbol=event["attributes"]["market_id"],
                timestamp=datetime.fromisoformat(event["attributes"]["timestamp"]),
                bids=json.loads(event["attributes"]["bids"]),
                asks=json.loads(event["attributes"]["asks"]),
                sequence=int(event["attributes"]["sequence"])
            )
            
            # Publish to internal data pipeline
            await self.publisher.publish("market_data", orderbook)
            
        except Exception as e:
            logger.error(f"Failed to process orderbook update: {e}")
            await self.metrics.increment("sei_websocket_errors")
```

### REST API Integration
**Purpose**: Order placement, cancellation, and account queries  
**Base URL**: `https://rpc.sei-apis.com`  
**Authentication**: None required for queries, wallet signature for transactions

#### Account Balance Queries
```python
class SeiRestClient:
    async def get_account_balance(self, address: str) -> Dict[str, float]:
        """Get account token balances"""
        
        url = f"{self.base_url}/cosmos/bank/v1beta1/balances/{address}"
        
        async with self.session.get(url) as response:
            data = await response.json()
            
            balances = {}
            for balance in data.get("balances", []):
                denom = balance["denom"]
                amount = float(balance["amount"]) / 1e6  # Adjust for decimals
                balances[denom] = amount
                
            return balances
    
    async def get_orders(self, address: str, market_id: str) -> List[Dict]:
        """Get active orders for account"""
        
        url = f"{self.base_url}/sei-protocol/seichain/dex/get_orders_by_user"
        params = {
            "contractAddr": self.clob_contract,
            "account": address,
            "marketId": market_id
        }
        
        async with self.session.get(url, params=params) as response:
            data = await response.json()
            return data.get("orders", [])
```

### Error Handling & Failover
```python
class SeiConnectionManager:
    def __init__(self):
        self.primary_ws = None
        self.backup_connections = []
        self.current_connection_index = 0
        self.connection_status = "disconnected"
        
    async def ensure_connection(self) -> None:
        """Ensure we have an active WebSocket connection"""
        
        if self.primary_ws and not self.primary_ws.closed:
            return
            
        # Try primary connection first
        try:
            self.primary_ws = await self._connect_websocket(self.config.primary_url)
            self.connection_status = "connected"
            return
        except Exception as e:
            logger.warning(f"Primary WebSocket failed: {e}")
        
        # Failover to backup connections
        for i, backup_url in enumerate(self.config.backup_urls):
            try:
                self.primary_ws = await self._connect_websocket(backup_url)
                self.current_connection_index = i + 1
                self.connection_status = "connected"
                logger.info(f"Failed over to backup connection {i+1}")
                return
            except Exception as e:
                logger.warning(f"Backup WebSocket {i+1} failed: {e}")
        
        # All connections failed
        self.connection_status = "failed"
        raise ConnectionError("All Sei WebSocket connections failed")
```

## Cambrian SDK Integration

### SDK Configuration
```python
from cambrian_sdk import CambrianClient

class CambrianIntegration:
    def __init__(self):
        self.client = CambrianClient(
            api_key=config.get("cambrian.api_key"),
            secret_key=config.get("cambrian.secret_key"),
            base_url=config.get("cambrian.base_url", "https://api.cambrian.com"),
            testnet=config.get("cambrian.testnet", True)
        )
        
        # Rate limiting
        self.rate_limiter = AsyncLimiter(10, 1)  # 10 requests per second
        self.order_semaphore = asyncio.Semaphore(5)  # Max 5 concurrent orders
```

### Order Management Integration
```python
class OrderRouter:
    async def place_order(self, order: Order) -> str:
        """Place order via Cambrian SDK"""
        
        async with self.rate_limiter, self.order_semaphore:
            try:
                # Prepare order payload
                order_request = {
                    "market_id": order.symbol,
                    "side": order.side.lower(),
                    "order_type": order.order_type.lower(),
                    "price": str(order.price),
                    "quantity": str(order.size),
                    "time_in_force": order.time_in_force
                }
                
                # Submit order
                response = await self.client.place_order(**order_request)
                
                # Track order
                order_id = response["order_id"]
                await self._track_order(order_id, order)
                
                return order_id
                
            except CambrianAPIError as e:
                logger.error(f"Cambrian API error: {e}")
                await self.metrics.increment("cambrian_api_errors")
                raise
            except Exception as e:
                logger.error(f"Order placement failed: {e}")
                await self.metrics.increment("order_placement_errors")
                raise
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel existing order"""
        
        async with self.rate_limiter:
            try:
                response = await self.client.cancel_order(order_id)
                success = response.get("status") == "cancelled"
                
                if success:
                    await self.metrics.increment("orders_cancelled")
                
                return success
                
            except Exception as e:
                logger.error(f"Order cancellation failed: {e}")
                await self.metrics.increment("order_cancellation_errors")  
                return False
    
    async def get_order_status(self, order_id: str) -> Dict:
        """Get current order status"""
        
        async with self.rate_limiter:
            try:
                return await self.client.get_order(order_id)
            except Exception as e:
                logger.error(f"Order status query failed: {e}")
                return {"status": "unknown", "error": str(e)}
```

### Fill Handling
```python
class FillHandler:
    async def process_fill(self, fill_data: Dict) -> None:
        """Process order fill from Cambrian"""
        
        try:
            fill = Fill(
                fill_id=fill_data["fill_id"],
                order_id=fill_data["order_id"],
                symbol=fill_data["market_id"],
                side=fill_data["side"],
                price=Decimal(fill_data["price"]),
                size=Decimal(fill_data["quantity"]),
                fee=Decimal(fill_data["fee"]),
                timestamp=datetime.fromisoformat(fill_data["timestamp"])
            )
            
            # Update position tracker
            await self.position_tracker.update_position(fill)
            
            # Update PnL calculation
            await self.pnl_tracker.record_fill(fill)
            
            # Publish metrics
            await self.metrics.record_fill(fill)
            
            # Notify risk manager
            await self.risk_manager.handle_fill(fill)
            
        except Exception as e:
            logger.error(f"Fill processing failed: {e}")
            await self.metrics.increment("fill_processing_errors")
```

## Grafana Cloud Integration

### Dashboard Configuration
```python
class GrafanaIntegration:
    def __init__(self):
        self.api_key = config.get("grafana.api_key")
        self.org_id = config.get("grafana.org_id")
        self.base_url = config.get("grafana.url", "https://flashmm.grafana.net")
        
        self.session = aiohttp.ClientSession(
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
        )
```

### Dashboard Provisioning
```python
# Dashboard configuration JSON
TRADING_DASHBOARD = {
    "dashboard": {
        "id": None,
        "title": "FlashMM Trading Dashboard",
        "tags": ["flashmm", "trading"],
        "timezone": "UTC",
        "panels": [
            {
                "id": 1,
                "title": "Spread Improvement",
                "type": "stat",
                "targets": [
                    {
                        "expr": "rate(flashmm_spread_improvement_bps[1m])",
                        "legendFormat": "{{market}}"
                    }
                ]
            },
            {
                "id": 2,
                "title": "PnL Over Time",
                "type": "timeseries",
                "targets": [
                    {
                        "expr": "flashmm_pnl_usdc",
                        "legendFormat": "Total PnL"
                    }
                ]
            },
            {
                "id": 3,
                "title": "Order Fill Rate",
                "type": "timeseries",
                "targets": [
                    {
                        "expr": "rate(flashmm_orders_filled_total[5m])",
                        "legendFormat": "Fills per second"
                    }
                ]
            }
        ]
    },
    "overwrite": True
}

class DashboardManager:
    async def create_dashboard(self) -> str:
        """Create or update Grafana dashboard"""
        
        url = f"{self.base_url}/api/dashboards/db"
        
        async with self.session.post(url, json=TRADING_DASHBOARD) as response:
            if response.status == 200:
                data = await response.json()
                dashboard_url = f"{self.base_url}/d/{data['uid']}"
                logger.info(f"Dashboard created: {dashboard_url}")
                return dashboard_url
            else:
                error = await response.text()
                logger.error(f"Dashboard creation failed: {error}")
                raise GrafanaAPIError(f"Failed to create dashboard: {error}")
```

### Alert Configuration
```python
ALERT_RULES = [
    {
        "alert": {
            "name": "High Drawdown Alert",
            "message": "FlashMM experiencing high drawdown",
            "frequency": "10s",
            "conditions": [
                {
                    "query": {
                        "queryType": "",
                        "refId": "A",
                        "datasourceUid": "prometheus_uid",
                        "model": {
                            "expr": "flashmm_drawdown_percent > 3",
                            "refId": "A"
                        }
                    },
                    "reducer": {
                        "type": "last",
                        "params": []
                    },
                    "evaluator": {
                        "params": [3],
                        "type": "gt"
                    }
                }
            ],
            "notifications": [
                {
                    "uid": "webhook_notification_uid"
                }
            ]
        }
    }
]

class AlertManager:
    async def setup_alerts(self) -> None:
        """Configure Grafana alerts"""
        
        for alert_rule in ALERT_RULES:
            url = f"{self.base_url}/api/alerts"
            
            async with self.session.post(url, json=alert_rule) as response:
                if response.status != 200:
                    error = await response.text()
                    logger.error(f"Alert setup failed: {error}")
```

## Twitter/X Integration

### Authentication Setup
```python
import tweepy

class TwitterIntegration:
    def __init__(self):
        self.client = tweepy.Client(
            bearer_token=config.get("twitter.bearer_token"),
            consumer_key=config.get("twitter.api_key"),
            consumer_secret=config.get("twitter.api_secret"),
            access_token=config.get("twitter.access_token"),
            access_token_secret=config.get("twitter.access_secret"),
            wait_on_rate_limit=True
        )
        
        self.tweet_queue = asyncio.Queue(maxsize=10)
        self.last_tweet_time = None
        self.min_tweet_interval = 3600  # 1 hour minimum between tweets
```

### Performance Report Generation
```python
class PerformanceReporter:
    async def generate_hourly_report(self) -> str:
        """Generate hourly performance summary"""
        
        # Fetch metrics from last hour
        metrics = await self.metrics_collector.get_hourly_metrics()
        
        # Generate performance chart
        chart_path = await self._generate_performance_chart(metrics)
        
        # Create tweet text
        tweet_text = f"""
🤖 FlashMM Hourly Update

📊 Spread Improvement: {metrics['spread_improvement']:.1f}%
💰 PnL: ${metrics['pnl_usdc']:.2f}
📈 Volume: ${metrics['volume_usdc']:,.0f}
🎯 Fill Rate: {metrics['fill_rate']:.1f}%
⚡ Uptime: {metrics['uptime_percent']:.1f}%

#FlashMM #MarketMaking #Sei #DeFi
"""
        
        return tweet_text, chart_path
    
    async def _generate_performance_chart(self, metrics: Dict) -> str:
        """Generate performance visualization"""
        
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        # PnL over time
        ax1.plot(metrics['timestamps'], metrics['pnl_history'])
        ax1.set_title('PnL Over Time')
        ax1.set_ylabel('USDC')
        
        # Spread improvement
        ax2.bar(metrics['markets'], metrics['spread_improvements'])
        ax2.set_title('Spread Improvement by Market')
        ax2.set_ylabel('Basis Points')
        
        # Fill rates
        ax3.pie(metrics['fill_rates'].values(), labels=metrics['fill_rates'].keys())
        ax3.set_title('Fill Rate Distribution')
        
        # Volume
        ax4.bar(metrics['markets'], metrics['volumes'])
        ax4.set_title('Volume by Market')
        ax4.set_ylabel('USDC')
        
        plt.tight_layout()
        
        # Save chart
        chart_path = f"/tmp/performance_chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return chart_path
```

### Social Media Publishing
```python
class SocialPublisher:
    async def publish_update(self, tweet_text: str, chart_path: str = None) -> bool:
        """Publish performance update to Twitter"""
        
        try:
            # Rate limiting check
            if self.last_tweet_time:
                time_since_last = datetime.now() - self.last_tweet_time
                if time_since_last.seconds < self.min_tweet_interval:
                    logger.info("Tweet rate limited, skipping")
                    return False
            
            # Upload media if provided
            media_id = None
            if chart_path and os.path.exists(chart_path):
                media = self.client.create_media(filename=chart_path)
                media_id = media.media_id
            
            # Post tweet
            tweet_params = {"text": tweet_text}
            if media_id:
                tweet_params["media_ids"] = [media_id]
            
            response = self.client.create_tweet(**tweet_params)
            
            # Update tracking
            self.last_tweet_time = datetime.now()
            
            # Cleanup media file
            if chart_path and os.path.exists(chart_path):
                os.remove(chart_path)
            
            logger.info(f"Tweet published: {response.data['id']}")
            return True
            
        except Exception as e:
            logger.error(f"Tweet publishing failed: {e}")
            await self.metrics.increment("twitter_publish_errors")
            return False
```

## Integration Health Monitoring

### Connection Health Checks
```python
class IntegrationHealthChecker:
    async def check_all_integrations(self) -> Dict[str, bool]:
        """Check health of all external integrations"""
        
        health_status = {}
        
        # Sei WebSocket health
        try:
            if self.sei_ws_manager.connection_status == "connected":
                health_status["sei_websocket"] = True
            else:
                health_status["sei_websocket"] = False
        except Exception:
            health_status["sei_websocket"] = False
        
        # Cambrian API health
        try:
            await self.cambrian_client.get_account_info()
            health_status["cambrian_api"] = True
        except Exception:
            health_status["cambrian_api"] = False
        
        # Grafana API health
        try:
            url = f"{self.grafana.base_url}/api/health"
            async with self.grafana.session.get(url) as response:
                health_status["grafana"] = response.status == 200
        except Exception:
            health_status["grafana"] = False
        
        # Twitter API health
        try:
            self.twitter.client.get_me()
            health_status["twitter"] = True
        except Exception:
            health_status["twitter"] = False
        
        return health_status
```

### Circuit Breaker Implementation
```python
class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    async def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        
        if self.state == "OPEN":
            if datetime.now() - self.last_failure_time > timedelta(seconds=self.timeout):
                self.state = "HALF_OPEN"
            else:
                raise CircuitBreakerOpenError("Circuit breaker is OPEN")
        
        try:
            result = await func(*args, **kwargs)
            
            # Success - reset failure count
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
                
            return result
            
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = datetime.now()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
                logger.warning(f"Circuit breaker opened after {self.failure_count} failures")
            
            raise e
```

This comprehensive integration strategy ensures reliable connectivity to all external services while providing robust error handling and failover capabilities essential for high-frequency trading operations.
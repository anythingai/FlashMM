# FlashMM API Specifications

## Overview
FlashMM uses a hybrid communication architecture:
- **External API**: FastAPI REST + WebSocket for dashboard and monitoring
- **Internal Communication**: Direct Python function calls + Redis pub/sub for decoupled components
- **Data Serialization**: Pydantic models with JSON/MessagePack serialization

## External API Endpoints

### Base URL
- **Development**: `http://localhost:8000`
- **Production**: `https://flashmm.example.com`

### Authentication
- **Method**: Bearer Token (JWT)
- **Header**: `Authorization: Bearer <token>`
- **Scope**: Read-only for monitoring, Admin for control

### Health & Status Endpoints

#### `GET /health`
System health check endpoint.

**Response**:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T00:00:00Z",
  "version": "1.0.0",
  "components": {
    "database": "healthy",
    "redis": "healthy",
    "websocket": "healthy",
    "ml_model": "healthy",
    "trading_engine": "healthy"
  },
  "uptime_seconds": 3600
}
```

#### `GET /health/detailed`
Detailed health information with performance metrics.

**Response**:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T00:00:00Z",
  "system": {
    "cpu_percent": 45.2,
    "memory_percent": 67.8,
    "disk_percent": 23.1,
    "network_latency_ms": 12.5
  },
  "components": {
    "websocket": {
      "status": "healthy",
      "last_message": "2024-01-01T00:00:00Z",
      "messages_per_second": 15.2,
      "latency_ms": 125
    },
    "ml_model": {
      "status": "healthy",
      "last_prediction": "2024-01-01T00:00:00Z",
      "predictions_per_second": 5.0,
      "inference_time_ms": 3.2,
      "model_confidence": 0.78
    },
    "trading_engine": {
      "status": "active",
      "orders_placed": 1250,
      "orders_filled": 1180,
      "last_order": "2024-01-01T00:00:00Z"
    }
  }
}
```

### Trading Status Endpoints

#### `GET /trading/status`
Current trading engine status.

**Response**:
```json
{
  "status": "active",
  "markets": [
    {
      "symbol": "SEI/USDC",
      "status": "trading",
      "last_quote_time": "2024-01-01T00:00:00Z",
      "spread_bps": 12.5,
      "position_usdc": -150.75,
      "daily_pnl_usdc": 25.30
    }
  ],
  "total_pnl_usdc": 25.30,
  "risk_level": "normal"
}
```

#### `GET /trading/positions`
Current inventory positions.

**Response**:
```json
{
  "positions": [
    {
      "symbol": "SEI/USDC",
      "base_balance": 1250.0,
      "quote_balance": 875.25,
      "position_usdc": -150.75,
      "position_percent": -1.2,
      "limit_percent": 2.0,
      "last_updated": "2024-01-01T00:00:00Z"
    }
  ],
  "total_position_usdc": -150.75,
  "risk_utilization_percent": 60.0
}
```

#### `GET /trading/orders`
Active orders and recent fills.

**Query Parameters**:
- `limit`: Number of records (default: 100, max: 1000)
- `status`: Filter by status (`active`, `filled`, `cancelled`)

**Response**:
```json
{
  "active_orders": [
    {
      "order_id": "ord_123456",
      "symbol": "SEI/USDC",
      "side": "buy",
      "price": "0.0421",
      "size": "500.0",
      "filled": "0.0",
      "status": "active",
      "created_at": "2024-01-01T00:00:00Z"
    }
  ],
  "recent_fills": [
    {
      "fill_id": "fill_789012",
      "order_id": "ord_123455",
      "symbol": "SEI/USDC",
      "side": "sell",
      "price": "0.0422",
      "size": "250.0",
      "fee": "0.25",
      "timestamp": "2024-01-01T00:00:00Z"
    }
  ]
}
```

### Metrics Endpoints

#### `GET /metrics`
Prometheus-compatible metrics endpoint.

**Response Format**: Prometheus exposition format
```
# HELP flashmm_predictions_total Total number of ML predictions made
# TYPE flashmm_predictions_total counter
flashmm_predictions_total 15420

# HELP flashmm_prediction_accuracy Accuracy of ML predictions
# TYPE flashmm_prediction_accuracy gauge
flashmm_prediction_accuracy 0.582

# HELP flashmm_spread_improvement_bps Current spread improvement in basis points
# TYPE flashmm_spread_improvement_bps gauge
flashmm_spread_improvement_bps{market="SEI/USDC"} 45.2
```

#### `GET /metrics/trading`
Trading-specific metrics in JSON format.

**Response**:
```json
{
  "timestamp": "2024-01-01T00:00:00Z",
  "metrics": {
    "total_volume_usdc": 125000.50,
    "total_trades": 1180,
    "average_spread_bps": 12.5,
    "spread_improvement_percent": 42.3,
    "maker_fee_earned_usdc": 125.50,
    "prediction_accuracy": 0.582,
    "system_uptime_seconds": 86400,
    "orders_per_minute": 2.5,
    "latency_p95_ms": 145.2
  },
  "by_market": {
    "SEI/USDC": {
      "volume_usdc": 75000.30,
      "trades": 720,
      "spread_bps": 11.8,
      "position_usdc": -150.75,
      "pnl_usdc": 18.25
    },
    "ETH/USDC": {
      "volume_usdc": 50000.20,
      "trades": 460,
      "spread_bps": 13.5,
      "position_usdc": 89.50,
      "pnl_usdc": 7.05
    }
  }
}
```

### Admin Control Endpoints

#### `POST /admin/pause`
Pause trading operations.

**Request**:
```json
{
  "reason": "Manual intervention required",
  "duration_seconds": 300
}
```

**Response**:
```json
{
  "status": "paused",
  "reason": "Manual intervention required",
  "paused_at": "2024-01-01T00:00:00Z",
  "resume_at": "2024-01-01T00:05:00Z"
}
```

#### `POST /admin/resume`
Resume trading operations.

**Response**:
```json
{
  "status": "active",
  "resumed_at": "2024-01-01T00:00:00Z"
}
```

#### `POST /admin/emergency-stop`
Emergency stop all trading.

**Request**:
```json
{
  "confirmation": "EMERGENCY_STOP_CONFIRMED"
}
```

**Response**:
```json
{
  "status": "emergency_stopped",
  "stopped_at": "2024-01-01T00:00:00Z",
  "orders_cancelled": 15,
  "positions_flattened": true
}
```

### WebSocket Endpoints

#### `WS /ws/live-metrics`
Real-time metrics stream.

**Connection**: Upgrade to WebSocket
**Message Format**: JSON

**Subscription Message**:
```json
{
  "action": "subscribe",
  "channels": ["trading", "positions", "predictions"]
}
```

**Trading Updates**:
```json
{
  "channel": "trading",
  "timestamp": "2024-01-01T00:00:00Z",
  "data": {
    "symbol": "SEI/USDC",
    "bid": "0.0420",
    "ask": "0.0422",
    "spread_bps": 47.6,
    "last_trade_price": "0.0421",
    "volume_24h": 125000.50
  }
}
```

**Position Updates**:
```json
{
  "channel": "positions",
  "timestamp": "2024-01-01T00:00:00Z",
  "data": {
    "symbol": "SEI/USDC",
    "position_usdc": -150.75,
    "pnl_usdc": 18.25,
    "risk_percent": 1.2
  }
}
```

**Prediction Updates**:
```json
{
  "channel": "predictions",
  "timestamp": "2024-01-01T00:00:00Z",
  "data": {
    "symbol": "SEI/USDC",
    "prediction": "bullish",
    "confidence": 0.78,
    "horizon_ms": 200,
    "price_change_bps": 5.2
  }
}
```

## Internal Service Communication

### Data Models (Pydantic Schemas)

#### Market Data Models
```python
from pydantic import BaseModel
from datetime import datetime
from decimal import Decimal

class OrderBookLevel(BaseModel):
    price: Decimal
    size: Decimal
    orders: int

class OrderBook(BaseModel):
    symbol: str
    timestamp: datetime
    sequence: int
    bids: List[OrderBookLevel]
    asks: List[OrderBookLevel]

class Trade(BaseModel):
    symbol: str
    timestamp: datetime
    price: Decimal
    size: Decimal
    side: str  # "buy" or "sell"
    trade_id: str
```

#### ML Models
```python
class FeatureVector(BaseModel):
    symbol: str
    timestamp: datetime
    features: List[float]
    lookback_ms: int

class Prediction(BaseModel):
    symbol: str
    timestamp: datetime
    direction: str  # "bullish", "bearish", "neutral"
    confidence: float  # 0.0 to 1.0
    price_change_bps: float
    horizon_ms: int
    model_version: str
```

#### Trading Models
```python
class Quote(BaseModel):
    symbol: str
    timestamp: datetime
    bid_price: Decimal
    bid_size: Decimal
    ask_price: Decimal
    ask_size: Decimal
    spread_bps: float

class Order(BaseModel):
    order_id: str
    symbol: str
    side: str  # "buy" or "sell"
    price: Decimal
    size: Decimal
    order_type: str  # "limit", "market"
    time_in_force: str  # "GTC", "IOC", "FOK"
    status: str
    created_at: datetime

class Fill(BaseModel):
    fill_id: str
    order_id: str
    symbol: str
    side: str
    price: Decimal
    size: Decimal
    fee: Decimal
    timestamp: datetime
```

### Redis Pub/Sub Channels

#### Channel Structure
```
flashmm:market_data:{symbol}     # Raw market data
flashmm:predictions:{symbol}     # ML predictions
flashmm:quotes:{symbol}          # Generated quotes
flashmm:orders:{symbol}          # Order updates
flashmm:fills:{symbol}           # Fill notifications
flashmm:positions:{symbol}       # Position updates
flashmm:alerts                   # System alerts
```

#### Message Format
All Redis messages use MessagePack serialization for efficiency:

```python
import msgpack

# Publishing
data = {"symbol": "SEI/USDC", "price": 0.0421, "timestamp": datetime.utcnow()}
redis_client.publish("flashmm:market_data:SEI/USDC", msgpack.packb(data))

# Subscribing
def handle_message(message):
    data = msgpack.unpackb(message['data'])
    # Process data...
```

### Internal API Interfaces

#### Data Ingestion Interface
```python
class DataFeedInterface:
    async def subscribe_orderbook(self, symbol: str) -> AsyncIterator[OrderBook]:
        """Stream order book updates"""
        
    async def subscribe_trades(self, symbol: str) -> AsyncIterator[Trade]:
        """Stream trade updates"""
        
    async def get_historical_data(self, symbol: str, start: datetime, end: datetime) -> List[OrderBook]:
        """Fetch historical order book data"""
```

#### ML Inference Interface
```python
class MLModelInterface:
    async def predict(self, features: FeatureVector) -> Prediction:
        """Generate price prediction"""
        
    async def batch_predict(self, features: List[FeatureVector]) -> List[Prediction]:
        """Batch prediction for efficiency"""
        
    def get_model_info(self) -> Dict:
        """Get model metadata"""
```

#### Trading Engine Interface
```python
class TradingEngineInterface:
    async def place_order(self, order: Order) -> str:
        """Place new order, return order_id"""
        
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel existing order"""
        
    async def get_position(self, symbol: str) -> Dict:
        """Get current position"""
        
    async def emergency_stop(self) -> None:
        """Emergency stop all trading"""
```

## Error Handling

### HTTP Status Codes
- `200`: Success
- `400`: Bad Request (invalid parameters)
- `401`: Unauthorized (invalid/missing token)
- `403`: Forbidden (insufficient permissions)  
- `404`: Not Found
- `429`: Too Many Requests (rate limited)
- `500`: Internal Server Error
- `503`: Service Unavailable (system paused)

### Error Response Format
```json
{
  "error": {
    "code": "INVALID_SYMBOL",
    "message": "Symbol 'INVALID/PAIR' not supported",
    "details": {
      "supported_symbols": ["SEI/USDC", "ETH/USDC"]
    },
    "timestamp": "2024-01-01T00:00:00Z",
    "request_id": "req_123456"
  }
}
```

### Internal Error Handling
```python
class FlashMMException(Exception):
    """Base exception for FlashMM"""
    
class DataFeedError(FlashMMException):
    """Data ingestion errors"""
    
class MLModelError(FlashMMException):
    """ML prediction errors"""
    
class TradingError(FlashMMException):
    """Trading execution errors"""
    
class RiskLimitError(TradingError):
    """Risk limit breached"""
```

## Rate Limiting

### External API Limits
- Health endpoints: 100 requests/minute
- Trading status: 60 requests/minute  
- Metrics: 30 requests/minute
- Admin controls: 10 requests/minute
- WebSocket connections: 10 concurrent per IP

### Internal Rate Limits
- Order placement: 10 orders/second per symbol
- Model predictions: 5 Hz (per design)
- Redis operations: No limit (local)
- External API calls: Respect vendor limits

## Security Considerations

### API Security
- JWT tokens with 1-hour expiration
- Rate limiting with IP-based blocking
- Input validation on all endpoints
- CORS configured for dashboard domain only

### Internal Security
- Redis AUTH password protection
- InfluxDB token-based authentication
- Encryption of sensitive configuration
- Audit logging of all admin actions

This API specification provides comprehensive coverage for both external monitoring/control and internal service communication within the FlashMM system.
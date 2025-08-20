# FlashMM API Documentation

## Table of Contents
- [Overview](#overview)
- [Authentication](#authentication)
- [REST API Endpoints](#rest-api-endpoints)
- [WebSocket API](#websocket-api)
- [Data Models](#data-models)
- [Error Handling](#error-handling)
- [Rate Limiting](#rate-limiting)
- [SDK Integration](#sdk-integration)
- [Examples](#examples)

---

## Overview

FlashMM provides a comprehensive API suite for monitoring, controlling, and integrating with the market making system:

- **🔗 REST API**: HTTP endpoints for system status, trading metrics, and administrative control
- **⚡ WebSocket API**: Real-time streaming data for live dashboards and monitoring
- **🔒 Authentication**: JWT-based security with role-based access control
- **📊 Metrics**: Prometheus-compatible metrics for monitoring integration
- **🐍 Python SDK**: Native Python integration for advanced users

### Base URLs

| Environment | REST API | WebSocket | Dashboard |
|-------------|----------|-----------|-----------|
| **Development** | `http://localhost:8000` | `ws://localhost:8000/ws` | `http://localhost:3000` |
| **Staging** | `https://api-staging.flashmm.com` | `wss://api-staging.flashmm.com/ws` | `https://dashboard-staging.flashmm.com` |
| **Production** | `https://api.flashmm.com` | `wss://api.flashmm.com/ws` | `https://dashboard.flashmm.com` |

### API Versioning

All API endpoints are versioned to ensure backward compatibility:
- **Current Version**: `v1`
- **Version Header**: `API-Version: v1`
- **URL Prefix**: `/api/v1/`

---

## Authentication

### JWT Token Authentication

FlashMM uses JWT (JSON Web Tokens) for secure API access:

```bash
# Obtain access token
curl -X POST https://api.flashmm.com/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "your_username", "password": "your_password"}'

# Response
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 3600,
  "scope": "read write admin"
}
```

### Using Authentication

Include the JWT token in the Authorization header:

```bash
curl -H "Authorization: Bearer your_jwt_token" \
     https://api.flashmm.com/api/v1/trading/status
```

### Scopes and Permissions

| Scope | Description | Endpoints |
|-------|-------------|-----------|
| **read** | View system status and metrics | `/health`, `/trading/status`, `/metrics` |
| **write** | Control trading parameters | `/trading/config`, `/admin/pause` |
| **admin** | Full system control | `/admin/emergency-stop`, `/admin/config` |

---

## REST API Endpoints

### Health and Status

#### `GET /api/v1/health`

Basic system health check.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-20T14:30:00Z",
  "version": "1.2.0",
  "environment": "production",
  "components": {
    "database": "healthy",
    "redis": "healthy",
    "websocket": "healthy",
    "ml_model": "healthy",
    "trading_engine": "healthy",
    "sei_blockchain": "healthy"
  },
  "uptime_seconds": 86400,
  "system_load": {
    "cpu_percent": 45.2,
    "memory_percent": 67.8,
    "disk_percent": 23.1
  }
}
```

#### `GET /api/v1/health/detailed`

Comprehensive health information with performance metrics.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-20T14:30:00Z",
  "system": {
    "hostname": "flashmm-prod-01",
    "cpu_cores": 4,
    "memory_total_gb": 8.0,
    "disk_total_gb": 50.0,
    "cpu_percent": 45.2,
    "memory_percent": 67.8,
    "disk_percent": 23.1,
    "network_latency_ms": 12.5,
    "sei_rpc_latency_ms": 145.2
  },
  "components": {
    "websocket": {
      "status": "healthy",
      "connection_count": 3,
      "last_message": "2024-01-20T14:29:58Z",
      "messages_per_second": 15.2,
      "latency_p95_ms": 125,
      "reconnect_count": 0
    },
    "ml_model": {
      "status": "healthy",
      "model_version": "v1.2.0",
      "last_prediction": "2024-01-20T14:29:59Z",
      "predictions_per_second": 5.0,
      "inference_time_p95_ms": 3.2,
      "model_confidence_avg": 0.78,
      "predictions_today": 432000
    },
    "trading_engine": {
      "status": "active",
      "orders_placed_today": 1250,
      "orders_filled_today": 1180,
      "fill_rate_percent": 94.4,
      "last_order": "2024-01-20T14:29:55Z",
      "active_orders": 5,
      "position_utilization_percent": 60.0
    },
    "sei_blockchain": {
      "status": "healthy",
      "network": "pacific-1",
      "block_height": 12345678,
      "block_time_seconds": 0.4,
      "rpc_endpoints_healthy": 3,
      "last_transaction": "2024-01-20T14:29:50Z"
    }
  },
  "performance": {
    "end_to_end_latency_p95_ms": 183.5,
    "orders_per_minute": 2.5,
    "api_requests_per_minute": 120.0,
    "error_rate_percent": 0.12
  }
}
```

### Trading Status and Metrics

#### `GET /api/v1/trading/status`

Current trading engine status and performance.

**Response:**
```json
{
  "status": "active",
  "trading_enabled": true,
  "start_time": "2024-01-20T00:00:00Z",
  "total_runtime_hours": 14.5,
  "markets": [
    {
      "symbol": "SEI/USDC",
      "status": "trading",
      "last_quote_time": "2024-01-20T14:29:58Z",
      "spread_bps": 12.5,
      "our_spread_bps": 8.2,
      "spread_improvement_percent": 34.4,
      "position_usdc": -150.75,
      "position_percent": -1.2,
      "daily_pnl_usdc": 25.30,
      "total_pnl_usdc": 156.78,
      "volume_24h_usdc": 75000.30,
      "trades_24h": 720,
      "maker_fees_earned_usdc": 37.50,
      "active_orders": 3
    },
    {
      "symbol": "ETH/USDC",
      "status": "trading",
      "last_quote_time": "2024-01-20T14:29:57Z",
      "spread_bps": 15.8,
      "our_spread_bps": 9.1,
      "spread_improvement_percent": 42.4,
      "position_usdc": 89.50,
      "position_percent": 0.7,
      "daily_pnl_usdc": 18.25,
      "total_pnl_usdc": 98.45,
      "volume_24h_usdc": 50000.20,
      "trades_24h": 460,
      "maker_fees_earned_usdc": 25.00,
      "active_orders": 2
    }
  ],
  "totals": {
    "total_pnl_usdc": 255.23,
    "daily_pnl_usdc": 43.55,
    "total_volume_usdc": 125000.50,
    "total_trades": 1180,
    "maker_fees_earned_usdc": 62.50,
    "average_spread_improvement_percent": 38.4
  },
  "risk_metrics": {
    "risk_level": "normal",
    "total_position_usdc": -61.25,
    "max_position_limit_usdc": 2000.0,
    "position_utilization_percent": 3.1,
    "var_95_usdc": 45.20,
    "max_drawdown_usdc": 15.30
  }
}
```

#### `GET /api/v1/trading/positions`

Current inventory positions and risk metrics.

**Query Parameters:**
- `symbol`: Filter by specific trading pair (optional)
- `include_history`: Include position history (default: false)

**Response:**
```json
{
  "timestamp": "2024-01-20T14:30:00Z",
  "positions": [
    {
      "symbol": "SEI/USDC",
      "base_asset": "SEI",
      "quote_asset": "USDC",
      "base_balance": 1250.0,
      "quote_balance": 875.25,
      "base_balance_available": 1200.0,
      "quote_balance_available": 825.25,
      "position_usdc": -150.75,
      "position_percent": -1.2,
      "position_limit_usdc": 2000.0,
      "target_position_usdc": 0.0,
      "inventory_skew": -0.012,
      "last_updated": "2024-01-20T14:29:58Z",
      "unrealized_pnl_usdc": 5.25,
      "realized_pnl_usdc": 20.05
    }
  ],
  "portfolio_summary": {
    "total_position_usdc": -150.75,
    "total_limit_usdc": 4000.0,
    "risk_utilization_percent": 3.8,
    "portfolio_value_usdc": 2125.25,
    "available_capital_usdc": 1950.0,
    "total_unrealized_pnl": 8.75,
    "total_realized_pnl": 43.55
  },
  "risk_metrics": {
    "var_95_1d_usdc": 45.20,
    "expected_shortfall_usdc": 67.80,
    "sharpe_ratio": 2.1,
    "max_drawdown_percent": 1.2,
    "correlation_matrix": {
      "SEI/USDC": {"ETH/USDC": 0.65}
    }
  }
}
```

#### `GET /api/v1/trading/orders`

Active orders and recent trading activity.

**Query Parameters:**
- `limit`: Number of records (default: 100, max: 1000)
- `status`: Filter by status (`active`, `filled`, `cancelled`, `all`)
- `symbol`: Filter by trading pair
- `start_time`: ISO timestamp for historical data
- `end_time`: ISO timestamp for historical data

**Response:**
```json
{
  "timestamp": "2024-01-20T14:30:00Z",
  "pagination": {
    "limit": 100,
    "offset": 0,
    "total": 1250,
    "has_more": true
  },
  "active_orders": [
    {
      "order_id": "ord_abc123456",
      "client_order_id": "flashmm_20240120_001",
      "symbol": "SEI/USDC",
      "side": "buy",
      "order_type": "limit",
      "price": "0.04210",
      "size": "500.0",
      "filled": "0.0",
      "remaining": "500.0",
      "status": "active",
      "time_in_force": "GTC",
      "created_at": "2024-01-20T14:25:30Z",
      "updated_at": "2024-01-20T14:25:30Z",
      "expires_at": null,
      "quote_generated_at": "2024-01-20T14:25:29Z",
      "spread_bps": 8.2,
      "is_maker": true
    }
  ],
  "recent_fills": [
    {
      "fill_id": "fill_xyz789012",
      "order_id": "ord_def456789",
      "client_order_id": "flashmm_20240120_000",
      "symbol": "SEI/USDC",
      "side": "sell",
      "price": "0.04220",
      "size": "250.0",
      "fee": "0.0525",
      "fee_asset": "USDC",
      "timestamp": "2024-01-20T14:20:15Z",
      "trade_id": "trade_987654321",
      "is_maker": true,
      "realized_pnl": "2.15"
    }
  ],
  "fill_statistics": {
    "total_fills_today": 1180,
    "fill_rate_percent": 94.4,
    "average_fill_time_seconds": 45.2,
    "maker_fill_percent": 89.5,
    "taker_fill_percent": 10.5
  }
}
```

### ML Predictions and Analytics

#### `GET /api/v1/ml/predictions`

Recent ML predictions and model performance.

**Query Parameters:**
- `symbol`: Trading pair (required)
- `limit`: Number of predictions (default: 100, max: 1000)
- `include_features`: Include feature vectors (default: false)

**Response:**
```json
{
  "timestamp": "2024-01-20T14:30:00Z",
  "model_info": {
    "version": "v1.2.0",
    "architecture": "transformer",
    "training_date": "2024-01-15T00:00:00Z",
    "parameters": 2850000,
    "input_features": 64,
    "inference_time_ms": 3.2
  },
  "predictions": [
    {
      "timestamp": "2024-01-20T14:29:58Z",
      "symbol": "SEI/USDC",
      "direction": "bullish",
      "confidence": 0.78,
      "price_change_bps": 5.2,
      "horizon_ms": 200,
      "current_price": "0.04215",
      "predicted_price": "0.04237",
      "features_hash": "sha256:abc123...",
      "model_version": "v1.2.0"
    }
  ],
  "performance_metrics": {
    "accuracy_1h": 0.582,
    "accuracy_24h": 0.567,
    "accuracy_7d": 0.573,
    "sharpe_ratio": 1.85,
    "max_drawdown": 0.023,
    "total_predictions": 432000,
    "correct_predictions": 247824
  }
}
```

#### `GET /api/v1/ml/model/info`

Detailed ML model information and statistics.

**Response:**
```json
{
  "model": {
    "version": "v1.2.0",
    "architecture": "transformer",
    "created_at": "2024-01-15T00:00:00Z",
    "trained_on": {
      "start_date": "2024-01-01T00:00:00Z",
      "end_date": "2024-01-14T23:59:59Z",
      "total_samples": 2016000,
      "validation_split": 0.2
    },
    "parameters": {
      "total_params": 2850000,
      "trainable_params": 2850000,
      "model_size_mb": 4.8,
      "input_features": 64,
      "sequence_length": 100,
      "output_classes": 3
    },
    "performance": {
      "training_accuracy": 0.651,
      "validation_accuracy": 0.587,
      "test_accuracy": 0.573,
      "training_loss": 0.892,
      "validation_loss": 1.045,
      "overfitting_score": 0.058
    }
  },
  "runtime_stats": {
    "inference_time_p50_ms": 2.8,
    "inference_time_p95_ms": 4.1,
    "inference_time_p99_ms": 6.2,
    "predictions_per_second": 357.1,
    "memory_usage_mb": 245.6,
    "cpu_utilization_percent": 12.5
  },
  "feature_importance": [
    {"feature": "bid_ask_spread", "importance": 0.142},
    {"feature": "volume_imbalance", "importance": 0.138},
    {"feature": "price_momentum_5s", "importance": 0.125},
    {"feature": "order_book_slope", "importance": 0.118}
  ]
}
```

### Metrics and Monitoring

#### `GET /api/v1/metrics`

Prometheus-compatible metrics endpoint.

**Response Format:** Prometheus exposition format
```
# HELP flashmm_predictions_total Total number of ML predictions made
# TYPE flashmm_predictions_total counter
flashmm_predictions_total{symbol="SEI/USDC"} 216000
flashmm_predictions_total{symbol="ETH/USDC"} 216000

# HELP flashmm_prediction_accuracy Accuracy of ML predictions
# TYPE flashmm_prediction_accuracy gauge
flashmm_prediction_accuracy{symbol="SEI/USDC",timeframe="1h"} 0.582
flashmm_prediction_accuracy{symbol="ETH/USDC",timeframe="1h"} 0.575

# HELP flashmm_spread_improvement_bps Current spread improvement in basis points
# TYPE flashmm_spread_improvement_bps gauge
flashmm_spread_improvement_bps{symbol="SEI/USDC"} 45.2
flashmm_spread_improvement_bps{symbol="ETH/USDC"} 62.1

# HELP flashmm_trading_volume_usdc_total Total trading volume in USDC
# TYPE flashmm_trading_volume_usdc_total counter
flashmm_trading_volume_usdc_total{symbol="SEI/USDC"} 1250000.50
flashmm_trading_volume_usdc_total{symbol="ETH/USDC"} 850000.25

# HELP flashmm_latency_seconds Request latency in seconds
# TYPE flashmm_latency_seconds histogram
flashmm_latency_seconds_bucket{le="0.1"} 8945
flashmm_latency_seconds_bucket{le="0.2"} 9512
flashmm_latency_seconds_bucket{le="0.5"} 9876
flashmm_latency_seconds_bucket{le="+Inf"} 10000
flashmm_latency_seconds_sum 1543.2
flashmm_latency_seconds_count 10000
```

#### `GET /api/v1/metrics/trading`

Detailed trading metrics in JSON format.

**Response:**
```json
{
  "timestamp": "2024-01-20T14:30:00Z",
  "period": "24h",
  "summary": {
    "total_volume_usdc": 125000.50,
    "total_trades": 1180,
    "unique_trading_pairs": 2,
    "average_spread_bps": 12.1,
    "spread_improvement_percent": 38.4,
    "maker_fee_earned_usdc": 62.50,
    "total_pnl_usdc": 43.55,
    "sharpe_ratio": 2.1,
    "max_drawdown_percent": 1.2,
    "system_uptime_percent": 99.8
  },
  "by_market": {
    "SEI/USDC": {
      "volume_usdc": 75000.30,
      "trades": 720,
      "spread_bps": 11.8,
      "our_spread_bps": 8.2,
      "improvement_percent": 30.5,
      "position_usdc": -150.75,
      "pnl_usdc": 25.30,
      "maker_fees_usdc": 37.50,
      "fill_rate_percent": 94.4,
      "prediction_accuracy": 0.582
    },
    "ETH/USDC": {
      "volume_usdc": 50000.20,
      "trades": 460,
      "spread_bps": 13.5,
      "our_spread_bps": 9.1,
      "improvement_percent": 32.6,
      "position_usdc": 89.50,
      "pnl_usdc": 18.25,
      "maker_fees_usdc": 25.00,
      "fill_rate_percent": 91.2,
      "prediction_accuracy": 0.575
    }
  },
  "performance": {
    "orders_per_minute": 2.5,
    "latency_p95_ms": 183.5,
    "api_requests_per_minute": 120.0,
    "error_rate_percent": 0.12,
    "predictions_per_second": 5.0,
    "ml_inference_time_ms": 3.2
  },
  "risk": {
    "total_position_usdc": -61.25,
    "max_position_limit_usdc": 4000.0,
    "position_utilization_percent": 1.5,
    "var_95_usdc": 45.20,
    "expected_shortfall_usdc": 67.80
  }
}
```

### Administrative Controls

#### `POST /api/v1/admin/pause`

Pause trading operations temporarily.

**Request:**
```json
{
  "reason": "Manual intervention required",
  "duration_seconds": 300,
  "markets": ["SEI/USDC"],  // Optional: specific markets
  "cancel_orders": true,    // Cancel existing orders
  "notify_social": true     // Post to social media
}
```

**Response:**
```json
{
  "status": "paused",
  "reason": "Manual intervention required",
  "paused_at": "2024-01-20T14:30:00Z",
  "resume_at": "2024-01-20T14:35:00Z",
  "affected_markets": ["SEI/USDC"],
  "orders_cancelled": 3,
  "positions_maintained": true,
  "notification_sent": true
}
```

#### `POST /api/v1/admin/resume`

Resume trading operations.

**Request:**
```json
{
  "markets": ["SEI/USDC"]  // Optional: specific markets
}
```

**Response:**
```json
{
  "status": "active",
  "resumed_at": "2024-01-20T14:30:00Z",
  "affected_markets": ["SEI/USDC", "ETH/USDC"],
  "orders_placed": 4,
  "system_health_check": "passed"
}
```

#### `POST /api/v1/admin/emergency-stop`

Emergency stop all trading immediately.

**Request:**
```json
{
  "confirmation": "EMERGENCY_STOP_CONFIRMED",
  "reason": "Unusual market conditions detected",
  "flatten_positions": false  // Whether to close positions
}
```

**Response:**
```json
{
  "status": "emergency_stopped",
  "stopped_at": "2024-01-20T14:30:00Z",
  "reason": "Unusual market conditions detected",
  "orders_cancelled": 15,
  "positions_flattened": false,
  "total_position_usdc": -61.25,
  "emergency_contacts_notified": true,
  "incident_id": "inc_20240120_001"
}
```

---

## WebSocket API

### Connection and Authentication

Connect to WebSocket endpoint with authentication:

```javascript
const ws = new WebSocket('wss://api.flashmm.com/ws/live-data');

// Authenticate after connection
ws.onopen = function() {
    ws.send(JSON.stringify({
        action: 'authenticate',
        token: 'your_jwt_token'
    }));
};
```

### Subscription Management

#### Subscribe to Channels

```javascript
// Subscribe to multiple channels
ws.send(JSON.stringify({
    action: 'subscribe',
    channels: [
        'trading.SEI/USDC',
        'positions',
        'predictions.SEI/USDC',
        'system.health'
    ]
}));
```

#### Unsubscribe from Channels

```javascript
ws.send(JSON.stringify({
    action: 'unsubscribe',
    channels: ['trading.ETH/USDC']
}));
```

### Channel Types and Data Formats

#### Trading Updates (`trading.{symbol}`)

Real-time trading data for specific market:

```json
{
  "channel": "trading.SEI/USDC",
  "timestamp": "2024-01-20T14:30:00.123Z",
  "type": "quote_update",
  "data": {
    "symbol": "SEI/USDC",
    "bid": "0.04200",
    "ask": "0.04220",
    "bid_size": "500.0",
    "ask_size": "750.0",
    "spread_bps": 47.6,
    "our_spread_bps": 8.2,
    "market_spread_bps": 47.6,
    "improvement_percent": 82.8,
    "last_trade_price": "0.04210",
    "volume_24h_usdc": 75000.30,
    "quote_generated_at": "2024-01-20T14:29:59.890Z",
    "orders_active": 2
  }
}
```

#### Position Updates (`positions`)

Real-time position and P&L updates:

```json
{
  "channel": "positions",
  "timestamp": "2024-01-20T14:30:00.123Z",
  "type": "position_update",
  "data": {
    "symbol": "SEI/USDC",
    "position_usdc": -150.75,
    "position_percent": -1.2,
    "unrealized_pnl_usdc": 5.25,
    "realized_pnl_usdc": 20.05,
    "daily_pnl_usdc": 25.30,
    "total_pnl_usdc": 156.78,
    "risk_percent": 1.2,
    "inventory_skew": -0.012,
    "last_trade": {
      "side": "sell",
      "price": "0.04220",
      "size": "100.0",
      "timestamp": "2024-01-20T14:29:58.567Z"
    }
  }
}
```

#### Prediction Updates (`predictions.{symbol}`)

Real-time ML predictions:

```json
{
  "channel": "predictions.SEI/USDC",
  "timestamp": "2024-01-20T14:30:00.123Z",
  "type": "prediction_update",
  "data": {
    "symbol": "SEI/USDC",
    "prediction": "bullish",
    "confidence": 0.78,
    "horizon_ms": 200,
    "price_change_bps": 5.2,
    "current_price": "0.04215",
    "predicted_price": "0.04237",
    "model_version": "v1.2.0",
    "features_used": 64,
    "inference_time_ms": 3.1,
    "previous_accuracy": 0.582
  }
}
```

#### System Health (`system.health`)

System status and performance metrics:

```json
{
  "channel": "system.health",
  "timestamp": "2024-01-20T14:30:00.123Z",
  "type": "health_update",
  "data": {
    "status": "healthy",
    "cpu_percent": 45.2,
    "memory_percent": 67.8,
    "disk_percent": 23.1,
    "network_latency_ms": 12.5,
    "sei_rpc_latency_ms": 145.2,
    "websocket_connections": 8,
    "api_requests_per_minute": 120.0,
    "error_rate_percent": 0.12,
    "uptime_seconds": 86400,
    "components": {
      "trading_engine": "active",
      "ml_model": "healthy",
      "database": "healthy",
      "redis": "healthy"
    }
  }
}
```

#### Order Updates (`orders.{symbol}`)

Order lifecycle events:

```json
{
  "channel": "orders.SEI/USDC",
  "timestamp": "2024-01-20T14:30:00.123Z",
  "type": "order_filled",
  "data": {
    "order_id": "ord_abc123456",
    "symbol": "SEI/USDC",
    "side": "buy",
    "price": "0.04210",
    "size": "500.0",
    "filled": "500.0",
    "remaining": "0.0",
    "fill_price": "0.04210",
    "fill_size": "500.0",
    "fee": "1.0525",
    "realized_pnl": "2.15",
    "is_maker": true,
    "timestamp": "2024-01-20T14:30:00.056Z"
  }
}
```

### Error Handling

WebSocket errors are sent as special messages:

```json
{
  "type": "error",
  "timestamp": "2024-01-20T14:30:00.123Z",
  "error": {
    "code": "INVALID_CHANNEL",
    "message": "Channel 'invalid.channel' does not exist",
    "details": {
      "available_channels": [
        "trading.*",
        "positions",
        "predictions.*",
        "system.health",
        "orders.*"
      ]
    }
  }
}
```

---

## Data Models

### Core Trading Models

#### OrderBook
```typescript
interface OrderBookLevel {
  price: string;      // Decimal as string
  size: string;       // Decimal as string  
  orders: number;     // Number of orders at level
}

interface OrderBook {
  symbol: string;
  timestamp: string;  // ISO 8601
  sequence: number;
  bids: OrderBookLevel
[];
  asks: OrderBookLevel[];
}
```

#### Trade
```typescript
interface Trade {
  symbol: string;
  timestamp: string;
  price: string;
  size: string;
  side: 'buy' | 'sell';
  trade_id: string;
  maker_order_id?: string;
  taker_order_id?: string;
}
```

#### Order
```typescript
interface Order {
  order_id: string;
  client_order_id: string;
  symbol: string;
  side: 'buy' | 'sell';
  order_type: 'limit' | 'market' | 'stop' | 'stop_limit';
  price: string;
  size: string;
  filled: string;
  remaining: string;
  status: 'pending' | 'active' | 'filled' | 'cancelled' | 'expired';
  time_in_force: 'GTC' | 'IOC' | 'FOK' | 'GTD';
  created_at: string;
  updated_at: string;
  expires_at?: string;
}
```

#### Position
```typescript
interface Position {
  symbol: string;
  base_asset: string;
  quote_asset: string;
  base_balance: string;
  quote_balance: string;
  base_balance_available: string;
  quote_balance_available: string;
  position_usdc: string;
  position_percent: number;
  position_limit_usdc: string;
  unrealized_pnl_usdc: string;
  realized_pnl_usdc: string;
  last_updated: string;
}
```

### ML Models

#### Prediction
```typescript
interface Prediction {
  timestamp: string;
  symbol: string;
  direction: 'bullish' | 'bearish' | 'neutral';
  confidence: number; // 0.0 to 1.0
  price_change_bps: number;
  horizon_ms: number;
  current_price: string;
  predicted_price: string;
  model_version: string;
  features_hash?: string;
}
```

#### FeatureVector
```typescript
interface FeatureVector {
  timestamp: string;
  symbol: string;
  features: number[];
  feature_names: string[];
  lookback_ms: number;
  normalization_method: string;
}
```

### Risk Models

#### RiskMetrics
```typescript
interface RiskMetrics {
  total_position_usdc: string;
  max_position_limit_usdc: string;
  position_utilization_percent: number;
  var_95_usdc: string;
  expected_shortfall_usdc: string;
  sharpe_ratio: number;
  max_drawdown_percent: number;
  correlation_matrix: Record<string, Record<string, number>>;
}
```

---

## Error Handling

### HTTP Status Codes

| Code | Status | Description | Common Causes |
|------|--------|-------------|---------------|
| 200 | OK | Success | Request completed successfully |
| 201 | Created | Resource created | Order placed, configuration saved |
| 400 | Bad Request | Invalid request | Missing parameters, invalid format |
| 401 | Unauthorized | Authentication failed | Invalid/missing token |
| 403 | Forbidden | Access denied | Insufficient permissions |
| 404 | Not Found | Resource not found | Invalid endpoint, missing data |
| 429 | Too Many Requests | Rate limited | Exceeded request limits |
| 500 | Internal Server Error | Server error | System malfunction |
| 503 | Service Unavailable | Service down | System maintenance, emergency stop |

### Error Response Format

All API errors return a consistent JSON structure:

```json
{
  "error": {
    "code": "INVALID_SYMBOL",
    "message": "Trading pair 'INVALID/PAIR' is not supported",
    "details": {
      "supported_symbols": ["SEI/USDC", "ETH/USDC"],
      "provided_symbol": "INVALID/PAIR"
    },
    "timestamp": "2024-01-20T14:30:00Z",
    "request_id": "req_abc123456",
    "documentation_url": "https://docs.flashmm.com/api#error-codes"
  }
}
```

### Common Error Codes

| Code | Description | Resolution |
|------|-------------|------------|
| `INVALID_SYMBOL` | Unsupported trading pair | Use supported symbols from `/trading/status` |
| `INSUFFICIENT_BALANCE` | Not enough funds | Check balance with `/trading/positions` |
| `POSITION_LIMIT_EXCEEDED` | Position would exceed limits | Reduce order size or adjust limits |
| `TRADING_PAUSED` | Trading temporarily disabled | Wait for resume or check `/trading/status` |
| `MODEL_UNAVAILABLE` | ML model not ready | Check `/ml/model/info` for status |
| `RATE_LIMIT_EXCEEDED` | Too many requests | Reduce request frequency |
| `INVALID_ORDER_SIZE` | Order size below minimum | Check market configuration |
| `STALE_PRICE` | Price quote too old | Request fresh quote |

---

## Rate Limiting

### Request Limits

FlashMM implements tiered rate limiting based on endpoint sensitivity:

| Endpoint Category | Limit | Window | Burst |
|------------------|-------|--------|-------|
| **Health/Status** | 100 requests | 1 minute | 10 |
| **Trading Data** | 60 requests | 1 minute | 5 |
| **Metrics** | 30 requests | 1 minute | 3 |
| **Admin Controls** | 10 requests | 1 minute | 2 |
| **ML Predictions** | 20 requests | 1 minute | 5 |

### Rate Limit Headers

All responses include rate limiting information:

```bash
HTTP/1.1 200 OK
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 45
X-RateLimit-Reset: 1642694400
X-RateLimit-Burst: 5
X-RateLimit-Burst-Remaining: 3
Retry-After: 15  # Only present when rate limited
```

### WebSocket Rate Limits

| Connection Type | Limit | Description |
|----------------|-------|-------------|
| **Concurrent Connections** | 10 per IP | Maximum simultaneous connections |
| **Message Rate** | 20/second | Maximum messages per second |
| **Subscription Limit** | 50 channels | Maximum channels per connection |
| **Authentication Attempts** | 5/minute | Failed authentication attempts |

---

## SDK Integration

### Python SDK

FlashMM provides a native Python SDK for seamless integration:

#### Installation
```bash
pip install flashmm-sdk
```

#### Basic Usage
```python
from flashmm_sdk import FlashMMClient

# Initialize client
client = FlashMMClient(
    api_key='your_api_key',
    base_url='https://api.flashmm.com',
    environment='production'
)

# Get trading status
status = await client.trading.get_status()
print(f"Total P&L: ${status.totals.total_pnl_usdc}")

# Subscribe to real-time data
async def handle_position_update(data):
    print(f"Position update: {data.symbol} = ${data.position_usdc}")

await client.websocket.subscribe('positions', handle_position_update)
```

---

This comprehensive API documentation provides everything needed to integrate with FlashMM's trading system. For additional support, see our other documentation:

- [Architecture Documentation](ARCHITECTURE.md)
- [Developer Guide](DEVELOPER.md) 
- [Configuration Reference](CONFIGURATION.md)
- [Operations Runbook](OPERATIONS.md)
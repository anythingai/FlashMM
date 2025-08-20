# FlashMM Technology Stack Specifications

## Core Technology Decisions

### Programming Language: Python 3.11+
**Rationale**: 
- Native PyTorch support for ML models
- Excellent async/await support for high-performance WebSocket handling
- Rich ecosystem for financial/trading applications
- Cambrian SDK native Python support
- Fast development cycle for hackathon timeline

### Runtime Environment: Docker + Docker Compose
**Rationale**:
- Single VPS deployment requirement
- Consistent environment across development/production
- Easy service orchestration and dependency management
- Resource isolation and monitoring

## Technology Stack by Component

### 1. Data Ingestion Layer

#### WebSocket Client
- **Primary**: `websockets` 11.0+
- **Backup**: `aiohttp` 3.8+ with WebSocket support
- **JSON Processing**: `orjson` 3.9+ (fastest JSON parser)
- **Async Framework**: `asyncio` (built-in)

**Key Dependencies**:
```python
websockets==11.0.2
aiohttp==3.8.5
orjson==3.9.4
```

#### Data Normalizer
- **Core**: Python `dataclasses` + `pydantic` 2.0+
- **Validation**: `pydantic` with custom validators
- **Serialization**: `msgpack` for efficient inter-service communication

**Key Dependencies**:
```python
pydantic==2.1.1
msgpack==1.0.5
```

### 2. Storage & Cache Layer

#### Redis Cache
- **Version**: Redis 7.0+
- **Python Client**: `redis-py` 4.6+ with async support
- **Use Cases**: 
  - Real-time order book snapshots
  - Feature vector caching
  - Session management
  - Rate limiting counters

**Configuration**:
```yaml
redis:
  version: "7.0-alpine"
  memory_policy: "allkeys-lru"
  max_memory: "512mb"
  persistence: false  # Performance over durability
```

#### InfluxDB
- **Version**: InfluxDB 2.7+
- **Python Client**: `influxdb-client` 1.37+
- **Use Cases**:
  - Time-series metrics storage
  - Performance analytics
  - Trading statistics
  - System telemetry

**Configuration**:
```yaml
influxdb:
  version: "2.7-alpine"
  storage_engine: "tsm1"
  retention_policy: "7d"  # Hackathon scope
```

### 3. ML Inference Pipeline

#### PyTorch Stack
- **Core**: `torch` 2.0+ (CPU inference only for cost efficiency)
- **Optimization**: `torchscript` for production deployment
- **Export Format**: `.pt` files < 5MB
- **Inference Engine**: Custom async wrapper

**Key Dependencies**:
```python
torch==2.0.1+cpu
torchvision==0.15.2+cpu  # If needed for preprocessing
numpy==1.24.3
scikit-learn==1.3.0  # For preprocessing pipelines
```

#### Model Architecture Options
1. **Transformer-based**: Custom lightweight attention mechanism
2. **LSTM**: Bidirectional LSTM with attention pooling
3. **Ensemble**: Voting classifier combining multiple models

**Performance Targets**:
- Model size: < 5MB
- Inference time: < 5ms
- Memory usage: < 100MB per model

### 4. Trading Engine

#### Cambrian SDK Integration
- **Version**: `cambrian-sdk` 0.4+
- **WebSocket**: For real-time order updates
- **REST**: For order placement and management
- **Authentication**: API key + signature-based

**Key Dependencies**:
```python
cambrian-sdk>=0.4.0
cryptography==41.0.3  # For signature generation
```

#### Order Management
- **State Management**: Custom finite state machine
- **Concurrency**: `asyncio` with semaphores for rate limiting
- **Error Handling**: Exponential backoff with circuit breakers

### 5. API & Web Services

#### FastAPI Framework
- **Version**: `fastapi` 0.100+
- **ASGI Server**: `uvicorn` 0.23+ with `uvloop`
- **WebSocket Support**: Native FastAPI WebSocket
- **Documentation**: Auto-generated OpenAPI/Swagger

**Key Dependencies**:
```python
fastapi==0.100.1
uvicorn[standard]==0.23.2
uvloop==0.17.0  # Linux performance boost
websockets==11.0.2
```

#### API Architecture
```
/health          - Health check endpoint
/metrics         - Prometheus-compatible metrics
/trading/status  - Current trading state
/trading/positions - Current inventory
/admin/pause     - Emergency controls
/ws/live-metrics - WebSocket for real-time data
```

### 6. Monitoring & Observability

#### Logging
- **Framework**: Python `structlog` 23.1+
- **Format**: JSON structured logging
- **Levels**: DEBUG/INFO/WARNING/ERROR/CRITICAL
- **Output**: Stdout (Docker logs) + InfluxDB

**Key Dependencies**:
```python
structlog==23.1.0
python-json-logger==2.0.7
```

#### Metrics Collection
- **Prometheus Client**: `prometheus-client` 0.17+
- **Custom Metrics**: Trading-specific metrics
- **System Metrics**: CPU, memory, network via `psutil`

**Key Dependencies**:
```python
prometheus-client==0.17.1
psutil==5.9.5
```

#### Grafana Integration
- **API Client**: `grafana-api` 1.0+
- **Dashboard Config**: JSON-based dashboard definitions
- **Alerting**: Grafana Cloud alerting rules

### 7. External Integrations

#### Social Media (X/Twitter)
- **Library**: `tweepy` 4.14+
- **Authentication**: OAuth 2.0 Bearer Token
- **Rate Limits**: Built-in rate limiting support
- **Content**: Performance summaries + charts

**Key Dependencies**:
```python
tweepy==4.14.0
Pillow==10.0.0  # For chart generation
matplotlib==3.7.2  # For performance charts
```

#### Sei Blockchain
- **RPC Client**: Custom HTTP client with connection pooling
- **WebSocket**: Direct WebSocket connection to Sei node
- **Backup Endpoints**: Multiple RPC endpoints for failover

### 8. Security & Authentication

#### Cryptography
- **Library**: `cryptography` 41.0+
- **Key Management**: Environment variables + optional HSM
- **Encryption**: AES-256 for sensitive data at rest
- **Signatures**: ECDSA for API authentication

**Key Dependencies**:
```python
cryptography==41.0.3
PyJWT==2.8.0  # For JWT tokens
python-dotenv==1.0.0  # Environment management
```

## Development & Testing Stack

### Testing Framework
- **Unit Tests**: `pytest` 7.4+
- **Async Testing**: `pytest-asyncio` 0.21+
- **Mocking**: `unittest.mock` (built-in) + `pytest-mock`
- **Coverage**: `pytest-cov` 4.1+

**Key Dependencies**:
```python
pytest==7.4.0
pytest-asyncio==0.21.1
pytest-cov==4.1.0
pytest-mock==3.11.1
```

### Code Quality
- **Linting**: `ruff` 0.0.280+ (replaces flake8, black, isort)
- **Type Checking**: `mypy` 1.5+
- **Security**: `bandit` 1.7+
- **Import Sorting**: Built into `ruff`

**Key Dependencies**:
```python
ruff==0.0.280
mypy==1.5.1
bandit==1.7.5
```

### Performance Profiling
- **Profiler**: `py-spy` for production profiling
- **Memory**: `memory-profiler` for memory analysis
- **Load Testing**: `locust` for API load testing

## Infrastructure Technologies

### Container Orchestration
```yaml
services:
  app:
    image: flashmm:latest
    ports: ["8000:8000"]
    environment:
      - REDIS_URL=redis://redis:6379
      - INFLUXDB_URL=http://influxdb:8086
    depends_on: [redis, influxdb]
    
  redis:
    image: redis:7.0-alpine
    command: redis-server --maxmemory 512mb --maxmemory-policy allkeys-lru
    
  influxdb:
    image: influxdb:2.7-alpine
    environment:
      - INFLUXDB_DB=flashmm
      - INFLUXDB_ADMIN_USER=admin
      - INFLUXDB_ADMIN_PASSWORD=${INFLUXDB_PASSWORD}
```

### Reverse Proxy (Optional)
- **Nginx**: For SSL termination and load balancing
- **Configuration**: Custom nginx.conf for WebSocket support

### Monitoring Tools
- **Grafana Cloud**: SaaS dashboard and alerting
- **InfluxDB**: Self-hosted time-series database
- **Custom Dashboards**: JSON-based dashboard as code

## Performance Optimizations

### Python Runtime
- **CPython**: Standard interpreter
- **Memory**: `PYTHONMALLOC=malloc` for better memory profiling
- **GC**: Tuned garbage collection for low-latency

### Async Optimizations
- **Event Loop**: `uvloop` on Linux for 2x performance boost
- **Connection Pooling**: Persistent connections to external APIs
- **Batch Processing**: Batch metric updates to reduce I/O

### Model Optimizations
- **TorchScript**: JIT compilation for faster inference
- **ONNX Export**: Optional ONNX format for cross-platform deployment
- **Quantization**: INT8 quantization if model size becomes critical

## Deployment Specifications

### VPS Requirements
- **OS**: Ubuntu 22.04 LTS
- **CPU**: 4+ cores
- **RAM**: 8GB minimum
- **Storage**: 50GB SSD
- **Network**: 1Gbps with low latency to Sei RPC

### Environment Variables
```bash
# Core Configuration
ENVIRONMENT=testnet
LOG_LEVEL=INFO
DEBUG=false

# Sei Network
SEI_RPC_URL=https://sei-testnet-rpc-url
SEI_WS_URL=wss://sei-testnet-ws-url
SEI_CHAIN_ID=sei-testnet-4

# Trading
CAMBRIAN_API_KEY=${CAMBRIAN_API_KEY}
CAMBRIAN_SECRET=${CAMBRIAN_SECRET}
MAX_POSITION_USDC=2000
QUOTE_FREQUENCY_HZ=5

# Storage
REDIS_URL=redis://localhost:6379
INFLUXDB_URL=http://localhost:8086
INFLUXDB_TOKEN=${INFLUXDB_TOKEN}

# External APIs
GRAFANA_API_KEY=${GRAFANA_API_KEY}
TWITTER_BEARER_TOKEN=${TWITTER_BEARER_TOKEN}

# Security
SECRET_KEY=${SECRET_KEY}
ENCRYPTION_KEY=${ENCRYPTION_KEY}
```

This technology stack provides a balance of performance, reliability, and development speed suitable for the hackathon timeline while meeting all technical requirements specified in the PRD.
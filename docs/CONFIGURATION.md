# FlashMM Configuration Reference

## Table of Contents
- [Configuration Overview](#configuration-overview)
- [Environment Variables](#environment-variables)
- [Configuration Files](#configuration-files)
- [Component Configuration](#component-configuration)
- [Environment-Specific Settings](#environment-specific-settings)
- [Security Configuration](#security-configuration)
- [Performance Tuning](#performance-tuning)
- [Validation and Troubleshooting](#validation-and-troubleshooting)
- [Migration Guide](#migration-guide)
- [Best Practices](#best-practices)

---

## Configuration Overview

FlashMM uses a **hierarchical configuration system** that allows flexible and secure configuration management across different environments and deployment scenarios.

### Configuration Hierarchy

```
Configuration Priority (highest to lowest):
1. Runtime Environment Variables
2. Environment-specific configuration files
3. Base configuration files  
4. Default values in code
```

### Configuration Sources

```mermaid
graph TD
    A[Environment Variables] --> E[Final Configuration]
    B[Environment Config Files] --> E
    C[Base Config Files] --> E
    D[Default Values] --> E
    
    F[config/environments/development.yml] --> B
    G[config/environments/staging.yml] --> B
    H[config/environments/production.yml] --> B
    
    I[.env.development] --> A
    J[.env.staging] --> A
    K[.env.production] --> A
    
    style A fill:#e8f5e8
    style E fill:#ffd700
```

### Configuration Loading Process

```python
# Configuration loading order
1. Load default values from code
2. Load base configuration from config/base.yml
3. Load environment-specific config from config/environments/{env}.yml
4. Override with environment variables
5. Validate final configuration
6. Apply runtime overrides (Redis-based)
```

---

## Environment Variables

### Core System Configuration

#### Application Settings

```bash
# Environment and debugging
ENVIRONMENT=development                    # development, staging, production
FLASHMM_DEBUG=true                        # Enable debug mode
FLASHMM_LOG_LEVEL=INFO                    # DEBUG, INFO, WARNING, ERROR, CRITICAL
FLASHMM_LOG_FORMAT=json                   # json, text
FLASHMM_CONFIG_PATH=./config              # Configuration directory path

# Application behavior
UVLOOP_ENABLED=true                       # Use uvloop for better async performance
MAX_WORKERS=4                             # Number of worker processes
GRACEFUL_SHUTDOWN_TIMEOUT=30              # Seconds to wait for graceful shutdown
```

#### Network and Connectivity

```bash
# API Server
HOST=0.0.0.0                             # Bind address
PORT=8000                                # Port number
API_PREFIX=/api/v1                       # API URL prefix
CORS_ORIGINS=*                           # CORS allowed origins (comma-separated)
CORS_CREDENTIALS=true                    # Allow credentials in CORS

# Request limits
MAX_REQUEST_SIZE=10485760                # Max request size in bytes (10MB)
REQUEST_TIMEOUT=30                       # Request timeout in seconds
KEEPALIVE_TIMEOUT=65                     # Keep-alive timeout
```

### Sei Blockchain Configuration

```bash
# Network settings
SEI_NETWORK=testnet                      # testnet, mainnet
SEI_CHAIN_ID=atlantic-2                 # Chain ID for testnet
SEI_RPC_URL=https://sei-testnet-rpc.polkachu.com
SEI_WS_URL=wss://sei-testnet-rpc.polkachu.com/websocket

# Backup endpoints (comma-separated)
SEI_RPC_BACKUP_URLS=https://sei-testnet-rpc.lavenderfive.com,https://rpc-sei.stingray.plus
SEI_WS_BACKUP_URLS=wss://sei-testnet-rpc.lavenderfive.com/websocket

# Connection settings
SEI_CONNECTION_TIMEOUT=30                # Connection timeout in seconds
SEI_READ_TIMEOUT=10                      # Read timeout in seconds
SEI_MAX_RETRIES=3                        # Maximum retry attempts
SEI_RETRY_DELAY=1                        # Delay between retries in seconds
SEI_CONNECTION_POOL_SIZE=10              # Connection pool size

# Authentication
SEI_PRIVATE_KEY=                         # Private key for transactions (required for live trading)
SEI_ADDRESS=                             # Sei wallet address
SEI_KEYRING_BACKEND=test                 # Keyring backend: test, file, os

# Gas settings
SEI_GAS_PRICE=0.1usei                    # Gas price for transactions
SEI_GAS_ADJUSTMENT=1.3                   # Gas adjustment multiplier
SEI_MAX_GAS=200000                       # Maximum gas per transaction
```

### Trading Configuration

```bash
# Trading mode
TRADING_ENABLED=false                    # Enable/disable trading
PAPER_TRADING=true                       # Paper trading mode (no real orders)
TRADING_MODE=market_making               # market_making, arbitrage, grid

# Position and risk limits
TRADING_MAX_POSITION_USDC=2000          # Max position size per market in USDC
TRADING_TOTAL_POSITION_LIMIT=5000       # Total portfolio position limit
TRADING_MAX_DRAWDOWN_PERCENT=5.0        # Maximum allowed drawdown
TRADING_STOP_LOSS_PERCENT=2.0           # Stop loss percentage

# Quote settings
TRADING_QUOTE_FREQUENCY_HZ=5            # Quote update frequency
TRADING_MIN_SPREAD_BPS=5                # Minimum spread in basis points
TRADING_MAX_SPREAD_BPS=100              # Maximum spread in basis points
TRADING_DEFAULT_ORDER_SIZE=100          # Default order size
TRADING_MAX_ORDER_SIZE=1000             # Maximum order size

# Supported trading pairs (comma-separated)
TRADING_SYMBOLS=SEI/USDC,ETH/USDC       # Trading pairs to support
TRADING_BASE_ASSETS=SEI,ETH             # Base assets
TRADING_QUOTE_ASSETS=USDC               # Quote assets

# Order management
ORDER_TIMEOUT_SECONDS=300               # Order timeout
ORDER_RETRY_ATTEMPTS=3                  # Order retry attempts
ORDER_BATCH_SIZE=10                     # Batch size for order operations
```

### Machine Learning Configuration

```bash
# Model settings
ML_MODEL_PATH=./models/                 # Path to ML models
ML_MODEL_VERSION=v1.2.0                # Model version to use
ML_INFERENCE_FREQUENCY_HZ=5            # Prediction frequency
ML_FEATURE_WINDOW_SECONDS=60           # Feature calculation window
ML_PREDICTION_HORIZON_MS=200           # Prediction time horizon

# Model performance
ML_MIN_CONFIDENCE=0.6                  # Minimum prediction confidence
ML_ACCURACY_THRESHOLD=0.55             # Minimum accuracy threshold
ML_MODEL_REFRESH_INTERVAL=3600         # Model refresh interval in seconds

# Azure OpenAI integration
AZURE_OPENAI_ENABLED=false             # Enable Azure OpenAI
AZURE_OPENAI_ENDPOINT=                 # Azure OpenAI endpoint
AZURE_OPENAI_API_KEY=                  # Azure OpenAI API key
AZURE_OPENAI_API_VERSION=2023-12-01-preview
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o    # Deployment name
AZURE_OPENAI_MAX_TOKENS=1000           # Max tokens per request
AZURE_OPENAI_TEMPERATURE=0.1           # Response temperature

# Fallback settings
ML_FALLBACK_ENABLED=true               # Enable rule-based fallback
ML_FALLBACK_MODE=conservative           # conservative, aggressive, neutral
```

### Data Storage Configuration

#### Database Settings

```bash
# PostgreSQL (primary database)
DATABASE_URL=postgresql://user:pass@localhost:5432/flashmm
DATABASE_MIN_CONNECTIONS=5             # Minimum pool connections
DATABASE_MAX_CONNECTIONS=20            # Maximum pool connections
DATABASE_CONNECTION_TIMEOUT=30         # Connection timeout
DATABASE_COMMAND_TIMEOUT=60            # Command timeout
DATABASE_SSL_MODE=prefer               # SSL mode: disable, allow, prefer, require

# Database optimization
DATABASE_POOL_RECYCLE=3600             # Connection recycle time
DATABASE_POOL_PRE_PING=true            # Test connections before use
DATABASE_ECHO=false                    # Log SQL queries (debug only)
```

#### Redis Configuration

```bash
# Redis (caching and real-time data)
REDIS_URL=redis://localhost:6379/0     # Redis connection URL
REDIS_PASSWORD=                        # Redis password
REDIS_SSL=false                        # Use SSL connection
REDIS_CONNECTION_POOL_SIZE=50          # Connection pool size
REDIS_SOCKET_TIMEOUT=5                 # Socket timeout
REDIS_SOCKET_CONNECT_TIMEOUT=5         # Connection timeout
REDIS_RETRY_ON_TIMEOUT=true            # Retry on timeout

# Redis optimization
REDIS_MAX_CONNECTIONS=100              # Maximum connections
REDIS_HEALTH_CHECK_INTERVAL=30         # Health check interval
REDIS_DECODE_RESPONSES=true            # Auto-decode responses
```

#### InfluxDB Configuration

```bash
# InfluxDB (time-series data)
INFLUXDB_URL=http://localhost:8086     # InfluxDB URL
INFLUXDB_TOKEN=                        # InfluxDB authentication token
INFLUXDB_ORG=flashmm                   # InfluxDB organization
INFLUXDB_BUCKET=metrics                # Default bucket name
INFLUXDB_TIMEOUT=30                    # Request timeout
INFLUXDB_VERIFY_SSL=true               # Verify SSL certificates

# Data retention
INFLUXDB_RETENTION_POLICY=7d           # Data retention period
INFLUXDB_BATCH_SIZE=1000               # Batch size for writes
INFLUXDB_FLUSH_INTERVAL=10             # Flush interval in seconds
```

### Monitoring and Observability

```bash
# Metrics collection
METRICS_ENABLED=true                   # Enable metrics collection
METRICS_PORT=8000                      # Metrics endpoint port
METRICS_PATH=/metrics                  # Metrics endpoint path
METRICS_UPDATE_INTERVAL=15             # Metrics update interval in seconds

# Prometheus settings
PROMETHEUS_ENABLED=true                # Enable Prometheus metrics
PROMETHEUS_JOB_NAME=flashmm           # Job name in Prometheus
PROMETHEUS_INSTANCE=flashmm-01        # Instance name

# Grafana integration
GRAFANA_URL=http://localhost:3000     # Grafana URL
GRAFANA_API_KEY=                      # Grafana API key
GRAFANA_DASHBOARD_ID=flashmm-overview # Default dashboard

# Logging
LOG_FILE_PATH=./logs/flashmm.log      # Log file path
LOG_MAX_SIZE_MB=100                   # Max log file size
LOG_BACKUP_COUNT=5                    # Number of backup files
LOG_ROTATION_INTERVAL=daily           # daily, weekly, monthly

# Alerting
ALERT_ENABLED=true                    # Enable alerting
ALERT_EMAIL_ENABLED=false             # Email alerts
ALERT_SLACK_ENABLED=false             # Slack alerts
ALERT_WEBHOOK_ENABLED=false           # Webhook alerts

# Social media integration
TWITTER_ENABLED=false                 # Enable Twitter integration
TWITTER_API_KEY=                      # Twitter API key
TWITTER_API_SECRET=                   # Twitter API secret
TWITTER_ACCESS_TOKEN=                 # Twitter access token
TWITTER_ACCESS_SECRET=                # Twitter access token secret
TWITTER_POST_FREQUENCY=3600           # Post frequency in seconds
```

### Security Configuration

```bash
# Authentication
JWT_SECRET_KEY=                       # JWT secret key (required)
JWT_ALGORITHM=HS256                   # JWT algorithm
JWT_EXPIRATION_HOURS=24               # JWT token expiration
API_KEY_HEADER=X-API-Key              # API key header name

# Rate limiting
RATE_LIMIT_ENABLED=true               # Enable rate limiting
RATE_LIMIT_REQUESTS_PER_MINUTE=100    # Requests per minute per IP
RATE_LIMIT_BURST=10                   # Burst allowance
RATE_LIMIT_STORAGE=redis              # Storage backend: memory, redis

# HTTPS and SSL
SSL_ENABLED=false                     # Enable SSL
SSL_CERT_PATH=                        # SSL certificate path
SSL_KEY_PATH=                         # SSL private key path
SSL_CA_PATH=                          # SSL CA certificate path

# Data encryption
ENCRYPTION_KEY=                       # Data encryption key
ENCRYPTION_ALGORITHM=AES-256-GCM      # Encryption algorithm
HASH_ALGORITHM=SHA-256                # Hash algorithm

# Security headers
SECURITY_HEADERS_ENABLED=true         # Enable security headers
HSTS_ENABLED=true                     # HTTP Strict Transport Security
CSP_ENABLED=true                      # Content Security Policy
```

---

## Configuration Files

### Base Configuration Structure

```yaml
# config/base.yml - Base configuration shared across environments
app:
  name: "FlashMM"
  version: "1.2.0"
  description: "AI-Powered Market Making Agent"
  
logging:
  level: "INFO"
  format: "json"
  handlers:
    - console
    - file
  
database:
  pool:
    min_connections: 5
    max_connections: 20
    timeout: 30
  
redis:
    connection_pool_size: 50
    socket_timeout: 5
    
trading:
  default_settings:
    quote_frequency_hz: 5
    min_spread_bps: 5
    max_spread_bps: 100
    
ml:
  feature_window_seconds: 60
  prediction_horizon_ms: 200
  min_confidence: 0.6
```

### Environment-Specific Configuration

#### Development Configuration

```yaml
# config/environments/development.yml
extends: base.yml

app:
  debug: true
  reload: true
  
logging:
  level: "DEBUG"
  handlers:
    - console
    
database:
  url: "postgresql://flashmm:dev@localhost:5433/flashmm_dev"
  echo: true  # Log SQL queries
  
redis:
  url: "redis://localhost:6379/0"
  
trading:
  enabled: false
  paper_trading: true
  max_position_usdc: 100
  symbols:
    - "SEI/USDC"
    
ml:
  azure_openai_enabled: false
  fallback_enabled: true
  
monitoring:
  metrics_enabled: true
  grafana_url: "http://localhost:3000"
  
security:
  rate_limit_enabled: false
  cors_origins: ["*"]
```

#### Staging Configuration

```yaml
# config/environments/staging.yml
extends: base.yml

app:
  debug: false
  
logging:
  level: "INFO"
  handlers:
    - console
    - file
  file:
    path: "/var/log/flashmm/staging.log"
    
database:
  url: "${STAGING_DATABASE_URL}"
  ssl_mode: "require"
  
redis:
  url: "${STAGING_REDIS_URL}"
  ssl: true
  
trading:
  enabled: true
  paper_trading: false
  max_position_usdc: 1000
  symbols:
    - "SEI/USDC"
    - "ETH/USDC"
    
ml:
  azure_openai_enabled: true
  model_version: "latest"
  
monitoring:
  metrics_enabled: true
  grafana_url: "https://grafana-staging.flashmm.com"
  alert_enabled: true
  
security:
  rate_limit_enabled: true
  rate_limit_requests_per_minute: 1000
  cors_origins: 
    - "https://dashboard-staging.flashmm.com"
```

#### Production Configuration

```yaml
# config/environments/production.yml
extends: base.yml

app:
  debug: false
  workers: 4
  
logging:
  level: "WARNING"
  handlers:
    - file
    - syslog
  file:
    path: "/var/log/flashmm/production.log"
    max_size_mb: 100
    backup_count: 10
    
database:
  url: "${PRODUCTION_DATABASE_URL}"
  ssl_mode: "require"
  pool:
    min_connections: 10
    max_connections: 50
    
redis:
  url: "${PRODUCTION_REDIS_URL}"
  ssl: true
  connection_pool_size: 100
  
trading:
  enabled: true
  paper_trading: false
  max_position_usdc: 10000
  symbols:
    - "SEI/USDC"
    - "ETH/USDC"
    - "BTC/USDC"
  quote_frequency_hz: 5
  
ml:
  azure_openai_enabled: true
  model_version: "v1.2.0"
  inference_frequency_hz: 5
  
monitoring:
  metrics_enabled: true
  grafana_url: "https://grafana.flashmm.com"
  alert_enabled: true
  prometheus_enabled: true
  
security:
  rate_limit_enabled: true
  rate_limit_requests_per_minute: 300
  cors_origins: 
    - "https://dashboard.flashmm.com"
  hsts_enabled: true
  csp_enabled: true
```

---

## Component Configuration

### Sei Blockchain Integration

#### Connection Configuration

```yaml
sei:
  network: "testnet"  # testnet, mainnet
  chain_id: "atlantic-2"
  
  # Primary endpoints
  rpc_url: "https://sei-testnet-rpc.polkachu.com"
  ws_url: "wss://sei-testnet-rpc.polkachu.com/websocket"
  
  # Backup endpoints for failover
  backup_endpoints:
    - rpc: "https://sei-testnet-rpc.lavenderfive.com"
      ws: "wss://sei-testnet-rpc.lavenderfive.com/websocket"
    - rpc: "https://rpc-sei.stingray.plus"
      ws: "wss://ws-sei.stingray.plus/websocket"
  
  # Connection settings
  connection:
    timeout: 30
    read_timeout: 10
    max_retries: 3
    retry_delay: 1
    pool_size: 10
    keepalive: true
  
  # Authentication
  auth:
    private_key: "${SEI_PRIVATE_KEY}"
    keyring_backend: "test"
  
  # Transaction settings
  gas:
    price: "0.1usei"
    adjustment: 1.3
    max_gas: 200000
  
  # Market data settings
  market_data:
    symbols:
      - symbol: "SEI/USDC"
        contract_address: "sei1abc..."
        decimals: 6
      - symbol: "ETH/USDC"
        contract_address: "sei1def..."
        decimals: 6
```

#### Market Configuration

```yaml
markets:
  SEI/USDC:
    enabled: true
    base_asset: "SEI"
    quote_asset: "USDC"
    min_order_size: 1.0
    max_order_size: 10000.0
    tick_size: 0.0001
    lot_size: 0.1
    
    # Risk parameters
    max_position: 2000.0  # USDC
    position_limit_percent: 2.0
    
    # Trading parameters
    min_spread_bps: 5
    max_spread_bps: 50
    default_order_size: 100
    
    # Market making settings
    quote_levels: 3
    size_increment: 0.1
    spread_increment_bps: 2
```

### Trading Engine Configuration

```yaml
trading:
  # Global settings
  enabled: true
  mode: "market_making"  # market_making, arbitrage, grid
  
  # Risk management
  risk:
    max_position_usdc: 10000
    max_drawdown_percent: 5.0
    stop_loss_percent: 2.0
    position_check_frequency_hz: 10
    
    # Circuit breakers
    circuit_breakers:
      high_latency:
        enabled: true
        threshold_ms: 500
        duration_seconds: 60
      
      low_accuracy:
        enabled: true
        threshold_percent: 50.0
        window_minutes: 30
      
      volatility:
        enabled: true
        threshold_percent: 10.0
        window_minutes: 5
  
  # Quote generation
  quoting:
    frequency_hz: 5
    update_threshold_bps: 1
    min_spread_bps: 5
    max_spread_bps: 100
    
    # Inventory skew
    inventory_skew:
      enabled: true
      max_skew_percent: 50.0
      skew_multiplier: 1.5
  
  # Order management
  orders:
    timeout_seconds: 300
    max_active_orders: 10
    batch_size: 5
    retry_attempts: 3
    
    # Order types
    default_type: "limit"
    time_in_force: "GTC"  # GTC, IOC, FOK
```

### ML Model Configuration

```yaml
ml:
  # Model settings
  model:
    path: "./models/"
    version: "v1.2.0"
    type: "transformer"  # transformer, lstm, ensemble
    
    # Performance requirements
    max_inference_time_ms: 5
    min_accuracy: 0.55
    confidence_threshold: 0.6
  
  # Feature engineering
  features:
    window_seconds: 60
    update_frequency_hz: 5
    
    # Feature types
    price_features:
      - momentum
      - volatility
      - returns
    
    volume_features:
      - volume_profile
      - volume_imbalance
      - vwap
    
    orderbook_features:
      - spread
      - depth
      - imbalance
      - slope
  
  # Prediction settings
  prediction:
    horizon_ms: 200
    frequency_hz: 5
    
    # Output format
    classes:
      - "bearish"    # -1
      - "neutral"    # 0  
      - "bullish"    # 1
  
  # Azure OpenAI integration
  azure_openai:
    enabled: false
    endpoint: "${AZURE_OPENAI_ENDPOINT}"
    api_key: "${AZURE_OPENAI_API_KEY}"
    deployment: "gpt-4o"
    max_tokens: 1000
    temperature: 0.1
    
    # Prompt configuration
    prompts:
      market_analysis: "market_analysis_v1"
      prediction_enhancement: "prediction_enhancement_v1"
  
  # Fallback engine
  fallback:
    enabled: true
    mode: "conservative"  # conservative, aggressive, neutral
    
    # Rule-based parameters
    rules:
      momentum_threshold: 0.02
      volatility_threshold: 0.05
      volume_threshold: 1.5
```

### Monitoring Configuration

```yaml
monitoring:
  # Metrics collection
  metrics:
    enabled: true
    update_interval: 15
    retention_days: 30
    
    # Metric types
    trading_metrics:
      - pnl
      - volume
      - fill_rate
      - spread_improvement
    
    system_metrics:
      - cpu_usage
      - memory_usage
      - latency
      - error_rate
    
    ml_metrics:
      - accuracy
      - confidence
      - inference_time
      - prediction_count
  
  # Alerting
  alerts:
    enabled: true
    
    # Alert channels
    channels:
      email:
        enabled: false
        smtp_server: "smtp.gmail.com"
        smtp_port: 587
        username: "${EMAIL_USERNAME}"
        password: "${EMAIL_PASSWORD}"
        recipients: ["admin@flashmm.com"]
      
      slack:
        enabled: false
        webhook_url: "${SLACK_WEBHOOK_URL}"
        channel: "#alerts"
      
      webhook:
        enabled: false
        url: "${WEBHOOK_URL}"
        timeout: 10
    
    # Alert rules
    rules:
      high_latency:
        metric: "latency_p95_ms"
        operator: ">"
        threshold: 350
        duration: "2m"
        severity: "warning"
      
      low_accuracy:
        metric: "ml_accuracy"
        operator: "<"
        threshold: 0.55
        duration: "10m"
        severity: "warning"
      
      position_limit:
        metric: "position_utilization_percent"
        operator: ">"
        threshold: 90
        duration: "1m"
        severity: "critical"
  
  # Dashboards
  dashboards:
    grafana:
      enabled: true
      url: "https://grafana.flashmm.com"
      api_key: "${GRAFANA_API_KEY}"
      
      # Dashboard definitions
      dashboards:
        - name: "FlashMM Overview"
          file: "flashmm-overview.json"
        - name: "Trading Performance"
          file: "trading-performance.json"
        - name: "ML Model Monitoring"
          file: "ml-monitoring.json"
```

---

## Environment-Specific Settings

### Development Environment

```bash
# .env.development
ENVIRONMENT=development
FLASHMM_DEBUG=true
FLASHMM_LOG_LEVEL=DEBUG

# Database
DATABASE_URL=postgresql://flashmm:dev@localhost:5433/flashmm_dev
REDIS_URL=redis://localhost:6379/0

# Trading (disabled by default)
TRADING_ENABLED=false
PAPER_TRADING=true
TRADING_MAX_POSITION_USDC=100

# External services (mocked)
AZURE_OPENAI_ENABLED=false
TWITTER_ENABLED=false
GRAFANA_URL=http://localhost:3000

# Security (relaxed for development)
RATE_LIMIT_ENABLED=false
CORS_ORIGINS=*
JWT_SECRET_KEY=dev-secret-key-change-in-production
```

### Staging Environment

```bash
# .env.staging
ENVIRONMENT=staging
FLASHMM_DEBUG=false
FLASHMM_LOG_LEVEL=INFO

# Database (managed service)
DATABASE_URL=${STAGING_DATABASE_URL}
REDIS_URL=${STAGING_REDIS_URL}

# Trading (limited)
TRADING_ENABLED=true
PAPER_TRADING=false
TRADING_MAX_POSITION_USDC=1000

# External services
AZURE_OPENAI_ENABLED=true
AZURE_OPENAI_ENDPOINT=${STAGING_AZURE_OPENAI_ENDPOINT}
TWITTER_ENABLED=true
GRAFANA_URL=https://grafana-staging.flashmm.com

# Security
RATE_LIMIT_ENABLED=true
RATE_LIMIT_REQUESTS_PER_MINUTE=1000
CORS_ORIGINS=https://dashboard-staging.flashmm.com
JWT_SECRET_KEY=${STAGING_JWT_SECRET_KEY}
```

### Production Environment

```bash
# .env.production
ENVIRONMENT=production
FLASHMM_DEBUG=false
FLASHMM_LOG_LEVEL=WARNING

# Database (high availability)
DATABASE_URL=${PRODUCTION_DATABASE_URL}
REDIS_URL=${PRODUCTION_REDIS_URL}

# Trading (full capacity)
TRADING_ENABLED=true
PAPER_TRADING=false
TRADING_MAX_POSITION_USDC=10000

# External services
AZURE_OPENAI_ENABLED=true
AZURE_OPENAI_ENDPOINT=${PRODUCTION_AZURE_OPENAI_ENDPOINT}
TWITTER_ENABLED=true
GRAFANA_URL=https://grafana.flashmm.com

# Security (strict)
RATE_LIMIT_ENABLED=true
RATE_LIMIT_REQUESTS_PER_MINUTE=300
CORS_ORIGINS=https://dashboard.flashmm.com
JWT_SECRET_KEY=${PRODUCTION_JWT_SECRET_KEY}
SSL_ENABLED=true
HSTS_ENABLED=true
```

---

## Security Configuration

### Authentication and Authorization

```yaml
security:
  # JWT Configuration
  jwt:
    secret_key: "${JWT_SECRET_KEY}"
    algorithm: "HS256"
    expiration_hours: 24
    issuer: "flashmm"
    audience: "flashmm-api"
  
  # API Key Authentication
  api_keys:
    header_name: "X-API-Key"
    query_param: "api_key"
    
    # Key validation
    min_length: 32
    required_permissions:
      - "read"    # View data
      - "write"   # Modify settings
      - "admin"   # Full access
  
  # Rate Limiting
  rate_limiting:
    enabled: true
    storage: "redis"
    
    # Global limits
    global:
      requests_per_minute: 1000
      burst: 50
    
    # Per-endpoint limits
    endpoints:
      "/api/v1/trading/orders":
        requests_per_minute: 60
        burst: 10
      "/api/v1/admin/*":
        requests_per_minute: 30
        burst: 5
  
  # CORS Configuration
  cors:
    allow_origins:
      - "https://dashboard.flashmm.com"
      - "https://grafana.flashmm.com"
    allow_methods: ["GET", "POST", "PUT", "DELETE"]
    allow_headers: ["*"]
    allow_credentials: true
    max_age: 3600
  
  # Security Headers
  headers:
    hsts:
      enabled: true
      max_age: 31536000
      include_subdomains: true
    
    csp:
      enabled: true
      policy: "default-src 'self'; script-src 'self' 'unsafe-inline'"
    
    frame_options: "DENY"
    content_type_options: "nosniff"
    xss_protection: "1; mode=block"
```

### Data Encryption

```yaml
encryption:
  # Encryption at rest
  at_rest:
    algorithm: "AES-256-GCM"
    key_derivation: "PBKDF2"
    iterations: 100000
    
    # Fields to encrypt
    encrypted_fields:
      - "private_keys"
      - "api_keys"
      - "passwords"
      - "tokens"
  
  # Encryption in transit
  in_transit:
    tls_version: "1.3"
    cipher_suites:
      - "TLS_AES_256_GCM_SHA384"
      - "TLS_CHACHA20_POLY1305_SHA256"
    
    # Certificate configuration
    certificates:
      cert_file: "/etc/ssl/certs/flashmm.crt"
      key_file: "/etc/ssl/private/flashmm.key"
      ca_file: "/etc/ssl/certs/ca.crt"
```

### Key Management

```yaml
key_management:
  # Key rotation
  rotation:
    jwt_secret:
      interval_days: 30
      overlap_days: 7  # Allow old keys during transition
    
    api_keys:
      interval_days:
90
      auto_revoke_unused_days: 90
    
    encryption_keys:
      interval_days: 365
      backup_count: 3
  
  # Key storage
  storage:
    type: "aws_secrets_manager"  # aws_secrets_manager, hashicorp_vault, file
    
    aws_secrets_manager:
      region: "us-east-1"
      secret_name_prefix: "flashmm/"
    
    hashicorp_vault:
      url: "https://vault.example.com"
      path: "flashmm/"
      auth_method: "aws"
    
    file:
      path: "/etc/flashmm/keys/"
      permissions: "0600"
```

---

## Performance Tuning

### Application Performance

```yaml
performance:
  # AsyncIO optimization
  asyncio:
    policy: "uvloop"  # uvloop, asyncio
    debug: false
    
    # Event loop settings
    loop_settings:
      slow_callback_duration: 0.1
      enable_async_generators: true
  
  # HTTP server optimization
  http_server:
    workers: 4
    worker_class: "uvicorn.workers.UvicornWorker"
    max_requests: 1000
    max_requests_jitter: 50
    timeout: 30
    keepalive: 2
    
    # Connection settings
    backlog: 2048
    max_connections: 1000
    h11_max_incomplete_event_size: 16384
  
  # Database optimization
  database:
    # Connection pooling
    pool:
      size: 20
      max_overflow: 30
      pool_recycle: 3600
      pool_pre_ping: true
    
    # Query optimization
    query:
      statement_timeout: 30000  # milliseconds
      lock_timeout: 5000
      idle_in_transaction_session_timeout: 600000
    
    # Bulk operations
    bulk:
      batch_size: 1000
      use_copy: true  # Use COPY for bulk inserts
      parallel_workers: 4
  
  # Redis optimization
  redis:
    # Connection settings
    connection:
      socket_keepalive: true
      socket_keepalive_options: {}
      health_check_interval: 30
    
    # Pipeline settings
    pipeline:
      buffer_size: 65536
      pack_response: true
    
    # Memory optimization
    memory:
      maxmemory_policy: "allkeys-lru"
      timeout: 300
      tcp_keepalive: 60
  
  # ML inference optimization
  ml:
    # Model optimization
    model:
      batch_size: 32
      num_threads: 4
      optimization_level: "O2"
      
      # Memory management
      memory_pool_size: 1024  # MB
      gc_frequency: 100  # predictions
    
    # Feature computation
    features:
      cache_size: 10000
      parallel_computation: true
      vectorization: true
```

### System-Level Optimization

```bash
# System configuration for high-performance trading

# Kernel parameters (add to /etc/sysctl.conf)
net.core.rmem_max=16777216
net.core.wmem_max=16777216
net.core.netdev_max_backlog=5000
net.ipv4.tcp_congestion_control=bbr
net.ipv4.tcp_rmem=4096 16384 16777216
net.ipv4.tcp_wmem=4096 16384 16777216

# CPU scaling governor
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Disable transparent huge pages
echo never > /sys/kernel/mm/transparent_hugepage/enabled

# Increase file descriptor limits
ulimit -n 65536

# Set process priority
nice -n -10 python -m flashmm.main

# CPU affinity (pin to specific cores)
taskset -c 0-3 python -m flashmm.main
```

### Container Optimization

```yaml
# Docker container optimization
version: '3.8'
services:
  flashmm:
    image: flashmm:latest
    
    # Resource limits
    deploy:
      resources:
        limits:
          cpus: '4.0'
          memory: 8G
        reservations:
          cpus: '2.0'
          memory: 4G
    
    # Security and performance
    security_opt:
      - no-new-privileges:true
    cap_drop:
      - ALL
    cap_add:
      - NET_BIND_SERVICE
    
    # Shared memory for better performance
    shm_size: 2gb
    
    # CPU affinity
    cpuset: "0-3"
    
    # Environment optimization
    environment:
      - PYTHONOPTIMIZE=2
      - PYTHONUNBUFFERED=1
      - PYTHONDONTWRITEBYTECODE=1
      - MALLOC_ARENA_MAX=2
      - UVLOOP_ENABLED=true
    
    # Volume mounts for performance
    volumes:
      - type: tmpfs
        target: /tmp
        tmpfs:
          size: 1G
          noexec: true
          nosuid: true
    
    # Network optimization
    networks:
      - flashmm-network
    
networks:
  flashmm-network:
    driver: bridge
    driver_opts:
      com.docker.network.driver.mtu: 9000  # Jumbo frames
```

---

## Validation and Troubleshooting

### Configuration Validation

#### Built-in Validation

```python
# Configuration validation script
# scripts/validate_config.py

import asyncio
import sys
from pathlib import Path

from flashmm.config.settings import Settings
from flashmm.config.validator import ConfigValidator

async def validate_configuration():
    """Validate FlashMM configuration."""
    
    print("🔍 Validating FlashMM configuration...")
    
    try:
        # Load configuration
        settings = Settings()
        validator = ConfigValidator()
        
        # Basic validation
        print("✓ Configuration loaded successfully")
        
        # Validate environment-specific settings
        env_issues = await validator.validate_environment(settings.environment)
        if env_issues:
            print("⚠️  Environment validation issues:")
            for issue in env_issues:
                print(f"  - {issue}")
        
        # Validate trading configuration
        if settings.trading.enabled:
            trading_issues = await validator.validate_trading_config(settings.trading)
            if trading_issues:
                print("⚠️  Trading configuration issues:")
                for issue in trading_issues:
                    print(f"  - {issue}")
        
        # Validate database connectivity
        db_status = await validator.validate_database_connection(settings.database_url)
        if db_status:
            print("✓ Database connection validated")
        else:
            print("❌ Database connection failed")
        
        # Validate Redis connectivity
        redis_status = await validator.validate_redis_connection(settings.redis_url)
        if redis_status:
            print("✓ Redis connection validated")
        else:
            print("❌ Redis connection failed")
        
        # Validate Sei network connectivity
        if settings.sei.network:
            sei_status = await validator.validate_sei_connection(settings.sei)
            if sei_status:
                print("✓ Sei network connection validated")
            else:
                print("❌ Sei network connection failed")
        
        # Validate ML model
        if settings.ml.model_path:
            model_status = await validator.validate_ml_model(settings.ml)
            if model_status:
                print("✓ ML model validated")
            else:
                print("❌ ML model validation failed")
        
        print("\n✅ Configuration validation completed")
        
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(validate_configuration())
```

#### Manual Validation Checklist

```bash
# Configuration validation checklist
# scripts/validation_checklist.sh

#!/bin/bash
set -euo pipefail

echo "📋 FlashMM Configuration Validation Checklist"
echo "=============================================="

# Check environment variables
echo "🔍 Checking environment variables..."
required_vars=(
    "ENVIRONMENT"
    "DATABASE_URL"
    "REDIS_URL"
    "SEI_RPC_URL"
    "JWT_SECRET_KEY"
)

for var in "${required_vars[@]}"; do
    if [[ -z "${!var:-}" ]]; then
        echo "❌ Missing required environment variable: $var"
        exit 1
    else
        echo "✓ $var is set"
    fi
done

# Check configuration files
echo "📄 Checking configuration files..."
config_files=(
    "config/base.yml"
    "config/environments/${ENVIRONMENT}.yml"
)

for file in "${config_files[@]}"; do
    if [[ -f "$file" ]]; then
        echo "✓ $file exists"
        # Validate YAML syntax
        python -c "import yaml; yaml.safe_load(open('$file'))" && echo "✓ $file has valid YAML syntax" || echo "❌ $file has invalid YAML syntax"
    else
        echo "❌ Missing configuration file: $file"
    fi
done

# Check network connectivity
echo "🌐 Checking network connectivity..."
if curl -s --max-time 5 "$SEI_RPC_URL/status" > /dev/null; then
    echo "✓ Sei RPC endpoint is reachable"
else
    echo "❌ Cannot reach Sei RPC endpoint"
fi

# Check database connection
echo "💾 Checking database connection..."
if python -c "
import asyncio
import asyncpg

async def test_db():
    try:
        conn = await asyncpg.connect('$DATABASE_URL')
        await conn.execute('SELECT 1')
        await conn.close()
        return True
    except:
        return False

result = asyncio.run(test_db())
exit(0 if result else 1)
"; then
    echo "✓ Database connection successful"
else
    echo "❌ Database connection failed"
fi

# Check Redis connection
echo "📦 Checking Redis connection..."
if python -c "
import redis
try:
    r = redis.from_url('$REDIS_URL')
    r.ping()
    print('✓ Redis connection successful')
except:
    print('❌ Redis connection failed')
    exit(1)
"; then
    :
fi

echo ""
echo "✅ Configuration validation completed successfully"
```

### Common Configuration Issues

#### Issue 1: Database Connection Failures

```bash
# Symptoms
SQLSTATE[08006] [7] connection to server failed

# Common causes and solutions:
# 1. Incorrect DATABASE_URL format
# Correct format: postgresql://user:password@host:port/database
DATABASE_URL=postgresql://flashmm:password@localhost:5432/flashmm

# 2. SSL configuration issues
DATABASE_URL=postgresql://user:pass@host:5432/db?sslmode=require

# 3. Connection pool exhaustion
DATABASE_MAX_CONNECTIONS=50  # Increase if needed
DATABASE_POOL_RECYCLE=3600   # Recycle connections

# 4. Network connectivity
# Test connection manually:
psql postgresql://user:pass@host:5432/database -c "SELECT 1;"
```

#### Issue 2: Redis Connection Issues

```bash
# Symptoms
redis.exceptions.ConnectionError: Error connecting to Redis

# Solutions:
# 1. Check Redis URL format
REDIS_URL=redis://localhost:6379/0

# 2. For Redis with password
REDIS_URL=redis://:password@localhost:6379/0

# 3. For Redis with SSL
REDIS_URL=rediss://user:pass@host:6380/0

# 4. Connection timeout issues
REDIS_SOCKET_TIMEOUT=5
REDIS_SOCKET_CONNECT_TIMEOUT=5

# Test Redis connection:
redis-cli -u "$REDIS_URL" ping
```

#### Issue 3: Sei Network Connectivity

```bash
# Symptoms
Failed to connect to Sei network

# Solutions:
# 1. Check RPC endpoint status
curl -s https://sei-testnet-rpc.polkachu.com/status

# 2. Use backup endpoints
SEI_RPC_BACKUP_URLS=https://sei-testnet-rpc.lavenderfive.com,https://rpc-sei.stingray.plus

# 3. Adjust timeouts
SEI_CONNECTION_TIMEOUT=30
SEI_READ_TIMEOUT=10

# 4. Network issues
# Test with curl:
curl -X POST -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","method":"status","id":1}' \
  https://sei-testnet-rpc.polkachu.com
```

#### Issue 4: ML Model Loading Failures

```bash
# Symptoms
Failed to load ML model

# Solutions:
# 1. Check model file exists
ls -la ${ML_MODEL_PATH}/model.pt

# 2. Verify model format
file ${ML_MODEL_PATH}/model.pt

# 3. Check model version compatibility
ML_MODEL_VERSION=v1.2.0

# 4. Memory issues
# Increase available memory or reduce model size
ML_MODEL_BATCH_SIZE=16  # Reduce if needed
```

### Configuration Testing

```python
# tests/test_configuration.py
import pytest
from flashmm.config.settings import Settings
from flashmm.config.validator import ConfigValidator

class TestConfiguration:
    """Test configuration validation and loading."""
    
    def test_default_configuration_loads(self):
        """Test that default configuration loads without errors."""
        settings = Settings()
        assert settings.environment is not None
        assert settings.database_url is not None
    
    def test_environment_specific_configuration(self):
        """Test environment-specific configuration loading."""
        for env in ["development", "staging", "production"]:
            settings = Settings(environment=env)
            assert settings.environment == env
    
    @pytest.mark.asyncio
    async def test_database_configuration_validation(self):
        """Test database configuration validation."""
        validator = ConfigValidator()
        settings = Settings()
        
        # Test valid configuration
        result = await validator.validate_database_connection(settings.database_url)
        assert result is True
        
        # Test invalid configuration
        result = await validator.validate_database_connection("invalid://url")
        assert result is False
    
    def test_trading_configuration_validation(self):
        """Test trading configuration validation."""
        validator = ConfigValidator()
        
        # Valid configuration
        config = {
            "enabled": True,
            "max_position_usdc": 1000,
            "symbols": ["SEI/USDC"]
        }
        issues = validator.validate_trading_config(config)
        assert len(issues) == 0
        
        # Invalid configuration
        config = {
            "enabled": True,
            "max_position_usdc": -1000,  # Invalid negative value
            "symbols": []  # Empty symbols list
        }
        issues = validator.validate_trading_config(config)
        assert len(issues) > 0
```

---

## Migration Guide

### Configuration Migration Between Versions

#### Version 1.1.x to 1.2.x Migration

```bash
#!/bin/bash
# scripts/migrate_config_v1.1_to_v1.2.sh

echo "🔄 Migrating configuration from v1.1.x to v1.2.x..."

# Backup existing configuration
cp .env .env.backup.$(date +%Y%m%d_%H%M%S)
cp -r config/ config.backup.$(date +%Y%m%d_%H%M%S)/

# New environment variables in v1.2.x
echo "# v1.2.x additions" >> .env
echo "AZURE_OPENAI_ENABLED=false" >> .env
echo "ML_PREDICTION_HORIZON_MS=200" >> .env
echo "TRADING_QUOTE_FREQUENCY_HZ=5" >> .env

# Renamed variables
sed -i 's/TRADING_FREQUENCY_HZ/TRADING_QUOTE_FREQUENCY_HZ/g' .env
sed -i 's/ML_CONFIDENCE_THRESHOLD/ML_MIN_CONFIDENCE/g' .env

# Updated configuration structure
if [ -f "config/trading.yml" ]; then
    echo "Moving trading.yml to config/components/trading.yml"
    mkdir -p config/components/
    mv config/trading.yml config/components/trading.yml
fi

echo "✅ Configuration migration completed"
echo "Please review the changes and test your configuration"
```

#### Database Schema Migration

```python
# migrations/migrate_v1.1_to_v1.2.py
"""
Migration script for database schema changes from v1.1.x to v1.2.x
"""

import asyncio
import asyncpg
from pathlib import Path

async def migrate_database(database_url: str):
    """Migrate database schema from v1.1.x to v1.2.x."""
    
    conn = await asyncpg.connect(database_url)
    
    try:
        # Check current schema version
        version = await conn.fetchval(
            "SELECT version FROM schema_version ORDER BY applied_at DESC LIMIT 1"
        )
        
        if version and version >= "1.2.0":
            print("Database already at v1.2.0 or higher")
            return
        
        print(f"Migrating database from {version} to v1.2.0...")
        
        # Add new columns for enhanced ML features
        await conn.execute("""
            ALTER TABLE predictions 
            ADD COLUMN IF NOT EXISTS azure_openai_enhanced BOOLEAN DEFAULT FALSE
        """)
        
        await conn.execute("""
            ALTER TABLE trading_metrics 
            ADD COLUMN IF NOT EXISTS spread_improvement_bps DECIMAL(10,4)
        """)
        
        # Create new tables for v1.2.x features
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS ml_model_versions (
                id SERIAL PRIMARY KEY,
                version VARCHAR(50) NOT NULL,
                path VARCHAR(500) NOT NULL,
                accuracy DECIMAL(5,4),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_active BOOLEAN DEFAULT FALSE
            )
        """)
        
        # Update schema version
        await conn.execute("""
            INSERT INTO schema_version (version, applied_at) 
            VALUES ('1.2.0', CURRENT_TIMESTAMP)
        """)
        
        print("✅ Database migration completed successfully")
        
    except Exception as e:
        print(f"❌ Database migration failed: {e}")
        raise
    finally:
        await conn.close()

if __name__ == "__main__":
    import os
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("❌ DATABASE_URL environment variable not set")
        exit(1)
    
    asyncio.run(migrate_database(database_url))
```

### Environment Migration

```yaml
# Migration mapping for environment changes
migration_v1.2:
  # Renamed variables
  renamed:
    TRADING_FREQUENCY_HZ: TRADING_QUOTE_FREQUENCY_HZ
    ML_CONFIDENCE_THRESHOLD: ML_MIN_CONFIDENCE
    REDIS_CONNECTION_POOL: REDIS_CONNECTION_POOL_SIZE
  
  # New variables with defaults
  new_variables:
    AZURE_OPENAI_ENABLED: false
    ML_PREDICTION_HORIZON_MS: 200
    TRADING_QUOTE_FREQUENCY_HZ: 5
    PERFORMANCE_MONITORING_ENABLED: true
  
  # Removed variables (no longer used)
  removed:
    - OLD_LEGACY_SETTING
    - DEPRECATED_ML_MODEL_PATH
  
  # Changed formats
  format_changes:
    DATABASE_URL:
      old_format: "postgres://user:pass@host/db"
      new_format: "postgresql://user:pass@host/db"
    
    REDIS_URL:
      old_format: "redis://host:port"
      new_format: "redis://host:port/db_number"
```

---

## Best Practices

### Configuration Organization

#### 1. Environment Separation

```bash
# Separate configurations by environment
config/
├── base.yml              # Shared settings
├── environments/
│   ├── development.yml   # Development overrides
│   ├── staging.yml       # Staging overrides
│   └── production.yml    # Production overrides
└── components/
    ├── trading.yml       # Trading-specific settings
    ├── ml.yml            # ML-specific settings
    └── monitoring.yml    # Monitoring settings
```

#### 2. Secrets Management

```bash
# Never commit secrets to version control
.env*
config/secrets/
*.key
*.pem

# Use environment variables for secrets
export JWT_SECRET_KEY=$(openssl rand -base64 32)
export DATABASE_PASSWORD=$(vault read -field=password secret/flashmm/db)

# Use external secret management in production
# AWS Secrets Manager, HashiCorp Vault, etc.
```

#### 3. Configuration Validation

```python
# Validate configuration at startup
from pydantic import BaseSettings, validator

class Settings(BaseSettings):
    trading_max_position_usdc: float
    
    @validator('trading_max_position_usdc')
    def validate_position_limit(cls, v):
        if v <= 0:
            raise ValueError('Position limit must be positive')
        if v > 100000:
            raise ValueError('Position limit too high for safety')
        return v
```

#### 4. Documentation

```yaml
# Document all configuration options
trading:
  max_position_usdc: 2000  # Maximum position size in USDC per market
  # Supported values: 100-10000
  # Higher values increase risk but allow larger profits
  # Recommended: 1000-5000 for most users
```

#### 5. Version Control

```bash
# Track configuration changes
git add config/
git commit -m "feat: add Azure OpenAI integration configuration"

# Use configuration versioning
config/
├── v1.1/
└── v1.2/  # Current version
```

### Security Best Practices

#### 1. Secrets Rotation

```bash
# Implement regular secret rotation
# scripts/rotate_secrets.sh

#!/bin/bash
echo "🔄 Starting secret rotation..."

# Generate new JWT secret
NEW_JWT_SECRET=$(openssl rand -base64 32)
echo "New JWT secret generated"

# Update in secret store (example: AWS Secrets Manager)
aws secretsmanager update-secret \
  --secret-id flashmm/production/jwt \
  --secret-string "{\"secret_key\":\"$NEW_JWT_SECRET\"}"

echo "✅ Secret rotation completed"
```

#### 2. Access Control

```yaml
# Implement role-based access to configurations
access_control:
  environments:
    development:
      - developers
      - testers
    
    staging:
      - developers
      - qa_team
      - devops
    
    production:
      - devops
      - system_admins
```

#### 3. Audit Logging

```python
# Log configuration changes
import logging

config_logger = logging.getLogger("config_audit")

def update_configuration(key: str, value: str, user: str):
    config_logger.info(
        f"Configuration updated: {key} by {user}",
        extra={
            "config_key": key,
            "user": user,
            "timestamp": datetime.utcnow(),
            "action": "update"
        }
    )
```

### Performance Best Practices

#### 1. Configuration Caching

```python
# Cache configuration to avoid repeated file I/O
from functools import lru_cache

@lru_cache(maxsize=1)
def get_settings():
    return Settings()

# Use cached settings
settings = get_settings()
```

#### 2. Lazy Loading

```python
# Load expensive configurations only when needed
class Settings:
    def __init__(self):
        self._ml_model = None
    
    @property
    def ml_model(self):
        if self._ml_model is None:
            self._ml_model = self._load_ml_model()
        return self._ml_model
```

#### 3. Environment-Specific Optimization

```yaml
# Optimize settings per environment
development:
  database:
    pool_size: 5  # Smaller pool for development
  
production:
  database:
    pool_size: 50  # Larger pool for production
    pool_recycle: 3600
```

---

## Conclusion

This comprehensive configuration reference provides:

### Key Features

1. **Hierarchical Configuration**: Environment variables override files for flexibility
2. **Environment Separation**: Different settings for dev/staging/production
3. **Component Organization**: Modular configuration for different system components
4. **Security Integration**: Built-in security settings and secret management
5. **Performance Tuning**: Optimization settings for high-frequency trading
6. **Validation Framework**: Comprehensive validation and troubleshooting tools
7. **Migration Support**: Tools and guides for configuration updates
8. **Best Practices**: Professional configuration management patterns

### Quick Reference

#### Essential Environment Variables
- `ENVIRONMENT`: Set deployment environment
- `DATABASE_URL`: Database connection string
- `REDIS_URL`: Redis connection string
- `SEI_RPC_URL`: Sei blockchain RPC endpoint
- `TRADING_ENABLED`: Enable/disable trading
- `JWT_SECRET_KEY`: JWT authentication secret

#### Configuration Files
- [`config/base.yml`](config/base.yml): Shared settings
- [`config/environments/{env}.yml`](config/environments/): Environment-specific overrides
- [`.env.{environment}`](.env.template): Environment variables

#### Validation Tools
- `scripts/validate_config.py`: Comprehensive configuration validation
- `make config-check`: Quick configuration verification
- `make config-test`: Test configuration with all services

### Support Resources

- **[Architecture Documentation](ARCHITECTURE.md)**: Understanding system design
- **[Deployment Guide](DEPLOYMENT_GUIDE.md)**: Environment setup procedures
- **[Developer Guide](DEVELOPER.md)**: Development configuration patterns
- **[Operations Manual](OPERATIONS.md)**: Production configuration management
- **[API Documentation](API.md)**: API configuration options

For additional configuration support or questions, please refer to the troubleshooting section or contact the FlashMM development team.
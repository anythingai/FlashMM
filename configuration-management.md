# FlashMM Configuration Management Strategy

## Overview
FlashMM uses a hierarchical configuration system that supports multiple environments, secure secret management, and runtime configuration updates. The strategy balances security, flexibility, and ease of deployment.

## Configuration Architecture

### Configuration Hierarchy
```
1. Default Values (code) 
2. Environment-specific files (config/environments/)
3. Environment variables
4. Runtime overrides (Redis config cache)
```

**Priority**: Runtime > Environment Variables > Config Files > Defaults

### Configuration Sources

#### 1. Default Configuration (Python Code)
Base configuration embedded in the application code.

**Location**: `src/flashmm/config/constants.py`
```python
# Default configuration values
DEFAULT_CONFIG = {
    "app": {
        "name": "FlashMM",
        "version": "1.0.0",
        "debug": False,
        "log_level": "INFO"
    },
    "trading": {
        "max_position_percent": 2.0,
        "quote_frequency_hz": 5.0,
        "max_quote_levels": 3,
        "spread_buffer_bps": 5.0
    },
    "ml": {
        "inference_timeout_ms": 5,
        "batch_size": 1,
        "confidence_threshold": 0.6
    },
    "api": {
        "host": "0.0.0.0",
        "port": 8000,
        "cors_origins": ["*"],
        "rate_limit_per_minute": 60
    }
}
```

#### 2. Environment Configuration Files
YAML files for environment-specific settings.

**Location**: `config/environments/`

**Development Configuration** (`config/environments/development.yml`):
```yaml
app:
  debug: true
  log_level: DEBUG

trading:
  max_position_usdc: 100.0  # Lower limits for dev
  quote_frequency_hz: 1.0   # Slower for debugging

sei:
  network: "testnet"
  rpc_url: "https://sei-testnet-rpc.polkachu.com"
  ws_url: "wss://sei-testnet-rpc.polkachu.com/websocket"
  chain_id: "atlantic-2"

storage:
  redis:
    host: "localhost"
    port: 6379
    db: 0
  influxdb:
    host: "localhost"
    port: 8086
    database: "flashmm_dev"

monitoring:
  grafana_enabled: false
  twitter_enabled: false
```

**Testnet Configuration** (`config/environments/testnet.yml`):
```yaml
app:
  debug: false
  log_level: INFO

trading:
  max_position_usdc: 2000.0
  quote_frequency_hz: 5.0

sei:
  network: "testnet"
  rpc_url: "https://rpc.sei-apis.com"
  ws_url: "wss://rpc.sei-apis.com/websocket"
  chain_id: "pacific-1"

storage:
  redis:
    host: "redis"
    port: 6379
    db: 0
  influxdb:
    host: "influxdb"
    port: 8086
    database: "flashmm_testnet"

monitoring:
  grafana_enabled: true
  twitter_enabled: true
  alert_thresholds:
    max_drawdown_percent: 5.0
    min_uptime_percent: 95.0
```

**Production Configuration** (`config/environments/production.yml`):
```yaml
app:
  debug: false
  log_level: WARNING

trading:
  max_position_usdc: 10000.0
  quote_frequency_hz: 5.0
  risk_multiplier: 1.5

sei:
  network: "mainnet"
  rpc_url: "https://rpc.sei-apis.com"
  ws_url: "wss://rpc.sei-apis.com/websocket"
  chain_id: "pacific-1"

storage:
  redis:
    host: "redis"
    port: 6379
    db: 0
    password_required: true
  influxdb:
    host: "influxdb"
    port: 8086
    database: "flashmm_prod"
    auth_required: true

monitoring:
  grafana_enabled: true
  twitter_enabled: true
  alert_thresholds:
    max_drawdown_percent: 2.0
    min_uptime_percent: 99.0
  
security:
  encryption_required: true
  audit_logging: true
  ip_whitelist_enabled: true
```

#### 3. Environment Variables
Sensitive configuration and deployment-specific overrides.

**Environment Variable Schema**:
```bash
# Application
ENVIRONMENT=testnet                 # development, testnet, production
FLASHMM_DEBUG=false
FLASHMM_LOG_LEVEL=INFO
FLASHMM_VERSION=1.0.0

# Sei Network
SEI_NETWORK=testnet
SEI_RPC_URL=https://rpc.sei-apis.com
SEI_WS_URL=wss://rpc.sei-apis.com/websocket
SEI_CHAIN_ID=pacific-1

# Trading Configuration
TRADING_MAX_POSITION_USDC=2000.0
TRADING_QUOTE_FREQUENCY_HZ=5.0
TRADING_MAX_QUOTE_LEVELS=3

# Cambrian SDK
CAMBRIAN_API_KEY=${CAMBRIAN_API_KEY}
CAMBRIAN_SECRET_KEY=${CAMBRIAN_SECRET_KEY}
CAMBRIAN_BASE_URL=https://api.cambrian.com

# Storage
REDIS_URL=redis://redis:6379/0
REDIS_PASSWORD=${REDIS_PASSWORD}
INFLUXDB_URL=http://influxdb:8086
INFLUXDB_TOKEN=${INFLUXDB_TOKEN}
INFLUXDB_ORG=flashmm
INFLUXDB_BUCKET=metrics

# External APIs
GRAFANA_API_KEY=${GRAFANA_API_KEY}
GRAFANA_URL=https://flashmm.grafana.net
TWITTER_BEARER_TOKEN=${TWITTER_BEARER_TOKEN}
TWITTER_API_KEY=${TWITTER_API_KEY}
TWITTER_API_SECRET=${TWITTER_API_SECRET}

# Security
SECRET_KEY=${SECRET_KEY}            # For JWT tokens
ENCRYPTION_KEY=${ENCRYPTION_KEY}    # For data encryption
API_AUTH_TOKEN=${API_AUTH_TOKEN}    # For admin endpoints

# ML Model
ML_MODEL_PATH=/app/models/latest.pt
ML_CONFIDENCE_THRESHOLD=0.6
ML_INFERENCE_TIMEOUT_MS=5

# Monitoring
HEALTH_CHECK_INTERVAL_SECONDS=30
METRICS_COLLECTION_INTERVAL_SECONDS=10
ALERT_WEBHOOK_URL=${ALERT_WEBHOOK_URL}
```

#### 4. Runtime Configuration (Redis Cache)
Dynamic configuration that can be updated without restart.

**Redis Key Schema**:
```
flashmm:config:trading:max_position_usdc
flashmm:config:trading:quote_frequency_hz
flashmm:config:ml:confidence_threshold
flashmm:config:monitoring:alert_enabled
```

## Configuration Loading Strategy

### Configuration Manager Class
```python
# src/flashmm/config/settings.py
import os
import yaml
import redis
from typing import Dict, Any, Optional
from pydantic import BaseSettings, Field
from functools import lru_cache

class Settings(BaseSettings):
    """Main configuration class using Pydantic"""
    
    # App settings
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=False, env="FLASHMM_DEBUG")
    log_level: str = Field(default="INFO", env="FLASHMM_LOG_LEVEL")
    
    # Trading settings
    max_position_usdc: float = Field(default=2000.0, env="TRADING_MAX_POSITION_USDC")
    quote_frequency_hz: float = Field(default=5.0, env="TRADING_QUOTE_FREQUENCY_HZ")
    
    # Sei network
    sei_rpc_url: str = Field(env="SEI_RPC_URL")
    sei_ws_url: str = Field(env="SEI_WS_URL")
    sei_chain_id: str = Field(env="SEI_CHAIN_ID")
    
    # Secrets
    cambrian_api_key: str = Field(env="CAMBRIAN_API_KEY")
    cambrian_secret_key: str = Field(env="CAMBRIAN_SECRET_KEY")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

class ConfigManager:
    """Configuration manager with hierarchical loading"""
    
    def __init__(self):
        self.settings = Settings()
        self.redis_client = None
        self._config_cache = {}
        
    async def initialize(self):
        """Initialize configuration manager"""
        await self._load_environment_config()
        await self._setup_redis_client()
        await self._load_runtime_config()
        
    async def _load_environment_config(self):
        """Load environment-specific YAML configuration"""
        env = self.settings.environment
        config_path = f"config/environments/{env}.yml"
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                env_config = yaml.safe_load(f)
            self._merge_config(env_config)
    
    async def _setup_redis_client(self):
        """Setup Redis client for runtime config"""
        try:
            self.redis_client = redis.Redis.from_url(
                os.getenv("REDIS_URL", "redis://localhost:6379/0")
            )
            await self.redis_client.ping()
        except Exception as e:
            print(f"Redis not available for runtime config: {e}")
    
    async def _load_runtime_config(self):
        """Load runtime configuration from Redis"""
        if not self.redis_client:
            return
            
        pattern = "flashmm:config:*"
        keys = await self.redis_client.keys(pattern)
        
        for key in keys:
            value = await self.redis_client.get(key)
            config_path = key.decode().replace("flashmm:config:", "").replace(":", ".")
            self._config_cache[config_path] = self._parse_value(value.decode())
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with fallback hierarchy"""
        # 1. Check runtime config (Redis cache)
        if key in self._config_cache:
            return self._config_cache[key]
        
        # 2. Check environment variables
        env_key = key.upper().replace(".", "_")
        env_value = os.getenv(env_key)
        if env_value is not None:
            return self._parse_value(env_value)
        
        # 3. Check Pydantic settings
        if hasattr(self.settings, key):
            return getattr(self.settings, key)
        
        # 4. Return default
        return default
    
    async def set_runtime_config(self, key: str, value: Any):
        """Set runtime configuration value"""
        if self.redis_client:
            redis_key = f"flashmm:config:{key.replace('.', ':')}"
            await self.redis_client.set(redis_key, str(value))
            self._config_cache[key] = value

# Global configuration instance
@lru_cache()
def get_config() -> ConfigManager:
    return ConfigManager()
```

## Environment Management

### Environment Detection
```python
def detect_environment() -> str:
    """Auto-detect environment based on various signals"""
    
    # Explicit environment variable
    if env := os.getenv("ENVIRONMENT"):
        return env
    
    # Docker container detection
    if os.path.exists("/.dockerenv"):
        return "testnet" if "testnet" in os.getenv("HOSTNAME", "") else "production"
    
    # Development indicators
    if os.getenv("PYTHONPATH") or os.path.exists("pyproject.toml"):
        return "development"
    
    return "development"  # Safe default
```

### Environment-Specific Behaviors

#### Development Environment
- Verbose logging (DEBUG level)
- Mock external APIs when not available
- Relaxed security settings
- File-based configuration reload
- Development-specific test data

#### Testnet Environment  
- Production-like setup with test funds
- Full external API integration
- Moderate security settings
- Real-time monitoring enabled
- Performance profiling enabled

#### Production Environment
- Minimal logging (WARNING+ only)
- Full security measures enabled
- Encrypted configuration storage
- Audit logging for all actions
- High-availability setup

## Secret Management

### Secret Storage Strategy

#### Development
- `.env` files (gitignored)
- Plain text for convenience
- Shared development secrets

#### Testnet/Production
- Environment variables only
- Container orchestration secrets
- Cloud provider secret managers (future)

### Secret Rotation Process

#### 1. API Keys
```bash
# Update Cambrian SDK keys
CAMBRIAN_API_KEY=new_key_here
CAMBRIAN_SECRET_KEY=new_secret_here

# Restart required: Yes
# Downtime: < 30 seconds
```

#### 2. Database Credentials
```bash
# Update Redis password
REDIS_PASSWORD=new_password_here

# Restart required: Yes
# Migration strategy: Blue-green deployment
```

#### 3. Encryption Keys
```bash
# Update encryption key with migration
OLD_ENCRYPTION_KEY=current_key
NEW_ENCRYPTION_KEY=new_key

# Migration script required: Yes
# Downtime: Depends on data volume
```

## Configuration Validation

### Schema Validation
```python
from pydantic import BaseModel, validator
from typing import List

class TradingConfig(BaseModel):
    max_position_usdc: float
    quote_frequency_hz: float
    max_quote_levels: int
    
    @validator('max_position_usdc')
    def validate_position_limit(cls, v):
        if v <= 0 or v > 100000:
            raise ValueError("Position limit must be between 0 and 100000")
        return v
    
    @validator('quote_frequency_hz')
    def validate_frequency(cls, v):
        if v <= 0 or v > 10:
            raise ValueError("Quote frequency must be between 0 and 10 Hz")
        return v

class NetworkConfig(BaseModel):
    rpc_url: str
    ws_url: str
    chain_id: str
    
    @validator('rpc_url', 'ws_url')
    def validate_urls(cls, v):
        if not v.startswith(('http://', 'https://', 'ws://', 'wss://')):
            raise ValueError("Invalid URL format")
        return v
```

### Runtime Validation
```python
async def validate_runtime_config():
    """Validate configuration at startup and periodically"""
    
    # Check network connectivity
    async with aiohttp.ClientSession() as session:
        async with session.get(config.get('sei.rpc_url') + '/health') as resp:
            if resp.status != 200:
                raise ConfigError("Sei RPC not accessible")
    
    # Validate API keys
    cambrian_client = CambrianClient(
        api_key=config.get('cambrian.api_key'),
        secret=config.get('cambrian.secret_key')
    )
    
    if not await cambrian_client.test_connection():
        raise ConfigError("Cambrian API credentials invalid")
    
    # Validate trading parameters
    max_position = config.get('trading.max_position_usdc')
    if max_position > await get_account_balance():
        raise ConfigError("Max position exceeds account balance")
```

## Configuration Hot-Reloading

### File-Based Reload (Development)
```python
import asyncio
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class ConfigReloadHandler(FileSystemEventHandler):
    def __init__(self, config_manager):
        self.config_manager = config_manager
    
    def on_modified(self, event):
        if event.src_path.endswith('.yml'):
            asyncio.create_task(self.config_manager.reload())

# Setup file watcher
observer = Observer()
observer.schedule(ConfigReloadHandler(config), 'config/', recursive=True)
observer.start()
```

### Redis-Based Runtime Updates
```python
async def setup_config_listener():
    """Listen for runtime configuration changes"""
    
    pubsub = redis_client.pubsub()
    await pubsub.subscribe('flashmm:config:updates')
    
    async for message in pubsub.listen():
        if message['type'] == 'message':
            config_key = message['data'].decode()
            await config_manager.reload_key(config_key)
            
            # Notify relevant components
            await notify_config_change(config_key)
```

## Configuration Deployment

### Docker Configuration
```dockerfile
# Multi-stage build with configuration
FROM python:3.11-slim as config-stage
COPY config/ /app/config/
COPY .env.example /app/.env.example

FROM python:3.11-slim as runtime
COPY --from=config-stage /app/config /app/config
ENV ENVIRONMENT=testnet
ENV PYTHONPATH=/app/src
```

### Docker Compose Integration
```yaml
version: '3.8'
services:
  flashmm:
    build: .
    environment:
      - ENVIRONMENT=${ENVIRONMENT:-testnet}
      - SEI_RPC_URL=${SEI_RPC_URL}
      - CAMBRIAN_API_KEY=${CAMBRIAN_API_KEY}
    env_file:
      - .env
    volumes:
      - ./config:/app/config:ro
    depends_on:
      - redis
      - influxdb
```

### Configuration Backup & Recovery
```bash
#!/bin/bash
# Backup current configuration
kubectl create configmap flashmm-config-backup \
  --from-file=config/ \
  --dry-run=client -o yaml > config-backup-$(date +%Y%m%d).yaml

# Restore configuration
kubectl apply -f config-backup-20240101.yaml
kubectl rollout restart deployment/flashmm
```

This configuration management strategy provides flexibility for development while ensuring security and reliability in production environments.
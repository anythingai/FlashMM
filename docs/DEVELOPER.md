# FlashMM Developer Guide

## Table of Contents
- [Development Environment Setup](#development-environment-setup)
- [Code Architecture](#code-architecture)
- [Development Workflow](#development-workflow)
- [Testing Framework](#testing-framework)
- [Code Standards](#code-standards)
- [Debugging and Profiling](#debugging-and-profiling)
- [Contributing Guidelines](#contributing-guidelines)
- [API Development](#api-development)
- [ML Model Development](#ml-model-development)
- [Extension Points](#extension-points)
- [CI/CD Integration](#cicd-integration)
- [Performance Optimization](#performance-optimization)

---

## Development Environment Setup

### Prerequisites

#### System Requirements

```bash
# Minimum requirements for development
OS: Linux (Ubuntu 22.04+), macOS (12.0+), or Windows 11 with WSL2
CPU: 4+ cores (Intel i5/AMD Ryzen 5 or better)
RAM: 8GB+ (16GB recommended)
Storage: 50GB+ free space (SSD recommended)
```

#### Required Software

```bash
# Core development tools
python >= 3.11
git >= 2.30
docker >= 20.10
docker-compose >= 2.0
node >= 18.0  # For frontend development
```

### Quick Setup

#### Automated Setup Script

```bash
#!/bin/bash
# scripts/setup-dev-environment.sh

set -euo pipefail

echo "🚀 Setting up FlashMM development environment..."

# Check Python version
python_version=$(python3 --version | cut -d' ' -f2)
required_version="3.11"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python $required_version or higher required. Found: $python_version"
    exit 1
fi

# Create virtual environment
echo "📦 Creating Python virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

# Install Poetry for dependency management
echo "📚 Installing Poetry..."
curl -sSL https://install.python-poetry.org | python3 -

# Install project dependencies
echo "📦 Installing project dependencies..."
poetry install --with dev,test,docs

# Install pre-commit hooks
echo "🔧 Setting up pre-commit hooks..."
pre-commit install

# Setup local configuration
echo "⚙️ Setting up local configuration..."
cp .env.template .env.dev
cp config/environments/development.yml.template config/environments/development.yml

# Install Node.js dependencies (for frontend)
if [ -d "frontend" ]; then
    echo "🌐 Installing frontend dependencies..."
    cd frontend
    npm install
    cd ..
fi

# Start development services
echo "🐳 Starting development services..."
docker-compose -f docker-compose.dev.yml up -d

# Run initial tests
echo "🧪 Running initial tests..."
pytest tests/unit/ -v

echo "✅ Development environment setup completed!"
echo ""
echo "📋 Next steps:"
echo "1. Activate virtual environment: source .venv/bin/activate"
echo "2. Start development server: make dev-server"
echo "3. Open browser: http://localhost:8000"
echo "4. Review documentation in docs/"
```

#### Manual Setup

```bash
# 1. Clone repository
git clone https://github.com/flashmm/flashmm.git
cd flashmm

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate     # Windows

# 3. Install Poetry
curl -sSL https://install.python-poetry.org | python3 -

# 4. Install dependencies
poetry install --with dev,test,docs

# 5. Setup pre-commit hooks
pre-commit install

# 6. Configure environment
cp .env.template .env.dev
# Edit .env.dev with your settings

# 7. Start services
docker-compose -f docker-compose.dev.yml up -d

# 8. Run tests
pytest
```

### Development Tools

#### Essential Tools

```bash
# Code quality tools (installed via Poetry)
black                    # Code formatting
isort                    # Import sorting
ruff                     # Fast linting
mypy                     # Static type checking
pytest                   # Testing framework
pytest-cov              # Coverage reporting
pre-commit               # Git hooks

# Development utilities
ipython                  # Enhanced Python REPL
jupyter                  # Notebook development
debugpy                  # Debug adapter
rich                     # Enhanced terminal output
```

#### IDE Configuration

**VS Code (Recommended)**

```json
// .vscode/settings.json
{
    "python.defaultInterpreterPath": "./.venv/bin/python",
    "python.formatting.provider": "black",
    "python.linting.enabled": true,
    "python.linting.ruffEnabled": true,
    "python.linting.mypyEnabled": true,
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests"],
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    },
    "files.exclude": {
        "**/__pycache__": true,
        "**/*.pyc": true,
        ".pytest_cache": true,
        ".mypy_cache": true,
        ".coverage": true
    }
}
```

**Extensions:**
- Python
- Pylance
- Python Docstring Generator
- GitLens
- Docker
- YAML
- REST Client

#### Environment Variables

```bash
# .env.dev - Development environment variables
ENVIRONMENT=development
FLASHMM_DEBUG=true
FLASHMM_LOG_LEVEL=DEBUG

# Development database
DATABASE_URL=postgresql://flashmm:dev@localhost:5433/flashmm_dev
REDIS_URL=redis://localhost:6379/0

# Testing configuration
TESTING=false
TEST_DATABASE_URL=postgresql://flashmm:test@localhost:5434/flashmm_test

# External services (use test endpoints)
SEI_NETWORK=testnet
SEI_RPC_URL=https://sei-testnet-rpc.polkachu.com
SEI_WS_URL=wss://sei-testnet-rpc.polkachu.com/websocket

# ML development
ML_MODEL_PATH=./models/dev/
AZURE_OPENAI_ENABLED=false  # Use mock for development

# Performance profiling
ENABLE_PROFILING=true
PROFILING_OUTPUT_DIR=./profiling/
```

---

## Code Architecture

### Project Structure Deep Dive

```
src/flashmm/
├── __init__.py                 # Package initialization
├── main.py                     # Application entry point
│
├── api/                        # FastAPI application
│   ├── __init__.py
│   ├── app.py                  # FastAPI app factory
│   ├── dependencies.py         # Dependency injection
│   ├── middleware.py           # Custom middleware
│   └── routers/               # API route handlers
│       ├── __init__.py
│       ├── health.py          # Health check endpoints
│       ├── trading.py         # Trading API endpoints
│       ├── ml.py              # ML prediction endpoints
│       ├── admin.py           # Administrative endpoints
│       └── websocket.py       # WebSocket handlers
│
├── blockchain/                 # Sei blockchain integration
│   ├── __init__.py
│   ├── sei_client.py          # Core Sei network client
│   ├── account_manager.py     # Account and balance management
│   ├── order_manager.py       # Order lifecycle management
│   ├── transaction_manager.py # Transaction handling
│   ├── blockchain_monitor.py  # Network health monitoring
│   ├── blockchain_service.py  # High-level blockchain service
│   └── market_config.py       # Market configuration management
│
├── config/                     # Configuration management
│   ├── __init__.py
│   ├── settings.py            # Settings models and validation
│   ├── environments.py        # Environment-specific configs
│   └── constants.py           # Application constants
│
├── data/                       # Data ingestion and storage
│   ├── __init__.py
│   ├── market_data_service.py # High-level data service
│   ├── ingestion/             # Data ingestion pipeline
│   │   ├── __init__.py
│   │   ├── websocket_client.py # WebSocket data client
│   │   ├── data_normalizer.py  # Data format normalization
│   │   └── feed_manager.py     # Data feed coordination
│   └── storage/               # Data storage backends
│       ├── __init__.py
│       ├── data_models.py     # Data model definitions
│       ├── influxdb_client.py # InfluxDB time-series storage
│       └── redis_client.py    # Redis caching layer
│
├── ml/                         # Machine learning components
│   ├── __init__.py
│   ├── prediction_service.py  # Main prediction service
│   ├── clients/               # External ML service clients
│   │   ├── __init__.py
│   │   └── azure_openai_client.py # Azure OpenAI integration
│   ├── fallback/              # Fallback prediction engines
│   │   ├── __init__.py
│   │   └── rule_based_engine.py # Rule-based fallback
│   ├── features/              # Feature engineering
│   │   ├── __init__.py
│   │   └── feature_extractor.py # Feature extraction pipeline
│   ├── inference/             # Model inference
│   │   ├── __init__.py
│   │   └── inference_engine.py # TorchScript inference
│   ├── models/                # Model definitions
│   │   ├── __init__.py
│   │   └── prediction_models.py # ML model classes
│   ├── prompts/               # LLM prompts
│   │   ├── __init__.py
│   │   └── market_prompts.py  # Market analysis prompts
│   └── reliability/           # ML reliability components
│       ├── __init__.py
│       └── circuit_breaker.py # ML circuit breaker
│
├── monitoring/                 # Monitoring and observability
│   ├── __init__.py
│   ├── monitoring_service.py  # Main monitoring service
│   ├── performance_tracker.py # Performance tracking
│   ├── ml_metrics.py          # ML-specific metrics
│   ├── alerts/                # Alert management
│   │   ├── __init__.py
│   │   └── alert_manager.py   # Alert processing
│   ├── analytics/             # Performance analytics
│   │   ├── __init__.py
│   │   └── performance_analyzer.py # Analytics engine
│   ├── dashboards/            # Dashboard integration
│   │   ├── __init__.py
│   │   ├── dashboard_generator.py # Dashboard creation
│   │   └── grafana_client.py  # Grafana API client
│   ├── social/                # Social media integration
│   │   ├── __init__.py
│   │   └── twitter_client.py  # Twitter/X integration
│   ├── streaming/             # Real-time data streaming
│   │   ├── __init__.py
│   │   └── data_streamer.py   # WebSocket data streaming
│   └── telemetry/             # Telemetry collection
│       ├── __init__.py
│       └── metrics_collector.py # Metrics collection
│
├── risk/                       # Risk management
│   ├── __init__.py
│   ├── market_risk_monitor.py # Market risk monitoring
│   ├── operational_risk.py    # Operational risk management
│   ├── pnl_controller.py      # P&L tracking and control
│   ├── position_limits.py     # Position limit management
│   ├── circuit_breakers.py    # Circuit breaker implementations
│   └── emergency_protocols.py # Emergency procedures
│
├── security/                   # Security components
│   ├── __init__.py
│   ├── auth.py                # Authentication
│   ├── encryption.py          # Data encryption
│   ├── key_management.py      # Key management
│   └── audit.py               # Audit logging
│
├── trading/                    # Trading engine
│   ├── __init__.py
│   ├── engine/                # Core trading engine
│   │   ├── __init__.py
│   │   └── market_making_engine.py # Main trading engine
│   ├── execution/             # Order execution
│   │   ├── __init__.py
│   │   └── order_executor.py  # Order execution logic
│   ├── optimization/          # Trading optimization
│   │   ├── __init__.py
│   │   └── portfolio_optimizer.py # Portfolio optimization
│   ├── quotes/                # Quote generation
│   │   ├── __init__.py
│   │   └── quote_generator.py # Dynamic quote generation
│   ├── risk/                  # Trading-specific risk
│   │   ├── __init__.py
│   │   └── position_manager.py # Position management
│   ├── state/                 # State management
│   │   ├── __init__.py
│   │   └── trading_state.py   # Trading state tracking
│   └── strategy/              # Trading strategies
│       ├── __init__.py
│       ├── base_strategy.py   # Base strategy interface
│       └── quoting_strategy.py # Market making strategy
│
├── utils/                      # Utility functions
│   ├── __init__.py
│   ├── logging.py             # Logging configuration
│   ├── metrics.py             # Metrics utilities
│   ├── time_utils.py          # Time and timezone utilities
│   ├── math_utils.py          # Mathematical utilities
│   └── async_utils.py         # Async programming utilities
│
└── tests/                      # Test modules (mirrors src structure)
    ├── __init__.py
    ├── conftest.py            # Pytest configuration
    ├── unit/                  # Unit tests
    ├── integration/           # Integration tests
    ├── performance/           # Performance tests
    └── fixtures/              # Test fixtures and data
```

### Design Patterns and Principles

#### Dependency Injection

```python
# src/flashmm/api/dependencies.py
from typing import Annotated
from fastapi import Depends

from flashmm.blockchain.sei_client import SeiClient
from flashmm.ml.prediction_service import PredictionService
from flashmm.trading.engine.market_making_engine import MarketMakingEngine

async def get_sei_client() -> SeiClient:
    """Dependency for Sei blockchain client."""
    return SeiClient()

async def get_prediction_service(
    sei_client: Annotated[SeiClient, Depends(get_sei_client)]
) -> PredictionService:
    """Dependency for ML prediction service."""
    return PredictionService(sei_client=sei_client)

async def get_trading_engine(
    sei_client: Annotated[SeiClient, Depends(get_sei_client)],
    prediction_service: Annotated[PredictionService, Depends(get_prediction_service)]
) -> MarketMakingEngine:
    """Dependency for trading engine."""
    return MarketMakingEngine(
        sei_client=sei_client,
        prediction_service=prediction_service
    )
```

#### Service Layer Pattern

```python
# src/flashmm/trading/engine/market_making_engine.py
from abc import ABC, abstractmethod
from typing import Protocol

class TradingEngineInterface(Protocol):
    """Interface for trading engines."""
    
    async def start(self) -> None:
        """Start the trading engine."""
        ...
    
    async def stop(self) -> None:
        """Stop the trading engine."""
        ...
    
    async def get_status(self) -> TradingStatus:
        """Get current trading status."""
        ...

class MarketMakingEngine:
    """Main market making trading engine."""
    
    def __init__(
        self,
        sei_client: SeiClient,
        prediction_service: PredictionService,
        risk_manager: RiskManager,
        order_manager: OrderManager
    ):
        self.sei_client = sei_client
        self.prediction_service = prediction_service
        self.risk_manager = risk_manager
        self.order_manager = order_manager
        self._running = False
    
    async def start(self) -> None:
        """Start the trading engine with proper initialization."""
        logger.info("Starting market making engine...")
        
        # Initialize components
        await self.sei_client.initialize()
        await self.prediction_service.initialize()
        
        # Start trading loop
        self._running = True
        asyncio.create_task(self._trading_loop())
        
        logger.info("Market making engine started successfully")
    
    async def _trading_loop(self) -> None:
        """Main trading loop with error handling."""
        while self._running:
            try:
                await self._execute_trading_cycle()
                await asyncio.sleep(0.2)  # 5Hz frequency
            except Exception as e:
                logger.error(f"Trading loop error: {e}")
                await self._handle_trading_error(e)
```

#### Error Handling Strategy

```python
# src/flashmm/utils/exceptions.py
class FlashMMException(Exception):
    """Base exception for FlashMM application."""
    def __init__(self, message: str, error_code: str = None, details: dict = None):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)

class TradingException(FlashMMException):
    """Trading-related exceptions."""
    pass

class ModelException(FlashMMException):
    """ML model-related exceptions."""
    pass

class NetworkException(FlashMMException):
    """Network and connectivity exceptions."""
    pass

# Usage in services
async def place_order(self, order: Order) -> OrderResult:
    try:
        result = await self.sei_client.submit_order(order)
        return result
    except ConnectionError as e:
        raise NetworkException(
            message="Failed to connect to Sei network",
            error_code="SEI_CONNECTION_ERROR",
            details={"original_error": str(e), "order_id": order.id}
        )
    except ValueError as e:
        raise TradingException(
            message="Invalid order parameters",
            error_code="INVALID_ORDER",
            details={"order": order.dict(), "validation_error": str(e)}
        )
```

### Configuration Management

#### Settings Architecture

```python
# src/flashmm/config/settings.py
from pydantic import BaseSettings, Field, validator
from typing import Optional, List
from enum import Enum

class Environment(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"

class SeiNetworkSettings(BaseSettings):
    """Sei blockchain network configuration."""
    
    network: str = Field(default="testnet", env="SEI_NETWORK")
    rpc_url: str = Field(env="SEI_RPC_URL")
    ws_url: str = Field(env="SEI_WS_URL")
    chain_id: str = Field(env="SEI_CHAIN_ID")
    private_key: Optional[str] = Field(default=None, env="SEI_PRIVATE_KEY")
    
    @validator("network")
    def validate_network(cls, v):
        if v not in ["testnet", "mainnet"]:
            raise ValueError("Network must be 'testnet' or 'mainnet'")
        return v

class TradingSettings(BaseSettings):
    """Trading engine configuration."""
    
    enabled: bool = Field(default=False, env="TRADING_ENABLED")
    paper_trading: bool = Field(default=True, env="PAPER_TRADING")
    max_position_usdc: float = Field(default=2000.0, env="TRADING_MAX_POSITION_USDC")
    quote_frequency_hz: int = Field(default=5, env="TRADING_QUOTE_FREQUENCY_HZ")
    symbols: List[str] = Field(default=["SEI/USDC"], env="TRADING_SYMBOLS")
    
    @validator("quote_frequency_hz")
    def validate_frequency(cls, v):
        if not 1 <= v <= 20:
            raise ValueError("Quote frequency must be between 1-20 Hz")
        return v

class MLSettings(BaseSettings):
    """Machine learning configuration."""
    
    model_path: str = Field(default="./models/", env="ML_MODEL_PATH")
    inference_frequency_hz: int = Field(default=5, env="ML_INFERENCE_FREQUENCY_HZ")
    azure_openai_enabled: bool = Field(default=False, env="AZURE_OPENAI_ENABLED")
    azure_openai_endpoint: Optional[str] = Field(default=None, env="AZURE_OPENAI_ENDPOINT")
    azure_openai_api_key: Optional[str] = Field(default=None, env="AZURE_OPENAI_API_KEY")

class Settings(BaseSettings):
    """Main application settings."""
    
    environment: Environment = Field(default=Environment.DEVELOPMENT)
    debug: bool = Field(default=False, env="FLASHMM_DEBUG")
    log_level: str = Field(default="INFO", env="FLASHMM_LOG_LEVEL")
    
    # Component settings
    sei: SeiNetworkSettings = SeiNetworkSettings()
    trading: TradingSettings = TradingSettings()
    ml: MLSettings = MLSettings()
    
    # Database settings
    database_url: str = Field(env="DATABASE_URL")
    redis_url: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

# Global settings instance
settings = Settings()
```

---

## Development Workflow

### Git Workflow

#### Branch Strategy

```
main branch (production-ready)
├── develop branch (integration branch)
│   ├── feature/trading-engine-improvements
│   ├── feature/ml-model-enhancements
│   ├── feature/api-rate-limiting
│   └── bugfix/websocket-reconnection
├── release/v1.2.0 (release preparation)
└── hotfix/critical-security-patch (emergency fixes)
```

#### Commit Message Convention

```bash
# Format: <type>(<scope>): <subject>
# Types: feat, fix, docs, style, refactor, test, chore

# Examples:
feat(trading): add dynamic spread adjustment based on volatility
fix(ml): resolve prediction confidence calibration issue
docs(api): update REST endpoint documentation
refactor(blockchain): improve error handling in sei client
test(integration): add comprehensive trading engine tests
chore(deps): update pytorch to version 2.1.0
```

#### Pull Request Template

```markdown
## Description
Brief description of changes and motivation.

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Performance tests pass (if applicable)
- [ ] Manual testing completed

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review of code completed
- [ ] Code is commented, particularly in hard-to-understand areas
- [ ] Corresponding changes to documentation made
- [ ] No new warnings generated
- [ ] Added tests that prove fix is effective or feature works
- [ ] New and existing unit tests pass locally

## Performance Impact
Describe any performance implications of your changes.

## Breaking Changes
List any breaking changes and migration instructions.
```

### Code Review Process

#### Review Checklist

**Code Quality:**
- [ ] Code follows established patterns and conventions
- [ ] Proper error handling implemented
- [ ] No obvious security vulnerabilities
- [ ] Performance implications considered
- [ ] Code is testable and well-structured

**Documentation:**
- [ ] Public APIs documented with docstrings
- [ ] Complex logic explained with comments
- [ ] README updated if necessary
- [ ] API documentation updated

**Testing:**
- [ ] Adequate test coverage (>80%)
- [ ] Tests cover edge cases
- [ ] Integration tests for new features
- [ ] Performance tests for critical paths

**Security:**
- [ ] Input validation implemented
- [ ] Sensitive data handled properly
- [ ] Authentication/authorization correct
- [ ] No hardcoded secrets or credentials

### Local Development Commands

#### Makefile Targets

```makefile
# Makefile for FlashMM development

.PHONY: help install dev-setup test lint format clean build

help: ## Show this help message
	@echo "FlashMM Development Commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install project dependencies
	poetry install --with dev,test,docs

dev-setup: install ## Complete development environment setup
	pre-commit install
	cp .env.template .env.dev
	docker-compose -f docker-compose.dev.yml up -d

dev-server: ## Start development server with hot reload
	poetry run uvicorn src.flashmm.main:app --host 0.0.0.0 --port 8000 --reload --reload-dir src

test: ## Run all tests
	poetry run pytest tests/ -v --cov=src/flashmm --cov-report=html --cov-report=term

test-unit: ## Run unit tests only
	poetry run pytest tests/unit/ -v

test-integration: ## Run integration tests
	poetry run pytest tests/integration/ -v

test-performance: ## Run performance tests
	poetry run pytest tests/performance/ -v

test-watch: ## Run tests in watch mode
	poetry run pytest-watch tests/ --runner "pytest --cov=src/flashmm"

lint: ## Run linting checks
	poetry run ruff check src/ tests/
	poetry run mypy src/

format: ## Format code
	poetry run black src/ tests/
	poetry run isort src/ tests/

format-check: ## Check code formatting
	poetry run black --check src/ tests/
	poetry run isort --check-only src/ tests/

security-scan: ## Run security scan
	poetry run bandit -r src/
	poetry run safety check

type-check: ## Run type checking
	poetry run mypy src/ --strict

docs: ## Generate documentation
	poetry run sphinx-build -b html docs/ docs/_build/html

docs-serve: ## Serve documentation locally
	poetry run sphinx-autobuild docs/ docs/_build/html --port 8080

clean: ## Clean generated files
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .mypy_cache .coverage htmlcov/ docs/_build/

build: ## Build Docker image
	docker build -t flashmm:dev .

run-dev: ## Run development container
	docker-compose -f docker-compose.dev.yml up

stop-dev: ## Stop development container
	docker-compose -f docker-compose.dev.yml down

logs: ## Show development logs
	docker-compose -f docker-compose.dev.yml logs -f

shell: ## Open shell in development container
	docker-compose -f docker-compose.dev.yml exec flashmm-app /bin/bash

db-migration: ## Run database migrations
	poetry run alembic upgrade head

db-reset: ## Reset database
	poetry run alembic downgrade base
	poetry run alembic upgrade head

profile: ## Run performance profiling
	poetry run python -m cProfile -o profile.stats src/flashmm/main.py
	poetry run python -c "import pstats; pstats.Stats('profile.stats').sort_stats('cumulative').print_stats(20)"

benchmark: ## Run benchmarks
	poetry run python scripts/benchmark.py

pre-commit: ## Run pre-commit hooks manually
	poetry run pre-commit run --all-files

ci-check: format-check lint type-check security-scan test ## Run all CI checks locally
```

---

## Testing Framework

### Testing Architecture

#### Test Categories

```python
# tests/conftest.py - Global test configuration
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from flashmm.config.settings import Settings
from flashmm.blockchain.sei_client import SeiClient
from flashmm.ml.prediction_service import PredictionService

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()

@pytest.fixture
def test_settings():
    """Test-specific settings."""
    return Settings(
        environment="testing",
        debug=True,
        trading=TradingSettings(
            enabled=False,
            paper_trading=True,
            max_position_usdc=100.0
        ),
        database_url="sqlite:///test.db",
        redis_url="redis://localhost:6380/1"
    )

@pytest.fixture
def mock_sei_client():
    """Mock Sei client for testing."""
    client = AsyncMock(spec=SeiClient)
    client.get_account_info.return_value = {
        "address": "sei1test...",
        "balances": [{"denom": "usdc", "amount": "10000"}]
    }
    client.get_market_data.return_value = {
        "symbol": "SEI/USDC",
        "price": "0.04210",
        "volume": "125000"
    }
    return client

@pytest.fixture
def mock_prediction_service():
    """Mock ML prediction service."""
    service = AsyncMock(spec=PredictionService)
    service.predict.return_value = {
        "direction": "bullish",
        "confidence": 0.75,
        "price_change_bps": 5.2
    }
    return service
```

#### Unit Tests

```python
# tests/unit/test_trading_engine.py
import pytest
from unittest.mock import AsyncMock, patch
from decimal import Decimal

from flashmm.trading.engine.market_making_engine import MarketMakingEngine
from flashmm.trading.models import Quote, Order, Position

class TestMarketMakingEngine:
    """Test suite for MarketMakingEngine."""
    
    @pytest.fixture
    def trading_engine(self, mock_sei_client, mock_prediction_service):
        """Create trading engine instance for testing."""
        return MarketMakingEngine(
            sei_client=mock_sei_client,
            prediction_service=mock_prediction_service
        )
    
    @pytest.mark.asyncio
    async def test_generate_quotes_bullish_prediction(self, trading_engine):
        """Test quote generation with bullish ML prediction."""
        # Arrange
        prediction = {
            "direction": "bullish",
            "confidence": 0.78,
            "price_change_bps": 5.2
        }
        
trading_engine.prediction_service.predict.return_value = prediction
        market_data = {
            "symbol": "SEI/USDC",
            "current_price": Decimal("0.04210"),
            "bid": Decimal("0.04200"),
            "ask": Decimal("0.04220")
        }
        
        # Act
        quotes = await trading_engine.generate_quotes("SEI/USDC", market_data)
        
        # Assert
        assert len(quotes) == 2  # bid and ask
        bid_quote = next(q for q in quotes if q.side == "buy")
        ask_quote = next(q for q in quotes if q.side == "sell")
        
        # Bullish prediction should tighten ask spread, widen bid spread
        assert bid_quote.price < Decimal("0.04200")  # More aggressive bid
        assert ask_quote.price < Decimal("0.04220")  # Tighter ask
        
        # Verify confidence affects spread
        expected_spread_reduction = prediction["confidence"] * 0.3  # 30% max reduction
        assert ask_quote.spread_bps < 20 * (1 - expected_spread_reduction)
    
    @pytest.mark.asyncio
    async def test_position_limit_enforcement(self, trading_engine):
        """Test that position limits are enforced."""
        # Arrange
        current_position = Position(
            symbol="SEI/USDC",
            base_balance=Decimal("1000"),
            quote_balance=Decimal("1900"),  # Close to 2000 limit
            position_usdc=Decimal("1900")
        )
        trading_engine.get_current_position.return_value = current_position
        
        order = Order(
            symbol="SEI/USDC",
            side="buy",
            price=Decimal("0.04200"),
            size=Decimal("500")  # Would exceed position limit
        )
        
        # Act & Assert
        with pytest.raises(PositionLimitError):
            await trading_engine.place_order(order)
    
    @pytest.mark.asyncio
    async def test_emergency_stop(self, trading_engine):
        """Test emergency stop functionality."""
        # Arrange
        trading_engine._running = True
        
        # Act
        await trading_engine.emergency_stop("Test emergency stop")
        
        # Assert
        assert not trading_engine._running
        trading_engine.sei_client.cancel_all_orders.assert_called_once()
        trading_engine.risk_manager.flatten_positions.assert_called_once()
```

#### Integration Tests

```python
# tests/integration/test_trading_workflow.py
import pytest
import asyncio
from decimal import Decimal

from flashmm.trading.engine.market_making_engine import MarketMakingEngine
from flashmm.blockchain.sei_client import SeiClient
from flashmm.ml.prediction_service import PredictionService

@pytest.mark.integration
class TestTradingWorkflow:
    """Integration tests for complete trading workflows."""
    
    @pytest.fixture
    async def live_trading_engine(self, test_settings):
        """Create trading engine with live connections (testnet)."""
        sei_client = SeiClient(
            rpc_url=test_settings.sei.rpc_url,
            private_key=test_settings.sei.private_key
        )
        prediction_service = PredictionService()
        
        engine = MarketMakingEngine(
            sei_client=sei_client,
            prediction_service=prediction_service
        )
        
        await engine.initialize()
        yield engine
        await engine.cleanup()
    
    @pytest.mark.asyncio
    async def test_complete_trading_cycle(self, live_trading_engine):
        """Test a complete trading cycle from prediction to execution."""
        # This test runs against testnet
        engine = live_trading_engine
        symbol = "SEI/USDC"
        
        # 1. Get market data
        market_data = await engine.sei_client.get_market_data(symbol)
        assert market_data is not None
        assert "current_price" in market_data
        
        # 2. Generate ML prediction
        prediction = await engine.prediction_service.predict(symbol, market_data)
        assert prediction is not None
        assert prediction.confidence > 0
        
        # 3. Generate quotes based on prediction
        quotes = await engine.generate_quotes(symbol, market_data, prediction)
        assert len(quotes) >= 2  # At least bid and ask
        
        # 4. Place orders (in paper trading mode)
        if engine.settings.trading.paper_trading:
            for quote in quotes:
                order_id = await engine.place_order(quote)
                assert order_id is not None
        
        # 5. Monitor position
        position = await engine.get_current_position(symbol)
        assert position.symbol == symbol
        
    @pytest.mark.asyncio
    async def test_websocket_data_pipeline(self, live_trading_engine):
        """Test real-time data ingestion pipeline."""
        engine = live_trading_engine
        symbol = "SEI/USDC"
        
        # Subscribe to market data
        data_received = asyncio.Event()
        received_data = []
        
        async def data_handler(data):
            received_data.append(data)
            data_received.set()
        
        await engine.data_service.subscribe_market_data(symbol, data_handler)
        
        # Wait for data
        await asyncio.wait_for(data_received.wait(), timeout=30.0)
        
        # Verify data received
        assert len(received_data) > 0
        latest_data = received_data[-1]
        assert latest_data["symbol"] == symbol
        assert "timestamp" in latest_data
        assert "price" in latest_data
```

#### Performance Tests

```python
# tests/performance/test_latency.py
import pytest
import asyncio
import time
from statistics import mean, median

from flashmm.ml.inference.inference_engine import InferenceEngine
from flashmm.trading.engine.market_making_engine import MarketMakingEngine

@pytest.mark.performance
class TestLatencyRequirements:
    """Test that system meets latency requirements."""
    
    @pytest.mark.asyncio
    async def test_ml_inference_latency(self):
        """Test ML inference meets <5ms requirement."""
        engine = InferenceEngine()
        await engine.load_model("./models/test_model.pt")
        
        # Prepare test data
        feature_vector = [0.1] * 64  # 64 features
        
        # Warm-up
        for _ in range(10):
            await engine.predict(feature_vector)
        
        # Measure latency
        latencies = []
        for _ in range(100):
            start_time = time.perf_counter()
            await engine.predict(feature_vector)
            end_time = time.perf_counter()
            latencies.append((end_time - start_time) * 1000)  # Convert to ms
        
        avg_latency = mean(latencies)
        p95_latency = sorted(latencies)[94]  # 95th percentile
        
        print(f"ML Inference - Avg: {avg_latency:.2f}ms, P95: {p95_latency:.2f}ms")
        
        assert avg_latency < 5.0, f"Average latency {avg_latency:.2f}ms exceeds 5ms target"
        assert p95_latency < 8.0, f"P95 latency {p95_latency:.2f}ms exceeds 8ms threshold"
    
    @pytest.mark.asyncio
    async def test_quote_generation_latency(self, trading_engine):
        """Test quote generation meets <5ms requirement."""
        market_data = {
            "symbol": "SEI/USDC",
            "current_price": 0.04210,
            "bid": 0.04200,
            "ask": 0.04220,
            "volume": 125000
        }
        
        prediction = {
            "direction": "bullish",
            "confidence": 0.75,
            "price_change_bps": 5.2
        }
        
        # Warm-up
        for _ in range(10):
            await trading_engine.generate_quotes("SEI/USDC", market_data, prediction)
        
        # Measure latency
        latencies = []
        for _ in range(100):
            start_time = time.perf_counter()
            await trading_engine.generate_quotes("SEI/USDC", market_data, prediction)
            end_time = time.perf_counter()
            latencies.append((end_time - start_time) * 1000)
        
        avg_latency = mean(latencies)
        p95_latency = sorted(latencies)[94]
        
        print(f"Quote Generation - Avg: {avg_latency:.2f}ms, P95: {p95_latency:.2f}ms")
        
        assert avg_latency < 5.0
        assert p95_latency < 8.0
    
    @pytest.mark.asyncio
    async def test_end_to_end_latency(self, live_trading_engine):
        """Test complete trading cycle meets <350ms requirement."""
        engine = live_trading_engine
        symbol = "SEI/USDC"
        
        # Measure end-to-end latency
        latencies = []
        for _ in range(50):  # Fewer iterations for integration test
            start_time = time.perf_counter()
            
            # Complete cycle: data -> prediction -> quotes -> orders
            market_data = await engine.get_market_data(symbol)
            prediction = await engine.generate_prediction(symbol, market_data)
            quotes = await engine.generate_quotes(symbol, market_data, prediction)
            
            # Simulate order placement (don't actually place in test)
            await engine.validate_orders(quotes)
            
            end_time = time.perf_counter()
            latencies.append((end_time - start_time) * 1000)
        
        avg_latency = mean(latencies)
        p95_latency = sorted(latencies)[47]  # 95th percentile
        
        print(f"End-to-End - Avg: {avg_latency:.2f}ms, P95: {p95_latency:.2f}ms")
        
        assert avg_latency < 350.0
        assert p95_latency < 500.0
```

---

## Code Standards

### Python Code Style

#### Formatting and Linting

```python
# pyproject.toml - Tool configuration
[tool.black]
line-length = 88
target-version = ['py311']
include = '\.pyi?$'
extend-exclude = '''
/(
  # directories
  \.eggs
  | \.git
  | \.hg
  | \.mypy_cache
  | \.pytest_cache
  | \.venv
  | build
  | dist
)/
'''

[tool.isort]
profile = "black"
multi_line_output = 3
line_length = 88
known_first_party = ["flashmm"]
known_third_party = ["fastapi", "pydantic", "sqlalchemy", "torch"]

[tool.ruff]
target-version = "py311"
line-length = 88
select = [
    "E",   # pycodestyle errors
    "W",   # pycodestyle warnings
    "F",   # pyflakes
    "I",   # isort
    "B",   # flake8-bugbear
    "C4",  # flake8-comprehensions
    "UP",  # pyupgrade
]
ignore = [
    "E501",  # line too long, handled by black
    "B008",  # do not perform function calls in argument defaults
]

[tool.mypy]
python_version = "3.11"
strict = true
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
disallow_incomplete_defs = true
check_untyped_defs = true
disallow_untyped_decorators = true
no_implicit_optional = true
warn_redundant_casts = true
warn_unused_ignores = true
warn_no_return = true
warn_unreachable = true
strict_equality = true

[[tool.mypy.overrides]]
module = [
    "torch.*",
    "numpy.*",
    "pandas.*",
]
ignore_missing_imports = true
```

#### Documentation Standards

```python
# Example of proper function documentation
async def place_market_order(
    self,
    symbol: str,
    side: Literal["buy", "sell"],
    size: Decimal,
    *,
    max_slippage_bps: Optional[int] = None,
    timeout_seconds: int = 30
) -> OrderResult:
    """Place a market order with slippage protection.
    
    Args:
        symbol: Trading pair symbol (e.g., "SEI/USDC")
        side: Order side ("buy" or "sell")
        size: Order size in base asset units
        max_slippage_bps: Maximum acceptable slippage in basis points.
            If None, uses default slippage tolerance.
        timeout_seconds: Order timeout in seconds
    
    Returns:
        OrderResult containing order ID, fill information, and execution details
    
    Raises:
        TradingException: If order placement fails
        ValidationError: If parameters are invalid
        TimeoutError: If order exceeds timeout
    
    Example:
        >>> result = await trader.place_market_order(
        ...     symbol="SEI/USDC",
        ...     side="buy",
        ...     size=Decimal("1000"),
        ...     max_slippage_bps=50
        ... )
        >>> print(f"Order {result.order_id} filled at {result.avg_price}")
    """
    # Validate parameters
    if size <= 0:
        raise ValidationError("Order size must be positive")
    
    if symbol not in self.supported_symbols:
        raise ValidationError(f"Unsupported symbol: {symbol}")
    
    # Implementation...
```

#### Type Hints and Annotations

```python
from typing import Dict, List, Optional, Union, Protocol, TypeVar, Generic
from decimal import Decimal
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

# Use dataclasses for data structures
@dataclass(frozen=True)
class MarketData:
    """Market data snapshot."""
    symbol: str
    timestamp: datetime
    price: Decimal
    volume: Decimal
    bid: Decimal
    ask: Decimal
    spread_bps: float

# Use enums for constants
class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"

class OrderStatus(str, Enum):
    PENDING = "pending"
    ACTIVE = "active"
    FILLED = "filled"
    CANCELLED = "cancelled"
    EXPIRED = "expired"

# Use protocols for interfaces
class TradingEngineProtocol(Protocol):
    """Protocol for trading engine implementations."""
    
    async def start(self) -> None:
        """Start the trading engine."""
        ...
    
    async def stop(self) -> None:
        """Stop the trading engine."""
        ...
    
    async def get_status(self) -> Dict[str, Any]:
        """Get current engine status."""
        ...

# Use generics when appropriate
T = TypeVar('T')

class TimeSeriesBuffer(Generic[T]):
    """Generic time-series data buffer."""
    
    def __init__(self, max_size: int) -> None:
        self._data: List[Tuple[datetime, T]] = []
        self._max_size = max_size
    
    def add(self, timestamp: datetime, value: T) -> None:
        """Add new value to buffer."""
        self._data.append((timestamp, value))
        if len(self._data) > self._max_size:
            self._data.pop(0)
    
    def get_latest(self) -> Optional[T]:
        """Get latest value."""
        return self._data[-1][1] if self._data else None
```

### Error Handling Patterns

#### Exception Hierarchy

```python
# src/flashmm/exceptions.py
class FlashMMError(Exception):
    """Base exception for all FlashMM errors."""
    
    def __init__(
        self,
        message: str,
        *,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.cause = cause
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for serialization."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "details": self.details,
            "cause": str(self.cause) if self.cause else None
        }

class ValidationError(FlashMMError):
    """Input validation errors."""
    pass

class NetworkError(FlashMMError):
    """Network and connectivity errors."""
    pass

class TradingError(FlashMMError):
    """Trading operation errors."""
    pass

class ModelError(FlashMMError):
    """ML model errors."""
    pass

# Usage example with proper error handling
async def execute_trade(self, order: Order) -> OrderResult:
    """Execute trade with comprehensive error handling."""
    try:
        # Validate order
        self._validate_order(order)
        
        # Check risk limits
        await self._check_risk_limits(order)
        
        # Execute order
        result = await self.sei_client.place_order(order)
        
        # Update internal state
        await self._update_position(result)
        
        return result
        
    except ValidationError:
        # Re-raise validation errors as-is
        raise
    except ConnectionError as e:
        raise NetworkError(
            "Failed to connect to Sei network",
            error_code="SEI_CONNECTION_FAILED",
            details={"order_id": order.id, "network": "sei-testnet"},
            cause=e
        )
    except Exception as e:
        # Catch all other errors and wrap them
        raise TradingError(
            "Unexpected error during trade execution",
            error_code="TRADE_EXECUTION_FAILED",
            details={"order": order.dict()},
            cause=e
        )
```

---

## Debugging and Profiling

### Debugging Tools

#### Logging Configuration

```python
# src/flashmm/utils/logging.py
import logging
import sys
from typing import Optional
from pathlib import Path
import json
from datetime import datetime

class JSONFormatter(logging.Formatter):
    """JSON log formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add extra fields
        if hasattr(record, 'user_id'):
            log_entry['user_id'] = record.user_id
        if hasattr(record, 'request_id'):
            log_entry['request_id'] = record.request_id
        if hasattr(record, 'symbol'):
            log_entry['symbol'] = record.symbol
        
        # Add exception info
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry, default=str)

def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[Path] = None,
    json_format: bool = True
) -> None:
    """Setup application logging."""
    
    # Create formatters
    if json_format:
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    # Setup console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # Setup file handler if specified
    handlers = [console_handler]
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        handlers=handlers,
        force=True
    )
    
    # Configure specific loggers
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)

# Usage in modules
logger = logging.getLogger(__name__)

async def place_order(self, order: Order) -> str:
    """Place order with structured logging."""
    logger.info(
        "Placing order",
        extra={
            "symbol": order.symbol,
            "side": order.side,
            "size": float(order.size),
            "price": float(order.price)
        }
    )
    
    try:
        result = await self.sei_client.submit_order(order)
        logger.info(
            "Order placed successfully",
            extra={
                "order_id": result.order_id,
                "symbol": order.symbol
            }
        )
        return result.order_id
    except Exception as e:
        logger.error(
            "Failed to place order",
            extra={
                "symbol": order.symbol,
                "error": str(e)
            },
            exc_info=True
        )
        raise
```

#### Debug Mode Configuration

```python
# src/flashmm/config/debug.py
import asyncio
import functools
import time
from typing import Any, Callable, TypeVar
from flashmm.utils.logging import logger

F = TypeVar('F', bound=Callable[..., Any])

def debug_async(func: F) -> F:
    """Decorator for debugging async functions."""
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        func_name = f"{func.__module__}.{func.__name__}"
        
        logger.debug(f"Entering {func_name}", extra={
            "args": str(args)[:100],  # Truncate long args
            "kwargs": str(kwargs)[:100]
        })
        
        try:
            result = await func(*args, **kwargs)
            end_time = time.perf_counter()
            duration = (end_time - start_time) * 1000  # ms
            
            logger.debug(f"Exiting {func_name}", extra={
                "duration_ms": duration,
                "result_type": type(result).__name__
            })
            
            return result
        except Exception as e:
            end_time = time.perf_counter()
            duration = (end_time - start_time) * 1000
            
            logger.debug(f"Exception in {func_name}", extra={
                "duration_ms": duration,
                "exception": str(e)
            })
            raise
    
    return wrapper

def performance_monitor(threshold_ms: float = 100.0):
    """Decorator to monitor function performance."""
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            result = await func(*args, **kwargs)
            end_time = time.perf_counter()
            
            duration_ms = (end_time - start_time) * 1000
            if duration_ms > threshold_ms:
                logger.warning(
                    f"Slow function detected: {func.__name__}",
                    extra={
                        "duration_ms": duration_ms,
                        "threshold_ms": threshold_ms,
                        "function": f"{func.__module__}.{func.__name__}"
                    }
                )
            
            return result
        return wrapper
    return decorator
```

### Performance Profiling

#### Profiling Tools

```python
# scripts/profile_application.py
import cProfile
import pstats
import asyncio
from pathlib import Path
import sys
sys.path.append("src")

from flashmm.main import create_app
from flashmm.trading.engine.market_making_engine import MarketMakingEngine

async def profile_trading_cycle():
    """Profile a complete trading cycle."""
    app = create_app()
    engine = app.state.trading_engine
    
    # Profile 100 trading cycles
    for _ in range(100):
        await engine.execute_trading_cycle()

def run_profiling():
    """Run performance profiling."""
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Run the code to profile
    asyncio.run(profile_trading_cycle())
    
    profiler.disable()
    
    # Save results
    profiler.dump_stats("profile_results.prof")
    
    # Print top functions
    stats = pstats.Stats("profile_results.prof")
    stats.sort_stats("cumulative")
    stats.print_stats(20)
    
    # Print slowest functions
    print("\n" + "="*50)
    print("SLOWEST FUNCTIONS")
    print("="*50)
    stats.sort_stats("tottime")
    stats.print_stats(10)

if __name__ == "__main__":
    run_profiling()
```

#### Memory Profiling

```python
# scripts/memory_profile.py
import tracemalloc
import asyncio
import gc
from flashmm.main import create_app

async def memory_profile():
    """Profile memory usage during operation."""
    tracemalloc.start()
    
    app = create_app()
    engine = app.state.trading_engine
    
    # Take initial snapshot
    snapshot1 = tracemalloc.take_snapshot()
    
    # Run trading cycles
    for i in range(1000):
        await engine.execute_trading_cycle()
        
        if i % 100 == 0:
            gc.collect()
            snapshot2 = tracemalloc.take_snapshot()
            
            # Compare snapshots
            top_stats = snapshot2.compare_to(snapshot1, 'lineno')
            
            print(f"\nMemory usage after {i} cycles:")
            for stat in top_stats[:10]:
                print(stat)
            
            snapshot1 = snapshot2

if __name__ == "__main__":
    asyncio.run(memory_profile())
```

#### Performance Benchmarking

```python
# scripts/benchmark.py
import asyncio
import time
import statistics
from typing import List, Callable, Any
from dataclasses import dataclass

@dataclass
class BenchmarkResult:
    name: str
    iterations: int
    avg_time_ms: float
    min_time_ms: float
    max_time_ms: float
    p95_time_ms: float
    total_time_s: float

class Benchmark:
    """Performance benchmarking utility."""
    
    def __init__(self):
        self.results: List[BenchmarkResult] = []
    
    async def run_async(
        self,
        name: str,
        func: Callable,
        iterations: int = 1000,
        warmup: int = 10
    ) -> BenchmarkResult:
        """Run async function benchmark."""
        print(f"Running benchmark: {name}")
        
        # Warmup
        for _ in range(warmup):
            await func()
        
        # Actual benchmark
        times = []
        start_total = time.perf_counter()
        
        for i in range(iterations):
            start = time.perf_counter()
            await func()
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms
            
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{iterations} completed")
        
        end_total = time.perf_counter()
        
        # Calculate statistics
        result = BenchmarkResult(
            name=name,
            iterations=iterations,
            avg_time_ms=statistics.mean(times),
            min_time_ms=min(times),
            max_time_ms=max(times),
            p95_time_ms=statistics.quantiles(times, n=20)[18],  # 95th percentile
            total_time_s=end_total - start_total
        )
        
        self.results.append(result)
        self.print_result(result)
        return result
    
    def print_result(self, result: BenchmarkResult):
        """Print benchmark result."""
        print(f"\nBenchmark: {result.name}")
        print(f"  Iterations: {result.iterations}")
        print(f"  Average: {result.avg_time_ms:.2f}ms")
        print(f"  Min: {result.min_time_ms:.2f}ms")
        print(f"  Max: {result.max_time_ms:.2f}ms")
        print(f"  P95: {result.p95_time_ms:.2f}ms")
        print(f"  Total: {result.total_time_s:.2f}s")
        print(f"  Rate: {result.iterations / result.total_time_s:.1f} ops/sec")
    
    def print_summary(self):
        """Print summary of all benchmarks."""
        print("\n" + "="*60)
        print("BENCHMARK SUMMARY")
        print("="*60)
        
        for result in self.results:
            print(f"{result.name:30} {result.avg_time_ms:8.2f}ms avg, {result.p95_time_ms:8.2f}ms p95")

# Usage example
async def main():
    from flashmm.ml.inference.inference_engine import InferenceEngine
    from flashmm.trading.quotes.quote_generator import QuoteGenerator
    
    benchmark = Benchmark()
    
    # Benchmark ML inference
    inference_engine = InferenceEngine()
    await inference_engine.load_model("./models/test_model.pt")
    
    async def ml_inference():
        features = [0.1] * 64
        return await inference_engine.predict(features)
    
    await benchmark.run_async("ML Inference", ml_inference, iterations=1000)
    
    # Benchmark quote generation
    quote_generator = QuoteGenerator()
    
    async def quote_generation():
        return await quote_generator.generate_quotes(
            symbol="SEI/USDC",
            market_data={"price": 0.042, "spread": 0.0005},
            prediction={"direction": "bullish", "confidence": 0.75}
        )
    
    await benchmark.run_async("Quote Generation", quote_generation, iterations=1000)
    
    benchmark.print_summary()

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Contributing Guidelines

### Getting Started

#### First-Time Contributors

1. **Fork the Repository**
   ```bash
   # Fork on GitHub, then clone your fork
   git clone https://github.com/YOUR_USERNAME/flashmm.git
   cd flashmm
   git remote add upstream https://github.com/flashmm/flashm
m.git
   ```

2. **Set Up Development Environment**
   ```bash
   # Run the setup script
   ./scripts/setup-dev-environment.sh
   
   # Or manually
   make dev-setup
   ```

3. **Find an Issue to Work On**
   - Look for issues labeled `good first issue` or `help wanted`
   - Check the project roadmap for upcoming features
   - Review the TODO comments in the codebase

4. **Create a Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

5. **Make Your Changes**
   - Follow the code standards outlined in this guide
   - Add tests for your changes
   - Update documentation as needed

6. **Submit a Pull Request**
   - Push your branch to your fork
   - Create a pull request with a clear description
   - Ensure all CI checks pass

#### Contribution Types

**Code Contributions:**
- Bug fixes
- New features
- Performance improvements
- Test coverage improvements
- Refactoring and code cleanup

**Documentation Contributions:**
- API documentation improvements
- Tutorial and guide creation
- Code example additions
- Translation work

**Community Contributions:**
- Issue triage and reproduction
- Community support in discussions
- Blog posts and tutorials
- Conference talks and presentations

### Code Review Process

#### As a Contributor

1. **Before Submitting:**
   - Run all tests locally (`make test`)
   - Check code formatting (`make format-check`)
   - Verify type checking (`make type-check`)
   - Run security scan (`make security-scan`)

2. **Pull Request Description:**
   - Clearly describe what your change does
   - Explain why the change is needed
   - Include any relevant issue numbers
   - Add screenshots for UI changes

3. **Responding to Reviews:**
   - Address all reviewer comments
   - Ask for clarification if feedback is unclear
   - Be open to suggestions and improvements
   - Update your branch with requested changes

#### As a Reviewer

1. **Review Criteria:**
   - Code correctness and functionality
   - Adherence to coding standards
   - Test coverage and quality
   - Performance implications
   - Security considerations
   - Documentation completeness

2. **Review Guidelines:**
   - Be constructive and respectful
   - Provide specific, actionable feedback
   - Explain the reasoning behind suggestions
   - Approve when all concerns are addressed

### Contributor License Agreement

By contributing to FlashMM, you agree that:

1. Your contributions will be licensed under the same MIT license
2. You have the right to submit the code you're contributing
3. Your contributions may be modified or redistributed
4. You understand the open-source nature of the project

---

## API Development

### FastAPI Application Structure

#### Application Factory Pattern

```python
# src/flashmm/api/app.py
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from contextlib import asynccontextmanager
import logging

from flashmm.api.routers import health, trading, ml, admin, websocket
from flashmm.api.middleware import LoggingMiddleware, MetricsMiddleware
from flashmm.api.dependencies import setup_dependencies
from flashmm.config.settings import settings

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    # Startup
    logger.info("Starting FlashMM API application")
    
    # Initialize dependencies
    await setup_dependencies(app)
    
    # Start background services
    await app.state.trading_engine.start()
    await app.state.monitoring_service.start()
    
    logger.info("FlashMM API application started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down FlashMM API application")
    
    # Stop background services
    await app.state.trading_engine.stop()
    await app.state.monitoring_service.stop()
    
    logger.info("FlashMM API application shut down complete")

def create_app() -> FastAPI:
    """Create FastAPI application instance."""
    
    app = FastAPI(
        title="FlashMM API",
        description="AI-Powered Market Making Agent for Sei Blockchain",
        version="1.2.0",
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
        lifespan=lifespan
    )
    
    # Add middleware
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.debug else ["https://dashboard.flashmm.com"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(LoggingMiddleware)
    app.add_middleware(MetricsMiddleware)
    
    # Include routers
    app.include_router(health.router, prefix="/health", tags=["health"])
    app.include_router(trading.router, prefix="/api/v1/trading", tags=["trading"])
    app.include_router(ml.router, prefix="/api/v1/ml", tags=["machine-learning"])
    app.include_router(admin.router, prefix="/api/v1/admin", tags=["admin"])
    app.include_router(websocket.router, prefix="/ws", tags=["websocket"])
    
    return app
```

#### Router Implementation

```python
# src/flashmm/api/routers/trading.py
from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Optional, Annotated
from datetime import datetime, timedelta

from flashmm.api.dependencies import get_trading_engine, get_current_user
from flashmm.api.models import (
    TradingStatusResponse,
    PositionResponse,
    OrderResponse,
    OrderRequest,
    TradingConfigRequest
)
from flashmm.trading.engine.market_making_engine import MarketMakingEngine
from flashmm.security.auth import User

router = APIRouter()

@router.get("/status", response_model=TradingStatusResponse)
async def get_trading_status(
    trading_engine: Annotated[MarketMakingEngine, Depends(get_trading_engine)]
) -> TradingStatusResponse:
    """Get current trading engine status and performance metrics."""
    try:
        status = await trading_engine.get_status()
        return TradingStatusResponse(**status)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get trading status: {str(e)}")

@router.get("/positions", response_model=List[PositionResponse])
async def get_positions(
    symbol: Optional[str] = Query(None, description="Filter by trading symbol"),
    trading_engine: Annotated[MarketMakingEngine, Depends(get_trading_engine)]
) -> List[PositionResponse]:
    """Get current trading positions."""
    try:
        positions = await trading_engine.get_positions(symbol=symbol)
        return [PositionResponse(**pos) for pos in positions]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get positions: {str(e)}")

@router.post("/orders", response_model=OrderResponse)
async def place_order(
    order_request: OrderRequest,
    trading_engine: Annotated[MarketMakingEngine, Depends(get_trading_engine)],
    current_user: Annotated[User, Depends(get_current_user)]
) -> OrderResponse:
    """Place a new trading order (admin only)."""
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    
    try:
        order_result = await trading_engine.place_order(order_request.to_order())
        return OrderResponse(**order_result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to place order: {str(e)}")

@router.post("/pause")
async def pause_trading(
    reason: str = "Manual pause",
    duration_seconds: Optional[int] = None,
    trading_engine: Annotated[MarketMakingEngine, Depends(get_trading_engine)],
    current_user: Annotated[User, Depends(get_current_user)]
):
    """Pause trading operations."""
    if not current_user.can_control_trading:
        raise HTTPException(status_code=403, detail="Trading control permission required")
    
    try:
        await trading_engine.pause(reason=reason, duration_seconds=duration_seconds)
        return {"status": "paused", "reason": reason, "paused_at": datetime.utcnow()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to pause trading: {str(e)}")

@router.post("/resume")
async def resume_trading(
    trading_engine: Annotated[MarketMakingEngine, Depends(get_trading_engine)],
    current_user: Annotated[User, Depends(get_current_user)]
):
    """Resume trading operations."""
    if not current_user.can_control_trading:
        raise HTTPException(status_code=403, detail="Trading control permission required")
    
    try:
        await trading_engine.resume()
        return {"status": "active", "resumed_at": datetime.utcnow()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to resume trading: {str(e)}")
```

#### WebSocket Implementation

```python
# src/flashmm/api/routers/websocket.py
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from typing import Set
import json
import asyncio
import logging

from flashmm.api.dependencies import get_websocket_manager
from flashmm.monitoring.streaming.websocket_manager import WebSocketManager

router = APIRouter()
logger = logging.getLogger(__name__)

@router.websocket("/live-data")
async def websocket_live_data(
    websocket: WebSocket,
    ws_manager: WebSocketManager = Depends(get_websocket_manager)
):
    """WebSocket endpoint for real-time trading data."""
    await websocket.accept()
    client_id = await ws_manager.connect(websocket)
    
    try:
        while True:
            # Receive subscription messages from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("action") == "subscribe":
                channels = message.get("channels", [])
                await ws_manager.subscribe_client(client_id, channels)
                await websocket.send_text(json.dumps({
                    "type": "subscription_confirmed",
                    "channels": channels
                }))
            
            elif message.get("action") == "unsubscribe":
                channels = message.get("channels", [])
                await ws_manager.unsubscribe_client(client_id, channels)
                await websocket.send_text(json.dumps({
                    "type": "unsubscription_confirmed",
                    "channels": channels
                }))
                
    except WebSocketDisconnect:
        logger.info(f"Client {client_id} disconnected")
    except Exception as e:
        logger.error(f"WebSocket error for client {client_id}: {e}")
    finally:
        await ws_manager.disconnect(client_id)
```

### API Models and Validation

#### Pydantic Models

```python
# src/flashmm/api/models.py
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from decimal import Decimal
from datetime import datetime
from enum import Enum

class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"

class OrderStatus(str, Enum):
    PENDING = "pending"
    ACTIVE = "active"
    FILLED = "filled"
    CANCELLED = "cancelled"

class TradingStatusResponse(BaseModel):
    """Trading engine status response."""
    status: str = Field(..., description="Current trading status")
    trading_enabled: bool = Field(..., description="Whether trading is enabled")
    start_time: datetime = Field(..., description="Trading engine start time")
    markets: List[Dict[str, Any]] = Field(..., description="Active trading markets")
    totals: Dict[str, Any] = Field(..., description="Aggregate performance metrics")
    risk_metrics: Dict[str, Any] = Field(..., description="Current risk metrics")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "active",
                "trading_enabled": True,
                "start_time": "2024-01-20T00:00:00Z",
                "markets": [
                    {
                        "symbol": "SEI/USDC",
                        "status": "trading",
                        "spread_improvement_percent": 34.4,
                        "daily_pnl_usdc": 25.30
                    }
                ],
                "totals": {
                    "total_pnl_usdc": 255.23,
                    "daily_pnl_usdc": 43.55
                },
                "risk_metrics": {
                    "risk_level": "normal",
                    "position_utilization_percent": 15.2
                }
            }
        }

class PositionResponse(BaseModel):
    """Position information response."""
    symbol: str = Field(..., description="Trading pair symbol")
    base_balance: Decimal = Field(..., description="Base asset balance")
    quote_balance: Decimal = Field(..., description="Quote asset balance")
    position_usdc: Decimal = Field(..., description="Position value in USDC")
    position_percent: float = Field(..., description="Position as percentage of limit")
    unrealized_pnl_usdc: Decimal = Field(..., description="Unrealized P&L in USDC")
    last_updated: datetime = Field(..., description="Last update timestamp")

class OrderRequest(BaseModel):
    """Order placement request."""
    symbol: str = Field(..., description="Trading pair symbol")
    side: OrderSide = Field(..., description="Order side (buy/sell)")
    price: Decimal = Field(..., gt=0, description="Order price")
    size: Decimal = Field(..., gt=0, description="Order size")
    
    @validator('symbol')
    def validate_symbol(cls, v):
        # Add symbol validation logic
        valid_symbols = ["SEI/USDC", "ETH/USDC"]  # Could be loaded from config
        if v not in valid_symbols:
            raise ValueError(f"Unsupported symbol: {v}")
        return v
    
    def to_order(self):
        """Convert to internal Order object."""
        from flashmm.trading.models import Order
        return Order(
            symbol=self.symbol,
            side=self.side.value,
            price=self.price,
            size=self.size
        )

class OrderResponse(BaseModel):
    """Order placement response."""
    order_id: str = Field(..., description="Unique order identifier")
    symbol: str = Field(..., description="Trading pair symbol")
    side: str = Field(..., description="Order side")
    price: Decimal = Field(..., description="Order price")
    size: Decimal = Field(..., description="Order size")
    status: OrderStatus = Field(..., description="Current order status")
    created_at: datetime = Field(..., description="Order creation timestamp")
```

---

## ML Model Development

### Model Training Pipeline

#### Data Preparation

```python
# src/flashmm/ml/training/data_preparation.py
import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
from datetime import datetime, timedelta
import asyncio

from flashmm.data.storage.influxdb_client import InfluxDBClient
from flashmm.ml.features.feature_extractor import FeatureExtractor

class TrainingDataPipeline:
    """Pipeline for preparing training data for ML models."""
    
    def __init__(self, influxdb_client: InfluxDBClient):
        self.influxdb_client = influxdb_client
        self.feature_extractor = FeatureExtractor()
    
    async def prepare_training_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        prediction_horizon_ms: int = 200
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data for model training."""
        
        # Fetch historical market data
        market_data = await self._fetch_market_data(symbol, start_date, end_date)
        
        # Extract features
        features = await self._extract_features(market_data)
        
        # Create labels (future price movements)
        labels = await self._create_labels(market_data, prediction_horizon_ms)
        
        # Clean and align data
        X, y = self._align_features_and_labels(features, labels)
        
        return X, y
    
    async def _fetch_market_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Fetch historical market data from InfluxDB."""
        query = f'''
        SELECT 
            time,
            price,
            volume,
            bid,
            ask,
            spread_bps
        FROM market_data 
        WHERE symbol = '{symbol}' 
        AND time >= '{start_date.isoformat()}'
        AND time <= '{end_date.isoformat()}'
        ORDER BY time
        '''
        
        result = await self.influxdb_client.query(query)
        return pd.DataFrame(result)
    
    async def _extract_features(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Extract features from market data."""
        features_list = []
        
        for i in range(len(market_data)):
            if i < 100:  # Need enough history for feature calculation
                continue
                
            window_data = market_data.iloc[i-100:i]
            features = await self.feature_extractor.extract_features(
                prices=window_data['price'].values,
                volumes=window_data['volume'].values,
                bids=window_data['bid'].values,
                asks=window_data['ask'].values
            )
            
            features_list.append({
                'timestamp': market_data.iloc[i]['time'],
                **features
            })
        
        return pd.DataFrame(features_list)
    
    async def _create_labels(
        self,
        market_data: pd.DataFrame,
        prediction_horizon_ms: int
    ) -> pd.DataFrame:
        """Create labels for supervised learning."""
        labels = []
        
        for i in range(len(market_data) - 1):
            current_price = market_data.iloc[i]['price']
            
            # Find price after prediction horizon
            current_time = pd.to_datetime(market_data.iloc[i]['time'])
            target_time = current_time + pd.Timedelta(milliseconds=prediction_horizon_ms)
            
            # Find closest data point to target time
            future_idx = i + 1
            for j in range(i + 1, len(market_data)):
                data_time = pd.to_datetime(market_data.iloc[j]['time'])
                if data_time >= target_time:
                    future_idx = j
                    break
            
            if future_idx < len(market_data):
                future_price = market_data.iloc[future_idx]['price']
                price_change_bps = (future_price - current_price) / current_price * 10000
                
                # Create classification labels
                if price_change_bps > 2:
                    direction = 2  # bullish
                elif price_change_bps < -2:
                    direction = 0  # bearish
                else:
                    direction = 1  # neutral
                
                labels.append({
                    'timestamp': market_data.iloc[i]['time'],
                    'direction': direction,
                    'price_change_bps': price_change_bps
                })
        
        return pd.DataFrame(labels)
    
    def _align_features_and_labels(
        self,
        features: pd.DataFrame,
        labels: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Align features and labels by timestamp."""
        # Merge on timestamp
        merged = pd.merge(features, labels, on='timestamp', how='inner')
        
        # Extract feature columns (exclude timestamp and labels)
        feature_cols = [col for col in merged.columns 
                       if col not in ['timestamp', 'direction', 'price_change_bps']]
        
        X = merged[feature_cols].values
        y = merged['direction'].values
        
        # Remove any rows with NaN values
        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X = X[mask]
        y = y[mask]
        
        return X, y
```

#### Model Training

```python
# src/flashmm/ml/training/trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from typing import Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)

class MarketPredictionModel(nn.Module):
    """Neural network model for market prediction."""
    
    def __init__(self, input_size: int, hidden_size: int = 128, num_classes: int = 3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, num_classes),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        return self.network(x)

class ModelTrainer:
    """Trainer for market prediction models."""
    
    def __init__(self, device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler = StandardScaler()
        logger.info(f"Using device: {self.device}")
    
    def train_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_path: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Train a market prediction model."""
        
        # Prepare data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train_scaled).to(self.device)
        y_train_tensor = torch.LongTensor(y_train).to(self.device)
        X_test_tensor = torch.FloatTensor(X_test_scaled).to(self.device)
        y_test_tensor = torch.LongTensor(y_test).to(self.device)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        
        # Initialize model
        model = MarketPredictionModel(
            input_size=X.shape[1],
            hidden_size=kwargs.get('hidden_size', 128),
            num_classes=len(np.unique(y))
        ).to(self.device)
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=kwargs.get('learning_rate', 0.001))
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
        
        # Training loop
        num_epochs = kwargs.get('num_epochs', 100)
        best_accuracy = 0.0
        
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            # Validation
            model.eval()
            with torch.no_grad():
                test_outputs = model(X_test_tensor)
                test_predictions = torch.argmax(test_outputs, dim=1)
                test_accuracy = accuracy_score(y_test, test_predictions.cpu().numpy())
            
            scheduler.step(total_loss)
            
            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                # Save best model
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'scaler': self.scaler,
                    'input_size': X.shape[1],
                    'accuracy': best_accuracy
                }, model_path)
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{num_epochs}, "
                          f"Loss: {total_loss/len(train_loader):.4f}, "
                          f"Test Accuracy: {test_accuracy:.4f}")
        
        # Final evaluation
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_tensor)
            test_predictions = torch.argmax(test_outputs, dim=1)
            final_accuracy = accuracy_score(y_test, test_predictions.cpu().numpy())
            
            # Generate classification report
            report = classification_report(
                y_test, 
                test_predictions.cpu().numpy(),
                target_names=['bearish', 'neutral', 'bullish'],
                output_dict=True
            )
        
        logger.info(f"Training completed. Final accuracy: {final_accuracy:.4f}")
        
        return {
            'final_accuracy': final_accuracy,
            'best_accuracy': best_accuracy,
            'classification_report': report,
            'model_path': model_path
        }
```

### Model Deployment

#### TorchScript Export

```python
# src/flashmm/ml/deployment/model_exporter.py
import torch
import joblib
from pathlib import Path
from typing import Dict, Any
import logging

from flashmm.ml.training.trainer import MarketPredictionModel

logger = logging.getLogger(__name__)

class ModelExporter:
    """Export trained models for production deployment."""
    
    @staticmethod
    def export_to_torchscript(
        model_path: str,
        output_path: str,
        example_input_size: int = 64
    ) -> Dict[str, Any]:
        """Export PyTorch model to TorchScript for production inference."""
        
        # Load trained model
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Recreate model
        model = MarketPredictionModel(
            input_size=checkpoint['input_size'],
            hidden_size=128,  # Should match training config
            num_classes=3
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Create example input for tracing
        example_input = torch.randn(1, checkpoint['input_size'])
        
        # Trace the model
        try:
            traced_model = torch.jit.trace(model, example_input)
            
            # Save TorchScript model
            traced_model.save(output_path)
            
            # Save scaler separately
            scaler_path = str(Path(output_path).with_suffix('.scaler.joblib'))
            joblib.dump(checkpoint['scaler'], scaler_path)
            
            # Test the traced model
            with torch.no_grad():
                original_output = model(example_input)
                traced_output = traced_model(example_input)
                max_diff = torch.max(torch.abs(original_output - traced_output))
            
            logger.info(f"Model exported to TorchScript: {output_path}")
            logger.info(f"Max output difference: {max_diff.item():.8f}")
            
            # Get model size
            model_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
            
            return {
                'torchscript_path': output_path,
                'scaler_path': scaler_path,
                'input_size': checkpoint['input_size'],
                'model_size_mb': model_size_mb,
                'accuracy': checkpoint['accuracy'],
                'max_output_diff': max_diff.item()
            }
            
        except Exception as e:
            logger.error(f"Failed to export model to TorchScript: {e}")
            raise
```

---

## Extension Points

### Plugin Architecture

#### Plugin Interface

```python
# src/flashmm/plugins/base.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class PluginInfo:
    name: str
    version: str
    description: str
    author: str
    dependencies: list

class FlashMMPlugin(ABC):
    """Base class for FlashMM plugins."""
    
    @property
    @abstractmethod
    def info(self) -> PluginInfo:
        """Plugin information."""
        pass
    
    @abstractmethod
    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the plugin with configuration."""
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """Cleanup plugin resources."""
        pass

class TradingStrategyPlugin(FlashMMPlugin):
    """Base class for trading strategy plugins."""
    
    @abstractmethod
    async def generate_quotes(
        self,
        symbol: str,
        market_data: Dict[str, Any],
        prediction: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Generate trading quotes based on market data and predictions."""
        pass

class MLModelPlugin(FlashMMPlugin):
    """Base class for ML model plugins."""
    
    @abstractmethod
    async def predict(
        self,
        features: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate predictions from input features."""
        pass
    
    @abstractmethod
    def get_feature_requirements(self) -> List[str]:
        """Get list of required features for prediction."""
        pass
```

#### Plugin Manager

```python
# src/flashmm/plugins/manager.py
import importlib
import pkgutil
from typing import Dict, List, Type, Any
from pathlib import Path
import logging

from flashmm.plugins.base import FlashMMPlugin, PluginInfo

logger = logging.getLogger(__name__)

class PluginManager:
    """Manage FlashMM plugins."""
    
def __init__(self):
        self.plugins: Dict[str, FlashMMPlugin] = {}
        self.plugin_configs: Dict[str, Dict[str, Any]] = {}
    
    def discover_plugins(self, plugin_directory: str = "plugins") -> List[PluginInfo]:
        """Discover available plugins in the plugin directory."""
        discovered = []
        plugin_path = Path(plugin_directory)
        
        if not plugin_path.exists():
            logger.warning(f"Plugin directory {plugin_directory} does not exist")
            return discovered
        
        for plugin_file in plugin_path.glob("*.py"):
            if plugin_file.name.startswith("_"):
                continue
                
            try:
                # Import plugin module
                spec = importlib.util.spec_from_file_location(
                    plugin_file.stem, plugin_file
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # Find plugin classes
                for name in dir(module):
                    obj = getattr(module, name)
                    if (isinstance(obj, type) and 
                        issubclass(obj, FlashMMPlugin) and 
                        obj != FlashMMPlugin):
                        
                        plugin_instance = obj()
                        discovered.append(plugin_instance.info)
                        logger.info(f"Discovered plugin: {plugin_instance.info.name}")
                        
            except Exception as e:
                logger.error(f"Failed to load plugin {plugin_file}: {e}")
        
        return discovered
    
    async def load_plugin(
        self, 
        plugin_name: str, 
        config: Dict[str, Any] = None
    ) -> None:
        """Load and initialize a plugin."""
        if plugin_name in self.plugins:
            logger.warning(f"Plugin {plugin_name} already loaded")
            return
        
        try:
            # Import and instantiate plugin
            plugin_instance = self._import_plugin(plugin_name)
            
            # Initialize with config
            await plugin_instance.initialize(config or {})
            
            # Store plugin
            self.plugins[plugin_name] = plugin_instance
            self.plugin_configs[plugin_name] = config or {}
            
            logger.info(f"Plugin {plugin_name} loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load plugin {plugin_name}: {e}")
            raise
    
    async def unload_plugin(self, plugin_name: str) -> None:
        """Unload a plugin."""
        if plugin_name not in self.plugins:
            logger.warning(f"Plugin {plugin_name} not loaded")
            return
        
        try:
            plugin = self.plugins[plugin_name]
            await plugin.cleanup()
            
            del self.plugins[plugin_name]
            del self.plugin_configs[plugin_name]
            
            logger.info(f"Plugin {plugin_name} unloaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to unload plugin {plugin_name}: {e}")
            raise
    
    def get_plugin(self, plugin_name: str) -> Optional[FlashMMPlugin]:
        """Get a loaded plugin instance."""
        return self.plugins.get(plugin_name)
    
    def list_loaded_plugins(self) -> List[str]:
        """List all loaded plugin names."""
        return list(self.plugins.keys())
```

#### Example Custom Strategy Plugin

```python
# plugins/custom_strategy.py
from typing import Dict, List, Any, Optional
import asyncio
from decimal import Decimal

from flashmm.plugins.base import TradingStrategyPlugin, PluginInfo

class CustomMomentumStrategy(TradingStrategyPlugin):
    """Custom momentum-based trading strategy plugin."""
    
    @property
    def info(self) -> PluginInfo:
        return PluginInfo(
            name="custom_momentum_strategy",
            version="1.0.0",
            description="Momentum-based market making strategy",
            author="FlashMM Community",
            dependencies=["numpy", "pandas"]
        )
    
    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the strategy plugin."""
        self.momentum_window = config.get("momentum_window", 10)
        self.spread_multiplier = config.get("spread_multiplier", 1.2)
        self.max_position_pct = config.get("max_position_pct", 0.8)
        
        # Initialize any strategy-specific state
        self.price_history = []
        
    async def cleanup(self) -> None:
        """Cleanup strategy resources."""
        self.price_history.clear()
    
    async def generate_quotes(
        self,
        symbol: str,
        market_data: Dict[str, Any],
        prediction: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Generate quotes using momentum strategy."""
        
        current_price = Decimal(str(market_data["price"]))
        base_spread = Decimal(str(market_data.get("spread", 0.001)))
        
        # Update price history
        self.price_history.append(float(current_price))
        if len(self.price_history) > self.momentum_window:
            self.price_history.pop(0)
        
        # Calculate momentum
        momentum = self._calculate_momentum()
        
        # Adjust spread based on momentum
        if momentum > 0.01:  # Strong upward momentum
            bid_spread = base_spread * Decimal("0.8")  # Tighter bid
            ask_spread = base_spread * Decimal("1.5")  # Wider ask
        elif momentum < -0.01:  # Strong downward momentum
            bid_spread = base_spread * Decimal("1.5")  # Wider bid
            ask_spread = base_spread * Decimal("0.8")  # Tighter ask
        else:  # Neutral momentum
            bid_spread = ask_spread = base_spread
        
        # Generate quotes
        quotes = [
            {
                "side": "buy",
                "price": current_price - bid_spread,
                "size": Decimal("100"),  # Could be dynamic
                "spread_bps": float(bid_spread / current_price * 10000)
            },
            {
                "side": "sell", 
                "price": current_price + ask_spread,
                "size": Decimal("100"),
                "spread_bps": float(ask_spread / current_price * 10000)
            }
        ]
        
        return quotes
    
    def _calculate_momentum(self) -> float:
        """Calculate price momentum from history."""
        if len(self.price_history) < 2:
            return 0.0
        
        # Simple momentum calculation
        recent_price = self.price_history[-1]
        older_price = self.price_history[0]
        
        return (recent_price - older_price) / older_price
```

---

## CI/CD Integration

### GitHub Actions Workflow

```yaml
# .github/workflows/ci.yml
name: CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]
  release:
    types: [ published ]

env:
  PYTHON_VERSION: "3.11"
  POETRY_VERSION: "1.4.2"

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.11", "3.12"]
    
    services:
      redis:
        image: redis:7-alpine
        ports:
          - 6379:6379
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
      
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: test
          POSTGRES_USER: test
          POSTGRES_DB: flashmm_test
        ports:
          - 5432:5432
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install Poetry
      uses: snok/install-poetry@v1
      with:
        version: ${{ env.POETRY_VERSION }}
        virtualenvs-create: true
        virtualenvs-in-project: true
    
    - name: Load cached venv
      id: cached-poetry-dependencies
      uses: actions/cache@v3
      with:
        path: .venv
        key: venv-${{ runner.os }}-${{ matrix.python-version }}-${{ hashFiles('**/poetry.lock') }}
    
    - name: Install dependencies
      if: steps.cached-poetry-dependencies.outputs.cache-hit != 'true'
      run: poetry install --with dev,test
    
    - name: Code formatting check
      run: |
        poetry run black --check src/ tests/
        poetry run isort --check-only src/ tests/
    
    - name: Linting
      run: |
        poetry run ruff check src/ tests/
        poetry run mypy src/
    
    - name: Security scan
      run: |
        poetry run bandit -r src/
        poetry run safety check
    
    - name: Run tests
      env:
        DATABASE_URL: postgresql://test:test@localhost:5432/flashmm_test
        REDIS_URL: redis://localhost:6379/1
        TESTING: true
      run: |
        poetry run pytest tests/ -v --cov=src/flashmm --cov-report=xml --cov-report=html
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: unittests
        name: codecov-umbrella
    
    - name: Performance tests
      env:
        DATABASE_URL: postgresql://test:test@localhost:5432/flashmm_test
        REDIS_URL: redis://localhost:6379/1
      run: |
        poetry run pytest tests/performance/ -v --benchmark-only

  security:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Run Trivy vulnerability scanner
      uses: aquasecurity/trivy-action@master
      with:
        scan-type: 'fs'
        scan-ref: '.'
        format: 'sarif'
        output: 'trivy-results.sarif'
    
    - name: Upload Trivy scan results to GitHub Security tab
      uses: github/codeql-action/upload-sarif@v2
      with:
        sarif_file: 'trivy-results.sarif'

  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v2
    
    - name: Login to Docker Hub
      if: github.event_name != 'pull_request'
      uses: docker/login-action@v2
      with:
        username: ${{ secrets.DOCKER_USERNAME }}
        password: ${{ secrets.DOCKER_PASSWORD }}
    
    - name: Extract metadata
      id: meta
      uses: docker/metadata-action@v4
      with:
        images: flashmm/flashmm
        tags: |
          type=ref,event=branch
          type=ref,event=pr
          type=semver,pattern={{version}}
          type=semver,pattern={{major}}.{{minor}}
          type=raw,value=latest,enable={{is_default_branch}}
    
    - name: Build and push Docker image
      uses: docker/build-push-action@v4
      with:
        context: .
        file: ./Dockerfile.production
        push: ${{ github.event_name != 'pull_request' }}
        tags: ${{ steps.meta.outputs.tags }}
        labels: ${{ steps.meta.outputs.labels }}
        cache-from: type=gha
        cache-to: type=gha,mode=max

  deploy-staging:
    needs: [test, build]
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/develop'
    environment: staging
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Configure AWS credentials
      uses: aws-actions/configure-aws-credentials@v2
      with:
        aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
        aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        aws-region: us-east-1
    
    - name: Update kubeconfig
      run: |
        aws eks update-kubeconfig --region us-east-1 --name flashmm-staging
    
    - name: Deploy to staging
      run: |
        helm upgrade --install flashmm ./helm/flashmm/ \
          -f environments/staging/values.yaml \
          --set image.tag=${GITHUB_SHA} \
          --namespace flashmm \
          --wait \
          --timeout 10m
    
    - name: Run integration tests
      run: |
        ./tests/integration/run_staging_tests.sh

  deploy-production:
    needs: [test, build]
    runs-on: ubuntu-latest
    if: github.event_name == 'release'
    environment: production
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Configure AWS credentials
      uses: aws-actions/configure-aws-credentials@v2
      with:
        aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
        aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        aws-region: us-east-1
    
    - name: Update kubeconfig
      run: |
        aws eks update-kubeconfig --region us-east-1 --name flashmm-production
    
    - name: Deploy to production
      run: |
        helm upgrade --install flashmm ./helm/flashmm/ \
          -f environments/production/values.yaml \
          --set image.tag=${GITHUB_REF#refs/tags/} \
          --namespace flashmm \
          --wait \
          --timeout 10m \
          --atomic
    
    - name: Run smoke tests
      run: |
        ./tests/integration/run_production_smoke_tests.sh
    
    - name: Notify Slack
      uses: 8398a7/action-slack@v3
      with:
        status: ${{ job.status }}
        channel: '#deployments'
        webhook_url: ${{ secrets.SLACK_WEBHOOK }}
      if: always()
```

### Pre-commit Hooks

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
      - id: check-merge-conflict
      - id: debug-statements
  
  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black
        language_version: python3
        args: [--line-length=88]
  
  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
        args: [--profile=black]
  
  - repo: https://github.com/charliermarsh/ruff-pre-commit
    rev: v0.0.270
    hooks:
      - id: ruff
        args: [--fix, --exit-non-zero-on-fix]
  
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.3.0
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
        exclude: ^tests/
  
  - repo: https://github.com/PyCQA/bandit
    rev: 1.7.5
    hooks:
      - id: bandit
        args: [-r, src/]
        exclude: tests/
  
  - repo: https://github.com/hadolint/hadolint
    rev: v2.12.0
    hooks:
      - id: hadolint-docker
        args: [--ignore, DL3008, --ignore, DL3009]
  
  - repo: local
    hooks:
      - id: pytest-check
        name: pytest-check
        entry: poetry run pytest tests/unit/ -x
        language: system
        pass_filenames: false
        always_run: true
```

---

## Performance Optimization

### Profiling and Optimization

#### AsyncIO Performance

```python
# src/flashmm/utils/async_utils.py
import asyncio
import time
import functools
from typing import Callable, Any, TypeVar, Optional
from contextlib import asynccontextmanager
import uvloop

T = TypeVar('T')

def setup_asyncio_optimization():
    """Setup asyncio for optimal performance."""
    # Use uvloop for better performance on Linux/macOS
    if hasattr(uvloop, 'install'):
        uvloop.install()
    
    # Configure asyncio settings
    loop = asyncio.get_event_loop()
    loop.set_debug(False)
    
    # Optimize task scheduling
    if hasattr(loop, 'set_task_factory'):
        loop.set_task_factory(asyncio.create_task)

class AsyncLimiter:
    """Rate limiter for async operations."""
    
    def __init__(self, max_rate: float, time_window: float = 1.0):
        self.max_rate = max_rate
        self.time_window = time_window
        self.tokens = max_rate
        self.last_update = time.monotonic()
        self._lock = asyncio.Lock()
    
    async def acquire(self, tokens: int = 1) -> None:
        """Acquire tokens from the rate limiter."""
        async with self._lock:
            now = time.monotonic()
            time_passed = now - self.last_update
            
            # Refill tokens
            self.tokens = min(
                self.max_rate,
                self.tokens + time_passed * (self.max_rate / self.time_window)
            )
            self.last_update = now
            
            # Wait if not enough tokens
            if self.tokens < tokens:
                wait_time = (tokens - self.tokens) / (self.max_rate / self.time_window)
                await asyncio.sleep(wait_time)
                self.tokens = 0
            else:
                self.tokens -= tokens

def async_lru_cache(maxsize: int = 128, ttl: Optional[float] = None):
    """LRU cache decorator for async functions with optional TTL."""
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        cache = {}
        cache_info = {'hits': 0, 'misses': 0}
        
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Create cache key
            key = (args, tuple(sorted(kwargs.items())))
            
            # Check cache
            if key in cache:
                result, timestamp = cache[key]
                if ttl is None or time.monotonic() - timestamp < ttl:
                    cache_info['hits'] += 1
                    return result
                else:
                    del cache[key]
            
            # Cache miss - execute function
            cache_info['misses'] += 1
            result = await func(*args, **kwargs)
            
            # Store in cache
            if len(cache) >= maxsize:
                # Remove oldest entry
                oldest_key = next(iter(cache))
                del cache[oldest_key]
            
            cache[key] = (result, time.monotonic())
            return result
        
        wrapper.cache_info = lambda: cache_info.copy()
        wrapper.cache_clear = cache.clear
        return wrapper
    
    return decorator

@asynccontextmanager
async def async_timer():
    """Context manager for timing async operations."""
    start_time = time.perf_counter()
    try:
        yield
    finally:
        end_time = time.perf_counter()
        duration = end_time - start_time
        print(f"Operation took {duration:.4f} seconds")
```

#### Memory Optimization

```python
# src/flashmm/utils/memory_utils.py
import gc
import sys
from typing import Any, Dict
import psutil
import asyncio
from dataclasses import dataclass

@dataclass
class MemoryStats:
    total_mb: float
    available_mb: float
    used_mb: float
    percent: float
    python_mb: float

class MemoryMonitor:
    """Monitor and optimize memory usage."""
    
    def __init__(self, gc_threshold: float = 80.0):
        self.gc_threshold = gc_threshold
        self._monitoring = False
    
    def get_memory_stats(self) -> MemoryStats:
        """Get current memory statistics."""
        process = psutil.Process()
        memory_info = process.memory_info()
        system_memory = psutil.virtual_memory()
        
        return MemoryStats(
            total_mb=system_memory.total / 1024 / 1024,
            available_mb=system_memory.available / 1024 / 1024,
            used_mb=system_memory.used / 1024 / 1024,
            percent=system_memory.percent,
            python_mb=memory_info.rss / 1024 / 1024
        )
    
    def optimize_gc(self):
        """Optimize garbage collection settings."""
        # Adjust GC thresholds for better performance
        gc.set_threshold(700, 10, 10)
        
        # Force garbage collection
        gc.collect()
    
    async def start_monitoring(self, interval: float = 60.0):
        """Start memory monitoring in background."""
        self._monitoring = True
        
        while self._monitoring:
            stats = self.get_memory_stats()
            
            if stats.percent > self.gc_threshold:
                print(f"High memory usage detected: {stats.percent:.1f}%")
                self.optimize_gc()
            
            await asyncio.sleep(interval)
    
    def stop_monitoring(self):
        """Stop memory monitoring."""
        self._monitoring = False

# Global memory monitor instance
memory_monitor = MemoryMonitor()
```

#### Database Connection Optimization

```python
# src/flashmm/data/storage/optimized_client.py
import asyncio
import asyncpg
from typing import Dict, Any, List, Optional
import logging
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

class OptimizedDatabaseClient:
    """Optimized database client with connection pooling."""
    
    def __init__(
        self,
        database_url: str,
        min_connections: int = 5,
        max_connections: int = 20,
        command_timeout: float = 60.0
    ):
        self.database_url = database_url
        self.min_connections = min_connections
        self.max_connections = max_connections
        self.command_timeout = command_timeout
        self._pool: Optional[asyncpg.Pool] = None
    
    async def initialize(self):
        """Initialize database connection pool."""
        self._pool = await asyncpg.create_pool(
            self.database_url,
            min_size=self.min_connections,
            max_size=self.max_connections,
            command_timeout=self.command_timeout,
            server_settings={
                'jit': 'off',  # Disable JIT for consistent performance
                'application_name': 'flashmm'
            }
        )
        logger.info("Database connection pool initialized")
    
    async def close(self):
        """Close database connection pool."""
        if self._pool:
            await self._pool.close()
            logger.info("Database connection pool closed")
    
    @asynccontextmanager
    async def get_connection(self):
        """Get database connection from pool."""
        if not self._pool:
            raise RuntimeError("Database pool not initialized")
        
        async with self._pool.acquire() as connection:
            yield connection
    
    async def execute_query(
        self,
        query: str,
        *args,
        fetch: str = "all"
    ) -> Any:
        """Execute query with connection pooling."""
        async with self.get_connection() as conn:
            if fetch == "all":
                return await conn.fetch(query, *args)
            elif fetch == "one":
                return await conn.fetchrow(query, *args)
            elif fetch == "none":
                return await conn.execute(query, *args)
            else:
                raise ValueError(f"Invalid fetch type: {fetch}")
    
    async def execute_batch(
        self,
        query: str,
        args_list: List[tuple]
    ) -> None:
        """Execute batch query for better performance."""
        async with self.get_connection() as conn:
            await conn.executemany(query, args_list)
    
    async def copy_records_to_table(
        self,
        table_name: str,
        records: List[Dict[str, Any]],
        columns: List[str]
    ) -> None:
        """Use COPY for bulk inserts - much faster than individual INSERTs."""
        if not records:
            return
        
        async with self.get_connection() as conn:
            # Prepare data for COPY
            data = []
            for record in records:
                row = [record.get(col) for col in columns]
                data.append(row)
            
            # Use COPY for bulk insert
            await conn.copy_records_to_table(
                table_name,
                records=data,
                columns=columns
            )
```

---

## Conclusion

This comprehensive developer guide provides everything needed to contribute to and extend FlashMM:

### Key Takeaways

1. **Development Environment**: Automated setup with Poetry, Docker, and pre-commit hooks
2. **Code Standards**: Strict typing, comprehensive testing, and professional documentation
3. **Architecture**: Modular design with clear separation of concerns and dependency injection
4. **Testing**: Comprehensive test suite with unit, integration, and performance tests
5. **Debugging**: Advanced profiling tools and structured logging
6. **Contributing**: Clear workflow for community contributions
7. **API Development**: FastAPI best practices with proper error handling
8. **ML Integration**: Complete ML pipeline from training to deployment
9. **Plugin System**: Extensible architecture for custom strategies and models
10. **CI/CD**: Automated testing, building, and deployment pipeline
11. **Performance**: Optimization techniques for high-frequency trading requirements

### Development Resources

- **📚 Complete Documentation**: [Full documentation suite](../README.md)
- **🏗️ Architecture Guide**: [ARCHITECTURE.md](ARCHITECTURE.md)
- **⚙️ Configuration**: [CONFIGURATION.md](CONFIGURATION.md)
- **👤 User Guide**: [USER_GUIDE.md](USER_GUIDE.md)
- **🚀 Deployment**: [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)
- **🔧 Operations**: [OPERATIONS.md](OPERATIONS.md)
- **🌐 API Reference**: [API.md](API.md)

### Getting Started

1. **Set up your environment** using the automated scripts
2. **Read the architecture documentation** to understand the system design
3. **Run the test suite** to ensure everything works
4. **Pick an issue** from the GitHub repository to work on
5. **Follow the contribution guidelines** when submitting changes

### Community

- **💬 Discussions**: Join GitHub Discussions for questions and ideas
- **🐛 Issues**: Report bugs and request features via GitHub Issues
- **📖 Documentation**: Help improve documentation and examples
- **🔧 Code**: Contribute bug fixes, features, and optimizations

FlashMM is designed to be hackathon-ready while maintaining production quality. The architecture supports rapid development while ensuring the system can handle real-world trading requirements.

Happy coding! 🚀
# FlashMM Development & Deployment Workflows

## Overview
FlashMM follows a streamlined development and deployment workflow optimized for hackathon speed while maintaining production-quality practices. The workflow supports local development, automated testing, and single-VPS deployment.

## Development Workflow

### Local Development Setup

#### Prerequisites
```bash
# System requirements
- Python 3.11+
- Docker & Docker Compose
- Git
- Make (optional but recommended)

# Development tools
pip install poetry  # Dependency management
pip install pre-commit  # Git hooks
```

#### Initial Setup
```bash
# Clone repository
git clone https://github.com/flashmm/flashmm.git
cd flashmm

# Install dependencies
poetry install --with dev,test

# Setup pre-commit hooks
pre-commit install

# Copy environment template
cp .env.example .env
# Edit .env with your configuration

# Start development services
docker-compose -f docker-compose.dev.yml up -d redis influxdb

# Run application
poetry run python -m flashmm.main
```

#### Development Environment Configuration
**`.env.development`**:
```bash
ENVIRONMENT=development
FLASHMM_DEBUG=true
FLASHMM_LOG_LEVEL=DEBUG

# Use local services
REDIS_URL=redis://localhost:6379/0
INFLUXDB_URL=http://localhost:8086

# Mock external APIs for development
SEI_RPC_URL=http://localhost:8545  # Local test node
CAMBRIAN_API_KEY=dev_key_12345
CAMBRIAN_SECRET_KEY=dev_secret_67890

# Disable external integrations in dev
TWITTER_ENABLED=false
GRAFANA_ENABLED=false
```

### Development Tools Integration

#### Pre-commit Configuration
**`.pre-commit-config.yaml`**:
```yaml
repos:
  - repo: https://github.com/charliermarsh/ruff-pre-commit
    rev: v0.0.280
    hooks:
      - id: ruff
        args: [--fix, --exit-non-zero-on-fix]
  
  - repo: https://github.com/psf/black
    rev: 23.7.0
    hooks:
      - id: black
        language_version: python3.11
  
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.5.1
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
  
  - repo: https://github.com/PyCQA/bandit
    rev: 1.7.5
    hooks:
      - id: bandit
        args: [-c, pyproject.toml]
  
  - repo: https://github.com/python-poetry/poetry
    rev: 1.5.1
    hooks:
      - id: poetry-check
```

#### Code Quality Configuration
**`pyproject.toml`** (relevant sections):
```toml
[tool.ruff]
target-version = "py311"
line-length = 100
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

[tool.black]
line-length = 100
target-version = ['py311']

[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true

[tool.bandit]
exclude_dirs = ["tests"]
skips = ["B101", "B601"]
```

### Testing Strategy

#### Test Structure
```
tests/
├── unit/                    # Fast, isolated tests
│   ├── test_data_ingestion/
│   ├── test_ml_models/
│   ├── test_trading_engine/
│   └── test_api/
├── integration/             # Component interaction tests
│   ├── test_data_pipeline.py
│   ├── test_trading_flow.py
│   └── test_external_apis.py
└── end_to_end/             # Full system tests
    └── test_full_system.py
```

#### Testing Configuration
**`pytest.ini`**:
```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    --strict-markers
    --disable-warnings
    --cov=src/flashmm
    --cov-report=term-missing
    --cov-report=html
    --cov-fail-under=80
markers =
    unit: Unit tests
    integration: Integration tests
    e2e: End-to-end tests
    slow: Slow tests
asyncio_mode = auto
```

#### Test Execution Commands
```bash
# Run all tests
make test

# Run specific test categories
make test-unit
make test-integration
make test-e2e

# Run with coverage
make test-coverage

# Run performance tests
make test-performance
```

#### Mock Strategy for External Dependencies
```python
# tests/conftest.py
import pytest
from unittest.mock import AsyncMock, MagicMock
from flashmm.trading.execution.order_router import CambrianClient

@pytest.fixture
def mock_cambrian_client():
    client = AsyncMock(spec=CambrianClient)
    client.place_order.return_value = {"order_id": "test_order_123"}
    client.get_balance.return_value = {"USDC": 10000.0, "SEI": 50000.0}
    return client

@pytest.fixture
def mock_sei_websocket():
    ws_mock = AsyncMock()
    ws_mock.recv.return_value = {
        "type": "orderbook",
        "symbol": "SEI/USDC",
        "bids": [["0.0420", "1000.0"]],
        "asks": [["0.0422", "1000.0"]]
    }
    return ws_mock
```

## Docker Configuration

### Multi-Stage Dockerfile
**`Dockerfile`**:
```dockerfile
# Build stage
FROM python:3.11-slim as builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry==1.5.1
RUN poetry config virtualenvs.create false

# Copy dependency files
COPY pyproject.toml poetry.lock ./

# Install Python dependencies
RUN poetry install --only main --no-dev

# Runtime stage
FROM python:3.11-slim as runtime

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY src/ ./src/
COPY config/ ./config/
COPY research/models/exported/ ./models/

# Create non-root user
RUN groupadd -r flashmm && useradd -r -g flashmm flashmm
RUN chown -R flashmm:flashmm /app
USER flashmm

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Set environment variables
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

# Start application
CMD ["python", "-m", "flashmm.main"]
```

### Docker Compose Configurations

#### Development Compose
**`docker-compose.dev.yml`**:
```yaml
version: '3.8'

services:
  redis:
    image: redis:7.0-alpine
    ports:
      - "6379:6379"
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data

  influxdb:
    image: influxdb:2.7-alpine
    ports:
      - "8086:8086"
    environment:
      - INFLUXDB_DB=flashmm_dev
      - INFLUXDB_ADMIN_ENABLED=true
      - INFLUXDB_ADMIN_USER=admin
      - INFLUXDB_ADMIN_PASSWORD=password123
    volumes:
      - influxdb_data:/var/lib/influxdb2

  grafana:
    image: grafana/grafana:10.0.0
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin123
    volumes:
      - grafana_data:/var/lib/grafana
      - ./infrastructure/grafana:/etc/grafana/provisioning

volumes:
  redis_data:
  influxdb_data:
  grafana_data:
```

#### Production Compose
**`docker-compose.prod.yml`**:
```yaml
version: '3.8'

services:
  flashmm:
    build:
      context: .
      dockerfile: Dockerfile
    restart: unless-stopped
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=testnet
    env_file:
      - .env
    volumes:
      - ./logs:/app/logs
    depends_on:
      - redis
      - influxdb
    networks:
      - flashmm-network
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '2.0'
        reservations:
          memory: 1G
          cpus: '1.0'

  redis:
    image: redis:7.0-alpine
    restart: unless-stopped
    command: redis-server --appendonly yes --requirepass ${REDIS_PASSWORD}
    volumes:
      - redis_data:/data
    networks:
      - flashmm-network

  influxdb:
    image: influxdb:2.7-alpine
    restart: unless-stopped
    environment:
      - INFLUXDB_DB=flashmm
      - INFLUXDB_ADMIN_USER=admin
      - INFLUXDB_ADMIN_PASSWORD=${INFLUXDB_PASSWORD}
    volumes:
      - influxdb_data:/var/lib/influxdb2
    networks:
      - flashmm-network

  nginx:
    image: nginx:alpine
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./nginx/ssl:/etc/nginx/ssl:ro
    depends_on:
      - flashmm
    networks:
      - flashmm-network

volumes:
  redis_data:
  influxdb_data:

networks:
  flashmm-network:
    driver: bridge
```

## CI/CD Pipeline

### GitHub Actions Workflow
**`.github/workflows/ci-cd.yml`**:
```yaml
name: FlashMM CI/CD

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: flashmm/flashmm

jobs:
  test:
    runs-on: ubuntu-latest
    
    services:
      redis:
        image: redis:7.0-alpine
        ports:
          - 6379:6379
      
      influxdb:
        image: influxdb:2.7-alpine
        ports:
          - 8086:8086
        env:
          INFLUXDB_DB: flashmm_test
          INFLUXDB_ADMIN_USER: admin
          INFLUXDB_ADMIN_PASSWORD: password123

    steps:
    - uses: actions/checkout@v4

    - name: Set up Python 3.11
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'

    - name: Install Poetry
      uses: abatilo/actions-poetry@v2
      with:
        poetry-version: '1.5.1'

    - name: Install dependencies
      run: |
        poetry config virtualenvs.create false
        poetry install --with dev,test

    - name: Run linting
      run: |
        ruff check src/ tests/
        black --check src/ tests/
        mypy src/

    - name: Run security checks
      run: bandit -r src/

    - name: Run unit tests
      run: |
        pytest tests/unit/ -v --cov=src/flashmm --cov-report=xml
      env:
        REDIS_URL: redis://localhost:6379/0
        INFLUXDB_URL: http://localhost:8086

    - name: Run integration tests
      run: |
        pytest tests/integration/ -v
      env:
        REDIS_URL: redis://localhost:6379/0
        INFLUXDB_URL: http://localhost:8086

    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml

  build:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'

    steps:
    - uses: actions/checkout@v4

    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v3

    - name: Log in to Container Registry
      uses: docker/login-action@v3
      with:
        registry: ${{ env.REGISTRY }}
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}

    - name: Extract metadata
      id: meta
      uses: docker/metadata-action@v5
      with:
        images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
        tags: |
          type=ref,event=branch
          type=sha
          type=raw,value=latest,enable={{is_default_branch}}

    - name: Build and push Docker image
      uses: docker/build-push-action@v5
      with:
        context: .
        push: true
        tags: ${{ steps.meta.outputs.tags }}
        labels: ${{ steps.meta.outputs.labels }}
        cache-from: type=gha
        cache-to: type=gha,mode=max

  deploy:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    environment: production

    steps:
    - uses: actions/checkout@v4

    - name: Deploy to VPS
      uses: appleboy/ssh-action@v1.0.0
      with:
        host: ${{ secrets.VPS_HOST }}
        username: ${{ secrets.VPS_USER }}
        key: ${{ secrets.VPS_SSH_KEY }}
        script: |
          cd /opt/flashmm
          
          # Pull latest changes
          git pull origin main
          
          # Update environment variables
          echo "${{ secrets.ENV_FILE }}" > .env
          
          # Pull latest Docker image
          docker pull ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:latest
          
          # Deploy with zero downtime
          docker-compose -f docker-compose.prod.yml up -d --force-recreate
          
          # Health check
          sleep 30
          curl -f http://localhost:8000/health || exit 1
          
          # Cleanup old images
          docker image prune -f
```

## Deployment Strategy

### VPS Setup Script
**`scripts/setup-vps.sh`**:
```bash
#!/bin/bash
set -e

echo "Setting up FlashMM on VPS..."

# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Create application directory
sudo mkdir -p /opt/flashmm
sudo chown $USER:$USER /opt/flashmm
cd /opt/flashmm

# Clone repository
git clone https://github.com/flashmm/flashmm.git .

# Setup SSL certificates (Let's Encrypt)
sudo apt install -y certbot
sudo certbot certonly --standalone -d flashmm.yourdomain.com

# Setup log rotation
sudo tee /etc/logrotate.d/flashmm << EOF
/opt/flashmm/logs/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    create 644 flashmm flashmm
}
EOF

# Setup systemd service for auto-restart
sudo tee /etc/systemd/system/flashmm.service << EOF
[Unit]
Description=FlashMM Market Making Agent
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/opt/flashmm
ExecStart=/usr/local/bin/docker-compose -f docker-compose.prod.yml up -d
ExecStop=/usr/local/bin/docker-compose -f docker-compose.prod.yml down
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl enable flashmm

echo "VPS setup complete!"
echo "1. Copy your .env file to /opt/flashmm/.env"
echo "2. Run: sudo systemctl start flashmm"
echo "3. Check logs: docker-compose -f docker-compose.prod.yml logs -f"
```

### Zero-Downtime Deployment
**`scripts/deploy.sh`**:
```bash
#!/bin/bash
set -e

COMPOSE_FILE="docker-compose.prod.yml"
SERVICE_NAME="flashmm"

echo "Starting zero-downtime deployment..."

# Pull latest image
docker-compose -f $COMPOSE_FILE pull $SERVICE_NAME

# Scale up with new version
docker-compose -f $COMPOSE_FILE up -d --scale $SERVICE_NAME=2 --no-recreate

# Wait for new instance to be healthy
echo "Waiting for new instance to be healthy..."
timeout 60s bash -c 'until curl -f http://localhost:8000/health; do sleep 5; done'

# Remove old instance
docker-compose -f $COMPOSE_FILE up -d --scale $SERVICE_NAME=1 --remove-orphans

echo "Deployment complete!"

# Cleanup
docker image prune -f
```

### Monitoring & Alerts
**`scripts/health-monitor.sh`**:
```bash
#!/bin/bash

# Health monitoring script for cron
WEBHOOK_URL=$1
SERVICE_URL="http://localhost:8000/health"

if ! curl -f $SERVICE_URL > /dev/null 2>&1; then
    # Service is down, send alert
    curl -X POST $WEBHOOK_URL \
        -H 'Content-Type: application/json' \
        -d '{
            "text": "🚨 FlashMM Health Check Failed",
            "attachments": [{
                "color": "danger",
                "fields": [{
                    "title": "Status",
                    "value": "Service appears to be down",
                    "short": true
                }, {
                    "title": "Time", 
                    "value": "'$(date)'",
                    "short": true
                }]
            }]
        }'
    
    # Attempt restart
    cd /opt/flashmm
    docker-compose -f docker-compose.prod.yml restart flashmm
fi
```

### Makefile for Development
**`Makefile`**:
```makefile
.PHONY: help install test lint format clean build deploy

help:
	@echo "Available commands:"
	@echo "  install     Install dependencies"
	@echo "  test        Run all tests"
	@echo "  lint        Run linting"
	@echo "  format      Format code"
	@echo "  build       Build Docker image"
	@echo "  deploy      Deploy to production"

install:
	poetry install --with dev,test
	pre-commit install

test:
	pytest tests/ -v

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v

test-coverage:
	pytest tests/ --cov=src/flashmm --cov-report=html

lint:
	ruff check src/ tests/
	black --check src/ tests/
	mypy src/

format:
	ruff check --fix src/ tests/
	black src/ tests/

security:
	bandit -r src/

clean:
	docker system prune -f
	docker volume prune -f

build:
	docker build -t flashmm:latest .

dev-up:
	docker-compose -f docker-compose.dev.yml up -d

dev-down:
	docker-compose -f docker-compose.dev.yml down

deploy-prod:
	./scripts/deploy.sh

logs:
	docker-compose -f docker-compose.prod.yml logs -f flashmm
```

This deployment workflow provides a robust foundation for rapid development and reliable production deployment suitable for the hackathon timeline and single-VPS constraint.
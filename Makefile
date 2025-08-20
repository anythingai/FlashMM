# FlashMM Development Makefile
.PHONY: help install test lint format clean build deploy docs

# Default target
help:
	@echo "FlashMM Development Commands:"
	@echo ""
	@echo "Setup & Installation:"
	@echo "  install          Install dependencies and setup development environment"
	@echo "  install-dev      Install with development dependencies"
	@echo "  setup-hooks      Setup pre-commit hooks"
	@echo ""
	@echo "Code Quality:"
	@echo "  lint             Run all linting checks"
	@echo "  format           Format code with black and ruff"
	@echo "  type-check       Run mypy type checking"
	@echo "  security         Run security checks with bandit"
	@echo ""
	@echo "Testing:"
	@echo "  test             Run all tests"
	@echo "  test-unit        Run unit tests only"
	@echo "  test-integration Run integration tests only"
	@echo "  test-e2e         Run end-to-end tests"
	@echo "  test-coverage    Run tests with coverage report"
	@echo "  test-performance Run performance tests"
	@echo ""
	@echo "Docker & Deployment:"
	@echo "  build            Build Docker image"
	@echo "  dev-up           Start development services"
	@echo "  dev-down         Stop development services"
	@echo "  deploy-test      Deploy to test environment"
	@echo ""
	@echo "Monitoring & Logs:"
	@echo "  logs             Show application logs"
	@echo "  logs-follow      Follow application logs"
	@echo "  monitoring       Start monitoring stack"
	@echo ""
	@echo "Utilities:"
	@echo "  clean            Clean up build artifacts"
	@echo "  docs             Generate documentation"
	@echo "  shell            Open development shell"

# =============================================================================
# Setup & Installation
# =============================================================================

install:
	@echo "Installing FlashMM dependencies..."
	poetry install
	@echo "Installation complete!"

install-dev:
	@echo "Installing FlashMM with development dependencies..."
	poetry install --with dev,test
	@echo "Development installation complete!"

setup-hooks:
	@echo "Setting up pre-commit hooks..."
	poetry run pre-commit install
	@echo "Pre-commit hooks installed!"

# =============================================================================
# Code Quality
# =============================================================================

lint:
	@echo "Running linting checks..."
	poetry run ruff check src/ tests/
	poetry run black --check src/ tests/
	poetry run mypy src/
	@echo "Linting complete!"

format:
	@echo "Formatting code..."
	poetry run ruff check --fix src/ tests/
	poetry run black src/ tests/
	@echo "Code formatting complete!"

type-check:
	@echo "Running type checks..."
	poetry run mypy src/
	@echo "Type checking complete!"

security:
	@echo "Running security checks..."
	poetry run bandit -r src/
	@echo "Security checks complete!"

# =============================================================================
# Testing
# =============================================================================

test:
	@echo "Running all tests..."
	poetry run pytest tests/ -v
	@echo "All tests complete!"

test-unit:
	@echo "Running unit tests..."
	poetry run pytest tests/unit/ -v -m "unit"
	@echo "Unit tests complete!"

test-integration:
	@echo "Running integration tests..."
	poetry run pytest tests/integration/ -v -m "integration"
	@echo "Integration tests complete!"

test-e2e:
	@echo "Running end-to-end tests..."
	poetry run pytest tests/end_to_end/ -v -m "e2e"
	@echo "E2E tests complete!"

test-coverage:
	@echo "Running tests with coverage..."
	poetry run pytest tests/ --cov=src/flashmm --cov-report=html --cov-report=term-missing
	@echo "Coverage report generated in htmlcov/"

test-performance:
	@echo "Running performance tests..."
	poetry run pytest tests/ -v -m "slow"
	@echo "Performance tests complete!"

# =============================================================================
# Docker & Deployment
# =============================================================================

build:
	@echo "Building Docker image..."
	docker build -t flashmm:latest .
	@echo "Docker image built successfully!"

dev-up:
	@echo "Starting development services..."
	docker-compose -f docker-compose.dev.yml up -d
	@echo "Development services started!"

dev-down:
	@echo "Stopping development services..."
	docker-compose -f docker-compose.dev.yml down
	@echo "Development services stopped!"

deploy-test:
	@echo "Deploying to test environment..."
	docker-compose -f docker-compose.yml up -d
	@echo "Test deployment complete!"

# =============================================================================
# Monitoring & Logs
# =============================================================================

logs:
	@echo "Showing application logs..."
	docker-compose logs flashmm

logs-follow:
	@echo "Following application logs..."
	docker-compose logs -f flashmm

monitoring:
	@echo "Starting monitoring stack..."
	docker-compose up -d grafana influxdb
	@echo "Monitoring stack started!"
	@echo "Grafana: http://localhost:3000 (admin/admin123)"
	@echo "InfluxDB: http://localhost:8086"

# =============================================================================
# Development Utilities
# =============================================================================

clean:
	@echo "Cleaning up build artifacts..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ .coverage htmlcov/ .pytest_cache/ .mypy_cache/ .ruff_cache/
	docker system prune -f
	@echo "Cleanup complete!"

docs:
	@echo "Generating documentation..."
	# Add documentation generation command here when ready
	@echo "Documentation generated!"

shell:
	@echo "Opening development shell..."
	poetry shell

# =============================================================================
# Database & Data Management
# =============================================================================

db-reset:
	@echo "Resetting databases..."
	docker-compose down -v
	docker volume prune -f
	docker-compose up -d redis influxdb
	@echo "Databases reset!"

db-backup:
	@echo "Backing up databases..."
	mkdir -p backups
	docker exec flashmm-redis redis-cli --rdb /data/dump.rdb
	docker cp flashmm-redis:/data/dump.rdb backups/redis-$(shell date +%Y%m%d-%H%M%S).rdb
	@echo "Database backup complete!"

# =============================================================================
# Model Management
# =============================================================================

models-download:
	@echo "Downloading ML models..."
	mkdir -p research/models/exported
	# Add model download commands here
	@echo "Models downloaded!"

models-validate:
	@echo "Validating ML models..."
	poetry run python -m flashmm.ml.utils.model_validator
	@echo "Model validation complete!"

# =============================================================================
# Development Server
# =============================================================================

dev-server:
	@echo "Starting development server..."
	ENVIRONMENT=development poetry run python -m flashmm.main

dev-server-debug:
	@echo "Starting development server with debug mode..."
	ENVIRONMENT=development FLASHMM_DEBUG=true poetry run python -m flashmm.main

# =============================================================================
# Health Checks
# =============================================================================

health-check:
	@echo "Running health checks..."
	curl -f http://localhost:8000/health || echo "Service not running"

ping-services:
	@echo "Pinging external services..."
	curl -f https://sei-testnet-rpc.polkachu.com/status || echo "Sei RPC not accessible"
	# Add other service pings here

# =============================================================================
# Configuration Management
# =============================================================================

config-validate:
	@echo "Validating configuration..."
	poetry run python -c "from flashmm.config.settings import get_config; config = get_config(); print('Configuration valid!')"

config-show:
	@echo "Showing current configuration..."
	poetry run python -c "from flashmm.config.settings import get_config; import json; config = get_config(); print(json.dumps(config.get_all(), indent=2))"

# =============================================================================
# Quick Development Shortcuts
# =============================================================================

quick-start: install-dev setup-hooks dev-up
	@echo "Quick start complete! Run 'make dev-server' to start the application."

quick-test: lint test-unit
	@echo "Quick testing complete!"

quick-deploy: build deploy-test health-check
	@echo "Quick deployment complete!"
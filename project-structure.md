# FlashMM Project Directory Structure

```
flashmm/
├── README.md
├── prd.md
├── .gitignore
├── .env.example
├── .env
├── requirements.txt
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml
├── setup.py
├── Makefile
│
├── src/
│   └── flashmm/
│       ├── __init__.py
│       ├── main.py                    # Application entry point
│       ├── config/
│       │   ├── __init__.py
│       │   ├── settings.py            # Configuration management
│       │   ├── environments.py        # Environment-specific configs
│       │   └── constants.py           # Application constants
│       │
│       ├── data/
│       │   ├── __init__.py
│       │   ├── ingestion/
│       │   │   ├── __init__.py
│       │   │   ├── websocket_client.py # Sei WebSocket client
│       │   │   ├── data_normalizer.py  # Order book normalization
│       │   │   └── feed_manager.py     # Data feed coordination
│       │   │
│       │   ├── storage/
│       │   │   ├── __init__.py
│       │   │   ├── influxdb_client.py  # InfluxDB integration
│       │   │   ├── redis_client.py     # Redis for real-time data
│       │   │   └── data_models.py      # Data schemas
│       │   │
│       │   └── preprocessing/
│       │       ├── __init__.py
│       │       ├── feature_engineering.py
│       │       └── data_validation.py
│       │
│       ├── ml/
│       │   ├── __init__.py
│       │   ├── models/
│       │   │   ├── __init__.py
│       │   │   ├── predictor.py        # Main prediction interface
│       │   │   ├── transformer_model.py
│       │   │   ├── lstm_model.py
│       │   │   └── ensemble_model.py
│       │   │
│       │   ├── training/
│       │   │   ├── __init__.py
│       │   │   ├── trainer.py
│       │   │   ├── data_loader.py
│       │   │   └── validation.py
│       │   │
│       │   ├── inference/
│       │   │   ├── __init__.py
│       │   │   ├── inference_engine.py
│       │   │   ├── model_loader.py
│       │   │   └── torchscript_optimizer.py
│       │   │
│       │   └── utils/
│       │       ├── __init__.py
│       │       ├── model_utils.py
│       │       └── metrics.py
│       │
│       ├── trading/
│       │   ├── __init__.py
│       │   ├── strategy/
│       │   │   ├── __init__.py
│       │   │   ├── quoting_strategy.py  # Main quoting logic
│       │   │   ├── spread_calculator.py
│       │   │   └── inventory_manager.py
│       │   │
│       │   ├── execution/
│       │   │   ├── __init__.py
│       │   │   ├── order_router.py      # Cambrian SDK integration
│       │   │   ├── order_manager.py     # Order lifecycle management
│       │   │   └── fill_handler.py      # Trade execution handling
│       │   │
│       │   └── risk/
│       │       ├── __init__.py
│       │       ├── risk_manager.py      # Risk controls
│       │       ├── position_tracker.py  # Inventory tracking
│       │       └── circuit_breakers.py  # Kill switches
│       │
│       ├── api/
│       │   ├── __init__.py
│       │   ├── app.py                   # FastAPI application
│       │   ├── routers/
│       │   │   ├── __init__.py
│       │   │   ├── health.py            # Health endpoints
│       │   │   ├── metrics.py           # Metrics endpoints
│       │   │   ├── trading.py           # Trading status endpoints
│       │   │   └── admin.py             # Admin controls
│       │   │
│       │   ├── middleware/
│       │   │   ├── __init__.py
│       │   │   ├── auth.py              # Authentication
│       │   │   ├── rate_limiter.py      # Rate limiting
│       │   │   └── cors.py              # CORS handling
│       │   │
│       │   └── schemas/
│       │       ├── __init__.py
│       │       ├── health.py
│       │       ├── metrics.py
│       │       └── trading.py
│       │
│       ├── monitoring/
│       │   ├── __init__.py
│       │   ├── telemetry/
│       │   │   ├── __init__.py
│       │   │   ├── metrics_collector.py # System metrics
│       │   │   ├── performance_tracker.py
│       │   │   └── alerting.py          # Alert system
│       │   │
│       │   ├── dashboard/
│       │   │   ├── __init__.py
│       │   │   ├── grafana_client.py    # Grafana integration
│       │   │   └── dashboard_generator.py
│       │   │
│       │   └── social/
│       │       ├── __init__.py
│       │       ├── twitter_bot.py       # X/Twitter integration
│       │       └── report_generator.py  # Performance reports
│       │
│       ├── utils/
│       │   ├── __init__.py
│       │   ├── logging.py               # Logging configuration
│       │   ├── exceptions.py            # Custom exceptions
│       │   ├── decorators.py            # Utility decorators
│       │   ├── math_utils.py            # Mathematical utilities
│       │   └── time_utils.py            # Time handling utilities
│       │
│       └── security/
│           ├── __init__.py
│           ├── key_manager.py           # Cryptographic key management
│           ├── encryption.py            # Encryption utilities
│           └── auth.py                  # Authentication helpers
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py                      # pytest configuration
│   ├── unit/
│   │   ├── __init__.py
│   │   ├── test_data/
│   │   ├── test_ml/
│   │   ├── test_trading/
│   │   ├── test_api/
│   │   └── test_utils/
│   │
│   ├── integration/
│   │   ├── __init__.py
│   │   ├── test_data_pipeline.py
│   │   ├── test_trading_flow.py
│   │   └── test_api_endpoints.py
│   │
│   └── end_to_end/
│       ├── __init__.py
│       └── test_full_system.py
│
├── research/
│   ├── notebooks/
│   │   ├── data_exploration.ipynb
│   │   ├── model_development.ipynb
│   │   ├── backtesting.ipynb
│   │   └── performance_analysis.ipynb
│   │
│   ├── data/
│   │   ├── raw/                         # Raw historical data
│   │   ├── processed/                   # Processed datasets
│   │   └── synthetic/                   # Synthetic data for testing
│   │
│   ├── models/
│   │   ├── checkpoints/                 # Training checkpoints
│   │   ├── exported/                    # TorchScript models
│   │   └── experiments/                 # Experimental models
│   │
│   └── scripts/
│       ├── data_collection.py           # Historical data collection
│       ├── model_training.py            # Training scripts
│       ├── backtesting.py               # Backtesting framework
│       └── model_export.py              # TorchScript export
│
├── config/
│   ├── environments/
│   │   ├── development.yml
│   │   ├── testnet.yml
│   │   ├── production.yml
│   │   └── local.yml
│   │
│   ├── logging/
│   │   ├── development.yml
│   │   ├── production.yml
│   │   └── testing.yml
│   │
│   ├── docker/
│   │   ├── app.dockerfile
│   │   ├── ml.dockerfile
│   │   └── nginx.dockerfile
│   │
│   └── kubernetes/                      # Future K8s deployment
│       ├── deployment.yml
│       ├── service.yml
│       └── configmap.yml
│
├── docs/
│   ├── architecture/
│   │   ├── system-design.md
│   │   ├── data-flow.md
│   │   ├── api-specification.md
│   │   └── security-model.md
│   │
│   ├── deployment/
│   │   ├── docker-setup.md
│   │   ├── vps-configuration.md
│   │   └── monitoring-setup.md
│   │
│   ├── development/
│   │   ├── getting-started.md
│   │   ├── contributing.md
│   │   ├── testing-guide.md
│   │   └── code-standards.md
│   │
│   └── operations/
│       ├── runbook.md
│       ├── troubleshooting.md
│       └── maintenance.md
│
├── scripts/
│   ├── setup/
│   │   ├── install-dependencies.sh
│   │   ├── setup-environment.sh
│   │   └── initialize-databases.sh
│   │
│   ├── deployment/
│   │   ├── deploy.sh
│   │   ├── rollback.sh
│   │   └── health-check.sh
│   │
│   ├── maintenance/
│   │   ├── backup-data.sh
│   │   ├── update-models.sh
│   │   └── cleanup-logs.sh
│   │
│   └── monitoring/
│       ├── system-health.sh
│       ├── performance-check.sh
│       └── alert-test.sh
│
├── infrastructure/
│   ├── terraform/                       # Future infrastructure as code
│   │   ├── main.tf
│   │   ├── variables.tf
│   │   └── outputs.tf
│   │
│   ├── grafana/
│   │   ├── dashboard-configs/
│   │   ├── alerting-rules/
│   │   └── provisioning/
│   │
│   └── monitoring/
│       ├── prometheus.yml
│       ├── influxdb.conf
│       └── telegraf.conf
│
└── deploy/
    ├── docker-compose.yml
    ├── docker-compose.dev.yml
    ├── docker-compose.prod.yml
    ├── .env.example
    └── nginx/
        ├── nginx.conf
        └── ssl/
```

## Key Directory Explanations

### `/src/flashmm/` - Core Application
- **`data/`**: Handles all data ingestion, storage, and preprocessing
- **`ml/`**: Machine learning models, training, and inference pipeline
- **`trading/`**: Trading strategy, execution, and risk management
- **`api/`**: FastAPI web service for metrics and control
- **`monitoring/`**: Telemetry, dashboards, and social media integration
- **`security/`**: Cryptographic operations and key management

### `/research/` - ML Development
- Jupyter notebooks for model development and analysis
- Training data and model artifacts
- Backtesting and performance analysis tools

### `/config/` - Configuration Management
- Environment-specific configurations
- Docker and deployment configurations
- Future Kubernetes manifests

### `/infrastructure/` - Monitoring & Infrastructure
- Grafana dashboard configurations
- Monitoring tool configurations
- Future Terraform infrastructure code

### `/deploy/` - Deployment Artifacts
- Docker Compose files for different environments
- Nginx configuration for reverse proxy
- SSL certificate management
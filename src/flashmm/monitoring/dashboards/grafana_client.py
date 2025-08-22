"""
FlashMM Grafana Dashboard Integration

Comprehensive Grafana Cloud integration with automatic dashboard provisioning,
real-time data source configuration, and alert rule management for FlashMM monitoring.
"""

from dataclasses import dataclass
from typing import Any

import aiohttp

from flashmm.config.settings import get_config
from flashmm.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class DashboardConfig:
    """Dashboard configuration."""
    title: str
    uid: str
    tags: list[str]
    folder: str
    refresh_interval: str = "5s"
    time_range: str = "1h"
    public_access: bool = False
    description: str = ""


@dataclass
class DataSourceConfig:
    """Data source configuration."""
    name: str
    type: str
    url: str
    database: str
    username: str = ""
    password: str = ""
    access: str = "proxy"
    is_default: bool = False


@dataclass
class AlertRule:
    """Alert rule configuration."""
    title: str
    uid: str
    condition: str
    evaluation_interval: str
    for_duration: str
    no_data_state: str = "NoData"
    exec_err_state: str = "Alerting"
    annotations: dict[str, str] | None = None
    labels: dict[str, str] | None = None


class GrafanaClient:
    """Grafana API client with comprehensive dashboard management."""

    def __init__(self):
        self.config = get_config()

        # Grafana configuration
        self.base_url = self.config.get("grafana.url", "https://flashmm.grafana.net")
        self.api_key = self.config.get("grafana.api_key", "")
        self.org_id = self.config.get("grafana.org_id", 1)

        # InfluxDB configuration for data sources
        self.influxdb_url = self.config.get("storage.influxdb.host", "localhost:8086")
        self.influxdb_database = self.config.get("storage.influxdb.database", "flashmm")
        self.influxdb_org = self.config.get("storage.influxdb.org", "flashmm")
        self.influxdb_bucket = self.config.get("storage.influxdb.bucket", "metrics")

        # Session for HTTP requests
        self.session: aiohttp.ClientSession | None = None

        # Dashboard templates cache
        self._dashboard_templates = {}

        logger.info(f"GrafanaClient initialized for {self.base_url}")

    async def initialize(self) -> None:
        """Initialize Grafana client and session."""
        try:
            if not self.api_key:
                logger.warning("No Grafana API key configured - dashboard features will be limited")
                return

            # Create session with authentication headers
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json"
            }

            connector = aiohttp.TCPConnector(limit=10, limit_per_host=5)
            timeout = aiohttp.ClientTimeout(total=30)

            self.session = aiohttp.ClientSession(
                headers=headers,
                connector=connector,
                timeout=timeout
            )

            # Verify connection
            await self._verify_connection()

            logger.info("GrafanaClient initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize GrafanaClient: {e}")
            raise

    async def _verify_connection(self) -> None:
        """Verify connection to Grafana API."""
        if not self.session:
            raise ConnectionError("Grafana session not initialized")

        try:
            async with self.session.get(f"{self.base_url}/api/org") as response:
                if response.status == 200:
                    org_info = await response.json()
                    logger.info(f"Connected to Grafana org: {org_info.get('name', 'Unknown')}")
                else:
                    logger.error(f"Failed to connect to Grafana: HTTP {response.status}")
                    raise ConnectionError(f"Grafana connection failed: {response.status}")
        except Exception as e:
            logger.error(f"Grafana connection verification failed: {e}")
            raise

    async def setup_data_sources(self) -> list[dict[str, Any]]:
        """Setup InfluxDB data sources for FlashMM metrics."""
        try:
            data_sources = []

            # Main InfluxDB data source for metrics
            influxdb_config = DataSourceConfig(
                name="FlashMM-InfluxDB",
                type="influxdb",
                url=f"http://{self.influxdb_url}",
                database=self.influxdb_database,
                access="proxy",
                is_default=True
            )

            # Create or update data source
            ds_result = await self._create_or_update_data_source(influxdb_config)
            data_sources.append(ds_result)

            # Additional data source for real-time metrics (Redis/WebSocket)
            realtime_config = DataSourceConfig(
                name="FlashMM-Realtime",
                type="redis",
                url="redis://localhost:6379",
                database="0",
                access="proxy"
            )

            # Create realtime data source if Redis plugin is available
            try:
                rt_result = await self._create_or_update_data_source(realtime_config)
                data_sources.append(rt_result)
            except Exception as e:
                logger.warning(f"Redis data source creation failed (plugin may not be installed): {e}")

            logger.info(f"Setup {len(data_sources)} data sources successfully")
            return data_sources

        except Exception as e:
            logger.error(f"Failed to setup data sources: {e}")
            raise

    async def _create_or_update_data_source(self, config: DataSourceConfig) -> dict[str, Any]:
        """Create or update a data source."""
        try:
            # Check if data source exists
            existing_ds = await self._get_data_source_by_name(config.name)

            ds_data = {
                "name": config.name,
                "type": config.type,
                "url": config.url,
                "access": config.access,
                "isDefault": config.is_default,
                "database": config.database,
                "user": config.username,
                "password": config.password,
                "jsonData": {},
                "secureJsonData": {}
            }

            if config.type == "influxdb":
                ds_data["jsonData"] = {
                    "organization": self.influxdb_org,
                    "defaultBucket": self.influxdb_bucket,
                    "version": "Flux",
                    "httpMode": "POST"
                }

            if not self.session:
                raise ConnectionError("Grafana session not initialized")

            if existing_ds:
                # Update existing data source
                ds_data["id"] = existing_ds["id"]
                async with self.session.put(
                    f"{self.base_url}/api/datasources/{existing_ds['id']}",
                    json=ds_data
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.info(f"Updated data source: {config.name}")
                        return result
                    else:
                        error_text = await response.text()
                        raise Exception(f"Failed to update data source: {error_text}")
            else:
                # Create new data source
                async with self.session.post(
                    f"{self.base_url}/api/datasources",
                    json=ds_data
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.info(f"Created data source: {config.name}")
                        return result
                    else:
                        error_text = await response.text()
                        raise Exception(f"Failed to create data source: {error_text}")

        except Exception as e:
            logger.error(f"Failed to create/update data source {config.name}: {e}")
            raise

    async def _get_data_source_by_name(self, name: str) -> dict[str, Any] | None:
        """Get data source by name."""
        if not self.session:
            return None

        try:
            async with self.session.get(f"{self.base_url}/api/datasources/name/{name}") as response:
                if response.status == 200:
                    return await response.json()
                elif response.status == 404:
                    return None
                else:
                    error_text = await response.text()
                    logger.warning(f"Failed to get data source {name}: {error_text}")
                    return None
        except Exception as e:
            logger.warning(f"Error getting data source {name}: {e}")
            return None

    async def provision_dashboards(self) -> list[dict[str, Any]]:
        """Provision all FlashMM dashboards."""
        try:
            dashboards = []

            # Trading Performance Dashboard
            trading_dashboard = await self._create_trading_dashboard()
            dashboards.append(trading_dashboard)

            # System Performance Dashboard
            system_dashboard = await self._create_system_dashboard()
            dashboards.append(system_dashboard)

            # Risk Monitoring Dashboard
            risk_dashboard = await self._create_risk_dashboard()
            dashboards.append(risk_dashboard)

            # ML Performance Dashboard
            ml_dashboard = await self._create_ml_dashboard()
            dashboards.append(ml_dashboard)

            # Public Demo Dashboard
            public_dashboard = await self._create_public_dashboard()
            dashboards.append(public_dashboard)

            logger.info(f"Provisioned {len(dashboards)} dashboards successfully")
            return dashboards

        except Exception as e:
            logger.error(f"Failed to provision dashboards: {e}")
            raise

    async def _create_trading_dashboard(self) -> dict[str, Any]:
        """Create comprehensive trading performance dashboard."""
        config = DashboardConfig(
            title="FlashMM Trading Performance",
            uid="flashmm-trading",
            tags=["flashmm", "trading", "performance"],
            folder="FlashMM",
            description="Real-time trading performance metrics including P&L, spreads, and volume"
        )

        dashboard_json = {
            "dashboard": {
                "id": None,
                "uid": config.uid,
                "title": config.title,
                "tags": config.tags,
                "timezone": "UTC",
                "refresh": config.refresh_interval,
                "time": {
                    "from": "now-1h",
                    "to": "now"
                },
                "timepicker": {
                    "refresh_intervals": ["5s", "10s", "30s", "1m", "5m", "15m", "30m", "1h"]
                },
                "panels": [
                    self._create_spread_improvement_panel(),
                    self._create_pnl_panel(),
                    self._create_volume_panel(),
                    self._create_fill_rate_panel(),
                    self._create_inventory_panel(),
                    self._create_trading_latency_panel()
                ],
                "templating": {
                    "list": [
                        {
                            "name": "symbol",
                            "type": "custom",
                            "options": [
                                {"text": "SEI/USDC", "value": "SEI/USDC", "selected": True},
                                {"text": "All", "value": "*", "selected": False}
                            ]
                        }
                    ]
                }
            },
            "folderId": 0,
            "overwrite": True
        }

        return await self._create_or_update_dashboard(dashboard_json)

    def _create_spread_improvement_panel(self) -> dict[str, Any]:
        """Create spread improvement panel."""
        return {
            "id": 1,
            "title": "Spread Improvement",
            "type": "stat",
            "gridPos": {"h": 8, "w": 6, "x": 0, "y": 0},
            "targets": [
                {
                    "query": 'from(bucket: "metrics") |> range(start: -1h) |> filter(fn: (r) => r._measurement == "trading_performance_metrics") |> filter(fn: (r) => r._field == "spread_improvement_percent") |> last()',
                    "refId": "A"
                }
            ],
            "fieldConfig": {
                "defaults": {
                    "unit": "percent",
                    "min": 0,
                    "max": 100,
                    "thresholds": {
                        "steps": [
                            {"color": "red", "value": 0},
                            {"color": "yellow", "value": 20},
                            {"color": "green", "value": 40}
                        ]
                    }
                }
            },
            "options": {
                "reduceOptions": {
                    "values": False,
                    "calcs": ["lastNotNull"],
                    "fields": ""
                },
                "orientation": "horizontal",
                "textMode": "value_and_name",
                "colorMode": "background",
                "graphMode": "area",
                "justifyMode": "center"
            }
        }

    def _create_pnl_panel(self) -> dict[str, Any]:
        """Create P&L tracking panel."""
        return {
            "id": 2,
            "title": "Profit & Loss",
            "type": "timeseries",
            "gridPos": {"h": 8, "w": 12, "x": 6, "y": 0},
            "targets": [
                {
                    "query": 'from(bucket: "metrics") |> range(start: -1h) |> filter(fn: (r) => r._measurement == "trading_performance_metrics") |> filter(fn: (r) => r._field == "total_pnl_usdc") |> aggregateWindow(every: 30s, fn: mean)',
                    "refId": "A",
                    "legendFormat": "Total P&L"
                },
                {
                    "query": 'from(bucket: "metrics") |> range(start: -1h) |> filter(fn: (r) => r._measurement == "trading_performance_metrics") |> filter(fn: (r) => r._field == "realized_pnl_usdc") |> aggregateWindow(every: 30s, fn: mean)',
                    "refId": "B",
                    "legendFormat": "Realized P&L"
                },
                {
                    "query": 'from(bucket: "metrics") |> range(start: -1h) |> filter(fn: (r) => r._measurement == "trading_performance_metrics") |> filter(fn: (r) => r._field == "unrealized_pnl_usdc") |> aggregateWindow(every: 30s, fn: mean)',
                    "refId": "C",
                    "legendFormat": "Unrealized P&L"
                }
            ],
            "fieldConfig": {
                "defaults": {
                    "unit": "currencyUSD",
                    "custom": {
                        "drawStyle": "line",
                        "lineInterpolation": "linear",
                        "lineWidth": 2,
                        "fillOpacity": 10,
                        "gradientMode": "none",
                        "spanNulls": False,
                        "insertNulls": False,
                        "showPoints": "never",
                        "pointSize": 5,
                        "stacking": {"mode": "none", "group": "A"},
                        "axisPlacement": "auto",
                        "axisLabel": "",
                        "scaleDistribution": {"type": "linear"},
                        "hideFrom": {"legend": False, "tooltip": False, "vis": False},
                        "thresholdsStyle": {"mode": "off"}
                    }
                }
            }
        }

    def _create_volume_panel(self) -> dict[str, Any]:
        """Create trading volume panel."""
        return {
            "id": 3,
            "title": "Trading Volume",
            "type": "stat",
            "gridPos": {"h": 8, "w": 6, "x": 18, "y": 0},
            "targets": [
                {
                    "query": 'from(bucket: "metrics") |> range(start: -1h) |> filter(fn: (r) => r._measurement == "trading_performance_metrics") |> filter(fn: (r) => r._field == "total_volume_usdc") |> last()',
                    "refId": "A"
                }
            ],
            "fieldConfig": {
                "defaults": {
                    "unit": "currencyUSD",
                    "decimals": 0
                }
            }
        }

    def _create_fill_rate_panel(self) -> dict[str, Any]:
        """Create fill rate panel."""
        return {
            "id": 4,
            "title": "Fill Rate",
            "type": "gauge",
            "gridPos": {"h": 8, "w": 6, "x": 0, "y": 8},
            "targets": [
                {
                    "query": 'from(bucket: "metrics") |> range(start: -1h) |> filter(fn: (r) => r._measurement == "trading_performance_metrics") |> filter(fn: (r) => r._field == "fill_rate_percent") |> last()',
                    "refId": "A"
                }
            ],
            "fieldConfig": {
                "defaults": {
                    "unit": "percent",
                    "min": 0,
                    "max": 100,
                    "thresholds": {
                        "steps": [
                            {"color": "red", "value": 0},
                            {"color": "yellow", "value": 50},
                            {"color": "green", "value": 80}
                        ]
                    }
                }
            }
        }

    def _create_inventory_panel(self) -> dict[str, Any]:
        """Create inventory utilization panel."""
        return {
            "id": 5,
            "title": "Inventory Utilization",
            "type": "bargauge",
            "gridPos": {"h": 8, "w": 6, "x": 6, "y": 8},
            "targets": [
                {
                    "query": 'from(bucket: "metrics") |> range(start: -1h) |> filter(fn: (r) => r._measurement == "trading_performance_metrics") |> filter(fn: (r) => r._field == "inventory_utilization_percent") |> last()',
                    "refId": "A"
                }
            ],
            "fieldConfig": {
                "defaults": {
                    "unit": "percent",
                    "min": 0,
                    "max": 100,
                    "thresholds": {
                        "steps": [
                            {"color": "green", "value": 0},
                            {"color": "yellow", "value": 70},
                            {"color": "red", "value": 90}
                        ]
                    }
                }
            }
        }

    def _create_trading_latency_panel(self) -> dict[str, Any]:
        """Create trading latency panel."""
        return {
            "id": 6,
            "title": "Order Latency",
            "type": "timeseries",
            "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8},
            "targets": [
                {
                    "query": 'from(bucket: "metrics") |> range(start: -1h) |> filter(fn: (r) => r._measurement == "trading_performance_metrics") |> filter(fn: (r) => r._field == "order_latency_ms") |> aggregateWindow(every: 30s, fn: mean)',
                    "refId": "A"
                }
            ],
            "fieldConfig": {
                "defaults": {
                    "unit": "ms",
                    "custom": {
                        "thresholdsStyle": {"mode": "line"},
                        "thresholds": {
                            "steps": [
                                {"color": "transparent", "value": None},
                                {"color": "red", "value": 100}
                            ]
                        }
                    }
                }
            }
        }

    async def _create_system_dashboard(self) -> dict[str, Any]:
        """Create system performance dashboard."""
        config = DashboardConfig(
            title="FlashMM System Performance",
            uid="flashmm-system",
            tags=["flashmm", "system", "performance"],
            folder="FlashMM",
            description="System resource monitoring and performance metrics"
        )

        dashboard_json = {
            "dashboard": {
                "id": None,
                "uid": config.uid,
                "title": config.title,
                "tags": config.tags,
                "timezone": "UTC",
                "refresh": config.refresh_interval,
                "time": {"from": "now-1h", "to": "now"},
                "panels": [
                    {
                        "id": 1,
                        "title": "CPU Usage",
                        "type": "timeseries",
                        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
                        "targets": [
                            {
                                "query": 'from(bucket: "metrics") |> range(start: -1h) |> filter(fn: (r) => r._measurement == "system_resource_metrics") |> filter(fn: (r) => r._field == "cpu_percent")',
                                "refId": "A"
                            }
                        ]
                    },
                    {
                        "id": 2,
                        "title": "Memory Usage",
                        "type": "timeseries",
                        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0},
                        "targets": [
                            {
                                "query": 'from(bucket: "metrics") |> range(start: -1h) |> filter(fn: (r) => r._measurement == "system_resource_metrics") |> filter(fn: (r) => r._field == "memory_percent")',
                                "refId": "A"
                            }
                        ]
                    }
                ]
            },
            "folderId": 0,
            "overwrite": True
        }

        return await self._create_or_update_dashboard(dashboard_json)

    async def _create_risk_dashboard(self) -> dict[str, Any]:
        """Create risk monitoring dashboard."""
        config = DashboardConfig(
            title="FlashMM Risk Monitoring",
            uid="flashmm-risk",
            tags=["flashmm", "risk", "monitoring"],
            folder="FlashMM",
            description="Risk management and compliance monitoring"
        )

        dashboard_json = {
            "dashboard": {
                "id": None,
                "uid": config.uid,
                "title": config.title,
                "tags": config.tags,
                "timezone": "UTC",
                "refresh": config.refresh_interval,
                "time": {"from": "now-1h", "to": "now"},
                "panels": [
                    {
                        "id": 1,
                        "title": "Risk Score",
                        "type": "gauge",
                        "gridPos": {"h": 8, "w": 6, "x": 0, "y": 0},
                        "targets": [
                            {
                                "query": 'from(bucket: "metrics") |> range(start: -1h) |> filter(fn: (r) => r._measurement == "risk_metrics") |> filter(fn: (r) => r._field == "risk_score") |> last()',
                                "refId": "A"
                            }
                        ]
                    },
                    {
                        "id": 2,
                        "title": "Value at Risk (95%)",
                        "type": "stat",
                        "gridPos": {"h": 8, "w": 6, "x": 6, "y": 0},
                        "targets": [
                            {
                                "query": 'from(bucket: "metrics") |> range(start: -1h) |> filter(fn: (r) => r._measurement == "risk_metrics") |> filter(fn: (r) => r._field == "var_1d_95_usdc") |> last()',
                                "refId": "A"
                            }
                        ]
                    }
                ]
            },
            "folderId": 0,
            "overwrite": True
        }

        return await self._create_or_update_dashboard(dashboard_json)

    async def _create_ml_dashboard(self) -> dict[str, Any]:
        """Create ML performance dashboard."""
        config = DashboardConfig(
            title="FlashMM ML Performance",
            uid="flashmm-ml",
            tags=["flashmm", "ml", "ai"],
            folder="FlashMM",
            description="Machine learning model performance and cost tracking"
        )

        dashboard_json = {
            "dashboard": {
                "id": None,
                "uid": config.uid,
                "title": config.title,
                "tags": config.tags,
                "timezone": "UTC",
                "refresh": config.refresh_interval,
                "time": {"from": "now-1h", "to": "now"},
                "panels": [
                    {
                        "id": 1,
                        "title": "Prediction Accuracy",
                        "type": "stat",
                        "gridPos": {"h": 8, "w": 6, "x": 0, "y": 0},
                        "targets": [
                            {
                                "query": 'from(bucket: "metrics") |> range(start: -1h) |> filter(fn: (r) => r._measurement == "ml_performance_metrics") |> filter(fn: (r) => r._field == "prediction_accuracy_percent") |> last()',
                                "refId": "A"
                            }
                        ]
                    },
                    {
                        "id": 2,
                        "title": "Inference Latency",
                        "type": "timeseries",
                        "gridPos": {"h": 8, "w": 12, "x": 6, "y": 0},
                        "targets": [
                            {
                                "query": 'from(bucket: "metrics") |> range(start: -1h) |> filter(fn: (r) => r._measurement == "ml_performance_metrics") |> filter(fn: (r) => r._field == "avg_inference_time_ms")',
                                "refId": "A"
                            }
                        ]
                    }
                ]
            },
            "folderId": 0,
            "overwrite": True
        }

        return await self._create_or_update_dashboard(dashboard_json)

    async def _create_public_dashboard(self) -> dict[str, Any]:
        """Create public demo dashboard."""
        config = DashboardConfig(
            title="FlashMM Public Demo",
            uid="flashmm-public",
            tags=["flashmm", "public", "demo"],
            folder="Public",
            public_access=True,
            description="Public demonstration dashboard showing FlashMM performance"
        )

        dashboard_json = {
            "dashboard": {
                "id": None,
                "uid": config.uid,
                "title": config.title,
                "tags": config.tags,
                "timezone": "UTC",
                "refresh": "10s",
                "time": {"from": "now-30m", "to": "now"},
                "panels": [
                    self._create_spread_improvement_panel(),
                    {
                        "id": 10,
                        "title": "Live Trading Metrics",
                        "type": "table",
                        "gridPos": {"h": 8, "w": 12, "x": 6, "y": 0},
                        "targets": [
                            {
                                "query": 'from(bucket: "metrics") |> range(start: -5m) |> filter(fn: (r) => r._measurement == "trading_performance_metrics") |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value") |> keep(columns: ["_time", "total_pnl_usdc", "total_volume_usdc", "fill_rate_percent"]) |> sort(columns: ["_time"], desc: true) |> limit(n: 10)',
                                "refId": "A"
                            }
                        ]
                    }
                ]
            },
            "folderId": 0,
            "overwrite": True
        }

        result = await self._create_or_update_dashboard(dashboard_json)

        # Enable public access if configured
        if config.public_access and result and "uid" in result:
            dashboard_uid = result.get("uid")
            if dashboard_uid:
                await self._enable_public_access(dashboard_uid)

        return result

    async def _create_or_update_dashboard(self, dashboard_json: dict[str, Any]) -> dict[str, Any]:
        """Create or update a dashboard."""
        if not self.session:
            raise ConnectionError("Grafana session not initialized")

        try:
            async with self.session.post(
                f"{self.base_url}/api/dashboards/db",
                json=dashboard_json
            ) as response:
                if response.status in [200, 201]:
                    result = await response.json()
                    logger.info(f"Created/updated dashboard: {dashboard_json['dashboard']['title']}")
                    return result
                else:
                    error_text = await response.text()
                    logger.error(f"Failed to create/update dashboard: {error_text}")
                    raise Exception(f"Dashboard creation failed: {error_text}")

        except Exception as e:
            logger.error(f"Error creating/updating dashboard: {e}")
            raise

    async def _enable_public_access(self, dashboard_uid: str) -> str | None:
        """Enable public access for a dashboard."""
        try:
            # Create public dashboard
            public_config = {
                "dashboardUid": dashboard_uid,
                "accessToken": "",
                "isEnabled": True,
                "annotationsEnabled": False,
                "timeSelectionEnabled": True
            }

            if not self.session:
                return None

            async with self.session.post(
                f"{self.base_url}/api/dashboards/public",
                json=public_config
            ) as response:
                if response.status in [200, 201]:
                    result = await response.json()
                    public_url = f"{self.base_url}/public-dashboards/{result.get('accessToken')}"
                    logger.info(f"Enabled public access for dashboard {dashboard_uid}: {public_url}")
                    return public_url
                else:
                    error_text = await response.text()
                    logger.warning(f"Failed to enable public access: {error_text}")
                    return None

        except Exception as e:
            logger.warning(f"Error enabling public access for dashboard {dashboard_uid}: {e}")
            return None

    async def setup_alert_rules(self) -> list[dict[str, Any]]:
        """Setup alert rules for FlashMM monitoring."""
        try:
            alert_rules = []

            # High latency alert
            latency_rule = AlertRule(
                title="High Trading Latency",
                uid="flashmm-high-latency",
                condition='avg(query(A, 5m, now)) > 350',
                evaluation_interval="1m",
                for_duration="2m",
                annotations={
                    "description": "Trading latency exceeded 350ms threshold",
                    "runbook_url": "https://github.com/flashmm/runbooks/latency"
                },
                labels={"severity": "warning", "service": "flashmm"}
            )

            # Spread degradation alert
            spread_rule = AlertRule(
                title="Spread Improvement Degraded",
                uid="flashmm-spread-degraded",
                condition='avg(query(A, 5m, now)) < 20',
                evaluation_interval="1m",
                for_duration="5m",
                annotations={
                    "description": "Spread improvement below 20% threshold",
                    "runbook_url": "https://github.com/flashmm/runbooks/spreads"
                },
                labels={"severity": "critical", "service": "flashmm"}
            )

            # System resource alert
            resource_rule = AlertRule(
                title="High System Resource Usage",
                uid="flashmm-high-resources",
                condition='avg(query(A, 5m, now)) > 85',
                evaluation_interval="1m",
                for_duration="3m",
                annotations={
                    "description": "System resource usage exceeded 85%",
                    "runbook_url": "https://github.com/flashmm/runbooks/resources"
                },
                labels={"severity": "warning", "service": "flashmm"}
            )

            # Create alert rules
            for rule in [latency_rule, spread_rule, resource_rule]:
                try:
                    result = await self._create_alert_rule(rule)
                    alert_rules.append(result)
                except Exception as e:
                    logger.error(f"Failed to create alert rule {rule.title}: {e}")

            logger.info(f"Setup {len(alert_rules)} alert rules successfully")
            return alert_rules

        except Exception as e:
            logger.error(f"Failed to setup alert rules: {e}")
            raise

    async def _create_alert_rule(self, rule: AlertRule) -> dict[str, Any]:
        """Create an alert rule."""
        try:
            rule_data = {
                "uid": rule.uid,
                "title": rule.title,
                "condition": rule.condition,
                "data": [
                    {
                        "refId": "A",
                        "queryType": "",
                        "relativeTimeRange": {
                            "from": 300,
                            "to": 0
                        },
                        "model": {
                            "expr": rule.condition,
                            "interval": "",
                            "refId": "A"
                        }
                    }
                ],
                "intervalSeconds": self._parse_interval(rule.evaluation_interval),
                "maxDataPoints": 43200,
                "noDataState": rule.no_data_state,
                "execErrState": rule.exec_err_state,
                "for": self._parse_duration(rule.for_duration),
                "annotations": rule.annotations or {},
                "labels": rule.labels or {}
            }

            if not self.session:
                raise ConnectionError("Grafana session not initialized")

            async with self.session.post(
                f"{self.base_url}/api/v1/provisioning/alert-rules",
                json=rule_data
            ) as response:
                if response.status in [200, 201]:
                    result = await response.json()
                    logger.info(f"Created alert rule: {rule.title}")
                    return result
                else:
                    error_text = await response.text()
                    raise Exception(f"Failed to create alert rule: {error_text}")

        except Exception as e:
            logger.error(f"Error creating alert rule {rule.title}: {e}")
            raise

    def _parse_interval(self, interval_str: str) -> int:
        """Parse interval string to seconds."""
        if interval_str.endswith('s'):
            return int(interval_str[:-1])
        elif interval_str.endswith('m'):
            return int(interval_str[:-1]) * 60
        elif interval_str.endswith('h'):
            return int(interval_str[:-1]) * 3600
        else:
            return 60  # Default to 1 minute

    def _parse_duration(self, duration_str: str) -> int:
        """Parse duration string to seconds."""
        return self._parse_interval(duration_str)

    async def get_dashboard_url(self, dashboard_uid: str, public: bool = False) -> str:
        """Get dashboard URL."""
        if public:
            # For public dashboards, we'd need to get the public access token
            return f"{self.base_url}/public-dashboards/{dashboard_uid}"
        else:
            return f"{self.base_url}/d/{dashboard_uid}"

    async def refresh_dashboard(self, dashboard_uid: str) -> bool:
        """Refresh a dashboard."""
        if not self.session:
            return False

        try:
            async with self.session.post(
                f"{self.base_url}/api/dashboards/uid/{dashboard_uid}/refresh"
            ) as response:
                if response.status == 200:
                    logger.info(f"Refreshed dashboard {dashboard_uid}")
                    return True
                else:
                    logger.warning(f"Failed to refresh dashboard {dashboard_uid}: {response.status}")
                    return False
        except Exception as e:
            logger.error(f"Error refreshing dashboard {dashboard_uid}: {e}")
            return False

    async def export_dashboard(self, dashboard_uid: str) -> dict[str, Any] | None:
        """Export dashboard JSON."""
        if not self.session:
            return None

        try:
            async with self.session.get(
                f"{self.base_url}/api/dashboards/uid/{dashboard_uid}"
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    logger.info(f"Exported dashboard {dashboard_uid}")
                    return result
                else:
                    logger.error(f"Failed to export dashboard {dashboard_uid}: {response.status}")
                    return None
        except Exception as e:
            logger.error(f"Error exporting dashboard {dashboard_uid}: {e}")
            return None

    async def list_dashboards(self, folder: str | None = None) -> list[dict[str, Any]]:
        """List all dashboards."""
        if not self.session:
            return []

        try:
            params = {}
            if folder:
                params["folderIds"] = folder

            async with self.session.get(
                f"{self.base_url}/api/search",
                params=params
            ) as response:
                if response.status == 200:
                    dashboards = await response.json()
                    logger.info(f"Listed {len(dashboards)} dashboards")
                    return dashboards
                else:
                    logger.error(f"Failed to list dashboards: {response.status}")
                    return []
        except Exception as e:
            logger.error(f"Error listing dashboards: {e}")
            return []

    async def cleanup(self) -> None:
        """Cleanup Grafana client resources."""
        try:
            if self.session:
                await self.session.close()
                logger.info("GrafanaClient session closed")
        except Exception as e:
            logger.error(f"Error during GrafanaClient cleanup: {e}")


# Global Grafana client instance
_grafana_client: GrafanaClient | None = None


async def get_grafana_client() -> GrafanaClient:
    """Get global Grafana client instance."""
    global _grafana_client
    if _grafana_client is None:
        _grafana_client = GrafanaClient()
        await _grafana_client.initialize()
    return _grafana_client


async def cleanup_grafana_client() -> None:
    """Cleanup global Grafana client."""
    global _grafana_client
    if _grafana_client:
        await _grafana_client.cleanup()
        _grafana_client = None

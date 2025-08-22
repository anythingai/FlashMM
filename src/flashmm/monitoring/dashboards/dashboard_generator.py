"""
FlashMM Dynamic Dashboard Generator

Generate dynamic Grafana dashboards based on system configuration, user roles,
and market-specific requirements with customizable layouts and permissions.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from flashmm.config.settings import get_config
from flashmm.monitoring.dashboards.grafana_client import DashboardConfig, GrafanaClient
from flashmm.utils.logging import get_logger

logger = get_logger(__name__)


class UserRole(Enum):
    """User roles for dashboard access."""
    ADMIN = "admin"
    TRADER = "trader"
    RISK_MANAGER = "risk_manager"
    VIEWER = "viewer"
    PUBLIC = "public"
    EXECUTIVE = "executive"


class DashboardTemplate(Enum):
    """Dashboard template types."""
    TRADING_OVERVIEW = "trading_overview"
    RISK_MONITORING = "risk_monitoring"
    SYSTEM_PERFORMANCE = "system_performance"
    ML_ANALYTICS = "ml_analytics"
    MARKET_SPECIFIC = "market_specific"
    EXECUTIVE_SUMMARY = "executive_summary"
    OPERATIONAL = "operational"


@dataclass
class PanelConfig:
    """Panel configuration for dashboard generation."""
    id: int
    title: str
    type: str
    query: str
    grid_pos: dict[str, int]
    field_config: dict[str, Any] = field(default_factory=dict)
    options: dict[str, Any] = field(default_factory=dict)
    targets: list[dict[str, Any]] = field(default_factory=list)
    thresholds: list[dict[str, Any]] = field(default_factory=list)
    alert_rules: list[str] = field(default_factory=list)


@dataclass
class DashboardLayout:
    """Dashboard layout configuration."""
    rows: int = 4
    columns: int = 24
    panel_height: int = 8
    panel_spacing: int = 1
    auto_arrange: bool = True


@dataclass
class RolePermissions:
    """Role-based permissions."""
    can_edit: bool = False
    can_share: bool = False
    can_export: bool = False
    can_create_alerts: bool = False
    visible_panels: list[str] = field(default_factory=list)
    hidden_panels: list[str] = field(default_factory=list)


class DashboardGenerator:
    """Dynamic dashboard generator with role-based customization."""

    def __init__(self, grafana_client: GrafanaClient):
        self.grafana_client = grafana_client
        self.config = get_config()

        # Role permissions mapping
        self.role_permissions = {
            UserRole.ADMIN: RolePermissions(
                can_edit=True,
                can_share=True,
                can_export=True,
                can_create_alerts=True,
                visible_panels=["*"]  # All panels visible
            ),
            UserRole.TRADER: RolePermissions(
                can_edit=False,
                can_share=True,
                can_export=True,
                can_create_alerts=False,
                visible_panels=[
                    "spread_improvement", "pnl", "volume", "fill_rate",
                    "inventory", "latency", "ml_predictions"
                ]
            ),
            UserRole.RISK_MANAGER: RolePermissions(
                can_edit=False,
                can_share=True,
                can_export=True,
                can_create_alerts=True,
                visible_panels=[
                    "risk_score", "var", "drawdown", "exposure", "compliance",
                    "position_limits", "inventory"
                ]
            ),
            UserRole.VIEWER: RolePermissions(
                can_edit=False,
                can_share=False,
                can_export=False,
                can_create_alerts=False,
                visible_panels=[
                    "spread_improvement", "volume", "system_health"
                ]
            ),
            UserRole.PUBLIC: RolePermissions(
                can_edit=False,
                can_share=False,
                can_export=False,
                can_create_alerts=False,
                visible_panels=[
                    "spread_improvement", "volume", "performance_summary"
                ]
            )
        }

        # Panel definitions
        self.panel_definitions = self._initialize_panel_definitions()

        logger.info("DashboardGenerator initialized")

    def _initialize_panel_definitions(self) -> dict[str, PanelConfig]:
        """Initialize panel definitions."""
        return {
            "spread_improvement": PanelConfig(
                id=1,
                title="Spread Improvement",
                type="stat",
                query='from(bucket: "metrics") |> range(start: -1h) |> filter(fn: (r) => r._measurement == "trading_performance_metrics") |> filter(fn: (r) => r._field == "spread_improvement_percent") |> last()',
                grid_pos={"h": 8, "w": 6, "x": 0, "y": 0},
                field_config={
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
                thresholds=[
                    {"field": "spread_improvement_percent", "value": 40, "condition": "lt", "severity": "critical"},
                    {"field": "spread_improvement_percent", "value": 20, "condition": "lt", "severity": "warning"}
                ]
            ),

            "pnl": PanelConfig(
                id=2,
                title="Profit & Loss",
                type="timeseries",
                query='from(bucket: "metrics") |> range(start: -1h) |> filter(fn: (r) => r._measurement == "trading_performance_metrics") |> filter(fn: (r) => r._field == "total_pnl_usdc")',
                grid_pos={"h": 8, "w": 12, "x": 6, "y": 0},
                field_config={
                    "defaults": {
                        "unit": "currencyUSD",
                        "custom": {
                            "drawStyle": "line",
                            "lineWidth": 2,
                            "fillOpacity": 10
                        }
                    }
                }
            ),

            "volume": PanelConfig(
                id=3,
                title="Trading Volume",
                type="stat",
                query='from(bucket: "metrics") |> range(start: -1h) |> filter(fn: (r) => r._measurement == "trading_performance_metrics") |> filter(fn: (r) => r._field == "total_volume_usdc") |> last()',
                grid_pos={"h": 8, "w": 6, "x": 18, "y": 0},
                field_config={
                    "defaults": {
                        "unit": "currencyUSD",
                        "decimals": 0
                    }
                }
            ),

            "fill_rate": PanelConfig(
                id=4,
                title="Fill Rate",
                type="gauge",
                query='from(bucket: "metrics") |> range(start: -1h) |> filter(fn: (r) => r._measurement == "trading_performance_metrics") |> filter(fn: (r) => r._field == "fill_rate_percent") |> last()',
                grid_pos={"h": 8, "w": 6, "x": 0, "y": 8},
                field_config={
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
            ),

            "inventory": PanelConfig(
                id=5,
                title="Inventory Utilization",
                type="bargauge",
                query='from(bucket: "metrics") |> range(start: -1h) |> filter(fn: (r) => r._measurement == "trading_performance_metrics") |> filter(fn: (r) => r._field == "inventory_utilization_percent") |> last()',
                grid_pos={"h": 8, "w": 6, "x": 6, "y": 8},
                field_config={
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
            ),

            "latency": PanelConfig(
                id=6,
                title="Order Latency",
                type="timeseries",
                query='from(bucket: "metrics") |> range(start: -1h) |> filter(fn: (r) => r._measurement == "trading_performance_metrics") |> filter(fn: (r) => r._field == "order_latency_ms")',
                grid_pos={"h": 8, "w": 12, "x": 12, "y": 8},
                field_config={
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
            ),

            "risk_score": PanelConfig(
                id=7,
                title="Risk Score",
                type="gauge",
                query='from(bucket: "metrics") |> range(start: -1h) |> filter(fn: (r) => r._measurement == "risk_metrics") |> filter(fn: (r) => r._field == "risk_score") |> last()',
                grid_pos={"h": 8, "w": 6, "x": 0, "y": 16},
                field_config={
                    "defaults": {
                        "unit": "short",
                        "min": 0,
                        "max": 100,
                        "thresholds": {
                            "steps": [
                                {"color": "green", "value": 0},
                                {"color": "yellow", "value": 50},
                                {"color": "red", "value": 80}
                            ]
                        }
                    }
                }
            ),

            "var": PanelConfig(
                id=8,
                title="Value at Risk (95%)",
                type="stat",
                query='from(bucket: "metrics") |> range(start: -1h) |> filter(fn: (r) => r._measurement == "risk_metrics") |> filter(fn: (r) => r._field == "var_1d_95_usdc") |> last()',
                grid_pos={"h": 8, "w": 6, "x": 6, "y": 16},
                field_config={
                    "defaults": {
                        "unit": "currencyUSD",
                        "decimals": 2
                    }
                }
            ),

            "ml_predictions": PanelConfig(
                id=9,
                title="ML Prediction Accuracy",
                type="stat",
                query='from(bucket: "metrics") |> range(start: -1h) |> filter(fn: (r) => r._measurement == "ml_performance_metrics") |> filter(fn: (r) => r._field == "prediction_accuracy_percent") |> last()',
                grid_pos={"h": 8, "w": 6, "x": 12, "y": 16},
                field_config={
                    "defaults": {
                        "unit": "percent",
                        "decimals": 1
                    }
                }
            ),

            "system_health": PanelConfig(
                id=10,
                title="System Health",
                type="table",
                query='from(bucket: "metrics") |> range(start: -5m) |> filter(fn: (r) => r._measurement == "component_health_metrics") |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value") |> sort(columns: ["_time"], desc: true) |> limit(n: 5)',
                grid_pos={"h": 8, "w": 6, "x": 18, "y": 16},
                field_config={
                    "defaults": {
                        "custom": {
                            "displayMode": "list",
                            "filterable": True
                        }
                    }
                }
            )
        }

    async def generate_dashboard(
        self,
        template: DashboardTemplate,
        role: UserRole,
        symbols: list[str] | None = None,
        custom_config: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Generate a dashboard based on template and user role."""
        try:
            # Get role permissions
            permissions = self.role_permissions.get(role, self.role_permissions[UserRole.VIEWER])

            # Create dashboard config
            dashboard_config = self._create_dashboard_config(template, role, symbols)

            # Select panels based on template and permissions
            selected_panels = self._select_panels(template, permissions)

            # Generate panels
            panels = []
            panel_id = 1

            for panel_name in selected_panels:
                if panel_name in self.panel_definitions:
                    panel_config = self.panel_definitions[panel_name]

                    # Customize panel for role and symbols
                    customized_panel = self._customize_panel(panel_config, role, symbols, panel_id)
                    panels.append(customized_panel)
                    panel_id += 1

            # Arrange panels in layout
            arranged_panels = self._arrange_panels(panels, template)

            # Create dashboard JSON
            dashboard_json = self._create_dashboard_json(
                dashboard_config,
                arranged_panels,
                permissions,
                custom_config
            )

            logger.info(f"Generated {template.value} dashboard for {role.value} with {len(panels)} panels")
            return dashboard_json

        except Exception as e:
            logger.error(f"Failed to generate dashboard: {e}")
            raise

    def _create_dashboard_config(
        self,
        template: DashboardTemplate,
        role: UserRole,
        symbols: list[str] | None
    ) -> DashboardConfig:
        """Create dashboard configuration."""
        symbol_suffix = f"-{'-'.join(symbols)}" if symbols else ""

        title_map = {
            DashboardTemplate.TRADING_OVERVIEW: f"Trading Overview - {role.value.title()}",
            DashboardTemplate.RISK_MONITORING: f"Risk Monitoring - {role.value.title()}",
            DashboardTemplate.SYSTEM_PERFORMANCE: f"System Performance - {role.value.title()}",
            DashboardTemplate.ML_ANALYTICS: f"ML Analytics - {role.value.title()}",
            DashboardTemplate.MARKET_SPECIFIC: f"Market Analysis{symbol_suffix} - {role.value.title()}",
            DashboardTemplate.EXECUTIVE_SUMMARY: f"Executive Summary - {role.value.title()}",
            DashboardTemplate.OPERATIONAL: f"Operational Dashboard - {role.value.title()}"
        }

        return DashboardConfig(
            title=title_map.get(template, f"FlashMM Dashboard - {role.value.title()}"),
            uid=f"flashmm-{template.value}-{role.value}{symbol_suffix}",
            tags=["flashmm", template.value, role.value] + (symbols or []),
            folder=f"FlashMM-{role.value.title()}",
            refresh_interval="5s" if role in [UserRole.TRADER, UserRole.ADMIN] else "30s",
            time_range="1h" if role != UserRole.EXECUTIVE else "24h",
            public_access=(role == UserRole.PUBLIC),
            description=f"FlashMM {template.value} dashboard customized for {role.value}"
        )

    def _select_panels(self, template: DashboardTemplate, permissions: RolePermissions) -> list[str]:
        """Select panels based on template and permissions."""
        template_panels = {
            DashboardTemplate.TRADING_OVERVIEW: [
                "spread_improvement", "pnl", "volume", "fill_rate", "inventory", "latency"
            ],
            DashboardTemplate.RISK_MONITORING: [
                "risk_score", "var", "inventory", "pnl", "system_health"
            ],
            DashboardTemplate.SYSTEM_PERFORMANCE: [
                "latency", "system_health", "volume"
            ],
            DashboardTemplate.ML_ANALYTICS: [
                "ml_predictions", "latency", "pnl"
            ],
            DashboardTemplate.MARKET_SPECIFIC: [
                "spread_improvement", "volume", "pnl", "fill_rate"
            ],
            DashboardTemplate.EXECUTIVE_SUMMARY: [
                "spread_improvement", "pnl", "volume", "risk_score"
            ],
            DashboardTemplate.OPERATIONAL: [
                "system_health", "latency", "inventory", "fill_rate"
            ]
        }

        base_panels = template_panels.get(template, [])

        # Filter panels based on permissions
        if "*" in permissions.visible_panels:
            return base_panels

        filtered_panels = [
            panel for panel in base_panels
            if panel in permissions.visible_panels and panel not in permissions.hidden_panels
        ]

        return filtered_panels

    def _customize_panel(
        self,
        panel_config: PanelConfig,
        role: UserRole,
        symbols: list[str] | None,
        panel_id: int
    ) -> dict[str, Any]:
        """Customize panel for specific role and symbols."""
        panel = {
            "id": panel_id,
            "title": panel_config.title,
            "type": panel_config.type,
            "gridPos": panel_config.grid_pos.copy(),
            "fieldConfig": panel_config.field_config.copy(),
            "options": panel_config.options.copy(),
            "targets": [
                {
                    "query": self._customize_query(panel_config.query, symbols),
                    "refId": "A"
                }
            ]
        }

        # Role-specific customizations
        if role == UserRole.PUBLIC:
            # Simplify panels for public view
            panel["fieldConfig"]["defaults"] = panel["fieldConfig"].get("defaults", {})
            panel["fieldConfig"]["defaults"]["custom"] = {"hideFrom": {"tooltip": True}}

        elif role == UserRole.RISK_MANAGER:
            # Add risk-specific annotations
            if panel_config.title in ["Inventory Utilization", "Risk Score"]:
                panel["alert"] = {
                    "conditions": panel_config.alert_rules,
                    "executionErrorState": "alerting",
                    "noDataState": "no_data",
                    "frequency": "10s"
                }

        return panel

    def _customize_query(self, base_query: str, symbols: list[str] | None) -> str:
        """Customize query with symbol filtering."""
        if not symbols:
            return base_query

        # Add symbol filtering to query
        symbol_filter = ' |> filter(fn: (r) => ' + ' or '.join([f'r.symbol == "{symbol}"' for symbol in symbols]) + ')'

        if '|> filter(' in base_query:
            # Insert after existing filters
            parts = base_query.split('|> filter(')
            if len(parts) > 1:
                return parts[0] + '|> filter(' + parts[1] + symbol_filter

        return base_query + symbol_filter

    def _arrange_panels(self, panels: list[dict[str, Any]], template: DashboardTemplate) -> list[dict[str, Any]]:
        """Arrange panels in optimal layout."""
        layout = DashboardLayout()

        if template == DashboardTemplate.EXECUTIVE_SUMMARY:
            # Executive layout: larger panels, less dense
            layout = DashboardLayout(rows=2, columns=24, panel_height=12)
        elif template == DashboardTemplate.OPERATIONAL:
            # Operational layout: compact, more panels
            layout = DashboardLayout(rows=6, columns=24, panel_height=6)

        arranged_panels = []
        current_x = 0
        current_y = 0
        panels_per_row = layout.columns // (layout.columns // min(len(panels), 4))

        for _i, panel in enumerate(panels):
            if layout.auto_arrange:
                # Auto-arrange panels
                panel_width = layout.columns // panels_per_row

                panel["gridPos"] = {
                    "h": layout.panel_height,
                    "w": panel_width,
                    "x": current_x,
                    "y": current_y
                }

                current_x += panel_width
                if current_x >= layout.columns:
                    current_x = 0
                    current_y += layout.panel_height

            arranged_panels.append(panel)

        return arranged_panels

    def _create_dashboard_json(
        self,
        config: DashboardConfig,
        panels: list[dict[str, Any]],
        permissions: RolePermissions,
        custom_config: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Create complete dashboard JSON."""
        dashboard_json = {
            "dashboard": {
                "id": None,
                "uid": config.uid,
                "title": config.title,
                "tags": config.tags,
                "timezone": "UTC",
                "refresh": config.refresh_interval,
                "time": {
                    "from": f"now-{config.time_range}",
                    "to": "now"
                },
                "timepicker": {
                    "refresh_intervals": ["5s", "10s", "30s", "1m", "5m", "15m", "30m", "1h"],
                    "time_options": ["5m", "15m", "1h", "6h", "12h", "24h", "2d", "7d", "30d"]
                },
                "panels": panels,
                "templating": {
                    "list": self._create_template_variables()
                },
                "annotations": {
                    "list": self._create_annotations(permissions)
                },
                "editable": permissions.can_edit,
                "hideControls": not permissions.can_edit,
                "graphTooltip": 1,
                "links": self._create_dashboard_links(config),
                "style": "dark",
                "version": 1
            },
            "folderId": 0,
            "overwrite": True,
            "message": f"Generated dashboard for {config.title}"
        }

        # Apply custom configuration
        if custom_config:
            dashboard_json["dashboard"].update(custom_config)

        return dashboard_json

    def _create_template_variables(self) -> list[dict[str, Any]]:
        """Create template variables for dashboard."""
        return [
            {
                "name": "symbol",
                "type": "custom",
                "label": "Trading Pair",
                "options": [
                    {"text": "All", "value": "*", "selected": True},
                    {"text": "SEI/USDC", "value": "SEI/USDC", "selected": False},
                    {"text": "BTC/USDC", "value": "BTC/USDC", "selected": False},
                    {"text": "ETH/USDC", "value": "ETH/USDC", "selected": False}
                ],
                "current": {"text": "All", "value": "*"}
            },
            {
                "name": "timerange",
                "type": "interval",
                "label": "Time Range",
                "options": [
                    {"text": "5m", "value": "5m", "selected": False},
                    {"text": "1h", "value": "1h", "selected": True},
                    {"text": "6h", "value": "6h", "selected": False},
                    {"text": "24h", "value": "24h", "selected": False}
                ]
            }
        ]

    def _create_annotations(self, permissions: RolePermissions) -> list[dict[str, Any]]:
        """Create annotations for dashboard."""
        annotations = []

        if permissions.can_create_alerts:
            annotations.append({
                "name": "Alerts",
                "datasource": "-- Grafana --",
                "enable": True,
                "hide": False,
                "iconColor": "red",
                "type": "dashboard"
            })

        annotations.append({
            "name": "Trading Events",
            "datasource": "FlashMM-InfluxDB",
            "enable": True,
            "hide": False,
            "iconColor": "blue",
            "query": 'from(bucket: "events") |> range(start: -1h) |> filter(fn: (r) => r._measurement == "trading_events")'
        })

        return annotations

    def _create_dashboard_links(self, config: DashboardConfig) -> list[dict[str, Any]]:
        """Create dashboard navigation links."""
        return [
            {
                "title": "FlashMM Home",
                "type": "link",
                "url": "/d/flashmm-overview",
                "icon": "external link"
            },
            {
                "title": "Documentation",
                "type": "link",
                "url": "https://github.com/flashmm/docs",
                "icon": "doc",
                "targetBlank": True
            }
        ]

    async def generate_market_specific_dashboard(
        self,
        symbol: str,
        role: UserRole = UserRole.TRADER
    ) -> dict[str, Any]:
        """Generate market-specific dashboard."""
        return await self.generate_dashboard(
            template=DashboardTemplate.MARKET_SPECIFIC,
            role=role,
            symbols=[symbol],
            custom_config={
                "title": f"FlashMM {symbol} Trading Dashboard",
                "tags": ["flashmm", "market", symbol.lower().replace("/", "-")],
                "refresh": "5s"
            }
        )

    async def generate_role_dashboard_suite(self, role: UserRole) -> list[dict[str, Any]]:
        """Generate complete dashboard suite for a role."""
        dashboards = []

        # Template mapping by role
        role_templates = {
            UserRole.ADMIN: [
                DashboardTemplate.TRADING_OVERVIEW,
                DashboardTemplate.RISK_MONITORING,
                DashboardTemplate.SYSTEM_PERFORMANCE,
                DashboardTemplate.ML_ANALYTICS,
                DashboardTemplate.OPERATIONAL
            ],
            UserRole.TRADER: [
                DashboardTemplate.TRADING_OVERVIEW,
                DashboardTemplate.ML_ANALYTICS
            ],
            UserRole.RISK_MANAGER: [
                DashboardTemplate.RISK_MONITORING,
                DashboardTemplate.TRADING_OVERVIEW
            ],
            UserRole.VIEWER: [
                DashboardTemplate.TRADING_OVERVIEW
            ],
            UserRole.PUBLIC: [
                DashboardTemplate.EXECUTIVE_SUMMARY
            ]
        }

        templates = role_templates.get(role, [DashboardTemplate.TRADING_OVERVIEW])

        for template in templates:
            try:
                dashboard = await self.generate_dashboard(template, role)
                dashboards.append(dashboard)
            except Exception as e:
                logger.error(f"Failed to generate {template.value} dashboard for {role.value}: {e}")

        logger.info(f"Generated {len(dashboards)} dashboards for role {role.value}")
        return dashboards

    async def export_dashboard_template(self, dashboard_json: dict[str, Any]) -> str:
        """Export dashboard as reusable template."""
        template = {
            "template_version": "1.0",
            "created_at": datetime.now().isoformat(),
            "dashboard": dashboard_json,
            "variables": {
                "datasource": "${DS_FLASHMM_INFLUXDB}",
                "refresh": "${refresh}",
                "time_range": "${time_range}"
            }
        }

        return json.dumps(template, indent=2)


# Global dashboard generator instance
_dashboard_generator: DashboardGenerator | None = None


async def get_dashboard_generator(grafana_client: GrafanaClient | None = None) -> DashboardGenerator:
    """Get global dashboard generator instance."""
    global _dashboard_generator
    if _dashboard_generator is None:
        if not grafana_client:
            from flashmm.monitoring.dashboards.grafana_client import get_grafana_client
            grafana_client = await get_grafana_client()
        _dashboard_generator = DashboardGenerator(grafana_client)
    return _dashboard_generator

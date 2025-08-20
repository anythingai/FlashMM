"""
FlashMM Multi-Channel Alert Management System

Comprehensive alerting system with configurable thresholds, multi-channel notifications,
alert correlation, escalation workflows, and performance analysis.
"""

import asyncio
import aiohttp
import smtplib
import json
from typing import Dict, Any, List, Optional, Set, Callable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
from email.mime.base import MimeBase
from email import encoders
import ssl

from flashmm.config.settings import get_config
from flashmm.utils.logging import get_logger
from flashmm.utils.exceptions import ValidationError

logger = get_logger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertStatus(Enum):
    """Alert status states."""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"
    ESCALATED = "escalated"


class AlertChannel(Enum):
    """Alert notification channels."""
    EMAIL = "email"
    SLACK = "slack"
    DISCORD = "discord"
    WEBHOOK = "webhook"
    SMS = "sms"
    DASHBOARD = "dashboard"
    TWITTER = "twitter"
    PAGERDUTY = "pagerduty"
    TELEGRAM = "telegram"


@dataclass
class AlertRule:
    """Alert rule configuration."""
    id: str
    name: str
    description: str
    metric: str
    condition: str  # e.g., "gt", "lt", "eq", "contains"
    threshold: Union[float, str]
    severity: AlertSeverity
    channels: List[AlertChannel]
    
    # Timing configuration
    evaluation_interval: int = 60  # seconds
    for_duration: int = 120  # seconds before firing
    cooldown_period: int = 900  # seconds before re-firing
    
    # Escalation configuration
    escalation_rules: List[Dict[str, Any]] = field(default_factory=list)
    
    # Suppression configuration
    suppression_rules: List[Dict[str, Any]] = field(default_factory=list)
    
    # Additional metadata
    tags: Dict[str, str] = field(default_factory=dict)
    runbook_url: str = ""
    owner: str = ""
    enabled: bool = True


@dataclass
class Alert:
    """Alert instance."""
    id: str
    rule_id: str
    name: str
    description: str
    severity: AlertSeverity
    status: AlertStatus
    
    # Alert data
    metric: str
    current_value: Union[float, str]
    threshold: Union[float, str]
    
    # Timing
    triggered_at: datetime
    last_updated: datetime
    resolved_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    
    # Metadata
    tags: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    
    # Tracking
    notification_count: int = 0
    escalation_level: int = 0
    suppressed_until: Optional[datetime] = None
    acknowledged_by: str = ""


@dataclass
class NotificationChannel:
    """Notification channel configuration."""
    type: AlertChannel
    name: str
    config: Dict[str, Any]
    enabled: bool = True
    rate_limit: int = 10  # max notifications per hour
    severity_filter: List[AlertSeverity] = field(default_factory=list)


@dataclass
class AlertCorrelation:
    """Alert correlation configuration."""
    name: str
    alerts: List[str]  # Alert rule IDs
    correlation_window: int = 300  # seconds
    threshold: int = 2  # minimum alerts to correlate
    action: str = "suppress"  # "suppress", "escalate", "merge"


class AlertManager:
    """Multi-channel alert management system."""
    
    def __init__(self):
        self.config = get_config()
        
        # Alert storage
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        
        # Notification channels
        self.channels: Dict[str, NotificationChannel] = {}
        
        # Correlation and suppression
        self.correlations: List[AlertCorrelation] = []
        self.suppression_rules: List[Dict[str, Any]] = []
        self.maintenance_windows: List[Dict[str, Any]] = []
        
        # Callback handlers
        self.alert_handlers: List[Callable[[Alert], None]] = []
        self.resolution_handlers: List[Callable[[Alert], None]] = []
        
        # Background tasks
        self.evaluation_task: Optional[asyncio.Task] = None
        self.notification_task: Optional[asyncio.Task] = None
        self.cleanup_task: Optional[asyncio.Task] = None
        
        # Statistics
        self.stats = {
            "total_alerts": 0,
            "alerts_by_severity": {s.value: 0 for s in AlertSeverity},
            "alerts_by_status": {s.value: 0 for s in AlertStatus},
            "notifications_sent": 0,
            "escalations": 0,
            "false_positives": 0,
            "avg_resolution_time": 0.0,
            "uptime": datetime.now()
        }
        
        self.running = False
        
        logger.info("AlertManager initialized")
    
    async def initialize(self) -> None:
        """Initialize alert manager."""
        try:
            # Load default alert rules
            await self._load_default_rules()
            
            # Setup notification channels
            await self._setup_notification_channels()
            
            # Load correlations and suppressions
            await self._load_correlations()
            
            # Start background tasks
            await self._start_background_tasks()
            
            self.running = True
            logger.info("AlertManager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize AlertManager: {e}")
            raise
    
    async def _load_default_rules(self) -> None:
        """Load default alert rules for FlashMM."""
        default_rules = [
            AlertRule(
                id="high_latency",
                name="High Trading Latency",
                description="Trading latency exceeded acceptable threshold",
                metric="trading.order_latency_ms",
                condition="gt",
                threshold=350.0,
                severity=AlertSeverity.WARNING,
                channels=[AlertChannel.EMAIL, AlertChannel.SLACK, AlertChannel.DASHBOARD],
                evaluation_interval=30,
                for_duration=120,
                tags={"component": "trading", "type": "performance"},
                runbook_url="https://github.com/flashmm/runbooks/latency"
            ),
            
            AlertRule(
                id="spread_degradation",
                name="Spread Improvement Degraded",
                description="Spread improvement below target threshold",
                metric="trading.spread_improvement_percent",
                condition="lt",
                threshold=20.0,
                severity=AlertSeverity.CRITICAL,
                channels=[AlertChannel.EMAIL, AlertChannel.SLACK, AlertChannel.PAGERDUTY],
                evaluation_interval=60,
                for_duration=300,
                escalation_rules=[
                    {"after_minutes": 15, "severity": "emergency", "channels": ["pagerduty", "sms"]},
                    {"after_minutes": 30, "severity": "emergency", "channels": ["sms"], "notify": ["admin"]}
                ],
                tags={"component": "trading", "type": "business_critical"},
                runbook_url="https://github.com/flashmm/runbooks/spreads"
            ),
            
            AlertRule(
                id="high_error_rate",
                name="High System Error Rate",
                description="System error rate exceeded threshold",
                metric="system.error_rate_percent",
                condition="gt",
                threshold=5.0,
                severity=AlertSeverity.CRITICAL,
                channels=[AlertChannel.EMAIL, AlertChannel.SLACK, AlertChannel.DASHBOARD],
                evaluation_interval=60,
                for_duration=180,
                tags={"component": "system", "type": "reliability"}
            ),
            
            AlertRule(
                id="inventory_critical",
                name="Critical Inventory Level",
                description="Inventory utilization in critical range",
                metric="trading.inventory_utilization_percent",
                condition="gt",
                threshold=90.0,
                severity=AlertSeverity.EMERGENCY,
                channels=[AlertChannel.EMAIL, AlertChannel.SLACK, AlertChannel.SMS, AlertChannel.PAGERDUTY],
                evaluation_interval=30,
                for_duration=60,
                tags={"component": "risk", "type": "business_critical"}
            ),
            
            AlertRule(
                id="ml_prediction_failure",
                name="ML Prediction Service Failure",
                description="ML prediction service experiencing high failure rate",
                metric="ml.api_success_rate_percent",
                condition="lt",
                threshold=90.0,
                severity=AlertSeverity.WARNING,
                channels=[AlertChannel.EMAIL, AlertChannel.SLACK],
                evaluation_interval=120,
                for_duration=300,
                tags={"component": "ml", "type": "service"}
            ),
            
            AlertRule(
                id="high_resource_usage",
                name="High System Resource Usage",
                description="System resources exceeded safe thresholds",
                metric="system.cpu_percent",
                condition="gt",
                threshold=85.0,
                severity=AlertSeverity.WARNING,
                channels=[AlertChannel.EMAIL, AlertChannel.DASHBOARD],
                evaluation_interval=60,
                for_duration=300,
                tags={"component": "system", "type": "resource"}
            ),
            
            AlertRule(
                id="pnl_significant_loss",
                name="Significant P&L Loss",
                description="Significant loss detected in trading P&L",
                metric="trading.total_pnl_usdc",
                condition="lt",
                threshold=-1000.0,
                severity=AlertSeverity.CRITICAL,
                channels=[AlertChannel.EMAIL, AlertChannel.SLACK, AlertChannel.SMS],
                evaluation_interval=60,
                for_duration=60,
                escalation_rules=[
                    {"after_minutes": 5, "severity": "emergency", "channels": ["pagerduty", "sms"]}
                ],
                tags={"component": "trading", "type": "financial"}
            )
        ]
        
        for rule in default_rules:
            self.alert_rules[rule.id] = rule
        
        logger.info(f"Loaded {len(default_rules)} default alert rules")
    
    async def _setup_notification_channels(self) -> None:
        """Setup notification channels."""
        # Email channel
        if self.config.get("alerts.email.enabled", False):
            self.channels["email"] = NotificationChannel(
                type=AlertChannel.EMAIL,
                name="Email Notifications",
                config={
                    "smtp_server": self.config.get("alerts.email.smtp_server", "smtp.gmail.com"),
                    "smtp_port": self.config.get("alerts.email.smtp_port", 587),
                    "username": self.config.get("alerts.email.username", ""),
                    "password": self.config.get("alerts.email.password", ""),
                    "from_email": self.config.get("alerts.email.from", ""),
                    "to_emails": self.config.get("alerts.email.recipients", [])
                }
            )
        
        # Slack channel
        if self.config.get("alerts.slack.enabled", False):
            self.channels["slack"] = NotificationChannel(
                type=AlertChannel.SLACK,
                name="Slack Notifications",
                config={
                    "webhook_url": self.config.get("alerts.slack.webhook_url", ""),
                    "channel": self.config.get("alerts.slack.channel", "#alerts"),
                    "username": self.config.get("alerts.slack.username", "FlashMM-Alerts")
                }
            )
        
        # Discord channel
        if self.config.get("alerts.discord.enabled", False):
            self.channels["discord"] = NotificationChannel(
                type=AlertChannel.DISCORD,
                name="Discord Notifications",
                config={
                    "webhook_url": self.config.get("alerts.discord.webhook_url", ""),
                    "username": self.config.get("alerts.discord.username", "FlashMM")
                }
            )
        
        # Webhook channel
        if self.config.get("alerts.webhook.enabled", False):
            self.channels["webhook"] = NotificationChannel(
                type=AlertChannel.WEBHOOK,
                name="Generic Webhook",
                config={
                    "url": self.config.get("alerts.webhook.url", ""),
                    "headers": self.config.get("alerts.webhook.headers", {}),
                    "method": self.config.get("alerts.webhook.method", "POST")
                }
            )
        
        # Dashboard channel (always enabled)
        self.channels["dashboard"] = NotificationChannel(
            type=AlertChannel.DASHBOARD,
            name="Dashboard Notifications",
            config={}
        )
        
        logger.info(f"Setup {len(self.channels)} notification channels")
    
    async def _load_correlations(self) -> None:
        """Load alert correlations."""
        self.correlations = [
            AlertCorrelation(
                name="Trading System Issues",
                alerts=["high_latency", "high_error_rate", "spread_degradation"],
                correlation_window=600,
                threshold=2,
                action="escalate"
            ),
            AlertCorrelation(
                name="Resource Exhaustion",
                alerts=["high_resource_usage", "high_latency", "high_error_rate"],
                correlation_window=300,
                threshold=2,
                action="suppress"
            )
        ]
    
    async def _start_background_tasks(self) -> None:
        """Start background tasks."""
        self.evaluation_task = asyncio.create_task(self._evaluation_loop())
        self.notification_task = asyncio.create_task(self._notification_loop())
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
    
    async def _evaluation_loop(self) -> None:
        """Main alert evaluation loop."""
        while self.running:
            try:
                await self._evaluate_alerts()
                await asyncio.sleep(30)  # Evaluate every 30 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in alert evaluation loop: {e}")
                await asyncio.sleep(60)
    
    async def _notification_loop(self) -> None:
        """Notification processing loop."""
        while self.running:
            try:
                await self._process_notifications()
                await asyncio.sleep(10)  # Process notifications every 10 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in notification loop: {e}")
                await asyncio.sleep(30)
    
    async def _cleanup_loop(self) -> None:
        """Cleanup old alerts and maintain statistics."""
        while self.running:
            try:
                await self._cleanup_old_alerts()
                await self._update_statistics()
                await asyncio.sleep(3600)  # Run every hour
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(1800)
    
    async def evaluate_metric(self, metric_name: str, current_value: Union[float, str]) -> List[Alert]:
        """Evaluate a metric against all applicable rules."""
        triggered_alerts = []
        
        for rule in self.alert_rules.values():
            if not rule.enabled or rule.metric != metric_name:
                continue
            
            try:
                if await self._evaluate_rule(rule, current_value):
                    alert = await self._create_alert(rule, current_value)
                    if alert:
                        triggered_alerts.append(alert)
            except Exception as e:
                logger.error(f"Error evaluating rule {rule.id}: {e}")
        
        return triggered_alerts
    
    async def _evaluate_rule(self, rule: AlertRule, current_value: Union[float, str]) -> bool:
        """Evaluate if a rule should trigger."""
        try:
            # Check if alert is already active
            existing_alert = None
            for alert in self.active_alerts.values():
                if alert.rule_id == rule.id and alert.status == AlertStatus.ACTIVE:
                    existing_alert = alert
                    break
            
            # Check cooldown period
            if existing_alert and existing_alert.last_updated:
                cooldown_end = existing_alert.last_updated + timedelta(seconds=rule.cooldown_period)
                if datetime.now() < cooldown_end:
                    return False
            
            # Evaluate condition
            should_trigger = self._evaluate_condition(rule.condition, current_value, rule.threshold)
            
            if should_trigger and not existing_alert:
                # New alert - check if it should trigger after for_duration
                return await self._check_for_duration(rule, current_value)
            elif not should_trigger and existing_alert:
                # Resolve existing alert
                await self._resolve_alert(existing_alert.id, "Condition no longer met")
            
            return False
            
        except Exception as e:
            logger.error(f"Error evaluating rule {rule.id}: {e}")
            return False
    
    def _evaluate_condition(self, condition: str, current_value: Union[float, str], threshold: Union[float, str]) -> bool:
        """Evaluate alert condition."""
        try:
            if condition == "gt":
                return float(current_value) > float(threshold)
            elif condition == "lt":
                return float(current_value) < float(threshold)
            elif condition == "eq":
                return current_value == threshold
            elif condition == "ne":
                return current_value != threshold
            elif condition == "gte":
                return float(current_value) >= float(threshold)
            elif condition == "lte":
                return float(current_value) <= float(threshold)
            elif condition == "contains":
                return str(threshold) in str(current_value)
            else:
                logger.warning(f"Unknown condition: {condition}")
                return False
        except (ValueError, TypeError) as e:
            logger.error(f"Error evaluating condition {condition}: {e}")
            return False
    
    async def _check_for_duration(self, rule: AlertRule, current_value: Union[float, str]) -> bool:
        """Check if condition has been true for the required duration."""
        # This is a simplified implementation
        # In a full implementation, you'd track condition states over time
        return True
    
    async def _create_alert(self, rule: AlertRule, current_value: Union[float, str]) -> Optional[Alert]:
        """Create a new alert."""
        try:
            alert_id = f"{rule.id}_{int(datetime.now().timestamp())}"
            
            alert = Alert(
                id=alert_id,
                rule_id=rule.id,
                name=rule.name,
                description=rule.description,
                severity=rule.severity,
                status=AlertStatus.ACTIVE,
                metric=rule.metric,
                current_value=current_value,
                threshold=rule.threshold,
                triggered_at=datetime.now(),
                last_updated=datetime.now(),
                tags=rule.tags.copy(),
                annotations={
                    "runbook_url": rule.runbook_url,
                    "owner": rule.owner
                }
            )
            
            # Store alert
            self.active_alerts[alert_id] = alert
            self.alert_history.append(alert)
            
            # Update statistics
            self.stats["total_alerts"] += 1
            self.stats["alerts_by_severity"][alert.severity.value] += 1
            self.stats["alerts_by_status"][alert.status.value] += 1
            
            # Trigger callbacks
            for handler in self.alert_handlers:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(alert)
                    else:
                        handler(alert)
                except Exception as e:
                    logger.error(f"Error in alert handler: {e}")
            
            logger.warning(f"ALERT TRIGGERED: {alert.name} - {alert.description}")
            return alert
            
        except Exception as e:
            logger.error(f"Error creating alert for rule {rule.id}: {e}")
            return None
    
    async def _resolve_alert(self, alert_id: str, reason: str = "") -> bool:
        """Resolve an active alert."""
        try:
            alert = self.active_alerts.get(alert_id)
            if not alert:
                return False
            
            alert.status = AlertStatus.RESOLVED
            alert.resolved_at = datetime.now()
            alert.last_updated = datetime.now()
            alert.annotations["resolution_reason"] = reason
            
            # Remove from active alerts
            del self.active_alerts[alert_id]
            
            # Update statistics
            self.stats["alerts_by_status"][AlertStatus.ACTIVE.value] -= 1
            self.stats["alerts_by_status"][AlertStatus.RESOLVED.value] += 1
            
            # Calculate resolution time
            resolution_time = (alert.resolved_at - alert.triggered_at).total_seconds()
            self._update_avg_resolution_time(resolution_time)
            
            # Trigger resolution callbacks
            for handler in self.resolution_handlers:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(alert)
                    else:
                        handler(alert)
                except Exception as e:
                    logger.error(f"Error in resolution handler: {e}")
            
            logger.info(f"ALERT RESOLVED: {alert.name} - {reason}")
            return True
            
        except Exception as e:
            logger.error(f"Error resolving alert {alert_id}: {e}")
            return False
    
    async def acknowledge_alert(self, alert_id: str, acknowledged_by: str, comment: str = "") -> bool:
        """Acknowledge an alert."""
        try:
            alert = self.active_alerts.get(alert_id)
            if not alert:
                return False
            
            alert.status = AlertStatus.ACKNOWLEDGED
            alert.acknowledged_at = datetime.now()
            alert.acknowledged_by = acknowledged_by
            alert.last_updated = datetime.now()
            
            if comment:
                alert.annotations["acknowledgment_comment"] = comment
            
            # Update statistics
            self.stats["alerts_by_status"][AlertStatus.ACTIVE.value] -= 1
            self.stats["alerts_by_status"][AlertStatus.ACKNOWLEDGED.value] += 1
            
            logger.info(f"ALERT ACKNOWLEDGED: {alert.name} by {acknowledged_by}")
            return True
            
        except Exception as e:
            logger.error(f"Error acknowledging alert {alert_id}: {e}")
            return False
    
    async def _evaluate_alerts(self) -> None:
        """Evaluate all alert rules (placeholder for metric integration)."""
        # This would integrate with the metrics collector
        # For now, this is a placeholder that would be called by the metrics collector
        pass
    
    async def _process_notifications(self) -> None:
        """Process pending notifications."""
        try:
            for alert in self.active_alerts.values():
                if alert.status == AlertStatus.ACTIVE:
                    await self._send_notifications(alert)
                    
                    # Check for escalation
                    await self._check_escalation(alert)
                    
                    # Check for correlation
                    await self._check_correlations(alert)
        
        except Exception as e:
            logger.error(f"Error processing notifications: {e}")
    
    async def _send_notifications(self, alert: Alert) -> None:
        """Send notifications for an alert."""
        try:
            rule = self.alert_rules.get(alert.rule_id)
            if not rule:
                return
            
            for channel_type in rule.channels:
                channel_name = channel_type.value
                channel = self.channels.get(channel_name)
                
                if channel and channel.enabled:
                    try:
                        success = await self._send_notification(channel, alert)
                        if success:
                            alert.notification_count += 1
                            self.stats["notifications_sent"] += 1
                    except Exception as e:
                        logger.error(f"Failed to send notification via {channel_name}: {e}")
        
        except Exception as e:
            logger.error(f"Error sending notifications for alert {alert.id}: {e}")
    
    async def _send_notification(self, channel: NotificationChannel, alert: Alert) -> bool:
        """Send notification via specific channel."""
        try:
            if channel.type == AlertChannel.EMAIL:
                return await self._send_email_notification(channel, alert)
            elif channel.type == AlertChannel.SLACK:
                return await self._send_slack_notification(channel, alert)
            elif channel.type == AlertChannel.DISCORD:
                return await self._send_discord_notification(channel, alert)
            elif channel.type == AlertChannel.WEBHOOK:
                return await self._send_webhook_notification(channel, alert)
            elif channel.type == AlertChannel.DASHBOARD:
                return await self._send_dashboard_notification(channel, alert)
            else:
                logger.warning(f"Unsupported notification channel: {channel.type}")
                return False
        
        except Exception as e:
            logger.error(f"Error sending {channel.type.value} notification: {e}")
            return False
    
    async def _send_email_notification(self, channel: NotificationChannel, alert: Alert) -> bool:
        """Send email notification."""
        try:
            config = channel.config
            
            msg = MimeMultipart()
            msg['From'] = config['from_email']
            msg['To'] = ', '.join(config['to_emails'])
            msg['Subject'] = f"[{alert.severity.value.upper()}] FlashMM Alert: {alert.name}"
            
            body = f"""
Alert: {alert.name}
Severity: {alert.severity.value.upper()}
Description: {alert.description}
Metric: {alert.metric}
Current Value: {alert.current_value}
Threshold: {alert.threshold}
Triggered At: {alert.triggered_at.strftime('%Y-%m-%d %H:%M:%S UTC')}

Runbook: {alert.annotations.get('runbook_url', 'N/A')}
Tags: {', '.join([f'{k}={v}' for k, v in alert.tags.items()])}
            """
            
            msg.attach(MimeText(body, 'plain'))
            
            # Send email
            server = smtplib.SMTP(config['smtp_server'], config['smtp_port'])
            server.starttls()
            server.login(config['username'], config['password'])
            text = msg.as_string()
            server.sendmail(config['from_email'], config['to_emails'], text)
            server.quit()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")
            return False
    
    async def _send_slack_notification(self, channel: NotificationChannel, alert: Alert) -> bool:
        """Send Slack notification."""
        try:
            config = channel.config
            
            color_map = {
                AlertSeverity.INFO: "good",
                AlertSeverity.WARNING: "warning",
                AlertSeverity.CRITICAL: "danger",
                AlertSeverity.EMERGENCY: "danger"
            }
            
            payload = {
                "channel": config['channel'],
                "username": config['username'],
                "icon_emoji": ":warning:",
                "attachments": [
                    {
                        "color": color_map.get(alert.severity, "warning"),
                        "title": f"{alert.severity.value.upper()}: {alert.name}",
                        "text": alert.description,
                        "fields": [
                            {"title": "Metric", "value": alert.metric, "short": True},
                            {"title": "Current Value", "value": str(alert.current_value), "short": True},
                            {"title": "Threshold", "value": str(alert.threshold), "short": True},
                            {"title": "Triggered At", "value": alert.triggered_at.strftime('%Y-%m-%d %H:%M:%S UTC'), "short": True}
                        ],
                        "footer": f"FlashMM Alert Manager | {alert.id}",
                        "ts": int(alert.triggered_at.timestamp())
                    }
                ]
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(config['webhook_url'], json=payload) as response:
                    return response.status == 200
            
        except Exception as e:
            logger.error(f"Failed to send Slack notification: {e}")
            return False
    
    async def _send_discord_notification(self, channel: NotificationChannel, alert: Alert) -> bool:
        """Send Discord notification."""
        try:
            config = channel.config
            
            color_map = {
                AlertSeverity.INFO: 0x00ff00,
                AlertSeverity.WARNING: 0xffff00,
                AlertSeverity.CRITICAL: 0xff4500,
                AlertSeverity.EMERGENCY: 0xff0000
            }
            
            payload = {
                "username": config.get('username', 'FlashMM'),
                "embeds": [
                    {
                        "title": f"{alert.severity.value.upper()}: {alert.name}",
                        "description": alert.description,
                        "color": color_map.get(alert.severity, 0xffff00),
                        "fields": [
                            {"name": "Metric", "value": alert.metric, "inline": True},
                            {"name": "Current Value", "value": str(alert.current_value), "inline": True},
                            {"name": "Threshold", "value": str(alert.threshold), "inline": True}
                        ],
                        "timestamp": alert.triggered_at.isoformat(),
                        "footer": {"text": f"Alert ID: {alert.id}"}
                    }
                ]
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(config['webhook_url'], json=payload) as response:
                    return response.status in [200, 204]
            
        except Exception as e:
            logger.error(f"Failed to send Discord notification: {e}")
            return False
    
    async def _send_webhook_notification(self, channel: NotificationChannel, alert: Alert) -> bool:
        """Send webhook notification."""
        try:
            config = channel.config
            
            payload = {
                "alert_id": alert.id,
                "rule_id": alert.rule_id,
                "name": alert.name,
                "description": alert.description,
                "severity": alert.severity.value,
                "status": alert.status.value,
                "metric": alert.metric,
                "current_value": alert.current_value,
                "threshold": alert.threshold,
                "triggered_at": alert.triggered_at.isoformat(),
                "tags": alert.tags,
                "annotations": alert.annotations
            }
            
            headers = config.get('headers', {})
            headers.setdefault('Content-Type', 'application/json')
            
            async with aiohttp.ClientSession() as session:
                method = config.get('method', 'POST').upper()
                async with session.request(method, config['url'], json=payload, headers=headers) as response:
                    return 200 <= response.status < 300
            
        except Exception as e:
            logger.error(f"Failed to send webhook notification: {e}")
            return False
    
    async def _send_dashboard_notification(self, channel: NotificationChannel, alert: Alert) -> bool:
        """Send dashboard notification (store for dashboard consumption)."""
        try:
            # This would integrate with a dashboard notification system
            # For now, we just log it as dashboard notifications are handled internally
            logger.info(f"Dashboard notification: {alert.name} - {alert.severity.value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send dashboard notification: {e}")
            return False
    
    async def _check_escalation(self, alert: Alert) -> None:
        """Check if alert should be escalated."""
        try:
            rule = self.alert_rules.get(alert.rule_id)
            if not rule or not rule.escalation_rules:
                return
            
            alert_age = (datetime.now() - alert.triggered_at).total_seconds() / 60  # minutes
            
            for escalation_rule in rule.escalation_rules:
                after_minutes = escalation_rule.get('after_minutes', 0)
                
                if alert_age >= after_minutes and alert.escalation_level < len(rule.escalation_rules):
                    # Escalate alert
                    alert.escalation_level += 1
                    alert.severity = AlertSeverity(escalation_rule.get('severity', alert.severity.value))
                    alert.status = AlertStatus.ESCALATED
                    alert.last_updated = datetime.now()
                    
                    # Send escalation notifications
                    escalation_channels = escalation_rule.get('channels', [])
                    for channel_name in escalation_channels:
                        channel = self.channels.get(channel_name)
                        if channel:
                            await self._send_notification(channel, alert)
                    
                    # Update statistics
                    self.stats["escalations"] += 1
                    
                    logger.warning(f"ALERT ESCALATED: {alert.name} to level {alert.escalation_level}")
                    break
        
        except Exception as e:
            logger.error(f"Error checking escalation for alert {alert.id}: {e}")
    
    async def _check_correlations(self, alert: Alert) -> None:
        """Check for alert correlations."""
        try:
            for correlation in self.correlations:
                if alert.rule_id not in correlation.alerts:
                    continue
                
                # Find related alerts within correlation window
                correlation_start = datetime.now() - timedelta(seconds=correlation.correlation_window)
                related_alerts = []
                
                for active_alert in self.active_alerts.values():
                    if (active_alert.rule_id in correlation.alerts and
                        active_alert.triggered_at >= correlation_start):
                        related_alerts.append(active_alert)
                
                if len(related_alerts) >= correlation.threshold:
                    await self._handle_correlation(correlation, related_alerts)
        
        except Exception as e:
            logger.error(f"Error checking correlations for alert {alert.id}: {e}")
    
    async def _handle_correlation(self, correlation: AlertCorrelation, alerts: List[Alert]) -> None:
        """Handle correlated alerts."""
        try:
            if correlation.action == "suppress":
                # Suppress related alerts except the first one
                for alert in alerts[1:]:
                    alert.status = AlertStatus.SUPPRESSED
                    alert.suppressed_until = datetime.now() + timedelta(seconds=correlation.correlation_window)
                    alert.last_updated = datetime.now()
                    logger.info(f"Alert suppressed due to correlation: {alert.name}")
            
            elif correlation.action == "escalate":
                # Escalate all related alerts
                for alert in alerts:
                    if alert.severity != AlertSeverity.EMERGENCY:
                        alert.severity = AlertSeverity.EMERGENCY
                        alert.status = AlertStatus.ESCALATED
                        alert.last_updated = datetime.now()
                        logger.warning(f"Alert escalated due to correlation: {alert.name}")
            
            elif correlation.action == "merge":
                # Create a new merged alert (simplified implementation)
                logger.info(f"Alert correlation detected: {correlation.name} with {len(alerts)} alerts")
        
        except Exception as e:
            logger.error(f"Error handling correlation {correlation.name}: {e}")
    
    async def _cleanup_old_alerts(self) -> None:
        """Clean up old resolved alerts."""
        try:
            cutoff_time = datetime.now() - timedelta(days=30)  # Keep 30 days of history
            
            # Clean up alert history
            self.alert_history = [
                alert for alert in self.alert_history
                if alert.triggered_at > cutoff_time or alert.status == AlertStatus.ACTIVE
            ]
            
            # Clean up suppressed alerts that have expired
            expired_suppressions = []
            for alert_id, alert in self.active_alerts.items():
                if (alert.status == AlertStatus.SUPPRESSED and
                    alert.suppressed_until and
                    datetime.now() > alert.suppressed_until):
                    expired_suppressions.append(alert_id)
            
            for alert_id in expired_suppressions:
                alert = self.active_alerts[alert_id]
                alert.status = AlertStatus.ACTIVE
                alert.suppressed_until = None
                alert.last_updated = datetime.now()
                logger.info(f"Alert suppression expired: {alert.name}")
            
            logger.debug(f"Cleaned up old alerts, keeping {len(self.alert_history)} in history")
        
        except Exception as e:
            logger.error(f"Error cleaning up old alerts: {e}")
    
    async def _update_statistics(self) -> None:
        """Update alert manager statistics."""
        try:
            # Update current counts
            self.stats["alerts_by_status"] = {s.value: 0 for s in AlertStatus}
            self.stats["alerts_by_severity"] = {s.value: 0 for s in AlertSeverity}
            
            for alert in self.active_alerts.values():
                self.stats["alerts_by_status"][alert.status.value] += 1
                self.stats["alerts_by_severity"][alert.severity.value] += 1
            
            # Calculate uptime
            uptime_seconds = (datetime.now() - self.stats["uptime"]).total_seconds()
            self.stats["uptime_hours"] = uptime_seconds / 3600
            
        except Exception as e:
            logger.error(f"Error updating statistics: {e}")
    
    def _update_avg_resolution_time(self, resolution_time: float) -> None:
        """Update average resolution time."""
        try:
            current_avg = self.stats.get("avg_resolution_time", 0.0)
            resolved_count = self.stats["alerts_by_status"][AlertStatus.RESOLVED.value]
            
            if resolved_count > 1:
                # Update running average
                self.stats["avg_resolution_time"] = ((current_avg * (resolved_count - 1)) + resolution_time) / resolved_count
            else:
                self.stats["avg_resolution_time"] = resolution_time
        
        except Exception as e:
            logger.error(f"Error updating average resolution time: {e}")
    
    # Public methods for rule management
    
    async def add_alert_rule(self, rule: AlertRule) -> bool:
        """Add a new alert rule."""
        try:
            if rule.id in self.alert_rules:
                logger.warning(f"Alert rule {rule.id} already exists, updating")
            
            self.alert_rules[rule.id] = rule
            logger.info(f"Added alert rule: {rule.id}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding alert rule {rule.id}: {e}")
            return False
    
    async def remove_alert_rule(self, rule_id: str) -> bool:
        """Remove an alert rule."""
        try:
            if rule_id not in self.alert_rules:
                logger.warning(f"Alert rule {rule_id} not found")
                return False
            
            # Resolve any active alerts for this rule
            alerts_to_resolve = [
                alert_id for alert_id, alert in self.active_alerts.items()
                if alert.rule_id == rule_id
            ]
            
            for alert_id in alerts_to_resolve:
                await self._resolve_alert(alert_id, "Rule removed")
            
            del self.alert_rules[rule_id]
            logger.info(f"Removed alert rule: {rule_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error removing alert rule {rule_id}: {e}")
            return False
    
    async def update_alert_rule(self, rule: AlertRule) -> bool:
        """Update an existing alert rule."""
        try:
            if rule.id not in self.alert_rules:
                logger.warning(f"Alert rule {rule.id} not found, adding new rule")
            
            self.alert_rules[rule.id] = rule
            logger.info(f"Updated alert rule: {rule.id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating alert rule {rule.id}: {e}")
            return False
    
    async def enable_alert_rule(self, rule_id: str) -> bool:
        """Enable an alert rule."""
        try:
            rule = self.alert_rules.get(rule_id)
            if not rule:
                logger.warning(f"Alert rule {rule_id} not found")
                return False
            
            rule.enabled = True
            logger.info(f"Enabled alert rule: {rule_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error enabling alert rule {rule_id}: {e}")
            return False
    
    async def disable_alert_rule(self, rule_id: str) -> bool:
        """Disable an alert rule."""
        try:
            rule = self.alert_rules.get(rule_id)
            if not rule:
                logger.warning(f"Alert rule {rule_id} not found")
                return False
            
            rule.enabled = False
            
            # Resolve any active alerts for this rule
            alerts_to_resolve = [
                alert_id for alert_id, alert in self.active_alerts.items()
                if alert.rule_id == rule_id
            ]
            
            for alert_id in alerts_to_resolve:
                await self._resolve_alert(alert_id, "Rule disabled")
            
            logger.info(f"Disabled alert rule: {rule_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error disabling alert rule {rule_id}: {e}")
            return False
    
    # Public methods for alert management
    
    def get_active_alerts(self, severity_filter: Optional[List[AlertSeverity]] = None) -> List[Alert]:
        """Get active alerts with optional severity filter."""
        alerts = list(self.active_alerts.values())
        
        if severity_filter:
            alerts = [alert for alert in alerts if alert.severity in severity_filter]
        
        return sorted(alerts, key=lambda x: x.triggered_at, reverse=True)
    
    def get_alert_history(self, limit: int = 100) -> List[Alert]:
        """Get alert history."""
        return sorted(self.alert_history, key=lambda x: x.triggered_at, reverse=True)[:limit]
    
    def get_alert_rules(self) -> Dict[str, AlertRule]:
        """Get all alert rules."""
        return self.alert_rules.copy()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get alert manager statistics."""
        return self.stats.copy()
    
    def add_alert_handler(self, handler: Callable[[Alert], None]) -> None:
        """Add an alert callback handler."""
        self.alert_handlers.append(handler)
    
    def add_resolution_handler(self, handler: Callable[[Alert], None]) -> None:
        """Add a resolution callback handler."""
        self.resolution_handlers.append(handler)
    
    async def create_maintenance_window(self, start_time: datetime, end_time: datetime,
                                     rules: List[str], reason: str = "") -> str:
        """Create a maintenance window to suppress alerts."""
        try:
            window_id = f"maint_{int(datetime.now().timestamp())}"
            
            maintenance_window = {
                "id": window_id,
                "start_time": start_time,
                "end_time": end_time,
                "rules": rules,
                "reason": reason,
                "created_at": datetime.now()
            }
            
            self.maintenance_windows.append(maintenance_window)
            logger.info(f"Created maintenance window: {window_id}")
            return window_id
            
        except Exception as e:
            logger.error(f"Error creating maintenance window: {e}")
            return ""
    
    async def cancel_maintenance_window(self, window_id: str) -> bool:
        """Cancel a maintenance window."""
        try:
            self.maintenance_windows = [
                window for window in self.maintenance_windows
                if window["id"] != window_id
            ]
            logger.info(f"Cancelled maintenance window: {window_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error cancelling maintenance window {window_id}: {e}")
            return False
    
    def is_in_maintenance_window(self, rule_id: str) -> bool:
        """Check if a rule is in a maintenance window."""
        try:
            now = datetime.now()
            
            for window in self.maintenance_windows:
                if (rule_id in window["rules"] and
                    window["start_time"] <= now <= window["end_time"]):
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking maintenance window for rule {rule_id}: {e}")
            return False
    
    async def test_notification_channel(self, channel_name: str) -> bool:
        """Test a notification channel."""
        try:
            channel = self.channels.get(channel_name)
            if not channel:
                logger.error(f"Notification channel {channel_name} not found")
                return False
            
            # Create a test alert
            test_alert = Alert(
                id="test_alert",
                rule_id="test_rule",
                name="Test Alert",
                description="This is a test alert to verify notification channel functionality",
                severity=AlertSeverity.INFO,
                status=AlertStatus.ACTIVE,
                metric="test.metric",
                current_value="test_value",
                threshold="test_threshold",
                triggered_at=datetime.now(),
                last_updated=datetime.now(),
                tags={"test": "true"},
                annotations={"test": "true"}
            )
            
            success = await self._send_notification(channel, test_alert)
            
            if success:
                logger.info(f"Test notification sent successfully via {channel_name}")
            else:
                logger.error(f"Test notification failed via {channel_name}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error testing notification channel {channel_name}: {e}")
            return False
    
    async def shutdown(self) -> None:
        """Shutdown the alert manager."""
        try:
            logger.info("Shutting down AlertManager...")
            
            self.running = False
            
            # Cancel background tasks
            tasks = [self.evaluation_task, self.notification_task, self.cleanup_task]
            for task in tasks:
                if task and not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
            
            # Resolve all active alerts
            for alert_id in list(self.active_alerts.keys()):
                await self._resolve_alert(alert_id, "System shutdown")
            
            logger.info("AlertManager shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during AlertManager shutdown: {e}")
    
    def __repr__(self) -> str:
        """String representation of AlertManager."""
        active_count = len(self.active_alerts)
        rule_count = len(self.alert_rules)
        channel_count = len(self.channels)
        
        return (f"AlertManager(rules={rule_count}, active_alerts={active_count}, "
                f"channels={channel_count}, running={self.running})")


# Utility functions for creating common alert rules

def create_latency_alert_rule(metric_name: str, threshold_ms: float,
                            severity: AlertSeverity = AlertSeverity.WARNING) -> AlertRule:
    """Create a latency-based alert rule."""
    return AlertRule(
        id=f"latency_{metric_name.replace('.', '_')}",
        name=f"High Latency - {metric_name}",
        description=f"Latency for {metric_name} exceeded {threshold_ms}ms",
        metric=metric_name,
        condition="gt",
        threshold=threshold_ms,
        severity=severity,
        channels=[AlertChannel.EMAIL, AlertChannel.SLACK, AlertChannel.DASHBOARD],
        evaluation_interval=60,
        for_duration=120,
        tags={"type": "latency", "metric": metric_name}
    )


def create_error_rate_alert_rule(metric_name: str, threshold_percent: float,
                               severity: AlertSeverity = AlertSeverity.CRITICAL) -> AlertRule:
    """Create an error rate alert rule."""
    return AlertRule(
        id=f"error_rate_{metric_name.replace('.', '_')}",
        name=f"High Error Rate - {metric_name}",
        description=f"Error rate for {metric_name} exceeded {threshold_percent}%",
        metric=metric_name,
        condition="gt",
        threshold=threshold_percent,
        severity=severity,
        channels=[AlertChannel.EMAIL, AlertChannel.SLACK, AlertChannel.PAGERDUTY],
        evaluation_interval=60,
        for_duration=180,
        tags={"type": "error_rate", "metric": metric_name}
    )


def create_threshold_alert_rule(metric_name: str, condition: str, threshold: Union[float, str],
                              severity: AlertSeverity = AlertSeverity.WARNING,
                              name: str = "", description: str = "") -> AlertRule:
    """Create a generic threshold-based alert rule."""
    rule_name = name or f"Threshold Alert - {metric_name}"
    rule_description = description or f"{metric_name} {condition} {threshold}"
    
    return AlertRule(
        id=f"threshold_{metric_name.replace('.', '_')}_{condition}_{str(threshold).replace('.', '_')}",
        name=rule_name,
        description=rule_description,
        metric=metric_name,
        condition=condition,
        threshold=threshold,
        severity=severity,
        channels=[AlertChannel.EMAIL, AlertChannel.DASHBOARD],
        evaluation_interval=60,
        for_duration=120,
        tags={"type": "threshold", "metric": metric_name}
    )
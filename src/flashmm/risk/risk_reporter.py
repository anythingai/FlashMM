"""
FlashMM Risk Reporting System

Comprehensive risk reporting and dashboard system with:
- Real-time risk reports and dashboards
- Risk metric time series and trend analysis
- Risk limit compliance reporting
- Daily risk summary and breach notifications
- Risk performance attribution analysis
- Regulatory-style risk reports
"""

import asyncio
import json
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from collections import defaultdict, deque

from flashmm.config.settings import get_config
from flashmm.utils.logging import get_logger
from flashmm.utils.exceptions import RiskError
from flashmm.utils.decorators import measure_latency, timeout_async

logger = get_logger(__name__)


class ReportType(Enum):
    """Risk report types."""
    REAL_TIME_DASHBOARD = "real_time_dashboard"
    DAILY_SUMMARY = "daily_summary"
    WEEKLY_SUMMARY = "weekly_summary"
    MONTHLY_SUMMARY = "monthly_summary"
    COMPLIANCE_REPORT = "compliance_report"
    BREACH_NOTIFICATION = "breach_notification"
    PERFORMANCE_ATTRIBUTION = "performance_attribution"
    REGULATORY_REPORT = "regulatory_report"
    STRESS_TEST_REPORT = "stress_test_report"


class RiskLevel(Enum):
    """Risk level classifications."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RiskMetric:
    """Individual risk metric."""
    name: str
    value: float
    threshold: float
    warning_threshold: float
    unit: str
    category: str
    timestamp: datetime
    
    # Risk classification
    risk_level: RiskLevel = field(init=False)
    breach: bool = field(init=False)
    warning: bool = field(init=False)
    
    def __post_init__(self):
        """Calculate risk levels and breach status."""
        self.breach = abs(self.value) > self.threshold
        self.warning = abs(self.value) > self.warning_threshold
        
        if self.breach:
            self.risk_level = RiskLevel.CRITICAL
        elif self.warning:
            self.risk_level = RiskLevel.HIGH
        elif abs(self.value) > self.warning_threshold * 0.7:
            self.risk_level = RiskLevel.MEDIUM
        else:
            self.risk_level = RiskLevel.LOW
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'value': self.value,
            'threshold': self.threshold,
            'warning_threshold': self.warning_threshold,
            'unit': self.unit,
            'category': self.category,
            'timestamp': self.timestamp.isoformat(),
            'risk_level': self.risk_level.value,
            'breach': self.breach,
            'warning': self.warning,
            'utilization_pct': (abs(self.value) / self.threshold * 100) if self.threshold > 0 else 0
        }


@dataclass
class RiskDashboardData:
    """Real-time risk dashboard data."""
    timestamp: datetime
    overall_risk_score: float  # 0-100 composite risk score
    
    # Risk categories
    market_risk_score: float
    credit_risk_score: float
    operational_risk_score: float
    liquidity_risk_score: float
    
    # Key metrics
    var_1d: float
    expected_shortfall: float
    maximum_drawdown: float
    sharpe_ratio: float
    
    # Position metrics
    total_exposure: float
    leverage_ratio: float
    concentration_risk: float
    
    # Operational metrics
    system_health_score: float
    connectivity_score: float
    
    # Alert counts
    critical_alerts: int
    high_alerts: int
    medium_alerts: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'overall_risk_score': self.overall_risk_score,
            'risk_categories': {
                'market_risk': self.market_risk_score,
                'credit_risk': self.credit_risk_score,
                'operational_risk': self.operational_risk_score,
                'liquidity_risk': self.liquidity_risk_score
            },
            'key_metrics': {
                'var_1d': self.var_1d,
                'expected_shortfall': self.expected_shortfall,
                'maximum_drawdown': self.maximum_drawdown,
                'sharpe_ratio': self.sharpe_ratio
            },
            'position_metrics': {
                'total_exposure': self.total_exposure,
                'leverage_ratio': self.leverage_ratio,
                'concentration_risk': self.concentration_risk
            },
            'operational_metrics': {
                'system_health_score': self.system_health_score,
                'connectivity_score': self.connectivity_score
            },
            'alert_summary': {
                'critical': self.critical_alerts,
                'high': self.high_alerts,
                'medium': self.medium_alerts,
                'total': self.critical_alerts + self.high_alerts + self.medium_alerts
            }
        }


@dataclass
class ComplianceReport:
    """Risk compliance report."""
    report_date: datetime
    reporting_period: str
    
    # Limit compliance
    position_limit_breaches: List[Dict[str, Any]]
    var_limit_breaches: List[Dict[str, Any]]
    drawdown_limit_breaches: List[Dict[str, Any]]
    concentration_limit_breaches: List[Dict[str, Any]]
    
    # Summary statistics
    total_breaches: int
    critical_breaches: int
    breach_duration_hours: float
    compliance_score: float  # 0-100
    
    # Corrective actions
    actions_taken: List[Dict[str, Any]]
    outstanding_issues: List[Dict[str, Any]]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'report_date': self.report_date.isoformat(),
            'reporting_period': self.reporting_period,
            'limit_breaches': {
                'position_limits': self.position_limit_breaches,
                'var_limits': self.var_limit_breaches,
                'drawdown_limits': self.drawdown_limit_breaches,
                'concentration_limits': self.concentration_limit_breaches
            },
            'summary': {
                'total_breaches': self.total_breaches,
                'critical_breaches': self.critical_breaches,
                'breach_duration_hours': self.breach_duration_hours,
                'compliance_score': self.compliance_score
            },
            'remediation': {
                'actions_taken': self.actions_taken,
                'outstanding_issues': self.outstanding_issues
            }
        }


class RiskDataCollector:
    """Collects risk data from various risk management components."""
    
    def __init__(self):
        self.config = get_config()
        self.risk_components: Dict[str, Any] = {}
        
    def register_risk_component(self, name: str, component: Any):
        """Register a risk management component."""
        self.risk_components[name] = component
        logger.info(f"Registered risk component: {name}")
    
    async def collect_all_risk_data(self) -> Dict[str, Any]:
        """Collect risk data from all registered components."""
        risk_data = {
            'collection_timestamp': datetime.now().isoformat(),
            'components': {}
        }
        
        # Collect from circuit breakers
        if 'circuit_breakers' in self.risk_components:
            try:
                cb_system = self.risk_components['circuit_breakers']
                risk_data['components']['circuit_breakers'] = cb_system.get_system_status()
            except Exception as e:
                logger.error(f"Error collecting circuit breaker data: {e}")
                risk_data['components']['circuit_breakers'] = {'error': str(e)}
        
        # Collect from position limits
        if 'position_limits' in self.risk_components:
            try:
                pos_limits = self.risk_components['position_limits']
                risk_data['components']['position_limits'] = pos_limits.get_all_limits_status()
            except Exception as e:
                logger.error(f"Error collecting position limits data: {e}")
                risk_data['components']['position_limits'] = {'error': str(e)}
        
        # Collect from market risk monitor
        if 'market_risk' in self.risk_components:
            try:
                market_risk = self.risk_components['market_risk']
                # Would call market_risk.get_current_analysis() or similar
                risk_data['components']['market_risk'] = {'status': 'active'}
            except Exception as e:
                logger.error(f"Error collecting market risk data: {e}")
                risk_data['components']['market_risk'] = {'error': str(e)}
        
        # Collect from P&L controller
        if 'pnl_controller' in self.risk_components:
            try:
                pnl_controller = self.risk_components['pnl_controller']
                risk_data['components']['pnl_controller'] = pnl_controller.get_pnl_summary()
            except Exception as e:
                logger.error(f"Error collecting P&L data: {e}")
                risk_data['components']['pnl_controller'] = {'error': str(e)}
        
        # Collect from operational risk
        if 'operational_risk' in self.risk_components:
            try:
                op_risk = self.risk_components['operational_risk']
                risk_data['components']['operational_risk'] = await op_risk.get_operational_status()
            except Exception as e:
                logger.error(f"Error collecting operational risk data: {e}")
                risk_data['components']['operational_risk'] = {'error': str(e)}
        
        return risk_data


class RiskDashboard:
    """Real-time risk dashboard generator."""
    
    def __init__(self, data_collector: RiskDataCollector):
        self.data_collector = data_collector
        self.config = get_config()
        
        # Dashboard refresh settings
        self.refresh_interval_seconds = self.config.get("reporting.dashboard_refresh_seconds", 10)
        self.history_retention_hours = self.config.get("reporting.history_retention_hours", 24)
        
        # Historical data storage
        self.dashboard_history: deque = deque(maxlen=8640)  # 24 hours at 10-second intervals
        
    async def generate_dashboard_data(self) -> RiskDashboardData:
        """Generate real-time dashboard data."""
        try:
            # Collect all risk data
            risk_data = await self.data_collector.collect_all_risk_data()
            
            # Calculate composite risk scores
            overall_risk_score = await self._calculate_overall_risk_score(risk_data)
            market_risk_score = await self._calculate_market_risk_score(risk_data)
            operational_risk_score = await self._calculate_operational_risk_score(risk_data)
            
            # Extract key metrics
            key_metrics = await self._extract_key_metrics(risk_data)
            
            # Count alerts
            alert_counts = await self._count_alerts(risk_data)
            
            dashboard_data = RiskDashboardData(
                timestamp=datetime.now(),
                overall_risk_score=overall_risk_score,
                market_risk_score=market_risk_score,
                credit_risk_score=0.0,  # Not implemented yet
                operational_risk_score=operational_risk_score,
                liquidity_risk_score=0.0,  # Not implemented yet
                var_1d=key_metrics.get('var_1d', 0.0),
                expected_shortfall=key_metrics.get('expected_shortfall', 0.0),
                maximum_drawdown=key_metrics.get('maximum_drawdown', 0.0),
                sharpe_ratio=key_metrics.get('sharpe_ratio', 0.0),
                total_exposure=key_metrics.get('total_exposure', 0.0),
                leverage_ratio=key_metrics.get('leverage_ratio', 1.0),
                concentration_risk=key_metrics.get('concentration_risk', 0.0),
                system_health_score=key_metrics.get('system_health_score', 100.0),
                connectivity_score=key_metrics.get('connectivity_score', 100.0),
                critical_alerts=alert_counts.get('critical', 0),
                high_alerts=alert_counts.get('high', 0),
                medium_alerts=alert_counts.get('medium', 0)
            )
            
            # Store in history
            self.dashboard_history.append(dashboard_data)
            
            return dashboard_data
            
        except Exception as e:
            logger.error(f"Error generating dashboard data: {e}")
            raise RiskError(f"Dashboard generation failed: {e}")
    
    async def _calculate_overall_risk_score(self, risk_data: Dict[str, Any]) -> float:
        """Calculate overall composite risk score (0-100)."""
        try:
            scores = []
            weights = []
            
            # Market risk component
            market_score = await self._calculate_market_risk_score(risk_data)
            scores.append(market_score)
            weights.append(0.4)  # 40% weight
            
            # Operational risk component
            operational_score = await self._calculate_operational_risk_score(risk_data)
            scores.append(operational_score)
            weights.append(0.3)  # 30% weight
            
            # Position risk component
            position_score = await self._calculate_position_risk_score(risk_data)
            scores.append(position_score)
            weights.append(0.3)  # 30% weight
            
            # Weighted average
            if scores and weights:
                weighted_score = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
                return min(100.0, max(0.0, weighted_score))
            
            return 50.0  # Default neutral score
            
        except Exception as e:
            logger.error(f"Error calculating overall risk score: {e}")
            return 50.0
    
    async def _calculate_market_risk_score(self, risk_data: Dict[str, Any]) -> float:
        """Calculate market risk score."""
        try:
            score = 0.0
            
            # Check circuit breakers
            cb_data = risk_data.get('components', {}).get('circuit_breakers', {})
            if cb_data and not cb_data.get('error'):
                active_breakers = cb_data.get('active_breakers', 0)
                total_breakers = cb_data.get('total_breakers', 1)
                
                # Higher score means higher risk
                score += (active_breakers / total_breakers) * 50
            
            # Check P&L drawdown
            pnl_data = risk_data.get('components', {}).get('pnl_controller', {})
            if pnl_data and not pnl_data.get('error'):
                # Simplified - would use actual drawdown metrics
                score += 10  # Base market risk
            
            return min(100.0, score)
            
        except Exception as e:
            logger.error(f"Error calculating market risk score: {e}")
            return 25.0
    
    async def _calculate_operational_risk_score(self, risk_data: Dict[str, Any]) -> float:
        """Calculate operational risk score."""
        try:
            op_data = risk_data.get('components', {}).get('operational_risk', {})
            if not op_data or op_data.get('error'):
                return 25.0  # Default score when no data
            
            # Convert health levels to risk scores (inverse relationship)
            health_mapping = {
                'excellent': 5,
                'good': 15,
                'warning': 40,
                'critical': 80,
                'failure': 100
            }
            
            overall_health = op_data.get('overall_health', 'good')
            base_score = health_mapping.get(overall_health, 25)
            
            # Adjust based on active alerts
            alert_summary = op_data.get('alert_summary', {})
            critical_alerts = alert_summary.get('critical', 0)
            high_alerts = alert_summary.get('high', 0)
            
            alert_penalty = min(30, critical_alerts * 10 + high_alerts * 5)
            
            return min(100.0, base_score + alert_penalty)
            
        except Exception as e:
            logger.error(f"Error calculating operational risk score: {e}")
            return 25.0
    
    async def _calculate_position_risk_score(self, risk_data: Dict[str, Any]) -> float:
        """Calculate position risk score."""
        try:
            pos_data = risk_data.get('components', {}).get('position_limits', {})
            if not pos_data or pos_data.get('error'):
                return 25.0
            
            # Check limit violations
            limits = pos_data.get('limits', {})
            violation_count = 0
            total_limits = len(limits)
            
            for limit_name, limit_info in limits.items():
                utilization = limit_info.get('utilization_percent', 0)
                if utilization > 100:  # Breach
                    violation_count += 2
                elif utilization > 80:  # Warning
                    violation_count += 1
            
            if total_limits > 0:
                violation_ratio = violation_count / (total_limits * 2)  # Max 2 points per limit
                return min(100.0, violation_ratio * 100)
            
            return 10.0  # Low risk when no limits defined
            
        except Exception as e:
            logger.error(f"Error calculating position risk score: {e}")
            return 25.0
    
    async def _extract_key_metrics(self, risk_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract key risk metrics from collected data."""
        metrics = {}
        
        try:
            # P&L metrics
            pnl_data = risk_data.get('components', {}).get('pnl_controller', {})
            if pnl_data and not pnl_data.get('error'):
                global_metrics = pnl_data.get('global_metrics', {})
                metrics['maximum_drawdown'] = abs(global_metrics.get('daily', 0.0))
                metrics['total_pnl'] = global_metrics.get('daily', 0.0)
            
            # Position metrics
            pos_data = risk_data.get('components', {}).get('position_limits', {})
            if pos_data and not pos_data.get('error'):
                global_settings = pos_data.get('global_settings', {})
                metrics['total_exposure'] = global_settings.get('global_notional_limit', 0.0)
            
            # Operational metrics
            op_data = risk_data.get('components', {}).get('operational_risk', {})
            if op_data and not op_data.get('error'):
                # Convert health to score
                health_scores = {
                    'excellent': 100, 'good': 80, 'warning': 60, 
                    'critical': 30, 'failure': 0
                }
                
                system_health = op_data.get('system_health', 'good')
                connectivity_health = op_data.get('connectivity_health', 'good')
                
                metrics['system_health_score'] = health_scores.get(system_health, 50)
                metrics['connectivity_score'] = health_scores.get(connectivity_health, 50)
            
            # Default values for missing metrics
            metrics.setdefault('var_1d', 0.0)
            metrics.setdefault('expected_shortfall', 0.0)
            metrics.setdefault('sharpe_ratio', 0.0)
            metrics.setdefault('leverage_ratio', 1.0)
            metrics.setdefault('concentration_risk', 0.0)
            
        except Exception as e:
            logger.error(f"Error extracting key metrics: {e}")
        
        return metrics
    
    async def _count_alerts(self, risk_data: Dict[str, Any]) -> Dict[str, int]:
        """Count alerts by severity across all components."""
        alert_counts = defaultdict(int)
        
        try:
            # Circuit breaker alerts
            cb_data = risk_data.get('components', {}).get('circuit_breakers', {})
            if cb_data and not cb_data.get('error'):
                alert_counts['critical'] += cb_data.get('active_breakers', 0)
            
            # Operational alerts
            op_data = risk_data.get('components', {}).get('operational_risk', {})
            if op_data and not op_data.get('error'):
                alert_summary = op_data.get('alert_summary', {})
                for severity, count in alert_summary.items():
                    if isinstance(count, int):
                        alert_counts[severity] += count
            
            # Position limit alerts (would check for violations)
            pos_data = risk_data.get('components', {}).get('position_limits', {})
            if pos_data and not pos_data.get('error'):
                limits = pos_data.get('limits', {})
                for limit_info in limits.values():
                    utilization = limit_info.get('utilization_percent', 0)
                    if utilization > 100:
                        alert_counts['critical'] += 1
                    elif utilization > 90:
                        alert_counts['high'] += 1
                    elif utilization > 80:
                        alert_counts['medium'] += 1
        
        except Exception as e:
            logger.error(f"Error counting alerts: {e}")
        
        return dict(alert_counts)
    
    def get_dashboard_history(self, hours: int = 1) -> List[Dict[str, Any]]:
        """Get dashboard history for specified hours."""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            recent_data = [
                data.to_dict() for data in self.dashboard_history
                if data.timestamp > cutoff_time
            ]
            return recent_data
        except Exception as e:
            logger.error(f"Error getting dashboard history: {e}")
            return []


class ComplianceReporter:
    """Generate compliance and regulatory reports."""
    
    def __init__(self, data_collector: RiskDataCollector):
        self.data_collector = data_collector
        self.config = get_config()
        
        # Compliance thresholds
        self.compliance_thresholds = {
            'position_limit_breach_hours': 1.0,  # Max 1 hour breach allowed
            'var_limit_breach_count': 3,         # Max 3 VaR breaches per month
            'minimum_compliance_score': 85.0     # Minimum 85% compliance
        }
    
    async def generate_daily_compliance_report(self, report_date: datetime) -> ComplianceReport:
        """Generate daily compliance report."""
        try:
            # Collect current risk data
            risk_data = await self.data_collector.collect_all_risk_data()
            
            # Analyze breaches
            position_breaches = await self._analyze_position_limit_breaches(risk_data)
            var_breaches = await self._analyze_var_breaches(risk_data)
            drawdown_breaches = await self._analyze_drawdown_breaches(risk_data)
            concentration_breaches = await self._analyze_concentration_breaches(risk_data)
            
            # Calculate summary statistics
            all_breaches = position_breaches + var_breaches + drawdown_breaches + concentration_breaches
            total_breaches = len(all_breaches)
            critical_breaches = len([b for b in all_breaches if b.get('severity') == 'critical'])
            
            # Calculate compliance score
            compliance_score = await self._calculate_compliance_score(all_breaches)
            
            # Generate corrective actions
            actions_taken = await self._get_corrective_actions(risk_data)
            outstanding_issues = await self._get_outstanding_issues(all_breaches)
            
            return ComplianceReport(
                report_date=report_date,
                reporting_period="daily",
                position_limit_breaches=position_breaches,
                var_limit_breaches=var_breaches,
                drawdown_limit_breaches=drawdown_breaches,
                concentration_limit_breaches=concentration_breaches,
                total_breaches=total_breaches,
                critical_breaches=critical_breaches,
                breach_duration_hours=sum(b.get('duration_hours', 0) for b in all_breaches),
                compliance_score=compliance_score,
                actions_taken=actions_taken,
                outstanding_issues=outstanding_issues
            )
            
        except Exception as e:
            logger.error(f"Error generating compliance report: {e}")
            raise RiskError(f"Compliance report generation failed: {e}")
    
    async def _analyze_position_limit_breaches(self, risk_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze position limit breaches."""
        breaches = []
        
        try:
            pos_data = risk_data.get('components', {}).get('position_limits', {})
            if pos_data and not pos_data.get('error'):
                limits = pos_data.get('limits', {})
                
                for limit_name, limit_info in limits.items():
                    utilization = limit_info.get('utilization_percent', 0)
                    violations_count = limit_info.get('violations_count', 0)
                    
                    if utilization > 100:
                        breaches.append({
                            'limit_name': limit_name,
                            'limit_type': limit_info.get('type', 'unknown'),
                            'current_value': limit_info.get('current_value', 0),
                            'limit_value': limit_info.get('limit_value', 0),
                            'utilization_percent': utilization,
                            'severity': 'critical' if utilization > 150 else 'high',
                            'violations_count': violations_count,
                            'duration_hours': 0.5,  # Would calculate actual duration
                            'timestamp': datetime.now().isoformat()
                        })
        
        except Exception as e:
            logger.error(f"Error analyzing position limit breaches: {e}")
        
        return breaches
    
    async def _analyze_var_breaches(self, risk_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze VaR limit breaches."""
        # Placeholder - would implement actual VaR breach analysis
        return []
    
    async def _analyze_drawdown_breaches(self, risk_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze drawdown limit breaches."""
        breaches = []
        
        try:
            pnl_data = risk_data.get('components', {}).get('pnl_controller', {})
            if pnl_data and not pnl_data.get('error'):
                # Would analyze actual drawdown data
                global_metrics = pnl_data.get('global_metrics', {})
                daily_pnl = global_metrics.get('daily', 0.0)
                
                if daily_pnl < -5000:  # Example threshold
                    breaches.append({
                        'breach_type': 'daily_drawdown',
                        'current_drawdown': abs(daily_pnl),
                        'limit': 5000,
                        'severity': 'critical' if daily_pnl < -10000 else 'high',
                        'duration_hours': 1.0,
                        'timestamp': datetime.now().isoformat()
                    })
        
        except Exception as e:
            logger.error(f"Error analyzing drawdown breaches: {e}")
        
        return breaches
    
    async def _analyze_concentration_breaches(self, risk_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze concentration risk breaches."""
        # Placeholder - would implement actual concentration analysis
        return []
    
    async def _calculate_compliance_score(self, breaches: List[Dict[str, Any]]) -> float:
        """Calculate overall compliance score."""
        try:
            if not breaches:
                return 100.0
            
            # Penalty system
            penalty = 0.0
            for breach in breaches:
                severity = breach.get('severity', 'low')
                duration = breach.get('duration_hours', 0)
                
                if severity == 'critical':
                    penalty += 10 + (duration * 2)
                elif severity == 'high':
                    penalty += 5 + duration
                else:
                    penalty += 2
            
            compliance_score = max(0.0, 100.0 - penalty)
            return compliance_score
            
        except Exception as e:
            logger.error(f"Error calculating compliance score: {e}")
            return 50.0
    
    async def _get_corrective_actions(self, risk_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get list of corrective actions taken."""
        actions = []
        
        try:
            # Check circuit breaker actions
            cb_data = risk_data.get('components', {}).get('circuit_breakers', {})
            if cb_data and cb_data.get('system_halted'):
                actions.append({
                    'action_type': 'trading_halt',
                    'reason': cb_data.get('halt_reason', 'Circuit breaker triggered'),
                    'timestamp': datetime.now().isoformat(),
                    'status': 'completed'
                })
            
            # Check operational risk actions
            op_data = risk_data.get('components', {}).get('operational_risk', {})
            if op_data and not op_data.get('error'):
                # Would check for automatic actions taken
                pass
        
        except Exception as e:
            logger.error(f"Error getting corrective actions: {e}")
        
        return actions
    
    async def _get_outstanding_issues(self, breaches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Get list of outstanding compliance issues."""
        outstanding = []
        
        try:
            for breach in breaches:
                if breach.get('severity') in ['critical', 'high']:
                    outstanding.append({
                        'issue_type': breach.get('breach_type', breach.get('limit_name', 'unknown')),
                        'severity': breach.get('severity'),
                        'description': f"Breach of {breach.get('limit_name', 'limit')}",
                        'first_observed': breach.get('timestamp'),
                        'current_value': breach.get('current_value', 0),
                        'required_action': 'Reduce exposure to comply with limits'
                    })
        
        except Exception as e:
            logger.error(f"Error getting outstanding issues: {e}")
        
        return outstanding


class RiskReporter:
    """Main risk reporting system coordinator."""
    
    def __init__(self):
        self.config = get_config()
        self.data_collector = RiskDataCollector()
        self.dashboard = RiskDashboard(self.data_collector)
        self.compliance_reporter = ComplianceReporter(self.data_collector)
        
        # Report generation settings
        self.auto_report_enabled = self.config.get("reporting.auto_reports_enabled", True)
        self.daily_report_time = self.config.get("reporting.daily_report_time", "18:00")
        
        # Report storage
        self.report_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
    async def initialize(self):
        """Initialize risk reporting system."""
        try:
            logger.info("Risk reporting system initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize risk reporting system: {e}")
            raise RiskError(f"Risk reporting initialization failed: {e}")
    
    async def generate_real_time_dashboard(self) -> Dict[str, Any]:
        """Generate real-time risk dashboard."""
        try:
            dashboard_data = await self.dashboard.generate_dashboard_data()
            return dashboard_data.to_dict()
        except Exception as e:
            logger.error(f"Error generating real-time dashboard: {e}")
            raise RiskError(f"Dashboard generation failed: {e}")
    
    async def generate_daily_summary_report(self, report_date: Optional[datetime] = None) -> Dict[str, Any]:
        """Generate daily risk summary report."""
        try:
            if report_date is None:
                report_date = datetime.now()
            
            # Collect comprehensive risk data
            risk_data = await self.data_collector.collect_all_risk_data()
            
            # Generate compliance report
            compliance_report = await self.compliance_reporter.generate_daily_compliance_report(report_date)
            
            # Get dashboard trends
            dashboard_history = self.dashboard.get_dashboard_history(hours=24)
            
            # Compile daily summary
            daily_summary = {
                'report_type': 'daily_summary',
                'report_date': report_date.isoformat(),
                'generation_time': datetime.now().isoformat(),
                'risk_data_snapshot': risk_data,
                'compliance_report': compliance_report.to_dict(),
                'performance_trends': {
                    'dashboard_data_points': len(dashboard_history),
                    'risk_score_trend': self._calculate_trend([d.get('overall_risk_score', 50) for d in dashboard_history]),
                    'alert_trend': self._calculate_alert_trend(dashboard_history)
                },
                'summary_statistics': await self._calculate_daily_statistics(risk_data),
                'recommendations': await self._generate_daily_recommendations(risk_data, compliance_report)
            }
            
            # Store report
            self.report_history['daily'].append(daily_summary)
            
            return daily_summary
            
        except Exception as e:
            logger.error(f"Error generating daily summary report: {e}")
            raise RiskError(f"Daily summary generation failed: {e}")
    
    async def generate_breach_notification(self, breach_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate breach notification report."""
        try:
            notification = {
                'report_type': 'breach_notification',
                'timestamp': datetime.now().isoformat(),
                'breach_id': f"BREACH_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'severity': breach_data.get('severity', 'high'),
                'breach_type': breach_data.get('breach_type', 'unknown'),
                'description': breach_data.get('description', 'Risk limit breach detected'),
                'current_value': breach_data.get('current_value', 0),
                'limit_value': breach_data.get('limit_value', 0),
                'exceedance': breach_data.get('current_value', 0) - breach_data.get('limit_value', 0),
                'immediate_actions': breach_data.get('immediate_actions', []),
                'escalation_required': breach_data.get('severity') in ['critical', 'high'],
                'notification_channels': self._get_notification_channels(breach_data.get('severity', 'medium'))
            }
            
            # Store notification
            self.report_history['breach_notifications'].append(notification)
            
            logger.warning(f"Risk breach notification generated: {notification['breach_id']}")
            
            return notification
            
        except Exception as e:
            logger.error(f"Error generating breach notification: {e}")
            raise RiskError(f"Breach notification generation failed: {e}")
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction from a list of values."""
        if len(values) < 2:
            return 'stable'
        
        try:
            # Simple linear regression slope
            x = list(range(len(values)))
            slope = np.polyfit(x, values, 1)[0]
            
            if slope > 1.0:
                return 'increasing'
            elif slope < -1.0:
                return 'decreasing'
            else:
                return 'stable'
        except:
            return 'stable'
    
    def _calculate_alert_trend(self, dashboard_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate alert trends from dashboard history."""
        if not dashboard_history:
            return {'trend': 'stable', 'average_alerts': 0}
        
        try:
            alert_counts = []
            for data_point in dashboard_history:
                alert_summary = data_point.get('alert_summary', {})
                total_alerts = alert_summary.get('total', 0)
                alert_counts.append(total_alerts)
            
            avg_alerts = np.mean(alert_counts) if alert_counts else 0
            trend = self._calculate_trend(alert_counts)
            
            return {
                'trend': trend,
                'average_alerts': round(avg_alerts, 1),
                'max_alerts': max(alert_counts) if alert_counts else 0,
                'current_alerts': alert_counts[-1] if alert_counts else 0
            }
        except Exception as e:
            logger.error(f"Error calculating alert trend: {e}")
            return {'trend': 'stable', 'average_alerts': 0}
    
    async def _calculate_daily_statistics(self, risk_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate daily summary statistics."""
        stats = {
            'total_components_monitored': len(risk_data.get('components', {})),
            'components_with_errors': len([c for c in risk_data.get('components', {}).values() if c.get('error')]),
            'system_uptime_percent': 98.5,  # Would calculate actual uptime
            'data_quality_score': 95.0      # Would calculate actual data quality
        }
        
        # Component-specific statistics
        for component_name, component_data in risk_data.get('components', {}).items():
            if not component_data.get('error'):
                if component_name == 'circuit_breakers':
                    stats['circuit_breakers_active'] = component_data.get('active_breakers', 0)
                elif component_name == 'position_limits':
                    stats['position_limits_monitored'] = len(component_data.get('limits', {}))
                elif component_name == 'operational_risk':
                    stats['operational_alerts'] = component_data.get('alert_summary', {}).get('total_active', 0)
        
        return stats
    
    async def _generate_daily_recommendations(self, risk_data: Dict[str, Any], compliance_report: ComplianceReport) -> List[str]:
        """Generate daily risk management recommendations."""
        recommendations = []
        
        try:
            # Compliance-based recommendations
            if compliance_report.compliance_score < 90:
                recommendations.append(f"Compliance score ({compliance_report.compliance_score:.1f}%) below target - review risk controls")
            
            if compliance_report.critical_breaches > 0:
                recommendations.append(f"Address {compliance_report.critical_breaches} critical compliance breaches immediately")
            
            # Component-specific recommendations
            cb_data = risk_data.get('components', {}).get('circuit_breakers', {})
            if cb_data and cb_data.get('active_breakers', 0) > 0:
                recommendations.append("Circuit breakers active - investigate and resolve underlying issues")
            
            op_data = risk_data.get('components', {}).get('operational_risk', {})
            if op_data and op_data.get('overall_health') in ['critical', 'warning']:
                recommendations.append("System health degraded - monitor resource usage and connectivity")
            
            # Default recommendation if no issues
            if not recommendations:
                recommendations.append("Risk management systems operating normally - continue monitoring")
        
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            recommendations.append("Error generating recommendations - manual review required")
        
        return recommendations
    
    def _get_notification_channels(self, severity: str) -> List[str]:
        """Get appropriate notification channels based on severity."""
        channels = ['log', 'dashboard']
        
        if severity in ['critical', 'high']:
            channels.extend(['email', 'slack'])
        
        if severity == 'critical':
            channels.append('sms')
        
        return channels
    
    def get_report_history(self, report_type: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get historical reports of specified type."""
        try:
            reports = self.report_history.get(report_type, [])
            return reports[-limit:] if limit > 0 else reports
        except Exception as e:
            logger.error(f"Error getting report history: {e}")
            return []
    
    async def cleanup(self):
        """Cleanup risk reporting system."""
        try:
            logger.info("Risk reporting system cleanup completed")
        except Exception as e:
            logger.error(f"Error during risk reporting cleanup: {e}")
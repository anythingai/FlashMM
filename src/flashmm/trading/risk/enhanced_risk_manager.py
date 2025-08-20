"""
Enhanced FlashMM Risk Manager

Enterprise-grade risk management system that integrates comprehensive risk monitoring
with the existing MarketMakingEngine. Provides seamless upgrade from basic risk controls
to full enterprise risk management while maintaining compatibility.
"""

import asyncio
from typing import Dict, Any, Optional, List, Callable
from decimal import Decimal
from datetime import datetime, timedelta
from dataclasses import dataclass

from flashmm.config.settings import get_config
from flashmm.utils.logging import get_logger, TradingEventLogger
from flashmm.utils.exceptions import RiskError, CircuitBreakerError, EmergencyStopError

# Import existing components
from flashmm.trading.risk.position_tracker import PositionTracker

# Import our comprehensive risk management system
from flashmm.risk import (
    CircuitBreakerSystem,
    PositionLimitsManager,
    MarketRiskMonitor,
    PnLRiskController,
    OperationalRiskManager,
    RiskReporter,
    EmergencyProtocolManager,
    EmergencyLevel,
    EmergencyType
)

logger = get_logger(__name__)
trading_logger = TradingEventLogger()


@dataclass
class RiskCheckResult:
    """Result of risk check with detailed analysis."""
    allowed: bool
    risk_level: str
    violations: List[str]
    warnings: List[str]
    metrics: Dict[str, Any]
    emergency_triggered: bool = False
    circuit_breakers_active: int = 0


class EnhancedRiskManager:
    """
    Enhanced risk manager that integrates enterprise risk management
    with existing trading engine for seamless compatibility.
    """
    
    def __init__(self):
        self.config = get_config()
        
        # Existing compatibility components
        self.position_tracker: Optional[PositionTracker] = None
        
        # Enterprise risk management components
        self.circuit_breakers: Optional[CircuitBreakerSystem] = None
        self.position_limits: Optional[PositionLimitsManager] = None
        self.market_monitor: Optional[MarketRiskMonitor] = None
        self.pnl_controller: Optional[PnLRiskController] = None
        self.operational_monitor: Optional[OperationalRiskManager] = None
        self.risk_reporter: Optional[RiskReporter] = None
        self.emergency_protocols: Optional[EmergencyProtocolManager] = None
        
        # Configuration
        self.enterprise_mode_enabled = self.config.get("risk.enterprise_mode", True)
        self.legacy_compatibility = self.config.get("risk.legacy_compatibility", True)
        
        # Legacy risk limits (for backward compatibility)
        self.max_position_usdc = self.config.get("trading.max_position_usdc", 2000.0)
        self.max_daily_volume = self.config.get("risk.max_daily_volume_usdc", 100000.0)
        self.circuit_breaker_loss_percent = self.config.get("risk.circuit_breaker_loss_percent", 10.0)
        
        # Legacy state tracking
        self.circuit_breaker_active = False
        self.daily_pnl = 0.0
        self.daily_volume = 0.0
        
        # Integration state
        self.initialized = False
        self.monitoring_active = False
        self.last_risk_check = datetime.now()
        
        # Callbacks for trading engine integration
        self.emergency_stop_callback: Optional[Callable] = None
        self.pause_trading_callback: Optional[Callable] = None
        self.cancel_orders_callback: Optional[Callable] = None
        
    async def initialize(self) -> None:
        """Initialize enhanced risk manager with all components."""
        try:
            logger.info("Initializing EnhancedRiskManager...")
            
            # Initialize position tracker (existing component)
            self.position_tracker = PositionTracker()
            await self.position_tracker.initialize()
            
            if self.enterprise_mode_enabled:
                # Initialize enterprise risk components
                await self._initialize_enterprise_components()
                
                # Setup component integration
                await self._setup_component_integration()
                
                # Setup trading engine callbacks
                await self._setup_trading_engine_integration()
                
                logger.info("Enterprise risk management system activated")
            else:
                logger.info("Running in legacy compatibility mode")
            
            self.initialized = True
            logger.info("EnhancedRiskManager initialization completed")
            
        except Exception as e:
            logger.error(f"Failed to initialize EnhancedRiskManager: {e}")
            raise RiskError(f"EnhancedRiskManager initialization failed: {e}")
    
    async def _initialize_enterprise_components(self):
        """Initialize all enterprise risk management components."""
        # Circuit breakers
        self.circuit_breakers = CircuitBreakerSystem()
        await self.circuit_breakers.initialize()
        
        # Position limits
        self.position_limits = PositionLimitsManager()
        await self.position_limits.initialize()
        
        # Market risk monitor
        self.market_monitor = MarketRiskMonitor()
        await self.market_monitor.initialize()
        
        # P&L controller
        self.pnl_controller = PnLRiskController()
        await self.pnl_controller.initialize()
        
        # Operational monitor
        self.operational_monitor = OperationalRiskManager()
        await self.operational_monitor.initialize()
        
        # Risk reporter
        self.risk_reporter = RiskReporter()
        await self.risk_reporter.initialize()
        
        # Emergency protocols
        self.emergency_protocols = EmergencyProtocolManager()
        await self.emergency_protocols.initialize()
    
    async def _setup_component_integration(self):
        """Setup integration between risk components."""
        if not self.enterprise_mode_enabled:
            return
        
        # Circuit breakers trigger emergency protocols
        async def circuit_breaker_alert(breaker_name, reason):
            if self.emergency_protocols:
                await self.emergency_protocols.check_emergency_conditions({
                    'components': {
                        'circuit_breakers': {
                            'triggered_breaker': breaker_name,
                            'reason': reason,
                            'active_breakers': 1
                        }
                    }
                })
        
        if self.circuit_breakers:
            self.circuit_breakers.set_emergency_callback(circuit_breaker_alert)
        
        # Market monitor alerts trigger circuit breakers
        async def market_alert_handler(alert_data):
            if alert_data.get('severity') == 'critical' and self.circuit_breakers:
                await self.circuit_breakers.emergency_stop("Critical market conditions detected")
        
        if self.market_monitor:
            self.market_monitor.set_alert_callback(market_alert_handler)
        
        # P&L controller triggers position limit adjustments
        async def pnl_alert_handler(pnl_alert):
            if (pnl_alert.get('drawdown_pct', 0) > 10 and 
                self.position_limits):
                await self.position_limits.emergency_limit_reduction(
                    0.5, "High drawdown detected"
                )
        
        if self.pnl_controller:
            self.pnl_controller.set_alert_callback(pnl_alert_handler)
    
    async def _setup_trading_engine_integration(self):
        """Setup integration with trading engine callbacks."""
        if not self.enterprise_mode_enabled:
            return
        
        # Setup emergency protocols callbacks
        if self.emergency_protocols:
            self.emergency_protocols.set_callbacks(
                halt_trading=self._handle_emergency_halt,
                cancel_orders=self._handle_emergency_cancel_orders,
                place_order=self._handle_emergency_place_order,
                get_positions=self._handle_get_positions,
                notify_stakeholders=self._handle_notify_stakeholders
            )
    
    def set_trading_engine_callbacks(self,
                                   emergency_stop: Optional[Callable] = None,
                                   pause_trading: Optional[Callable] = None,
                                   cancel_orders: Optional[Callable] = None):
        """Set callbacks to trading engine for emergency actions."""
        self.emergency_stop_callback = emergency_stop
        self.pause_trading_callback = pause_trading
        self.cancel_orders_callback = cancel_orders
    
    async def _handle_emergency_halt(self, reason: str):
        """Handle emergency halt trading."""
        logger.critical(f"EMERGENCY HALT: {reason}")
        if self.emergency_stop_callback:
            await self.emergency_stop_callback(reason)
    
    async def _handle_emergency_cancel_orders(self, reason: str):
        """Handle emergency order cancellation."""
        logger.warning(f"Emergency order cancellation: {reason}")
        if self.cancel_orders_callback:
            result = await self.cancel_orders_callback(reason)
            return result if result else {'cancelled_count': 0}
        return {'cancelled_count': 0}
    
    async def _handle_emergency_place_order(self, order_data: Dict[str, Any]):
        """Handle emergency order placement (e.g., for hedging)."""
        logger.info(f"Emergency order placement: {order_data}")
        # Would integrate with order router
        return {'status': 'simulated', 'order_id': 'emergency_001'}
    
    async def _handle_get_positions(self):
        """Get current positions for emergency protocols."""
        if self.position_tracker:
            try:
                all_positions = await self.position_tracker.get_all_positions()
                return [
                    {
                        'symbol': symbol,
                        'base_balance': pos.get('base_balance', 0),
                        'notional_value': pos.get('value_usdc', 0),
                        'sector': 'crypto' if 'USD' in symbol else 'unknown'
                    }
                    for symbol, pos in all_positions.items()
                ]
            except Exception as e:
                logger.error(f"Error getting positions: {e}")
        return []
    
    async def _handle_notify_stakeholders(self, alert_data: Dict[str, Any]):
        """Handle stakeholder notification."""
        logger.warning(f"Stakeholder notification: {alert_data}")
        # Would integrate with notification system
    
    # Main interface methods (compatible with existing RiskManager)
    async def check_trading_allowed(self, symbol: str) -> bool:
        """
        Enhanced trading permission check with comprehensive risk analysis.
        Maintains compatibility with existing interface.
        """
        try:
            self.last_risk_check = datetime.now()
            
            # Perform comprehensive risk check
            risk_result = await self.perform_comprehensive_risk_check(symbol)
            
            # Legacy compatibility: Update legacy state
            if self.legacy_compatibility:
                self.circuit_breaker_active = risk_result.emergency_triggered
            
            return risk_result.allowed
            
        except Exception as e:
            logger.error(f"Risk check failed for {symbol}: {e}")
            return False
    
    async def perform_comprehensive_risk_check(self, symbol: str) -> RiskCheckResult:
        """Perform comprehensive risk analysis across all components."""
        violations = []
        warnings = []
        metrics = {}
        emergency_triggered = False
        circuit_breakers_active = 0
        
        try:
            if self.enterprise_mode_enabled:
                # Get current market data for analysis
                market_data = await self._get_current_market_data(symbol)
                
                # Circuit breaker checks
                if self.circuit_breakers:
                    triggered_breakers = await self.circuit_breakers.check_all_breakers(market_data)
                    circuit_breakers_active = len(triggered_breakers)
                    
                    if triggered_breakers:
                        violations.extend([f"Circuit breaker active: {cb}" for cb in triggered_breakers])
                
                # Position limit checks
                if self.position_limits:
                    position_check = await self._check_enhanced_position_limits(symbol)
                    if not position_check['allowed']:
                        violations.extend(position_check['violations'])
                
                # Market risk checks
                if self.market_monitor:
                    market_analysis = await self.market_monitor.analyze_market_conditions(market_data)
                    if market_analysis.get('overall_risk_level') in ['high', 'critical']:
                        warnings.append(f"High market risk: {market_analysis.get('overall_risk_level')}")
                
                # P&L risk checks
                if self.pnl_controller:
                    pnl_data = await self._get_current_pnl_data()
                    pnl_analysis = await self.pnl_controller.assess_pnl_risk(pnl_data)
                    if pnl_analysis.get('overall_risk_level') in ['high', 'critical']:
                        violations.append(f"P&L risk: {pnl_analysis.get('overall_risk_level')}")
                
                # Operational risk checks
                if self.operational_monitor:
                    ops_health = await self.operational_monitor.perform_health_check()
                    if ops_health.get('overall_score', 100) < 50:
                        warnings.append("Poor system health")
                
                # Emergency protocol checks
                if self.emergency_protocols:
                    emergency_conditions = {
                        'components': {
                            'circuit_breakers': {'active_breakers': circuit_breakers_active},
                            'market_risk': market_analysis if 'market_analysis' in locals() else {},
                            'pnl_risk': pnl_analysis if 'pnl_analysis' in locals() else {},
                            'operational_risk': ops_health if 'ops_health' in locals() else {}
                        }
                    }
                    
                    triggered_protocols = await self.emergency_protocols.check_emergency_conditions(
                        emergency_conditions
                    )
                    
                    if triggered_protocols:
                        emergency_triggered = True
                        violations.append(f"Emergency protocols triggered: {triggered_protocols}")
                
                metrics = {
                    'circuit_breakers_active': circuit_breakers_active,
                    'market_risk_level': market_analysis.get('overall_risk_level', 'unknown') if 'market_analysis' in locals() else 'unknown',
                    'pnl_risk_level': pnl_analysis.get('overall_risk_level', 'unknown') if 'pnl_analysis' in locals() else 'unknown',
                    'system_health_score': ops_health.get('overall_score', 0) if 'ops_health' in locals() else 0,
                    'emergency_protocols_active': len(triggered_protocols) if 'triggered_protocols' in locals() else 0
                }
            
            else:
                # Legacy risk checks for backward compatibility
                legacy_result = await self._perform_legacy_risk_checks(symbol)
                if not legacy_result:
                    violations.append("Legacy risk check failed")
            
            # Determine overall risk level
            if emergency_triggered or circuit_breakers_active > 2:
                risk_level = 'critical'
            elif len(violations) > 0:
                risk_level = 'high'
            elif len(warnings) > 0:
                risk_level = 'medium'
            else:
                risk_level = 'normal'
            
            # Trading allowed if no violations and no emergency
            allowed = len(violations) == 0 and not emergency_triggered
            
            return RiskCheckResult(
                allowed=allowed,
                risk_level=risk_level,
                violations=violations,
                warnings=warnings,
                metrics=metrics,
                emergency_triggered=emergency_triggered,
                circuit_breakers_active=circuit_breakers_active
            )
            
        except Exception as e:
            logger.error(f"Comprehensive risk check failed: {e}")
            return RiskCheckResult(
                allowed=False,
                risk_level='error',
                violations=[f"Risk check error: {e}"],
                warnings=[],
                metrics={}
            )
    
    async def _perform_legacy_risk_checks(self, symbol: str) -> bool:
        """Perform legacy risk checks for backward compatibility."""
        try:
            # Check circuit breaker
            if self.circuit_breaker_active:
                return False
            
            # Check position limits
            if not await self._check_position_limits(symbol):
                return False
            
            # Check daily volume limits
            if not await self._check_volume_limits():
                return False
            
            # Check P&L limits
            if not await self._check_pnl_limits():
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Legacy risk check failed: {e}")
            return False
    
    async def _check_enhanced_position_limits(self, symbol: str) -> Dict[str, Any]:
        """Check enhanced position limits."""
        if not self.position_limits:
            return {'allowed': True, 'violations': []}
        
        try:
            # Get current positions
            positions = await self._handle_get_positions()
            portfolio_value = Decimal('100000')  # Would get from position tracker
            
            # Check portfolio limits
            limit_check = await self.position_limits.check_portfolio_limits(
                positions, portfolio_value
            )
            
            violations = []
            if limit_check.get('position_limit_violations'):
                violations.extend(limit_check['position_limit_violations'])
            
            if limit_check.get('concentration_violations'):
                violations.extend([f"Concentration: {v}" for v in limit_check['concentration_violations']])
            
            return {
                'allowed': len(violations) == 0,
                'violations': violations
            }
            
        except Exception as e:
            logger.error(f"Enhanced position limit check failed: {e}")
            return {'allowed': False, 'violations': [f"Position check error: {e}"]}
    
    async def _get_current_market_data(self, symbol: str) -> Dict[str, Any]:
        """Get current market data for risk analysis."""
        # This would integrate with market data feeds
        # For now, return simulated data structure
        return {
            'price_data': {
                symbol: {
                    'current_price': 50000.0,
                    'previous_price': 49500.0,
                    'timestamp': datetime.now()
                }
            },
            'volume_data': {
                symbol: {
                    'volume': 1000000.0,
                    'timestamp': datetime.now()
                }
            }
        }
    
    async def _get_current_pnl_data(self) -> Dict[str, Any]:
        """Get current P&L data for risk analysis."""
        try:
            portfolio_summary = {}
            if self.position_tracker:
                portfolio_summary = self.position_tracker.get_portfolio_summary()
            
            return {
                'total_pnl': portfolio_summary.get('total_pnl', 0.0),
                'daily_pnl': self.daily_pnl,
                'unrealized_pnl': portfolio_summary.get('total_unrealized_pnl', 0.0),
                'realized_pnl': portfolio_summary.get('total_realized_pnl', 0.0),
                'positions': await self._handle_get_positions(),
                'timestamp': datetime.now()
            }
        except Exception as e:
            logger.error(f"Error getting P&L data: {e}")
            return {
                'total_pnl': 0.0,
                'daily_pnl': 0.0,
                'positions': [],
                'timestamp': datetime.now()
            }
    
    # Legacy compatibility methods
    async def _check_position_limits(self, symbol: str) -> bool:
        """Legacy position limit check."""
        if not self.position_tracker:
            return True
        
        try:
            position = await self.position_tracker.get_position(symbol)
            position_value = abs(position.get("value_usdc", 0.0))
            
            if position_value > self.max_position_usdc:
                logger.warning(f"Position limit exceeded for {symbol}: {position_value}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Position limit check failed: {e}")
            return False
    
    async def _check_volume_limits(self) -> bool:
        """Legacy volume limit check."""
        if self.daily_volume > self.max_daily_volume:
            logger.warning(f"Daily volume limit exceeded: {self.daily_volume}")
            return False
        return True
    
    async def _check_pnl_limits(self) -> bool:
        """Legacy P&L limit check."""
        loss_percent = abs(self.daily_pnl / self.max_position_usdc) * 100
        
        if loss_percent > self.circuit_breaker_loss_percent:
            await self._trigger_circuit_breaker("pnl_limit_exceeded")
            return False
        
        return True
    
    async def _trigger_circuit_breaker(self, reason: str) -> None:
        """Legacy circuit breaker trigger."""
        self.circuit_breaker_active = True
        
        logger.critical(f"Circuit breaker triggered: {reason}")
        await trading_logger.log_pnl_event(
            "ALL",
            0.0,
            self.daily_pnl,
            self.daily_pnl,
            circuit_breaker_reason=reason
        )
        
        # Trigger emergency protocols if available
        if self.enterprise_mode_enabled and self.emergency_protocols:
            await self.emergency_protocols.manual_emergency_stop(
                f"Circuit breaker: {reason}",
                EmergencyType.EXCESSIVE_LOSSES
            )
        
        raise CircuitBreakerError(
            f"Trading halted: {reason}",
            breaker_name="main_circuit_breaker",
            trigger_reason=reason
        )
    
    # Position and trade management (compatible interface)
    async def update_position(self, symbol: str, side: str, size: float, price: float) -> None:
        """Update position after trade execution (compatible interface)."""
        if not self.position_tracker:
            return
        
        try:
            await self.position_tracker.update_position(symbol, side, size, price)
            
            # Update daily volume
            trade_value = size * price
            self.daily_volume += trade_value
            
            # Update enterprise components if enabled
            if self.enterprise_mode_enabled:
                await self._update_enterprise_components_after_trade(symbol, side, size, price)
            
        except Exception as e:
            logger.error(f"Failed to update position: {e}")
    
    async def _update_enterprise_components_after_trade(self, symbol: str, side: str, size: float, price: float):
        """Update enterprise risk components after trade execution."""
        try:
            # Update P&L controller
            if self.pnl_controller:
                pnl_data = await self._get_current_pnl_data()
                await self.pnl_controller.update_pnl(pnl_data)
            
            # Update market monitor with trade data
            if self.market_monitor:
                trade_data = {
                    symbol: {
                        'price': price,
                        'volume': size * price,
                        'timestamp': datetime.now()
                    }
                }
                await self.market_monitor.add_market_data(trade_data)
            
            # Update position limits
            if self.position_limits:
                positions = await self._handle_get_positions()
                portfolio_value = Decimal('100000')  # Would calculate actual value
                await self.position_limits.check_portfolio_limits(positions, portfolio_value)
            
        except Exception as e:
            logger.error(f"Error updating enterprise components after trade: {e}")
    
    # Metrics and reporting (compatible interface)
    async def get_risk_metrics(self) -> Dict[str, Any]:
        """Get comprehensive risk metrics (enhanced compatible interface)."""
        base_metrics = {
            "circuit_breaker_active": self.circuit_breaker_active,
            "daily_pnl": self.daily_pnl,
            "daily_volume": self.daily_volume,
            "max_position_usdc": self.max_position_usdc,
            "max_daily_volume": self.max_daily_volume,
            "positions": {},
            "enterprise_mode_enabled": self.enterprise_mode_enabled,
            "last_risk_check": self.last_risk_check.isoformat()
        }
        
        # Add positions from position tracker
        if self.position_tracker:
            try:
                base_metrics["positions"] = await self.position_tracker.get_all_positions()
            except Exception as e:
                logger.error(f"Failed to get positions: {e}")
        
        # Add enterprise risk metrics if enabled
        if self.enterprise_mode_enabled:
            try:
                enterprise_metrics = await self._get_enterprise_risk_metrics()
                base_metrics.update(enterprise_metrics)
            except Exception as e:
                logger.error(f"Failed to get enterprise metrics: {e}")
        
        return base_metrics
    
    async def _get_enterprise_risk_metrics(self) -> Dict[str, Any]:
        """Get metrics from all enterprise risk components."""
        metrics = {}
        
        try:
            # Circuit breaker metrics
            if self.circuit_breakers:
                cb_status = self.circuit_breakers.get_system_status()
                metrics['circuit_breakers'] = cb_status
            
            # Position limits metrics
            if self.position_limits:
                limits = await self.position_limits.get_current_limits()
                metrics['position_limits'] = {
                    'current_limits': {k: float(v) for k, v in limits.items()},
                    'emergency_mode': hasattr(self.position_limits, 'emergency_mode')
                }
            
            # Market risk metrics
            if self.market_monitor:
                # Would get current market analysis
                metrics['market_risk'] = {'status': 'monitoring'}
            
            # Operational risk metrics
            if self.operational_monitor:
                ops_status = await self.operational_monitor.emergency_health_check()
                metrics['operational_risk'] = ops_status
            
            # Emergency protocol status
            if self.emergency_protocols:
                emergency_status = self.emergency_protocols.get_emergency_status()
                metrics['emergency_protocols'] = emergency_status
            
        except Exception as e:
            logger.error(f"Error collecting enterprise metrics: {e}")
        
        return {'enterprise_metrics': metrics}
    
    # Legacy compatibility methods
    async def reset_daily_metrics(self) -> None:
        """Reset daily P&L and volume metrics (compatible interface)."""
        self.daily_pnl = 0.0
        self.daily_volume = 0.0
        logger.info("Daily risk metrics reset")
    
    async def reset_circuit_breaker(self) -> None:
        """Reset circuit breaker (compatible interface)."""
        self.circuit_breaker_active = False
        
        # Reset enterprise circuit breakers if available
        if self.enterprise_mode_enabled and self.circuit_breakers:
            # Would implement circuit breaker reset
            pass
        
        logger.warning("Circuit breaker manually reset")
    
    # Monitoring and reporting
    async def start_monitoring(self) -> None:
        """Start continuous risk monitoring."""
        if not self.enterprise_mode_enabled:
            return
        
        self.monitoring_active = True
        logger.info("Enterprise risk monitoring started")
        
        # Start individual component monitoring
        components = [
            self.operational_monitor,
            self.market_monitor,
            self.pnl_controller
        ]
        
        for component in components:
            if component and hasattr(component, 'start_monitoring'):
                try:
                    await component.start_monitoring()
                except Exception as e:
                    logger.error(f"Failed to start monitoring for {component}: {e}")
    
    async def stop_monitoring(self) -> None:
        """Stop continuous risk monitoring."""
        self.monitoring_active = False
        logger.info("Enterprise risk monitoring stopped")
    
    async def generate_risk_report(self) -> Dict[str, Any]:
        """Generate comprehensive risk report."""
        if not self.enterprise_mode_enabled or not self.risk_reporter:
            return await self.get_risk_metrics()
        
        try:
            # Generate comprehensive report
            risk_data = await self.get_risk_metrics()
            report = await self.risk_reporter.generate_comprehensive_report(risk_data)
            return report
        except Exception as e:
            logger.error(f"Error generating risk report: {e}")
            return await self.get_risk_metrics()
    
    async def cleanup(self) -> None:
        """Cleanup resources."""
        logger.info("Cleaning up EnhancedRiskManager...")
        
        try:
            await self.stop_monitoring()
            
            # Cleanup enterprise components
            if self.enterprise_mode_enabled:
                components = [
                    self.circuit_breakers,
                    self.position_limits,
                    self.market_monitor,
                    self.pnl_controller,
                    self.operational_monitor,
                    self.risk_reporter,
                    self.emergency_protocols
                ]
                
                for component in components:
                    if component and hasattr(component, 'cleanup'):
                        try:
                            await component.cleanup()
                        except Exception as e:
                            logger.error(f"Error cleaning up {component}: {e}")
            
            # Cleanup position tracker
            if self.position_tracker:
                await self.position_tracker.cleanup()
            
            logger.info("EnhancedRiskManager cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


# Factory function for backward compatibility
async def create_risk_manager(enterprise_mode: bool = True) -> EnhancedRiskManager:
    """Create and initialize enhanced risk manager."""
    risk_manager = EnhancedRiskManager()
    risk_manager.enterprise_mode_enabled = enterprise_mode
    await risk_manager.initialize()
    return risk_manager
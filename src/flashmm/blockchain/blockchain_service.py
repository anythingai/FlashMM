"""
Blockchain Service

Main service orchestrating all blockchain interactions, integrating with
existing trading engine and providing comprehensive blockchain functionality.
"""

import asyncio
from typing import Dict, Any, Optional, List, Callable
from decimal import Decimal
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from flashmm.config.settings import get_config
from flashmm.utils.logging import get_logger, TradingEventLogger
from flashmm.utils.exceptions import BlockchainError, ValidationError, ServiceError
from flashmm.utils.decorators import timeout_async, measure_latency, retry_async
from flashmm.trading.execution.order_router import Order, OrderStatus
from flashmm.blockchain.sei_client import SeiClient
from flashmm.blockchain.market_config import MarketConfigManager
from flashmm.blockchain.order_manager import BlockchainOrderManager
from flashmm.blockchain.transaction_manager import TransactionManager, BatchTransactionType, TransactionPriority
from flashmm.blockchain.account_manager import AccountManager, AccountType
from flashmm.blockchain.blockchain_monitor import BlockchainMonitor, AlertSeverity

logger = get_logger(__name__)
trading_logger = TradingEventLogger()


class ServiceStatus(Enum):
    """Blockchain service status."""
    INITIALIZING = "initializing"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    EMERGENCY = "emergency"
    SHUTDOWN = "shutdown"


@dataclass
class ServiceHealthCheck:
    """Service health check result."""
    status: ServiceStatus
    timestamp: datetime
    components: Dict[str, bool]
    error_count: int
    last_error: Optional[str]
    performance_metrics: Dict[str, float]


class BlockchainService:
    """Main blockchain service orchestrating all blockchain operations."""
    
    def __init__(self):
        self.config = get_config()
        
        # Service status
        self.status = ServiceStatus.INITIALIZING
        self.initialization_time: Optional[datetime] = None
        self.last_health_check: Optional[ServiceHealthCheck] = None
        
        # Core components
        self.sei_client: Optional[SeiClient] = None
        self.market_config_manager: Optional[MarketConfigManager] = None
        self.order_manager: Optional[BlockchainOrderManager] = None
        self.transaction_manager: Optional[TransactionManager] = None
        self.account_manager: Optional[AccountManager] = None
        self.blockchain_monitor: Optional[BlockchainMonitor] = None
        
        # Configuration
        self.enabled = self.config.get("blockchain.enabled", True)
        self.auto_recovery = self.config.get("blockchain.auto_recovery", True)
        self.health_check_interval = self.config.get("blockchain.health_check_interval_seconds", 60)
        self.emergency_stop_enabled = self.config.get("blockchain.emergency_stop_enabled", True)
        
        # Performance tracking
        self.performance_stats = {
            'orders_submitted': 0,
            'orders_successful': 0,
            'orders_failed': 0,
            'avg_latency_ms': 0.0,
            'uptime_percentage': 100.0,
            'last_restart': None
        }
        
        # Error tracking
        self.error_count = 0
        self.last_error: Optional[str] = None
        self.error_history: List[Dict[str, Any]] = []
        
        # Background tasks
        self._health_check_task: Optional[asyncio.Task] = None
        self._recovery_task: Optional[asyncio.Task] = None
        
        # Event callbacks
        self.status_change_callbacks: List[Callable[[ServiceStatus], None]] = []
        self.error_callbacks: List[Callable[[str, Exception], None]] = []
        
    async def initialize(self) -> None:
        """Initialize the blockchain service and all components."""
        try:
            logger.info("Initializing blockchain service")
            self.status = ServiceStatus.INITIALIZING
            
            if not self.enabled:
                logger.info("Blockchain service disabled by configuration")
                self.status = ServiceStatus.SHUTDOWN
                return
            
            # Initialize components in order
            await self._initialize_sei_client()
            await self._initialize_market_config_manager()
            await self._initialize_transaction_manager()
            await self._initialize_account_manager()
            await self._initialize_order_manager()
            await self._initialize_blockchain_monitor()
            
            # Setup inter-component connections
            await self._setup_component_connections()
            
            # Start background tasks
            await self._start_background_tasks()
            
            # Perform initial health check
            health_check = await self._perform_health_check()
            
            # Set service status based on health check
            self.status = health_check.status
            self.initialization_time = datetime.now()
            
            # Notify status change
            await self._notify_status_change(self.status)
            
            logger.info(f"Blockchain service initialized successfully - Status: {self.status.value}")
            
        except Exception as e:
            self.status = ServiceStatus.UNHEALTHY
            self.last_error = str(e)
            self.error_count += 1
            
            logger.error(f"Failed to initialize blockchain service: {e}")
            await self._notify_error("initialization_failed", e)
            raise ServiceError(f"Blockchain service initialization failed: {e}")
    
    async def _initialize_sei_client(self) -> None:
        """Initialize Sei blockchain client."""
        try:
            logger.debug("Initializing Sei client")
            self.sei_client = SeiClient()
            await self.sei_client.initialize()
            logger.debug("Sei client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Sei client: {e}")
            raise BlockchainError(f"Sei client initialization failed: {e}")
    
    async def _initialize_market_config_manager(self) -> None:
        """Initialize market configuration manager."""
        try:
            logger.debug("Initializing market configuration manager")
            self.market_config_manager = MarketConfigManager(self.sei_client)
            await self.market_config_manager.initialize()
            logger.debug("Market configuration manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize market config manager: {e}")
            raise BlockchainError(f"Market config manager initialization failed: {e}")
    
    async def _initialize_transaction_manager(self) -> None:
        """Initialize transaction manager."""
        try:
            logger.debug("Initializing transaction manager")
            self.transaction_manager = TransactionManager(self.sei_client)
            await self.transaction_manager.initialize()
            logger.debug("Transaction manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize transaction manager: {e}")
            raise BlockchainError(f"Transaction manager initialization failed: {e}")
    
    async def _initialize_account_manager(self) -> None:
        """Initialize account manager."""
        try:
            logger.debug("Initializing account manager")
            self.account_manager = AccountManager(self.sei_client)
            await self.account_manager.initialize()
            logger.debug("Account manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize account manager: {e}")
            raise BlockchainError(f"Account manager initialization failed: {e}")
    
    async def _initialize_order_manager(self) -> None:
        """Initialize blockchain order manager."""
        try:
            logger.debug("Initializing blockchain order manager")
            self.order_manager = BlockchainOrderManager(self.sei_client, self.market_config_manager)
            await self.order_manager.initialize()
            logger.debug("Blockchain order manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize order manager: {e}")
            raise BlockchainError(f"Order manager initialization failed: {e}")
    
    async def _initialize_blockchain_monitor(self) -> None:
        """Initialize blockchain monitor."""
        try:
            logger.debug("Initializing blockchain monitor")
            self.blockchain_monitor = BlockchainMonitor(self.sei_client)
            await self.blockchain_monitor.initialize()
            logger.debug("Blockchain monitor initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize blockchain monitor: {e}")
            raise BlockchainError(f"Blockchain monitor initialization failed: {e}")
    
    async def _setup_component_connections(self) -> None:
        """Setup connections between components."""
        try:
            # Setup blockchain monitor alerts
            if self.blockchain_monitor:
                self.blockchain_monitor.add_alert_callback(self._handle_blockchain_alert)
            
            logger.debug("Component connections setup completed")
            
        except Exception as e:
            logger.error(f"Failed to setup component connections: {e}")
            raise
    
    async def _start_background_tasks(self) -> None:
        """Start background maintenance tasks."""
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        
        if self.auto_recovery:
            self._recovery_task = asyncio.create_task(self._recovery_loop())
    
    async def _health_check_loop(self) -> None:
        """Background health check loop."""
        while True:
            try:
                await asyncio.sleep(self.health_check_interval)
                
                if self.status not in [ServiceStatus.SHUTDOWN, ServiceStatus.EMERGENCY]:
                    health_check = await self._perform_health_check()
                    
                    # Update service status if needed
                    if health_check.status != self.status:
                        old_status = self.status
                        self.status = health_check.status
                        logger.info(f"Service status changed: {old_status.value} -> {self.status.value}")
                        await self._notify_status_change(self.status)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check loop error: {e}")
                await asyncio.sleep(60)
    
    async def _recovery_loop(self) -> None:
        """Background recovery loop."""
        while True:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes
                
                if self.status in [ServiceStatus.UNHEALTHY, ServiceStatus.DEGRADED]:
                    await self._attempt_recovery()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Recovery loop error: {e}")
                await asyncio.sleep(300)
    
    @timeout_async(30.0)
    @measure_latency("blockchain_health_check")
    async def _perform_health_check(self) -> ServiceHealthCheck:
        """Perform comprehensive health check."""
        try:
            components = {}
            error_count = 0
            performance_metrics = {}
            
            # Check Sei client
            try:
                if self.sei_client:
                    network_health = await self.sei_client.check_network_health()
                    components["sei_client"] = network_health.is_healthy
                    performance_metrics["sei_rpc_latency_ms"] = network_health.rpc_latency_ms
                else:
                    components["sei_client"] = False
                    error_count += 1
            except Exception as e:
                components["sei_client"] = False
                error_count += 1
                logger.debug(f"Sei client health check failed: {e}")
            
            # Check market config manager
            try:
                if self.market_config_manager:
                    active_markets = len(self.market_config_manager.get_active_markets())
                    components["market_config"] = active_markets > 0
                    performance_metrics["active_markets"] = active_markets
                else:
                    components["market_config"] = False
                    error_count += 1
            except Exception as e:
                components["market_config"] = False
                error_count += 1
                logger.debug(f"Market config health check failed: {e}")
            
            # Check transaction manager
            try:
                if self.transaction_manager:
                    tx_stats = self.transaction_manager.get_performance_stats()
                    components["transaction_manager"] = True
                    performance_metrics["tx_success_rate"] = tx_stats.get("success_rate", 0.0)
                else:
                    components["transaction_manager"] = False
                    error_count += 1
            except Exception as e:
                components["transaction_manager"] = False
                error_count += 1
                logger.debug(f"Transaction manager health check failed: {e}")
            
            # Check account manager
            try:
                if self.account_manager:
                    account_summary = self.account_manager.get_account_summary()
                    components["account_manager"] = account_summary["active_accounts"] > 0
                    performance_metrics["active_accounts"] = account_summary["active_accounts"]
                else:
                    components["account_manager"] = False
                    error_count += 1
            except Exception as e:
                components["account_manager"] = False
                error_count += 1
                logger.debug(f"Account manager health check failed: {e}")
            
            # Check order manager
            try:
                if self.order_manager:
                    order_stats = self.order_manager.get_performance_stats()
                    components["order_manager"] = True
                    performance_metrics["order_success_rate"] = order_stats.get("success_rate", 0.0)
                else:
                    components["order_manager"] = False
                    error_count += 1
            except Exception as e:
                components["order_manager"] = False
                error_count += 1
                logger.debug(f"Order manager health check failed: {e}")
            
            # Check blockchain monitor
            try:
                if self.blockchain_monitor:
                    monitor_stats = self.blockchain_monitor.get_monitoring_stats()
                    components["blockchain_monitor"] = True
                    performance_metrics["active_alerts"] = monitor_stats.get("active_alerts_count", 0)
                else:
                    components["blockchain_monitor"] = False
                    error_count += 1
            except Exception as e:
                components["blockchain_monitor"] = False
                error_count += 1
                logger.debug(f"Blockchain monitor health check failed: {e}")
            
            # Determine overall status
            total_components = len(components)
            healthy_components = sum(1 for healthy in components.values() if healthy)
            
            if healthy_components == total_components:
                status = ServiceStatus.HEALTHY
            elif healthy_components >= total_components * 0.8:  # 80% healthy
                status = ServiceStatus.DEGRADED
            elif healthy_components >= total_components * 0.5:  # 50% healthy
                status = ServiceStatus.UNHEALTHY
            else:
                status = ServiceStatus.EMERGENCY
            
            health_check = ServiceHealthCheck(
                status=status,
                timestamp=datetime.now(),
                components=components,
                error_count=error_count,
                last_error=self.last_error,
                performance_metrics=performance_metrics
            )
            
            self.last_health_check = health_check
            return health_check
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return ServiceHealthCheck(
                status=ServiceStatus.UNHEALTHY,
                timestamp=datetime.now(),
                components={},
                error_count=1,
                last_error=str(e),
                performance_metrics={}
            )
    
    async def _attempt_recovery(self) -> None:
        """Attempt to recover from unhealthy state."""
        try:
            logger.info("Attempting service recovery")
            
            # Try to reinitialize failed components
            if self.last_health_check:
                for component, healthy in self.last_health_check.components.items():
                    if not healthy:
                        await self._recover_component(component)
            
            # Perform health check to verify recovery
            health_check = await self._perform_health_check()
            
            if health_check.status in [ServiceStatus.HEALTHY, ServiceStatus.DEGRADED]:
                logger.info(f"Service recovery successful - Status: {health_check.status.value}")
                self.performance_stats['last_restart'] = datetime.now()
            else:
                logger.warning(f"Service recovery partially successful - Status: {health_check.status.value}")
            
        except Exception as e:
            logger.error(f"Service recovery failed: {e}")
            await self._notify_error("recovery_failed", e)
    
    async def _recover_component(self, component: str) -> None:
        """Attempt to recover a specific component."""
        try:
            logger.debug(f"Recovering component: {component}")
            
            if component == "sei_client" and self.sei_client:
                await self.sei_client.initialize()
            elif component == "market_config" and self.market_config_manager:
                await self.market_config_manager.reload_market_config("SEI/USDC")
            elif component == "transaction_manager" and self.transaction_manager:
                # Transaction manager usually doesn't need recovery
                pass
            elif component == "account_manager" and self.account_manager:
                # Account manager usually doesn't need recovery
                pass
            elif component == "order_manager" and self.order_manager:
                # Order manager usually doesn't need recovery
                pass
            elif component == "blockchain_monitor" and self.blockchain_monitor:
                await self.blockchain_monitor.force_health_check()
            
            logger.debug(f"Component recovery completed: {component}")
            
        except Exception as e:
            logger.error(f"Failed to recover component {component}: {e}")
    
    # Public API methods for trading engine integration
    
    @timeout_async(10.0)
    async def submit_order_to_blockchain(self, order: Order) -> bool:
        """Submit order to blockchain via integrated order manager."""
        try:
            if not self._is_service_ready():
                raise ServiceError("Blockchain service not ready")
            
            # Submit via order manager
            success = await self.order_manager.submit_order_to_blockchain(order)
            
            # Update performance stats
            self.performance_stats['orders_submitted'] += 1
            if success:
                self.performance_stats['orders_successful'] += 1
            else:
                self.performance_stats['orders_failed'] += 1
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to submit order to blockchain: {e}")
            self.performance_stats['orders_failed'] += 1
            await self._handle_error("order_submission_failed", e)
            return False
    
    @timeout_async(5.0)
    async def cancel_order_on_blockchain(self, order_id: str) -> bool:
        """Cancel order on blockchain."""
        try:
            if not self._is_service_ready():
                raise ServiceError("Blockchain service not ready")
            
            return await self.order_manager.cancel_order_on_blockchain(order_id)
            
        except Exception as e:
            logger.error(f"Failed to cancel order on blockchain: {e}")
            await self._handle_error("order_cancellation_failed", e)
            return False
    
    def get_supported_markets(self) -> List[str]:
        """Get list of supported trading markets."""
        try:
            if self.market_config_manager:
                return list(self.market_config_manager.get_supported_symbols())
            return []
            
        except Exception as e:
            logger.error(f"Failed to get supported markets: {e}")
            return []
    
    def validate_order_for_blockchain(self, order: Order) -> Dict[str, Any]:
        """Validate order parameters for blockchain submission."""
        try:
            if not self.market_config_manager:
                return {'valid': False, 'error': 'Market configuration not available'}
            
            return self.market_config_manager.validate_market_order(
                order.symbol, order.price, order.size
            )
            
        except Exception as e:
            logger.error(f"Order validation failed: {e}")
            return {'valid': False, 'error': f'Validation error: {e}'}
    
    def get_account_balance(self) -> Optional[Dict[str, Any]]:
        """Get current account balance information."""
        try:
            if not self.account_manager:
                return None
            
            active_account = asyncio.create_task(self.account_manager.get_active_account())
            # Note: This is simplified - in production, handle async properly
            return None
            
        except Exception as e:
            logger.error(f"Failed to get account balance: {e}")
            return None
    
    def get_network_status(self) -> Dict[str, Any]:
        """Get current blockchain network status."""
        try:
            if self.blockchain_monitor:
                return self.blockchain_monitor.get_current_network_status()
            return {'status': 'monitoring_unavailable'}
            
        except Exception as e:
            logger.error(f"Failed to get network status: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _is_service_ready(self) -> bool:
        """Check if service is ready for operations."""
        return self.status in [ServiceStatus.HEALTHY, ServiceStatus.DEGRADED]
    
    async def _handle_blockchain_alert(self, metric: str, severity: AlertSeverity, data: Dict[str, Any]) -> None:
        """Handle blockchain monitoring alerts."""
        try:
            logger.warning(f"Blockchain alert received: {metric} ({severity.value})")
            
            # Handle critical alerts
            if severity == AlertSeverity.CRITICAL:
                # Consider switching to degraded mode
                if self.status == ServiceStatus.HEALTHY:
                    self.status = ServiceStatus.DEGRADED
                    await self._notify_status_change(self.status)
            
            # Handle emergency alerts
            elif severity == AlertSeverity.EMERGENCY:
                if self.emergency_stop_enabled:
                    await self._trigger_emergency_stop(f"Emergency alert: {metric}")
            
        except Exception as e:
            logger.error(f"Failed to handle blockchain alert: {e}")
    
    async def _trigger_emergency_stop(self, reason: str) -> None:
        """Trigger emergency stop procedures."""
        try:
            logger.critical(f"EMERGENCY STOP TRIGGERED: {reason}")
            
            old_status = self.status
            self.status = ServiceStatus.EMERGENCY
            
            # Cancel all active orders if possible
            if self.order_manager:
                try:
                    # In production, implement emergency order cancellation
                    pass
                except Exception as e:
                    logger.error(f"Failed to cancel orders during emergency stop: {e}")
            
            # Notify status change
            await self._notify_status_change(self.status)
            
            # Record emergency event
            self.error_history.append({
                'timestamp': datetime.now().isoformat(),
                'type': 'emergency_stop',
                'reason': reason,
                'previous_status': old_status.value
            })
            
        except Exception as e:
            logger.error(f"Emergency stop procedure failed: {e}")
    
    async def _handle_error(self, error_type: str, error: Exception) -> None:
        """Handle service errors."""
        self.error_count += 1
        self.last_error = str(error)
        
        # Record error
        self.error_history.append({
            'timestamp': datetime.now().isoformat(),
            'type': error_type,
            'message': str(error)
        })
        
        # Keep only recent errors
        if len(self.error_history) > 100:
            self.error_history = self.error_history[-100:]
        
        # Notify error callbacks
        await self._notify_error(error_type, error)
    
    async def _notify_status_change(self, new_status: ServiceStatus) -> None:
        """Notify registered callbacks of status change."""
        for callback in self.status_change_callbacks:
            try:
                await callback(new_status)
            except Exception as e:
                logger.warning(f"Status change callback failed: {e}")
    
    async def _notify_error(self, error_type: str, error: Exception) -> None:
        """Notify registered callbacks of errors."""
        for callback in self.error_callbacks:
            try:
                await callback(error_type, error)
            except Exception as e:
                logger.warning(f"Error callback failed: {e}")
    
    # Public management methods
    
    def add_status_change_callback(self, callback: Callable[[ServiceStatus], None]) -> None:
        """Add status change callback."""
        self.status_change_callbacks.append(callback)
    
    def add_error_callback(self, callback: Callable[[str, Exception], None]) -> None:
        """Add error callback."""
        self.error_callbacks.append(callback)
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get comprehensive service status."""
        return {
            'status': self.status.value,
            'enabled': self.enabled,
            'initialization_time': self.initialization_time.isoformat() if self.initialization_time else None,
            'uptime_seconds': (datetime.now() - self.initialization_time).total_seconds() if self.initialization_time else 0,
            'last_health_check': {
                'timestamp': self.last_health_check.timestamp.isoformat(),
                'status': self.last_health_check.status.value,
                'components': self.last_health_check.components,
                'error_count': self.last_health_check.error_count,
                'performance_metrics': self.last_health_check.performance_metrics
            } if self.last_health_check else None,
            'performance_stats': self.performance_stats,
            'error_count': self.error_count,
            'last_error': self.last_error,
            'recent_errors': self.error_history[-5:] if self.error_history else []
        }
    
    def get_component_status(self) -> Dict[str, Any]:
        """Get status of individual components."""
        return {
            'sei_client': self.sei_client is not None,
            'market_config_manager': self.market_config_manager is not None,
            'order_manager': self.order_manager is not None,
            'transaction_manager': self.transaction_manager is not None,
            'account_manager': self.account_manager is not None,
            'blockchain_monitor': self.blockchain_monitor is not None
        }
    
    async def force_health_check(self) -> Dict[str, Any]:
        """Force immediate health check."""
        try:
            health_check = await self._perform_health_check()
            return {
                'success': True,
                'health_check': {
                    'status': health_check.status.value,
                    'timestamp': health_check.timestamp.isoformat(),
                    'components': health_check.components,
                    'error_count': health_check.error_count,
                    'performance_metrics': health_check.performance_metrics
                }
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    async def restart_service(self) -> bool:
        """Restart the blockchain service."""
        try:
            logger.info("Restarting blockchain service")
            
            # Cleanup existing components
            await self.cleanup()
            
            # Reinitialize
            await self.initialize()
            
            return self.status in [ServiceStatus.HEALTHY, ServiceStatus.DEGRADED]
            
        except Exception as e:
            logger.error(f"Service restart failed: {e}")
            return False
    
    async def cleanup(self) -> None:
        """Cleanup all resources and stop background tasks."""
        try:
            logger.info("Cleaning up blockchain service")
            
            # Cancel background tasks
            tasks = [self._health_check_task, self._recovery_task]
            for task in tasks:
                if task and not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
            
            # Cleanup components
            components = [
                self.blockchain_monitor,
                self.order_manager,
                self.account_manager,
                self.transaction_manager,
                self.market_config_manager,
                self.sei_client
            ]
            
            for component in components:
                if component and hasattr(component, 'cleanup'):
                    try:
                        await component.cleanup()
                    except Exception as e:
                        logger.warning(f"Component cleanup failed: {e}")
            
            # Clear references
            self.sei_client = None
            self.market_config_manager = None
            self.order_manager = None
            self.transaction_manager = None
            self.account_manager = None
            self.blockchain_monitor = None
            
            self.status = ServiceStatus.SHUTDOWN
            
            logger.info("Blockchain service cleanup completed")
            
        except Exception as e:
            logger.error(f"Service cleanup failed: {e}")


# Global blockchain service instance
_blockchain_service: Optional[BlockchainService] = None


def get_blockchain_service() -> BlockchainService:
    """Get global blockchain service instance."""
    global _blockchain_service
    if _blockchain_service is None:
        _blockchain_service = BlockchainService()
    return _blockchain_service
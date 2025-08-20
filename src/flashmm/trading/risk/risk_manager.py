"""
FlashMM Risk Manager

Core risk management with position limits, circuit breakers, and P&L tracking.
"""

from typing import Dict, Any, Optional
from decimal import Decimal
from datetime import datetime

from flashmm.config.settings import get_config
from flashmm.trading.risk.position_tracker import PositionTracker
from flashmm.utils.logging import get_logger, TradingEventLogger
from flashmm.utils.exceptions import RiskError, CircuitBreakerError

logger = get_logger(__name__)
trading_logger = TradingEventLogger()


class RiskManager:
    """Core risk management system."""
    
    def __init__(self):
        self.config = get_config()
        self.position_tracker: Optional[PositionTracker] = None
        
        # Risk limits
        self.max_position_usdc = self.config.get("trading.max_position_usdc", 2000.0)
        self.max_daily_volume = self.config.get("risk.max_daily_volume_usdc", 100000.0)
        self.circuit_breaker_loss_percent = self.config.get("risk.circuit_breaker_loss_percent", 10.0)
        
        # Circuit breaker state
        self.circuit_breaker_active = False
        self.daily_pnl = 0.0
        self.daily_volume = 0.0
    
    async def initialize(self) -> None:
        """Initialize risk manager."""
        try:
            self.position_tracker = PositionTracker()
            await self.position_tracker.initialize()
            
            logger.info("RiskManager initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize RiskManager: {e}")
            raise RiskError(f"RiskManager initialization failed: {e}")
    
    async def check_trading_allowed(self, symbol: str) -> bool:
        """Check if trading is allowed for a symbol."""
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
            logger.error(f"Risk check failed for {symbol}: {e}")
            return False
    
    async def _check_position_limits(self, symbol: str) -> bool:
        """Check position size limits."""
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
        """Check daily volume limits."""
        if self.daily_volume > self.max_daily_volume:
            logger.warning(f"Daily volume limit exceeded: {self.daily_volume}")
            return False
        
        return True
    
    async def _check_pnl_limits(self) -> bool:
        """Check P&L limits and circuit breaker."""
        loss_percent = abs(self.daily_pnl / self.max_position_usdc) * 100
        
        if loss_percent > self.circuit_breaker_loss_percent:
            await self._trigger_circuit_breaker("pnl_limit_exceeded")
            return False
        
        return True
    
    async def _trigger_circuit_breaker(self, reason: str) -> None:
        """Trigger circuit breaker to stop trading."""
        self.circuit_breaker_active = True
        
        logger.critical(f"Circuit breaker triggered: {reason}")
        await trading_logger.log_pnl_event(
            "ALL",
            0.0,
            self.daily_pnl,
            self.daily_pnl,
            circuit_breaker_reason=reason
        )
        
        raise CircuitBreakerError(
            f"Trading halted: {reason}",
            breaker_name="main_circuit_breaker",
            trigger_reason=reason
        )
    
    async def update_position(self, symbol: str, side: str, size: float, price: float) -> None:
        """Update position after trade execution."""
        if not self.position_tracker:
            return
        
        try:
            await self.position_tracker.update_position(symbol, side, size, price)
            
            # Update daily volume
            trade_value = size * price
            self.daily_volume += trade_value
            
        except Exception as e:
            logger.error(f"Failed to update position: {e}")
    
    async def get_risk_metrics(self) -> Dict[str, Any]:
        """Get current risk metrics."""
        positions = {}
        if self.position_tracker:
            try:
                positions = await self.position_tracker.get_all_positions()
            except Exception as e:
                logger.error(f"Failed to get positions: {e}")
        
        return {
            "circuit_breaker_active": self.circuit_breaker_active,
            "daily_pnl": self.daily_pnl,
            "daily_volume": self.daily_volume,
            "max_position_usdc": self.max_position_usdc,
            "max_daily_volume": self.max_daily_volume,
            "positions": positions,
        }
    
    async def reset_daily_metrics(self) -> None:
        """Reset daily P&L and volume metrics."""
        self.daily_pnl = 0.0
        self.daily_volume = 0.0
        logger.info("Daily risk metrics reset")
    
    async def reset_circuit_breaker(self) -> None:
        """Reset circuit breaker (manual override)."""
        self.circuit_breaker_active = False
        logger.warning("Circuit breaker manually reset")
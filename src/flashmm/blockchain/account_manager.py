"""
Account Manager

Multi-account support with hot, warm, and cold key management,
balance monitoring, rotation, and security features.
"""

import asyncio
from typing import Dict, Any, Optional, List, Set, Tuple
from decimal import Decimal
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import uuid
import json
from pathlib import Path

from cosmpy.crypto.keypairs import PrivateKey
from cosmpy.aerial.wallet import LocalWallet

from flashmm.config.settings import get_config
from flashmm.utils.logging import get_logger
from flashmm.utils.exceptions import SecurityError, ValidationError, BlockchainError
from flashmm.utils.decorators import timeout_async, measure_latency
from flashmm.security.key_manager import KeyManager
from flashmm.blockchain.sei_client import SeiClient, AccountInfo

logger = get_logger(__name__)


class AccountType(Enum):
    """Account type classification."""
    HOT = "hot"          # Active trading account
    WARM = "warm"        # Backup trading account
    COLD = "cold"        # Long-term storage account
    EMERGENCY = "emergency"  # Emergency procedures account


class AccountStatus(Enum):
    """Account status."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    COMPROMISED = "compromised"
    EMERGENCY = "emergency"


@dataclass
class AccountConfig:
    """Account configuration."""
    account_id: str
    account_type: AccountType
    address: str
    nickname: str
    max_daily_volume: Decimal
    max_single_transaction: Decimal
    enabled_operations: Set[str]
    status: AccountStatus = AccountStatus.ACTIVE
    created_at: datetime = field(default_factory=datetime.now)
    last_used: Optional[datetime] = None
    rotation_due: Optional[datetime] = None
    emergency_contact: Optional[str] = None


@dataclass
class AccountBalance:
    """Account balance information."""
    account_id: str
    address: str
    balances: Dict[str, Decimal]
    total_value_usdc: Decimal
    last_updated: datetime
    pending_transactions: int = 0


@dataclass
class AccountActivity:
    """Account activity record."""
    account_id: str
    activity_type: str
    amount: Optional[Decimal]
    asset: Optional[str]
    transaction_hash: Optional[str]
    timestamp: datetime
    status: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class AccountManager:
    """Advanced account management with multi-key security."""
    
    def __init__(self, sei_client: SeiClient):
        self.sei_client = sei_client
        self.config = get_config()
        self.key_manager = KeyManager()
        
        # Account storage
        self.accounts: Dict[str, AccountConfig] = {}
        self.wallets: Dict[str, LocalWallet] = {}
        self.balances: Dict[str, AccountBalance] = {}
        self.activity_log: List[AccountActivity] = []
        
        # Configuration
        self.auto_rotation_enabled = self.config.get("accounts.auto_rotation", True)
        self.rotation_interval_days = self.config.get("accounts.rotation_interval_days", 30)
        self.balance_alert_threshold = self.config.get("accounts.balance_alert_threshold", 100.0)
        self.max_accounts = self.config.get("accounts.max_accounts", 10)
        
        # Security settings
        self.daily_volume_limits = {
            AccountType.HOT: Decimal("50000"),      # $50k daily limit
            AccountType.WARM: Decimal("100000"),    # $100k daily limit  
            AccountType.COLD: Decimal("1000000"),   # $1M daily limit
            AccountType.EMERGENCY: Decimal("10000") # $10k emergency limit
        }
        
        self.single_tx_limits = {
            AccountType.HOT: Decimal("10000"),      # $10k single tx limit
            AccountType.WARM: Decimal("25000"),     # $25k single tx limit
            AccountType.COLD: Decimal("100000"),    # $100k single tx limit
            AccountType.EMERGENCY: Decimal("5000")  # $5k emergency limit
        }
        
        # Background tasks
        self._monitoring_task: Optional[asyncio.Task] = None
        self._rotation_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        
        # Active account tracking
        self.current_active_account: Optional[str] = None
        self.account_usage_stats: Dict[str, Dict[str, Any]] = {}
        
    async def initialize(self) -> None:
        """Initialize the account manager."""
        try:
            logger.info("Initializing account manager")
            
            # Load existing accounts
            await self._load_accounts()
            
            # Create default accounts if none exist
            if not self.accounts:
                await self._create_default_accounts()
            
            # Update account balances
            await self._update_all_balances()
            
            # Start background tasks
            await self._start_background_tasks()
            
            logger.info(f"Account manager initialized with {len(self.accounts)} accounts")
            
        except Exception as e:
            logger.error(f"Failed to initialize account manager: {e}")
            raise BlockchainError(f"Account manager initialization failed: {e}")
    
    async def _load_accounts(self) -> None:
        """Load accounts from configuration or storage."""
        try:
            # In production, load from secure storage
            # For now, load from configuration
            
            accounts_config = self.config.get("accounts.configured_accounts", [])
            
            for account_data in accounts_config:
                account_config = AccountConfig(
                    account_id=account_data.get("id", str(uuid.uuid4())),
                    account_type=AccountType(account_data.get("type", "hot")),
                    address=account_data.get("address", ""),
                    nickname=account_data.get("nickname", ""),
                    max_daily_volume=Decimal(str(account_data.get("max_daily_volume", "10000"))),
                    max_single_transaction=Decimal(str(account_data.get("max_single_tx", "1000"))),
                    enabled_operations=set(account_data.get("enabled_operations", ["trade", "withdraw"]))
                )
                
                self.accounts[account_config.account_id] = account_config
                
                # Initialize wallet if private key is available
                private_key_env = account_data.get("private_key_env")
                if private_key_env:
                    await self._initialize_account_wallet(account_config.account_id, private_key_env)
            
            logger.info(f"Loaded {len(self.accounts)} accounts from configuration")
            
        except Exception as e:
            logger.warning(f"Failed to load accounts: {e}")
    
    async def _create_default_accounts(self) -> None:
        """Create default accounts for initial setup."""
        try:
            # Create hot trading account
            hot_account = await self._create_account(
                account_type=AccountType.HOT,
                nickname="Primary Trading",
                max_daily_volume=self.daily_volume_limits[AccountType.HOT],
                enabled_operations={"trade", "cancel", "modify"}
            )
            
            # Set as current active account
            self.current_active_account = hot_account.account_id
            
            # Create warm backup account
            await self._create_account(
                account_type=AccountType.WARM,
                nickname="Backup Trading",
                max_daily_volume=self.daily_volume_limits[AccountType.WARM],
                enabled_operations={"trade", "cancel", "modify", "withdraw"}
            )
            
            logger.info("Default accounts created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create default accounts: {e}")
            raise
    
    async def _initialize_account_wallet(self, account_id: str, private_key_env: str) -> None:
        """Initialize wallet for account."""
        try:
            private_key_hex = self.config.get(private_key_env)
            if not private_key_hex:
                logger.warning(f"Private key not found for account {account_id}")
                return
            
            # Decrypt private key
            decrypted_key = await self.key_manager.decrypt_private_key(private_key_hex)
            private_key = PrivateKey.from_hex(decrypted_key)
            
            # Create wallet
            wallet = LocalWallet(private_key)
            self.wallets[account_id] = wallet
            
            # Update account address
            account = self.accounts.get(account_id)
            if account:
                account.address = wallet.address()
            
            logger.debug(f"Wallet initialized for account {account_id}")
            
        except Exception as e:
            logger.error(f"Failed to initialize wallet for account {account_id}: {e}")
    
    async def _start_background_tasks(self) -> None:
        """Start background maintenance tasks."""
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        self._rotation_task = asyncio.create_task(self._rotation_loop())
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
    
    async def _monitoring_loop(self) -> None:
        """Background account monitoring loop."""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                # Update balances
                await self._update_all_balances()
                
                # Check account health
                await self._check_account_health()
                
                # Check usage limits
                await self._check_usage_limits()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Account monitoring error: {e}")
                await asyncio.sleep(60)
    
    async def _rotation_loop(self) -> None:
        """Background account rotation loop."""
        while True:
            try:
                await asyncio.sleep(3600)  # Check every hour
                
                if self.auto_rotation_enabled:
                    await self._check_rotation_schedule()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Account rotation error: {e}")
                await asyncio.sleep(3600)
    
    async def _cleanup_loop(self) -> None:
        """Background cleanup loop."""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                await self._cleanup_old_activity()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")
                await asyncio.sleep(3600)
    
    @timeout_async(5.0)
    async def create_account(
        self,
        account_type: AccountType,
        nickname: str,
        max_daily_volume: Optional[Decimal] = None,
        enabled_operations: Optional[Set[str]] = None
    ) -> AccountConfig:
        """Create a new account."""
        if len(self.accounts) >= self.max_accounts:
            raise SecurityError(f"Maximum number of accounts ({self.max_accounts}) reached")
        
        return await self._create_account(account_type, nickname, max_daily_volume, enabled_operations)
    
    async def _create_account(
        self,
        account_type: AccountType,
        nickname: str,
        max_daily_volume: Optional[Decimal] = None,
        enabled_operations: Optional[Set[str]] = None
    ) -> AccountConfig:
        """Internal account creation."""
        try:
            # Generate new account
            account_id = str(uuid.uuid4())
            
            # Create account configuration
            account_config = AccountConfig(
                account_id=account_id,
                account_type=account_type,
                address="",  # Will be set when wallet is created
                nickname=nickname,
                max_daily_volume=max_daily_volume or self.daily_volume_limits[account_type],
                max_single_transaction=self.single_tx_limits[account_type],
                enabled_operations=enabled_operations or {"trade"},
                rotation_due=datetime.now() + timedelta(days=self.rotation_interval_days)
            )
            
            # Generate new wallet
            private_key = PrivateKey.generate()
            wallet = LocalWallet(private_key)
            account_config.address = wallet.address()
            
            # Store encrypted private key
            encrypted_key = await self.key_manager.encrypt_private_key(private_key.hex())
            await self._store_account_key(account_id, encrypted_key)
            
            # Store account and wallet
            self.accounts[account_id] = account_config
            self.wallets[account_id] = wallet
            
            # Initialize usage stats
            self.account_usage_stats[account_id] = {
                'daily_volume': Decimal('0'),
                'transaction_count': 0,
                'last_reset': datetime.now().date()
            }
            
            # Log account creation
            await self._log_activity(
                account_id,
                "account_created",
                metadata={
                    'type': account_type.value,
                    'nickname': nickname,
                    'address': account_config.address
                }
            )
            
            logger.info(f"Account created: {account_id} ({nickname}) - {account_config.address}")
            return account_config
            
        except Exception as e:
            logger.error(f"Failed to create account: {e}")
            raise SecurityError(f"Account creation failed: {e}")
    
    async def _store_account_key(self, account_id: str, encrypted_key: str) -> None:
        """Store encrypted account private key."""
        # In production, store in secure key management system
        # For now, store in memory only
        pass
    
    @timeout_async(3.0)
    async def get_account_balance(self, account_id: str) -> Optional[AccountBalance]:
        """Get current balance for account."""
        account = self.accounts.get(account_id)
        if not account:
            return None
        
        try:
            # Get account info from blockchain
            account_info = await self.sei_client.get_account_info(account.address)
            
            # Calculate total value in USDC
            total_value = Decimal('0')
            for asset, balance in account_info.balances.items():
                # In production, convert all assets to USDC value
                if asset.lower() == 'usdc':
                    total_value += balance
                elif asset.lower() == 'sei':
                    # Mock conversion rate
                    total_value += balance * Decimal('0.5')  # Assume 1 SEI = $0.5
            
            balance_info = AccountBalance(
                account_id=account_id,
                address=account.address,
                balances=account_info.balances,
                total_value_usdc=total_value,
                last_updated=datetime.now()
            )
            
            # Cache balance
            self.balances[account_id] = balance_info
            
            return balance_info
            
        except Exception as e:
            logger.error(f"Failed to get balance for account {account_id}: {e}")
            return None
    
    async def _update_all_balances(self) -> None:
        """Update balances for all accounts."""
        try:
            for account_id in self.accounts.keys():
                await self.get_account_balance(account_id)
        except Exception as e:
            logger.error(f"Failed to update all balances: {e}")
    
    async def switch_active_account(self, account_id: str) -> bool:
        """Switch to a different active account."""
        try:
            account = self.accounts.get(account_id)
            if not account:
                logger.error(f"Account not found: {account_id}")
                return False
            
            if account.status != AccountStatus.ACTIVE:
                logger.error(f"Account not active: {account_id}")
                return False
            
            if account_id not in self.wallets:
                logger.error(f"Wallet not available for account: {account_id}")
                return False
            
            old_account = self.current_active_account
            self.current_active_account = account_id
            
            # Update Sei client wallet
            self.sei_client.wallet = self.wallets[account_id]
            
            # Log account switch
            await self._log_activity(
                account_id,
                "account_activated",
                metadata={'previous_account': old_account}
            )
            
            logger.info(f"Switched to account: {account_id} ({account.nickname})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to switch account: {e}")
            return False
    
    async def get_active_account(self) -> Optional[AccountConfig]:
        """Get currently active account."""
        if self.current_active_account:
            return self.accounts.get(self.current_active_account)
        return None
    
    async def validate_transaction(
        self,
        account_id: str,
        operation: str,
        amount: Optional[Decimal] = None
    ) -> Dict[str, Any]:
        """Validate if transaction is allowed for account."""
        try:
            account = self.accounts.get(account_id)
            if not account:
                return {'valid': False, 'error': 'Account not found'}
            
            if account.status != AccountStatus.ACTIVE:
                return {'valid': False, 'error': f'Account status: {account.status.value}'}
            
            if operation not in account.enabled_operations:
                return {'valid': False, 'error': f'Operation {operation} not enabled'}
            
            # Check transaction limits
            if amount:
                if amount > account.max_single_transaction:
                    return {
                        'valid': False,
                        'error': f'Amount exceeds single transaction limit: {amount} > {account.max_single_transaction}'
                    }
                
                # Check daily volume limit
                usage = self.account_usage_stats.get(account_id, {})
                daily_volume = usage.get('daily_volume', Decimal('0'))
                
                if daily_volume + amount > account.max_daily_volume:
                    return {
                        'valid': False,
                        'error': f'Would exceed daily volume limit: {daily_volume + amount} > {account.max_daily_volume}'
                    }
            
            return {'valid': True}
            
        except Exception as e:
            logger.error(f"Transaction validation failed: {e}")
            return {'valid': False, 'error': f'Validation error: {e}'}
    
    async def record_transaction(
        self,
        account_id: str,
        operation: str,
        amount: Optional[Decimal] = None,
        asset: Optional[str] = None,
        transaction_hash: Optional[str] = None
    ) -> None:
        """Record transaction for usage tracking."""
        try:
            # Update usage statistics
            if account_id in self.account_usage_stats:
                usage = self.account_usage_stats[account_id]
                
                # Reset daily stats if new day
                if usage['last_reset'] != datetime.now().date():
                    usage['daily_volume'] = Decimal('0')
                    usage['transaction_count'] = 0
                    usage['last_reset'] = datetime.now().date()
                
                # Update stats
                if amount:
                    usage['daily_volume'] += amount
                usage['transaction_count'] += 1
            
            # Update last used timestamp
            account = self.accounts.get(account_id)
            if account:
                account.last_used = datetime.now()
            
            # Log activity
            await self._log_activity(
                account_id,
                operation,
                amount,
                asset,
                transaction_hash,
                "completed"
            )
            
        except Exception as e:
            logger.error(f"Failed to record transaction: {e}")
    
    async def _check_account_health(self) -> None:
        """Check health of all accounts."""
        try:
            for account_id, account in self.accounts.items():
                # Check balance alerts
                balance = self.balances.get(account_id)
                if balance and balance.total_value_usdc < self.balance_alert_threshold:
                    logger.warning(f"Low balance alert for account {account_id}: ${balance.total_value_usdc}")
                
                # Check for inactive accounts
                if account.last_used:
                    inactive_days = (datetime.now() - account.last_used).days
                    if inactive_days > 7:  # 7 days inactive
                        logger.info(f"Account {account_id} inactive for {inactive_days} days")
                
                # Check rotation due
                if account.rotation_due and datetime.now() > account.rotation_due:
                    logger.warning(f"Account {account_id} rotation overdue")
                    
        except Exception as e:
            logger.error(f"Account health check failed: {e}")
    
    async def _check_usage_limits(self) -> None:
        """Check usage limits for all accounts."""
        try:
            for account_id, account in self.accounts.items():
                usage = self.account_usage_stats.get(account_id)
                if not usage:
                    continue
                
                # Check if approaching daily limit
                daily_volume = usage['daily_volume']
                limit_percentage = (daily_volume / account.max_daily_volume) * 100
                
                if limit_percentage > 80:  # 80% of daily limit
                    logger.warning(f"Account {account_id} approaching daily limit: {limit_percentage:.1f}%")
                    
        except Exception as e:
            logger.error(f"Usage limit check failed: {e}")
    
    async def _check_rotation_schedule(self) -> None:
        """Check and perform scheduled account rotations."""
        try:
            for account_id, account in self.accounts.items():
                if account.rotation_due and datetime.now() > account.rotation_due:
                    if account.account_type == AccountType.HOT:
                        await self._rotate_hot_account(account_id)
                        
        except Exception as e:
            logger.error(f"Rotation schedule check failed: {e}")
    
    async def _rotate_hot_account(self, account_id: str) -> None:
        """Rotate hot account for security."""
        try:
            logger.info(f"Rotating hot account: {account_id}")
            
            # Find warm account to promote
            warm_accounts = [
                acc for acc in self.accounts.values()
                if acc.account_type == AccountType.WARM and acc.status == AccountStatus.ACTIVE
            ]
            
            if not warm_accounts:
                logger.warning("No warm account available for rotation")
                return
            
            # Switch to warm account
            new_hot_account = warm_accounts[0]
            success = await self.switch_active_account(new_hot_account.account_id)
            
            if success:
                # Demote old hot account to warm
                old_account = self.accounts[account_id]
                old_account.account_type = AccountType.WARM
                
                # Promote new account to hot
                new_hot_account.account_type = AccountType.HOT
                new_hot_account.rotation_due = datetime.now() + timedelta(days=self.rotation_interval_days)
                
                logger.info(f"Account rotation completed: {account_id} -> {new_hot_account.account_id}")
                
        except Exception as e:
            logger.error(f"Hot account rotation failed: {e}")
    
    async def _log_activity(
        self,
        account_id: str,
        activity_type: str,
        amount: Optional[Decimal] = None,
        asset: Optional[str] = None,
        transaction_hash: Optional[str] = None,
        status: str = "completed",
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log account activity."""
        try:
            activity = AccountActivity(
                account_id=account_id,
                activity_type=activity_type,
                amount=amount,
                asset=asset,
                transaction_hash=transaction_hash,
                timestamp=datetime.now(),
                status=status,
                metadata=metadata or {}
            )
            
            self.activity_log.append(activity)
            
            # Keep only recent activity (last 1000 entries)
            if len(self.activity_log) > 1000:
                self.activity_log = self.activity_log[-1000:]
                
        except Exception as e:
            logger.error(f"Failed to log activity: {e}")
    
    async def _cleanup_old_activity(self) -> None:
        """Clean up old activity logs."""
        try:
            cutoff_time = datetime.now() - timedelta(days=7)
            
            old_count = len(self.activity_log)
            self.activity_log = [
                activity for activity in self.activity_log
                if activity.timestamp > cutoff_time
            ]
            
            cleaned_count = old_count - len(self.activity_log)
            if cleaned_count > 0:
                logger.debug(f"Cleaned up {cleaned_count} old activity records")
                
        except Exception as e:
            logger.error(f"Activity cleanup failed: {e}")
    
    def get_account_summary(self) -> Dict[str, Any]:
        """Get summary of all accounts."""
        active_accounts = len([acc for acc in self.accounts.values() if acc.status == AccountStatus.ACTIVE])
        total_balance = sum(
            balance.total_value_usdc for balance in self.balances.values()
        )
        
        return {
            'total_accounts': len(self.accounts),
            'active_accounts': active_accounts,
            'current_active_account': self.current_active_account,
            'total_balance_usdc': float(total_balance),
            'accounts_by_type': {
                account_type.value: len([
                    acc for acc in self.accounts.values()
                    if acc.account_type == account_type
                ])
                for account_type in AccountType
            },
            'recent_activity_count': len(self.activity_log)
        }
    
    def get_account_details(self, account_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information for specific account."""
        account = self.accounts.get(account_id)
        if not account:
            return None
        
        balance = self.balances.get(account_id)
        usage = self.account_usage_stats.get(account_id, {})
        
        return {
            'account_id': account.account_id,
            'type': account.account_type.value,
            'nickname': account.nickname,
            'address': account.address,
            'status': account.status.value,
            'created_at': account.created_at.isoformat(),
            'last_used': account.last_used.isoformat() if account.last_used else None,
            'rotation_due': account.rotation_due.isoformat() if account.rotation_due else None,
            'limits': {
                'max_daily_volume': float(account.max_daily_volume),
                'max_single_transaction': float(account.max_single_transaction)
            },
            'enabled_operations': list(account.enabled_operations),
            'balance': {
                'balances': {k: float(v) for k, v in balance.balances.items()} if balance else {},
                'total_value_usdc': float(balance.total_value_usdc) if balance else 0,
                'last_updated': balance.last_updated.isoformat() if balance else None
            },
            'daily_usage': {
                'volume': float(usage.get('daily_volume', 0)),
                'transaction_count': usage.get('transaction_count', 0),
                'volume_percentage': float(usage.get('daily_volume', 0) / account.max_daily_volume * 100) if account.max_daily_volume > 0 else 0
            }
        }
    
    async def emergency_freeze_account(self, account_id: str, reason: str) -> bool:
        """Emergency freeze account."""
        try:
            account = self.accounts.get(account_id)
            if not account:
                return False
            
            account.status = AccountStatus.SUSPENDED
            
            # Switch to emergency account if this was active
            if self.current_active_account == account_id:
                emergency_accounts = [
                    acc for acc in self.accounts.values()
                    if acc.account_type == AccountType.EMERGENCY and acc.status == AccountStatus.ACTIVE
                ]
                
                if emergency_accounts:
                    await self.switch_active_account(emergency_accounts[0].account_id)
            
            # Log emergency action
            await self._log_activity(
                account_id,
                "emergency_freeze",
                metadata={'reason': reason}
            )
            
            logger.critical(f"Emergency freeze applied to account {account_id}: {reason}")
            return True
            
        except Exception as e:
            logger.error(f"Emergency freeze failed: {e}")
            return False
    
    async def cleanup(self) -> None:
        """Cleanup resources and stop background tasks."""
        # Cancel background tasks
        tasks = [self._monitoring_task, self._rotation_task, self._cleanup_task]
        
        for task in tasks:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Clear sensitive data
        self.wallets.clear()
        
        logger.info("Account manager cleanup completed")
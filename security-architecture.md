# FlashMM Security Architecture & Key Management

## Overview
FlashMM handles sensitive trading operations with real funds, requiring a comprehensive security architecture that balances robust protection with hackathon development speed. This document outlines security controls, key management strategies, and risk mitigation approaches.

## Security Threat Model

### Threat Categories
1. **Financial Theft**: Unauthorized access to trading keys
2. **Data Breach**: Exposure of sensitive configuration/strategies
3. **Service Disruption**: DoS attacks on trading operations
4. **Strategy Theft**: Intellectual property exposure
5. **Regulatory Compliance**: KYC/AML requirements
6. **Insider Threats**: Malicious or negligent team members

### Attack Vectors
- API key compromise
- Container/VM breakout
- Man-in-the-middle attacks
- Social engineering
- Supply chain attacks
- Misconfigured cloud resources

## Key Management Architecture

### Key Classification & Storage Strategy

#### Hot Keys (Online, Active Trading)
**Purpose**: Real-time order signing and API authentication  
**Risk Level**: High  
**Storage**: Environment variables in secure containers  
**Rotation**: Daily during hackathon, hourly in production

```python
class HotKeyManager:
    """Manages keys needed for active trading operations"""
    
    def __init__(self):
        self.cambrian_api_key = os.getenv("CAMBRIAN_API_KEY")
        self.cambrian_secret = os.getenv("CAMBRIAN_SECRET_KEY")
        self.sei_private_key = os.getenv("SEI_PRIVATE_KEY")  # For transaction signing
        
        # Validate keys are present
        self._validate_keys()
        
        # Setup key rotation schedule
        self.rotation_interval = timedelta(hours=24)  # Hackathon setting
        self.last_rotation = datetime.now()
    
    def _validate_keys(self):
        """Validate all required keys are present and valid"""
        required_keys = [
            "CAMBRIAN_API_KEY", 
            "CAMBRIAN_SECRET_KEY", 
            "SEI_PRIVATE_KEY"
        ]
        
        missing_keys = [key for key in required_keys if not os.getenv(key)]
        if missing_keys:
            raise SecurityError(f"Missing required keys: {missing_keys}")
        
        # Test key validity
        if not self._test_cambrian_keys():
            raise SecurityError("Cambrian API keys invalid")
    
    async def _test_cambrian_keys(self) -> bool:
        """Test if Cambrian keys are valid"""
        try:
            client = CambrianClient(self.cambrian_api_key, self.cambrian_secret)
            await client.get_account_info()
            return True
        except Exception:
            return False
```

#### Warm Keys (Configuration, Less Sensitive)
**Purpose**: API access for monitoring services  
**Risk Level**: Medium  
**Storage**: Encrypted configuration files  
**Rotation**: Weekly

```python
class WarmKeyManager:
    """Manages moderately sensitive keys for monitoring/reporting"""
    
    def __init__(self):
        self.grafana_api_key = self._decrypt_key("GRAFANA_API_KEY_ENCRYPTED")
        self.twitter_bearer_token = self._decrypt_key("TWITTER_TOKEN_ENCRYPTED")
        self.influxdb_token = self._decrypt_key("INFLUXDB_TOKEN_ENCRYPTED")
    
    def _decrypt_key(self, encrypted_env_var: str) -> str:
        """Decrypt key using master encryption key"""
        encrypted_value = os.getenv(encrypted_env_var)
        if not encrypted_value:
            raise SecurityError(f"Missing encrypted key: {encrypted_env_var}")
        
        return self.cipher.decrypt(encrypted_value.encode()).decode()
    
    @property
    def cipher(self):
        """Get Fernet cipher for key decryption"""
        master_key = os.getenv("MASTER_ENCRYPTION_KEY")
        if not master_key:
            raise SecurityError("Master encryption key not found")
        
        return Fernet(master_key.encode())
```

#### Cold Keys (Backup, Emergency)
**Purpose**: Recovery and emergency access  
**Risk Level**: Critical  
**Storage**: Offline, encrypted, multiple locations  
**Rotation**: Monthly or on compromise

```python
class ColdKeyManager:
    """Manages offline keys for emergency recovery"""
    
    def __init__(self):
        # Cold keys are not loaded by default
        self.recovery_keys = {}
        self.emergency_shutdown_key = None
    
    async def emergency_recovery(self, recovery_phrase: str) -> Dict[str, str]:
        """Recover system using cold keys"""
        
        # Verify recovery phrase
        if not self._verify_recovery_phrase(recovery_phrase):
            raise SecurityError("Invalid recovery phrase")
        
        # Load cold keys from secure storage
        cold_keys = await self._load_cold_keys()
        
        # Audit log emergency access
        await self.audit_logger.log_critical_event(
            event="emergency_cold_key_access",
            user="system",
            details={"timestamp": datetime.now().isoformat()}
        )
        
        return cold_keys
```

### Key Rotation Strategy

#### Automatic Rotation
```python
class KeyRotationManager:
    async def rotate_hot_keys(self) -> None:
        """Rotate hot keys automatically"""
        
        logger.info("Starting hot key rotation")
        
        try:
            # Generate new Cambrian API keys
            new_cambrian_keys = await self.cambrian_admin.generate_new_keys()
            
            # Test new keys
            if not await self._test_new_keys(new_cambrian_keys):
                raise SecurityError("New keys failed validation")
            
            # Atomic key swap
            await self._atomic_key_update(new_cambrian_keys)
            
            # Revoke old keys
            await self.cambrian_admin.revoke_old_keys()
            
            # Update last rotation timestamp
            await self.redis.set("last_key_rotation", datetime.now().isoformat())
            
            logger.info("Hot key rotation completed successfully")
            
        except Exception as e:
            logger.error(f"Key rotation failed: {e}")
            await self.alert_manager.send_critical_alert("Key rotation failed")
            raise
    
    async def _atomic_key_update(self, new_keys: Dict[str, str]) -> None:
        """Update keys atomically to prevent service interruption"""
        
        # Pause trading temporarily
        await self.trading_engine.pause()
        
        try:
            # Update environment variables
            os.environ["CAMBRIAN_API_KEY"] = new_keys["api_key"]
            os.environ["CAMBRIAN_SECRET_KEY"] = new_keys["secret_key"]
            
            # Reinitialize clients with new keys
            await self.order_router.reinitialize_client()
            
            # Resume trading
            await self.trading_engine.resume()
            
        except Exception as e:
            # Rollback on failure
            logger.error("Key rotation rollback initiated")
            await self.trading_engine.emergency_stop()
            raise
```

## Authentication & Authorization

### API Authentication
```python
from datetime import datetime, timedelta
import jwt
from passlib.context import CryptContext

class AuthenticationManager:
    def __init__(self):
        self.secret_key = os.getenv("JWT_SECRET_KEY")
        self.algorithm = "HS256"
        self.access_token_expire = timedelta(hours=1)
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        
        # API key store (in production, use database)
        self.api_keys = {
            "admin": self._hash_password(os.getenv("ADMIN_API_KEY")),
            "readonly": self._hash_password(os.getenv("READONLY_API_KEY"))
        }
    
    def create_access_token(self, subject: str, permissions: List[str]) -> str:
        """Create JWT access token"""
        
        expire = datetime.utcnow() + self.access_token_expire
        to_encode = {
            "sub": subject,
            "permissions": permissions,
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "access"
        }
        
        return jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify JWT token and return payload"""
        
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            # Check token type
            if payload.get("type") != "access":
                raise AuthenticationError("Invalid token type")
            
            return payload
            
        except jwt.ExpiredSignatureError:
            raise AuthenticationError("Token expired")
        except jwt.JWTError:
            raise AuthenticationError("Invalid token")
    
    def verify_api_key(self, api_key: str) -> Optional[str]:
        """Verify API key and return role"""
        
        for role, hashed_key in self.api_keys.items():
            if self.pwd_context.verify(api_key, hashed_key):
                return role
        
        return None
```

### Role-Based Access Control
```python
class AuthorizationManager:
    PERMISSIONS = {
        "admin": [
            "trading.pause",
            "trading.resume", 
            "trading.emergency_stop",
            "config.read",
            "config.write",
            "metrics.read",
            "keys.rotate"
        ],
        "readonly": [
            "metrics.read",
            "config.read"
        ],
        "system": [
            "trading.execute",
            "metrics.write",
            "internal.all"
        ]
    }
    
    def check_permission(self, role: str, permission: str) -> bool:
        """Check if role has specific permission"""
        
        role_permissions = self.PERMISSIONS.get(role, [])
        
        # Check exact match
        if permission in role_permissions:
            return True
        
        # Check wildcard permissions
        for perm in role_permissions:
            if perm.endswith(".all"):
                prefix = perm[:-4]  # Remove ".all"
                if permission.startswith(prefix):
                    return True
        
        return False
    
    def require_permission(self, permission: str):
        """Decorator to require specific permission"""
        
        def decorator(func):
            @wraps(func)
            async def wrapper(request, *args, **kwargs):
                # Get user context from request
                user_context = getattr(request, "user_context", None)
                if not user_context:
                    raise AuthorizationError("Authentication required")
                
                # Check permission
                if not self.check_permission(user_context["role"], permission):
                    raise AuthorizationError(f"Permission denied: {permission}")
                
                return await func(request, *args, **kwargs)
            
            return wrapper
        return decorator
```

## Data Encryption

### Encryption at Rest
```python
class DataEncryption:
    def __init__(self):
        self.master_key = os.getenv("MASTER_ENCRYPTION_KEY")
        if not self.master_key:
            raise SecurityError("Master encryption key not configured")
        
        self.cipher = Fernet(self.master_key.encode())
    
    def encrypt_sensitive_data(self, data: str) -> str:
        """Encrypt sensitive data for storage"""
        
        if not isinstance(data, str):
            data = json.dumps(data)
        
        encrypted = self.cipher.encrypt(data.encode())
        return base64.b64encode(encrypted).decode()
    
    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data from storage"""
        
        try:
            encrypted_bytes = base64.b64decode(encrypted_data.encode())
            decrypted = self.cipher.decrypt(encrypted_bytes)
            return decrypted.decode()
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise SecurityError("Data decryption failed")
    
    def encrypt_config_file(self, config_path: str) -> None:
        """Encrypt configuration file"""
        
        with open(config_path, 'r') as f:
            config_data = f.read()
        
        encrypted_data = self.encrypt_sensitive_data(config_data)
        
        with open(f"{config_path}.encrypted", 'w') as f:
            f.write(encrypted_data)
        
        # Securely delete original
        os.remove(config_path)
```

### TLS/SSL Configuration
```python
# Nginx SSL configuration
SSL_CONFIG = """
server {
    listen 443 ssl http2;
    server_name flashmm.yourdomain.com;
    
    # SSL Configuration
    ssl_certificate /etc/nginx/ssl/flashmm.crt;
    ssl_certificate_key /etc/nginx/ssl/flashmm.key;
    
    # SSL Security Settings
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;
    
    # Security Headers
    add_header Strict-Transport-Security "max-age=63072000" always;
    add_header X-Frame-Options DENY always;
    add_header X-Content-Type-Options nosniff always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;
    
    location / {
        proxy_pass http://flashmm:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket support
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
"""
```

## Network Security

### Firewall Configuration
```bash
# UFW firewall rules for VPS
#!/bin/bash

# Reset firewall
ufw --force reset

# Default policies
ufw default deny incoming
ufw default allow outgoing

# SSH access (change default port)
ufw allow 2222/tcp comment 'SSH'

# HTTP/HTTPS
ufw allow 80/tcp comment 'HTTP'
ufw allow 443/tcp comment 'HTTPS'

# Application port (internal only)
ufw allow from 172.0.0.0/8 to any port 8000 comment 'FlashMM internal'

# Database ports (internal only)
ufw allow from 172.0.0.0/8 to any port 6379 comment 'Redis internal'
ufw allow from 172.0.0.0/8 to any port 8086 comment 'InfluxDB internal'

# Enable firewall
ufw --force enable

# Show status
ufw status verbose
```

### Docker Network Security
```yaml
# docker-compose.prod.yml security settings
version: '3.8'

services:
  flashmm:
    security_opt:
      - no-new-privileges:true
    user: flashmm:flashmm
    read_only: true
    tmpfs:
      - /tmp
      - /var/tmp
    volumes:
      - ./logs:/app/logs:rw
    cap_drop:
      - ALL
    cap_add:
      - NET_BIND_SERVICE

networks:
  flashmm-network:
    driver: bridge
    ipam:
      driver: default
      config:
        - subnet: 172.20.0.0/16
    driver_opts:
      com.docker.network.bridge.enable_icc: "false"
      com.docker.network.bridge.enable_ip_masquerade: "true"
```

## Application Security

### Input Validation & Sanitization
```python
from pydantic import BaseModel, validator
from decimal import Decimal

class OrderRequest(BaseModel):
    symbol: str
    side: str
    price: Decimal
    size: Decimal
    order_type: str = "limit"
    
    @validator('symbol')
    def validate_symbol(cls, v):
        allowed_symbols = ["SEI/USDC", "ETH/USDC"]
        if v not in allowed_symbols:
            raise ValueError(f"Symbol must be one of: {allowed_symbols}")
        return v
    
    @validator('side')
    def validate_side(cls, v):
        if v.lower() not in ['buy', 'sell']:
            raise ValueError("Side must be 'buy' or 'sell'")
        return v.lower()
    
    @validator('price')
    def validate_price(cls, v):
        if v <= 0:
            raise ValueError("Price must be positive")
        if v > Decimal('1000000'):  # Reasonable upper bound
            raise ValueError("Price too high")
        return v
    
    @validator('size') 
    def validate_size(cls, v):
        if v <= 0:
            raise ValueError("Size must be positive")
        if v > Decimal('1000000'):  # Position limit
            raise ValueError("Size too large")
        return v
```

### SQL Injection Prevention
```python
# Use parameterized queries (if using SQL database)
class PositionTracker:
    async def get_position(self, symbol: str, user_id: str) -> Dict:
        """Get position using parameterized query"""
        
        query = """
        SELECT symbol, base_balance, quote_balance, last_updated
        FROM positions 
        WHERE symbol = $1 AND user_id = $2
        """
        
        result = await self.db.fetchrow(query, symbol, user_id)
        return dict(result) if result else {}
    
    async def update_position(self, position: Position) -> None:
        """Update position safely"""
        
        query = """
        INSERT INTO positions (symbol, user_id, base_balance, quote_balance, last_updated)
        VALUES ($1, $2, $3, $4, $5)
        ON CONFLICT (symbol, user_id) 
        DO UPDATE SET 
            base_balance = $3,
            quote_balance = $4,
            last_updated = $5
        """
        
        await self.db.execute(
            query,
            position.symbol,
            position.user_id, 
            position.base_balance,
            position.quote_balance,
            position.last_updated
        )
```

## Audit Logging & Monitoring

### Security Event Logging
```python
class SecurityAuditLogger:
    def __init__(self):
        self.logger = structlog.get_logger("security_audit")
        self.influx_client = InfluxDBClient()
    
    async def log_authentication_event(self, event_type: str, user: str, 
                                     success: bool, ip_address: str = None) -> None:
        """Log authentication events"""
        
        event_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "user": user,
            "success": success,
            "ip_address": ip_address,
            "user_agent": getattr(request, "user_agent", "unknown") if 'request' in locals() else "unknown"
        }
        
        # Structured logging
        self.logger.info("authentication_event", **event_data)
        
        # Time-series storage
        point = Point("security_events") \
            .tag("event_type", event_type) \
            .tag("user", user) \
            .field("success", success) \
            .time(datetime.utcnow())
        
        await self.influx_client.write_api().write(point=point)
    
    async def log_trading_event(self, event_type: str, order_id: str, 
                              symbol: str, amount: Decimal) -> None:
        """Log trading events for audit trail"""
        
        event_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "order_id": order_id,
            "symbol": symbol,
            "amount": str(amount)
        }
        
        self.logger.info("trading_event", **event_data)
        
        # Store in InfluxDB for analysis
        point = Point("trading_events") \
            .tag("event_type", event_type) \
            .tag("symbol", symbol) \
            .field("amount", float(amount)) \
            .time(datetime.utcnow())
        
        await self.influx_client.write_api().write(point=point)
```

### Security Monitoring & Alerting
```python
class SecurityMonitor:
    def __init__(self):
        self.alert_thresholds = {
            "failed_logins": 5,  # per 5 minutes
            "large_trades": 10000,  # USDC value
            "position_limit": 2000,  # USDC
            "api_error_rate": 0.1  # 10% error rate
        }
    
    async def monitor_failed_logins(self) -> None:
        """Monitor for brute force attacks"""
        
        query = """
        SELECT COUNT(*) as failed_count, ip_address
        FROM security_events 
        WHERE event_type = 'login_failed' 
        AND timestamp > NOW() - INTERVAL '5 minutes'
        GROUP BY ip_address
        HAVING COUNT(*) > $1
        """
        
        results = await self.db.fetch(query, self.alert_thresholds["failed_logins"])
        
        for result in results:
            await self.send_security_alert(
                "brute_force_detected",
                f"IP {result['ip_address']} has {result['failed_count']} failed logins"
            )
    
    async def monitor_unusual_trading(self) -> None:
        """Monitor for unusual trading patterns"""
        
        # Check for large trades
        large_trades = await self.get_recent_large_trades()
        for trade in large_trades:
            await self.send_security_alert(
                "large_trade_detected",
                f"Large trade detected: {trade['amount']} USDC in {trade['symbol']}"
            )
        
        # Check position limits
        positions = await self.get_current_positions()
        for position in positions:
            if abs(position['value_usdc']) > self.alert_thresholds["position_limit"]:
                await self.send_security_alert(
                    "position_limit_exceeded", 
                    f"Position limit exceeded: {position['value_usdc']} USDC"
                )
    
    async def send_security_alert(self, alert_type: str, message: str) -> None:
        """Send security alert to monitoring systems"""
        
        alert_data = {
            "alert_type": alert_type,
            "message": message,
            "timestamp": datetime.utcnow().isoformat(),
            "severity": "high"
        }
        
        # Log alert
        logger.warning("security_alert", **alert_data)
        
        # Send to external monitoring (Grafana, Slack, etc.)
        await self.external_alerting.send_alert(alert_data)
```

## Incident Response

### Emergency Procedures
```python
class EmergencyResponse:
    async def emergency_shutdown(self, reason: str, initiated_by: str) -> None:
        """Emergency shutdown procedure"""
        
        logger.critical(f"EMERGENCY SHUTDOWN: {reason} (by: {initiated_by})")
        
        # 1. Stop all trading immediately
        await self.trading_engine.emergency_stop()
        
        # 2. Cancel all open orders
        await self.order_manager.cancel_all_orders()
        
        # 3. Pause data ingestion
        await self.data_ingestion.pause()
        
        # 4. Send alerts
        await self.alert_manager.send_critical_alert(
            f"EMERGENCY SHUTDOWN: {reason}"
        )
        
        # 5. Audit log
        await self.audit_logger.log_critical_event(
            "emergency_shutdown",
            initiated_by,
            {"reason": reason, "timestamp": datetime.utcnow().isoformat()}
        )
        
        # 6. Update system status
        await self.redis.set("system_status", "emergency_shutdown")
        
        logger.critical("Emergency shutdown completed")
    
    async def security_incident_response(self, incident_type: str, 
                                       details: Dict) -> None:
        """Handle security incidents"""
        
        logger.critical(f"SECURITY INCIDENT: {incident_type}")
        
        # Immediate containment
        if incident_type in ["key_compromise", "unauthorized_access"]:
            await self.emergency_shutdown(f"Security incident: {incident_type}", "security_system")
            
            # Rotate all keys immediately
            await self.key_manager.emergency_key_rotation()
        
        # Evidence preservation
        await self._preserve_evidence(incident_type, details)
        
        # Notification
        await self._notify_incident_response_team(incident_type, details)
    
    async def _preserve_evidence(self, incident_type: str, details: Dict) -> None:
        """Preserve evidence for forensic analysis"""
        
        evidence_package = {
            "incident_type": incident_type,
            "timestamp": datetime.utcnow().isoformat(),
            "details": details,
            "system_logs": await self._collect_system_logs(),
            "audit_trail": await self._collect_audit_trail(),
            "system_state": await self._collect_system_state()
        }
        
        # Store evidence securely
        evidence_path = f"/secure/evidence/{incident_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        encrypted_evidence = self.encryption.encrypt_sensitive_data(json.dumps(evidence_package))
        
        with open(evidence_path, 'w') as f:
            f.write(encrypted_evidence)
        
        logger.info(f"Evidence preserved: {evidence_path}")
```

## Compliance & Regulatory Considerations

### Data Retention Policy
```python
class DataRetentionManager:
    RETENTION_POLICIES = {
        "trading_data": timedelta(days=2555),  # 7 years
        "audit_logs": timedelta(days=2555),    # 7 years
        "user_data": timedelta(days=365),      # 1 year
        "system_logs": timedelta(days=90),     # 3 months
        "metrics_data": timedelta(days=30)     # 1 month
    }
    
    async def cleanup_expired_data(self) -> None:
        """Clean up data based on retention policies"""
        
        for data_type, retention_period in self.RETENTION_POLICIES.items():
            cutoff_date = datetime.utcnow() - retention_period
            
            try:
                deleted_count = await self._delete_expired_data(data_type, cutoff_date)
                logger.info(f"Cleaned up {deleted_count} {data_type} records older than {cutoff_date}")
                
            except Exception as e:
                logger.error(f"Failed to cleanup {data_type}: {e}")
```

This comprehensive security architecture provides multiple layers of protection while maintaining the operational requirements for high-frequency trading in a hackathon environment.
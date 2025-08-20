# FlashMM Operations Runbook

## Table of Contents
- [Overview](#overview)
- [Daily Operations](#daily-operations)
- [System Monitoring](#system-monitoring)
- [Incident Response](#incident-response)
- [Performance Management](#performance-management)
- [Backup and Recovery](#backup-and-recovery)
- [Security Operations](#security-operations)
- [Maintenance Procedures](#maintenance-procedures)
- [Emergency Procedures](#emergency-procedures)
- [Scaling and Capacity Planning](#scaling-and-capacity-planning)
- [Change Management](#change-management)
- [Troubleshooting Guide](#troubleshooting-guide)

---

## Overview

This operational runbook provides comprehensive procedures for maintaining, monitoring, and operating FlashMM in production environments. It serves as the primary reference for operations teams, SREs, and system administrators.

### Operational Objectives

- **Availability**: Maintain 99.9% uptime for trading operations
- **Performance**: Keep latency below 350ms end-to-end
- **Security**: Protect trading funds and sensitive data
- **Reliability**: Ensure consistent trading performance
- **Compliance**: Meet regulatory and audit requirements

### Operational Roles and Responsibilities

| Role | Responsibilities | Escalation |
|------|------------------|-------------|
| **L1 Operations** | Monitoring, basic troubleshooting, incident reporting | L2 Operations |
| **L2 Operations** | Advanced troubleshooting, system maintenance, deployments | L3/DevOps |
| **L3/DevOps** | Architecture changes, complex incidents, performance tuning | Engineering |
| **Security Team** | Security incidents, compliance, access management | CISO |
| **Trading Team** | Trading strategy, risk parameters, P&L review | CTO |

### Communication Channels

```
🚨 Emergency: Immediate phone call + Slack #incidents
⚠️  Critical: Slack #alerts + PagerDuty
📊 Monitoring: Slack #monitoring
📋 Operations: Slack #operations
📈 Trading: Slack #trading-ops
```

---

## Daily Operations

### Morning Checklist (9:00 AM UTC)

```bash
#!/bin/bash
# scripts/daily_morning_checklist.sh

echo "🌅 FlashMM Daily Morning Operations Checklist"
echo "============================================="

# 1. System Health Check
echo "🔍 1. Checking system health..."
./scripts/health-check.sh -e production -v

# 2. Trading Performance Review
echo "📊 2. Reviewing overnight trading performance..."
curl -s -H "Authorization: Bearer $API_TOKEN" \
  https://api.flashmm.com/api/v1/metrics/trading | jq '.summary'

# 3. P&L and Position Review
echo "💰 3. Checking P&L and positions..."
./scripts/pnl-report.sh --period=24h

# 4. System Resources Check
echo "⚡ 4. Checking system resources..."
kubectl top nodes
kubectl top pods -n flashmm

# 5. Security Check
echo "🔒 5. Running security checks..."
./scripts/security-check.sh --daily

# 6. Backup Verification
echo "💾 6. Verifying backups..."
./scripts/verify-backups.sh --check-last-24h

# 7. Alert Review
echo "🚨 7. Reviewing overnight alerts..."
./scripts/review-alerts.sh --since="24 hours ago"

echo "✅ Morning checklist completed"
```

### Business Hours Monitoring (9:00 AM - 6:00 PM UTC)

#### Hourly Tasks
- Monitor trading performance metrics
- Check system latency and throughput
- Review active alerts and warnings
- Verify ML model accuracy
- Monitor position utilization

#### Every 4 Hours
- Review P&L performance
- Check system resource utilization
- Validate backup completion
- Review security logs
- Test emergency procedures

### Evening Checklist (6:00 PM UTC)

```bash
#!/bin/bash
# scripts/daily_evening_checklist.sh

echo "🌇 FlashMM Daily Evening Operations Checklist"
echo "============================================="

# 1. Daily P&L Summary
echo "💰 1. Generating daily P&L summary..."
./scripts/pnl-report.sh --period=today --format=summary

# 2. Performance Summary
echo "📈 2. Generating performance summary..."
./scripts/performance-report.sh --period=today

# 3. System Cleanup
echo "🧹 3. Running system cleanup..."
./scripts/cleanup-logs.sh --older-than=7d
./scripts/cleanup-temp-files.sh

# 4. Backup Execution
echo "💾 4. Starting daily backup..."
./scripts/backup-production.sh

# 5. Security Summary
echo "🔒 5. Security summary..."
./scripts/security-summary.sh --period=today

# 6. Next Day Preparation
echo "📋 6. Preparing for next day..."
./scripts/prepare-next-day.sh

echo "✅ Evening checklist completed"
```

---

## System Monitoring

### Key Performance Indicators (KPIs)

#### Trading Performance Metrics

```bash
# Primary KPIs to monitor continuously
CRITICAL_METRICS=(
    "trading_pnl_usdc"                    # Target: Positive daily P&L
    "trading_spread_improvement_percent"   # Target: >40%
    "ml_prediction_accuracy"              # Target: >55%
    "system_latency_p95_ms"               # Target: <350ms
    "trading_uptime_percent"              # Target: >99.9%
    "position_utilization_percent"        # Target: <80%
)

WARNING_THRESHOLDS=(
    "trading_pnl_usdc < -50"              # Daily loss > $50
    "ml_prediction_accuracy < 0.50"       # Accuracy below 50%
    "system_latency_p95_ms > 300"         # Latency approaching limit
    "position_utilization_percent > 70"   # High position utilization
)

CRITICAL_THRESHOLDS=(
    "trading_pnl_usdc < -100"             # Daily loss > $100
    "ml_prediction_accuracy < 0.45"       # Very poor accuracy
    "system_latency_p95_ms > 400"         # Latency exceeds target
    "position_utilization_percent > 90"   # Near position limit
    "trading_uptime_percent < 95"         # Low uptime
)
```

#### System Health Monitoring

```yaml
# monitoring/health_checks.yml
health_checks:
  # Application health
  api_health:
    endpoint: "https://api.flashmm.com/health"
    timeout: 5s
    expected_status: 200
    check_interval: 30s
    
  # Database connectivity
  database_health:
    query: "SELECT 1"
    timeout: 5s
    check_interval: 60s
    
  # Redis connectivity
  redis_health:
    command: "PING"
    timeout: 2s
    check_interval: 30s
    
  # Sei network connectivity
  sei_health:
    endpoint: "https://sei-testnet-rpc.polkachu.com/status"
    timeout: 10s
    check_interval: 60s
    
  # ML model availability
  ml_health:
    test_prediction: true
    timeout: 10s
    check_interval: 300s

# Resource monitoring
resource_monitoring:
  cpu_usage:
    warning_threshold: 70
    critical_threshold: 85
    
  memory_usage:
    warning_threshold: 80
    critical_threshold: 90
    
  disk_usage:
    warning_threshold: 80
    critical_threshold: 90
    
  network_latency:
    warning_threshold: 200  # ms
    critical_threshold: 350  # ms
```

### Grafana Dashboard Monitoring

#### Primary Dashboard Panels

```json
{
  "dashboard_monitoring_schedule": {
    "continuous_monitoring": [
      "Trading P&L (Real-time)",
      "System Latency (P95)",
      "ML Prediction Accuracy",
      "Active Orders Count",
      "Position Utilization"
    ],
    
    "hourly_review": [
      "Trading Volume",
      "Fill Rate",
      "Spread Improvement",
      "System Resource Usage",
      "Error Rate"
    ],
    
    "daily_review": [
      "Daily P&L Summary",
      "Performance Trends",
      "System Uptime",
      "Security Events",
      "Capacity Utilization"
    ]
  }
}
```

#### Alert Rules Configuration

```yaml
# monitoring/alert_rules.yml
groups:
- name: flashmm.trading
  rules:
  - alert: TradingSystemDown
    expr: up{job="flashmm"} == 0
    for: 30s
    labels:
      severity: critical
      team: operations
    annotations:
      summary: "FlashMM trading system is down"
      description: "The FlashMM trading system has been down for more than 30 seconds"
      runbook: "https://docs.flashmm.com/operations#system-down"
      
  - alert: HighLatency
    expr: histogram_quantile(0.95, rate(flashmm_latency_seconds_bucket[5m])) > 0.35
    for: 2m
    labels:
      severity: warning
      team: operations
    annotations:
      summary: "High latency detected"
      description: "95th percentile latency is {{ $value }}s"
      
  - alert: LowMLAccuracy
    expr: avg_over_time(flashmm_prediction_accuracy[30m]) < 0.50
    for: 10m
    labels:
      severity: warning
      team: trading
    annotations:
      summary: "ML prediction accuracy is low"
      description: "ML accuracy is {{ $value }} over the last 30 minutes"
      
  - alert: HighPositionUtilization
    expr: flashmm_position_utilization_percent > 85
    for: 1m
    labels:
      severity: critical
      team: trading
    annotations:
      summary: "Position utilization is high"
      description: "Position utilization is {{ $value }}%"
      
  - alert: DailyLossThreshold
    expr: flashmm_daily_pnl_usdc < -100
    for: 0s
    labels:
      severity: critical
      team: trading
    annotations:
      summary: "Daily loss threshold exceeded"
      description: "Daily P&L is {{ $value }} USDC"
```

---

## Incident Response

### Incident Classification

| Severity | Definition | Response Time | Escalation |
|----------|------------|---------------|------------|
| **P0 - Critical** | Trading system down, major fund loss | Immediate | All hands |
| **P1 - High** | Degraded performance, minor fund loss | 15 minutes | L2 + Manager |
| **P2 - Medium** | System warnings, no immediate impact | 1 hour | L2 Operations |
| **P3 - Low** | Minor issues, maintenance items | 4 hours | L1 Operations |

### Incident Response Procedures

#### P0 - Critical Incident Response

```bash
#!/bin/bash
# scripts/incident_response_p0.sh

echo "🚨 P0 CRITICAL INCIDENT RESPONSE"
echo "==============================="

# 1. Immediate Assessment
echo "1. 📊 Immediate system assessment..."
./scripts/system-status.sh --emergency

# 2. Emergency Stop if Trading Loss
echo "2. 🛑 Checking for emergency stop conditions..."
DAILY_PNL=$(curl -s -H "Authorization: Bearer $API_TOKEN" \
  https://api.flashmm.com/api/v1/metrics/trading | jq -r '.summary.daily_pnl_usdc')

if (( $(echo "$DAILY_PNL < -200" | bc -l) )); then
    echo "🚨 CRITICAL LOSS DETECTED: $DAILY_PNL USDC"
    echo "Executing emergency stop..."
    ./scripts/emergency-stop.sh --reason="Critical loss threshold"
fi

# 3. Notification
echo "3. 📢 Sending critical alerts..."
./scripts/send-alert.sh --severity=P0 --message="Critical incident detected"

# 4. War Room Setup
echo "4. 🏛️ Setting up war room..."
./scripts/setup-war-room.sh

# 5. Data Collection
echo "5. 📋 Collecting incident data..."
mkdir -p ./incidents/$(date +%Y%m%d_%H%M%S)
./scripts/collect-incident-data.sh --output=./incidents/$(date +%Y%m%d_%H%M%S)

echo "✅ P0 incident response initiated"
echo "Next steps:"
echo "1. Join war room: https://meet.google.com/flashmm-war-room"
echo "2. Review incident data in ./incidents/"
echo "3. Follow incident commander instructions"
```

#### Incident War Room Procedures

```markdown
## War Room Protocol

### Roles
- **Incident Commander**: Coordinates response, makes decisions
- **Technical Lead**: Diagnoses technical issues
- **Communications Lead**: Manages stakeholder communications
- **Trading SME**: Assesses trading impact
- **Security SME**: Handles security aspects (if applicable)

### War Room Checklist
1. [ ] All key personnel joined
2. [ ] Current system status assessed
3. [ ] Impact scope determined
4. [ ] Emergency actions taken (if needed)
5. [ ] Root cause investigation started
6. [ ] Stakeholder communications sent
7. [ ] Fix implementation plan created
8. [ ] Fix deployed and verified
9. [ ] System restored to normal
10. [ ] Post-incident review scheduled

### Communication Templates
- **Initial Alert**: "P0 incident detected. War room active. Updates every 15 minutes."
- **Update**: "Status update: [current status]. ETA to resolution: [estimate]."
- **Resolution**: "Incident resolved. System restored. Post-mortem scheduled for [date]."
```

### Common Incident Scenarios

#### Scenario 1: High Latency

```bash
# Diagnosis and Resolution Steps
echo "🔍 High Latency Incident Response"

# 1. Check system resources
kubectl top nodes
kubectl top pods -n flashmm

# 2. Check network connectivity
./scripts/check-network-latency.sh

# 3. Check Sei network status
curl -s https://sei-testnet-rpc.polkachu.com/status | jq '.result.sync_info'

# 4. Check database performance
./scripts/check-db-performance.sh

# 5. Restart services if needed
if [[ "$LATENCY_P95" -gt 500 ]]; then
    echo "Restarting application pods..."
    kubectl rollout restart deployment/flashmm-app -n flashmm
fi
```

#### Scenario 2: ML Model Degradation

```bash
# ML Model Performance Issues
echo "🤖 ML Model Degradation Response"

# 1. Check model accuracy
ACCURACY=$(curl -s -H "Authorization: Bearer $API_TOKEN" \
  https://api.flashmm.com/api/v1/ml/model/info | jq -r '.performance.accuracy')

echo "Current model accuracy: $ACCURACY"

# 2. Switch to fallback if accuracy too low
if (( $(echo "$ACCURACY < 0.45" | bc -l) )); then
    echo "Switching to fallback engine..."
    ./scripts/switch-to-fallback.sh
fi

# 3. Check for model drift
./scripts/check-model-drift.sh

# 4. Reload model if needed
./scripts/reload-ml-model.sh --version=latest
```

#### Scenario 3: Position Limit Breach

```bash
# Position Limit Management
echo "⚖️ Position Limit Breach Response"

# 1. Check current positions
./scripts/check-positions.sh --detailed

# 2. Emergency position flattening if critical
POSITION_UTIL=$(curl -s -H "Authorization: Bearer $API_TOKEN" \
  https://api.flashmm.com/api/v1/trading/positions | jq -r '.portfolio_summary.risk_utilization_percent')

if (( $(echo "$POSITION_UTIL > 95" | bc -l) )); then
    echo "🚨 CRITICAL: Position utilization at $POSITION_UTIL%"
    echo "Initiating emergency position flattening..."
    ./scripts/flatten-positions.sh --emergency
fi

# 3. Adjust risk parameters
./scripts/adjust-risk-limits.sh --reduce-by=20
```

---

## Performance Management

### Performance Monitoring Framework

#### Latency Monitoring

```python
# scripts/monitor_latency.py
import asyncio
import time
import statistics
from typing import List, Dict
import aiohttp

class LatencyMonitor:
    """Monitor end-to-end system latency."""
    
    def __init__(self):
        self.api_base = "https://api.flashmm.com"
        self.measurements = []
        
    async def measure_end_to_end_latency(self) -> float:
        """Measure complete trading cycle latency."""
        start_time = time.perf_counter()
        
        async with aiohttp.ClientSession() as session:
            # 1. Get market data
            async with session.get(f"{self.api_base}/api/v1/trading/status") as resp:
                await resp.json()
            
            # 2. Get ML prediction (simulated)
            async with session.get(f"{self.api_base}/api/v1/ml/predictions?symbol=SEI/USDC&limit=1") as resp:
                await resp.json()
            
        end_time = time.perf_counter()
        return (end_time - start_time) * 1000  # Convert to milliseconds
    
    async def run_continuous_monitoring(self, duration_minutes: int = 60):
        """Run continuous latency monitoring."""
        end_time = time.time() + (duration_minutes * 60)
        
        while time.time() < end_time:
            try:
                latency = await self.measure_end_to_end_latency()
                self.measurements.append({
                    'timestamp': time.time(),
                    'latency_ms': latency
                })
                
                # Alert if latency exceeds threshold
                if latency > 350:
                    print(f"⚠️ HIGH LATENCY ALERT: {latency:.2f}ms")
                
                await asyncio.sleep(10)  # Measure every 10 seconds
                
            except Exception as e:
                print(f"❌ Latency measurement failed: {e}")
                await asyncio.sleep(5)
        
        # Generate report
        self.generate_latency_report()
    
    def generate_latency_report(self):
        """Generate latency performance report."""
        if not self.measurements:
            return
        
        latencies = [m['latency_ms'] for m in self.measurements]
        
        report = {
            'total_measurements': len(latencies),
            'avg_latency_ms': statistics.mean(latencies),
            'median_latency_ms': statistics.median(latencies),
            'p95_latency_ms': statistics.quantiles(latencies, n=20)[18],
            'p99_latency_ms': statistics.quantiles(latencies, n=100)[98],
            'max_latency_ms': max(latencies),
            'violations_over_350ms': len([l for l in latencies if l > 350])
        }
        
        print("📊 Latency Performance Report:")
        for key, value in report.items():
            print(f"  {key}: {value:.2f}" if isinstance(value, float) else f"  {key}: {value}")

if __name__ == "__main__":
    monitor = LatencyMonitor()
    asyncio.run(monitor.run_continuous_monitoring(60))
```

#### Performance Optimization Procedures

```bash
#!/bin/bash
# scripts/optimize_performance.sh

echo "⚡ FlashMM Performance Optimization"
echo "==================================="

# 1. Database Optimization
echo "💾 1. Optimizing database performance..."
./scripts/optimize-database.sh

# 2. Redis Optimization
echo "📦 2. Optimizing Redis performance..."
./scripts/optimize-redis.sh

# 3. Application Optimization
echo "🐍 3. Optimizing application performance..."
./scripts/optimize-application.sh

# 4. Network Optimization
echo "🌐 4. Optimizing network performance..."
./scripts/optimize-network.sh

# 5. System Resource Optimization
echo "⚙️ 5. Optimizing system resources..."
./scripts/optimize-system-resources.sh

echo "✅ Performance optimization completed"
```

### Capacity Planning

#### Resource Utilization Monitoring

```yaml
# Resource monitoring thresholds
resource_thresholds:
  cpu:
    warning: 70%
    critical: 85%
    scaling_trigger: 80%
    
  memory:
    warning: 75%
    critical: 90%
    scaling_trigger: 85%
    
  disk:
    warning: 80%
    critical: 90%
    cleanup_trigger: 85%
    
  network:
    warning: 70%
    critical: 85%
    
  database_connections:
    warning: 80%
    critical: 95%
    
  redis_memory:
    warning: 80%
    critical: 90%
```

#### Scaling Procedures

```bash
#!/bin/bash
# scripts/scale_system.sh

SCALE_ACTION=$1  # scale-up, scale-down, auto-scale

case $SCALE_ACTION in
    "scale-up")
        echo "📈 Scaling up FlashMM system..."
        
        # Increase Kubernetes replicas
        kubectl scale deployment flashmm-app -n flashmm --replicas=5
        
        # Increase database connections
        kubectl patch configmap flashmm-config -n flashmm \
          --patch='{"data":{"DATABASE_MAX_CONNECTIONS":"50"}}'
        
        # Increase Redis memory
        kubectl patch configmap redis-config -n flashmm \
          --patch='{"data":{"maxmemory":"2gb"}}'
        ;;
        
    "scale-down")
        echo "📉 Scaling down FlashMM system..."
        
        # Decrease Kubernetes replicas
        kubectl scale deployment flashmm-app -n flashmm --replicas=2
        
        # Reduce database connections
        kubectl patch configmap flashmm-config -n flashmm \
          --patch='{"data":{"DATABASE_MAX_CONNECTIONS":"20"}}'
        ;;
        
    "auto-scale")
        echo "🔄 Enabling auto-scaling..."
        
        # Apply HPA configuration
        kubectl apply -f k8s/hpa.yaml
        
        # Monitor scaling events
        kubectl get hpa -n flashmm -w
        ;;
esac
```

---

## Backup and Recovery

### Backup Strategy

#### Daily Backup Procedures

```bash
#!/bin/bash
# scripts/backup-production.sh

BACKUP_DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups/flashmm"
S3_BUCKET="flashmm-backups-production"

echo "💾 Starting FlashMM Production Backup - $BACKUP_DATE"
echo "=================================================="

# 1. Database Backup
echo "📊 1. Backing up PostgreSQL database..."
kubectl exec -n flashmm deployment/postgresql -- \
    pg_dump -U flashmm flashmm | gzip > "$BACKUP_DIR/db_$BACKUP_DATE.sql.gz"

# 2. Redis Backup
echo "📦 2. Backing up Redis data..."
kubectl exec -n flashmm deployment/redis -- redis-cli BGSAVE
sleep 10  # Wait for background save to complete
kubectl cp flashmm/$(kubectl get pods -n flashmm -l app=redis -o jsonpath='{.items[0].metadata.name}'):/data/dump.rdb \
    "$BACKUP_DIR/redis_$BACKUP_DATE.rdb"

# 3. Configuration Backup
echo "⚙️ 3. Backing up configurations..."
kubectl get configmaps -n flashmm -o yaml > "$BACKUP_DIR/configmaps_$BACKUP_DATE.yaml"
kubectl get secrets -n flashmm -o yaml > "$BACKUP_DIR/secrets_$BACKUP_DATE.yaml"

# 4. ML Models Backup
echo "🤖 4. Backing up ML models..."
tar -czf "$BACKUP_DIR/models_$BACKUP_DATE.tar.gz" -C /app/models .

# 5. Application Logs Backup (last 24 hours)
echo "📋 5. Backing up recent logs..."
kubectl logs -n flashmm --since=24h -l app.kubernetes.io/name=flashmm > \
    "$BACKUP_DIR/logs_$BACKUP_DATE.txt"

# 6. Upload to S3
echo "☁️ 6. Uploading backups to S3..."
aws s3 sync "$BACKUP_DIR/" "s3://$S3_BUCKET/daily/$BACKUP_DATE/"

# 7. Cleanup old local backups (keep last 7 days)
echo "🧹 7. Cleaning up old backups..."
find "$BACKUP_DIR" -name "*" -type f -mtime +7 -delete

# 8. Verify backup integrity
echo "✅ 8. Verifying backup integrity..."
./scripts/verify-backup-integrity.sh "$BACKUP_DIR" "$BACKUP_DATE"

# 9. Update backup inventory
echo "📝 9. Updating backup inventory..."
echo "$BACKUP_DATE,$(date),$(du -sh $BACKUP_DIR | cut -f1)" >> "$BACKUP_DIR/backup_inventory.csv"

echo "✅ Backup completed successfully: $BACKUP_DATE"

# 10. Send notification
./scripts/send-notification.sh --type=backup-complete --backup-id="$BACKUP_DATE"
```

#### Recovery Procedures

```bash
#!/bin/bash
# scripts/restore-production.sh

BACKUP_DATE=${1:-latest}
BACKUP_DIR="/backups/flashmm"
S3_BUCKET="flashmm-backups-production"

echo "🔄 Starting FlashMM Production Recovery - $BACKUP_DATE"
echo "==================================================="

# 1. Download backup from S3 if needed
if [[ "$BACKUP_DATE" == "latest" ]]; then
    BACKUP_DATE=$(aws s3 ls "s3://$S3_BUCKET/daily/" | tail -1 | awk '{print $2}' | tr -d '/')
fi

echo "📦 Using backup: $BACKUP_DATE"

if [[ ! -d "$BACKUP_DIR/$BACKUP_DATE" ]]; then
    echo "⬇️ Downloading backup from S3..."
    aws s3 sync "s3://$S3_BUCKET/daily/$BACKUP_DATE/" "$BACKUP_DIR/$BACKUP_DATE/"
fi

# 2. Confirmation prompt
read -p "⚠️ This will restore production data. Continue? (yes/no): " -r
if [[ ! $REPLY =~ ^yes$ ]]; then
    echo "❌ Recovery cancelled"
    exit 1
fi

# 3. Scale down application
echo "⬇️ Scaling down application..."
kubectl scale deployment flashmm-app -n flashmm --replicas=0

# 4. Restore database
echo "📊 Restoring PostgreSQL database..."
gunzip -c "$BACKUP_DIR/$BACKUP_DATE/db_$BACKUP_DATE.sql.gz" | \
kubectl exec -i -n flashmm deployment/postgresql -- psql -U flashmm flashmm

# 5. Restore Redis
echo "📦 Restoring Redis data..."
kubectl cp "$BACKUP_DIR/$BACKUP_DATE/redis_$BACKUP_DATE.rdb" \
    flashmm/$(kubectl get pods -n flashmm -l app=redis -o jsonpath='{.items[0].metadata.name}'):/data/dump.rdb
kubectl exec -n flashmm deployment/redis -- redis-cli DEBUG RELOAD

# 6. Restore configurations
echo "⚙️ Restoring configurations..."
kubectl apply -f "$BACKUP_DIR/$BACKUP_DATE/configmaps_$BACKUP_DATE.yaml"
kubectl apply -f "$BACKUP_DIR/$BACKUP_DATE/secrets_$BACKUP_DATE.yaml"

# 7. Restore ML models
echo "🤖 Restoring ML models..."
kubectl exec -n flashmm deployment/flashmm-app -- rm -rf /app/models/*
tar -xzf "$BACKUP_DIR/$BACKUP_DATE/models_$BACKUP_DATE.tar.gz" -C /tmp/
kubectl cp /tmp/models flashmm/$(kubectl get pods -n flashmm -l app.kubernetes.io/name=flashmm -o jsonpath='{.items[0].metadata.name}'):/app/

# 8. Scale up application
echo "⬆️ Scaling up application..."
kubectl scale deployment flashmm-app -n flashmm --replicas=3

# 9. Wait for pods to be ready
echo "⏳ Waiting for pods to be ready..."
kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=flashmm -n flashmm --timeout=300s

# 10. Verify recovery
echo "✅ Verifying recovery..."
./scripts/health-check.sh -e production

echo "✅ Recovery completed successfully"
```

### Disaster Recovery

#### Disaster Recovery Plan

```yaml
# disaster_recovery_plan.yml
disaster_recovery:
  scenarios:
    - name: "Complete Data Center Failure"
      rto: "4 hours"  # Recovery Time Objective
      rpo: "1 hour"   # Recovery Point Objective
      
      steps:
        1. "Activate secondary AWS region"
        2. "Restore from S3 cross-region backup"
        3. "Update DNS to point to DR environment"
        4. "Verify all services operational"
        5. "Resume trading operations"
    
    - name: "Database Corruption"
      rto: "2 hours"
      rpo: "15 minutes"
      
      steps:
        1. "Stop application to prevent further corruption"
        2. "Restore database from latest backup"
        3. "Replay transaction logs if available"
        4. "Verify data integrity"
        5. "Resume operations"
    
    - name: "Security Breach"
      rto: "1 hour"
      rpo: "Immediate"
      
      steps:
        1. "Isolate affected systems"
        2. "Revoke all API keys and tokens"
        3. "Change all passwords and secrets"
        4. "Restore from clean backup"
        5. "Implement additional security measures"
        6. "Resume operations after security clearance"

  emergency_contacts:
    - role: "Incident Commander"
      primary: "+1-555-0101"
      secondary: "+1-555-0102"
    
    - role: "Technical Lead"
      primary: "+1-555-0201"
      secondary: "+1-555-0202"
    
    - role: "Security Lead"
      primary: "+1-555-0301"
      secondary:
"+1-555-0302"

  runbook_locations:
    primary: "https://docs.flashmm.com/operations"
    backup: "https://wiki.flashmm.com/operations"
    offline: "/opt/flashmm/docs/operations/"
```

---

## Security Operations

### Security Monitoring

#### Daily Security Checks

```bash
#!/bin/bash
# scripts/daily_security_check.sh

echo "🔒 Daily Security Check"
echo "======================"

# 1. Check for unauthorized login attempts
echo "👤 1. Checking unauthorized access attempts..."
./scripts/check-failed-logins.sh --last-24h

# 2. API key usage audit
echo "🔑 2. Auditing API key usage..."
./scripts/audit-api-keys.sh --suspicious-activity

# 3. Network security scan
echo "🌐 3. Running network security scan..."
./scripts/network-security-scan.sh

# 4. Vulnerability scan
echo "🔍 4. Running vulnerability scan..."
./scripts/vulnerability-scan.sh --critical-only

# 5. Certificate expiry check
echo "📜 5. Checking certificate expiry..."
./scripts/check-cert-expiry.sh --warn-days=30

# 6. Security patch status
echo "🛡️ 6. Checking security patch status..."
./scripts/check-security-patches.sh

echo "✅ Daily security check completed"
```

#### Security Incident Response

```bash
#!/bin/bash
# scripts/security_incident_response.sh

INCIDENT_TYPE=$1  # breach, suspicious-activity, key-compromise

echo "🚨 Security Incident Response: $INCIDENT_TYPE"
echo "=============================================="

case $INCIDENT_TYPE in
    "breach")
        echo "🔒 Data breach response protocol..."
        
        # 1. Isolate systems
        ./scripts/isolate-systems.sh
        
        # 2. Revoke all active sessions
        ./scripts/revoke-all-sessions.sh
        
        # 3. Change all passwords and keys
        ./scripts/rotate-all-credentials.sh --emergency
        
        # 4. Enable additional logging
        ./scripts/enable-enhanced-logging.sh
        
        # 5. Notify stakeholders
        ./scripts/notify-security-incident.sh --type=breach
        ;;
        
    "suspicious-activity")
        echo "👁️ Suspicious activity response..."
        
        # 1. Increase monitoring
        ./scripts/increase-monitoring-sensitivity.sh
        
        # 2. Block suspicious IPs
        ./scripts/block-suspicious-ips.sh
        
        # 3. Audit recent activity
        ./scripts/audit-recent-activity.sh --detailed
        ;;
        
    "key-compromise")
        echo "🔑 Key compromise response..."
        
        # 1. Revoke compromised keys
        ./scripts/revoke-compromised-keys.sh
        
        # 2. Generate new keys
        ./scripts/generate-new-keys.sh
        
        # 3. Update applications
        ./scripts/update-application-keys.sh
        
        # 4. Audit key usage
        ./scripts/audit-key-usage.sh --compromised-period
        ;;
esac

echo "✅ Security incident response completed"
```

---

## Maintenance Procedures

### Scheduled Maintenance

#### Weekly Maintenance (Sundays 2:00 AM UTC)

```bash
#!/bin/bash
# scripts/weekly_maintenance.sh

echo "🔧 Weekly Maintenance Procedure"
echo "==============================="

# 1. System updates
echo "📦 1. Applying system updates..."
./scripts/apply-system-updates.sh --non-critical

# 2. Database maintenance
echo "💾 2. Database maintenance..."
./scripts/database-maintenance.sh --vacuum --analyze --reindex

# 3. Log rotation and cleanup
echo "📋 3. Log maintenance..."
./scripts/rotate-logs.sh
./scripts/cleanup-old-logs.sh --older-than=30d

# 4. Certificate renewal check
echo "📜 4. Certificate maintenance..."
./scripts/renew-certificates.sh --check-expiry

# 5. Performance optimization
echo "⚡ 5. Performance optimization..."
./scripts/optimize-database.sh --weekly
./scripts/cleanup-temp-files.sh

# 6. Security scan
echo "🔒 6. Security maintenance..."
./scripts/security-scan.sh --full

# 7. Backup verification
echo "💾 7. Backup verification..."
./scripts/verify-backups.sh --test-restore

echo "✅ Weekly maintenance completed"
```

#### Monthly Maintenance (First Sunday 1:00 AM UTC)

```bash
#!/bin/bash
# scripts/monthly_maintenance.sh

echo "🔧 Monthly Maintenance Procedure"
echo "================================"

# 1. Full system backup
echo "💾 1. Full system backup..."
./scripts/full-system-backup.sh

# 2. Security audit
echo "🔒 2. Security audit..."
./scripts/comprehensive-security-audit.sh

# 3. Performance review
echo "📊 3. Performance review..."
./scripts/performance-analysis.sh --monthly

# 4. Capacity planning review
echo "📈 4. Capacity planning..."
./scripts/capacity-planning-review.sh

# 5. Dependency updates
echo "📦 5. Dependency updates..."
./scripts/update-dependencies.sh --security-only

# 6. Documentation update
echo "📚 6. Documentation review..."
./scripts/update-runbook.sh --check-accuracy

echo "✅ Monthly maintenance completed"
```

### Emergency Maintenance

```bash
#!/bin/bash
# scripts/emergency_maintenance.sh

MAINTENANCE_TYPE=$1  # security-patch, critical-bug, performance-issue

echo "🚨 Emergency Maintenance: $MAINTENANCE_TYPE"
echo "=========================================="

# 1. Announce maintenance window
echo "📢 1. Announcing maintenance..."
./scripts/announce-maintenance.sh --type=emergency --duration=60

# 2. Create maintenance mode
echo "🔧 2. Enabling maintenance mode..."
./scripts/enable-maintenance-mode.sh

# 3. Perform emergency maintenance
case $MAINTENANCE_TYPE in
    "security-patch")
        ./scripts/apply-security-patch.sh --emergency
        ;;
    "critical-bug")
        ./scripts/deploy-hotfix.sh --critical
        ;;
    "performance-issue")
        ./scripts/emergency-performance-fix.sh
        ;;
esac

# 4. Verify fix
echo "✅ 4. Verifying fix..."
./scripts/verify-emergency-fix.sh

# 5. Exit maintenance mode
echo "🔧 5. Disabling maintenance mode..."
./scripts/disable-maintenance-mode.sh

# 6. Post-maintenance verification
echo "🔍 6. Post-maintenance checks..."
./scripts/post-maintenance-verification.sh

echo "✅ Emergency maintenance completed"
```

---

## Emergency Procedures

### Emergency Stop Procedures

```bash
#!/bin/bash
# scripts/emergency_stop.sh

REASON=${1:-"Manual emergency stop"}
STOP_TYPE=${2:-"graceful"}  # graceful, immediate

echo "🚨 EMERGENCY STOP INITIATED"
echo "=========================="
echo "Reason: $REASON"
echo "Type: $STOP_TYPE"

# 1. Immediate notification
echo "📢 1. Sending emergency notifications..."
./scripts/send-emergency-alert.sh --reason="$REASON"

# 2. Stop trading immediately
echo "🛑 2. Stopping all trading operations..."
if [[ "$STOP_TYPE" == "immediate" ]]; then
    # Kill all trading processes immediately
    kubectl delete pods -n flashmm -l component=trading --grace-period=0 --force
else
    # Graceful stop
    curl -X POST -H "Authorization: Bearer $EMERGENCY_TOKEN" \
      https://api.flashmm.com/api/v1/admin/emergency-stop \
      -d '{"confirmation": "EMERGENCY_STOP_CONFIRMED", "reason": "'$REASON'"}'
fi

# 3. Cancel all active orders
echo "❌ 3. Cancelling all active orders..."
./scripts/cancel-all-orders.sh --emergency

# 4. Flatten positions if requested
read -p "Flatten all positions? (y/N): " -n 1 -r
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "⚖️ Flattening all positions..."
    ./scripts/flatten-positions.sh --market-orders
fi

# 5. Secure system state
echo "🔒 5. Securing system state..."
./scripts/secure-system-state.sh

# 6. Create incident record
echo "📋 6. Creating incident record..."
./scripts/create-incident-record.sh --type=emergency-stop --reason="$REASON"

echo "✅ Emergency stop completed"
echo "System is now in safe state"
echo "Next steps:"
echo "1. Investigate root cause"
echo "2. Implement fixes"
echo "3. Test thoroughly"
echo "4. Resume operations when safe"
```

### System Recovery Procedures

```bash
#!/bin/bash
# scripts/system_recovery.sh

RECOVERY_TYPE=${1:-"standard"}  # standard, emergency, partial

echo "🔄 System Recovery Procedure"
echo "============================"
echo "Recovery type: $RECOVERY_TYPE"

# 1. System health assessment
echo "🔍 1. Assessing system health..."
./scripts/comprehensive-health-check.sh --recovery-mode

# 2. Verify data integrity
echo "🔐 2. Verifying data integrity..."
./scripts/verify-data-integrity.sh

# 3. Check external dependencies
echo "🌐 3. Checking external services..."
./scripts/check-external-dependencies.sh

# 4. Restore from backup if needed
if [[ "$RECOVERY_TYPE" == "emergency" ]]; then
    echo "💾 4. Emergency restore from backup..."
    ./scripts/restore-production.sh --latest --emergency
fi

# 5. Restart services
echo "🚀 5. Restarting services..."
case $RECOVERY_TYPE in
    "standard")
        kubectl rollout restart deployment/flashmm-app -n flashmm
        ;;
    "emergency")
        ./scripts/emergency-restart.sh
        ;;
    "partial")
        ./scripts/partial-restart.sh --failed-components-only
        ;;
esac

# 6. Verify recovery
echo "✅ 6. Verifying recovery..."
./scripts/verify-recovery.sh --comprehensive

# 7. Resume trading operations
read -p "Resume trading operations? (y/N): " -n 1 -r
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "▶️ Resuming trading operations..."
    ./scripts/resume-trading.sh --verify-first
fi

echo "✅ System recovery completed"
```

---

## Scaling and Capacity Planning

### Capacity Monitoring

```python
# scripts/capacity_monitor.py
import psutil
import kubernetes
import json
from datetime import datetime, timedelta

class CapacityMonitor:
    """Monitor system capacity and recommend scaling actions."""
    
    def __init__(self):
        self.k8s_apps_v1 = kubernetes.client.AppsV1Api()
        self.k8s_core_v1 = kubernetes.client.CoreV1Api()
        
    def get_resource_utilization(self):
        """Get current resource utilization."""
        # Get pod metrics
        pods = self.k8s_core_v1.list_namespaced_pod(namespace="flashmm")
        
        utilization = {
            'timestamp': datetime.utcnow().isoformat(),
            'pods': [],
            'total_cpu_usage': 0,
            'total_memory_usage': 0,
            'pod_count': len(pods.items)
        }
        
        for pod in pods.items:
            if pod.status.phase == "Running":
                # Get pod metrics (would require metrics server)
                pod_metrics = self.get_pod_metrics(pod.metadata.name)
                utilization['pods'].append({
                    'name': pod.metadata.name,
                    'cpu_usage': pod_metrics.get('cpu', 0),
                    'memory_usage': pod_metrics.get('memory', 0),
                    'status': pod.status.phase
                })
        
        return utilization
    
    def recommend_scaling(self, utilization_data):
        """Recommend scaling actions based on utilization."""
        recommendations = []
        
        avg_cpu = utilization_data['total_cpu_usage'] / utilization_data['pod_count']
        avg_memory = utilization_data['total_memory_usage'] / utilization_data['pod_count']
        
        # CPU-based recommendations
        if avg_cpu > 80:
            recommendations.append({
                'type': 'scale_up',
                'reason': f'High CPU utilization: {avg_cpu}%',
                'action': 'Increase pod replicas',
                'priority': 'high'
            })
        elif avg_cpu < 30 and utilization_data['pod_count'] > 2:
            recommendations.append({
                'type': 'scale_down',
                'reason': f'Low CPU utilization: {avg_cpu}%',
                'action': 'Decrease pod replicas',
                'priority': 'low'
            })
        
        # Memory-based recommendations
        if avg_memory > 85:
            recommendations.append({
                'type': 'scale_up',
                'reason': f'High memory utilization: {avg_memory}%',
                'action': 'Increase memory limits or pod replicas',
                'priority': 'high'
            })
        
        return recommendations
    
    def generate_capacity_report(self):
        """Generate comprehensive capacity report."""
        utilization = self.get_resource_utilization()
        recommendations = self.recommend_scaling(utilization)
        
        report = {
            'timestamp': datetime.utcnow().isoformat(),
            'current_utilization': utilization,
            'scaling_recommendations': recommendations,
            'capacity_forecast': self.forecast_capacity(),
            'cost_implications': self.estimate_costs()
        }
        
        return report

if __name__ == "__main__":
    monitor = CapacityMonitor()
    report = monitor.generate_capacity_report()
    print(json.dumps(report, indent=2))
```

### Auto-scaling Configuration

```yaml
# k8s/hpa-advanced.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: flashmm-hpa
  namespace: flashmm
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: flashmm-app
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  - type: Pods
    pods:
      metric:
        name: trading_latency_p95_milliseconds
      target:
        type: AverageValue
        averageValue: "300"
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
      - type: Pods
        value: 2
        periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
```

---

## Change Management

### Change Process

#### Standard Change Process

```bash
#!/bin/bash
# scripts/standard_change.sh

CHANGE_ID=$1
CHANGE_DESCRIPTION=$2
SCHEDULED_TIME=$3

echo "📋 Standard Change Process"
echo "========================="
echo "Change ID: $CHANGE_ID"
echo "Description: $CHANGE_DESCRIPTION"
echo "Scheduled: $SCHEDULED_TIME"

# 1. Pre-change validation
echo "✅ 1. Pre-change validation..."
./scripts/pre-change-validation.sh --change-id=$CHANGE_ID

# 2. Backup current state
echo "💾 2. Creating pre-change backup..."
./scripts/create-change-backup.sh --change-id=$CHANGE_ID

# 3. Implement change
echo "🔧 3. Implementing change..."
./scripts/implement-change.sh --change-id=$CHANGE_ID

# 4. Post-change validation
echo "✅ 4. Post-change validation..."
./scripts/post-change-validation.sh --change-id=$CHANGE_ID

# 5. Update documentation
echo "📚 5. Updating documentation..."
./scripts/update-change-documentation.sh --change-id=$CHANGE_ID

echo "✅ Standard change completed: $CHANGE_ID"
```

#### Emergency Change Process

```bash
#!/bin/bash
# scripts/emergency_change.sh

CHANGE_DESCRIPTION=$1
APPROVER=$2

echo "🚨 Emergency Change Process"
echo "============================"
echo "Description: $CHANGE_DESCRIPTION"
echo "Approved by: $APPROVER"

# 1. Create emergency change record
CHANGE_ID="EMRG-$(date +%Y%m%d-%H%M%S)"
echo "📋 1. Creating emergency change record: $CHANGE_ID"
./scripts/create-change-record.sh --emergency --id=$CHANGE_ID --description="$CHANGE_DESCRIPTION"

# 2. Notify stakeholders
echo "📢 2. Notifying stakeholders..."
./scripts/notify-emergency-change.sh --change-id=$CHANGE_ID

# 3. Quick backup
echo "💾 3. Quick backup..."
./scripts/quick-backup.sh --change-id=$CHANGE_ID

# 4. Implement emergency change
echo "🔧 4. Implementing emergency change..."
./scripts/implement-emergency-change.sh --change-id=$CHANGE_ID

# 5. Immediate validation
echo "✅ 5. Immediate validation..."
./scripts/immediate-validation.sh --change-id=$CHANGE_ID

# 6. Schedule post-implementation review
echo "📅 6. Scheduling post-implementation review..."
./scripts/schedule-pir.sh --change-id=$CHANGE_ID --within=24h

echo "✅ Emergency change completed: $CHANGE_ID"
```

---

## Troubleshooting Guide

### Common Issues and Solutions

#### Issue 1: High Latency

```bash
# Diagnosis and Resolution
echo "🔍 Troubleshooting High Latency"
echo "=============================="

# Step 1: Identify latency source
echo "1. Identifying latency source..."
./scripts/trace-latency.sh --detailed

# Step 2: Check system resources
echo "2. Checking system resources..."
kubectl top nodes
kubectl top pods -n flashmm

# Step 3: Check network connectivity
echo "3. Checking network connectivity..."
./scripts/network-diagnostics.sh

# Step 4: Check external dependencies
echo "4. Checking external dependencies..."
./scripts/check-external-latency.sh

# Step 5: Apply fixes based on findings
echo "5. Applying appropriate fixes..."
if [[ $(kubectl top nodes | awk 'NR>1 {print $3}' | sed 's/%//g' | head -1) -gt 80 ]]; then
    echo "High CPU detected - scaling up..."
    kubectl scale deployment flashmm-app -n flashmm --replicas=5
fi
```

#### Issue 2: ML Model Performance Degradation

```bash
# ML Performance Troubleshooting
echo "🤖 Troubleshooting ML Performance"
echo "================================="

# Check model accuracy
CURRENT_ACCURACY=$(curl -s -H "Authorization: Bearer $API_TOKEN" \
  https://api.flashmm.com/api/v1/ml/model/info | jq -r '.performance.accuracy')

echo "Current accuracy: $CURRENT_ACCURACY"

if (( $(echo "$CURRENT_ACCURACY < 0.50" | bc -l) )); then
    echo "Low accuracy detected - investigating..."
    
    # Check for data quality issues
    ./scripts/check-data-quality.sh
    
    # Check for model drift
    ./scripts/detect-model-drift.sh
    
    # Switch to fallback if necessary
    if (( $(echo "$CURRENT_ACCURACY < 0.45" | bc -l) )); then
        echo "Switching to fallback engine..."
        ./scripts/enable-fallback-engine.sh
    fi
fi
```

#### Issue 3: Database Connection Issues

```bash
# Database Troubleshooting
echo "💾 Troubleshooting Database Issues"
echo "=================================="

# Test database connectivity
if ! pg_isready -h $DB_HOST -p $DB_PORT -U $DB_USER; then
    echo "Database not reachable - checking alternatives..."
    
    # Check if it's a connection pool issue
    ./scripts/check-connection-pool.sh
    
    # Try connecting to read replica
    ./scripts/test-read-replica.sh
    
    # Restart database if necessary
    kubectl rollout restart statefulset/postgresql -n flashmm
fi
```

### Performance Troubleshooting

```python
# scripts/performance_diagnostics.py
import asyncio
import time
import psutil
import aiohttp
from typing import Dict, List

class PerformanceDiagnostics:
    """Comprehensive performance diagnostics tool."""
    
    async def diagnose_system_performance(self) -> Dict:
        """Run comprehensive performance diagnostics."""
        results = {
            'timestamp': time.time(),
            'system_metrics': self.get_system_metrics(),
            'application_metrics': await self.get_application_metrics(),
            'network_metrics': await self.get_network_metrics(),
            'recommendations': []
        }
        
        # Analyze results and generate recommendations
        results['recommendations'] = self.generate_recommendations(results)
        
        return results
    
    def get_system_metrics(self) -> Dict:
        """Get system-level performance metrics."""
        return {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_io': psutil.disk_io_counters()._asdict(),
            'network_io': psutil.net_io_counters()._asdict(),
            'load_average': psutil.getloadavg()
        }
    
    async def get_application_metrics(self) -> Dict:
        """Get application-specific metrics."""
        async with aiohttp.ClientSession() as session:
            async with session.get("http://localhost:8000/metrics") as resp:
                metrics_text = await resp.text()
                return self.parse_prometheus_metrics(metrics_text)
    
    def generate_recommendations(self, results: Dict) -> List[str]:
        """Generate performance improvement recommendations."""
        recommendations = []
        
        # CPU recommendations
        if results['system_metrics']['cpu_percent'] > 80:
            recommendations.append("High CPU usage detected - consider scaling up")
        
        # Memory recommendations
        if results['system_metrics']['memory_percent'] > 85:
            recommendations.append("High memory usage - check for memory leaks")
        
        # Application-specific recommendations
        app_metrics = results['application_metrics']
        if app_metrics.get('latency_p95', 0) > 350:
            recommendations.append("High latency detected - investigate bottlenecks")
        
        return recommendations

if __name__ == "__main__":
    diagnostics = PerformanceDiagnostics()
    results = asyncio.run(diagnostics.diagnose_system_performance())
    
    print("🔍 Performance Diagnostics Results")
    print("==================================")
    for rec in results['recommendations']:
        print(f"⚠️  {rec}")
```

---

## Conclusion

This comprehensive operations runbook provides:

### Key Operational Procedures

1. **Daily Operations**: Morning and evening checklists, continuous monitoring
2. **System Monitoring**: KPI tracking, health checks, performance monitoring
3. **Incident Response**: Classification, procedures, war room protocols
4. **Performance Management**: Latency monitoring, optimization, capacity planning
5. **Backup and Recovery**: Daily backups, disaster recovery, system restoration
6. **Security Operations**: Daily checks, incident response, threat monitoring
7. **Maintenance**: Scheduled and emergency maintenance procedures
8. **Emergency Procedures**: System stops, recovery, crisis management
9. **Scaling**: Auto-scaling, capacity monitoring, resource optimization
10. **Change Management**: Standard and emergency change processes
11. **Troubleshooting**: Common issues, diagnostic tools, resolution steps

### Operational Excellence Features

- **Automated Monitoring**: Continuous health and performance monitoring
- **Proactive Alerts**: Multi-tier alerting with appropriate escalation
- **Documented Procedures**: Step-by-step runbooks for all scenarios
- **Emergency Response**: Well-defined incident response and recovery procedures
- **Comprehensive Backups**: Multiple backup strategies with tested recovery
- **Security Operations**: Continuous security monitoring and incident response
- **Performance Optimization**: Automated scaling and performance tuning

### Support Resources

- **[Architecture Documentation](ARCHITECTURE.md)**: Understanding system design
- **[Deployment Guide](DEPLOYMENT_GUIDE.md)**: Infrastructure and deployment
- **[Configuration Reference](CONFIGURATION.md)**: System configuration options
- **[Developer Guide](DEVELOPER.md)**: Development and troubleshooting
- **[User Guide](USER_GUIDE.md)**: User-facing features and interfaces
- **[API Documentation](API.md)**: API endpoints and integration

### Emergency Contacts

- **Operations Team**: Slack #operations, PagerDuty escalation
- **Security Team**: Slack #security-incidents, Phone: +1-555-SECURITY
- **Development Team**: Slack #dev-support, Email: dev@flashmm.com
- **Management Escalation**: Phone: +1-555-ESCALATE

This runbook ensures FlashMM operates with the highest levels of reliability, security, and performance required for production high-frequency trading operations.
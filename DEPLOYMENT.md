# 🚀 FlashMM Deployment Guide

**Quick deployment guide for FlashMM High-Frequency Trading Platform**

## 📋 Prerequisites

### Required Tools
```bash
# Container and orchestration tools
docker >= 20.10
docker-compose >= 2.0
kubectl >= 1.24
helm >= 3.10

# Infrastructure tools (for production)
terraform >= 1.5
aws-cli >= 2.0  # or gcloud/az CLI

# Development tools
python >= 3.11
git
make
```

### Cloud Provider Setup
Choose your cloud provider and configure access:

**AWS:**
```bash
aws configure
export AWS_DEFAULT_REGION=us-east-1
```

**GCP:**
```bash
gcloud auth login
gcloud config set project flashmm-prod
```

**Azure:**
```bash
az login
az account set --subscription "flashmm-subscription"
```

## ⚡ Quick Start (5 Minutes)

### Development Environment
```bash
# 1. Clone and setup
git clone https://github.com/flashmm/flashmm.git
cd flashmm
cp .env.template .env.dev

# 2. Start development stack
docker-compose -f docker-compose.dev.yml up -d

# 3. Access services
echo "🚀 FlashMM API: http://localhost:8000"
echo "📊 Grafana: http://localhost:3000 (admin/dev_admin_123)"
echo "🔍 Redis UI: http://localhost:8081"
echo "📧 MailHog: http://localhost:8025"
```

### Staging Deployment
```bash
# 1. Configure environment
cp environments/staging/values.yaml my-staging-values.yaml
# Edit my-staging-values.yaml with your settings

# 2. Deploy to staging cluster
./environments/deploy.sh -e staging

# 3. Verify deployment
./scripts/health-check.sh -e staging
```

### Production Deployment
```bash
# 1. Deploy infrastructure (AWS example)
cd terraform/
terraform init
terraform apply \
  -var="environment=production" \
  -var="cloud_provider=aws" \
  -var="flashmm_replica_count=3"

# 2. Configure kubectl
aws eks update-kubeconfig --region us-east-1 --name flashmm-production

# 3. Deploy application
./environments/deploy.sh -e production

# 4. Validate deployment
./tests/deployment/run_all_tests.sh -e production --full
```

## 🎯 Deployment Options

### Option 1: Docker Compose (Development/Small Production)

**Best for:** Development, demos, small deployments
**Complexity:** ⭐⭐☆☆☆

```bash
# Development
docker-compose -f docker-compose.dev.yml up -d

# Staging-like
docker-compose -f docker-compose.staging.yml up -d

# Production (single server)
docker-compose -f docker-compose.prod.yml up -d
```

**Features:**
- ✅ Quick setup and deployment
- ✅ Integrated monitoring stack
- ✅ Automated backups
- ✅ Health checks and restarts
- ❌ Limited scalability
- ❌ No automatic failover

### Option 2: Kubernetes + Helm (Recommended for Production)

**Best for:** Production, high availability, scalability
**Complexity:** ⭐⭐⭐⭐☆

```bash
# Using Helm charts
helm install flashmm ./helm/flashmm/ \
  -f environments/production/values.yaml \
  --namespace flashmm \
  --create-namespace

# Using raw Kubernetes manifests
kubectl apply -f k8s/
```

**Features:**
- ✅ High availability and auto-scaling
- ✅ Rolling updates and rollbacks
- ✅ Advanced monitoring and alerting
- ✅ Security policies and compliance
- ✅ Multi-environment support
- ⚠️ Requires Kubernetes expertise

### Option 3: Infrastructure as Code (Enterprise)

**Best for:** Enterprise, compliance, multi-cloud
**Complexity:** ⭐⭐⭐⭐⭐

```bash
# Full infrastructure deployment
cd terraform/
terraform init
terraform plan -out=production.tfplan
terraform apply production.tfplan
```

**Features:**
- ✅ Complete infrastructure automation
- ✅ Multi-cloud support (AWS, GCP, Azure)
- ✅ Compliance and security controls
- ✅ Disaster recovery and backup
- ✅ Cost optimization
- ⚠️ Requires infrastructure expertise

## 🔧 Configuration

### Environment Variables

Create environment-specific configuration files:

**Development:** `.env.dev`
```bash
ENVIRONMENT=development
FLASHMM_DEBUG=true
SEI_NETWORK=testnet
TRADING_ENABLED=false
```

**Staging:** `.env.staging`
```bash
ENVIRONMENT=staging
FLASHMM_DEBUG=false
SEI_NETWORK=testnet
TRADING_ENABLED=true
TRADING_MAX_POSITION_USDC=1000
```

**Production:** `.env.prod`
```bash
ENVIRONMENT=production
FLASHMM_DEBUG=false
SEI_NETWORK=mainnet
TRADING_ENABLED=true
TRADING_MAX_POSITION_USDC=10000
```

### Secrets Management

**Docker Compose:**
```bash
# Update .env file with actual values
cp .env.template .env
# Edit .env with your API keys and secrets
```

**Kubernetes:**
```bash
# Update secrets template
kubectl apply -f k8s/secrets.yaml
kubectl edit secret flashmm-secrets -n flashmm
```

**Production Secrets (External Secrets Operator):**
```yaml
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: flashmm-secrets
spec:
  secretStoreRef:
    name: aws-secrets-manager
    kind: SecretStore
  target:
    name: flashmm-secrets
  data:
    - secretKey: SEI_PRIVATE_KEY
      remoteRef:
        key: flashmm/production/sei
        property: private_key
```

## 📊 Monitoring Setup

### Accessing Dashboards

| Environment | Grafana | Prometheus | Kibana |
|-------------|---------|------------|--------|
| Development | http://localhost:3000 | http://localhost:9090 | http://localhost:5601 |
| Staging | https://grafana.staging.flashmm.com | https://prometheus.staging.flashmm.com | https://kibana.staging.flashmm.com |
| Production | https://grafana.flashmm.com | https://prometheus.flashmm.com | https://kibana.flashmm.com |

### Default Credentials
- **Development:** admin/dev_admin_123
- **Staging/Production:** Set via secrets

## 🧪 Testing and Validation

### Quick Validation
```bash
# Basic health check
curl http://localhost:8000/health

# Kubernetes health check
./scripts/health-check.sh -e staging

# Run quick tests
./tests/deployment/run_all_tests.sh --quick
```

### Comprehensive Testing
```bash
# Full test suite
./tests/deployment/run_all_tests.sh -e production --full

# Performance testing
python3 tests/performance/load_test.py --kubernetes --namespace flashmm

# Security compliance
./security/scripts/security-scan.sh -e production -t compliance
```

## 🚨 Emergency Procedures

### System Recovery
```bash
# 1. Check system status
kubectl get pods -n flashmm
./scripts/health-check.sh -e production

# 2. Emergency scaling
./scripts/scale.sh emergency-scale -e production

# 3. Rollback if needed
./scripts/rollback.sh -e production

# 4. Restore from backup
./scripts/restore.sh -e production -b latest
```

### Incident Response
1. **Assess impact:** Check monitoring dashboards
2. **Contain issue:** Scale resources or isolate components
3. **Investigate:** Review logs and metrics
4. **Resolve:** Apply fixes or rollback
5. **Document:** Record incident and lessons learned

## 📞 Support

### Getting Help
- **Documentation:** [`docs/deployment/README.md`](docs/deployment/README.md)
- **Troubleshooting:** Check logs and monitoring dashboards
- **Community:** GitHub Issues and Discussions
- **Enterprise Support:** Contact ops@flashmm.com

### Useful Commands
```bash
# View logs
kubectl logs -f deployment/flashmm-app -n flashmm

# Port forward for local access
kubectl port-forward -n flashmm service/flashmm-app 8000:8000

# Execute shell in pod
kubectl exec -it -n flashmm deployment/flashmm-app -- /bin/sh

# Check resource usage
kubectl top pods -n flashmm

# Scale manually
kubectl scale deployment flashmm-app -n flashmm --replicas=5
```

---

## 🎉 Ready to Deploy!

Choose your deployment method based on your requirements:

- **🔧 Development:** Use Docker Compose for quick local setup
- **🚀 Staging:** Use Kubernetes with Helm for testing
- **🏢 Production:** Use Terraform + Kubernetes for enterprise deployment

**Need help?** Check the comprehensive guide at [`docs/deployment/README.md`](docs/deployment/README.md)
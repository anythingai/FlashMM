#!/bin/bash
# FlashMM Production Deployment Script
# Comprehensive production deployment with safety checks and rollback capabilities

set -euo pipefail

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Default configuration
ENVIRONMENT="production"
NAMESPACE="flashmm"
KUBECTL_TIMEOUT="600s"
HELM_TIMEOUT="600s"
BACKUP_BEFORE_DEPLOY=true
HEALTH_CHECK_RETRIES=10
HEALTH_CHECK_INTERVAL=30

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

# Error handling
error_exit() {
    log_error "$1"
    exit 1
}

cleanup() {
    log_info "Performing cleanup..."
    # Remove temporary files if any
    if [[ -n "${TEMP_DIR:-}" ]] && [[ -d "$TEMP_DIR" ]]; then
        rm -rf "$TEMP_DIR"
    fi
}

trap cleanup EXIT

# Pre-flight checks
preflight_checks() {
    log_info "Running pre-flight checks..."
    
    # Check required tools
    command -v kubectl >/dev/null 2>&1 || error_exit "kubectl is required but not installed"
    command -v helm >/dev/null 2>&1 || error_exit "helm is required but not installed"
    command -v jq >/dev/null 2>&1 || error_exit "jq is required but not installed"
    
    # Check cluster connectivity
    kubectl cluster-info >/dev/null 2>&1 || error_exit "Cannot connect to Kubernetes cluster"
    
    # Check if namespace exists
    kubectl get namespace "$NAMESPACE" >/dev/null 2>&1 || error_exit "Namespace $NAMESPACE does not exist"
    
    # Check if current user has required permissions
    kubectl auth can-i create deployments -n "$NAMESPACE" >/dev/null 2>&1 || error_exit "Insufficient permissions to deploy"
    
    # Check cluster resources
    local available_cpu=$(kubectl top nodes --no-headers | awk '{sum+=$3} END {print sum}' || echo "0")
    local available_memory=$(kubectl top nodes --no-headers | awk '{sum+=$5} END {print sum}' || echo "0")
    
    log_info "Cluster resources - CPU: ${available_cpu}, Memory: ${available_memory}"
    
    log_success "Pre-flight checks completed"
}

# Backup current deployment
backup_deployment() {
    if [[ "$BACKUP_BEFORE_DEPLOY" != "true" ]]; then
        log_info "Skipping backup as requested"
        return 0
    fi
    
    log_info "Creating backup of current deployment..."
    
    local backup_dir="${PROJECT_ROOT}/backups/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$backup_dir"
    
    # Backup Helm values
    if helm list -n "$NAMESPACE" | grep -q flashmm; then
        helm get values flashmm -n "$NAMESPACE" > "$backup_dir/helm-values.yaml"
        helm get manifest flashmm -n "$NAMESPACE" > "$backup_dir/helm-manifest.yaml"
        log_info "Helm configuration backed up to $backup_dir"
    fi
    
    # Backup Kubernetes resources
    kubectl get all -n "$NAMESPACE" -o yaml > "$backup_dir/k8s-resources.yaml"
    kubectl get configmaps -n "$NAMESPACE" -o yaml > "$backup_dir/configmaps.yaml"
    kubectl get secrets -n "$NAMESPACE" -o yaml > "$backup_dir/secrets.yaml"
    
    # Backup database (if possible)
    if kubectl get deployment -n "$NAMESPACE" postgres >/dev/null 2>&1; then
        log_info "Creating database backup..."
        kubectl exec -n "$NAMESPACE" deployment/postgres -- pg_dump -U flashmm flashmm_prod > "$backup_dir/database.sql" || log_warning "Database backup failed"
    fi
    
    log_success "Backup completed: $backup_dir"
    echo "$backup_dir" > /tmp/flashmm_backup_path
}

# Deploy using Helm
deploy_application() {
    log_info "Deploying FlashMM to production..."
    
    local values_file="${PROJECT_ROOT}/environments/production/values.yaml"
    [[ -f "$values_file" ]] || error_exit "Production values file not found: $values_file"
    
    # Get current image tag from CI/CD or use latest
    local image_tag="${FLASHMM_IMAGE_TAG:-latest}"
    log_info "Deploying image tag: $image_tag"
    
    # Deploy with Helm
    helm upgrade --install flashmm "${PROJECT_ROOT}/helm/flashmm" \
        --namespace "$NAMESPACE" \
        --values "$values_file" \
        --set global.imageTag="$image_tag" \
        --set global.environment="production" \
        --timeout "$HELM_TIMEOUT" \
        --wait \
        --atomic || error_exit "Helm deployment failed"
    
    log_success "Helm deployment completed"
}

# Wait for deployment to be ready
wait_for_deployment() {
    log_info "Waiting for deployment to be ready..."
    
    # Wait for pods to be ready
    kubectl rollout status deployment/flashmm-app -n "$NAMESPACE" --timeout="$KUBECTL_TIMEOUT" || error_exit "Deployment rollout failed"
    
    # Wait for all pods to be ready
    local ready_pods=0
    local total_pods=0
    local retries=0
    
    while [[ $retries -lt $HEALTH_CHECK_RETRIES ]]; do
        ready_pods=$(kubectl get pods -n "$NAMESPACE" -l app.kubernetes.io/name=flashmm --field-selector=status.phase=Running -o json | jq '.items | length')
        total_pods=$(kubectl get pods -n "$NAMESPACE" -l app.kubernetes.io/name=flashmm -o json | jq '.items | length')
        
        if [[ $ready_pods -eq $total_pods ]] && [[ $total_pods -gt 0 ]]; then
            log_success "All $total_pods pods are ready"
            break
        fi
        
        log_info "Waiting for pods to be ready: $ready_pods/$total_pods (attempt $((retries + 1))/$HEALTH_CHECK_RETRIES)"
        sleep $HEALTH_CHECK_INTERVAL
        ((retries++))
    done
    
    if [[ $retries -eq $HEALTH_CHECK_RETRIES ]]; then
        error_exit "Timeout waiting for pods to be ready"
    fi
}

# Health checks
run_health_checks() {
    log_info "Running health checks..."
    
    # Check application health endpoint
    local health_check_passed=false
    local retries=0
    
    while [[ $retries -lt $HEALTH_CHECK_RETRIES ]]; do
        if kubectl run health-check-$$-$RANDOM \
            --rm -i --restart=Never \
            --image=curlimages/curl \
            --timeout=60s \
            -n "$NAMESPACE" \
            -- curl -f -m 10 "http://flashmm-app:8000/health/detailed" >/dev/null 2>&1; then
            health_check_passed=true
            break
        fi
        
        log_info "Health check failed, retrying... (attempt $((retries + 1))/$HEALTH_CHECK_RETRIES)"
        sleep $HEALTH_CHECK_INTERVAL
        ((retries++))
    done
    
    if [[ "$health_check_passed" != "true" ]]; then
        error_exit "Health checks failed after $HEALTH_CHECK_RETRIES attempts"
    fi
    
    # Check metrics endpoint
    if kubectl run metrics-check-$$-$RANDOM \
        --rm -i --restart=Never \
        --image=curlimages/curl \
        --timeout=60s \
        -n "$NAMESPACE" \
        -- curl -f -m 10 "http://flashmm-app:8000/metrics" >/dev/null 2>&1; then
        log_success "Metrics endpoint is healthy"
    else
        log_warning "Metrics endpoint check failed"
    fi
    
    # Check database connectivity
    if kubectl get deployment -n "$NAMESPACE" postgres >/dev/null 2>&1; then
        if kubectl exec -n "$NAMESPACE" deployment/postgres -- pg_isready -U flashmm >/dev/null 2>&1; then
            log_success "Database connectivity check passed"
        else
            log_warning "Database connectivity check failed"
        fi
    fi
    
    # Check Redis connectivity
    if kubectl get deployment -n "$NAMESPACE" redis >/dev/null 2>&1; then
        if kubectl exec -n "$NAMESPACE" deployment/redis -- redis-cli ping >/dev/null 2>&1; then
            log_success "Redis connectivity check passed"
        else
            log_warning "Redis connectivity check failed"
        fi
    fi
    
    log_success "Health checks completed"
}

# Smoke tests
run_smoke_tests() {
    log_info "Running smoke tests..."
    
    # Test API functionality
    local test_results=()
    
    # Test health endpoint
    if kubectl run smoke-test-health-$$-$RANDOM \
        --rm -i --restart=Never \
        --image=curlimages/curl \
        --timeout=60s \
        -n "$NAMESPACE" \
        -- curl -f "http://flashmm-app:8000/health" >/dev/null 2>&1; then
        test_results+=("✓ Health endpoint")
    else
        test_results+=("✗ Health endpoint")
    fi
    
    # Test metrics endpoint
    if kubectl run smoke-test-metrics-$$-$RANDOM \
        --rm -i --restart=Never \
        --image=curlimages/curl \
        --timeout=60s \
        -n "$NAMESPACE" \
        -- curl -f "http://flashmm-app:8000/metrics" >/dev/null 2>&1; then
        test_results+=("✓ Metrics endpoint")
    else
        test_results+=("✗ Metrics endpoint")
    fi
    
    # Display results
    log_info "Smoke test results:"
    for result in "${test_results[@]}"; do
        echo "  $result"
    done
    
    # Check if any tests failed
    if [[ "${test_results[*]}" == *"✗"* ]]; then
        log_warning "Some smoke tests failed"
        return 1
    else
        log_success "All smoke tests passed"
        return 0
    fi
}

# Monitor deployment
monitor_deployment() {
    log_info "Monitoring deployment for 5 minutes..."
    
    local start_time=$(date +%s)
    local monitor_duration=300  # 5 minutes
    
    while [[ $(($(date +%s) - start_time)) -lt $monitor_duration ]]; do
        # Check pod status
        local unhealthy_pods=$(kubectl get pods -n "$NAMESPACE" -l app.kubernetes.io/name=flashmm --field-selector=status.phase!=Running -o name | wc -l)
        
        if [[ $unhealthy_pods -gt 0 ]]; then
            log_warning "Found $unhealthy_pods unhealthy pods"
            kubectl get pods -n "$NAMESPACE" -l app.kubernetes.io/name=flashmm
        fi
        
        # Check for recent errors in logs
        local error_count=$(kubectl logs -n "$NAMESPACE" deployment/flashmm-app --since=1m | grep -i error | wc -l || echo "0")
        if [[ $error_count -gt 5 ]]; then
            log_warning "High error rate detected in logs: $error_count errors in last minute"
        fi
        
        sleep 30
    done
    
    log_success "Deployment monitoring completed"
}

# Post-deployment actions
post_deployment_actions() {
    log_info "Running post-deployment actions..."
    
    # Update deployment status
    kubectl annotate deployment/flashmm-app -n "$NAMESPACE" \
        deployment.kubernetes.io/last-successful-deployment="$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        --overwrite
    
    # Send notifications (if configured)
    if [[ -n "${SLACK_WEBHOOK_URL:-}" ]]; then
        curl -X POST "$SLACK_WEBHOOK_URL" \
            -H 'Content-Type: application/json' \
            -d "{\"text\":\"🚀 FlashMM production deployment completed successfully!\"}" \
            >/dev/null 2>&1 || log_warning "Failed to send Slack notification"
    fi
    
    # Clean up old resources
    log_info "Cleaning up old resources..."
    kubectl delete pods -n "$NAMESPACE" --field-selector=status.phase=Succeeded >/dev/null 2>&1 || true
    kubectl delete pods -n "$NAMESPACE" --field-selector=status.phase=Failed >/dev/null 2>&1 || true
    
    log_success "Post-deployment actions completed"
}

# Display deployment summary
show_deployment_summary() {
    log_info "Deployment Summary"
    echo "====================="
    
    # Deployment info
    kubectl get deployment -n "$NAMESPACE" flashmm-app -o wide
    echo ""
    
    # Pod info
    echo "Pods:"
    kubectl get pods -n "$NAMESPACE" -l app.kubernetes.io/name=flashmm
    echo ""
    
    # Service info
    echo "Services:"
    kubectl get services -n "$NAMESPACE" -l app.kubernetes.io/name=flashmm
    echo ""
    
    # Ingress info
    if kubectl get ingress -n "$NAMESPACE" >/dev/null 2>&1; then
        echo "Ingress:"
        kubectl get ingress -n "$NAMESPACE"
        echo ""
    fi
    
    # URLs
    echo "URLs:"
    echo "  API: https://api.flashmm.com"
    echo "  Grafana: https://grafana.flashmm.com"
    echo "  Prometheus: https://prometheus.flashmm.com"
    echo ""
    
    # Resource usage
    echo "Resource Usage:"
    kubectl top pods -n "$NAMESPACE" -l app.kubernetes.io/name=flashmm 2>/dev/null || echo "  Resource metrics not available"
}

# Main execution
main() {
    log_info "Starting FlashMM production deployment..."
    log_info "Environment: $ENVIRONMENT"
    log_info "Namespace: $NAMESPACE"
    
    # Create temporary directory
    TEMP_DIR=$(mktemp -d)
    
    # Execute deployment steps
    preflight_checks
    backup_deployment
    deploy_application
    wait_for_deployment
    run_health_checks
    
    if run_smoke_tests; then
        monitor_deployment
        post_deployment_actions
        show_deployment_summary
        log_success "🎉 FlashMM production deployment completed successfully!"
    else
        log_error "Smoke tests failed. Deployment may need attention."
        exit 1
    fi
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --no-backup)
            BACKUP_BEFORE_DEPLOY=false
            shift
            ;;
        --image-tag)
            FLASHMM_IMAGE_TAG="$2"
            shift 2
            ;;
        --timeout)
            KUBECTL_TIMEOUT="$2"
            HELM_TIMEOUT="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --no-backup          Skip backup before deployment"
            echo "  --image-tag TAG      Specify image tag to deploy"
            echo "  --timeout SECONDS    Set deployment timeout"
            echo "  --help               Show this help message"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Execute main function
main "$@"
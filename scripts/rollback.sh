#!/bin/bash
# FlashMM Rollback Script
# Automated rollback with safety checks and validation

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Configuration
ENVIRONMENT="production"
NAMESPACE="flashmm"
ROLLBACK_TIMEOUT="300s"
VALIDATE_ROLLBACK=true

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"; }

show_help() {
    cat << EOF
FlashMM Rollback Script

Usage: $0 [OPTIONS]

Options:
    -e, --environment ENV    Target environment (default: production)
    -n, --namespace NS       Kubernetes namespace (default: flashmm)
    -r, --revision NUM       Specific revision to rollback to
    -f, --force             Force rollback without validation
    --helm-rollback         Use Helm rollback instead of kubectl
    --list-revisions        List available revisions
    -h, --help              Show this help message

Examples:
    $0                      # Rollback to previous revision
    $0 -r 5                 # Rollback to revision 5
    $0 --list-revisions     # Show available revisions
    $0 --helm-rollback      # Use Helm for rollback

EOF
}

# Parse arguments
ROLLBACK_REVISION=""
FORCE_ROLLBACK=false
USE_HELM=false
LIST_REVISIONS=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -n|--namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        -r|--revision)
            ROLLBACK_REVISION="$2"
            shift 2
            ;;
        -f|--force)
            FORCE_ROLLBACK=true
            shift
            ;;
        --helm-rollback)
            USE_HELM=true
            shift
            ;;
        --list-revisions)
            LIST_REVISIONS=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    command -v kubectl >/dev/null 2>&1 || { log_error "kubectl is required"; exit 1; }
    command -v helm >/dev/null 2>&1 || { log_error "helm is required"; exit 1; }
    
    kubectl cluster-info >/dev/null 2>&1 || { log_error "Cannot connect to cluster"; exit 1; }
    kubectl get namespace "$NAMESPACE" >/dev/null 2>&1 || { log_error "Namespace $NAMESPACE not found"; exit 1; }
    
    log_success "Prerequisites check passed"
}

# List available revisions
list_revisions() {
    log_info "Available revisions:"
    
    if [[ "$USE_HELM" == "true" ]]; then
        echo "Helm Release History:"
        helm history flashmm -n "$NAMESPACE" || log_error "Failed to get Helm history"
    else
        echo "Kubernetes Deployment History:"
        kubectl rollout history deployment/flashmm-app -n "$NAMESPACE" || log_error "Failed to get deployment history"
    fi
}

# Get current status
get_current_status() {
    log_info "Current deployment status:"
    
    kubectl get deployment flashmm-app -n "$NAMESPACE" -o wide
    kubectl get pods -n "$NAMESPACE" -l app.kubernetes.io/name=flashmm
    
    # Get current revision
    local current_revision
    if [[ "$USE_HELM" == "true" ]]; then
        current_revision=$(helm list -n "$NAMESPACE" -o json | jq -r '.[] | select(.name=="flashmm") | .revision')
        log_info "Current Helm revision: $current_revision"
    else
        current_revision=$(kubectl get deployment flashmm-app -n "$NAMESPACE" -o jsonpath='{.metadata.annotations.deployment\.kubernetes\.io/revision}')
        log_info "Current deployment revision: $current_revision"
    fi
}

# Pre-rollback validation
pre_rollback_validation() {
    if [[ "$FORCE_ROLLBACK" == "true" ]]; then
        log_warning "Skipping pre-rollback validation (force mode)"
        return 0
    fi
    
    log_info "Running pre-rollback validation..."
    
    # Check if there are any critical alerts
    if command -v curl >/dev/null 2>&1 && [[ -n "${PROMETHEUS_URL:-}" ]]; then
        local critical_alerts=$(curl -s "${PROMETHEUS_URL}/api/v1/query?query=ALERTS{severity=\"critical\"}" | jq -r '.data.result | length')
        if [[ "$critical_alerts" -gt 0 ]]; then
            log_warning "Found $critical_alerts critical alerts. Consider investigating before rollback."
            read -p "Continue with rollback? (y/N) " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                log_info "Rollback cancelled by user"
                exit 0
            fi
        fi
    fi
    
    # Check current pod health
    local unhealthy_pods=$(kubectl get pods -n "$NAMESPACE" -l app.kubernetes.io/name=flashmm --field-selector=status.phase!=Running -o name | wc -l)
    if [[ $unhealthy_pods -eq 0 ]]; then
        log_warning "All pods are currently healthy. Are you sure you want to rollback?"
        if [[ "$FORCE_ROLLBACK" != "true" ]]; then
            read -p "Continue with rollback? (y/N) " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                log_info "Rollback cancelled by user"
                exit 0
            fi
        fi
    fi
    
    log_success "Pre-rollback validation completed"
}

# Perform rollback
perform_rollback() {
    log_info "Performing rollback..."
    
    if [[ "$USE_HELM" == "true" ]]; then
        # Helm rollback
        local rollback_cmd="helm rollback flashmm"
        if [[ -n "$ROLLBACK_REVISION" ]]; then
            rollback_cmd="$rollback_cmd $ROLLBACK_REVISION"
        fi
        rollback_cmd="$rollback_cmd -n $NAMESPACE --timeout $ROLLBACK_TIMEOUT --wait"
        
        log_info "Executing: $rollback_cmd"
        if eval "$rollback_cmd"; then
            log_success "Helm rollback completed"
        else
            log_error "Helm rollback failed"
            return 1
        fi
    else
        # Kubernetes rollback
        local rollback_cmd="kubectl rollout undo deployment/flashmm-app -n $NAMESPACE"
        if [[ -n "$ROLLBACK_REVISION" ]]; then
            rollback_cmd="$rollback_cmd --to-revision=$ROLLBACK_REVISION"
        fi
        
        log_info "Executing: $rollback_cmd"
        if eval "$rollback_cmd"; then
            # Wait for rollback to complete
            kubectl rollout status deployment/flashmm-app -n "$NAMESPACE" --timeout="$ROLLBACK_TIMEOUT"
            log_success "Kubernetes rollback completed"
        else
            log_error "Kubernetes rollback failed"
            return 1
        fi
    fi
}

# Post-rollback validation
post_rollback_validation() {
    if [[ "$VALIDATE_ROLLBACK" != "true" ]]; then
        log_info "Skipping post-rollback validation"
        return 0
    fi
    
    log_info "Running post-rollback validation..."
    
    # Wait for pods to be ready
    local ready_pods=0
    local total_pods=0
    local retries=0
    local max_retries=10
    
    while [[ $retries -lt $max_retries ]]; do
        ready_pods=$(kubectl get pods -n "$NAMESPACE" -l app.kubernetes.io/name=flashmm --field-selector=status.phase=Running -o json | jq '.items | length')
        total_pods=$(kubectl get pods -n "$NAMESPACE" -l app.kubernetes.io/name=flashmm -o json | jq '.items | length')
        
        if [[ $ready_pods -eq $total_pods ]] && [[ $total_pods -gt 0 ]]; then
            log_success "All $total_pods pods are ready after rollback"
            break
        fi
        
        log_info "Waiting for pods to be ready: $ready_pods/$total_pods (attempt $((retries + 1))/$max_retries)"
        sleep 30
        ((retries++))
    done
    
    if [[ $retries -eq $max_retries ]]; then
        log_error "Timeout waiting for pods to be ready after rollback"
        return 1
    fi
    
    # Health check
    log_info "Running health check..."
    if kubectl run rollback-health-check-$$-$RANDOM \
        --rm -i --restart=Never \
        --image=curlimages/curl \
        --timeout=60s \
        -n "$NAMESPACE" \
        -- curl -f -m 10 "http://flashmm-app:8000/health" >/dev/null 2>&1; then
        log_success "Health check passed"
    else
        log_error "Health check failed after rollback"
        return 1
    fi
    
    log_success "Post-rollback validation completed"
}

# Send notifications
send_notifications() {
    log_info "Sending rollback notifications..."
    
    local rollback_info="FlashMM rollback completed"
    if [[ -n "$ROLLBACK_REVISION" ]]; then
        rollback_info="$rollback_info to revision $ROLLBACK_REVISION"
    fi
    rollback_info="$rollback_info in $ENVIRONMENT environment"
    
    # Slack notification
    if [[ -n "${SLACK_WEBHOOK_URL:-}" ]]; then
        curl -X POST "$SLACK_WEBHOOK_URL" \
            -H 'Content-Type: application/json' \
            -d "{\"text\":\"⏪ $rollback_info\"}" \
            >/dev/null 2>&1 || log_warning "Failed to send Slack notification"
    fi
    
    # Update deployment annotation
    kubectl annotate deployment/flashmm-app -n "$NAMESPACE" \
        deployment.kubernetes.io/rollback-timestamp="$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        --overwrite >/dev/null 2>&1 || true
    
    log_success "Notifications sent"
}

# Show rollback summary
show_rollback_summary() {
    log_info "Rollback Summary"
    echo "=================="
    
    get_current_status
    
    echo ""
    echo "Rollback Details:"
    echo "  Environment: $ENVIRONMENT"
    echo "  Namespace: $NAMESPACE"
    echo "  Method: $(if [[ "$USE_HELM" == "true" ]]; then echo "Helm"; else echo "Kubernetes"; fi)"
    echo "  Target Revision: ${ROLLBACK_REVISION:-previous}"
    echo "  Timestamp: $(date)"
}

# Main execution
main() {
    log_info "Starting FlashMM rollback process..."
    
    check_prerequisites
    
    if [[ "$LIST_REVISIONS" == "true" ]]; then
        list_revisions
        exit 0
    fi
    
    get_current_status
    pre_rollback_validation
    
    if perform_rollback; then
        if post_rollback_validation; then
            send_notifications
            show_rollback_summary
            log_success "🔄 FlashMM rollback completed successfully!"
        else
            log_error "Post-rollback validation failed. System may be in unstable state."
            exit 1
        fi
    else
        log_error "Rollback failed. System state unchanged."
        exit 1
    fi
}

# Execute main function
main "$@"
#!/bin/bash
# FlashMM Scaling Script
# Dynamic scaling operations for production workloads

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Configuration
ENVIRONMENT="production"
NAMESPACE="flashmm"
COMPONENT="flashmm-app"
SCALE_ACTION=""
TARGET_REPLICAS=""
MAX_REPLICAS=10
MIN_REPLICAS=1
TIMEOUT="300s"

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
FlashMM Scaling Script

Usage: $0 [OPTIONS] ACTION [TARGET]

Actions:
    up [replicas]           Scale up to specified replicas or +1 if not specified
    down [replicas]         Scale down to specified replicas or -1 if not specified
    set REPLICAS            Set exact number of replicas
    auto                    Enable/configure horizontal pod autoscaling
    status                  Show current scaling status
    emergency-scale         Emergency scaling for high load situations

Options:
    -e, --environment ENV    Environment (default: production)
    -n, --namespace NS       Kubernetes namespace (default: flashmm)
    -c, --component NAME     Component to scale (default: flashmm-app)
    --max-replicas NUM      Maximum allowed replicas (default: 10)
    --min-replicas NUM      Minimum allowed replicas (default: 1)
    --timeout DURATION      Operation timeout (default: 300s)
    --dry-run               Show what would be done without executing
    -h, --help              Show this help message

Examples:
    $0 up                   # Scale up by 1 replica
    $0 up 5                 # Scale up to 5 replicas
    $0 down                 # Scale down by 1 replica
    $0 set 3                # Set exactly 3 replicas
    $0 auto                 # Enable auto-scaling
    $0 status               # Show current status
    $0 emergency-scale      # Emergency scale for high load

EOF
}

# Parse arguments
DRY_RUN=false

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
        -c|--component)
            COMPONENT="$2"
            shift 2
            ;;
        --max-replicas)
            MAX_REPLICAS="$2"
            shift 2
            ;;
        --min-replicas)
            MIN_REPLICAS="$2"
            shift 2
            ;;
        --timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        up|down|set|auto|status|emergency-scale)
            SCALE_ACTION="$1"
            if [[ $# -gt 1 && ! "$2" =~ ^- ]]; then
                TARGET_REPLICAS="$2"
                shift 2
            else
                shift
            fi
            ;;
        *)
            if [[ -z "$SCALE_ACTION" ]]; then
                log_error "Unknown action: $1"
                show_help
                exit 1
            else
                TARGET_REPLICAS="$1"
                shift
            fi
            ;;
    esac
done

if [[ -z "$SCALE_ACTION" ]]; then
    log_error "No action specified"
    show_help
    exit 1
fi

# Prerequisites check
check_prerequisites() {
    command -v kubectl >/dev/null 2>&1 || { log_error "kubectl is required"; exit 1; }
    kubectl cluster-info >/dev/null 2>&1 || { log_error "Cannot connect to cluster"; exit 1; }
    kubectl get namespace "$NAMESPACE" >/dev/null 2>&1 || { log_error "Namespace $NAMESPACE not found"; exit 1; }
    kubectl get deployment "$COMPONENT" -n "$NAMESPACE" >/dev/null 2>&1 || { log_error "Deployment $COMPONENT not found"; exit 1; }
}

# Get current status
get_current_status() {
    local current_replicas=$(kubectl get deployment "$COMPONENT" -n "$NAMESPACE" -o jsonpath='{.status.replicas}')
    local ready_replicas=$(kubectl get deployment "$COMPONENT" -n "$NAMESPACE" -o jsonpath='{.status.readyReplicas}')
    local available_replicas=$(kubectl get deployment "$COMPONENT" -n "$NAMESPACE" -o jsonpath='{.status.availableReplicas}')
    
    echo "Current Status:"
    echo "  Desired: $current_replicas"
    echo "  Ready: ${ready_replicas:-0}"
    echo "  Available: ${available_replicas:-0}"
    
    # Check HPA status
    if kubectl get hpa "${COMPONENT}-hpa" -n "$NAMESPACE" >/dev/null 2>&1; then
        echo ""
        echo "HPA Status:"
        kubectl get hpa "${COMPONENT}-hpa" -n "$NAMESPACE"
    fi
    
    return "$current_replicas"
}

# Validate scaling parameters
validate_scaling() {
    local target="$1"
    
    if [[ $target -lt $MIN_REPLICAS ]]; then
        log_error "Target replicas ($target) below minimum ($MIN_REPLICAS)"
        return 1
    fi
    
    if [[ $target -gt $MAX_REPLICAS ]]; then
        log_error "Target replicas ($target) above maximum ($MAX_REPLICAS)"
        return 1
    fi
    
    # Check if environment supports the target
    case "$ENVIRONMENT" in
        development)
            if [[ $target -gt 2 ]]; then
                log_warning "Development environment scaling to $target replicas (consider resource limits)"
            fi
            ;;
        staging)
            if [[ $target -gt 5 ]]; then
                log_warning "Staging environment scaling to $target replicas"
            fi
            ;;
        production)
            if [[ $target -gt $MAX_REPLICAS ]]; then
                log_error "Production scaling exceeds configured maximum"
                return 1
            fi
            ;;
    esac
    
    return 0
}

# Scale up operation
scale_up() {
    local current_replicas=$1
    local target_replicas
    
    if [[ -n "$TARGET_REPLICAS" ]]; then
        target_replicas="$TARGET_REPLICAS"
    else
        target_replicas=$((current_replicas + 1))
    fi
    
    if ! validate_scaling "$target_replicas"; then
        return 1
    fi
    
    log_info "Scaling up from $current_replicas to $target_replicas replicas"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would scale deployment $COMPONENT to $target_replicas replicas"
        return 0
    fi
    
    kubectl scale deployment "$COMPONENT" -n "$NAMESPACE" --replicas="$target_replicas"
    kubectl rollout status deployment "$COMPONENT" -n "$NAMESPACE" --timeout="$TIMEOUT"
    
    log_success "Scaled up to $target_replicas replicas"
}

# Scale down operation
scale_down() {
    local current_replicas=$1
    local target_replicas
    
    if [[ -n "$TARGET_REPLICAS" ]]; then
        target_replicas="$TARGET_REPLICAS"
    else
        target_replicas=$((current_replicas - 1))
    fi
    
    if ! validate_scaling "$target_replicas"; then
        return 1
    fi
    
    log_info "Scaling down from $current_replicas to $target_replicas replicas"
    
    if [[ "$ENVIRONMENT" == "production" ]] && [[ $target_replicas -lt 2 ]]; then
        log_warning "Scaling production to less than 2 replicas reduces availability"
        read -p "Continue? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "Scaling cancelled by user"
            return 0
        fi
    fi
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would scale deployment $COMPONENT to $target_replicas replicas"
        return 0
    fi
    
    kubectl scale deployment "$COMPONENT" -n "$NAMESPACE" --replicas="$target_replicas"
    kubectl rollout status deployment "$COMPONENT" -n "$NAMESPACE" --timeout="$TIMEOUT"
    
    log_success "Scaled down to $target_replicas replicas"
}

# Set exact replicas
set_replicas() {
    local current_replicas=$1
    local target_replicas="$TARGET_REPLICAS"
    
    if [[ -z "$target_replicas" ]]; then
        log_error "Target replica count required for 'set' action"
        return 1
    fi
    
    if ! validate_scaling "$target_replicas"; then
        return 1
    fi
    
    log_info "Setting replicas from $current_replicas to $target_replicas"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would set deployment $COMPONENT to $target_replicas replicas"
        return 0
    fi
    
    kubectl scale deployment "$COMPONENT" -n "$NAMESPACE" --replicas="$target_replicas"
    kubectl rollout status deployment "$COMPONENT" -n "$NAMESPACE" --timeout="$TIMEOUT"
    
    log_success "Set to $target_replicas replicas"
}

# Configure auto-scaling
configure_autoscaling() {
    log_info "Configuring horizontal pod autoscaling..."
    
    local hpa_name="${COMPONENT}-hpa"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would configure HPA for $COMPONENT"
        return 0
    fi
    
    # Check if HPA already exists
    if kubectl get hpa "$hpa_name" -n "$NAMESPACE" >/dev/null 2>&1; then
        log_info "HPA already exists, updating configuration..."
        
        kubectl patch hpa "$hpa_name" -n "$NAMESPACE" --patch '{
            "spec": {
                "minReplicas": '$MIN_REPLICAS',
                "maxReplicas": '$MAX_REPLICAS',
                "metrics": [
                    {
                        "type": "Resource",
                        "resource": {
                            "name": "cpu",
                            "target": {
                                "type": "Utilization",
                                "averageUtilization": 70
                            }
                        }
                    },
                    {
                        "type": "Resource", 
                        "resource": {
                            "name": "memory",
                            "target": {
                                "type": "Utilization",
                                "averageUtilization": 80
                            }
                        }
                    }
                ]
            }
        }'
    else
        log_info "Creating new HPA..."
        
        kubectl autoscale deployment "$COMPONENT" -n "$NAMESPACE" \
            --cpu-percent=70 \
            --min="$MIN_REPLICAS" \
            --max="$MAX_REPLICAS"
    fi
    
    log_success "Auto-scaling configured"
    kubectl get hpa "$hpa_name" -n "$NAMESPACE"
}

# Emergency scaling for high load
emergency_scale() {
    log_warning "⚡ EMERGENCY SCALING ACTIVATED ⚡"
    
    local current_replicas
    current_replicas=$(kubectl get deployment "$COMPONENT" -n "$NAMESPACE" -o jsonpath='{.status.replicas}')
    
    # Calculate emergency target (double current or max allowed)
    local emergency_target=$((current_replicas * 2))
    if [[ $emergency_target -gt $MAX_REPLICAS ]]; then
        emergency_target=$MAX_REPLICAS
    fi
    
    log_warning "Emergency scaling from $current_replicas to $emergency_target replicas"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would emergency scale to $emergency_target replicas"
        return 0
    fi
    
    # Disable HPA temporarily during emergency scaling
    if kubectl get hpa "${COMPONENT}-hpa" -n "$NAMESPACE" >/dev/null 2>&1; then
        log_info "Temporarily disabling HPA"
        kubectl patch hpa "${COMPONENT}-hpa" -n "$NAMESPACE" --patch '{
            "spec": {"maxReplicas": '$emergency_target'}
        }'
    fi
    
    # Perform emergency scaling
    kubectl scale deployment "$COMPONENT" -n "$NAMESPACE" --replicas="$emergency_target"
    
    # Wait for scale-up with reduced timeout for urgency
    kubectl rollout status deployment "$COMPONENT" -n "$NAMESPACE" --timeout="120s" || {
        log_error "Emergency scaling did not complete in time"
        return 1
    }
    
    # Send emergency notification
    if [[ -n "${SLACK_WEBHOOK_URL:-}" ]]; then
        curl -X POST "$SLACK_WEBHOOK_URL" \
            -H 'Content-Type: application/json' \
            -d "{\"text\":\"⚡ EMERGENCY SCALING: FlashMM scaled to $emergency_target replicas in $ENVIRONMENT\"}" \
            >/dev/null 2>&1 || log_warning "Failed to send emergency notification"
    fi
    
    log_success "Emergency scaling completed"
    log_warning "Remember to review and adjust scaling after the emergency situation resolves"
}

# Show scaling status
show_status() {
    echo ""
    echo "======================================"
    echo "FlashMM Scaling Status"
    echo "======================================"
    echo "Environment: $ENVIRONMENT"
    echo "Namespace: $NAMESPACE"
    echo "Component: $COMPONENT"
    echo ""
    
    # Deployment status
    echo "Deployment Status:"
    kubectl get deployment "$COMPONENT" -n "$NAMESPACE" -o wide
    echo ""
    
    # Pod status
    echo "Pod Status:"
    kubectl get pods -n "$NAMESPACE" -l app.kubernetes.io/name=flashmm
    echo ""
    
    # HPA status (if exists)
    if kubectl get hpa "${COMPONENT}-hpa" -n "$NAMESPACE" >/dev/null 2>&1; then
        echo "Horizontal Pod Autoscaler:"
        kubectl get hpa "${COMPONENT}-hpa" -n "$NAMESPACE"
        echo ""
        
        echo "HPA Details:"
        kubectl describe hpa "${COMPONENT}-hpa" -n "$NAMESPACE" | tail -20
        echo ""
    fi
    
    # Resource usage
    echo "Current Resource Usage:"
    kubectl top pods -n "$NAMESPACE" -l app.kubernetes.io/name=flashmm 2>/dev/null || echo "  Resource metrics not available"
    echo ""
    
    # Recent scaling events
    echo "Recent Scaling Events:"
    kubectl get events -n "$NAMESPACE" --field-selector involvedObject.name="$COMPONENT" \
        --sort-by='.lastTimestamp' | tail -5 || echo "  No recent events"
}

# Pre-scaling validation
pre_scaling_validation() {
    local target="$1"
    
    log_info "Running pre-scaling validation for $target replicas..."
    
    # Check node capacity
    local total_nodes=$(kubectl get nodes --no-headers | wc -l)
    local allocatable_cpu=$(kubectl describe nodes | grep -A5 "Allocatable:" | grep cpu | awk '{sum+=substr($2,1,length($2)-1)} END {print sum}')
    local allocatable_memory=$(kubectl describe nodes | grep -A5 "Allocatable:" | grep memory | awk '{sum+=substr($2,1,length($2)-2)} END {print sum}')
    
    log_info "Cluster capacity - Nodes: $total_nodes, CPU: ${allocatable_cpu:-unknown}, Memory: ${allocatable_memory:-unknown}Ki"
    
    # Estimate resource requirements
    local estimated_cpu=$((target * 500))  # 500m per replica
    local estimated_memory=$((target * 1024))  # 1Gi per replica
    
    if [[ -n "$allocatable_cpu" ]] && [[ $estimated_cpu -gt $allocatable_cpu ]]; then
        log_warning "Target scaling may exceed cluster CPU capacity"
    fi
    
    if [[ -n "$allocatable_memory" ]] && [[ $estimated_memory -gt $allocatable_memory ]]; then
        log_warning "Target scaling may exceed cluster memory capacity"
    fi
    
    # Check current system load
    local current_cpu_usage
    current_cpu_usage=$(kubectl top pods -n "$NAMESPACE" -l app.kubernetes.io/name=flashmm --no-headers 2>/dev/null | \
        awk '{sum+=$2} END {print sum}' | sed 's/m//' || echo "0")
    
    if [[ $current_cpu_usage -gt 1000 ]] && [[ "$SCALE_ACTION" == "down" ]]; then
        log_warning "Current CPU usage is high ($current_cpu_usage), scaling down may impact performance"
    fi
    
    log_success "Pre-scaling validation completed"
}

# Post-scaling validation
post_scaling_validation() {
    local target="$1"
    
    log_info "Running post-scaling validation..."
    
    # Wait for rollout to complete
    kubectl rollout status deployment "$COMPONENT" -n "$NAMESPACE" --timeout="$TIMEOUT"
    
    # Verify target replicas achieved
    local actual_replicas=$(kubectl get deployment "$COMPONENT" -n "$NAMESPACE" -o jsonpath='{.status.replicas}')
    local ready_replicas=$(kubectl get deployment "$COMPONENT" -n "$NAMESPACE" -o jsonpath='{.status.readyReplicas}')
    
    if [[ $actual_replicas -eq $target && $ready_replicas -eq $target ]]; then
        log_success "Scaling validation passed: $ready_replicas/$target replicas ready"
    else
        log_error "Scaling validation failed: $ready_replicas/$target replicas ready"
        return 1
    fi
    
    # Run basic health check
    sleep 30  # Allow time for services to stabilize
    
    if kubectl run post-scale-health-$$-$RANDOM \
        --rm -i --restart=Never \
        --image=curlimages/curl \
        --timeout=60s \
        -n "$NAMESPACE" \
        -- curl -f -m 10 "http://flashmm-app:8000/health" >/dev/null 2>&1; then
        log_success "Post-scaling health check passed"
    else
        log_error "Post-scaling health check failed"
        return 1
    fi
}

# Main scaling logic
perform_scaling() {
    local current_replicas
    current_replicas=$(get_current_status)
    
    case "$SCALE_ACTION" in
        up)
            scale_up "$current_replicas"
            ;;
        down)
            scale_down "$current_replicas"
            ;;
        set)
            set_replicas "$current_replicas"
            ;;
        auto)
            configure_autoscaling
            ;;
        status)
            show_status
            return 0
            ;;
        emergency-scale)
            emergency_scale
            ;;
        *)
            log_error "Unknown scaling action: $SCALE_ACTION"
            return 1
            ;;
    esac
}

# Send scaling notification
send_notification() {
    local action="$1"
    local target="${2:-}"
    
    local message="FlashMM scaling operation completed: $action"
    if [[ -n "$target" ]]; then
        message="$message to $target replicas"
    fi
    message="$message in $ENVIRONMENT environment"
    
    if [[ -n "${SLACK_WEBHOOK_URL:-}" ]]; then
        curl -X POST "$SLACK_WEBHOOK_URL" \
            -H 'Content-Type: application/json' \
            -d "{\"text\":\"📊 $message\"}" \
            >/dev/null 2>&1 || log_warning "Failed to send scaling notification"
    fi
    
    # Update deployment annotation
    kubectl annotate deployment "$COMPONENT" -n "$NAMESPACE" \
        deployment.kubernetes.io/last-scaling-action="$action-$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        --overwrite >/dev/null 2>&1 || true
}

# Main execution
main() {
    log_info "Starting FlashMM scaling operation..."
    
    check_prerequisites
    
    if [[ "$SCALE_ACTION" != "status" ]]; then
        local current_replicas
        current_replicas=$(kubectl get deployment "$COMPONENT" -n "$NAMESPACE" -o jsonpath='{.status.replicas}')
        
        # Determine target for validation
        local validation_target
        case "$SCALE_ACTION" in
            up)
                validation_target="${TARGET_REPLICAS:-$((current_replicas + 1))}"
                ;;
            down)
                validation_target="${TARGET_REPLICAS:-$((current_replicas - 1))}"
                ;;
            set)
                validation_target="$TARGET_REPLICAS"
                ;;
            emergency-scale)
                validation_target=$((current_replicas * 2))
                if [[ $validation_target -gt $MAX_REPLICAS ]]; then
                    validation_target=$MAX_REPLICAS
                fi
                ;;
        esac
        
        if [[ -n "$validation_target" && "$SCALE_ACTION" != "auto" ]]; then
            pre_scaling_validation "$validation_target"
        fi
    fi
    
    # Perform the scaling operation
    if perform_scaling; then
        if [[ "$SCALE_ACTION" != "status" && "$SCALE_ACTION" != "auto" ]]; then
            if [[ -n "${validation_target:-}" ]]; then
                post_scaling_validation "$validation_target"
            fi
            send_notification "$SCALE_ACTION" "${validation_target:-}"
        fi
        
        show_status
        log_success "🎯 FlashMM scaling operation completed successfully!"
    else
        log_error "Scaling operation failed"
        exit 1
    fi
}

# Execute main function
main "$@"
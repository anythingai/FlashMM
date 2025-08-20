#!/bin/bash
# FlashMM Environment Deployment Script
# Unified deployment script for all environments

set -euo pipefail

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Default values
ENVIRONMENT=""
NAMESPACE=""
DRY_RUN=false
FORCE=false
SKIP_TESTS=false
TIMEOUT=600
HELM_CHART_PATH="${PROJECT_ROOT}/helm/flashmm"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Help function
show_help() {
    cat << EOF
FlashMM Environment Deployment Script

Usage: $0 [OPTIONS]

Options:
    -e, --environment ENVIRONMENT    Target environment (development, staging, production)
    -n, --namespace NAMESPACE        Kubernetes namespace (default: flashmm-ENVIRONMENT)
    -d, --dry-run                   Perform a dry run without making changes
    -f, --force                     Force deployment even if validation fails
    -s, --skip-tests                Skip pre-deployment tests
    -t, --timeout SECONDS          Deployment timeout in seconds (default: 600)
    -h, --help                      Show this help message

Examples:
    $0 -e development                Deploy to development environment
    $0 -e staging -d                 Dry run deployment to staging
    $0 -e production -f              Force deployment to production
    $0 -e staging -n custom-ns       Deploy to staging with custom namespace

Environments:
    development     Local development environment
    staging         Staging environment for testing
    production      Production environment

Prerequisites:
    - kubectl configured for target cluster
    - helm 3.x installed
    - Required secrets configured in target namespace

EOF
}

# Parse command line arguments
parse_args() {
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
            -d|--dry-run)
                DRY_RUN=true
                shift
                ;;
            -f|--force)
                FORCE=true
                shift
                ;;
            -s|--skip-tests)
                SKIP_TESTS=true
                shift
                ;;
            -t|--timeout)
                TIMEOUT="$2"
                shift 2
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

    # Validate required arguments
    if [[ -z "$ENVIRONMENT" ]]; then
        log_error "Environment is required. Use -e or --environment."
        show_help
        exit 1
    fi

    # Set default namespace if not provided
    if [[ -z "$NAMESPACE" ]]; then
        NAMESPACE="flashmm-${ENVIRONMENT}"
    fi

    # Validate environment
    case "$ENVIRONMENT" in
        development|staging|production)
            ;;
        *)
            log_error "Invalid environment: $ENVIRONMENT"
            log_error "Valid environments: development, staging, production"
            exit 1
            ;;
    esac
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."

    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is required but not installed"
        exit 1
    fi

    # Check helm
    if ! command -v helm &> /dev/null; then
        log_error "helm is required but not installed"
        exit 1
    fi

    # Check cluster connectivity
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        log_error "Please configure kubectl for the target cluster"
        exit 1
    fi

    # Check if values file exists
    local values_file="${SCRIPT_DIR}/${ENVIRONMENT}/values.yaml"
    if [[ ! -f "$values_file" ]]; then
        log_error "Values file not found: $values_file"
        exit 1
    fi

    # Check Helm chart
    if [[ ! -d "$HELM_CHART_PATH" ]]; then
        log_error "Helm chart not found: $HELM_CHART_PATH"
        exit 1
    fi

    log_success "Prerequisites check passed"
}

# Validate Helm chart and values
validate_deployment() {
    log_info "Validating deployment configuration..."

    local values_file="${SCRIPT_DIR}/${ENVIRONMENT}/values.yaml"
    
    # Lint Helm chart
    if ! helm lint "$HELM_CHART_PATH" -f "$values_file"; then
        log_error "Helm chart validation failed"
        if [[ "$FORCE" != "true" ]]; then
            exit 1
        else
            log_warning "Continuing due to --force flag"
        fi
    fi

    # Template and validate Kubernetes manifests
    local temp_dir=$(mktemp -d)
    if ! helm template flashmm "$HELM_CHART_PATH" \
        -f "$values_file" \
        --namespace "$NAMESPACE" \
        --output-dir "$temp_dir" &> /dev/null; then
        log_error "Helm template generation failed"
        rm -rf "$temp_dir"
        if [[ "$FORCE" != "true" ]]; then
            exit 1
        else
            log_warning "Continuing due to --force flag"
        fi
    fi

    # Validate generated manifests with kubectl
    if ! kubectl apply --dry-run=client -R -f "$temp_dir" &> /dev/null; then
        log_error "Kubernetes manifest validation failed"
        rm -rf "$temp_dir"
        if [[ "$FORCE" != "true" ]]; then
            exit 1
        else
            log_warning "Continuing due to --force flag"
        fi
    fi

    rm -rf "$temp_dir"
    log_success "Deployment configuration validation passed"
}

# Run pre-deployment tests
run_pre_deployment_tests() {
    if [[ "$SKIP_TESTS" == "true" ]]; then
        log_warning "Skipping pre-deployment tests"
        return 0
    fi

    log_info "Running pre-deployment tests..."

    # Check if there are existing resources that might conflict
    local existing_release=$(helm list -n "$NAMESPACE" -q | grep -w "flashmm" || true)
    if [[ -n "$existing_release" ]]; then
        log_info "Found existing FlashMM release in namespace $NAMESPACE"
        
        # Check if it's healthy
        local release_status=$(helm status flashmm -n "$NAMESPACE" -o json | jq -r '.info.status')
        if [[ "$release_status" != "deployed" ]]; then
            log_warning "Existing release is in $release_status state"
            if [[ "$FORCE" != "true" ]]; then
                log_error "Use --force to proceed with unhealthy release"
                exit 1
            fi
        fi
    fi

    # Environment-specific tests
    case "$ENVIRONMENT" in
        production)
            run_production_pre_tests
            ;;
        staging)
            run_staging_pre_tests
            ;;
        development)
            run_development_pre_tests
            ;;
    esac

    log_success "Pre-deployment tests passed"
}

# Production-specific pre-deployment tests
run_production_pre_tests() {
    log_info "Running production-specific tests..."

    # Check backup status
    if kubectl get cronjob -n "$NAMESPACE" flashmm-backup &> /dev/null; then
        local last_backup=$(kubectl get job -n "$NAMESPACE" -l job-name -o jsonpath='{.items[0].status.completionTime}' 2>/dev/null || echo "never")
        log_info "Last backup: $last_backup"
        
        if [[ "$last_backup" == "never" ]]; then
            log_warning "No recent backup found"
            if [[ "$FORCE" != "true" ]]; then
                log_error "Production deployment requires recent backup. Use --force to override."
                exit 1
            fi
        fi
    fi

    # Check monitoring availability
    if ! kubectl get deployment -n flashmm-monitoring prometheus-server &> /dev/null; then
        log_warning "Prometheus monitoring not found"
        if [[ "$FORCE" != "true" ]]; then
            log_error "Production deployment requires monitoring. Use --force to override."
            exit 1
        fi
    fi
}

# Staging-specific pre-deployment tests
run_staging_pre_tests() {
    log_info "Running staging-specific tests..."
    
    # Check if development environment is healthy (as a baseline)
    if kubectl get namespace flashmm-development &> /dev/null; then
        local dev_pods=$(kubectl get pods -n flashmm-development -l app.kubernetes.io/name=flashmm --field-selector=status.phase=Running -o name | wc -l)
        if [[ "$dev_pods" -eq 0 ]]; then
            log_warning "No running FlashMM pods in development environment"
        fi
    fi
}

# Development-specific pre-deployment tests
run_development_pre_tests() {
    log_info "Running development-specific tests..."
    
    # Check available resources
    local node_count=$(kubectl get nodes --no-headers | wc -l)
    if [[ "$node_count" -lt 1 ]]; then
        log_error "No available nodes for deployment"
        exit 1
    fi
    
    log_info "Available nodes: $node_count"
}

# Create namespace if it doesn't exist
ensure_namespace() {
    log_info "Ensuring namespace $NAMESPACE exists..."

    if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
        if [[ "$DRY_RUN" == "true" ]]; then
            log_info "[DRY RUN] Would create namespace: $NAMESPACE"
        else
            kubectl create namespace "$NAMESPACE"
            kubectl label namespace "$NAMESPACE" \
                name="$NAMESPACE" \
                environment="$ENVIRONMENT" \
                app.kubernetes.io/managed-by=helm \
                --overwrite
            log_success "Created namespace: $NAMESPACE"
        fi
    else
        log_info "Namespace $NAMESPACE already exists"
    fi
}

# Deploy using Helm
deploy_with_helm() {
    log_info "Deploying FlashMM to $ENVIRONMENT environment..."

    local values_file="${SCRIPT_DIR}/${ENVIRONMENT}/values.yaml"
    local helm_args=(
        "upgrade" "--install" "flashmm" "$HELM_CHART_PATH"
        "--namespace" "$NAMESPACE"
        "--create-namespace"
        "--values" "$values_file"
        "--timeout" "${TIMEOUT}s"
        "--wait"
        "--atomic"
    )

    # Add environment-specific arguments
    case "$ENVIRONMENT" in
        production)
            helm_args+=(
                "--set" "global.environment=production"
                "--set" "flashmm.image.tag=latest"
                "--set" "monitoring.enabled=true"
                "--set" "backup.enabled=true"
            )
            ;;
        staging)
            helm_args+=(
                "--set" "global.environment=staging"
                "--set" "flashmm.image.tag=staging"
                "--set" "monitoring.enabled=true"
            )
            ;;
        development)
            helm_args+=(
                "--set" "global.environment=development"
                "--set" "flashmm.image.tag=dev"
                "--set" "flashmm.config.debug=true"
            )
            ;;
    esac

    if [[ "$DRY_RUN" == "true" ]]; then
        helm_args+=("--dry-run")
        log_info "[DRY RUN] Would execute: helm ${helm_args[*]}"
    fi

    # Execute Helm deployment
    if helm "${helm_args[@]}"; then
        if [[ "$DRY_RUN" != "true" ]]; then
            log_success "FlashMM deployed successfully to $ENVIRONMENT"
        else
            log_success "Dry run completed successfully"
        fi
    else
        log_error "Deployment failed"
        
        # Show recent events for debugging
        log_info "Recent events in namespace $NAMESPACE:"
        kubectl get events -n "$NAMESPACE" --sort-by='.lastTimestamp' | tail -10
        
        exit 1
    fi
}

# Verify deployment
verify_deployment() {
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "Skipping verification for dry run"
        return 0
    fi

    log_info "Verifying deployment..."

    # Wait for pods to be ready
    log_info "Waiting for pods to be ready..."
    if ! kubectl wait --for=condition=ready pod \
        -l app.kubernetes.io/name=flashmm \
        -n "$NAMESPACE" \
        --timeout="${TIMEOUT}s"; then
        log_error "Pods failed to become ready"
        
        # Show pod status for debugging
        log_info "Pod status:"
        kubectl get pods -n "$NAMESPACE" -l app.kubernetes.io/name=flashmm
        
        exit 1
    fi

    # Check service endpoints
    log_info "Checking service endpoints..."
    local service_name="flashmm-app"
    local endpoints=$(kubectl get endpoints "$service_name" -n "$NAMESPACE" -o jsonpath='{.subsets[0].addresses[*].ip}' 2>/dev/null || echo "")
    
    if [[ -z "$endpoints" ]]; then
        log_error "No service endpoints found"
        exit 1
    fi
    
    log_info "Service endpoints: $endpoints"

    # Health check
    log_info "Performing health check..."
    if kubectl run health-check-$$-$RANDOM \
        --rm -i --restart=Never \
        --image=curlimages/curl \
        --timeout=60s \
        -n "$NAMESPACE" \
        -- curl -f "http://${service_name}:8000/health" &> /dev/null; then
        log_success "Health check passed"
    else
        log_error "Health check failed"
        exit 1
    fi

    log_success "Deployment verification completed"
}

# Show deployment status
show_status() {
    if [[ "$DRY_RUN" == "true" ]]; then
        return 0
    fi

    log_info "Deployment Status:"
    echo "===================="
    
    # Helm release status
    echo "Helm Release:"
    helm status flashmm -n "$NAMESPACE"
    echo ""
    
    # Pod status
    echo "Pods:"
    kubectl get pods -n "$NAMESPACE" -l app.kubernetes.io/name=flashmm
    echo ""
    
    # Service status
    echo "Services:"
    kubectl get services -n "$NAMESPACE" -l app.kubernetes.io/name=flashmm
    echo ""
    
    # Ingress status (if applicable)
    if kubectl get ingress -n "$NAMESPACE" &> /dev/null; then
        echo "Ingress:"
        kubectl get ingress -n "$NAMESPACE"
        echo ""
    fi
    
    # Show URLs based on environment
    case "$ENVIRONMENT" in
        production)
            echo "URLs:"
            echo "  API: https://api.flashmm.com"
            echo "  Grafana: https://grafana.flashmm.com"
            echo "  Prometheus: https://prometheus.flashmm.com"
            ;;
        staging)
            echo "URLs:"
            echo "  API: https://api.staging.flashmm.com"
            echo "  Grafana: https://grafana.staging.flashmm.com"
            ;;
        development)
            echo "URLs:"
            echo "  API: http://api.dev.flashmm.local"
            echo "  Grafana: http://grafana.dev.flashmm.local"
            ;;
    esac
}

# Cleanup function
cleanup() {
    log_info "Cleaning up temporary files..."
    # Add any cleanup logic here
}

# Main execution flow
main() {
    trap cleanup EXIT

    log_info "Starting FlashMM deployment..."
    log_info "Environment: $ENVIRONMENT"
    log_info "Namespace: $NAMESPACE"
    log_info "Dry run: $DRY_RUN"

    parse_args "$@"
    check_prerequisites
    validate_deployment
    run_pre_deployment_tests
    ensure_namespace
    deploy_with_helm
    verify_deployment
    show_status

    log_success "FlashMM deployment completed successfully!"
}

# Execute main function with all arguments
main "$@"
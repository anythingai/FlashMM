#!/bin/bash
# FlashMM Health Check Script
# Comprehensive system health monitoring and validation

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Configuration
ENVIRONMENT="production"
NAMESPACE="flashmm"
CHECK_TYPE="full"
OUTPUT_FORMAT="text"
TIMEOUT=30
ALERT_THRESHOLDS=true

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

# Health check results
declare -A HEALTH_RESULTS
OVERALL_HEALTH="healthy"

show_help() {
    cat << EOF
FlashMM Health Check Script

Usage: $0 [OPTIONS]

Options:
    -e, --environment ENV    Environment to check (default: production)
    -n, --namespace NS       Kubernetes namespace (default: flashmm)
    -t, --type TYPE         Check type: quick, full, external (default: full)
    -f, --format FORMAT     Output format: text, json, prometheus (default: text)
    --timeout SECONDS       Request timeout (default: 30)
    --no-alerts             Skip alert threshold checks
    -h, --help              Show this help message

Check Types:
    quick       Basic pod and service health
    full        Comprehensive health including databases and metrics
    external    External API and dependency health

Output Formats:
    text        Human-readable text output
    json        Structured JSON output
    prometheus  Prometheus metrics format

Examples:
    $0                      # Full health check with text output
    $0 -t quick             # Quick health check
    $0 -f json              # JSON output for monitoring integration
    $0 -t external          # Check external dependencies only

EOF
}

# Parse arguments
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
        -t|--type)
            CHECK_TYPE="$2"
            shift 2
            ;;
        -f|--format)
            OUTPUT_FORMAT="$2"
            shift 2
            ;;
        --timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        --no-alerts)
            ALERT_THRESHOLDS=false
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

# Update health result
update_health_result() {
    local component="$1"
    local status="$2"
    local message="$3"
    local metric_value="${4:-}"
    
    HEALTH_RESULTS["${component}_status"]="$status"
    HEALTH_RESULTS["${component}_message"]="$message"
    if [[ -n "$metric_value" ]]; then
        HEALTH_RESULTS["${component}_value"]="$metric_value"
    fi
    
    if [[ "$status" != "healthy" ]]; then
        OVERALL_HEALTH="unhealthy"
    fi
}

# Check FlashMM application health
check_application_health() {
    log_info "Checking FlashMM application health..."
    
    # Check if pods are running
    local running_pods=$(kubectl get pods -n "$NAMESPACE" -l app.kubernetes.io/name=flashmm --field-selector=status.phase=Running -o json | jq '.items | length')
    local total_pods=$(kubectl get pods -n "$NAMESPACE" -l app.kubernetes.io/name=flashmm -o json | jq '.items | length')
    
    if [[ $running_pods -eq $total_pods ]] && [[ $total_pods -gt 0 ]]; then
        update_health_result "application" "healthy" "All $total_pods pods running" "$running_pods/$total_pods"
    else
        update_health_result "application" "unhealthy" "$running_pods/$total_pods pods running" "$running_pods/$total_pods"
    fi
    
    # Check application endpoints
    if kubectl run health-check-app-$$-$RANDOM \
        --rm -i --restart=Never \
        --image=curlimages/curl \
        --timeout="${TIMEOUT}s" \
        -n "$NAMESPACE" \
        -- curl -f -m "$TIMEOUT" "http://flashmm-app:8000/health" >/dev/null 2>&1; then
        update_health_result "app_endpoint" "healthy" "Health endpoint responding"
    else
        update_health_result "app_endpoint" "unhealthy" "Health endpoint not responding"
    fi
    
    # Check metrics endpoint
    if kubectl run metrics-check-app-$$-$RANDOM \
        --rm -i --restart=Never \
        --image=curlimages/curl \
        --timeout="${TIMEOUT}s" \
        -n "$NAMESPACE" \
        -- curl -f -m "$TIMEOUT" "http://flashmm-app:8000/metrics" >/dev/null 2>&1; then
        update_health_result "app_metrics" "healthy" "Metrics endpoint responding"
    else
        update_health_result "app_metrics" "unhealthy" "Metrics endpoint not responding"
    fi
}

# Check database health
check_database_health() {
    log_info "Checking database health..."
    
    # PostgreSQL health
    if kubectl get deployment -n "$NAMESPACE" postgres >/dev/null 2>&1; then
        if kubectl exec -n "$NAMESPACE" deployment/postgres -- pg_isready -U flashmm -d flashmm_prod >/dev/null 2>&1; then
            # Get connection count
            local connections=$(kubectl exec -n "$NAMESPACE" deployment/postgres -- \
                psql -U flashmm -d flashmm_prod -t -c "SELECT count(*) FROM pg_stat_activity;" 2>/dev/null | tr -d '[:space:]' || echo "unknown")
            update_health_result "postgresql" "healthy" "Database responding" "$connections connections"
        else
            update_health_result "postgresql" "unhealthy" "Database not responding"
        fi
    else
        update_health_result "postgresql" "not_configured" "PostgreSQL not deployed"
    fi
    
    # Redis health
    if kubectl get deployment -n "$NAMESPACE" redis >/dev/null 2>&1; then
        if kubectl exec -n "$NAMESPACE" deployment/redis -- redis-cli ping >/dev/null 2>&1; then
            # Get memory usage
            local memory_usage=$(kubectl exec -n "$NAMESPACE" deployment/redis -- \
                redis-cli info memory | grep used_memory_human | cut -d: -f2 | tr -d '\r\n' || echo "unknown")
            update_health_result "redis" "healthy" "Cache responding" "$memory_usage"
        else
            update_health_result "redis" "unhealthy" "Cache not responding"
        fi
    else
        update_health_result "redis" "not_configured" "Redis not deployed"
    fi
    
    # InfluxDB health
    if kubectl get deployment -n "$NAMESPACE" influxdb >/dev/null 2>&1; then
        if kubectl run influx-check-$$-$RANDOM \
            --rm -i --restart=Never \
            --image=curlimages/curl \
            --timeout="${TIMEOUT}s" \
            -n "$NAMESPACE" \
            -- curl -f -m "$TIMEOUT" "http://influxdb:8086/ping" >/dev/null 2>&1; then
            update_health_result "influxdb" "healthy" "Time series database responding"
        else
            update_health_result "influxdb" "unhealthy" "Time series database not responding"
        fi
    else
        update_health_result "influxdb" "not_configured" "InfluxDB not deployed"
    fi
}

# Check monitoring stack health
check_monitoring_health() {
    if [[ "$CHECK_TYPE" != "quick" ]]; then
        log_info "Checking monitoring stack health..."
        
        # Prometheus health
        if kubectl get deployment -n flashmm-monitoring prometheus-server >/dev/null 2>&1; then
            if kubectl run prometheus-check-$$-$RANDOM \
                --rm -i --restart=Never \
                --image=curlimages/curl \
                --timeout="${TIMEOUT}s" \
                -n flashmm-monitoring \
                -- curl -f -m "$TIMEOUT" "http://prometheus:9090/-/healthy" >/dev/null 2>&1; then
                update_health_result "prometheus" "healthy" "Monitoring system responding"
            else
                update_health_result "prometheus" "unhealthy" "Monitoring system not responding"
            fi
        else
            update_health_result "prometheus" "not_configured" "Prometheus not deployed"
        fi
        
        # Grafana health
        if kubectl get deployment -n flashmm-monitoring grafana >/dev/null 2>&1; then
            if kubectl run grafana-check-$$-$RANDOM \
                --rm -i --restart=Never \
                --image=curlimages/curl \
                --timeout="${TIMEOUT}s" \
                -n flashmm-monitoring \
                -- curl -f -m "$TIMEOUT" "http://flashmm-grafana:3000/api/health" >/dev/null 2>&1; then
                update_health_result "grafana" "healthy" "Dashboard system responding"
            else
                update_health_result "grafana" "unhealthy" "Dashboard system not responding"
            fi
        else
            update_health_result "grafana" "not_configured" "Grafana not deployed"
        fi
    fi
}

# Check external dependencies
check_external_health() {
    if [[ "$CHECK_TYPE" == "external" || "$CHECK_TYPE" == "full" ]]; then
        log_info "Checking external dependencies..."
        
        # Sei Network RPC
        if curl -f -m "$TIMEOUT" "https://sei-testnet-rpc.polkachu.com/status" >/dev/null 2>&1; then
            update_health_result "sei_rpc" "healthy" "Sei RPC endpoint responding"
        else
            update_health_result "sei_rpc" "unhealthy" "Sei RPC endpoint not responding"
        fi
        
        # Cambrian API
        if curl -f -m "$TIMEOUT" -H "Content-Type: application/json" \
            "https://api.cambrian.com/health" >/dev/null 2>&1; then
            update_health_result "cambrian_api" "healthy" "Cambrian API responding"
        else
            update_health_result "cambrian_api" "unhealthy" "Cambrian API not responding"
        fi
        
        # Azure OpenAI (if configured)
        if [[ -n "${AZURE_OPENAI_ENDPOINT:-}" ]]; then
            if curl -f -m "$TIMEOUT" -H "api-key: ${AZURE_OPENAI_API_KEY:-}" \
                "${AZURE_OPENAI_ENDPOINT}/openai/deployments" >/dev/null 2>&1; then
                update_health_result "azure_openai" "healthy" "Azure OpenAI responding"
            else
                update_health_result "azure_openai" "unhealthy" "Azure OpenAI not responding"
            fi
        else
            update_health_result "azure_openai" "not_configured" "Azure OpenAI not configured"
        fi
    fi
}

# Check resource usage and thresholds
check_resource_usage() {
    if [[ "$ALERT_THRESHOLDS" == "true" && "$CHECK_TYPE" == "full" ]]; then
        log_info "Checking resource usage thresholds..."
        
        # Check CPU usage
        local cpu_usage
        cpu_usage=$(kubectl top pods -n "$NAMESPACE" -l app.kubernetes.io/name=flashmm --no-headers 2>/dev/null | \
            awk '{sum+=$2} END {print sum}' | sed 's/m//' || echo "0")
        
        if [[ $cpu_usage -gt 1500 ]]; then  # 1.5 CPU cores
            update_health_result "cpu_usage" "warning" "High CPU usage detected" "${cpu_usage}m"
        else
            update_health_result "cpu_usage" "healthy" "CPU usage within limits" "${cpu_usage}m"
        fi
        
        # Check memory usage
        local memory_usage
        memory_usage=$(kubectl top pods -n "$NAMESPACE" -l app.kubernetes.io/name=flashmm --no-headers 2>/dev/null | \
            awk '{sum+=$3} END {print sum}' | sed 's/Mi//' || echo "0")
        
        if [[ $memory_usage -gt 3072 ]]; then  # 3GB
            update_health_result "memory_usage" "warning" "High memory usage detected" "${memory_usage}Mi"
        else
            update_health_result "memory_usage" "healthy" "Memory usage within limits" "${memory_usage}Mi"
        fi
        
        # Check disk usage
        local disk_usage
        disk_usage=$(kubectl exec -n "$NAMESPACE" deployment/flashmm-app -- df -h /app | tail -1 | awk '{print $5}' | sed 's/%//' || echo "0")
        
        if [[ $disk_usage -gt 85 ]]; then
            update_health_result "disk_usage" "warning" "High disk usage detected" "${disk_usage}%"
        else
            update_health_result "disk_usage" "healthy" "Disk usage within limits" "${disk_usage}%"
        fi
    fi
}

# Check trading system health
check_trading_health() {
    if [[ "$CHECK_TYPE" == "full" ]]; then
        log_info "Checking trading system health..."
        
        # This would typically query the application's internal metrics
        # For now, we'll check if the trading endpoints respond
        
        if kubectl run trading-check-$$-$RANDOM \
            --rm -i --restart=Never \
            --image=curlimages/curl \
            --timeout="${TIMEOUT}s" \
            -n "$NAMESPACE" \
            -- curl -f -m "$TIMEOUT" "http://flashmm-app:8000/api/v1/trading/status" >/dev/null 2>&1; then
            update_health_result "trading_engine" "healthy" "Trading engine responding"
        else
            update_health_result "trading_engine" "unhealthy" "Trading engine not responding"
        fi
        
        # Check ML model health
        if kubectl run ml-check-$$-$RANDOM \
            --rm -i --restart=Never \
            --image=curlimages/curl \
            --timeout="${TIMEOUT}s" \
            -n "$NAMESPACE" \
            -- curl -f -m "$TIMEOUT" "http://flashmm-app:8000/api/v1/ml/health" >/dev/null 2>&1; then
            update_health_result "ml_engine" "healthy" "ML engine responding"
        else
            update_health_result "ml_engine" "unhealthy" "ML engine not responding"
        fi
    fi
}

# Output results in different formats
output_results() {
    case "$OUTPUT_FORMAT" in
        json)
            output_json
            ;;
        prometheus)
            output_prometheus
            ;;
        text|*)
            output_text
            ;;
    esac
}

# Text output
output_text() {
    echo ""
    echo "======================================"
    echo "FlashMM Health Check Report"
    echo "======================================"
    echo "Timestamp: $(date)"
    echo "Environment: $ENVIRONMENT"
    echo "Namespace: $NAMESPACE"
    echo "Check Type: $CHECK_TYPE"
    echo "Overall Status: $OVERALL_HEALTH"
    echo ""
    
    echo "Component Health:"
    echo "----------------"
    
    for key in "${!HEALTH_RESULTS[@]}"; do
        if [[ "$key" == *"_status" ]]; then
            local component=$(echo "$key" | sed 's/_status$//')
            local status="${HEALTH_RESULTS[$key]}"
            local message="${HEALTH_RESULTS[${component}_message]:-}"
            local value="${HEALTH_RESULTS[${component}_value]:-}"
            
            local status_icon
            case "$status" in
                healthy) status_icon="✅" ;;
                unhealthy) status_icon="❌" ;;
                warning) status_icon="⚠️" ;;
                not_configured) status_icon="⚪" ;;
                *) status_icon="❓" ;;
            esac
            
            printf "  %-20s %s %-12s %s" "$component" "$status_icon" "$status" "$message"
            if [[ -n "$value" ]]; then
                printf " (%s)" "$value"
            fi
            echo ""
        fi
    done
    
    echo ""
    if [[ "$OVERALL_HEALTH" == "healthy" ]]; then
        echo -e "${GREEN}Overall System Status: HEALTHY${NC} ✅"
    else
        echo -e "${RED}Overall System Status: UNHEALTHY${NC} ❌"
    fi
}

# JSON output
output_json() {
    local json_output='{"timestamp":"'$(date -u +%Y-%m-%dT%H:%M:%SZ)'","environment":"'$ENVIRONMENT'","namespace":"'$NAMESPACE'","check_type":"'$CHECK_TYPE'","overall_health":"'$OVERALL_HEALTH'","components":{}}'
    
    for key in "${!HEALTH_RESULTS[@]}"; do
        if [[ "$key" == *"_status" ]]; then
            local component=$(echo "$key" | sed 's/_status$//')
            local status="${HEALTH_RESULTS[$key]}"
            local message="${HEALTH_RESULTS[${component}_message]:-}"
            local value="${HEALTH_RESULTS[${component}_value]:-}"
            
            json_output=$(echo "$json_output" | jq --arg comp "$component" --arg status "$status" --arg msg "$message" --arg val "$value" \
                '.components[$comp] = {"status": $status, "message": $msg, "value": $val}')
        fi
    done
    
    echo "$json_output" | jq .
}

# Prometheus metrics output
output_prometheus() {
    echo "# HELP flashmm_health_status Health status of FlashMM components (1=healthy, 0=unhealthy)"
    echo "# TYPE flashmm_health_status gauge"
    
    for key in "${!HEALTH_RESULTS[@]}"; do
        if [[ "$key" == *"_status" ]]; then
            local component=$(echo "$key" | sed 's/_status$//')
            local status="${HEALTH_RESULTS[$key]}"
            local message="${HEALTH_RESULTS[${component}_message]:-}"
            
            local metric_value
            case "$status" in
                healthy) metric_value=1 ;;
                warning) metric_value=0.5 ;;
                *) metric_value=0 ;;
            esac
            
            echo "flashmm_health_status{component=\"$component\",environment=\"$ENVIRONMENT\",namespace=\"$NAMESPACE\"} $metric_value"
        fi
    done
    
    # Overall health metric
    local overall_value
    case "$OVERALL_HEALTH" in
        healthy) overall_value=1 ;;
        *) overall_value=0 ;;
    esac
    
    echo "flashmm_health_overall{environment=\"$ENVIRONMENT\",namespace=\"$NAMESPACE\"} $overall_value"
}

# Main execution
main() {
    # Prerequisites check
    command -v kubectl >/dev/null 2>&1 || { log_error "kubectl is required"; exit 1; }
    command -v jq >/dev/null 2>&1 || { log_error "jq is required"; exit 1; }
    
    kubectl cluster-info >/dev/null 2>&1 || { log_error "Cannot connect to cluster"; exit 1; }
    kubectl get namespace "$NAMESPACE" >/dev/null 2>&1 || { log_error "Namespace $NAMESPACE not found"; exit 1; }
    
    if [[ "$OUTPUT_FORMAT" == "text" ]]; then
        log_info "Starting FlashMM health check..."
        log_info "Environment: $ENVIRONMENT, Namespace: $NAMESPACE, Type: $CHECK_TYPE"
    fi
    
    # Run health checks based on type
    case "$CHECK_TYPE" in
        quick)
            check_application_health
            ;;
        full)
            check_application_health
            check_database_health
            check_monitoring_health
            check_resource_usage
            check_trading_health
            ;;
        external)
            check_external_health
            ;;
    esac
    
    # Output results
    output_results
    
    # Exit with appropriate code
    if [[ "$OVERALL_HEALTH" == "healthy" ]]; then
        exit 0
    else
        exit 1
    fi
}

# Execute main function
main "$@"
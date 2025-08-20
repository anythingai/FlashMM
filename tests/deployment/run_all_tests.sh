#!/bin/bash
# FlashMM Comprehensive Test Suite Runner
# Orchestrates all deployment and infrastructure tests

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Configuration
ENVIRONMENT="staging"
NAMESPACE="flashmm"
TEST_TYPES=("unit" "integration" "infrastructure" "security" "performance")
PARALLEL_TESTS=false
GENERATE_REPORT=true
TIMEOUT=1800  # 30 minutes total timeout

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

# Test results
declare -A TEST_RESULTS
OVERALL_SUCCESS=true

show_help() {
    cat << EOF
FlashMM Comprehensive Test Suite Runner

Usage: $0 [OPTIONS]

Options:
    -e, --environment ENV    Environment to test (default: staging)
    -n, --namespace NS       Kubernetes namespace (default: flashmm)
    -t, --types TYPES        Test types to run (comma-separated)
    -p, --parallel           Run tests in parallel
    --no-report             Skip report generation
    --timeout SECONDS       Total timeout for all tests (default: 1800)
    --quick                 Run only quick tests
    --full                  Run comprehensive test suite
    -h, --help              Show this help message

Test Types:
    unit            Unit tests for individual components
    integration     Integration tests across components
    infrastructure  Infrastructure and deployment tests
    security        Security compliance and vulnerability tests
    performance     Load and performance tests

Examples:
    $0                      # Run all tests
    $0 -t unit,integration  # Run only unit and integration tests
    $0 --quick              # Quick validation test suite
    $0 --full -p            # Full test suite in parallel
    $0 -e production        # Test production environment

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
        -t|--types)
            IFS=',' read -ra TEST_TYPES <<< "$2"
            shift 2
            ;;
        -p|--parallel)
            PARALLEL_TESTS=true
            shift
            ;;
        --no-report)
            GENERATE_REPORT=false
            shift
            ;;
        --timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        --quick)
            TEST_TYPES=("infrastructure" "security")
            shift
            ;;
        --full)
            TEST_TYPES=("unit" "integration" "infrastructure" "security" "performance")
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
    
    # Check required tools
    local required_tools=("kubectl" "python3" "pip")
    for tool in "${required_tools[@]}"; do
        if ! command -v "$tool" >/dev/null 2>&1; then
            log_error "$tool is required but not installed"
            exit 1
        fi
    done
    
    # Check Python dependencies
    if ! python3 -c "import pytest, requests, aiohttp" >/dev/null 2>&1; then
        log_info "Installing Python test dependencies..."
        pip install pytest pytest-asyncio requests aiohttp
    fi
    
    # Check cluster connectivity
    kubectl cluster-info >/dev/null 2>&1 || {
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    }
    
    # Check namespace exists
    kubectl get namespace "$NAMESPACE" >/dev/null 2>&1 || {
        log_error "Namespace $NAMESPACE not found"
        exit 1
    }
    
    log_success "Prerequisites check passed"
}

# Run unit tests
run_unit_tests() {
    log_info "Running unit tests..."
    
    cd "$PROJECT_ROOT"
    
    if python3 -m pytest tests/unit/ -v \
        --tb=short \
        --junitxml="test-results/unit-tests.xml" \
        --cov=src/flashmm \
        --cov-report=xml:test-results/unit-coverage.xml \
        --cov-report=html:test-results/unit-coverage-html \
        --timeout=300; then
        TEST_RESULTS["unit"]="pass"
        log_success "Unit tests passed"
    else
        TEST_RESULTS["unit"]="fail"
        OVERALL_SUCCESS=false
        log_error "Unit tests failed"
    fi
}

# Run integration tests
run_integration_tests() {
    log_info "Running integration tests..."
    
    cd "$PROJECT_ROOT"
    
    # Set environment variables for integration tests
    export ENVIRONMENT="$ENVIRONMENT"
    export KUBERNETES_NAMESPACE="$NAMESPACE"
    export REDIS_URL="redis://localhost:6379/0"
    export POSTGRES_URL="postgresql://flashmm:test@localhost:5432/flashmm_test"
    
    if python3 -m pytest tests/integration/ -v \
        --tb=short \
        --junitxml="test-results/integration-tests.xml" \
        --timeout=600; then
        TEST_RESULTS["integration"]="pass"
        log_success "Integration tests passed"
    else
        TEST_RESULTS["integration"]="fail"
        OVERALL_SUCCESS=false
        log_error "Integration tests failed"
    fi
}

# Run infrastructure tests
run_infrastructure_tests() {
    log_info "Running infrastructure tests..."
    
    cd "$PROJECT_ROOT"
    
    if python3 tests/deployment/test_infrastructure.py \
        -e "$ENVIRONMENT" \
        -n "$NAMESPACE" \
        -t full \
        -v; then
        TEST_RESULTS["infrastructure"]="pass"
        log_success "Infrastructure tests passed"
    else
        TEST_RESULTS["infrastructure"]="fail"
        OVERALL_SUCCESS=false
        log_error "Infrastructure tests failed"
    fi
}

# Run security tests
run_security_tests() {
    log_info "Running security compliance tests..."
    
    # Run security compliance scan
    if "${PROJECT_ROOT}/security/scripts/security-scan.sh" \
        -e "$ENVIRONMENT" \
        -n "$NAMESPACE" \
        -t full \
        -f json > "test-results/security-scan.json"; then
        TEST_RESULTS["security"]="pass"
        log_success "Security tests passed"
    else
        TEST_RESULTS["security"]="fail"
        OVERALL_SUCCESS=false
        log_error "Security tests failed"
    fi
    
    # Run container vulnerability scan
    if command -v trivy >/dev/null 2>&1; then
        log_info "Running container vulnerability scan..."
        trivy image --format json --output "test-results/trivy-scan.json" \
            "ghcr.io/flashmm/flashmm:latest" || log_warning "Trivy scan failed"
    fi
}

# Run performance tests  
run_performance_tests() {
    log_info "Running performance and load tests..."
    
    cd "$PROJECT_ROOT"
    
    # Determine base URL based on environment
    local base_url
    case "$ENVIRONMENT" in
        production)
            base_url="https://api.flashmm.com"
            ;;
        staging)
            base_url="https://api.staging.flashmm.com"
            ;;
        development)
            base_url="http://api.dev.flashmm.local"
            ;;
        *)
            base_url="http://localhost:8000"
            ;;
    esac
    
    if python3 tests/performance/load_test.py \
        --base-url "$base_url" \
        --test-type api \
        --users 50 \
        --duration 120 \
        --output "test-results/load-test-results.json"; then
        TEST_RESULTS["performance"]="pass"
        log_success "Performance tests passed"
    else
        TEST_RESULTS["performance"]="fail"
        OVERALL_SUCCESS=false
        log_error "Performance tests failed"
    fi
}

# Run tests in parallel
run_tests_parallel() {
    log_info "Running tests in parallel..."
    
    local pids=()
    
    for test_type in "${TEST_TYPES[@]}"; do
        case "$test_type" in
            unit)
                run_unit_tests &
                pids+=($!)
                ;;
            integration)
                run_integration_tests &
                pids+=($!)
                ;;
            infrastructure)
                run_infrastructure_tests &
                pids+=($!)
                ;;
            security)
                run_security_tests &
                pids+=($!)
                ;;
            performance)
                run_performance_tests &
                pids+=($!)
                ;;
        esac
    done
    
    # Wait for all tests to complete
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            OVERALL_SUCCESS=false
        fi
    done
}

# Run tests sequentially
run_tests_sequential() {
    log_info "Running tests sequentially..."
    
    for test_type in "${TEST_TYPES[@]}"; do
        case "$test_type" in
            unit)
                run_unit_tests
                ;;
            integration)
                run_integration_tests
                ;;
            infrastructure)
                run_infrastructure_tests
                ;;
            security)
                run_security_tests
                ;;
            performance)
                run_performance_tests
                ;;
            *)
                log_warning "Unknown test type: $test_type"
                ;;
        esac
    done
}

# Generate comprehensive test report
generate_test_report() {
    if [[ "$GENERATE_REPORT" != "true" ]]; then
        return 0
    fi
    
    log_info "Generating comprehensive test report..."
    
    local report_file="test-results/comprehensive-test-report.html"
    mkdir -p "$(dirname "$report_file")"
    
    cat > "$report_file" << EOF
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FlashMM Deployment Test Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; }
        .header { text-align: center; margin-bottom: 30px; }
        .summary { display: flex; justify-content: space-around; margin-bottom: 30px; }
        .summary-box { padding: 20px; border-radius: 8px; text-align: center; min-width: 150px; }
        .success { background-color: #d4edda; border: 1px solid #c3e6cb; color: #155724; }
        .failure { background-color: #f8d7da; border: 1px solid #f5c6cb; color: #721c24; }
        .warning { background-color: #fff3cd; border: 1px solid #ffeaa7; color: #856404; }
        .test-section { margin-bottom: 30px; }
        .test-result { margin: 10px 0; padding: 10px; border-radius: 5px; }
        .pass { background-color: #d4edda; }
        .fail { background-color: #f8d7da; }
        .timestamp { text-align: right; font-size: 0.9em; color: #666; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>FlashMM Deployment Test Report</h1>
            <p>Environment: <strong>$ENVIRONMENT</strong> | Namespace: <strong>$NAMESPACE</strong></p>
            <p class="timestamp">Generated: $(date)</p>
        </div>
        
        <div class="summary">
EOF

    # Calculate summary statistics
    local total_tests=${#TEST_TYPES[@]}
    local passed_tests=0
    local failed_tests=0
    
    for test_type in "${TEST_TYPES[@]}"; do
        if [[ "${TEST_RESULTS[$test_type]:-fail}" == "pass" ]]; then
            ((passed_tests++))
        else
            ((failed_tests++))
        fi
    done
    
    # Add summary boxes
    cat >> "$report_file" << EOF
            <div class="summary-box $([ $failed_tests -eq 0 ] && echo "success" || echo "failure")">
                <h3>Overall Status</h3>
                <p>$([ $failed_tests -eq 0 ] && echo "✅ PASS" || echo "❌ FAIL")</p>
            </div>
            <div class="summary-box">
                <h3>Tests Run</h3>
                <p>$total_tests</p>
            </div>
            <div class="summary-box success">
                <h3>Passed</h3>
                <p>$passed_tests</p>
            </div>
            <div class="summary-box $([ $failed_tests -gt 0 ] && echo "failure" || echo "success")">
                <h3>Failed</h3>
                <p>$failed_tests</p>
            </div>
        </div>
        
        <div class="test-section">
            <h2>Test Results</h2>
EOF

    # Add individual test results
    for test_type in "${TEST_TYPES[@]}"; do
        local result="${TEST_RESULTS[$test_type]:-fail}"
        local css_class="$result"
        local icon=$([ "$result" == "pass" ] && echo "✅" || echo "❌")
        
        cat >> "$report_file" << EOF
            <div class="test-result $css_class">
                <h3>$icon ${test_type^} Tests</h3>
                <p>Status: $(echo "$result" | tr '[:lower:]' '[:upper:]')</p>
            </div>
EOF
    done
    
    cat >> "$report_file" << EOF
        </div>
        
        <div class="test-section">
            <h2>Environment Information</h2>
            <ul>
                <li><strong>Environment:</strong> $ENVIRONMENT</li>
                <li><strong>Namespace:</strong> $NAMESPACE</li>
                <li><strong>Test Types:</strong> ${TEST_TYPES[*]}</li>
                <li><strong>Parallel Execution:</strong> $PARALLEL_TESTS</li>
                <li><strong>Total Duration:</strong> $(($(date +%s) - START_TIME))s</li>
            </ul>
        </div>
        
        <div class="test-section">
            <h2>Next Steps</h2>
            <ul>
EOF

    if [[ $failed_tests -gt 0 ]]; then
        cat >> "$report_file" << EOF
                <li>❌ Review failed tests and address issues</li>
                <li>🔧 Check deployment logs: <code>kubectl logs -n $NAMESPACE deployment/flashmm-app</code></li>
                <li>📊 Monitor system health: <code>./scripts/health-check.sh -e $ENVIRONMENT</code></li>
                <li>🔄 Re-run tests after fixes: <code>$0 -e $ENVIRONMENT</code></li>
EOF
    else
        cat >> "$report_file" << EOF
                <li>✅ All tests passed! Deployment is ready for use</li>
                <li>📊 Monitor ongoing performance: Access Grafana dashboard</li>
                <li>🔍 Review security posture: Run security scans periodically</li>
                <li>💾 Verify backup procedures: Test backup and recovery processes</li>
EOF
    fi
    
    cat >> "$report_file" << EOF
            </ul>
        </div>
    </div>
</body>
</html>
EOF
    
    log_success "Test report generated: $report_file"
}

# Create test results directory
setup_test_environment() {
    mkdir -p "${PROJECT_ROOT}/test-results"
    
    # Record start time
    START_TIME=$(date +%s)
    
    # Create test summary file
    cat > "${PROJECT_ROOT}/test-results/test-summary.json" << EOF
{
    "test_run": {
        "start_time": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
        "environment": "$ENVIRONMENT", 
        "namespace": "$NAMESPACE",
        "test_types": [$(printf '"%s",' "${TEST_TYPES[@]}" | sed 's/,$//')]
    },
    "results": {}
}
EOF
}

# Update test summary
update_test_summary() {
    local test_type="$1"
    local result="$2"
    local duration="${3:-0}"
    
    local temp_file=$(mktemp)
    jq --arg type "$test_type" --arg result "$result" --arg duration "$duration" \
        '.results[$type] = {"status": $result, "duration": ($duration | tonumber)}' \
        "${PROJECT_ROOT}/test-results/test-summary.json" > "$temp_file" && \
        mv "$temp_file" "${PROJECT_ROOT}/test-results/test-summary.json"
}

# Show test summary
show_test_summary() {
    echo ""
    echo "======================================"
    echo "FlashMM Test Suite Summary"
    echo "======================================"
    echo "Environment: $ENVIRONMENT"
    echo "Namespace: $NAMESPACE"
    echo "Start Time: $(date -d @$START_TIME)"
    echo "Duration: $(($(date +%s) - START_TIME))s"
    echo ""
    
    echo "Test Results:"
    echo "-------------"
    
    for test_type in "${TEST_TYPES[@]}"; do
        local result="${TEST_RESULTS[$test_type]:-skip}"
        local icon
        case "$result" in
            pass) icon="✅" ;;
            fail) icon="❌" ;;
            skip) icon="⏭️" ;;
            *) icon="❓" ;;
        esac
        
        printf "  %-15s %s %s\n" "$test_type" "$icon" "$(echo "$result" | tr '[:lower:]' '[:upper:]')"
    done
    
    echo ""
    
    if [[ "$OVERALL_SUCCESS" == "true" ]]; then
        echo -e "${GREEN}🎉 All tests passed! FlashMM deployment is validated.${NC}"
    else
        echo -e "${RED}❌ Some tests failed. Review results and address issues.${NC}"
    fi
    
    echo ""
    echo "Detailed results available in: ${PROJECT_ROOT}/test-results/"
}

# Cleanup function
cleanup() {
    log_info "Cleaning up test resources..."
    
    # Clean up any test pods that might be hanging around
    kubectl delete pods -n "$NAMESPACE" -l "app=test" --ignore-not-found=true >/dev/null 2>&1 || true
    
    # Final test summary update
    if [[ -f "${PROJECT_ROOT}/test-results/test-summary.json" ]]; then
        local temp_file=$(mktemp)
        jq --arg end_time "$(date -u +%Y-%m-%dT%H:%M:%SZ)" --arg duration "$(($(date +%s) - START_TIME))" \
            '.test_run.end_time = $end_time | .test_run.duration = ($duration | tonumber)' \
            "${PROJECT_ROOT}/test-results/test-summary.json" > "$temp_file" && \
            mv "$temp_file" "${PROJECT_ROOT}/test-results/test-summary.json"
    fi
}

trap cleanup EXIT

# Main execution
main() {
    log_info "Starting FlashMM comprehensive test suite..."
    
    check_prerequisites
    setup_test_environment
    
    # Run tests based on configuration
    if [[ "$PARALLEL_TESTS" == "true" ]]; then
        run_tests_parallel
    else
        run_tests_sequential
    fi
    
    # Generate reports
    if [[ "$GENERATE_REPORT" == "true" ]]; then
        generate_test_report
    fi
    
    show_test_summary
    
    # Exit with appropriate code
    if [[ "$OVERALL_SUCCESS" == "true" ]]; then
        log_success "🚀 All tests completed successfully!"
        exit 0
    else
        log_error "🔥 Test suite failed. Check individual test results."
        exit 1
    fi
}

# Execute main function
main "$@"
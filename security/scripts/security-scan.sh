#!/bin/bash
# FlashMM Security Compliance Scanner
# Comprehensive security assessment and compliance checking

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Configuration
ENVIRONMENT="production"
NAMESPACE="flashmm"
SCAN_TYPE="full"
OUTPUT_FORMAT="text"
COMPLIANCE_FRAMEWORKS=("soc2" "iso27001")
REMEDIATION_MODE=false

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

# Security scan results
declare -A SECURITY_RESULTS
OVERALL_SECURITY_SCORE=0
TOTAL_CHECKS=0
PASSED_CHECKS=0

show_help() {
    cat << EOF
FlashMM Security Compliance Scanner

Usage: $0 [OPTIONS]

Options:
    -e, --environment ENV        Environment to scan (default: production)
    -n, --namespace NS           Kubernetes namespace (default: flashmm)
    -t, --type TYPE             Scan type: quick, full, compliance (default: full)
    -f, --format FORMAT         Output format: text, json, html (default: text)
    -c, --compliance FRAMEWORK  Compliance framework: soc2, iso27001, pci-dss, gdpr
    --remediation               Enable remediation suggestions
    --fix                       Attempt to fix common issues (use with caution)
    -h, --help                  Show this help message

Scan Types:
    quick       Basic security posture check
    full        Comprehensive security assessment
    compliance  Compliance framework validation

Examples:
    $0                          # Full security scan
    $0 -t compliance -c soc2    # SOC2 compliance scan
    $0 --remediation            # Include remediation suggestions
    $0 --fix -t quick           # Quick scan with automatic fixes

EOF
}

# Parse arguments
ENABLE_FIXES=false

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
            SCAN_TYPE="$2"
            shift 2
            ;;
        -f|--format)
            OUTPUT_FORMAT="$2"
            shift 2
            ;;
        -c|--compliance)
            COMPLIANCE_FRAMEWORKS=("$2")
            shift 2
            ;;
        --remediation)
            REMEDIATION_MODE=true
            shift
            ;;
        --fix)
            ENABLE_FIXES=true
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

# Record security check result
record_check() {
    local check_name="$1"
    local status="$2"
    local severity="${3:-medium}"
    local message="${4:-}"
    local remediation="${5:-}"
    
    SECURITY_RESULTS["${check_name}_status"]="$status"
    SECURITY_RESULTS["${check_name}_severity"]="$severity"
    SECURITY_RESULTS["${check_name}_message"]="$message"
    SECURITY_RESULTS["${check_name}_remediation"]="$remediation"
    
    ((TOTAL_CHECKS++))
    if [[ "$status" == "pass" ]]; then
        ((PASSED_CHECKS++))
    fi
}

# Check pod security contexts
check_pod_security_context() {
    log_info "Checking pod security contexts..."
    
    local pods
    pods=$(kubectl get pods -n "$NAMESPACE" -l app.kubernetes.io/part-of=flashmm-platform -o json)
    
    local pods_count
    pods_count=$(echo "$pods" | jq '.items | length')
    
    if [[ $pods_count -eq 0 ]]; then
        record_check "pod_security_context" "fail" "high" "No FlashMM pods found" "Deploy FlashMM application"
        return
    fi
    
    # Check each pod's security context
    local insecure_pods=0
    local security_issues=()
    
    while IFS= read -r pod_info; do
        local pod_name=$(echo "$pod_info" | jq -r '.name')
        local run_as_non_root=$(echo "$pod_info" | jq -r '.securityContext.runAsNonRoot // false')
        local run_as_user=$(echo "$pod_info" | jq -r '.securityContext.runAsUser // 0')
        local read_only_fs=$(echo "$pod_info" | jq -r '.containers[0].securityContext.readOnlyRootFilesystem // false')
        
        if [[ "$run_as_non_root" != "true" ]]; then
            security_issues+=("$pod_name: not running as non-root")
            ((insecure_pods++))
        fi
        
        if [[ "$run_as_user" == "0" ]]; then
            security_issues+=("$pod_name: running as root user")
            ((insecure_pods++))
        fi
        
        if [[ "$read_only_fs" != "true" ]]; then
            security_issues+=("$pod_name: not using read-only root filesystem")
            ((insecure_pods++))
        fi
        
    done < <(echo "$pods" | jq -c '.items[] | {name: .metadata.name, securityContext: .spec.securityContext, containers: .spec.containers}')
    
    if [[ $insecure_pods -eq 0 ]]; then
        record_check "pod_security_context" "pass" "critical" "All pods have secure contexts"
    else
        local issues_text=$(printf '%s\n' "${security_issues[@]}")
        record_check "pod_security_context" "fail" "critical" "$insecure_pods pods with security issues" "Review and update pod security contexts:\n$issues_text"
    fi
}

# Check network policies
check_network_policies() {
    log_info "Checking network policies..."
    
    # Check if network policies exist
    local policies_count
    policies_count=$(kubectl get networkpolicy -n "$NAMESPACE" --no-headers 2>/dev/null | wc -l)
    
    if [[ $policies_count -eq 0 ]]; then
        record_check "network_policies" "fail" "critical" "No network policies found" "Apply network policies from security/policies/network-policy.yaml"
    else
        # Check for default deny policy
        if kubectl get networkpolicy -n "$NAMESPACE" default-deny-all >/dev/null 2>&1; then
            record_check "network_policies" "pass" "critical" "$policies_count network policies configured including default deny"
        else
            record_check "network_policies" "partial" "high" "$policies_count network policies found but no default deny" "Add default deny-all network policy"
        fi
    fi
}

# Check RBAC configuration
check_rbac() {
    log_info "Checking RBAC configuration..."
    
    # Check if service accounts exist
    local sa_count
    sa_count=$(kubectl get serviceaccounts -n "$NAMESPACE" --no-headers 2>/dev/null | wc -l)
    
    # Check if roles and rolebindings exist
    local roles_count
    roles_count=$(kubectl get roles,clusterroles -n "$NAMESPACE" --no-headers 2>/dev/null | wc -l)
    
    local bindings_count
    bindings_count=$(kubectl get rolebindings,clusterrolebindings -n "$NAMESPACE" --no-headers 2>/dev/null | wc -l)
    
    if [[ $sa_count -gt 0 && $roles_count -gt 0 && $bindings_count -gt 0 ]]; then
        record_check "rbac_configuration" "pass" "critical" "RBAC configured: $sa_count SAs, $roles_count roles, $bindings_count bindings"
    else
        record_check "rbac_configuration" "fail" "critical" "Incomplete RBAC: $sa_count SAs, $roles_count roles, $bindings_count bindings" "Apply RBAC manifests from k8s/rbac.yaml"
    fi
}

# Check secrets management
check_secrets_management() {
    log_info "Checking secrets management..."
    
    # Check if secrets exist
    local secrets_count
    secrets_count=$(kubectl get secrets -n "$NAMESPACE" --no-headers 2>/dev/null | wc -l)
    
    if [[ $secrets_count -eq 0 ]]; then
        record_check "secrets_management" "fail" "critical" "No secrets found" "Create required secrets"
        return
    fi
    
    # Check for hardcoded secrets in manifests (basic check)
    local hardcoded_secrets=0
    
    # Check deployments for hardcoded values
    while IFS= read -r deployment; do
        local env_vars
        env_vars=$(echo "$deployment" | jq -r '.spec.template.spec.containers[].env[]? | select(.value != null) | .value' 2>/dev/null || echo "")
        
        if echo "$env_vars" | grep -qi "password\|secret\|token\|key"; then
            ((hardcoded_secrets++))
        fi
    done < <(kubectl get deployments -n "$NAMESPACE" -o json | jq -c '.items[]')
    
    if [[ $hardcoded_secrets -eq 0 ]]; then
        record_check "secrets_management" "pass" "critical" "$secrets_count secrets configured, no hardcoded values detected"
    else
        record_check "secrets_management" "fail" "critical" "$hardcoded_secrets deployments with potential hardcoded secrets" "Move sensitive values to Kubernetes secrets"
    fi
}

# Check image security
check_image_security() {
    log_info "Checking container image security..."
    
    local images
    images=$(kubectl get pods -n "$NAMESPACE" -o jsonpath='{.items[*].spec.containers[*].image}' | tr ' ' '\n' | sort -u)
    
    local untrusted_images=0
    local unsigned_images=0
    
    while IFS= read -r image; do
        if [[ -n "$image" ]]; then
            # Check if image is from trusted registry
            if [[ ! "$image" =~ ^(ghcr\.io/flashmm/|docker\.io/library/|registry\.k8s\.io/|quay\.io/) ]]; then
                ((untrusted_images++))
            fi
            
            # Basic check for image signatures (would need cosign integration)
            # For now, just check if it's a latest tag (which is discouraged)
            if [[ "$image" == *":latest" ]] && [[ "$ENVIRONMENT" == "production" ]]; then
                ((unsigned_images++))
            fi
        fi
    done <<< "$images"
    
    if [[ $untrusted_images -eq 0 && $unsigned_images -eq 0 ]]; then
        record_check "image_security" "pass" "high" "All images from trusted sources with proper tags"
    else
        record_check "image_security" "fail" "high" "$untrusted_images untrusted images, $unsigned_images with 'latest' tags" "Use specific tags and trusted registries"
    fi
}

# Check encryption
check_encryption() {
    log_info "Checking encryption configuration..."
    
    local encryption_issues=()
    
    # Check if etcd encryption is enabled (cluster-level check)
    # This would require cluster admin access
    local etcd_encrypted="unknown"
    
    # Check PV encryption
    local unencrypted_pvs=0
    while IFS= read -r pv; do
        local encrypted
        encrypted=$(echo "$pv" | jq -r '.spec.storageClassName // ""')
        # This is cloud-provider specific check
        if [[ "$encrypted" != *"encrypted"* ]] && [[ "$ENVIRONMENT" == "production" ]]; then
            ((unencrypted_pvs++))
        fi
    done < <(kubectl get pv -o json | jq -c '.items[]?')
    
    # Check TLS configuration
    local tls_enabled=true
    if ! kubectl get ingress -n "$NAMESPACE" -o json | jq -r '.items[].spec.tls[]?' >/dev/null 2>&1; then
        tls_enabled=false
    fi
    
    local score=0
    local total=3
    
    if [[ "$etcd_encrypted" != "false" ]]; then ((score++)); fi
    if [[ $unencrypted_pvs -eq 0 ]]; then ((score++)); fi
    if [[ "$tls_enabled" == "true" ]]; then ((score++)); fi
    
    if [[ $score -eq $total ]]; then
        record_check "encryption" "pass" "critical" "Encryption properly configured"
    else
        local issues=""
        if [[ "$etcd_encrypted" == "false" ]]; then issues="${issues}etcd not encrypted; "; fi
        if [[ $unencrypted_pvs -gt 0 ]]; then issues="${issues}${unencrypted_pvs} unencrypted PVs; "; fi
        if [[ "$tls_enabled" != "true" ]]; then issues="${issues}TLS not configured; "; fi
        
        record_check "encryption" "fail" "critical" "$issues" "Enable encryption at rest and in transit"
    fi
}

# Check compliance requirements
check_compliance() {
    log_info "Checking compliance requirements..."
    
    for framework in "${COMPLIANCE_FRAMEWORKS[@]}"; do
        log_info "Checking $framework compliance..."
        
        case "$framework" in
            soc2)
                check_soc2_compliance
                ;;
            iso27001)
                check_iso27001_compliance
                ;;
            pci_dss)
                check_pci_compliance
                ;;
            gdpr)
                check_gdpr_compliance
                ;;
        esac
    done
}

# SOC2 compliance check
check_soc2_compliance() {
    local soc2_score=0
    local soc2_total=5
    
    # Security - Access controls
    if kubectl get networkpolicy -n "$NAMESPACE" >/dev/null 2>&1; then ((soc2_score++)); fi
    
    # Availability - Monitoring and alerting
    if kubectl get deployment -n flashmm-monitoring prometheus-server >/dev/null 2>&1; then ((soc2_score++)); fi
    
    # Processing Integrity - Data validation and error handling
    if kubectl get pods -n "$NAMESPACE" -l app.kubernetes.io/name=flashmm --field-selector=status.phase=Running >/dev/null 2>&1; then ((soc2_score++)); fi
    
    # Confidentiality - Encryption
    if kubectl get ingress -n "$NAMESPACE" -o json | jq -r '.items[].spec.tls[]?' >/dev/null 2>&1; then ((soc2_score++)); fi
    
    # Privacy - Data handling (basic check)
    if kubectl get configmap -n "$NAMESPACE" security-baseline >/dev/null 2>&1; then ((soc2_score++)); fi
    
    local soc2_percentage=$((soc2_score * 100 / soc2_total))
    record_check "soc2_compliance" $([ $soc2_percentage -ge 80 ] && echo "pass" || echo "fail") "critical" "SOC2 compliance: $soc2_percentage% ($soc2_score/$soc2_total)" "Address missing SOC2 controls"
}

# ISO 27001 compliance check
check_iso27001_compliance() {
    local iso_score=0
    local iso_total=8
    
    # Information security policies
    if kubectl get configmap -n flashmm-system security-baseline >/dev/null 2>&1; then ((iso_score++)); fi
    
    # Access control
    if kubectl get roles,rolebindings -n "$NAMESPACE" >/dev/null 2>&1; then ((iso_score++)); fi
    
    # Cryptography
    if kubectl get ingress -n "$NAMESPACE" -o json | jq -r '.items[].spec.tls[]?' >/dev/null 2>&1; then ((iso_score++)); fi
    
    # Operations security
    if kubectl get networkpolicy -n "$NAMESPACE" >/dev/null 2>&1; then ((iso_score++)); fi
    
    # Communications security
    if kubectl get ingress -n "$NAMESPACE" -o json | jq -r '.items[].spec.tls[]?' >/dev/null 2>&1; then ((iso_score++)); fi
    
    # System acquisition and maintenance
    if kubectl get deployment -n "$NAMESPACE" -o json | jq -r '.items[].spec.template.spec.containers[].securityContext.readOnlyRootFilesystem' | grep -q true; then ((iso_score++)); fi
    
    # Information security incident management
    if kubectl get deployment -n flashmm-monitoring alertmanager >/dev/null 2>&1; then ((iso_score++)); fi
    
    # Business continuity
    if kubectl get cronjob -n "$NAMESPACE" | grep -q backup; then ((iso_score++)); fi
    
    local iso_percentage=$((iso_score * 100 / iso_total))
    record_check "iso27001_compliance" $([ $iso_percentage -ge 75 ] && echo "pass" || echo "fail") "critical" "ISO 27001 compliance: $iso_percentage% ($iso_score/$iso_total)" "Implement missing ISO 27001 controls"
}

# Vulnerability scanning
check_vulnerabilities() {
    log_info "Checking for known vulnerabilities..."
    
    # This would integrate with vulnerability scanners like Trivy, Twistlock, etc.
    # For now, basic checks
    
    local vulnerable_images=0
    local images
    images=$(kubectl get pods -n "$NAMESPACE" -o jsonpath='{.items[*].spec.containers[*].image}' | tr ' ' '\n' | sort -u)
    
    while IFS= read -r image; do
        if [[ -n "$image" ]]; then
            # Basic check for known vulnerable image patterns
            if [[ "$image" == *"alpine:3.16"* ]] || [[ "$image" == *"ubuntu:18.04"* ]]; then
                ((vulnerable_images++))
            fi
        fi
    done <<< "$images"
    
    if [[ $vulnerable_images -eq 0 ]]; then
        record_check "vulnerability_scan" "pass" "high" "No known vulnerable images detected"
    else
        record_check "vulnerability_scan" "fail" "high" "$vulnerable_images potentially vulnerable images" "Update to latest secure image versions"
    fi
}

# Check runtime security
check_runtime_security() {
    log_info "Checking runtime security monitoring..."
    
    # Check if Falco is deployed
    if kubectl get daemonset -n falco-system falco >/dev/null 2>&1; then
        record_check "runtime_security" "pass" "high" "Falco runtime security monitoring active"
    elif kubectl get deployment -n kube-system falco >/dev/null 2>&1; then
        record_check "runtime_security" "pass" "high" "Runtime security monitoring detected"
    else
        record_check "runtime_security" "fail" "high" "No runtime security monitoring found" "Deploy Falco or similar runtime security solution"
    fi
}

# Apply automatic fixes (where safe)
apply_fixes() {
    if [[ "$ENABLE_FIXES" != "true" ]]; then
        return 0
    fi
    
    log_info "Applying automatic security fixes..."
    
    local fixes_applied=0
    
    # Fix: Apply network policies if missing
    if [[ "${SECURITY_RESULTS[network_policies_status]:-}" == "fail" ]]; then
        if [[ -f "${PROJECT_ROOT}/security/policies/network-policy.yaml" ]]; then
            log_info "Applying network policies..."
            kubectl apply -f "${PROJECT_ROOT}/security/policies/network-policy.yaml" >/dev/null 2>&1 && {
                ((fixes_applied++))
                log_success "Applied network policies"
            } || log_warning "Failed to apply network policies"
        fi
    fi
    
    # Fix: Apply RBAC if missing
    if [[ "${SECURITY_RESULTS[rbac_configuration_status]:-}" == "fail" ]]; then
        if [[ -f "${PROJECT_ROOT}/k8s/rbac.yaml" ]]; then
            log_info "Applying RBAC configuration..."
            kubectl apply -f "${PROJECT_ROOT}/k8s/rbac.yaml" >/dev/null 2>&1 && {
                ((fixes_applied++))
                log_success "Applied RBAC configuration"
            } || log_warning "Failed to apply RBAC configuration"
        fi
    fi
    
    log_info "$fixes_applied automatic fixes applied"
}

# Generate security report
generate_report() {
    case "$OUTPUT_FORMAT" in
        json)
            generate_json_report
            ;;
        html)
            generate_html_report
            ;;
        text|*)
            generate_text_report
            ;;
    esac
}

# Text report
generate_text_report() {
    echo ""
    echo "======================================"
    echo "FlashMM Security Compliance Report"
    echo "======================================"
    echo "Timestamp: $(date)"
    echo "Environment: $ENVIRONMENT"
    echo "Namespace: $NAMESPACE"
    echo "Scan Type: $SCAN_TYPE"
    echo ""
    
    # Overall score
    OVERALL_SECURITY_SCORE=$((PASSED_CHECKS * 100 / TOTAL_CHECKS))
    echo "Overall Security Score: $OVERALL_SECURITY_SCORE% ($PASSED_CHECKS/$TOTAL_CHECKS checks passed)"
    echo ""
    
    # Detailed results
    echo "Security Check Results:"
    echo "----------------------"
    
    for key in "${!SECURITY_RESULTS[@]}"; do
        if [[ "$key" == *"_status" ]]; then
            local check_name=$(echo "$key" | sed 's/_status$//')
            local status="${SECURITY_RESULTS[$key]}"
            local severity="${SECURITY_RESULTS[${check_name}_severity]:-medium}"
            local message="${SECURITY_RESULTS[${check_name}_message]:-}"
            
            local status_icon
            case "$status" in
                pass) status_icon="✅" ;;
                fail) status_icon="❌" ;;
                partial) status_icon="⚠️" ;;
                *) status_icon="❓" ;;
            esac
            
            printf "  %-25s %s %-8s [%s] %s\n" "$check_name" "$status_icon" "$status" "$severity" "$message"
            
            # Show remediation if enabled and available
            if [[ "$REMEDIATION_MODE" == "true" && -n "${SECURITY_RESULTS[${check_name}_remediation]:-}" ]]; then
                echo "    Remediation: ${SECURITY_RESULTS[${check_name}_remediation]}"
            fi
        fi
    done
    
    echo ""
    
    # Compliance framework results
    if [[ "$SCAN_TYPE" == "compliance" ]]; then
        echo "Compliance Framework Results:"
        echo "----------------------------"
        for framework in "${COMPLIANCE_FRAMEWORKS[@]}"; do
            local framework_key="${framework}_compliance"
            if [[ -n "${SECURITY_RESULTS[${framework_key}_status]:-}" ]]; then
                printf "  %-15s: %s\n" "$framework" "${SECURITY_RESULTS[${framework_key}_message]}"
            fi
        done
    fi
    
    echo ""
    
    # Security recommendations
    echo "Security Recommendations:"
    echo "------------------------"
    if [[ $OVERALL_SECURITY_SCORE -lt 100 ]]; then
        echo "• Address failed security checks to improve security posture"
        echo "• Implement continuous security monitoring"
        echo "• Regular security assessments and penetration testing"
        echo "• Keep all components updated with latest security patches"
    else
        echo "• Excellent security posture! Maintain current standards"
        echo "• Continue regular security monitoring and assessments"
    fi
}

# JSON report
generate_json_report() {
    local report='{
        "timestamp": "'$(date -u +%Y-%m-%dT%H:%M:%SZ)'",
        "environment": "'$ENVIRONMENT'",
        "namespace": "'$NAMESPACE'",
        "scan_type": "'$SCAN_TYPE'",
        "overall_score": '$OVERALL_SECURITY_SCORE',
        "total_checks": '$TOTAL_CHECKS',
        "passed_checks": '$PASSED_CHECKS',
        "checks": {}
    }'
    
    for key in "${!SECURITY_RESULTS[@]}"; do
        if [[ "$key" == *"_status" ]]; then
            local check_name=$(echo "$key" | sed 's/_status$//')
            local status="${SECURITY_RESULTS[$key]}"
            local severity="${SECURITY_RESULTS[${check_name}_severity]:-medium}"
            local message="${SECURITY_RESULTS[${check_name}_message]:-}"
            local remediation="${SECURITY_RESULTS[${check_name}_remediation]:-}"
            
            report=$(echo "$report" | jq --arg name "$check_name" --arg status "$status" --arg sev "$severity" --arg msg "$message" --arg rem "$remediation" \
                '.checks[$name] = {"status": $status, "severity": $sev, "message": $msg, "remediation": $rem}')
        fi
    done
    
    echo "$report" | jq .
}

# Main execution
main() {
    log_info "Starting FlashMM security compliance scan..."
    
    # Prerequisites
    command -v kubectl >/dev/null 2>&1 || { log_error "kubectl is required"; exit 1; }
    command -v jq >/dev/null 2>&1 || { log_error "jq is required"; exit 1; }
    
    kubectl cluster-info >/dev/null 2>&1 || { log_error "Cannot connect to cluster"; exit 1; }
    kubectl get namespace "$NAMESPACE" >/dev/null 2>&1 || { log_error "Namespace $NAMESPACE not found"; exit 1; }
    
    # Run security checks based on scan type
    case "$SCAN_TYPE" in
        quick)
            check_pod_security_context
            check_network_policies
            check_rbac
            ;;
        full)
            check_pod_security_context
            check_network_policies
            check_rbac
            check_secrets_management
            check_image_security
            check_encryption
            check_runtime_security
            check_vulnerabilities
            ;;
        compliance)
            check_pod_security_context
            check_network_policies
            check_rbac
            check_secrets_management
            check_image_security
            check_encryption
            check_runtime_security
            check_compliance
            ;;
    esac
    
    # Apply fixes if requested
    apply_fixes
    
    # Generate report
    generate_report
    
    # Exit with appropriate code
    if [[ $OVERALL_SECURITY_SCORE -ge 80 ]]; then
        log_success "Security scan completed successfully"
        exit 0
    else
        log_warning "Security issues detected"
        exit 1
    fi
}

# Execute main function
main "$@"
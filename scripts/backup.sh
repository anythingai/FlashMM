#!/bin/bash
# FlashMM Backup Script
# Comprehensive backup solution for all FlashMM components

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Configuration
ENVIRONMENT="production"
NAMESPACE="flashmm"
BACKUP_TYPE="full"
RETENTION_DAYS=30
PARALLEL_BACKUPS=3

# Backup destinations
LOCAL_BACKUP_DIR="${PROJECT_ROOT}/backups"
S3_BUCKET="${BACKUP_S3_BUCKET:-flashmm-backups-prod}"
AWS_REGION="${AWS_DEFAULT_REGION:-us-east-1}"

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
FlashMM Backup Script

Usage: $0 [OPTIONS]

Options:
    -e, --environment ENV    Environment to backup (default: production)
    -n, --namespace NS       Kubernetes namespace (default: flashmm)
    -t, --type TYPE         Backup type: full, config, data (default: full)
    -r, --retention DAYS    Retention period in days (default: 30)
    -s, --s3-bucket BUCKET  S3 bucket for remote backup
    --local-only            Skip S3 upload
    --verify                Verify backup integrity
    -h, --help              Show this help message

Backup Types:
    full        Complete backup including data, config, and manifests
    config      Configuration and Kubernetes manifests only
    data        Database and persistent data only

Examples:
    $0                      # Full backup with default settings
    $0 -t data              # Data-only backup
    $0 --local-only         # Backup locally without S3 upload
    $0 --verify             # Verify backup after creation

EOF
}

# Parse arguments
LOCAL_ONLY=false
VERIFY_BACKUP=false

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
            BACKUP_TYPE="$2"
            shift 2
            ;;
        -r|--retention)
            RETENTION_DAYS="$2"
            shift 2
            ;;
        -s|--s3-bucket)
            S3_BUCKET="$2"
            shift 2
            ;;
        --local-only)
            LOCAL_ONLY=true
            shift
            ;;
        --verify)
            VERIFY_BACKUP=true
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

# Validate backup type
case "$BACKUP_TYPE" in
    full|config|data)
        ;;
    *)
        log_error "Invalid backup type: $BACKUP_TYPE"
        show_help
        exit 1
        ;;
esac

# Initialize backup
BACKUP_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="${LOCAL_BACKUP_DIR}/${ENVIRONMENT}/${BACKUP_TIMESTAMP}"
BACKUP_MANIFEST="${BACKUP_DIR}/backup_manifest.json"

mkdir -p "$BACKUP_DIR"

# Create backup manifest
create_backup_manifest() {
    log_info "Creating backup manifest..."
    
    cat > "$BACKUP_MANIFEST" << EOF
{
  "backup_info": {
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "environment": "$ENVIRONMENT",
    "namespace": "$NAMESPACE",
    "type": "$BACKUP_TYPE",
    "retention_days": $RETENTION_DAYS,
    "created_by": "$(whoami)",
    "hostname": "$(hostname)",
    "version": "1.0"
  },
  "components": {},
  "files": [],
  "checksums": {}
}
EOF
    
    log_success "Backup manifest created: $BACKUP_MANIFEST"
}

# Update manifest with component info
update_manifest() {
    local component="$1"
    local status="$2"
    local file_path="${3:-}"
    local checksum="${4:-}"
    
    local temp_file=$(mktemp)
    jq --arg comp "$component" --arg status "$status" --arg file "$file_path" --arg sum "$checksum" \
        '.components[$comp] = {"status": $status, "file": $file, "checksum": $sum}' \
        "$BACKUP_MANIFEST" > "$temp_file" && mv "$temp_file" "$BACKUP_MANIFEST"
}

# Backup Kubernetes configurations
backup_kubernetes_config() {
    if [[ "$BACKUP_TYPE" != "data" ]]; then
        log_info "Backing up Kubernetes configurations..."
        
        local k8s_dir="${BACKUP_DIR}/kubernetes"
        mkdir -p "$k8s_dir"
        
        # Backup all resources in namespace
        kubectl get all -n "$NAMESPACE" -o yaml > "${k8s_dir}/all-resources.yaml"
        kubectl get configmaps -n "$NAMESPACE" -o yaml > "${k8s_dir}/configmaps.yaml"
        kubectl get secrets -n "$NAMESPACE" -o yaml > "${k8s_dir}/secrets.yaml"
        kubectl get pvc -n "$NAMESPACE" -o yaml > "${k8s_dir}/persistent-volumes.yaml"
        kubectl get ingress -n "$NAMESPACE" -o yaml > "${k8s_dir}/ingress.yaml" 2>/dev/null || true
        
        # Backup Helm releases
        if helm list -n "$NAMESPACE" | grep -q flashmm; then
            helm get all flashmm -n "$NAMESPACE" > "${k8s_dir}/helm-release.yaml"
            helm get values flashmm -n "$NAMESPACE" > "${k8s_dir}/helm-values.yaml"
        fi
        
        # Create archive
        local k8s_archive="${BACKUP_DIR}/kubernetes-config.tar.gz"
        tar -czf "$k8s_archive" -C "$BACKUP_DIR" kubernetes/
        local checksum=$(sha256sum "$k8s_archive" | cut -d' ' -f1)
        
        update_manifest "kubernetes_config" "success" "kubernetes-config.tar.gz" "$checksum"
        log_success "Kubernetes configuration backup completed"
    fi
}

# Backup PostgreSQL database
backup_postgresql() {
    if [[ "$BACKUP_TYPE" != "config" ]]; then
        log_info "Backing up PostgreSQL database..."
        
        local db_dir="${BACKUP_DIR}/database"
        mkdir -p "$db_dir"
        
        # Check if PostgreSQL deployment exists
        if kubectl get deployment -n "$NAMESPACE" postgres >/dev/null 2>&1; then
            # Create database dump
            local dump_file="${db_dir}/postgresql_dump.sql"
            if kubectl exec -n "$NAMESPACE" deployment/postgres -- \
                pg_dump -U flashmm flashmm_prod > "$dump_file"; then
                
                # Compress dump
                gzip "$dump_file"
                local compressed_file="${dump_file}.gz"
                local checksum=$(sha256sum "$compressed_file" | cut -d' ' -f1)
                
                update_manifest "postgresql" "success" "database/postgresql_dump.sql.gz" "$checksum"
                log_success "PostgreSQL backup completed"
            else
                update_manifest "postgresql" "failed" "" ""
                log_error "PostgreSQL backup failed"
            fi
        else
            update_manifest "postgresql" "not_found" "" ""
            log_warning "PostgreSQL deployment not found"
        fi
    fi
}

# Backup Redis data
backup_redis() {
    if [[ "$BACKUP_TYPE" != "config" ]]; then
        log_info "Backing up Redis data..."
        
        local redis_dir="${BACKUP_DIR}/redis"
        mkdir -p "$redis_dir"
        
        # Check if Redis deployment exists
        if kubectl get deployment -n "$NAMESPACE" redis >/dev/null 2>&1; then
            # Create Redis backup
            local backup_file="${redis_dir}/redis_backup.rdb"
            if kubectl exec -n "$NAMESPACE" deployment/redis -- \
                redis-cli --rdb "$backup_file" >/dev/null 2>&1; then
                
                # Copy backup file from container
                kubectl cp "${NAMESPACE}/$(kubectl get pods -n "$NAMESPACE" -l app=redis -o jsonpath='{.items[0].metadata.name}'):${backup_file}" \
                    "$backup_file" >/dev/null 2>&1 || log_warning "Could not copy Redis backup file"
                
                if [[ -f "$backup_file" ]]; then
                    gzip "$backup_file"
                    local compressed_file="${backup_file}.gz"
                    local checksum=$(sha256sum "$compressed_file" | cut -d' ' -f1)
                    
                    update_manifest "redis" "success" "redis/redis_backup.rdb.gz" "$checksum"
                    log_success "Redis backup completed"
                else
                    update_manifest "redis" "partial" "" ""
                    log_warning "Redis backup partially completed"
                fi
            else
                update_manifest "redis" "failed" "" ""
                log_error "Redis backup failed"
            fi
        else
            update_manifest "redis" "not_found" "" ""
            log_warning "Redis deployment not found"
        fi
    fi
}

# Backup InfluxDB data
backup_influxdb() {
    if [[ "$BACKUP_TYPE" != "config" ]]; then
        log_info "Backing up InfluxDB data..."
        
        local influx_dir="${BACKUP_DIR}/influxdb"
        mkdir -p "$influx_dir"
        
        # Check if InfluxDB deployment exists
        if kubectl get deployment -n "$NAMESPACE" influxdb >/dev/null 2>&1; then
            # Create InfluxDB backup
            local backup_file="${influx_dir}/influxdb_backup.tar.gz"
            if kubectl exec -n "$NAMESPACE" deployment/influxdb -- \
                influx backup /tmp/backup >/dev/null 2>&1; then
                
                # Create archive and copy
                kubectl exec -n "$NAMESPACE" deployment/influxdb -- \
                    tar -czf /tmp/influx_backup.tar.gz -C /tmp backup/ >/dev/null 2>&1
                
                kubectl cp "${NAMESPACE}/$(kubectl get pods -n "$NAMESPACE" -l app=influxdb -o jsonpath='{.items[0].metadata.name}'):/tmp/influx_backup.tar.gz" \
                    "$backup_file" >/dev/null 2>&1 || log_warning "Could not copy InfluxDB backup file"
                
                if [[ -f "$backup_file" ]]; then
                    local checksum=$(sha256sum "$backup_file" | cut -d' ' -f1)
                    update_manifest "influxdb" "success" "influxdb/influxdb_backup.tar.gz" "$checksum"
                    log_success "InfluxDB backup completed"
                else
                    update_manifest "influxdb" "partial" "" ""
                    log_warning "InfluxDB backup partially completed"
                fi
            else
                update_manifest "influxdb" "failed" "" ""
                log_error "InfluxDB backup failed"
            fi
        else
            update_manifest "influxdb" "not_found" "" ""
            log_warning "InfluxDB deployment not found"
        fi
    fi
}

# Backup persistent volumes
backup_persistent_volumes() {
    if [[ "$BACKUP_TYPE" != "config" ]]; then
        log_info "Backing up persistent volume data..."
        
        local pv_dir="${BACKUP_DIR}/persistent-volumes"
        mkdir -p "$pv_dir"
        
        # Get list of PVCs
        local pvcs=$(kubectl get pvc -n "$NAMESPACE" -o jsonpath='{.items[*].metadata.name}' 2>/dev/null || echo "")
        
        if [[ -n "$pvcs" ]]; then
            for pvc in $pvcs; do
                log_info "Backing up PVC: $pvc"
                
                # Create a job to backup the PVC
                local job_name="backup-${pvc}-${BACKUP_TIMESTAMP}"
                local backup_file="${pv_dir}/${pvc}.tar.gz"
                
                # Create backup job manifest
                cat > "/tmp/${job_name}.yaml" << EOF
apiVersion: batch/v1
kind: Job
metadata:
  name: ${job_name}
  namespace: ${NAMESPACE}
spec:
  template:
    spec:
      containers:
      - name: backup
        image: alpine:latest
        command: ["/bin/sh"]
        args: ["-c", "tar -czf /backup/${pvc}.tar.gz -C /data . && sleep 10"]
        volumeMounts:
        - name: data
          mountPath: /data
        - name: backup-storage
          mountPath: /backup
      volumes:
      - name: data
        persistentVolumeClaim:
          claimName: ${pvc}
      - name: backup-storage
        emptyDir: {}
      restartPolicy: Never
  backoffLimit: 1
EOF
                
                # Run backup job (simplified - in production you'd want more robust handling)
                kubectl apply -f "/tmp/${job_name}.yaml" >/dev/null 2>&1 || log_warning "Could not create backup job for $pvc"
                rm -f "/tmp/${job_name}.yaml"
            done
            
            update_manifest "persistent_volumes" "partial" "persistent-volumes/" ""
            log_success "Persistent volume backup initiated"
        else
            update_manifest "persistent_volumes" "not_found" "" ""
            log_info "No persistent volumes found"
        fi
    fi
}

# Backup monitoring data
backup_monitoring() {
    if [[ "$BACKUP_TYPE" == "full" ]]; then
        log_info "Backing up monitoring configurations..."
        
        local monitor_dir="${BACKUP_DIR}/monitoring"
        mkdir -p "$monitor_dir"
        
        # Copy monitoring configurations
        if [[ -d "${PROJECT_ROOT}/monitoring" ]]; then
            cp -r "${PROJECT_ROOT}/monitoring"/* "$monitor_dir/"
            
            # Create archive
            local monitor_archive="${BACKUP_DIR}/monitoring-config.tar.gz"
            tar -czf "$monitor_archive" -C "$BACKUP_DIR" monitoring/
            local checksum=$(sha256sum "$monitor_archive" | cut -d' ' -f1)
            
            update_manifest "monitoring_config" "success" "monitoring-config.tar.gz" "$checksum"
            log_success "Monitoring configuration backup completed"
        else
            update_manifest "monitoring_config" "not_found" "" ""
            log_warning "Monitoring configuration directory not found"
        fi
    fi
}

# Create final backup archive
create_backup_archive() {
    log_info "Creating final backup archive..."
    
    local final_archive="${LOCAL_BACKUP_DIR}/${ENVIRONMENT}_${BACKUP_TYPE}_${BACKUP_TIMESTAMP}.tar.gz"
    tar -czf "$final_archive" -C "${LOCAL_BACKUP_DIR}/${ENVIRONMENT}" "${BACKUP_TIMESTAMP}/"
    
    local archive_checksum=$(sha256sum "$final_archive" | cut -d' ' -f1)
    local archive_size=$(du -h "$final_archive" | cut -f1)
    
    # Update manifest with final info
    local temp_file=$(mktemp)
    jq --arg archive "$(basename "$final_archive")" --arg checksum "$archive_checksum" --arg size "$archive_size" \
        '.backup_info.final_archive = $archive | .backup_info.checksum = $checksum | .backup_info.size = $size' \
        "$BACKUP_MANIFEST" > "$temp_file" && mv "$temp_file" "$BACKUP_MANIFEST"
    
    log_success "Final backup archive created: $final_archive ($archive_size)"
    echo "$final_archive"
}

# Upload to S3
upload_to_s3() {
    if [[ "$LOCAL_ONLY" == "true" ]]; then
        log_info "Skipping S3 upload (local-only mode)"
        return 0
    fi
    
    log_info "Uploading backup to S3..."
    
    local final_archive="$1"
    local s3_key="${ENVIRONMENT}/$(basename "$final_archive")"
    
    if command -v aws >/dev/null 2>&1; then
        if aws s3 cp "$final_archive" "s3://${S3_BUCKET}/${s3_key}" --region "$AWS_REGION"; then
            log_success "Backup uploaded to S3: s3://${S3_BUCKET}/${s3_key}"
            
            # Upload manifest as well
            aws s3 cp "$BACKUP_MANIFEST" "s3://${S3_BUCKET}/${ENVIRONMENT}/manifests/$(basename "$BACKUP_MANIFEST")" --region "$AWS_REGION"
        else
            log_error "Failed to upload backup to S3"
            return 1
        fi
    else
        log_warning "AWS CLI not available, skipping S3 upload"
    fi
}

# Verify backup integrity
verify_backup() {
    if [[ "$VERIFY_BACKUP" != "true" ]]; then
        log_info "Skipping backup verification"
        return 0
    fi
    
    log_info "Verifying backup integrity..."
    
    local final_archive="$1"
    
    # Verify archive can be extracted
    if tar -tzf "$final_archive" >/dev/null 2>&1; then
        log_success "Backup archive integrity verified"
    else
        log_error "Backup archive is corrupted"
        return 1
    fi
    
    # Verify manifest
    if jq empty "$BACKUP_MANIFEST" 2>/dev/null; then
        log_success "Backup manifest is valid JSON"
    else
        log_error "Backup manifest is invalid"
        return 1
    fi
    
    # Verify checksums
    local temp_dir=$(mktemp -d)
    tar -xzf "$final_archive" -C "$temp_dir"
    
    local manifest_backup_dir=$(find "$temp_dir" -name "backup_manifest.json" -exec dirname {} \;)
    if [[ -n "$manifest_backup_dir" ]]; then
        # Check individual file checksums from manifest
        local checksum_errors=0
        while IFS=$'\t' read -r file expected_checksum; do
            if [[ -n "$file" && -n "$expected_checksum" ]]; then
                local actual_checksum=$(sha256sum "${manifest_backup_dir}/${file}" 2>/dev/null | cut -d' ' -f1)
                if [[ "$actual_checksum" != "$expected_checksum" ]]; then
                    log_error "Checksum mismatch for $file"
                    ((checksum_errors++))
                fi
            fi
        done < <(jq -r '.components[] | select(.checksum != "") | "\(.file)\t\(.checksum)"' "$BACKUP_MANIFEST")
        
        if [[ $checksum_errors -eq 0 ]]; then
            log_success "All file checksums verified"
        else
            log_error "$checksum_errors checksum verification failures"
        fi
    fi
    
    rm -rf "$temp_dir"
}

# Clean old backups
cleanup_old_backups() {
    log_info "Cleaning up old backups (retention: ${RETENTION_DAYS} days)..."
    
    # Clean local backups
    find "${LOCAL_BACKUP_DIR}/${ENVIRONMENT}" -name "*${BACKUP_TYPE}*" -type f -mtime +$RETENTION_DAYS -delete 2>/dev/null || true
    find "${LOCAL_BACKUP_DIR}/${ENVIRONMENT}" -type d -empty -delete 2>/dev/null || true
    
    # Clean S3 backups
    if [[ "$LOCAL_ONLY" != "true" ]] && command -v aws >/dev/null 2>&1; then
        local cutoff_date=$(date -d "${RETENTION_DAYS} days ago" +%Y-%m-%d)
        aws s3 ls "s3://${S3_BUCKET}/${ENVIRONMENT}/" --region "$AWS_REGION" | \
            awk -v cutoff="$cutoff_date" '$1 < cutoff {print $4}' | \
            while read -r file; do
                if [[ -n "$file" ]]; then
                    aws s3 rm "s3://${S3_BUCKET}/${ENVIRONMENT}/${file}" --region "$AWS_REGION" >/dev/null 2>&1 || true
                fi
            done
    fi
    
    log_success "Old backup cleanup completed"
}

# Main execution
main() {
    log_info "Starting FlashMM backup process..."
    log_info "Environment: $ENVIRONMENT"
    log_info "Namespace: $NAMESPACE"
    log_info "Backup type: $BACKUP_TYPE"
    log_info "Backup directory: $BACKUP_DIR"
    
    # Prerequisites check
    command -v kubectl >/dev/null 2>&1 || { log_error "kubectl is required"; exit 1; }
    command -v jq >/dev/null 2>&1 || { log_error "jq is required"; exit 1; }
    
    kubectl cluster-info >/dev/null 2>&1 || { log_error "Cannot connect to cluster"; exit 1; }
    kubectl get namespace "$NAMESPACE" >/dev/null 2>&1 || { log_error "Namespace $NAMESPACE not found"; exit 1; }
    
    # Initialize backup
    create_backup_manifest
    
    # Perform backups based on type
    case "$BACKUP_TYPE" in
        full)
            backup_kubernetes_config &
            backup_postgresql &
            backup_redis &
            backup_influxdb &
            backup_persistent_volumes &
            backup_monitoring &
            wait
            ;;
        config)
            backup_kubernetes_config
            backup_monitoring
            ;;
        data)
            backup_postgresql &
            backup_redis &
            backup_influxdb &
            backup_persistent_volumes &
            wait
            ;;
    esac
    
    # Create final archive
    local final_archive
    final_archive=$(create_backup_archive)
    
    # Upload to S3
    upload_to_s3 "$final_archive"
    
    # Verify backup
    verify_backup "$final_archive"
    
    # Cleanup old backups
    cleanup_old_backups
    
    # Show summary
    log_info "Backup Summary:"
    echo "  Archive: $final_archive"
    echo "  Size: $(du -h "$final_archive" | cut -f1)"
    echo "  Components:"
    jq -r '.components | to_entries[] | "    \(.key): \(.value.status)"' "$BACKUP_MANIFEST"
    
    log_success "🗄️ FlashMM backup completed successfully!"
}

# Execute main function
main "$@"
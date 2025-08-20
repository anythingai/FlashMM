# FlashMM Terraform Outputs
# Export important infrastructure values

# =============================================================================
# Cloud Provider Outputs
# =============================================================================

output "cloud_provider" {
  description = "The cloud provider used for deployment"
  value       = var.cloud_provider
}

output "environment" {
  description = "The deployment environment"
  value       = var.environment
}

output "region" {
  description = "The deployment region"
  value = var.cloud_provider == "aws" ? var.aws_region : (
    var.cloud_provider == "gcp" ? var.gcp_region : var.azure_location
  )
}

# =============================================================================
# Kubernetes Cluster Outputs
# =============================================================================

output "cluster_name" {
  description = "Kubernetes cluster name"
  value = var.cloud_provider == "aws" ? (
    length(module.aws_infrastructure) > 0 ? module.aws_infrastructure[0].cluster_name : ""
  ) : var.cloud_provider == "gcp" ? (
    length(module.gcp_infrastructure) > 0 ? module.gcp_infrastructure[0].cluster_name : ""
  ) : var.cloud_provider == "azure" ? (
    length(module.azure_infrastructure) > 0 ? module.azure_infrastructure[0].cluster_name : ""
  ) : ""
}

output "cluster_endpoint" {
  description = "Kubernetes cluster endpoint"
  value = var.cloud_provider == "aws" ? (
    length(module.aws_infrastructure) > 0 ? module.aws_infrastructure[0].cluster_endpoint : ""
  ) : var.cloud_provider == "gcp" ? (
    length(module.gcp_infrastructure) > 0 ? module.gcp_infrastructure[0].cluster_endpoint : ""
  ) : var.cloud_provider == "azure" ? (
    length(module.azure_infrastructure) > 0 ? module.azure_infrastructure[0].cluster_endpoint : ""
  ) : ""
  sensitive = true
}

output "cluster_version" {
  description = "Kubernetes cluster version"
  value = var.cloud_provider == "aws" ? (
    length(module.aws_infrastructure) > 0 ? module.aws_infrastructure[0].cluster_version : ""
  ) : var.cloud_provider == "gcp" ? (
    length(module.gcp_infrastructure) > 0 ? module.gcp_infrastructure[0].cluster_version : ""
  ) : var.cloud_provider == "azure" ? (
    length(module.azure_infrastructure) > 0 ? module.azure_infrastructure[0].cluster_version : ""
  ) : ""
}

# =============================================================================
# Network Outputs
# =============================================================================

output "vpc_id" {
  description = "VPC ID"
  value = var.cloud_provider == "aws" ? (
    length(module.aws_infrastructure) > 0 ? module.aws_infrastructure[0].vpc_id : ""
  ) : var.cloud_provider == "gcp" ? (
    length(module.gcp_infrastructure) > 0 ? module.gcp_infrastructure[0].vpc_name : ""
  ) : var.cloud_provider == "azure" ? (
    length(module.azure_infrastructure) > 0 ? module.azure_infrastructure[0].vnet_id : ""
  ) : ""
}

output "public_subnet_ids" {
  description = "Public subnet IDs"
  value = var.cloud_provider == "aws" ? (
    length(module.aws_infrastructure) > 0 ? module.aws_infrastructure[0].public_subnet_ids : []
  ) : var.cloud_provider == "gcp" ? (
    length(module.gcp_infrastructure) > 0 ? module.gcp_infrastructure[0].public_subnet_names : []
  ) : var.cloud_provider == "azure" ? (
    length(module.azure_infrastructure) > 0 ? module.azure_infrastructure[0].public_subnet_ids : []
  ) : []
}

output "private_subnet_ids" {
  description = "Private subnet IDs"
  value = var.cloud_provider == "aws" ? (
    length(module.aws_infrastructure) > 0 ? module.aws_infrastructure[0].private_subnet_ids : []
  ) : var.cloud_provider == "gcp" ? (
    length(module.gcp_infrastructure) > 0 ? module.gcp_infrastructure[0].private_subnet_names : []
  ) : var.cloud_provider == "azure" ? (
    length(module.azure_infrastructure) > 0 ? module.azure_infrastructure[0].private_subnet_ids : []
  ) : []
}

# =============================================================================
# Database Outputs
# =============================================================================

output "database_endpoint" {
  description = "Database endpoint"
  value = var.cloud_provider == "aws" ? (
    length(module.aws_infrastructure) > 0 ? module.aws_infrastructure[0].rds_endpoint : ""
  ) : var.cloud_provider == "gcp" ? (
    length(module.gcp_infrastructure) > 0 ? module.gcp_infrastructure[0].database_connection_name : ""
  ) : var.cloud_provider == "azure" ? (
    length(module.azure_infrastructure) > 0 ? module.azure_infrastructure[0].postgresql_fqdn : ""
  ) : ""
  sensitive = true
}

output "database_port" {
  description = "Database port"
  value = var.cloud_provider == "aws" ? (
    length(module.aws_infrastructure) > 0 ? module.aws_infrastructure[0].rds_port : 0
  ) : var.cloud_provider == "gcp" ? (
    length(module.gcp_infrastructure) > 0 ? 5432 : 0
  ) : var.cloud_provider == "azure" ? (
    length(module.azure_infrastructure) > 0 ? 5432 : 0
  ) : 0
}

# =============================================================================
# Cache Outputs
# =============================================================================

output "redis_endpoint" {
  description = "Redis endpoint"
  value = var.cloud_provider == "aws" ? (
    length(module.aws_infrastructure) > 0 ? module.aws_infrastructure[0].redis_endpoint : ""
  ) : var.cloud_provider == "gcp" ? (
    length(module.gcp_infrastructure) > 0 ? module.gcp_infrastructure[0].redis_host : ""
  ) : var.cloud_provider == "azure" ? (
    length(module.azure_infrastructure) > 0 ? module.azure_infrastructure[0].redis_hostname : ""
  ) : ""
  sensitive = true
}

output "redis_port" {
  description = "Redis port"
  value = var.cloud_provider == "aws" ? (
    length(module.aws_infrastructure) > 0 ? module.aws_infrastructure[0].redis_port : 0
  ) : var.cloud_provider == "gcp" ? (
    length(module.gcp_infrastructure) > 0 ? 6379 : 0
  ) : var.cloud_provider == "azure" ? (
    length(module.azure_infrastructure) > 0 ? 6380 : 0
  ) : 0
}

# =============================================================================
# Load Balancer Outputs
# =============================================================================

output "load_balancer_dns" {
  description = "Load balancer DNS name"
  value = var.cloud_provider == "aws" ? (
    length(module.aws_infrastructure) > 0 ? module.aws_infrastructure[0].load_balancer_dns : ""
  ) : var.cloud_provider == "gcp" ? (
    length(module.gcp_infrastructure) > 0 ? module.gcp_infrastructure[0].load_balancer_ip : ""
  ) : var.cloud_provider == "azure" ? (
    length(module.azure_infrastructure) > 0 ? module.azure_infrastructure[0].load_balancer_ip : ""
  ) : ""
}

# =============================================================================
# Application Outputs
# =============================================================================

output "flashmm_namespace" {
  description = "FlashMM Kubernetes namespace"
  value       = module.flashmm_application.namespace
}

output "flashmm_service_urls" {
  description = "FlashMM service URLs"
  value       = module.flashmm_application.service_urls
}

output "flashmm_ingress_urls" {
  description = "FlashMM ingress URLs"
  value       = module.flashmm_application.ingress_urls
}

# =============================================================================
# Monitoring Outputs
# =============================================================================

output "monitoring_urls" {
  description = "Monitoring service URLs"
  value = var.monitoring_enabled ? {
    prometheus = length(module.monitoring) > 0 ? module.monitoring[0].prometheus_url : ""
    grafana    = length(module.monitoring) > 0 ? module.monitoring[0].grafana_url : ""
    alertmanager = length(module.monitoring) > 0 ? module.monitoring[0].alertmanager_url : ""
  } : {}
}

output "logging_urls" {
  description = "Logging service URLs"
  value = var.logging_enabled ? {
    kibana = length(module.monitoring) > 0 ? module.monitoring[0].kibana_url : ""
    elasticsearch = length(module.monitoring) > 0 ? module.monitoring[0].elasticsearch_url : ""
  } : {}
}

# =============================================================================
# Security Outputs
# =============================================================================

output "cluster_security_group_id" {
  description = "Cluster security group ID"
  value = var.cloud_provider == "aws" ? (
    length(module.aws_infrastructure) > 0 ? module.aws_infrastructure[0].cluster_security_group_id : ""
  ) : ""
}

output "node_security_group_id" {
  description = "Node security group ID"
  value = var.cloud_provider == "aws" ? (
    length(module.aws_infrastructure) > 0 ? module.aws_infrastructure[0].node_security_group_id : ""
  ) : ""
}

# =============================================================================
# Backup Outputs
# =============================================================================

output "backup_bucket" {
  description = "Backup storage bucket"
  value = var.backup_enabled ? (
    length(module.backup) > 0 ? module.backup[0].backup_bucket : ""
  ) : ""
}

output "backup_schedule" {
  description = "Backup schedule"
  value       = var.backup_enabled ? var.backup_schedule : ""
}

# =============================================================================
# Cost and Resource Outputs
# =============================================================================

output "estimated_monthly_cost" {
  description = "Estimated monthly cost (USD)"
  value = var.cloud_provider == "aws" ? (
    length(module.aws_infrastructure) > 0 ? module.aws_infrastructure[0].estimated_monthly_cost : 0
  ) : var.cloud_provider == "gcp" ? (
    length(module.gcp_infrastructure) > 0 ? module.gcp_infrastructure[0].estimated_monthly_cost : 0
  ) : var.cloud_provider == "azure" ? (
    length(module.azure_infrastructure) > 0 ? module.azure_infrastructure[0].estimated_monthly_cost : 0
  ) : 0
}

output "resource_summary" {
  description = "Summary of deployed resources"
  value = {
    cluster_nodes = var.node_desired_size
    database_instances = 1
    redis_nodes = var.redis_num_cache_nodes
    storage_gb = var.rds_allocated_storage + tonumber(replace(var.prometheus_storage_size, "Gi", ""))
    replicas = var.flashmm_replica_count
    monitoring_enabled = var.monitoring_enabled
    backup_enabled = var.backup_enabled
  }
}

# =============================================================================
# Connection Information
# =============================================================================

output "kubectl_config_command" {
  description = "Command to configure kubectl"
  value = var.cloud_provider == "aws" ? (
    length(module.aws_infrastructure) > 0 ? 
    "aws eks update-kubeconfig --region ${var.aws_region} --name ${module.aws_infrastructure[0].cluster_name}" : ""
  ) : var.cloud_provider == "gcp" ? (
    length(module.gcp_infrastructure) > 0 ? 
    "gcloud container clusters get-credentials ${module.gcp_infrastructure[0].cluster_name} --region ${var.gcp_region}" : ""
  ) : var.cloud_provider == "azure" ? (
    length(module.azure_infrastructure) > 0 ? 
    "az aks get-credentials --resource-group ${module.azure_infrastructure[0].resource_group_name} --name ${module.azure_infrastructure[0].cluster_name}" : ""
  ) : ""
}

output "deployment_instructions" {
  description = "Instructions for accessing the deployed application"
  value = <<-EOT
    # FlashMM Deployment Information
    
    ## Cluster Access
    Configure kubectl: ${local.kubectl_config_command}
    
    ## Application URLs
    API: ${lookup(module.flashmm_application.ingress_urls, "api", "Not configured")}
    
    ## Monitoring URLs
    Grafana: ${var.monitoring_enabled ? lookup(module.monitoring[0].service_urls, "grafana", "Not enabled") : "Disabled"}
    Prometheus: ${var.monitoring_enabled ? lookup(module.monitoring[0].service_urls, "prometheus", "Not enabled") : "Disabled"}
    
    ## Database Connection
    Endpoint: ${local.database_endpoint}
    Port: ${local.database_port}
    
    ## Redis Connection
    Endpoint: ${local.redis_endpoint}
    Port: ${local.redis_port}
    
    ## Estimated Monthly Cost: $${local.estimated_monthly_cost}
    
    ## Next Steps
    1. Verify cluster connectivity: kubectl get nodes
    2. Check application status: kubectl get pods -n flashmm
    3. Access monitoring dashboard: ${var.monitoring_enabled ? "Open Grafana URL above" : "Enable monitoring first"}
    4. Review logs: kubectl logs -f deployment/flashmm-app -n flashmm
  EOT
}

# =============================================================================
# Local Values for Outputs
# =============================================================================

locals {
  kubectl_config_command = var.cloud_provider == "aws" ? (
    length(module.aws_infrastructure) > 0 ? 
    "aws eks update-kubeconfig --region ${var.aws_region} --name ${module.aws_infrastructure[0].cluster_name}" : ""
  ) : var.cloud_provider == "gcp" ? (
    length(module.gcp_infrastructure) > 0 ? 
    "gcloud container clusters get-credentials ${module.gcp_infrastructure[0].cluster_name} --region ${var.gcp_region}" : ""
  ) : var.cloud_provider == "azure" ? (
    length(module.azure_infrastructure) > 0 ? 
    "az aks get-credentials --resource-group ${module.azure_infrastructure[0].resource_group_name} --name ${module.azure_infrastructure[0].cluster_name}" : ""
  ) : ""
  
  database_endpoint = var.cloud_provider == "aws" ? (
    length(module.aws_infrastructure) > 0 ? module.aws_infrastructure[0].rds_endpoint : ""
  ) : var.cloud_provider == "gcp" ? (
    length(module.gcp_infrastructure) > 0 ? module.gcp_infrastructure[0].database_connection_name : ""
  ) : var.cloud_provider == "azure" ? (
    length(module.azure_infrastructure) > 0 ? module.azure_infrastructure[0].postgresql_fqdn : ""
  ) : ""
  
  database_port = var.cloud_provider == "aws" ? (
    length(module.aws_infrastructure) > 0 ? module.aws_infrastructure[0].rds_port : 0
  ) : 5432
  
  redis_endpoint = var.cloud_provider == "aws" ? (
    length(module.aws_infrastructure) > 0 ? module.aws_infrastructure[0].redis_endpoint : ""
  ) : var.cloud_provider == "gcp" ? (
    length(module.gcp_infrastructure) > 0 ? module.gcp_infrastructure[0].redis_host : ""
  ) : var.cloud_provider == "azure" ? (
    length(module.azure_infrastructure) > 0 ? module.azure_infrastructure[0].redis_hostname : ""
  ) : ""
  
  redis_port = var.cloud_provider == "aws" ? (
    length(module.aws_infrastructure) > 0 ? module.aws_infrastructure[0].redis_port : 0
  ) : var.cloud_provider == "gcp" ? 6379 : var.cloud_provider == "azure" ? 6380 : 0
  
  estimated_monthly_cost = var.cloud_provider == "aws" ? (
    length(module.aws_infrastructure) > 0 ? module.aws_infrastructure[0].estimated_monthly_cost : 0
  ) : var.cloud_provider == "gcp" ? (
    length(module.gcp_infrastructure) > 0 ? module.gcp_infrastructure[0].estimated_monthly_cost : 0
  ) : var.cloud_provider == "azure" ? (
    length(module.azure_infrastructure) > 0 ? module.azure_infrastructure[0].estimated_monthly_cost : 0
  ) : 0
}
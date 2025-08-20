# FlashMM Terraform Variables
# Input variables for infrastructure configuration

# =============================================================================
# Core Configuration
# =============================================================================

variable "project_name" {
  description = "Name of the project"
  type        = string
  default     = "flashmm"
  
  validation {
    condition     = can(regex("^[a-z0-9-]+$", var.project_name))
    error_message = "Project name must contain only lowercase letters, numbers, and hyphens."
  }
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  
  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be one of: dev, staging, prod."
  }
}

variable "cloud_provider" {
  description = "Cloud provider to use (aws, gcp, azure)"
  type        = string
  default     = "aws"
  
  validation {
    condition     = contains(["aws", "gcp", "azure"], var.cloud_provider)
    error_message = "Cloud provider must be one of: aws, gcp, azure."
  }
}

# =============================================================================
# Network Configuration
# =============================================================================

variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
  
  validation {
    condition     = can(cidrhost(var.vpc_cidr, 0))
    error_message = "VPC CIDR must be a valid IPv4 CIDR block."
  }
}

variable "public_subnet_cidrs" {
  description = "CIDR blocks for public subnets"
  type        = list(string)
  default     = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
  
  validation {
    condition     = length(var.public_subnet_cidrs) >= 2
    error_message = "At least 2 public subnets are required for high availability."
  }
}

variable "private_subnet_cidrs" {
  description = "CIDR blocks for private subnets"
  type        = list(string)
  default     = ["10.0.10.0/24", "10.0.11.0/24", "10.0.12.0/24"]
  
  validation {
    condition     = length(var.private_subnet_cidrs) >= 2
    error_message = "At least 2 private subnets are required for high availability."
  }
}

# =============================================================================
# AWS Configuration
# =============================================================================

variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "rds_instance_class" {
  description = "RDS instance class"
  type        = string
  default     = "db.r5.large"
}

variable "rds_allocated_storage" {
  description = "RDS allocated storage in GB"
  type        = number
  default     = 100
  
  validation {
    condition     = var.rds_allocated_storage >= 20
    error_message = "RDS allocated storage must be at least 20 GB."
  }
}

variable "redis_node_type" {
  description = "ElastiCache Redis node type"
  type        = string
  default     = "cache.r6g.large"
}

variable "redis_num_cache_nodes" {
  description = "Number of cache nodes in the Redis cluster"
  type        = number
  default     = 2
  
  validation {
    condition     = var.redis_num_cache_nodes >= 1 && var.redis_num_cache_nodes <= 20
    error_message = "Redis cache nodes must be between 1 and 20."
  }
}

# =============================================================================
# GCP Configuration
# =============================================================================

variable "gcp_project_id" {
  description = "GCP project ID"
  type        = string
  default     = ""
}

variable "gcp_region" {
  description = "GCP region"
  type        = string
  default     = "us-central1"
}

variable "gcp_zone" {
  description = "GCP zone"
  type        = string
  default     = "us-central1-a"
}

variable "gcp_node_machine_type" {
  description = "GKE node machine type"
  type        = string
  default     = "e2-standard-4"
}

variable "gcp_database_tier" {
  description = "Cloud SQL database tier"
  type        = string
  default     = "db-standard-2"
}

# =============================================================================
# Azure Configuration
# =============================================================================

variable "azure_subscription_id" {
  description = "Azure subscription ID"
  type        = string
  default     = ""
}

variable "azure_tenant_id" {
  description = "Azure tenant ID"
  type        = string
  default     = ""
}

variable "azure_location" {
  description = "Azure location"
  type        = string
  default     = "East US"
}

variable "azure_node_vm_size" {
  description = "AKS node VM size"
  type        = string
  default     = "Standard_D4s_v3"
}

variable "azure_postgresql_sku" {
  description = "Azure PostgreSQL SKU"
  type        = string
  default     = "GP_Standard_D4s_v3"
}

# =============================================================================
# Kubernetes Configuration
# =============================================================================

variable "k8s_version" {
  description = "Kubernetes version"
  type        = string
  default     = "1.28"
}

variable "node_instance_types" {
  description = "EC2 instance types for EKS nodes"
  type        = list(string)
  default     = ["t3.large", "t3.xlarge"]
}

variable "node_desired_size" {
  description = "Desired number of nodes"
  type        = number
  default     = 3
  
  validation {
    condition     = var.node_desired_size >= 1
    error_message = "Node desired size must be at least 1."
  }
}

variable "node_max_size" {
  description = "Maximum number of nodes"
  type        = number
  default     = 10
  
  validation {
    condition     = var.node_max_size >= var.node_desired_size
    error_message = "Node max size must be greater than or equal to desired size."
  }
}

variable "node_min_size" {
  description = "Minimum number of nodes"
  type        = number
  default     = 1
  
  validation {
    condition     = var.node_min_size <= var.node_desired_size
    error_message = "Node min size must be less than or equal to desired size."
  }
}

# =============================================================================
# FlashMM Application Configuration
# =============================================================================

variable "flashmm_image_tag" {
  description = "FlashMM Docker image tag"
  type        = string
  default     = "latest"
}

variable "flashmm_replica_count" {
  description = "Number of FlashMM application replicas"
  type        = number
  default     = 3
  
  validation {
    condition     = var.flashmm_replica_count >= 1
    error_message = "FlashMM replica count must be at least 1."
  }
}

variable "flashmm_resources_requests" {
  description = "FlashMM resource requests"
  type = object({
    cpu    = string
    memory = string
  })
  default = {
    cpu    = "500m"
    memory = "1Gi"
  }
}

variable "flashmm_resources_limits" {
  description = "FlashMM resource limits"
  type = object({
    cpu    = string
    memory = string
  })
  default = {
    cpu    = "2"
    memory = "4Gi"
  }
}

# =============================================================================
# External API Configuration
# =============================================================================

variable "sei_network" {
  description = "Sei network (testnet, mainnet)"
  type        = string
  default     = "testnet"
  
  validation {
    condition     = contains(["testnet", "mainnet"], var.sei_network)
    error_message = "Sei network must be either testnet or mainnet."
  }
}

variable "sei_rpc_url" {
  description = "Sei RPC URL"
  type        = string
  default     = "https://sei-testnet-rpc.polkachu.com"
}

variable "cambrian_api_key" {
  description = "Cambrian API key"
  type        = string
  default     = ""
  sensitive   = true
}

variable "azure_openai_endpoint" {
  description = "Azure OpenAI endpoint"
  type        = string
  default     = ""
}

# =============================================================================
# Monitoring Configuration
# =============================================================================

variable "monitoring_enabled" {
  description = "Enable monitoring stack (Prometheus, Grafana)"
  type        = bool
  default     = true
}

variable "logging_enabled" {
  description = "Enable logging stack (ELK)"
  type        = bool
  default     = true
}

variable "prometheus_retention" {
  description = "Prometheus data retention period"
  type        = string
  default     = "30d"
}

variable "prometheus_storage_size" {
  description = "Prometheus storage size"
  type        = string
  default     = "100Gi"
}

variable "grafana_admin_password" {
  description = "Grafana admin password"
  type        = string
  default     = ""
  sensitive   = true
}

variable "grafana_storage_size" {
  description = "Grafana storage size"
  type        = string
  default     = "10Gi"
}

variable "slack_webhook_url" {
  description = "Slack webhook URL for alerts"
  type        = string
  default     = ""
  sensitive   = true
}

variable "pagerduty_integration_key" {
  description = "PagerDuty integration key"
  type        = string
  default     = ""
  sensitive   = true
}

# =============================================================================
# Backup Configuration
# =============================================================================

variable "backup_enabled" {
  description = "Enable backup functionality"
  type        = bool
  default     = true
}

variable "backup_schedule" {
  description = "Backup schedule (cron format)"
  type        = string
  default     = "0 2 * * *"  # Daily at 2 AM
  
  validation {
    condition     = can(regex("^[0-9*,-/]+ [0-9*,-/]+ [0-9*,-/]+ [0-9*,-/]+ [0-9*,-/]+$", var.backup_schedule))
    error_message = "Backup schedule must be a valid cron expression."
  }
}

variable "backup_retention" {
  description = "Backup retention period in days"
  type        = number
  default     = 30
  
  validation {
    condition     = var.backup_retention >= 1 && var.backup_retention <= 365
    error_message = "Backup retention must be between 1 and 365 days."
  }
}

variable "backup_bucket_name" {
  description = "S3 bucket name for backups"
  type        = string
  default     = ""
}

# =============================================================================
# Security Configuration
# =============================================================================

variable "enable_encryption" {
  description = "Enable encryption at rest for databases"
  type        = bool
  default     = true
}

variable "enable_backup_encryption" {
  description = "Enable encryption for backups"
  type        = bool
  default     = true
}

variable "allowed_cidr_blocks" {
  description = "CIDR blocks allowed to access the cluster"
  type        = list(string)
  default     = ["0.0.0.0/0"]  # Restrict this in production
}

variable "certificate_arn" {
  description = "SSL certificate ARN for load balancer"
  type        = string
  default     = ""
}

# =============================================================================
# Cost Optimization
# =============================================================================

variable "enable_spot_instances" {
  description = "Enable spot instances for cost optimization"
  type        = bool
  default     = false
}

variable "spot_instance_percentage" {
  description = "Percentage of spot instances in node groups"
  type        = number
  default     = 50
  
  validation {
    condition     = var.spot_instance_percentage >= 0 && var.spot_instance_percentage <= 100
    error_message = "Spot instance percentage must be between 0 and 100."
  }
}

# =============================================================================
# Feature Flags
# =============================================================================

variable "enable_service_mesh" {
  description = "Enable service mesh (Istio)"
  type        = bool
  default     = false
}

variable "enable_auto_scaling" {
  description = "Enable horizontal pod autoscaling"
  type        = bool
  default     = true
}

variable "enable_network_policies" {
  description = "Enable Kubernetes network policies"
  type        = bool
  default     = true
}

variable "enable_pod_security_policies" {
  description = "Enable pod security policies"
  type        = bool
  default     = true
}
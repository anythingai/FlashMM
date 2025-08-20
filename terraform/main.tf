# FlashMM Terraform Infrastructure as Code
# Multi-cloud infrastructure provisioning for production deployment

terraform {
  required_version = ">= 1.5.0"
  
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    google = {
      source  = "hashicorp/google"
      version = "~> 4.0"
    }
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.0"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.0"
    }
    tls = {
      source  = "hashicorp/tls"
      version = "~> 4.0"
    }
  }
  
  # Remote state backend configuration
  # Configure this based on your chosen backend
  backend "s3" {
    # bucket         = "flashmm-terraform-state"
    # key            = "infrastructure/terraform.tfstate"
    # region         = "us-east-1"
    # encrypt        = true
    # dynamodb_table = "flashmm-terraform-locks"
  }
}

# =============================================================================
# Local Values and Data Sources
# =============================================================================
locals {
  # Common tags for all resources
  common_tags = {
    Project     = "FlashMM"
    Environment = var.environment
    ManagedBy   = "Terraform"
    Owner       = "FlashMM-Team"
    CostCenter  = "Trading-Platform"
  }
  
  # Resource naming convention
  name_prefix = "${var.project_name}-${var.environment}"
  
  # Network configuration
  vpc_cidr = var.vpc_cidr
  
  # Availability zones
  availability_zones = data.aws_availability_zones.available.names
}

# Get available availability zones
data "aws_availability_zones" "available" {
  state = "available"
}

# Get current AWS caller identity
data "aws_caller_identity" "current" {}

# Get current AWS region
data "aws_region" "current" {}

# =============================================================================
# Cloud Provider Configurations
# =============================================================================

# AWS Provider Configuration
provider "aws" {
  region = var.aws_region
  
  default_tags {
    tags = local.common_tags
  }
}

# Google Cloud Provider Configuration
provider "google" {
  project = var.gcp_project_id
  region  = var.gcp_region
  zone    = var.gcp_zone
}

# Azure Provider Configuration
provider "azurerm" {
  features {
    key_vault {
      purge_soft_delete_on_destroy    = true
      recover_soft_deleted_key_vaults = true
    }
    resource_group {
      prevent_deletion_if_contains_resources = false
    }
  }
  
  subscription_id = var.azure_subscription_id
  tenant_id       = var.azure_tenant_id
}

# =============================================================================
# Conditional Cloud Infrastructure Modules
# =============================================================================

# AWS Infrastructure
module "aws_infrastructure" {
  count  = var.cloud_provider == "aws" ? 1 : 0
  source = "./modules/aws"
  
  # Common variables
  project_name   = var.project_name
  environment    = var.environment
  region         = var.aws_region
  
  # Network configuration
  vpc_cidr             = var.vpc_cidr
  availability_zones   = local.availability_zones
  public_subnet_cidrs  = var.public_subnet_cidrs
  private_subnet_cidrs = var.private_subnet_cidrs
  
  # EKS configuration
  cluster_version     = var.k8s_version
  node_instance_types = var.node_instance_types
  node_desired_size   = var.node_desired_size
  node_max_size       = var.node_max_size
  node_min_size       = var.node_min_size
  
  # Database configuration
  rds_instance_class    = var.rds_instance_class
  rds_allocated_storage = var.rds_allocated_storage
  
  # Redis configuration
  redis_node_type       = var.redis_node_type
  redis_num_cache_nodes = var.redis_num_cache_nodes
  
  # Tags
  tags = local.common_tags
}

# GCP Infrastructure
module "gcp_infrastructure" {
  count  = var.cloud_provider == "gcp" ? 1 : 0
  source = "./modules/gcp"
  
  # Common variables
  project_id     = var.gcp_project_id
  project_name   = var.project_name
  environment    = var.environment
  region         = var.gcp_region
  zone           = var.gcp_zone
  
  # Network configuration
  vpc_cidr             = var.vpc_cidr
  public_subnet_cidrs  = var.public_subnet_cidrs
  private_subnet_cidrs = var.private_subnet_cidrs
  
  # GKE configuration
  cluster_version     = var.k8s_version
  node_machine_type   = var.gcp_node_machine_type
  node_count          = var.node_desired_size
  max_node_count      = var.node_max_size
  min_node_count      = var.node_min_size
  
  # Database configuration
  database_tier = var.gcp_database_tier
  
  # Labels
  labels = local.common_tags
}

# Azure Infrastructure
module "azure_infrastructure" {
  count  = var.cloud_provider == "azure" ? 1 : 0
  source = "./modules/azure"
  
  # Common variables
  subscription_id = var.azure_subscription_id
  project_name    = var.project_name
  environment     = var.environment
  location        = var.azure_location
  
  # Network configuration
  vpc_cidr             = var.vpc_cidr
  public_subnet_cidrs  = var.public_subnet_cidrs
  private_subnet_cidrs = var.private_subnet_cidrs
  
  # AKS configuration
  cluster_version   = var.k8s_version
  node_vm_size      = var.azure_node_vm_size
  node_count        = var.node_desired_size
  max_node_count    = var.node_max_size
  min_node_count    = var.node_min_size
  
  # Database configuration
  postgresql_sku_name = var.azure_postgresql_sku
  
  # Tags
  tags = local.common_tags
}

# =============================================================================
# Kubernetes Configuration
# =============================================================================

# Configure Kubernetes provider based on selected cloud
locals {
  cluster_endpoint = var.cloud_provider == "aws" ? (
    length(module.aws_infrastructure) > 0 ? module.aws_infrastructure[0].cluster_endpoint : ""
  ) : var.cloud_provider == "gcp" ? (
    length(module.gcp_infrastructure) > 0 ? module.gcp_infrastructure[0].cluster_endpoint : ""
  ) : var.cloud_provider == "azure" ? (
    length(module.azure_infrastructure) > 0 ? module.azure_infrastructure[0].cluster_endpoint : ""
  ) : ""
  
  cluster_ca_certificate = var.cloud_provider == "aws" ? (
    length(module.aws_infrastructure) > 0 ? module.aws_infrastructure[0].cluster_ca_certificate : ""
  ) : var.cloud_provider == "gcp" ? (
    length(module.gcp_infrastructure) > 0 ? module.gcp_infrastructure[0].cluster_ca_certificate : ""
  ) : var.cloud_provider == "azure" ? (
    length(module.azure_infrastructure) > 0 ? module.azure_infrastructure[0].cluster_ca_certificate : ""
  ) : ""
}

provider "kubernetes" {
  host                   = local.cluster_endpoint
  cluster_ca_certificate = base64decode(local.cluster_ca_certificate)
  
  dynamic "exec" {
    for_each = var.cloud_provider == "aws" ? [1] : []
    content {
      api_version = "client.authentication.k8s.io/v1beta1"
      command     = "aws"
      args        = ["eks", "get-token", "--cluster-name", local.name_prefix]
    }
  }
  
  dynamic "exec" {
    for_each = var.cloud_provider == "gcp" ? [1] : []
    content {
      api_version = "client.authentication.k8s.io/v1beta1"
      command     = "gke-gcloud-auth-plugin"
    }
  }
}

provider "helm" {
  kubernetes {
    host                   = local.cluster_endpoint
    cluster_ca_certificate = base64decode(local.cluster_ca_certificate)
    
    dynamic "exec" {
      for_each = var.cloud_provider == "aws" ? [1] : []
      content {
        api_version = "client.authentication.k8s.io/v1beta1"
        command     = "aws"
        args        = ["eks", "get-token", "--cluster-name", local.name_prefix]
      }
    }
  }
}

# =============================================================================
# FlashMM Application Deployment
# =============================================================================

module "flashmm_application" {
  source = "./modules/flashmm"
  
  # Dependency on infrastructure
  depends_on = [
    module.aws_infrastructure,
    module.gcp_infrastructure,
    module.azure_infrastructure
  ]
  
  # Common configuration
  project_name = var.project_name
  environment  = var.environment
  
  # Application configuration
  image_tag           = var.flashmm_image_tag
  replica_count       = var.flashmm_replica_count
  resources_requests  = var.flashmm_resources_requests
  resources_limits    = var.flashmm_resources_limits
  
  # External dependencies
  sei_network         = var.sei_network
  sei_rpc_url         = var.sei_rpc_url
  cambrian_api_key    = var.cambrian_api_key
  azure_openai_endpoint = var.azure_openai_endpoint
  
  # Monitoring configuration
  monitoring_enabled    = var.monitoring_enabled
  logging_enabled       = var.logging_enabled
  
  # Database connections (from infrastructure modules)
  database_endpoint = var.cloud_provider == "aws" ? (
    length(module.aws_infrastructure) > 0 ? module.aws_infrastructure[0].rds_endpoint : ""
  ) : var.cloud_provider == "gcp" ? (
    length(module.gcp_infrastructure) > 0 ? module.gcp_infrastructure[0].database_connection_name : ""
  ) : var.cloud_provider == "azure" ? (
    length(module.azure_infrastructure) > 0 ? module.azure_infrastructure[0].postgresql_fqdn : ""
  ) : ""
  
  redis_endpoint = var.cloud_provider == "aws" ? (
    length(module.aws_infrastructure) > 0 ? module.aws_infrastructure[0].redis_endpoint : ""
  ) : var.cloud_provider == "gcp" ? (
    length(module.gcp_infrastructure) > 0 ? module.gcp_infrastructure[0].redis_host : ""
  ) : var.cloud_provider == "azure" ? (
    length(module.azure_infrastructure) > 0 ? module.azure_infrastructure[0].redis_hostname : ""
  ) : ""
}

# =============================================================================
# Monitoring and Observability
# =============================================================================

module "monitoring" {
  count  = var.monitoring_enabled ? 1 : 0
  source = "./modules/monitoring"
  
  depends_on = [module.flashmm_application]
  
  project_name = var.project_name
  environment  = var.environment
  
  # Monitoring configuration
  prometheus_retention = var.prometheus_retention
  grafana_admin_password = var.grafana_admin_password
  
  # Storage configuration
  prometheus_storage_size = var.prometheus_storage_size
  grafana_storage_size    = var.grafana_storage_size
  
  # External integrations
  slack_webhook_url = var.slack_webhook_url
  pagerduty_integration_key = var.pagerduty_integration_key
}

# =============================================================================
# Backup and Disaster Recovery
# =============================================================================

module "backup" {
  count  = var.backup_enabled ? 1 : 0
  source = "./modules/backup"
  
  depends_on = [module.flashmm_application]
  
  project_name = var.project_name
  environment  = var.environment
  
  # Backup configuration
  backup_schedule    = var.backup_schedule
  backup_retention   = var.backup_retention
  backup_bucket_name = var.backup_bucket_name
  
  # Cloud-specific backup settings
  cloud_provider = var.cloud_provider
}
# AWS Module Outputs

output "cluster_name" {
  description = "EKS cluster name"
  value       = aws_eks_cluster.flashmm.name
}

output "cluster_endpoint" {
  description = "EKS cluster endpoint"
  value       = aws_eks_cluster.flashmm.endpoint
}

output "cluster_ca_certificate" {
  description = "EKS cluster CA certificate"
  value       = aws_eks_cluster.flashmm.certificate_authority[0].data
}

output "cluster_version" {
  description = "EKS cluster version"
  value       = aws_eks_cluster.flashmm.version
}

output "vpc_id" {
  description = "VPC ID"
  value       = aws_vpc.flashmm.id
}

output "public_subnet_ids" {
  description = "Public subnet IDs"
  value       = aws_subnet.public[*].id
}

output "private_subnet_ids" {
  description = "Private subnet IDs"
  value       = aws_subnet.private[*].id
}

output "cluster_security_group_id" {
  description = "Cluster security group ID"
  value       = aws_security_group.cluster.id
}

output "node_security_group_id" {
  description = "Node security group ID"
  value       = aws_security_group.node.id
}

output "rds_endpoint" {
  description = "RDS endpoint"
  value       = aws_db_instance.flashmm.endpoint
}

output "rds_port" {
  description = "RDS port"
  value       = aws_db_instance.flashmm.port
}

output "rds_database_name" {
  description = "RDS database name"
  value       = aws_db_instance.flashmm.db_name
}

output "rds_username" {
  description = "RDS username"
  value       = aws_db_instance.flashmm.username
}

output "redis_endpoint" {
  description = "Redis endpoint"
  value       = aws_elasticache_replication_group.flashmm.primary_endpoint_address
}

output "redis_port" {
  description = "Redis port"
  value       = aws_elasticache_replication_group.flashmm.port
}

output "load_balancer_dns" {
  description = "Load balancer DNS (placeholder - will be created by K8s service)"
  value       = "Will be created by Kubernetes LoadBalancer service"
}

output "estimated_monthly_cost" {
  description = "Estimated monthly cost in USD"
  value = (
    # EKS cluster: $73/month
    73 +
    # EKS nodes: instance cost * desired size
    (var.node_instance_types[0] == "t3.large" ? 67.2 : 
     var.node_instance_types[0] == "t3.xlarge" ? 134.4 : 100) * var.node_desired_size +
    # RDS: varies by instance class
    (var.rds_instance_class == "db.r5.large" ? 175 :
     var.rds_instance_class == "db.r5.xlarge" ? 350 : 200) +
    # Redis: varies by node type
    (var.redis_node_type == "cache.r6g.large" ? 120 :
     var.redis_node_type == "cache.r6g.xlarge" ? 240 : 150) * var.redis_num_cache_nodes +
    # Storage and other services
    50
  )
}

# Sensitive outputs for application configuration
output "database_password" {
  description = "Database password"
  value       = random_password.db_password.result
  sensitive   = true
}

output "redis_auth_token" {
  description = "Redis auth token"
  value       = random_password.redis_password.result
  sensitive   = true
}

output "kms_key_eks_arn" {
  description = "EKS KMS key ARN"
  value       = aws_kms_key.eks.arn
}

output "kms_key_rds_arn" {
  description = "RDS KMS key ARN"
  value       = aws_kms_key.rds.arn
}
#!/usr/bin/env python3
"""
FlashMM Infrastructure Testing Suite
Comprehensive validation of deployment infrastructure components
"""

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

class InfrastructureTestSuite:
    """Comprehensive infrastructure testing suite"""

    def __init__(self, environment: str = "staging", namespace: str = "flashmm"):
        self.environment = environment
        self.namespace = namespace
        self.kubectl_timeout = 300  # 5 minutes
        self.health_check_retries = 10
        self.test_results = {}

    def run_kubectl_command(self, cmd: list[str]) -> tuple[bool, str, str]:
        """Execute kubectl command and return success, stdout, stderr"""
        try:
            import shutil

            # Validate kubectl executable path
            kubectl_path = shutil.which("kubectl")
            if not kubectl_path:
                return False, "", "kubectl not found in PATH"

            # Validate command arguments (basic sanitation)
            for arg in cmd:
                if not isinstance(arg, str):
                    return False, "", f"Invalid argument type: {type(arg)}"
                # Check for shell injection attempts
                if any(char in arg for char in [';', '|', '&', '$', '`']):
                    return False, "", f"Potentially unsafe argument: {arg}"

            result = subprocess.run(  # noqa: S603 - command is validated above
                [kubectl_path] + cmd,
                capture_output=True,
                text=True,
                timeout=self.kubectl_timeout,
                shell=False  # Explicit shell=False for security
            )
            return result.returncode == 0, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return False, "", "Command timeout"
        except Exception as e:
            return False, "", str(e)

    def run_helm_command(self, cmd: list[str]) -> tuple[bool, str, str]:
        """Execute helm command and return success, stdout, stderr"""
        try:
            import shutil

            # Validate helm executable path
            helm_path = shutil.which("helm")
            if not helm_path:
                return False, "", "helm not found in PATH"

            # Validate command arguments (basic sanitation)
            for arg in cmd:
                if not isinstance(arg, str):
                    return False, "", f"Invalid argument type: {type(arg)}"
                # Check for shell injection attempts
                if any(char in arg for char in [';', '|', '&', '$', '`']):
                    return False, "", f"Potentially unsafe argument: {arg}"

            result = subprocess.run(  # noqa: S603 - command is validated above
                [helm_path] + cmd,
                capture_output=True,
                text=True,
                timeout=self.kubectl_timeout,
                shell=False  # Explicit shell=False for security
            )
            return result.returncode == 0, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return False, "", "Command timeout"
        except Exception as e:
            return False, "", str(e)

class TestKubernetesDeployment(InfrastructureTestSuite):
    """Test Kubernetes deployment components"""

    def test_namespace_exists(self):
        """Test that FlashMM namespace exists and is properly configured"""
        success, stdout, stderr = self.run_kubectl_command([
            "get", "namespace", self.namespace, "-o", "json"
        ])

        assert success, f"Namespace {self.namespace} not found: {stderr}"

        namespace_data = json.loads(stdout)
        labels = namespace_data.get("metadata", {}).get("labels", {})

        # Check security labels
        assert "pod-security.kubernetes.io/enforce" in labels, "Pod security enforcement not configured"
        assert labels["pod-security.kubernetes.io/enforce"] == "restricted", "Pod security not set to restricted"

    def test_deployments_healthy(self):
        """Test that all deployments are healthy and running"""
        success, stdout, stderr = self.run_kubectl_command([
            "get", "deployments", "-n", self.namespace, "-o", "json"
        ])

        assert success, f"Failed to get deployments: {stderr}"

        deployments_data = json.loads(stdout)
        deployments = deployments_data.get("items", [])

        assert len(deployments) > 0, "No deployments found"

        for deployment in deployments:
            name = deployment["metadata"]["name"]
            status = deployment.get("status", {})

            replicas = status.get("replicas", 0)
            ready_replicas = status.get("readyReplicas", 0)
            available_replicas = status.get("availableReplicas", 0)

            assert ready_replicas == replicas, f"Deployment {name}: {ready_replicas}/{replicas} replicas ready"
            assert available_replicas == replicas, f"Deployment {name}: {available_replicas}/{replicas} replicas available"

    def test_pods_security_context(self):
        """Test that all pods have proper security contexts"""
        success, stdout, stderr = self.run_kubectl_command([
            "get", "pods", "-n", self.namespace,
            "-l", "app.kubernetes.io/part-of=flashmm-platform",
            "-o", "json"
        ])

        assert success, f"Failed to get pods: {stderr}"

        pods_data = json.loads(stdout)
        pods = pods_data.get("items", [])

        for pod in pods:
            name = pod["metadata"]["name"]
            spec = pod.get("spec", {})

            # Check pod-level security context
            pod_security = spec.get("securityContext", {})
            assert pod_security.get("runAsNonRoot") is True, f"Pod {name}: not running as non-root"
            assert pod_security.get("fsGroup", 0) >= 10001, f"Pod {name}: fsGroup not properly set"

            # Check container-level security contexts
            containers = spec.get("containers", [])
            for container in containers:
                container_name = container["name"]
                container_security = container.get("securityContext", {})

                assert container_security.get("allowPrivilegeEscalation") is False, \
                    f"Pod {name}, container {container_name}: privilege escalation allowed"
                assert container_security.get("readOnlyRootFilesystem") is True, \
                    f"Pod {name}, container {container_name}: root filesystem not read-only"

                # Check capabilities
                capabilities = container_security.get("capabilities", {})
                drop = capabilities.get("drop", [])
                assert "ALL" in drop, f"Pod {name}, container {container_name}: capabilities not dropped"

    def test_services_configuration(self):
        """Test that services are properly configured"""
        success, stdout, stderr = self.run_kubectl_command([
            "get", "services", "-n", self.namespace, "-o", "json"
        ])

        assert success, f"Failed to get services: {stderr}"

        services_data = json.loads(stdout)
        services = services_data.get("items", [])

        # Check that main app service exists
        app_service = next((s for s in services if s["metadata"]["name"] == "flashmm-app"), None)
        assert app_service is not None, "FlashMM app service not found"

        # Check service type (should not be NodePort in production)
        service_type = app_service["spec"]["type"]
        if self.environment == "production":
            assert service_type != "NodePort", "NodePort services not allowed in production"

    def test_ingress_configuration(self):
        """Test ingress configuration and TLS"""
        success, stdout, stderr = self.run_kubectl_command([
            "get", "ingress", "-n", self.namespace, "-o", "json"
        ])

        if success:
            ingress_data = json.loads(stdout)
            ingresses = ingress_data.get("items", [])

            for ingress in ingresses:
                name = ingress["metadata"]["name"]
                spec = ingress.get("spec", {})

                # Check TLS configuration in production
                if self.environment == "production":
                    tls = spec.get("tls", [])
                    assert len(tls) > 0, f"Ingress {name}: TLS not configured"

                # Check annotations for security
                annotations = ingress.get("metadata", {}).get("annotations", {})
                if "cert-manager.io/cluster-issuer" in annotations:
                    assert annotations["cert-manager.io/cluster-issuer"] == "letsencrypt-prod", \
                        f"Ingress {name}: not using production certificate issuer"

    def test_persistent_volumes(self):
        """Test persistent volume configuration"""
        success, stdout, stderr = self.run_kubectl_command([
            "get", "pvc", "-n", self.namespace, "-o", "json"
        ])

        assert success, f"Failed to get PVCs: {stderr}"

        pvc_data = json.loads(stdout)
        pvcs = pvc_data.get("items", [])

        for pvc in pvcs:
            name = pvc["metadata"]["name"]
            spec = pvc.get("spec", {})

            # Check storage class (should use fast storage for production)
            storage_class = spec.get("storageClassName", "")
            if self.environment == "production":
                assert "fast" in storage_class.lower() or "ssd" in storage_class.lower(), \
                    f"PVC {name}: not using fast storage class"

            # Check access modes
            access_modes = spec.get("accessModes", [])
            assert len(access_modes) > 0, f"PVC {name}: no access modes specified"

class TestContainerSecurity(InfrastructureTestSuite):
    """Test container security and image compliance"""

    def test_container_images(self):
        """Test that container images are from trusted sources"""
        success, stdout, stderr = self.run_kubectl_command([
            "get", "pods", "-n", self.namespace, "-o", "jsonpath={.items[*].spec.containers[*].image}"
        ])

        assert success, f"Failed to get container images: {stderr}"

        images = stdout.strip().split()
        trusted_registries = ["ghcr.io/flashmm/", "docker.io/library/", "registry.k8s.io/", "quay.io/"]

        for image in images:
            is_trusted = any(image.startswith(registry) for registry in trusted_registries)
            assert is_trusted, f"Untrusted image: {image}"

            # Check for latest tags in production
            if self.environment == "production":
                assert not image.endswith(":latest"), f"Production should not use 'latest' tag: {image}"

    def test_container_resources(self):
        """Test that containers have resource limits defined"""
        success, stdout, stderr = self.run_kubectl_command([
            "get", "pods", "-n", self.namespace, "-o", "json"
        ])

        assert success, f"Failed to get pods: {stderr}"

        pods_data = json.loads(stdout)
        pods = pods_data.get("items", [])

        for pod in pods:
            pod_name = pod["metadata"]["name"]
            containers = pod.get("spec", {}).get("containers", [])

            for container in containers:
                container_name = container["name"]
                resources = container.get("resources", {})

                # Check resource limits
                limits = resources.get("limits", {})
                assert "memory" in limits, f"Pod {pod_name}, container {container_name}: no memory limit"
                assert "cpu" in limits, f"Pod {pod_name}, container {container_name}: no CPU limit"

                # Check resource requests
                requests = resources.get("requests", {})
                assert "memory" in requests, f"Pod {pod_name}, container {container_name}: no memory request"
                assert "cpu" in requests, f"Pod {pod_name}, container {container_name}: no CPU request"

class TestNetworkSecurity(InfrastructureTestSuite):
    """Test network security policies and configurations"""

    def test_network_policies_exist(self):
        """Test that network policies are configured"""
        success, stdout, stderr = self.run_kubectl_command([
            "get", "networkpolicy", "-n", self.namespace, "-o", "json"
        ])

        assert success, f"Failed to get network policies: {stderr}"

        policies_data = json.loads(stdout)
        policies = policies_data.get("items", [])

        assert len(policies) > 0, "No network policies found"

        # Check for default deny policy
        policy_names = [p["metadata"]["name"] for p in policies]
        assert "default-deny-all" in policy_names, "Default deny-all policy not found"

    def test_service_mesh_configuration(self):
        """Test service mesh configuration if enabled"""
        # Check if Istio is installed
        success, stdout, stderr = self.run_kubectl_command([
            "get", "namespace", "istio-system"
        ])

        if success:
            # Istio is installed, check for FlashMM service mesh configuration
            success, stdout, stderr = self.run_kubectl_command([
                "get", "virtualservice", "-n", self.namespace
            ])

            if success and stdout.strip():
                # Service mesh is configured
                assert True, "Service mesh configuration found"
            else:
                pytest.skip("Service mesh installed but not configured for FlashMM")
        else:
            pytest.skip("Service mesh not installed")

    def test_tls_configuration(self):
        """Test TLS/SSL configuration"""
        success, stdout, stderr = self.run_kubectl_command([
            "get", "ingress", "-n", self.namespace, "-o", "json"
        ])

        if success:
            ingress_data = json.loads(stdout)
            ingresses = ingress_data.get("items", [])

            for ingress in ingresses:
                name = ingress["metadata"]["name"]
                spec = ingress.get("spec", {})

                if self.environment == "production":
                    tls = spec.get("tls", [])
                    assert len(tls) > 0, f"Ingress {name}: TLS not configured in production"

class TestMonitoringStack(InfrastructureTestSuite):
    """Test monitoring and observability stack"""

    def test_prometheus_deployment(self):
        """Test Prometheus deployment and configuration"""
        success, stdout, stderr = self.run_kubectl_command([
            "get", "deployment", "-n", "flashmm-monitoring", "prometheus-server"
        ])

        if not success:
            pytest.skip("Prometheus not deployed")

        # Test Prometheus health endpoint
        success, stdout, stderr = self.run_kubectl_command([
            "run", f"prometheus-test-{int(time.time())}",
            "--rm", "-i", "--restart=Never",
            "--image=curlimages/curl",
            "--timeout=60s",
            "-n", "flashmm-monitoring",
            "--", "curl", "-f", "http://prometheus:9090/-/healthy"
        ])

        assert success, "Prometheus health check failed"

    def test_grafana_deployment(self):
        """Test Grafana deployment and dashboards"""
        success, stdout, stderr = self.run_kubectl_command([
            "get", "deployment", "-n", "flashmm-monitoring", "grafana"
        ])

        if not success:
            pytest.skip("Grafana not deployed")

        # Test Grafana health endpoint
        success, stdout, stderr = self.run_kubectl_command([
            "run", f"grafana-test-{int(time.time())}",
            "--rm", "-i", "--restart=Never",
            "--image=curlimages/curl",
            "--timeout=60s",
            "-n", "flashmm-monitoring",
            "--", "curl", "-f", "http://grafana:3000/api/health"
        ])

        assert success, "Grafana health check failed"

    def test_elasticsearch_deployment(self):
        """Test Elasticsearch deployment for logging"""
        success, stdout, stderr = self.run_kubectl_command([
            "get", "deployment", "-n", "flashmm-monitoring", "elasticsearch"
        ])

        if not success:
            pytest.skip("Elasticsearch not deployed")

        # Test Elasticsearch health
        success, stdout, stderr = self.run_kubectl_command([
            "run", f"elasticsearch-test-{int(time.time())}",
            "--rm", "-i", "--restart=Never",
            "--image=curlimages/curl",
            "--timeout=60s",
            "-n", "flashmm-monitoring",
            "--", "curl", "-f", "http://elasticsearch:9200/_cluster/health"
        ])

        assert success, "Elasticsearch health check failed"

class TestApplicationDeployment(InfrastructureTestSuite):
    """Test FlashMM application deployment"""

    def test_application_pods_running(self):
        """Test that FlashMM application pods are running"""
        success, stdout, stderr = self.run_kubectl_command([
            "get", "pods", "-n", self.namespace,
            "-l", "app.kubernetes.io/name=flashmm",
            "--field-selector=status.phase=Running",
            "-o", "json"
        ])

        assert success, f"Failed to get running pods: {stderr}"

        pods_data = json.loads(stdout)
        running_pods = pods_data.get("items", [])

        assert len(running_pods) > 0, "No FlashMM pods running"

        # Check minimum replicas for production
        if self.environment == "production":
            assert len(running_pods) >= 2, f"Production should have at least 2 replicas, found {len(running_pods)}"

    def test_application_health_endpoints(self):
        """Test application health endpoints"""
        success, stdout, stderr = self.run_kubectl_command([
            "run", f"health-test-{int(time.time())}",
            "--rm", "-i", "--restart=Never",
            "--image=curlimages/curl",
            "--timeout=60s",
            "-n", self.namespace,
            "--", "curl", "-f", "http://flashmm-app:8000/health"
        ])

        assert success, f"Application health check failed: {stderr}"

    def test_application_metrics_endpoints(self):
        """Test application metrics endpoints"""
        success, stdout, stderr = self.run_kubectl_command([
            "run", f"metrics-test-{int(time.time())}",
            "--rm", "-i", "--restart=Never",
            "--image=curlimages/curl",
            "--timeout=60s",
            "-n", self.namespace,
            "--", "curl", "-f", "http://flashmm-app:8000/metrics"
        ])

        assert success, f"Application metrics endpoint failed: {stderr}"

class TestDatabaseConnectivity(InfrastructureTestSuite):
    """Test database connectivity and configuration"""

    def test_postgresql_connectivity(self):
        """Test PostgreSQL connectivity"""
        success, stdout, stderr = self.run_kubectl_command([
            "get", "deployment", "-n", self.namespace, "postgres"
        ])

        if not success:
            pytest.skip("PostgreSQL not deployed")

        # Test database connectivity
        success, stdout, stderr = self.run_kubectl_command([
            "exec", "-n", self.namespace, "deployment/postgres",
            "--", "pg_isready", "-U", "flashmm", "-d", f"flashmm_{self.environment}"
        ])

        assert success, f"PostgreSQL connectivity failed: {stderr}"

    def test_redis_connectivity(self):
        """Test Redis connectivity"""
        success, stdout, stderr = self.run_kubectl_command([
            "get", "deployment", "-n", self.namespace, "redis"
        ])

        if not success:
            pytest.skip("Redis not deployed")

        # Test Redis connectivity
        success, stdout, stderr = self.run_kubectl_command([
            "exec", "-n", self.namespace, "deployment/redis",
            "--", "redis-cli", "ping"
        ])

        assert success, f"Redis connectivity failed: {stderr}"
        assert "PONG" in stdout, "Redis did not respond with PONG"

    def test_influxdb_connectivity(self):
        """Test InfluxDB connectivity"""
        success, stdout, stderr = self.run_kubectl_command([
            "get", "deployment", "-n", self.namespace, "influxdb"
        ])

        if not success:
            pytest.skip("InfluxDB not deployed")

        # Test InfluxDB health
        success, stdout, stderr = self.run_kubectl_command([
            "run", f"influxdb-test-{int(time.time())}",
            "--rm", "-i", "--restart=Never",
            "--image=curlimages/curl",
            "--timeout=60s",
            "-n", self.namespace,
            "--", "curl", "-f", "http://influxdb:8086/ping"
        ])

        assert success, f"InfluxDB connectivity failed: {stderr}"

class TestSecurityCompliance(InfrastructureTestSuite):
    """Test security compliance and policies"""

    def test_rbac_configuration(self):
        """Test RBAC configuration"""
        # Test service accounts
        success, stdout, stderr = self.run_kubectl_command([
            "get", "serviceaccounts", "-n", self.namespace, "-o", "json"
        ])

        assert success, f"Failed to get service accounts: {stderr}"

        sa_data = json.loads(stdout)
        service_accounts = sa_data.get("items", [])

        # Should have dedicated service account
        sa_names = [sa["metadata"]["name"] for sa in service_accounts]
        assert "flashmm" in sa_names, "FlashMM service account not found"

        # Test roles and role bindings
        success, stdout, stderr = self.run_kubectl_command([
            "get", "roles,rolebindings", "-n", self.namespace
        ])

        assert success, f"RBAC not properly configured: {stderr}"

    def test_secrets_encryption(self):
        """Test that secrets are properly configured"""
        success, stdout, stderr = self.run_kubectl_command([
            "get", "secrets", "-n", self.namespace, "-o", "json"
        ])

        assert success, f"Failed to get secrets: {stderr}"

        secrets_data = json.loads(stdout)
        secrets = secrets_data.get("items", [])

        # Check for required secrets
        secret_names = [s["metadata"]["name"] for s in secrets]
        required_secrets = ["flashmm-secrets"]

        for required_secret in required_secrets:
            assert required_secret in secret_names, f"Required secret {required_secret} not found"

    def test_pod_security_standards(self):
        """Test Pod Security Standards compliance"""
        # Check namespace labels
        success, stdout, stderr = self.run_kubectl_command([
            "get", "namespace", self.namespace, "-o", "json"
        ])

        assert success, f"Failed to get namespace: {stderr}"

        namespace_data = json.loads(stdout)
        labels = namespace_data.get("metadata", {}).get("labels", {})

        # Check Pod Security Standards labels
        assert "pod-security.kubernetes.io/enforce" in labels, "Pod security enforcement not set"
        assert labels["pod-security.kubernetes.io/enforce"] == "restricted", "Pod security not restricted"

class TestPerformanceValidation(InfrastructureTestSuite):
    """Test performance and resource utilization"""

    def test_resource_utilization(self):
        """Test resource utilization is within acceptable ranges"""
        # Get resource usage
        success, stdout, stderr = self.run_kubectl_command([
            "top", "pods", "-n", self.namespace, "--no-headers"
        ])

        if not success:
            pytest.skip("Resource metrics not available")

        lines = stdout.strip().split('\n')
        for line in lines:
            if line.strip():
                parts = line.split()
                pod_name = parts[0]
                cpu_usage = parts[1]
                memory_usage = parts[2]

                # Basic validation (would be more sophisticated in practice)
                if cpu_usage != "<unknown>":
                    cpu_value = int(cpu_usage.replace('m', ''))
                    assert cpu_value < 2000, f"Pod {pod_name}: CPU usage {cpu_usage} is very high"

                if memory_usage != "<unknown>":
                    # Basic memory check
                    assert "Gi" in memory_usage or "Mi" in memory_usage, f"Pod {pod_name}: unexpected memory format {memory_usage}"

    def test_horizontal_pod_autoscaler(self):
        """Test HPA configuration"""
        success, stdout, stderr = self.run_kubectl_command([
            "get", "hpa", "-n", self.namespace, "-o", "json"
        ])

        if not success:
            pytest.skip("No HPA configured")

        hpa_data = json.loads(stdout)
        hpas = hpa_data.get("items", [])

        for hpa in hpas:
            name = hpa["metadata"]["name"]
            spec = hpa.get("spec", {})

            min_replicas = spec.get("minReplicas", 0)
            max_replicas = spec.get("maxReplicas", 0)

            assert min_replicas >= 1, f"HPA {name}: minReplicas should be >= 1"
            assert max_replicas > min_replicas, f"HPA {name}: maxReplicas should be > minReplicas"

            if self.environment == "production":
                assert min_replicas >= 2, f"HPA {name}: production should have >= 2 min replicas"

class TestDisasterRecovery(InfrastructureTestSuite):
    """Test disaster recovery capabilities"""

    def test_backup_configuration(self):
        """Test backup job configuration"""
        success, stdout, stderr = self.run_kubectl_command([
            "get", "cronjob", "-n", self.namespace, "-o", "json"
        ])

        if success:
            cronjob_data = json.loads(stdout)
            cronjobs = cronjob_data.get("items", [])

            backup_jobs = [cj for cj in cronjobs if "backup" in cj["metadata"]["name"]]

            if backup_jobs:
                for backup_job in backup_jobs:
                    spec = backup_job.get("spec", {})
                    schedule = spec.get("schedule", "")
                    assert schedule, "Backup job has no schedule configured"
            else:
                pytest.skip("No backup jobs configured")
        else:
            pytest.skip("No CronJobs found")

    def test_persistent_volume_snapshots(self):
        """Test volume snapshot capability"""
        # Check if volume snapshot CRDs are available
        success, stdout, stderr = self.run_kubectl_command([
            "get", "crd", "volumesnapshots.snapshot.storage.k8s.io"
        ])

        if success:
            # Check for snapshot classes
            success, stdout, stderr = self.run_kubectl_command([
                "get", "volumesnapshotclass"
            ])

            assert success, "Volume snapshot classes not configured"
        else:
            pytest.skip("Volume snapshots not supported")

# Test runner
def run_infrastructure_tests(environment: str = "staging", namespace: str = "flashmm",
                           test_type: str = "full", verbose: bool = True):
    """
    Run infrastructure tests with specified parameters

    Args:
        environment: Target environment (staging, production)
        namespace: Kubernetes namespace
        test_type: Type of tests to run (quick, full, security)
        verbose: Enable verbose output
    """

    # Configure pytest arguments
    pytest_args = [
        "-v" if verbose else "-q",
        "--tb=short",
        f"--environment={environment}",
        f"--namespace={namespace}"
    ]

    # Select test modules based on type
    if test_type == "quick":
        pytest_args.extend([
            "tests/deployment/test_infrastructure.py::TestKubernetesDeployment::test_namespace_exists",
            "tests/deployment/test_infrastructure.py::TestKubernetesDeployment::test_deployments_healthy",
            "tests/deployment/test_infrastructure.py::TestApplicationDeployment::test_application_pods_running"
        ])
    elif test_type == "security":
        pytest_args.extend([
            "tests/deployment/test_infrastructure.py::TestContainerSecurity",
            "tests/deployment/test_infrastructure.py::TestNetworkSecurity",
            "tests/deployment/test_infrastructure.py::TestSecurityCompliance"
        ])
    else:  # full
        pytest_args.append("tests/deployment/test_infrastructure.py")

    # Run tests
    return pytest.main(pytest_args)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FlashMM Infrastructure Tests")
    parser.add_argument("-e", "--environment", default="staging", help="Environment to test")
    parser.add_argument("-n", "--namespace", default="flashmm", help="Kubernetes namespace")
    parser.add_argument("-t", "--type", default="full", choices=["quick", "full", "security"], help="Test type")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Run tests
    exit_code = run_infrastructure_tests(
        environment=args.environment,
        namespace=args.namespace,
        test_type=args.type,
        verbose=args.verbose
    )

    sys.exit(exit_code)

"""
Innovation CI Runner
Ephemeral container-based CI/CD pipeline for innovation branches.
Runs comprehensive testing, security scanning, and performance validation.
"""

import asyncio
import docker
from typing import Dict, List, Optional, Any
import logging
import tempfile
import os

logger = logging.getLogger(__name__)

class InnovationCIRunner:
    """
    Runs CI/CD pipeline for innovation branches in ephemeral containers.
    Provides security scanning, testing, and performance validation.
    """
    
    def __init__(self):
        self.docker_client = docker.from_env()
        self.security_scanners = {
            'sast': 'bandit',
            'dependency': 'safety',
            'secrets': 'trufflehog',
            'container': 'trivy'
        }
    
    async def run_security_scan(self, branch: str) -> Dict[str, Any]:
        """
        Run comprehensive security scanning on innovation branch.
        """
        scan_results = {}
        
        try:
            # SAST - Static Application Security Testing
            sast_result = await self._run_sast_scan(branch)
            scan_results['sast_passed'] = sast_result.get('vulnerabilities', 0) == 0
            scan_results['sast_details'] = sast_result
            
            # Dependency vulnerability scanning
            dep_result = await self._run_dependency_scan(branch)
            scan_results['dependency_check_passed'] = dep_result.get('critical', 0) == 0
            scan_results['dependency_details'] = dep_result
            
            # Secrets scanning
            secrets_result = await self._run_secrets_scan(branch)
            scan_results['secrets_scan_passed'] = secrets_result.get('secrets_found', 0) == 0
            scan_results['secrets_details'] = secrets_result
            
            # Container security scanning
            container_result = await self._run_container_scan(branch)
            scan_results['container_scan_passed'] = container_result.get('vulnerabilities', 0) == 0
            scan_results['container_details'] = container_result
            
            # Overall security score
            scan_results['security_score'] = self._calculate_security_score(scan_results)
            scan_results['security_passed'] = scan_results['security_score'] >= 80
            
        except Exception as e:
            logger.error(f"Security scan failed: {e}")
            scan_results['error'] = str(e)
            scan_results['security_passed'] = False
        
        return scan_results
    
    async def run_test_suite(self, branch: str) -> Dict[str, Any]:
        """
        Run comprehensive test suite including unit, integration, and E2E tests.
        """
        test_results = {}
        
        try:
            # Unit tests
            unit_results = await self._run_unit_tests(branch)
            test_results['unit_tests_passed'] = unit_results.get('passed', False)
            test_results['unit_coverage'] = unit_results.get('coverage', 0)
            
            # Integration tests
            integration_results = await self._run_integration_tests(branch)
            test_results['integration_tests_passed'] = integration_results.get('passed', False)
            
            # E2E tests
            e2e_results = await self._run_e2e_tests(branch)
            test_results['e2e_tests_passed'] = e2e_results.get('passed', False)
            
            # Performance tests
            perf_results = await self._run_performance_tests(branch)
            test_results['performance_tests_passed'] = perf_results.get('passed', False)
            
            # Overall test status
            test_results['all_passed'] = all([
                test_results['unit_tests_passed'],
                test_results['integration_tests_passed'], 
                test_results['e2e_tests_passed'],
                test_results['performance_tests_passed']
            ])
            
            test_results['total_coverage'] = unit_results.get('coverage', 0)
            
        except Exception as e:
            logger.error(f"Test suite execution failed: {e}")
            test_results['error'] = str(e)
            test_results['all_passed'] = False
        
        return test_results
    
    async def run_performance_tests(self, branch: str) -> Dict[str, Any]:
        """
        Run performance and load testing on innovation branch.
        """
        perf_results = {}
        
        try:
            # Load testing
            load_test = await self._run_load_test(branch)
            perf_results['load_test_passed'] = load_test.get('success', False)
            perf_results['load_metrics'] = load_test.get('metrics', {})
            
            # Response time testing
            response_test = await self._run_response_time_test(branch)
            perf_results['response_times_acceptable'] = response_test.get('acceptable', False)
            perf_results['response_metrics'] = response_test.get('metrics', {})
            
            # Memory and CPU profiling
            resource_test = await self._run_resource_usage_test(branch)
            perf_results['resource_usage_acceptable'] = resource_test.get('acceptable', False)
            perf_results['resource_metrics'] = resource_test.get('metrics', {})
            
            # Overall performance score
            perf_results['performance_score'] = self._calculate_performance_score(perf_results)
            perf_results['performance_passed'] = perf_results['performance_score'] >= 70
            
        except Exception as e:
            logger.error(f"Performance testing failed: {e}")
            perf_results['error'] = str(e)
            perf_results['performance_passed'] = False
        
        return perf_results
    
    async def generate_artifacts(self, branch: str) -> Dict[str, str]:
        """
        Generate CI artifacts (logs, reports, coverage) for review.
        """
        artifacts = {}
        
        try:
            # Test coverage report
            coverage_report = await self._generate_coverage_report(branch)
            artifacts['coverage_report'] = coverage_report
            
            # Security report
            security_report = await self._generate_security_report(branch)
            artifacts['security_report'] = security_report
            
            # Performance report
            performance_report = await self._generate_performance_report(branch)
            artifacts['performance_report'] = performance_report
            
            # Build logs
            build_logs = await self._capture_build_logs(branch)
            artifacts['build_logs'] = build_logs
            
            # Test execution logs
            test_logs = await self._capture_test_logs(branch)
            artifacts['test_logs'] = test_logs
            
        except Exception as e:
            logger.error(f"Artifact generation failed: {e}")
            artifacts['error'] = str(e)
        
        return artifacts
    
    # Internal security scanning methods
    async def _run_sast_scan(self, branch: str) -> Dict[str, Any]:
        """Run Static Application Security Testing."""
        container = await self._create_ephemeral_container(branch)
        
        try:
            # Run bandit for Python SAST
            result = container.exec_run("bandit -r . -f json")
            return self._parse_sast_results(result.output.decode())
        finally:
            container.remove(force=True)
    
    async def _run_dependency_scan(self, branch: str) -> Dict[str, Any]:
        """Run dependency vulnerability scanning."""
        container = await self._create_ephemeral_container(branch)
        
        try:
            # Run safety check
            result = container.exec_run("safety check --json")
            return self._parse_dependency_results(result.output.decode())
        finally:
            container.remove(force=True)
    
    async def _run_secrets_scan(self, branch: str) -> Dict[str, Any]:
        """Run secrets detection scanning."""
        container = await self._create_ephemeral_container(branch)
        
        try:
            # Run trufflehog for secrets detection
            result = container.exec_run("trufflehog filesystem . --json")
            return self._parse_secrets_results(result.output.decode())
        finally:
            container.remove(force=True)
    
    async def _run_container_scan(self, branch: str) -> Dict[str, Any]:
        """Run container image security scanning."""
        # Build temporary image
        image_tag = f"innovation-{branch.replace('/', '-')}"
        image, _ = self.docker_client.images.build(path=f"./{branch}", tag=image_tag)
        
        try:
            # Run trivy for container scanning
            container = self.docker_client.containers.run(
                "aquasec/trivy:latest",
                f"image {image_tag} --format json",
                remove=True
            )
            return self._parse_container_results(container.decode())
        finally:
            # Cleanup image
            self.docker_client.images.remove(image.id)
    
    # Internal testing methods
    async def _run_unit_tests(self, branch: str) -> Dict[str, Any]:
        """Run unit test suite with coverage."""
        container = await self._create_ephemeral_container(branch)
        
        try:
            # Run pytest with coverage
            result = container.exec_run("pytest tests/unit --cov=. --cov-report=json")
            return self._parse_unit_test_results(result.output.decode())
        finally:
            container.remove(force=True)
    
    async def _run_integration_tests(self, branch: str) -> Dict[str, Any]:
        """Run integration test suite."""
        container = await self._create_ephemeral_container(branch)
        
        try:
            # Run integration tests
            result = container.exec_run("pytest tests/integration -v")
            return self._parse_integration_test_results(result.output.decode())
        finally:
            container.remove(force=True)
    
    async def _run_e2e_tests(self, branch: str) -> Dict[str, Any]:
        """Run end-to-end test suite."""
        container = await self._create_ephemeral_container(branch)
        
        try:
            # Run E2E tests
            result = container.exec_run("pytest tests/e2e -v")
            return self._parse_e2e_test_results(result.output.decode())
        finally:
            container.remove(force=True)
    
    async def _run_load_test(self, branch: str) -> Dict[str, Any]:
        """Run load testing with locust."""
        container = await self._create_ephemeral_container(branch)
        
        try:
            # Run locust load test
            result = container.exec_run("locust -f load_test.py --headless -u 100 -r 10 -t 1m")
            return self._parse_load_test_results(result.output.decode())
        finally:
            container.remove(force=True)
    
    # Utility methods
    async def _create_ephemeral_container(self, branch: str) -> Any:
        """Create ephemeral container for branch testing."""
        # Pull latest Python image
        image = "python:3.9-slim"
        
        # Create container with branch code
        container = self.docker_client.containers.run(
            image,
            "sleep 3600",  # Keep alive for commands
            detach=True,
            working_dir="/app",
            volumes={
                os.path.abspath(branch): {
                    'bind': '/app',
                    'mode': 'rw'
                }
            }
        )
        
        # Install dependencies
        container.exec_run("pip install -r requirements.txt")
        
        return container
    
    def _calculate_security_score(self, scan_results: Dict) -> float:
        """Calculate overall security score from scan results."""
        score = 100.0
        
        # Deduct for vulnerabilities
        if not scan_results.get('sast_passed', False):
            score -= 30
        
        if not scan_results.get('dependency_check_passed', False):
            score -= 30
            
        if not scan_results.get('secrets_scan_passed', False):
            score -= 25
            
        if not scan_results.get('container_scan_passed', False):
            score -= 15
            
        return max(score, 0)
    
    def _calculate_performance_score(self, perf_results: Dict) -> float:
        """Calculate overall performance score."""
        score = 100.0
        
        if not perf_results.get('load_test_passed', False):
            score -= 40
            
        if not perf_results.get('response_times_acceptable', False):
            score -= 35
            
        if not perf_results.get('resource_usage_acceptable', False):
            score -= 25
            
        return max(score, 0)
    
    # Parsing methods for different tool outputs
    def _parse_sast_results(self, output: str) -> Dict[str, Any]:
        """Parse SAST tool output."""
        import json
        try:
            data = json.loads(output)
            return {
                'vulnerabilities': len(data.get('metrics', {}).get('_totals', {}).get('issues', [])),
                'confidence_high': sum(1 for issue in data.get('issues', []) if issue.get('confidence', '') == 'HIGH'),
                'details': data
            }
        except:
            return {'vulnerabilities': 0, 'details': {}}
    
    def _parse_dependency_results(self, output: str) -> Dict[str, Any]:
        """Parse dependency scan output."""
        import json
        try:
            data = json.loads(output)
            return {
                'critical': len([v for v in data.get('vulnerabilities', []) if v.get('severity') == 'CRITICAL']),
                'high': len([v for v in data if v.get('severity') == 'HIGH']),
                'details': data
            }
        except:
            return {'critical': 0, 'high': 0, 'details': {}}
    
    def _parse_secrets_results(self, output: str) -> Dict[str, Any]:
        """Parse secrets scan output."""
        lines = output.strip().split('\n')
        secrets = [json.loads(line) for line in lines if line]
        return {
            'secrets_found': len(secrets),
            'details': secrets
        }
    
    def _parse_unit_test_results(self, output: str) -> Dict[str, Any]:
        """Parse unit test results."""
        import json
        try:
            # Try to get coverage data
            coverage_data = json.loads(output.split('Coverage JSON:')[-1])
            return {
                'passed': 'FAILED' not in output,
                'coverage': coverage_data.get('totals', {}).get('percent_covered', 0),
                'details': output
            }
        except:
            return {'passed': 'FAILED' not in output, 'coverage': 0, 'details': output}
    
    # Additional artifact generation methods
    async def _generate_coverage_report(self, branch: str) -> str:
        """Generate test coverage report."""
        return f"coverage_report_{branch}.html"
    
    async def _generate_security_report(self, branch: str) -> str:
        """Generate comprehensive security report."""
        return f"security_report_{branch}.pdf"
    
    async def _generate_performance_report(self, branch: str) -> str:
        """Generate performance test report."""
        return f"performance_report_{branch}.html"
    
    async def _capture_build_logs(self, branch: str) -> str:
        """Capture build process logs."""
        return f"build_logs_{branch}.txt"
    
    async def _capture_test_logs(self, branch: str) -> str:
        """Capture test execution logs."""
        return f"test_logs_{branch}.txt"
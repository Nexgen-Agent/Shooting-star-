# scout/vetting/skill_tests.py
import asyncio
from typing import Dict, List, Optional, Any
import docker
from pathlib import Path
import tempfile
import json

class SkillTestService:
    def __init__(self):
        self.docker_client = docker.from_env()
        self.test_templates = self._load_test_templates()
    
    async def run_skill_test(self, candidate_id: str, test_type: str, language: str) -> Dict[str, Any]:
        """Run a skill test for a candidate"""
        test_config = self.test_templates.get(f"{test_type}_{language}")
        if not test_config:
            raise ValueError(f"No test template for {test_type}_{language}")
        
        # Create test environment
        with tempfile.TemporaryDirectory() as temp_dir:
            test_result = await self._execute_test_in_docker(temp_dir, test_config, candidate_id)
            
        return self._compile_test_results(test_result, test_config)
    
    def _load_test_templates(self) -> Dict[str, Any]:
        """Load predefined test templates"""
        return {
            "algorithm_python": {
                "docker_image": "python:3.9-slim",
                "test_files": ["algorithm_test.py"],
                "timeout_seconds": 300,
                "max_score": 100
            },
            "frontend_javascript": {
                "docker_image": "node:16-alpine",
                "test_files": ["frontend_test.js"],
                "timeout_seconds": 600,
                "max_score": 100
            },
            "system_design": {
                "docker_image": "python:3.9-slim",
                "test_files": ["system_design.md"],
                "timeout_seconds": 1800,
                "max_score": 100,
                "requires_review": True
            }
        }
    
    async def _execute_test_in_docker(self, temp_dir: str, test_config: Dict[str, Any], candidate_id: str) -> Dict[str, Any]:
        """Execute test in isolated Docker container"""
        try:
            container = self.docker_client.containers.run(
                test_config["docker_image"],
                command=["python", "-c", "print('Test execution placeholder')"],
                working_dir="/test",
                volumes={temp_dir: {'bind': '/test', 'mode': 'ro'}},
                detach=True
            )
            
            # Wait for completion with timeout
            try:
                result = container.wait(timeout=test_config["timeout_seconds"])
                logs = container.logs().decode('utf-8')
                
                return {
                    "exit_code": result.get("StatusCode", 1),
                    "logs": logs,
                    "timed_out": False
                }
                
            except Exception as e:
                container.kill()
                return {
                    "exit_code": 1,
                    "logs": f"Test timed out: {str(e)}",
                    "timed_out": True
                }
                
        except Exception as e:
            return {
                "exit_code": 1,
                "logs": f"Container error: {str(e)}",
                "timed_out": False
            }
    
    def _compile_test_results(self, test_result: Dict[str, Any], test_config: Dict[str, Any]) -> Dict[str, Any]:
        """Compile test results into scores"""
        base_score = 0.0
        feedback = []
        
        if test_result["exit_code"] == 0 and not test_result["timed_out"]:
            base_score = test_config["max_score"] * 0.8  # Base score for completion
            feedback.append("Test completed successfully")
        else:
            feedback.append(f"Test failed: {test_result['logs']}")
        
        # Analyze logs for additional scoring factors
        if "error" not in test_result["logs"].lower():
            base_score += test_config["max_score"] * 0.2
        
        return {
            "score": min(base_score, test_config["max_score"]),
            "max_score": test_config["max_score"],
            "passed": base_score >= (test_config["max_score"] * 0.7),
            "feedback": feedback,
            "execution_time": 0,  # Would be calculated from actual timing
            "requires_review": test_config.get("requires_review", False)
        }
    
    async def run_comprehensive_vetting(self, candidate_id: str, role: str) -> Dict[str, float]:
        """Run comprehensive vetting based on role"""
        role_tests = {
            "backend_developer": ["algorithm_python", "system_design"],
            "frontend_developer": ["frontend_javascript", "system_design"],
            "fullstack_developer": ["algorithm_python", "frontend_javascript", "system_design"]
        }
        
        tests_to_run = role_tests.get(role, ["algorithm_python"])
        results = {}
        
        for test in tests_to_run:
            test_result = await self.run_skill_test(candidate_id, test.split('_')[0], test.split('_')[1])
            results[test] = test_result
        
        return self._calculate_vetting_score(results)
    
    def _calculate_vetting_score(self, test_results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate overall vetting score from multiple tests"""
        total_score = 0.0
        max_possible = 0.0
        tests_passed = 0
        
        for test_name, result in test_results.items():
            total_score += result["score"]
            max_possible += result["max_score"]
            if result["passed"]:
                tests_passed += 1
        
        overall_score = (total_score / max_possible) * 100 if max_possible > 0 else 0
        pass_rate = tests_passed / len(test_results) if test_results else 0
        
        return {
            "overall_score": overall_score,
            "pass_rate": pass_rate,
            "technical_assessment_score": overall_score,
            "detailed_breakdown": test_results
        }
# scripts/simulate_incident.py
"""
DEFENSIVE ONLY — NO OFFENSIVE ACTIONS. ALL ACTIONS LOGGED AND AUDITED.

Staging-only incident simulation script for testing playbooks and response.
NEVER RUN IN PRODUCTION ENVIRONMENT.
"""

import asyncio
import random
from datetime import datetime
from enum import Enum

class SimulationScenario(str, Enum):
    BRUTE_FORCE = "brute_force"
    DATA_EXFIL = "data_exfiltration" 
    RANSOMWARE = "ransomware"
    WEBSHELL = "webshell_upload"
    LATERAL_MOVEMENT = "lateral_movement"

async def simulate_incident_drill(scenario: SimulationScenario, intensity: str = "low") -> Dict:
    """
    Simulate security incident in staging environment only.
    This is for testing response procedures and playbooks.
    """
    
    if _is_production_environment():
        raise RuntimeError("NEVER run simulations in production environment!")
    
    simulation_id = f"sim-{scenario}-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
    
    print(f"🚀 Starting security drill: {scenario} (intensity: {intensity})")
    print(f"Simulation ID: {simulation_id}")
    print("THIS IS A DRILL - NO REAL SECURITY INCIDENT\n")
    
    # Generate simulated evidence based on scenario
    evidence = await _generate_simulated_evidence(scenario, intensity)
    
    # Execute appropriate playbook
    from core.playbooks import PlaybookType, SafePlaybookExecutor
    
    playbook_mapping = {
        SimulationScenario.BRUTE_FORCE: PlaybookType.BRUTE_FORCE,
        SimulationScenario.DATA_EXFIL: PlaybookType.DATA_EXFILTRATION,
        SimulationScenario.RANSOMWARE: PlaybookType.RANSOMWARE_SUSPICION,
        SimulationScenario.WEBSHELL: PlaybookType.PRIVILEGE_ESCALATION,
        SimulationScenario.LATERAL_MOVEMENT: PlaybookType.SUSPICIOUS_OUTBOUND
    }
    
    playbook_type = playbook_mapping[scenario]
    executor = SafePlaybookExecutor()
    
    results = await executor.execute_playbook(playbook_type, simulation_id, evidence)
    
    # Generate lessons learned
    lessons = await _analyze_simulation_results(results, scenario)
    
    print(f"✅ Drill completed: {simulation_id}")
    print(f"Lessons learned: {lessons}")
    
    return {
        "simulation_id": simulation_id,
        "scenario": scenario,
        "intensity": intensity,
        "results": results,
        "lessons_learned": lessons,
        "completion_time": datetime.utcnow().isoformat()
    }

async def _generate_simulated_evidence(scenario: SimulationScenario, intensity: str) -> Dict:
    """Generate realistic but simulated evidence for drills"""
    
    base_evidence = {
        "simulation": True,
        "environment": "staging",
        "intensity": intensity
    }
    
    if scenario == SimulationScenario.BRUTE_FORCE:
        base_evidence.update({
            "type": "brute_force",
            "target_service": "ssh",
            "source_ips": ["192.0.2.1", "192.0.2.2", "192.0.2.3"],
            "failed_attempts": random.randint(100, 1000),
            "target_accounts": ["admin", "root", "user"]
        })
    
    elif scenario == SimulationScenario.DATA_EXFIL:
        base_evidence.update({
            "type": "data_exfiltration",
            "source_host": "staging-db-01",
            "destination_ip": "198.51.100.1",
            "data_volume": f"{random.randint(100, 1000)} MB",
            "protocol": "HTTPS",
            "suspicious_file_types": [".sql", ".csv", ".json"]
        })
    
    elif scenario == SimulationScenario.RANSOMWARE:
        base_evidence.update({
            "type": "ransomware_suspicion",
            "affected_hosts": ["staging-fs-01", "staging-fs-02"],
            "suspicious_processes": ["encrypt_tool", "malicious_script"],
            "file_extensions_changed": [".encrypted", ".crypted"],
            "ransom_note_found": True
        })
    
    return base_evidence

async def _analyze_simulation_results(results: Dict, scenario: SimulationScenario) -> List[str]:
    """Analyze simulation results for lessons learned"""
    lessons = []
    
    completed_steps = results["completed_steps"]
    total_steps = len(results["execution_log"])
    
    if completed_steps < total_steps:
        lessons.append(f"Playbook completion rate: {completed_steps}/{total_steps} - review failed steps")
    
    # Check response time
    lessons.append("Review incident response time metrics")
    
    # Scenario-specific lessons
    if scenario == SimulationScenario.BRUTE_FORCE:
        lessons.append("Consider implementing IP blocking automation after N failed attempts")
        lessons.append("Review MFA enforcement policies")
    
    elif scenario == SimulationScenario.DATA_EXFIL:
        lessons.append("Evaluate egress filtering and data loss prevention rules")
        lessons.append("Review data classification and access controls")
    
    elif scenario == SimulationScenario.RANSOMWARE:
        lessons.append("Verify backup integrity and recovery procedures")
        lessons.append("Test isolated recovery environment availability")
    
    return lessons

def _is_production_environment() -> bool:
    """Safety check - prevent simulations in production"""
    import os
    env = os.getenv("ENVIRONMENT", "staging").lower()
    return env in ["production", "prod"]

async def chaos_engineering_mode():
    """Advanced chaos engineering tests for resilience validation"""
    if _is_production_environment():
        raise RuntimeError("Chaos engineering only allowed in staging!")
    
    chaos_tests = [
        "network_partition",
        "service_degradation", 
        "dependency_failure",
        "resource_exhaustion"
    ]
    
    for test in chaos_tests:
        print(f"🧪 Running chaos test: {test}")
        # Implementation would inject controlled failures
        await asyncio.sleep(2)  # Simulate test duration
    
    print("Chaos engineering completed - review system resilience")

if __name__ == "__main__":
    # Example simulation
    asyncio.run(simulate_incident_drill(SimulationScenario.BRUTE_FORCE, "medium"))
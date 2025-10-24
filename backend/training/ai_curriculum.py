# training/ai_curriculum.py
"""
COMPREHENSIVE TRAINING CURRICULUM FOR CYBER DEFENSE AI
"""

class AICyberCurriculum:
    def __init__(self):
        self.training_modules = self._build_curriculum()
    
    def _build_curriculum(self):
        return {
            "phase_1_fundamentals": {
                "network_attacks": ["ddos", "man_in_the_middle", "arp_spoofing"],
                "web_attacks": ["sql_injection", "xss", "csrf", "file_inclusion"],
                "system_attacks": ["buffer_overflow", "privilege_escalation", "rootkits"]
            },
            "phase_2_advanced": {
                "advanced_persistent_threats": ["apt1", "apt29", "apt34"],
                "ransomware_families": ["wannacry", "notpetya", "revil"],
                "supply_chain_attacks": ["solarwinds", "codecov", "kaseya"]
            },
            "phase_3_ai_vs_ai": {
                "adversarial_machine_learning": ["model_poisoning", "evasion_attacks"],
                "ai_powered_attacks": ["deepfake_social_engineering", "ai_generated_malware"],
                "autonomous_attack_systems": ["self_propagating_worms", "ai_red_teams"]
            }
        }
    
    async def train_ai_comprehensive(self):
        """Complete training curriculum for defense AI"""
        for phase_name, modules in self.training_modules.items():
            print(f"🎓 Training Phase: {phase_name}")
            
            for module_name, attacks in modules.items():
                success_rate = await self._train_module(module_name, attacks)
                
                if success_rate < 0.9:
                    await self._remedial_training(module_name, attacks)
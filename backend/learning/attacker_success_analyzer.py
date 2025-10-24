# learning/attacker_success_analyzer.py
"""
ATTACKER SUCCESS ANALYZER - LEARNS FROM EVERY ATTACKER VICTORY
Turns attacker wins into defense evolution opportunities.
"""

class AttackerSuccessAnalyzer:
    def __init__(self):
        self.attacker_victories = []
        self.defense_breach_analysis = {}
        self.countermeasure_development = {}
    
    async def analyze_attacker_victory(self, successful_attack: Dict):
        """Deep analysis of every successful attack"""
        print(f"🔍 Analyzing attacker victory: {successful_attack['type']}")
        
        # 1. Attack Chain Reconstruction
        attack_chain = await self._reconstruct_attack_chain(successful_attack)
        
        # 2. Defense Failure Points
        failure_points = await self._identify_defense_failures(attack_chain)
        
        # 3. Attacker Innovation Detection
        innovations = await self._detect_attacker_innovations(attack_chain)
        
        # 4. Develop Counter-Innovations
        counter_innovations = await self._develop_counter_innovations(innovations)
        
        # 5. Update All Defenses
        await self._update_all_defense_layers(failure_points, counter_innovations)
        
        return {
            "lessons_learned": len(counter_innovations),
            "defense_improvements": await self._count_improvements(),
            "future_prevention_rate": await self._calculate_prevention_rate()
        }
    
    async def _reconstruct_attack_chain(self, attack):
        """Reconstruct exactly how the attack succeeded"""
        return {
            "initial_access_method": await self._analyze_access_method(attack),
            "defense_evasion_techniques": await self._analyze_evasion_tactics(attack),
            "lateral_movement_path": await self._map_movement_path(attack),
            "persistence_mechanisms": await self._identify_persistence(attack),
            "data_exfiltration_method": await self._analyze_exfiltration(attack)
        }
    
    async def _identify_defense_failures(self, attack_chain):
        """Identify exactly where defenses failed"""
        failures = []
        
        # Check each defense layer
        for layer_name, layer in self.defense_layers.items():
            effectiveness = await self._test_layer_against_attack(layer, attack_chain)
            if effectiveness < 1.0:
                failures.append({
                    "layer": layer_name,
                    "failure_point": await self._pinpoint_failure(layer, attack_chain),
                    "severity": 1.0 - effectiveness,
                    "improvement_opportunity": await self._suggest_improvement(layer)
                })
        
        return failures
    
    async def _detect_attacker_innovations(self, attack_chain):
        """Detect novel techniques used by attackers"""
        innovations = []
        
        # Compare with known techniques
        known_techniques = await self.knowledge_base.get_known_techniques()
        
        for technique in attack_chain['techniques']:
            if not await self._is_known_technique(technique, known_techniques):
                innovations.append({
                    "novel_technique": technique,
                    "detection_gap": await self._identify_detection_gap(technique),
                    "prevention_gap": await self._identify_prevention_gap(technique)
                })
        
        return innovations
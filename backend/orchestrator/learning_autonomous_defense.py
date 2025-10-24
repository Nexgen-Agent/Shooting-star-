# orchestrator/learning_autonomous_defense.py
"""
ENHANCED AUTONOMOUS DEFENSE WITH HUMAN & ATTACKER LEARNING
The complete system that never stops learning.
"""

class LearningAutonomousDefense:
    def __init__(self):
        self.mistake_learner = MistakeLearningEngine()
        self.attacker_learner = AttackerSuccessAnalyzer()
        self.incident_learner = RealWorldIncidentLearner()
        self.defender_analyzer = DefenderBehaviorAnalyzer()
        self.psychology_learner = AttackerPsychologyLearner()
        self.improvement_engine = ContinuousImprovementEngine()
        self.evolution_engine = MistakeDrivenEvolution()
        
        self.learning_effectiveness = 0.0
        self.defense_maturity = 0.0
    
    async def start_complete_learning_system(self):
        """Start the complete learning-based defense system"""
        print("🧠 Starting Learning-Based Autonomous Defense")
        
        # Start all learning components
        learning_tasks = [
            self.mistake_learner.analyze_human_mistakes(),
            self.incident_learner.continuous_incident_learning(),
            self.defender_analyzer.analyze_defender_patterns(),
            self.psychology_learner.study_attacker_psychology(),
            self.improvement_engine.run_improvement_cycles(),
            self.evolution_engine.evolve_from_mistakes(),
            self._track_learning_effectiveness()
        ]
        
        await asyncio.gather(*learning_tasks)
    
    async def _track_learning_effectiveness(self):
        """Track how effectively the system is learning"""
        while True:
            # Calculate learning effectiveness
            self.learning_effectiveness = await self._calculate_learning_effectiveness()
            
            # Calculate defense maturity
            self.defense_maturity = await self._calculate_defense_maturity()
            
            # Print learning progress
            print(f"📊 Learning Effectiveness: {self.learning_effectiveness:.1%}")
            print(f"🛡️ Defense Maturity: {self.defense_maturity:.1%}")
            
            # If learning effectiveness drops, trigger interventions
            if self.learning_effectiveness < 0.8:
                await self._trigger_learning_interventions()
            
            await asyncio.sleep(3600)  # Check hourly
    
    async def handle_attack_with_learning(self, attack_data: Dict) -> Dict:
        """Handle attack while simultaneously learning from it"""
        # 1. Defend against attack
        defense_result = await self._execute_defense(attack_data)
        
        # 2. Learn regardless of outcome
        if defense_result['success']:
            await self._learn_from_successful_defense(attack_data, defense_result)
        else:
            await self._learn_from_failed_defense(attack_data, defense_result)
        
        # 3. Extract maximum learning value
        learning_value = await self._extract_learning_value(attack_data, defense_result)
        
        # 4. Update all learning systems
        await self._update_learning_systems(attack_data, defense_result, learning_value)
        
        return {
            **defense_result,
            "learning_extracted": learning_value,
            "system_improved": True
        }

# Start the ultimate learning system
async def main():
    defense_system = LearningAutonomousDefense()
    await defense_system.start_complete_learning_system()

if __name__ == "__main__":
    asyncio.run(main())
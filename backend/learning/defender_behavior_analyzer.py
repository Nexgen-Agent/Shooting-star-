# learning/defender_behavior_analyzer.py
"""
DEFENDER BEHAVIOR ANALYZER - LEARNS FROM SECURITY TEAM PATTERNS
Identifies common mistakes and builds automated safeguards.
"""

class DefenderBehaviorAnalyzer:
    def __init__(self):
        self.team_behavior_db = TeamBehaviorDatabase()
        self.mistake_patterns = MistakePatterns()
        self.automation_builder = AutomationBuilder()
    
    async def analyze_defender_patterns(self):
        """Analyze security team behavior for improvement opportunities"""
        # 1. Analyze alert response patterns
        response_patterns = await self._analyze_alert_responses()
        
        # 2. Identify common investigation mistakes
        investigation_issues = await self._find_investigation_mistakes()
        
        # 3. Study tool usage inefficiencies
        tool_inefficiencies = await self._find_tool_issues()
        
        # 4. Detect knowledge gaps
        knowledge_gaps = await self._identify_knowledge_gaps()
        
        # 5. Build automated assistants
        await self._build_automated_assistants(
            response_patterns, 
            investigation_issues,
            tool_inefficiencies,
            knowledge_gaps
        )
    
    async def _analyze_alert_responses(self):
        """Analyze how security team responds to alerts"""
        patterns = {
            "false_positive_patterns": await self._find_false_positive_tendencies(),
            "missed_true_positives": await self._find_missed_detections(),
            "response_time_issues": await self._analyze_response_times(),
            "escalation_patterns": await self._study_escalation_behavior()
        }
        
        return patterns
    
    async def _find_investigation_mistakes(self):
        """Find common investigation errors"""
        return [
            "premature_closure": await self._find_premature_investigations(),
            "tunnel_vision": await self._detect_investigation_bias(),
            "evidence_mishandling": await self._find_evidence_issues(),
            "scope_failures": await self._find_scope_mistakes()
        ]
    
    async def _build_automated_assistants(self, patterns, issues, inefficiencies, gaps):
        """Build AI assistants to prevent common mistakes"""
        assistants = []
        
        # Alert Triage Assistant
        if patterns['false_positive_patterns']:
            assistants.append(await self._build_triage_assistant(patterns))
        
        # Investigation Guide Assistant  
        if issues['premature_closure'] or issues['tunnel_vision']:
            assistants.append(await self._build_investigation_assistant(issues))
        
        # Tool Optimization Assistant
        if inefficiencies:
            assistants.append(await self._build_tool_assistant(inefficiencies))
        
        # Knowledge Gap Filler
        if gaps:
            assistants.append(await self._build_knowledge_assistant(gaps))
        
        return assistants
"""
Autonomous Innovation Engine (AIE) "Forge"
Core engine that orchestrates AI-driven feature development with secure human gating.
Integrates with private ledger, crypto signatures, and scout services for full lifecycle management.
"""

import asyncio
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging

from core.private_ledger import PrivateLedger
from services.scout_service import ScoutService
from crypto.key_manager import KeyManager
from config.settings import INNOVATION_SETTINGS
from ai.innovation_task_manager import InnovationTaskManager
from scout.innovation_recruiter import InnovationRecruiter
from ci.innovation_ci_runner import InnovationCIRunner

logger = logging.getLogger(__name__)

@dataclass
class InnovationProposal:
    proposal_id: str
    summary: str
    cost_estimate: float
    tasks: List[Dict]
    spec: Dict
    created_by: str
    created_at: datetime
    status: str = "draft"  # draft, recruiting, development, testing, approval_pending, approved, merged

class AutonomousInnovationEngine:
    """
    Main AIE engine that coordinates autonomous feature development
    with cryptographic founder approval gates.
    """
    
    def __init__(self):
        self.ledger = PrivateLedger()
        self.key_manager = KeyManager()
        self.task_manager = InnovationTaskManager()
        self.recruiter = InnovationRecruiter()
        self.ci_runner = InnovationCIRunner()
        self.scout_service = ScoutService()
        self.active_proposals: Dict[str, InnovationProposal] = {}
        
    async def scan_and_prototype(self, spec: Dict) -> Dict[str, Any]:
        """
        Scan codebase and create prototype based on specification.
        Returns staging details and metrics.
        """
        try:
            # Generate unique proposal ID
            proposal_id = f"innov_{uuid.uuid4().hex[:12]}"
            branch_name = f"innov/{proposal_id}"
            
            # Log initiation
            await self.ledger.log_innovation_event(
                event_type="prototype_initiated",
                proposal_id=proposal_id,
                actor="aie_engine",
                metadata={"spec": spec, "branch": branch_name}
            )
            
            # Analyze codebase for integration points
            integration_analysis = await self._analyze_codebase_integration(spec)
            
            # Create prototype branch
            staging_url = await self._create_prototype_branch(branch_name, spec)
            
            # Run initial metrics
            metrics = await self._calculate_prototype_metrics(spec, integration_analysis)
            
            # Store proposal
            proposal = InnovationProposal(
                proposal_id=proposal_id,
                summary=spec.get('summary', ''),
                cost_estimate=metrics.get('cost_estimate', 0),
                tasks=[],
                spec=spec,
                created_by="aie_engine",
                created_at=datetime.utcnow()
            )
            self.active_proposals[proposal_id] = proposal
            
            return {
                "proposal_id": proposal_id,
                "branch": branch_name,
                "staging_url": staging_url,
                "metrics": metrics
            }
            
        except Exception as e:
            logger.error(f"Prototype creation failed: {e}")
            await self.ledger.log_innovation_event(
                event_type="prototype_failed",
                proposal_id=proposal_id,
                actor="aie_engine",
                metadata={"error": str(e)}
            )
            raise
    
    async def generate_feature_proposal(self, spec: Dict) -> Dict[str, Any]:
        """
        Generate detailed feature proposal with cost estimates and task breakdown.
        """
        proposal_id = f"prop_{uuid.uuid4().hex[:12]}"
        
        # Analyze requirements and generate tasks
        tasks = await self.task_manager.analyze_requirements(spec)
        
        # Calculate cost estimate
        cost_estimate = await self._calculate_cost_estimate(tasks)
        
        # Generate summary
        summary = await self._generate_proposal_summary(spec, tasks, cost_estimate)
        
        # Store proposal
        proposal = InnovationProposal(
            proposal_id=proposal_id,
            summary=summary,
            cost_estimate=cost_estimate,
            tasks=tasks,
            spec=spec,
            created_by="aie_engine",
            created_at=datetime.utcnow()
        )
        self.active_proposals[proposal_id] = proposal
        
        await self.ledger.log_innovation_event(
            event_type="proposal_created",
            proposal_id=proposal_id,
            actor="aie_engine",
            metadata={"summary": summary, "cost_estimate": cost_estimate}
        )
        
        return {
            "summary": summary,
            "cost_estimate": cost_estimate,
            "tasks": tasks
        }
    
    async def create_task_bundle(self, proposal_id: str) -> Dict[str, List[str]]:
        """
        Convert proposal into executable task bundle with dependencies.
        """
        if proposal_id not in self.active_proposals:
            raise ValueError(f"Proposal {proposal_id} not found")
            
        proposal = self.active_proposals[proposal_id]
        task_bundle = await self.task_manager.create_task_bundle(proposal.tasks)
        
        proposal.status = "recruiting"
        await self.ledger.log_innovation_event(
            event_type="task_bundle_created",
            proposal_id=proposal_id,
            actor="aie_engine",
            metadata={"task_count": len(task_bundle.get('task_ids', []))}
        )
        
        return task_bundle
    
    async def recruit_experts(self, task_ids: List[str], criteria: Dict) -> Dict[str, List[Dict]]:
        """
        Recruit Python experts based on task requirements and skill tags.
        """
        candidates = await self.recruiter.find_candidates(task_ids, criteria)
        
        # Auto-send NDAs to qualified candidates
        qualified_candidates = []
        for candidate in candidates.get('candidates', []):
            if candidate.get('vetting_score', 0) >= criteria.get('min_score', 80):
                nda_status = await self.recruiter.send_nda_agreement(candidate['id'])
                if nda_status.get('accepted', False):
                    qualified_candidates.append(candidate)
        
        return {"candidates": qualified_candidates}
    
    async def assign_tasks(self, task_ids: List[str], candidate_ids: List[str]):
        """
        Assign tasks to selected candidates with proper scope and permissions.
        """
        assignments = await self.task_manager.assign_tasks(task_ids, candidate_ids)
        
        # Log assignments in ledger
        for assignment in assignments:
            await self.ledger.log_innovation_event(
                event_type="task_assigned",
                proposal_id=assignment['proposal_id'],
                actor="aie_engine",
                metadata=assignment
            )
    
    async def run_staging_pipeline(self, branch: str) -> Dict[str, Any]:
        """
        Execute full CI/CD pipeline for staging branch with security scanning.
        """
        # Run security and quality checks
        security_report = await self.ci_runner.run_security_scan(branch)
        test_results = await self.ci_runner.run_test_suite(branch)
        perf_report = await self.ci_runner.run_performance_tests(branch)
        
        # Check if all critical security tests passed
        security_passed = all(
            security_report.get(check, False) 
            for check in ['sast_passed', 'dependency_check_passed', 'secrets_scan_passed']
        )
        
        return {
            "tests_passed": test_results.get('all_passed', False),
            "security_report": security_report,
            "perf_report": perf_report,
            "pipeline_success": security_passed and test_results.get('all_passed', False)
        }
    
    async def request_production_approval(self, proposal_id: str) -> Dict[str, Any]:
        """
        Request Founder approval with cryptographic signature requirement.
        """
        if proposal_id not in self.active_proposals:
            raise ValueError(f"Proposal {proposal_id} not found")
            
        proposal = self.active_proposals[proposal_id]
        proposal.status = "approval_pending"
        
        # Generate approval request with hash
        approval_hash = await self._generate_approval_hash(proposal)
        
        # Store in ledger awaiting signature
        await self.ledger.log_innovation_event(
            event_type="approval_requested",
            proposal_id=proposal_id,
            actor="aie_engine",
            metadata={
                "approval_hash": approval_hash,
                "proposal_summary": proposal.summary,
                "cost_estimate": proposal.cost_estimate
            }
        )
        
        return {
            "approval_required": True,
            "approval_hash": approval_hash,
            "founder_key_required": True,
            "proposal_id": proposal_id
        }
    
    async def apply_founder_approval(self, proposal_id: str, signature: str) -> Dict[str, Any]:
        """
        Apply Founder cryptographic approval to merge and deploy.
        Only proceeds with valid signature verification.
        """
        # Verify founder signature
        is_valid = await self.key_manager.verify_founder_signature(
            proposal_id, signature
        )
        
        if not is_valid:
            await self.ledger.log_innovation_event(
                event_type="approval_rejected",
                proposal_id=proposal_id,
                actor="founder",
                metadata={"reason": "Invalid signature"}
            )
            raise SecurityError("Invalid founder signature")
        
        # Merge to main and deploy
        merge_result = await self._merge_to_main(proposal_id)
        deploy_result = await self._deploy_to_production(proposal_id)
        
        proposal = self.active_proposals[proposal_id]
        proposal.status = "merged"
        
        await self.ledger.log_innovation_event(
            event_type="proposal_approved_and_merged",
            proposal_id=proposal_id,
            actor="founder",
            metadata={
                "merge_result": merge_result,
                "deploy_result": deploy_result,
                "signature_verified": True
            }
        )
        
        return {
            "merged": True,
            "deployed": deploy_result.get('success', False),
            "proposal_id": proposal_id
        }
    
    # Internal methods
    async def _analyze_codebase_integration(self, spec: Dict) -> Dict:
        """Analyze existing codebase for integration points."""
        return {
            "integration_points": [],
            "dependencies": [],
            "conflict_analysis": {}
        }
    
    async def _create_prototype_branch(self, branch_name: str, spec: Dict) -> str:
        """Create prototype branch and return staging URL."""
        # Implementation would integrate with git and deployment system
        return f"https://staging-{branch_name}.shootingstar.ai"
    
    async def _calculate_prototype_metrics(self, spec: Dict, analysis: Dict) -> Dict:
        """Calculate development metrics and cost estimates."""
        return {
            "cost_estimate": 0.0,
            "development_hours": 0,
            "complexity_score": 0,
            "risk_assessment": "low"
        }
    
    async def _calculate_cost_estimate(self, tasks: List[Dict]) -> float:
        """Calculate total cost estimate from tasks."""
        return sum(task.get('cost_estimate', 0) for task in tasks)
    
    async def _generate_proposal_summary(self, spec: Dict, tasks: List[Dict], cost: float) -> str:
        """Generate human-readable proposal summary."""
        return f"Feature proposal: {spec.get('title', 'Unknown')} - Estimated cost: ${cost:.2f}"
    
    async def _generate_approval_hash(self, proposal: InnovationProposal) -> str:
        """Generate cryptographic hash for approval request."""
        return await self.key_manager.generate_approval_hash(proposal.proposal_id)
    
    async def _merge_to_main(self, proposal_id: str) -> Dict:
        """Merge approved changes to main branch."""
        return {"success": True, "merge_commit": "abc123"}
    
    async def _deploy_to_production(self, proposal_id: str) -> Dict:
        """Deploy merged changes to production."""
        return {"success": True, "deployment_id": "dep_123"}

class SecurityError(Exception):
    """Security violation in innovation engine."""
    pass
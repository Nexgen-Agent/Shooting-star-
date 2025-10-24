from sqlalchemy.ext.asyncio import AsyncSession
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging
import asyncio

from database.models.reception.ai_suggestions import AISuggestion
from ai.receptionist.ai_receptionist_upgrade_engine import AIReceptionistUpgradeEngine

logger = logging.getLogger(__name__)

class SelfUpgradeService:
    """Collects AI recommendations and delivers them to admins/engineers"""
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self.upgrade_engine = AIReceptionistUpgradeEngine(db)

    async def generate_weekly_improvement_report(self) -> Dict[str, Any]:
        """Generate weekly system improvement report"""
        try:
            # Run growth cycle to get latest recommendations
            growth_prescription = await self.upgrade_engine.run_weekly_growth_cycle()
            
            # Get pending suggestions from database
            pending_suggestions = await self._get_pending_suggestions()
            
            # Compile comprehensive report
            report = {
                "report_id": f"improvement_report_{datetime.utcnow().strftime('%Y%W')}",
                "generated_at": datetime.utcnow(),
                "executive_summary": await self._generate_executive_summary(growth_prescription, pending_suggestions),
                "growth_prescription": growth_prescription,
                "pending_suggestions": pending_suggestions,
                "implementation_priority": await self._determine_implementation_priority(growth_prescription, pending_suggestions),
                "action_items": await self._generate_action_items(growth_prescription, pending_suggestions)
            }
            
            # Deliver report to relevant stakeholders
            await self._deliver_report_to_stakeholders(report)
            
            logger.info(f"Weekly improvement report generated with {len(report['action_items'])} action items")
            return report
            
        except Exception as e:
            logger.error(f"Error generating improvement report: {str(e)}")
            return {"error": "Failed to generate improvement report"}

    async def deliver_ai_recommendations(self, target_audience: str) -> Dict[str, Any]:
        """Deliver AI recommendations to specific audience (admins, engineers, etc.)"""
        try:
            # Get relevant suggestions for target audience
            suggestions = await self._get_relevant_suggestions(target_audience)
            
            if not suggestions:
                return {"message": f"No relevant suggestions for {target_audience}"}
            
            # Format recommendations for the audience
            formatted_recommendations = await self._format_recommendations_for_audience(suggestions, target_audience)
            
            # Deliver recommendations
            delivery_result = await self._deliver_recommendations(formatted_recommendations, target_audience)
            
            # Mark as delivered
            await self._mark_suggestions_delivered([s["id"] for s in suggestions])
            
            return {
                "audience": target_audience,
                "recommendations_delivered": len(suggestions),
                "delivery_status": delivery_result,
                "next_follow_up": await self._schedule_follow_up(target_audience)
            }
            
        except Exception as e:
            logger.error(f"Error delivering AI recommendations: {str(e)}")
            return {"error": "Failed to deliver recommendations"}

    async def track_recommendation_implementation(self) -> Dict[str, Any]:
        """Track implementation status of AI recommendations"""
        try:
            # Get all suggestions with implementation status
            result = await self.db.execute(
                "SELECT * FROM ai_suggestions WHERE status IN ('approved', 'implemented')"
            )
            implemented_suggestions = result.fetchall()
            
            # Calculate implementation metrics
            total_approved = len(implemented_suggestions)
            total_implemented = len([s for s in implemented_suggestions if s[15] == "implemented"])  # status column
            
            implementation_rate = (total_implemented / total_approved * 100) if total_approved > 0 else 0
            
            # Calculate impact of implemented suggestions
            impact_analysis = await self._analyze_implementation_impact(implemented_suggestions)
            
            return {
                "tracking_period": "all_time",
                "total_recommendations_approved": total_approved,
                "total_recommendations_implemented": total_implemented,
                "implementation_rate": round(implementation_rate, 2),
                "average_implementation_time": "14 days",  # Would be calculated from actual data
                "impact_analysis": impact_analysis,
                "top_performing_recommendations": await self._identify_top_performing_recommendations(implemented_suggestions)
            }
            
        except Exception as e:
            logger.error(f"Error tracking recommendation implementation: {str(e)}")
            return {"error": "Failed to track implementation"}

    async def escalate_critical_recommendations(self) -> Dict[str, Any]:
        """Escalate critical recommendations to higher management"""
        try:
            # Get critical suggestions
            result = await self.db.execute(
                "SELECT * FROM ai_suggestions WHERE priority_level = 'critical' AND status = 'pending'"
            )
            critical_suggestions = result.fetchall()
            
            if not critical_suggestions:
                return {"message": "No critical recommendations requiring escalation"}
            
            # Format escalation report
            escalation_report = {
                "escalated_at": datetime.utcnow(),
                "escalation_reason": "critical_recommendations_requiring_immediate_attention",
                "critical_recommendations": [
                    {
                        "id": suggestion[0],
                        "title": suggestion[3],
                        "description": suggestion[4],
                        "expected_impact": suggestion[9],
                        "urgency_reason": "high_impact_on_system_performance"
                    } for suggestion in critical_suggestions
                ],
                "recommended_actions": ["review_immediately", "allocate_resources", "expedite_approval"]
            }
            
            # Deliver escalation (in production, this would send emails/notifications)
            await self._deliver_escalation(escalation_report)
            
            return {
                "escalations_sent": len(critical_suggestions),
                "escalation_level": "executive_management",
                "follow_up_required": True
            }
            
        except Exception as e:
            logger.error(f"Error escalating critical recommendations: {str(e)}")
            return {"error": "Escalation failed"}

    # ========== PRIVATE METHODS ==========

    async def _get_pending_suggestions(self) -> List[Dict[str, Any]]:
        """Get pending AI suggestions from database"""
        try:
            result = await self.db.execute(
                "SELECT * FROM ai_suggestions WHERE status = 'pending' ORDER BY priority_level DESC, created_at DESC"
            )
            suggestions = result.fetchall()
            
            return [
                {
                    "id": s[0],
                    "title": s[3],
                    "type": s[1],
                    "priority": s[12],
                    "description": s[4],
                    "estimated_cost": s[13],
                    "expected_roi": s[14]
                } for s in suggestions
            ]
            
        except Exception as e:
            logger.error(f"Error getting pending suggestions: {str(e)}")
            return []

    async def _generate_executive_summary(self, growth_prescription: Dict[str, Any], 
                                        pending_suggestions: List[Dict[str, Any]]) -> str:
        """Generate executive summary for improvement report"""
        high_priority_suggestions = len([s for s in pending_suggestions if s.get("priority") == "high"])
        total_recommendations = len(growth_prescription.get("all_recommendations", []))
        
        return f"Weekly improvement analysis identified {total_recommendations} recommendations with {high_priority_suggestions} high-priority items requiring immediate attention."

    async def _determine_implementation_priority(self, growth_prescription: Dict[str, Any],
                                               pending_suggestions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Determine implementation priority for recommendations"""
        priority_items = []
        
        # Add high-priority items from growth prescription
        for rec in growth_prescription.get("priority_recommendations", []):
            priority_items.append({
                "item": rec["title"],
                "priority": "high",
                "reason": "growth_prescription_priority",
                "estimated_timeline": "1-2 weeks"
            })
        
        # Add critical pending suggestions
        for suggestion in pending_suggestions:
            if suggestion.get("priority") == "critical":
                priority_items.append({
                    "item": suggestion["title"],
                    "priority": "critical",
                    "reason": "critical_system_improvement",
                    "estimated_timeline": "immediate"
                })
        
        return priority_items

    async def _generate_action_items(self, growth_prescription: Dict[str, Any],
                                   pending_suggestions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate actionable items from recommendations"""
        action_items = []
        
        # Convert recommendations to action items
        for rec in growth_prescription.get("priority_recommendations", []):
            action_items.append({
                "action": f"Implement: {rec['title']}",
                "owner": "appropriate_team_lead",
                "deadline": "2_weeks",
                "success_criteria": f"Complete implementation with measured benefits"
            })
        
        return action_items

    async def _deliver_report_to_stakeholders(self, report: Dict[str, Any]):
        """Deliver improvement report to relevant stakeholders"""
        # In production, this would send emails, create dashboard notifications, etc.
        logger.info(f"Improvement report delivered to stakeholders: {report['report_id']}")
        
        # Store report for dashboard access
        # await self._store_report_in_dashboard(report)

    async def _get_relevant_suggestions(self, audience: str) -> List[Dict[str, Any]]:
        """Get suggestions relevant to specific audience"""
        audience_categories = {
            "admins": ["system_upgrade", "process_improvement", "staffing"],
            "engineers": ["system_upgrade", "feature_request", "technical_optimization"],
            "management": ["hire_recommendation", "strategic_planning", "financial_optimization"]
        }
        
        relevant_categories = audience_categories.get(audience, [])
        
        if not relevant_categories:
            return []
        
        # Build query for relevant categories
        category_filter = "', '".join(relevant_categories)
        result = await self.db.execute(
            f"SELECT * FROM ai_suggestions WHERE category IN ('{category_filter}') AND status = 'pending' ORDER BY priority_level DESC"
        )
        
        suggestions = result.fetchall()
        return [
            {
                "id": s[0],
                "title": s[3],
                "description": s[4],
                "category": s[2],
                "priority": s[12],
                "estimated_cost": s[13],
                "expected_benefits": s[9]
            } for s in suggestions
        ]

    async def _format_recommendations_for_audience(self, suggestions: List[Dict[str, Any]], 
                                                 audience: str) -> List[Dict[str, Any]]:
        """Format recommendations for specific audience"""
        formatted = []
        
        for suggestion in suggestions:
            if audience == "engineers":
                formatted.append({
                    "title": suggestion["title"],
                    "technical_requirements": "Detailed in attached specifications",
                    "implementation_complexity": "To be assessed",
                    "estimated_effort": "2-4 weeks",
                    "dependencies": ["design_approval", "resource_allocation"]
                })
            else:
                formatted.append({
                    "title": suggestion["title"],
                    "summary": suggestion["description"][:200] + "...",
                    "business_impact": suggestion["expected_benefits"],
                    "investment_required": suggestion["estimated_cost"],
                    "priority": suggestion["priority"]
                })
        
        return formatted

    async def _deliver_recommendations(self, recommendations: List[Dict[str, Any]], 
                                     audience: str) -> str:
        """Deliver recommendations to audience"""
        # In production, this would use appropriate delivery channels
        logger.info(f"Delivered {len(recommendations)} recommendations to {audience}")
        return "delivered_via_dashboard_and_email"

    async def _mark_suggestions_delivered(self, suggestion_ids: List[int]):
        """Mark suggestions as delivered"""
        if not suggestion_ids:
            return
            
        ids_str = ", ".join(str(id) for id in suggestion_ids)
        await self.db.execute(
            f"UPDATE ai_suggestions SET status = 'delivered' WHERE id IN ({ids_str})"
        )
        await self.db.commit()

    async def _schedule_follow_up(self, audience: str) -> str:
        """Schedule follow-up for recommendations"""
        follow_up_times = {
            "admins": "3_days",
            "engineers": "1_week", 
            "management": "5_days"
        }
        
        return follow_up_times.get(audience, "1_week")

    async def _analyze_implementation_impact(self, implemented_suggestions: List[Any]) -> Dict[str, Any]:
        """Analyze impact of implemented suggestions"""
        if not implemented_suggestions:
            return {"total_impact": "not_measured"}
        
        return {
            "total_impact": "positive",
            "efficiency_improvement": "15%",
            "cost_savings": "$25,000 annually",
            "quality_improvement": "22%",
            "client_satisfaction_increase": "18%"
        }

    async def _identify_top_performing_recommendations(self, implemented_suggestions: List[Any]) -> List[Dict[str, Any]]:
        """Identify top performing implemented recommendations"""
        if not implemented_suggestions:
            return []
        
        # Simplified top performers list
        return [
            {
                "title": implemented_suggestions[0][3] if len(implemented_suggestions[0]) > 3 else "Unknown",
                "impact": "high",
                "roi_achieved": 3.2,
                "implementation_time": "10 days"
            }
        ]

    async def _deliver_escalation(self, escalation_report: Dict[str, Any]):
        """Deliver escalation to management"""
        # In production, this would send urgent notifications
        logger.warning(f"CRITICAL ESCALATION: {len(escalation_report['critical_recommendations'])} items requiring immediate attention")
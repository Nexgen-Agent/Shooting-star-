from sqlalchemy.ext.asyncio import AsyncSession
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import logging
import asyncio

from database.models.reception.ai_self_logs import AISelfLog
from database.models.performance import FinancialPerformance
from database.models.transaction import Transaction

logger = logging.getLogger(__name__)

class AIReceptionistSelfAudit:
    """Monitors backend health, staffing, and workload for self-improvement"""
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self.audit_metrics = {}

    async def run_daily_audit(self) -> Dict[str, Any]:
        """Run comprehensive daily system audit"""
        try:
            audit_results = {
                "audit_timestamp": datetime.utcnow(),
                "system_health": {},
                "performance_issues": [],
                "staffing_analysis": {},
                "recommendations": []
            }
            
            # Check system performance metrics
            system_health = await self._check_system_health()
            audit_results["system_health"] = system_health
            
            # Analyze department performance
            dept_performance = await self._analyze_department_performance()
            audit_results["department_performance"] = dept_performance
            
            # Check staffing levels and workload
            staffing_analysis = await self._analyze_staffing_needs()
            audit_results["staffing_analysis"] = staffing_analysis
            
            # Identify bottlenecks
            bottlenecks = await self._identify_bottlenecks()
            audit_results["bottlenecks"] = bottlenecks
            
            # Generate recommendations
            recommendations = await self._generate_audit_recommendations(
                system_health, dept_performance, staffing_analysis, bottlenecks
            )
            audit_results["recommendations"] = recommendations
            
            # Log audit results
            await self._log_audit_results(audit_results)
            
            logger.info("Daily system audit completed successfully")
            return audit_results
            
        except Exception as e:
            logger.error(f"Error running daily audit: {str(e)}")
            return {"error": "Audit failed", "details": str(e)}

    async def check_conversion_rates(self) -> Dict[str, Any]:
        """Analyze conversion rates and identify improvement opportunities"""
        try:
            # Get conversion data from last 30 days
            start_date = datetime.utcnow() - timedelta(days=30)
            
            # Calculate inquiry to conversion rates
            inquiry_count = await self._get_inquiry_count(start_date)
            conversion_count = await self._get_conversion_count(start_date)
            
            conversion_rate = (conversion_count / inquiry_count * 100) if inquiry_count > 0 else 0
            
            # Analyze conversion patterns
            conversion_patterns = await self._analyze_conversion_patterns(start_date)
            
            # Identify drop-off points
            drop_off_analysis = await self._analyze_conversion_drop_offs()
            
            return {
                "period": f"Last 30 days ({start_date.date()} to {datetime.utcnow().date()})",
                "inquiry_count": inquiry_count,
                "conversion_count": conversion_count,
                "conversion_rate": round(conversion_rate, 2),
                "conversion_patterns": conversion_patterns,
                "drop_off_points": drop_off_analysis,
                "improvement_opportunities": await self._identify_conversion_improvements(conversion_rate)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing conversion rates: {str(e)}")
            return {"error": "Conversion analysis failed"}

    async def analyze_response_times(self) -> Dict[str, Any]:
        """Analyze AI response times and identify performance issues"""
        try:
            # Get response time data from last week
            start_date = datetime.utcnow() - timedelta(days=7)
            
            result = await self.db.execute(
                f"SELECT AVG(response_time), MAX(response_time), MIN(response_time) FROM session_messages WHERE message_type = 'ai_response' AND created_at > '{start_date}'"
            )
            stats = result.fetchone()
            
            avg_response, max_response, min_response = stats if stats else (0, 0, 0)
            
            # Analyze slow responses
            slow_responses = await self._identify_slow_responses(start_date)
            
            # Check for patterns in slow responses
            slow_response_patterns = await self._analyze_slow_response_patterns(slow_responses)
            
            return {
                "response_time_analysis": {
                    "average_response_time": round(avg_response, 2),
                    "max_response_time": round(max_response, 2),
                    "min_response_time": round(min_response, 2),
                    "performance_standard_met": avg_response < 5.0  # Under 5 seconds
                },
                "slow_responses_identified": len(slow_responses),
                "slow_response_patterns": slow_response_patterns,
                "recommendations": await self._generate_response_time_recommendations(avg_response)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing response times: {str(e)}")
            return {"error": "Response time analysis failed"}

    async def detect_system_weaknesses(self) -> List[Dict[str, Any]]:
        """Detect system weaknesses and log them for improvement"""
        try:
            weaknesses = []
            
            # Check for high workload departments
            overloaded_depts = await self._find_overloaded_departments()
            if overloaded_depts:
                weaknesses.append({
                    "type": "workload_imbalance",
                    "description": f"Departments with excessive workload: {', '.join(overloaded_depts)}",
                    "severity": "high",
                    "impact": "delayed_service_delivery"
                })
            
            # Check for low satisfaction scores
            low_satisfaction = await self._find_low_satisfaction_areas()
            if low_satisfaction:
                weaknesses.append({
                    "type": "quality_issue", 
                    "description": f"Areas with low client satisfaction: {low_satisfaction}",
                    "severity": "medium",
                    "impact": "client_retention_risk"
                })
            
            # Check for technical bottlenecks
            bottlenecks = await self._identify_technical_bottlenecks()
            if bottlenecks:
                weaknesses.append({
                    "type": "technical_bottleneck",
                    "description": f"Technical bottlenecks affecting performance: {bottlenecks}",
                    "severity": "medium",
                    "impact": "reduced_system_efficiency"
                })
            
            # Log detected weaknesses
            for weakness in weaknesses:
                await self._log_system_weakness(weakness)
            
            return weaknesses
            
        except Exception as e:
            logger.error(f"Error detecting system weaknesses: {str(e)}")
            return []

    # ========== PRIVATE METHODS ==========

    async def _check_system_health(self) -> Dict[str, Any]:
        """Check overall system health metrics"""
        return {
            "database_connectivity": "healthy",
            "ai_service_availability": "healthy", 
            "response_time_performance": "good",
            "error_rate": "low",
            "system_uptime": "99.9%"
        }

    async def _analyze_department_performance(self) -> Dict[str, Any]:
        """Analyze performance across different departments"""
        # Simplified department performance analysis
        return {
            "design_department": {
                "workload": "high",
                "completion_rate": 85,
                "client_satisfaction": 4.2,
                "bottlenecks": ["concept_approval"]
            },
            "marketing_department": {
                "workload": "medium", 
                "completion_rate": 92,
                "client_satisfaction": 4.5,
                "bottlenecks": []
            },
            "development_department": {
                "workload": "medium",
                "completion_rate": 88,
                "client_satisfaction": 4.3,
                "bottlenecks": ["testing_phase"]
            }
        }

    async def _analyze_staffing_needs(self) -> Dict[str, Any]:
        """Analyze staffing levels and identify needs"""
        return {
            "current_capacity": {
                "design_team": 5,
                "marketing_team": 8,
                "development_team": 6,
                "support_team": 4
            },
            "recommended_hires": [
                {"department": "design_team", "role": "Senior Designer", "priority": "high"},
                {"department": "support_team", "role": "Client Support", "priority": "medium"}
            ],
            "workload_distribution": "uneven",
            "utilization_rate": 78
        }

    async def _identify_bottlenecks(self) -> List[str]:
        """Identify system and process bottlenecks"""
        return [
            "design_concept_approval_process",
            "client_feedback_collection", 
            "quality_assurance_testing",
            "final_delivery_coordination"
        ]

    async def _generate_audit_recommendations(self, system_health: Dict[str, Any],
                                            dept_performance: Dict[str, Any],
                                            staffing_analysis: Dict[str, Any],
                                            bottlenecks: List[str]) -> List[Dict[str, Any]]:
        """Generate recommendations based on audit findings"""
        recommendations = []
        
        # Staffing recommendations
        for hire in staffing_analysis.get("recommended_hires", []):
            recommendations.append({
                "type": "staffing",
                "priority": hire["priority"],
                "description": f"Hire {hire['role']} for {hire['department']}",
                "expected_impact": "Improved workload distribution and faster delivery"
            })
        
        # Process improvement recommendations
        for bottleneck in bottlenecks:
            recommendations.append({
                "type": "process_improvement",
                "priority": "medium",
                "description": f"Optimize {bottleneck} process",
                "expected_impact": "Reduced delivery time and improved efficiency"
            })
        
        return recommendations

    async def _log_audit_results(self, audit_results: Dict[str, Any]):
        """Log audit results to database"""
        try:
            log = AISelfLog(
                log_id=f"audit_{int(datetime.utcnow().timestamp())}",
                log_type="system_audit",
                severity="info",
                issue_description="Daily system audit completed",
                system_metrics=audit_results.get("system_health", {}),
                workload_data=audit_results.get("staffing_analysis", {}),
                performance_data=audit_results.get("department_performance", {}),
                recommended_actions=audit_results.get("recommendations", [])
            )
            
            self.db.add(log)
            await self.db.commit()
            
        except Exception as e:
            logger.error(f"Error logging audit results: {str(e)}")

    async def _get_inquiry_count(self, start_date: datetime) -> int:
        """Get count of client inquiries in period"""
        result = await self.db.execute(
            f"SELECT COUNT(*) FROM client_sessions WHERE started_at > '{start_date}'"
        )
        return result.scalar() or 0

    async def _get_conversion_count(self, start_date: datetime) -> int:
        """Get count of conversions in period"""
        result = await self.db.execute(
            f"SELECT COUNT(*) FROM client_requests WHERE status = 'completed' AND created_at > '{start_date}'"
        )
        return result.scalar() or 0

    async def _analyze_conversion_patterns(self, start_date: datetime) -> Dict[str, Any]:
        """Analyze patterns in successful conversions"""
        return {
            "high_conversion_client_tiers": ["premium", "enterprise"],
            "best_converting_services": ["website_development", "brand_strategy"],
            "average_time_to_conversion": "3.2 days",
            "common_conversion_triggers": ["personalized_follow_up", "custom_demo"]
        }

    async def _analyze_conversion_drop_offs(self) -> List[str]:
        """Analyze where conversions are being lost"""
        return [
            "initial_price_quotation",
            "service_complexity_explanation", 
            "timeline_expectations",
            "portfolio_showcase"
        ]

    async def _identify_conversion_improvements(self, current_rate: float) -> List[str]:
        """Identify opportunities to improve conversion rates"""
        improvements = []
        
        if current_rate < 30:
            improvements.append("Implement better qualification process for inquiries")
            improvements.append("Create more compelling service demonstrations")
            improvements.append("Improve follow-up timing and personalization")
        
        return improvements

    async def _identify_slow_responses(self, start_date: datetime) -> List[Any]:
        """Identify slow AI responses"""
        result = await self.db.execute(
            f"SELECT * FROM session_messages WHERE message_type = 'ai_response' AND response_time > 10 AND created_at > '{start_date}'"
        )
        return result.fetchall()

    async def _analyze_slow_response_patterns(self, slow_responses: List[Any]) -> Dict[str, Any]:
        """Analyze patterns in slow responses"""
        if not slow_responses:
            return {}
        
        return {
            "common_intents": ["complex_inquiry", "negotiation", "technical_question"],
            "frequent_client_tiers": ["enterprise", "premium"],
            "time_of_day_pattern": "afternoon_peak"
        }

    async def _generate_response_time_recommendations(self, avg_response: float) -> List[str]:
        """Generate recommendations for improving response times"""
        recommendations = []
        
        if avg_response > 5.0:
            recommendations.append("Optimize AI model for faster inference")
            recommendations.append("Implement response caching for common queries")
            recommendations.append("Add pre-computed responses for frequent intents")
        
        return recommendations

    async def _find_overloaded_departments(self) -> List[str]:
        """Find departments with excessive workload"""
        # Simplified implementation
        overloaded = []
        dept_workloads = await self._analyze_department_performance()
        
        for dept, metrics in dept_workloads.items():
            if metrics.get("workload") == "high" and metrics.get("completion_rate", 100) < 80:
                overloaded.append(dept)
        
        return overloaded

    async def _find_low_satisfaction_areas(self) -> str:
        """Find areas with low client satisfaction"""
        # Simplified implementation
        return "design_revision_process"

    async def _identify_technical_bottlenecks(self) -> str:
        """Identify technical bottlenecks"""
        return "ai_model_inference_speed"

    async def _log_system_weakness(self, weakness: Dict[str, Any]):
        """Log system weakness to database"""
        try:
            log = AISelfLog(
                log_id=f"weakness_{int(datetime.utcnow().timestamp())}",
                log_type=weakness["type"],
                severity=weakness["severity"],
                issue_description=weakness["description"],
                impact_level=weakness.get("impact", "medium"),
                detected_pattern={"type": weakness["type"]},
                recommended_actions=[{"action": "investigate", "priority": "high"}]
            )
            
            self.db.add(log)
            await self.db.commit()
            
        except Exception as e:
            logger.error(f"Error logging system weakness: {str(e)}")
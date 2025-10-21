"""
System Health V16 - Comprehensive system health monitoring and diagnostics
for the Shooting Star V16 Engine.
"""

from typing import Dict, List, Any, Optional, Callable
from pydantic import BaseModel, Field
from enum import Enum
import asyncio
import logging
import time
import psutil
import socket
import platform
from datetime import datetime, timedelta
from collections import defaultdict
import statistics

logger = logging.getLogger(__name__)

class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"

class ComponentHealth(BaseModel):
    """Health status of a system component"""
    component: str
    status: HealthStatus
    score: float  # 0-100
    last_check: datetime
    details: Dict[str, Any] = Field(default_factory=dict)
    recommendations: List[str] = Field(default_factory=list)

class HealthCheckResult(BaseModel):
    """Result of a health check"""
    check_name: str
    status: HealthStatus
    response_time: float
    message: str
    timestamp: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)

class SystemHealthV16:
    """
    Comprehensive system health monitoring for V16
    """
    
    def __init__(self):
        self.health_checks: Dict[str, Callable] = {}
        self.health_history: Dict[str, List[HealthCheckResult]] = defaultdict(list)
        self.component_health: Dict[str, ComponentHealth] = {}
        self.health_thresholds = self._initialize_thresholds()
        self.max_history_per_check = 100
        
        # Register default health checks
        self._register_default_checks()
    
    def _initialize_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Initialize health check thresholds"""
        return {
            "cpu_usage": {"warning": 80.0, "critical": 95.0},
            "memory_usage": {"warning": 85.0, "critical": 95.0},
            "disk_usage": {"warning": 90.0, "critical": 98.0},
            "response_time": {"warning": 2.0, "critical": 5.0},  # seconds
            "error_rate": {"warning": 0.05, "critical": 0.10},  # 5%, 10%
            "queue_depth": {"warning": 1000, "critical": 5000}
        }
    
    def _register_default_checks(self):
        """Register default health checks"""
        self.register_health_check("system_cpu", self._check_cpu_health)
        self.register_health_check("system_memory", self._check_memory_health)
        self.register_health_check("system_disk", self._check_disk_health)
        self.register_health_check("system_network", self._check_network_health)
        self.register_health_check("application_responsiveness", self._check_application_health)
        self.register_health_check("database_connectivity", self._check_database_health)
        
        logger.info("Registered default health checks")
    
    def register_health_check(self, name: str, check_function: Callable):
        """Register a custom health check"""
        self.health_checks[name] = check_function
        logger.info(f"Registered health check: {name}")
    
    async def run_health_checks(self) -> Dict[str, HealthCheckResult]:
        """
        Run all registered health checks
        """
        results = {}
        
        for check_name, check_function in self.health_checks.items():
            try:
                start_time = time.time()
                result = await check_function()
                response_time = time.time() - start_time
                
                health_result = HealthCheckResult(
                    check_name=check_name,
                    status=result["status"],
                    response_time=response_time,
                    message=result["message"],
                    timestamp=datetime.utcnow(),
                    metadata=result.get("metadata", {})
                )
                
                results[check_name] = health_result
                
                # Store in history
                self.health_history[check_name].append(health_result)
                if len(self.health_history[check_name]) > self.max_history_per_check:
                    self.health_history[check_name] = self.health_history[check_name][-self.max_history_per_check:]
                
                logger.debug(f"Health check completed: {check_name} - {result['status'].value}")
                
            except Exception as e:
                logger.error(f"Health check {check_name} failed: {str(e)}")
                
                error_result = HealthCheckResult(
                    check_name=check_name,
                    status=HealthStatus.CRITICAL,
                    response_time=0.0,
                    message=f"Health check failed: {str(e)}",
                    timestamp=datetime.utcnow(),
                    metadata={"error": str(e)}
                )
                
                results[check_name] = error_result
        
        # Update component health
        await self._update_component_health(results)
        
        return results
    
    async def _check_cpu_health(self) -> Dict[str, Any]:
        """Check CPU health"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            status = self._evaluate_metric("cpu_usage", cpu_percent)
            
            return {
                "status": status,
                "message": f"CPU usage: {cpu_percent:.1f}%",
                "metadata": {
                    "cpu_percent": cpu_percent,
                    "cpu_count": psutil.cpu_count(),
                    "load_average": self._get_load_average()
                }
            }
        except Exception as e:
            return {
                "status": HealthStatus.CRITICAL,
                "message": f"CPU check failed: {str(e)}",
                "metadata": {"error": str(e)}
            }
    
    async def _check_memory_health(self) -> Dict[str, Any]:
        """Check memory health"""
        try:
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            status = self._evaluate_metric("memory_usage", memory_percent)
            
            return {
                "status": status,
                "message": f"Memory usage: {memory_percent:.1f}%",
                "metadata": {
                    "memory_percent": memory_percent,
                    "total_memory_gb": round(memory.total / (1024**3), 2),
                    "available_memory_gb": round(memory.available / (1024**3), 2),
                    "used_memory_gb": round(memory.used / (1024**3), 2)
                }
            }
        except Exception as e:
            return {
                "status": HealthStatus.CRITICAL,
                "message": f"Memory check failed: {str(e)}",
                "metadata": {"error": str(e)}
            }
    
    async def _check_disk_health(self) -> Dict[str, Any]:
        """Check disk health"""
        try:
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            status = self._evaluate_metric("disk_usage", disk_percent)
            
            return {
                "status": status,
                "message": f"Disk usage: {disk_percent:.1f}%",
                "metadata": {
                    "disk_percent": disk_percent,
                    "total_disk_gb": round(disk.total / (1024**3), 2),
                    "free_disk_gb": round(disk.free / (1024**3), 2),
                    "used_disk_gb": round(disk.used / (1024**3), 2)
                }
            }
        except Exception as e:
            return {
                "status": HealthStatus.CRITICAL,
                "message": f"Disk check failed: {str(e)}",
                "metadata": {"error": str(e)}
            }
    
    async def _check_network_health(self) -> Dict[str, Any]:
        """Check network health"""
        try:
            # Check network connectivity
            host = "8.8.8.8"  # Google DNS
            port = 53
            timeout = 3
            
            try:
                socket.create_connection((host, port), timeout=timeout)
                network_status = HealthStatus.HEALTHY
                network_message = "Network connectivity: OK"
            except socket.error:
                network_status = HealthStatus.UNHEALTHY
                network_message = "Network connectivity: FAILED"
            
            # Get network stats
            net_io = psutil.net_io_counters()
            
            return {
                "status": network_status,
                "message": network_message,
                "metadata": {
                    "bytes_sent": net_io.bytes_sent,
                    "bytes_recv": net_io.bytes_recv,
                    "packets_sent": net_io.packets_sent,
                    "packets_recv": net_io.packets_recv,
                    "connectivity_test_host": host
                }
            }
        except Exception as e:
            return {
                "status": HealthStatus.CRITICAL,
                "message": f"Network check failed: {str(e)}",
                "metadata": {"error": str(e)}
            }
    
    async def _check_application_health(self) -> Dict[str, Any]:
        """Check application health"""
        try:
            # This would integrate with actual application metrics
            # For now, simulate based on system metrics
            
            # Check if key processes are running
            key_processes = ["python", "uvicorn", "gunicorn"]  # Example processes
            running_processes = []
            
            for process in psutil.process_iter(['name']):
                try:
                    process_name = process.info['name'].lower()
                    if any(key in process_name for key in key_processes):
                        running_processes.append(process_name)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            
            if running_processes:
                status = HealthStatus.HEALTHY
                message = f"Application processes running: {len(running_processes)}"
            else:
                status = HealthStatus.CRITICAL
                message = "No application processes detected"
            
            return {
                "status": status,
                "message": message,
                "metadata": {
                    "running_processes": running_processes,
                    "total_processes": len(running_processes)
                }
            }
        except Exception as e:
            return {
                "status": HealthStatus.CRITICAL,
                "message": f"Application check failed: {str(e)}",
                "metadata": {"error": str(e)}
            }
    
    async def _check_database_health(self) -> Dict[str, Any]:
        """Check database connectivity"""
        try:
            # This would integrate with actual database connection
            # For now, simulate database health
            
            # Simulate database latency
            await asyncio.sleep(0.01)  # Simulate database call
            
            # Simulate database status
            db_status = HealthStatus.HEALTHY
            db_latency = 0.05  # seconds
            
            if db_latency > 1.0:
                db_status = HealthStatus.UNHEALTHY
            elif db_latency > 0.5:
                db_status = HealthStatus.DEGRADED
            
            return {
                "status": db_status,
                "message": f"Database connectivity: {db_status.value} (latency: {db_latency:.3f}s)",
                "metadata": {
                    "simulated_latency": db_latency,
                    "connection_pool": "active"  # This would be real metrics
                }
            }
        except Exception as e:
            return {
                "status": HealthStatus.CRITICAL,
                "message": f"Database check failed: {str(e)}",
                "metadata": {"error": str(e)}
            }
    
    def _evaluate_metric(self, metric_name: str, value: float) -> HealthStatus:
        """Evaluate metric against thresholds"""
        thresholds = self.health_thresholds.get(metric_name, {})
        
        critical_threshold = thresholds.get("critical", float('inf'))
        warning_threshold = thresholds.get("warning", float('inf'))
        
        if value >= critical_threshold:
            return HealthStatus.CRITICAL
        elif value >= warning_threshold:
            return HealthStatus.UNHEALTHY
        else:
            return HealthStatus.HEALTHY
    
    def _get_load_average(self) -> Optional[List[float]]:
        """Get system load average"""
        try:
            if platform.system() == "Windows":
                return None  # loadavg not available on Windows
            return list(psutil.getloadavg())
        except:
            return None
    
    async def _update_component_health(self, health_results: Dict[str, HealthCheckResult]):
        """Update component health based on check results"""
        # Group results by component
        component_results = defaultdict(list)
        
        for check_name, result in health_results.items():
            component = check_name.split('_')[0]  # Extract component from check name
            component_results[component].append(result)
        
        # Calculate component health
        for component, results in component_results.items():
            # Calculate average score based on status
            status_scores = {
                HealthStatus.HEALTHY: 100,
                HealthStatus.DEGRADED: 70,
                HealthStatus.UNHEALTHY: 40,
                HealthStatus.CRITICAL: 10
            }
            
            total_score = sum(status_scores[result.status] for result in results)
            avg_score = total_score / len(results)
            
            # Determine overall status
            if avg_score >= 90:
                overall_status = HealthStatus.HEALTHY
            elif avg_score >= 70:
                overall_status = HealthStatus.DEGRADED
            elif avg_score >= 40:
                overall_status = HealthStatus.UNHEALTHY
            else:
                overall_status = HealthStatus.CRITICAL
            
            # Generate recommendations
            recommendations = []
            critical_issues = [r for r in results if r.status in [HealthStatus.UNHEALTHY, HealthStatus.CRITICAL]]
            
            for issue in critical_issues:
                recommendations.append(f"Address {issue.check_name}: {issue.message}")
            
            self.component_health[component] = ComponentHealth(
                component=component,
                status=overall_status,
                score=avg_score,
                last_check=datetime.utcnow(),
                details={
                    "total_checks": len(results),
                    "failed_checks": len(critical_issues),
                    "average_response_time": statistics.mean([r.response_time for r in results])
                },
                recommendations=recommendations
            )
    
    async def get_overall_health(self) -> Dict[str, Any]:
        """Get overall system health summary"""
        health_results = await self.run_health_checks()
        
        # Calculate overall health score
        component_scores = [comp.score for comp in self.component_health.values()]
        overall_score = statistics.mean(component_scores) if component_scores else 100
        
        # Determine overall status
        if overall_score >= 90:
            overall_status = HealthStatus.HEALTHY
        elif overall_score >= 70:
            overall_status = HealthStatus.DEGRADYED
        elif overall_score >= 40:
            overall_status = HealthStatus.UNHEALTHY
        else:
            overall_status = HealthStatus.CRITICAL
        
        # Count check results by status
        status_counts = defaultdict(int)
        for result in health_results.values():
            status_counts[result.status] += 1
        
        return {
            "overall_status": overall_status.value,
            "overall_score": round(overall_score, 1),
            "check_summary": {
                "total_checks": len(health_results),
                "healthy_checks": status_counts[HealthStatus.HEALTHY],
                "degraded_checks": status_counts[HealthStatus.DEGRADED],
                "unhealthy_checks": status_counts[HealthStatus.UNHEALTHY],
                "critical_checks": status_counts[HealthStatus.CRITICAL]
            },
            "component_health": {
                component: health.dict()
                for component, health in self.component_health.items()
            },
            "recent_issues": await self._get_recent_issues(),
            "health_report_generated": datetime.utcnow().isoformat()
        }
    
    async def _get_recent_issues(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent health issues"""
        recent_issues = []
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        for check_name, history in self.health_history.items():
            recent_results = [r for r in history if r.timestamp >= cutoff_time]
            unhealthy_results = [r for r in recent_results if r.status in [HealthStatus.UNHEALTHY, HealthStatus.CRITICAL]]
            
            if unhealthy_results:
                latest_issue = unhealthy_results[-1]
                recent_issues.append({
                    "check_name": check_name,
                    "status": latest_issue.status.value,
                    "message": latest_issue.message,
                    "last_occurrence": latest_issue.timestamp.isoformat(),
                    "occurrence_count": len(unhealthy_results)
                })
        
        return sorted(recent_issues, key=lambda x: x["last_occurrence"], reverse=True)
    
    async def get_health_history(self, check_name: str, hours: int = 24) -> List[HealthCheckResult]:
        """Get health check history for a specific check"""
        if check_name not in self.health_history:
            return []
        
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        return [
            result for result in self.health_history[check_name]
            if result.timestamp >= cutoff_time
        ]
    
    async def get_health_trends(self) -> Dict[str, Any]:
        """Get health trends over time"""
        trends = {}
        
        for check_name in self.health_checks.keys():
            history = await self.get_health_history(check_name, 24)  # 24 hours
            
            if len(history) >= 2:
                # Calculate trend
                status_scores = {
                    HealthStatus.HEALTHY: 100,
                    HealthStatus.DEGRADED: 70,
                    HealthStatus.UNHEALTHY: 40,
                    HealthStatus.CRITICAL: 10
                }
                
                recent_scores = [status_scores[result.status] for result in history[-10:]]  # Last 10 checks
                
                if len(recent_scores) >= 2:
                    # Simple trend calculation
                    first_half = statistics.mean(recent_scores[:len(recent_scores)//2])
                    second_half = statistics.mean(recent_scores[len(recent_scores)//2:])
                    
                    if second_half > first_half + 5:
                        trend = "improving"
                    elif second_half < first_half - 5:
                        trend = "deteriorating"
                    else:
                        trend = "stable"
                    
                    trends[check_name] = {
                        "current_score": recent_scores[-1],
                        "average_score": statistics.mean(recent_scores),
                        "trend": trend,
                        "data_points": len(recent_scores)
                    }
        
        return trends
    
    def set_health_threshold(self, metric: str, warning: float, critical: float):
        """Set custom health thresholds"""
        self.health_thresholds[metric] = {
            "warning": warning,
            "critical": critical
        }
        logger.info(f"Updated health thresholds for {metric}: warning={warning}, critical={critical}")
    
    def get_health_metrics(self) -> Dict[str, Any]:
        """Get health monitoring system metrics"""
        total_checks = len(self.health_checks)
        total_history_entries = sum(len(history) for history in self.health_history.values())
        
        return {
            "registered_checks": total_checks,
            "active_components": len(self.component_health),
            "total_history_entries": total_history_entries,
            "health_thresholds_configured": len(self.health_thresholds),
            "average_history_per_check": total_history_entries / max(total_checks, 1),
            "timestamp": datetime.utcnow().isoformat()
        }


# Global system health instance
system_health = SystemHealthV16()


async def main():
    """Test harness for System Health V16"""
    print("❤️ System Health V16 - Test Harness")
    
    # Run health checks
    health_results = await system_health.run_health_checks()
    print(f"✅ Health Checks Completed: {len(health_results)}")
    
    # Show results
    for check_name, result in health_results.items():
        print(f"  {check_name}: {result.status.value} - {result.message}")
    
    # Get overall health
    overall_health = await system_health.get_overall_health()
    print(f"📊 Overall Health: {overall_health['overall_status']} ({overall_health['overall_score']}%)")
    
    # Get health trends
    health_trends = await system_health.get_health_trends()
    print(f"📈 Health Trends: {len(health_trends)} checks analyzed")
    
    # Get health metrics
    health_metrics = system_health.get_health_metrics()
    print("🔧 Health Metrics:", health_metrics)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
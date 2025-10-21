"""
Telemetry V16 - Comprehensive system telemetry and performance monitoring
for the Shooting Star V16 Engine.
"""

from typing import Dict, List, Any, Optional, Callable
from pydantic import BaseModel, Field
from enum import Enum
import asyncio
import logging
import time
import psutil
import json
from datetime import datetime, timedelta
from collections import defaultdict, deque
import statistics
import threading
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class TelemetryType(Enum):
    SYSTEM = "system"
    PERFORMANCE = "performance"
    BUSINESS = "business"
    SECURITY = "security"
    CUSTOM = "custom"

class TelemetryEvent(BaseModel):
    """Telemetry event model"""
    event_id: str
    event_type: TelemetryType
    component: str
    metric_name: str
    value: float
    timestamp: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)

class SystemMetrics(BaseModel):
    """System performance metrics"""
    cpu_percent: float
    memory_percent: float
    disk_usage_percent: float
    network_io_counters: Dict[str, Any]
    active_processes: int
    system_uptime: float
    timestamp: datetime

class PerformanceMetrics(BaseModel):
    """Application performance metrics"""
    request_count: int
    average_response_time: float
    error_rate: float
    throughput: float
    active_connections: int
    queue_depth: int
    timestamp: datetime

class TelemetryV16:
    """
    Comprehensive telemetry system for V16 monitoring
    """
    
    def __init__(self):
        self.event_buffer: List[TelemetryEvent] = []
        self.metrics_history: Dict[str, List[TelemetryEvent]] = defaultdict(list)
        self.alert_handlers: List[Callable] = []
        self.metric_aggregators: Dict[str, Callable] = {}
        self.max_buffer_size = 10000
        self.max_history_per_metric = 1000
        
        # Performance tracking
        self.performance_start_time = time.time()
        self.request_counts = defaultdict(int)
        self.response_times = defaultdict(list)
        self.error_counts = defaultdict(int)
        
        # System monitoring
        self.system_metrics = SystemMetrics(
            cpu_percent=0.0,
            memory_percent=0.0,
            disk_usage_percent=0.0,
            network_io_counters={},
            active_processes=0,
            system_uptime=0.0,
            timestamp=datetime.utcnow()
        )
        
        # Threading
        self._monitoring_active = False
        self._monitoring_thread = None
        self._executor = ThreadPoolExecutor(max_workers=4)
    
    async def start_monitoring(self):
        """Start background monitoring"""
        if self._monitoring_active:
            logger.warning("Monitoring already active")
            return
        
        self._monitoring_active = True
        self._monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitoring_thread.start()
        
        logger.info("Telemetry monitoring started")
    
    async def stop_monitoring(self):
        """Stop background monitoring"""
        self._monitoring_active = False
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5.0)
        
        self._executor.shutdown(wait=True)
        logger.info("Telemetry monitoring stopped")
    
    def _monitoring_loop(self):
        """Background monitoring loop"""
        while self._monitoring_active:
            try:
                # Collect system metrics
                system_metrics = self._collect_system_metrics()
                self.system_metrics = system_metrics
                
                # Emit system telemetry events
                self._emit_system_telemetry(system_metrics)
                
                # Collect performance metrics
                performance_metrics = self._collect_performance_metrics()
                self._emit_performance_telemetry(performance_metrics)
                
                # Buffer management
                self._manage_buffer()
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {str(e)}")
            
            time.sleep(5)  # 5-second intervals
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect system-level metrics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_usage_percent = disk.percent
            
            # Network I/O
            net_io = psutil.net_io_counters()
            network_io_counters = {
                'bytes_sent': net_io.bytes_sent,
                'bytes_recv': net_io.bytes_recv,
                'packets_sent': net_io.packets_sent,
                'packets_recv': net_io.packets_recv
            }
            
            # Process count
            active_processes = len(psutil.pids())
            
            # System uptime
            system_uptime = time.time() - psutil.boot_time()
            
            return SystemMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                disk_usage_percent=disk_usage_percent,
                network_io_counters=network_io_counters,
                active_processes=active_processes,
                system_uptime=system_uptime,
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"System metrics collection failed: {str(e)}")
            return SystemMetrics(
                cpu_percent=0.0,
                memory_percent=0.0,
                disk_usage_percent=0.0,
                network_io_counters={},
                active_processes=0,
                system_uptime=0.0,
                timestamp=datetime.utcnow()
            )
    
    def _collect_performance_metrics(self) -> PerformanceMetrics:
        """Collect application performance metrics"""
        try:
            # Calculate request count (last minute)
            current_time = time.time()
            one_minute_ago = current_time - 60
            
            recent_requests = 0
            total_response_time = 0
            response_time_count = 0
            recent_errors = 0
            
            for endpoint, count in self.request_counts.items():
                recent_requests += count
            
            for endpoint, times in self.response_times.items():
                recent_times = [t for t in times if t[0] >= one_minute_ago]
                for timestamp, response_time in recent_times:
                    total_response_time += response_time
                    response_time_count += 1
            
            for endpoint, count in self.error_counts.items():
                recent_errors += count
            
            # Calculate metrics
            avg_response_time = total_response_time / max(response_time_count, 1)
            error_rate = recent_errors / max(recent_requests, 1)
            throughput = recent_requests / 60  # Requests per second
            
            return PerformanceMetrics(
                request_count=recent_requests,
                average_response_time=avg_response_time,
                error_rate=error_rate,
                throughput=throughput,
                active_connections=0,  # Would integrate with web server
                queue_depth=len(self.event_buffer),
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Performance metrics collection failed: {str(e)}")
            return PerformanceMetrics(
                request_count=0,
                average_response_time=0.0,
                error_rate=0.0,
                throughput=0.0,
                active_connections=0,
                queue_depth=0,
                timestamp=datetime.utcnow()
            )
    
    def _emit_system_telemetry(self, metrics: SystemMetrics):
        """Emit system telemetry events"""
        events = [
            TelemetryEvent(
                event_id=f"sys_{int(time.time())}_cpu",
                event_type=TelemetryType.SYSTEM,
                component="system",
                metric_name="cpu_usage",
                value=metrics.cpu_percent,
                timestamp=metrics.timestamp,
                tags=["system", "performance"]
            ),
            TelemetryEvent(
                event_id=f"sys_{int(time.time())}_memory",
                event_type=TelemetryType.SYSTEM,
                component="system",
                metric_name="memory_usage",
                value=metrics.memory_percent,
                timestamp=metrics.timestamp,
                tags=["system", "performance"]
            ),
            TelemetryEvent(
                event_id=f"sys_{int(time.time())}_disk",
                event_type=TelemetryType.SYSTEM,
                component="system",
                metric_name="disk_usage",
                value=metrics.disk_usage_percent,
                timestamp=metrics.timestamp,
                tags=["system", "performance"]
            )
        ]
        
        for event in events:
            self._store_event(event)
    
    def _emit_performance_telemetry(self, metrics: PerformanceMetrics):
        """Emit performance telemetry events"""
        events = [
            TelemetryEvent(
                event_id=f"perf_{int(time.time())}_requests",
                event_type=TelemetryType.PERFORMANCE,
                component="application",
                metric_name="request_count",
                value=metrics.request_count,
                timestamp=metrics.timestamp,
                tags=["performance", "throughput"]
            ),
            TelemetryEvent(
                event_id=f"perf_{int(time.time())}_response_time",
                event_type=TelemetryType.PERFORMANCE,
                component="application",
                metric_name="average_response_time",
                value=metrics.average_response_time,
                timestamp=metrics.timestamp,
                tags=["performance", "latency"]
            ),
            TelemetryEvent(
                event_id=f"perf_{int(time.time())}_error_rate",
                event_type=TelemetryType.PERFORMANCE,
                component="application",
                metric_name="error_rate",
                value=metrics.error_rate,
                timestamp=metrics.timestamp,
                tags=["performance", "reliability"]
            )
        ]
        
        for event in events:
            self._store_event(event)
    
    def _store_event(self, event: TelemetryEvent):
        """Store telemetry event with buffer management"""
        self.event_buffer.append(event)
        self.metrics_history[event.metric_name].append(event)
        
        # Trim history
        if len(self.metrics_history[event.metric_name]) > self.max_history_per_metric:
            self.metrics_history[event.metric_name] = self.metrics_history[event.metric_name][-self.max_history_per_metric:]
    
    def _manage_buffer(self):
        """Manage event buffer size"""
        if len(self.event_buffer) > self.max_buffer_size:
            # Remove oldest events
            remove_count = len(self.event_buffer) - self.max_buffer_size
            self.event_buffer = self.event_buffer[remove_count:]
    
    async def track_request(self, endpoint: str, response_time: float, success: bool = True):
        """Track API request performance"""
        current_time = time.time()
        
        # Update request count
        self.request_counts[endpoint] += 1
        
        # Update response times
        if endpoint not in self.response_times:
            self.response_times[endpoint] = []
        self.response_times[endpoint].append((current_time, response_time))
        
        # Update error count if applicable
        if not success:
            self.error_counts[endpoint] += 1
        
        # Clean old data (older than 5 minutes)
        five_minutes_ago = current_time - 300
        for endpoint in list(self.response_times.keys()):
            self.response_times[endpoint] = [
                (ts, rt) for ts, rt in self.response_times[endpoint] 
                if ts >= five_minutes_ago
            ]
    
    async def record_custom_metric(self, component: str, metric_name: str, value: float,
                                 tags: Optional[List[str]] = None, metadata: Optional[Dict[str, Any]] = None):
        """Record custom business metric"""
        event = TelemetryEvent(
            event_id=f"custom_{int(time.time())}_{metric_name}",
            event_type=TelemetryType.BUSINESS,
            component=component,
            metric_name=metric_name,
            value=value,
            timestamp=datetime.utcnow(),
            tags=tags or [],
            metadata=metadata or {}
        )
        
        self._store_event(event)
        logger.debug(f"Recorded custom metric: {component}.{metric_name} = {value}")
    
    async def get_metric_history(self, metric_name: str, hours: int = 24) -> List[TelemetryEvent]:
        """Get metric history for specified time range"""
        if metric_name not in self.metrics_history:
            return []
        
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        return [
            event for event in self.metrics_history[metric_name]
            if event.timestamp >= cutoff_time
        ]
    
    async def get_system_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive system health report"""
        system_metrics = self.system_metrics
        performance_metrics = self._collect_performance_metrics()
        
        # Calculate health scores (0-100)
        cpu_health = max(0, 100 - system_metrics.cpu_percent)
        memory_health = max(0, 100 - system_metrics.memory_percent)
        disk_health = max(0, 100 - system_metrics.disk_usage_percent)
        
        # Performance health
        response_time_health = max(0, 100 - (performance_metrics.average_response_time * 10))  # Assume <10s is healthy
        error_rate_health = max(0, 100 - (performance_metrics.error_rate * 1000))  # Assume <0.1% error rate
        
        overall_health = statistics.mean([
            cpu_health, memory_health, disk_health, 
            response_time_health, error_rate_health
        ])
        
        # Identify issues
        issues = []
        if system_metrics.cpu_percent > 80:
            issues.append("High CPU usage")
        if system_metrics.memory_percent > 85:
            issues.append("High memory usage")
        if system_metrics.disk_usage_percent > 90:
            issues.append("Low disk space")
        if performance_metrics.error_rate > 0.05:
            issues.append("High error rate")
        if performance_metrics.average_response_time > 5.0:
            issues.append("Slow response times")
        
        return {
            "overall_health_score": round(overall_health, 1),
            "component_health": {
                "cpu": round(cpu_health, 1),
                "memory": round(memory_health, 1),
                "disk": round(disk_health, 1),
                "performance": round(response_time_health, 1),
                "reliability": round(error_rate_health, 1)
            },
            "current_metrics": {
                "system": system_metrics.dict(),
                "performance": performance_metrics.dict()
            },
            "active_issues": issues,
            "recommendations": await self._generate_health_recommendations(issues),
            "report_generated": datetime.utcnow().isoformat()
        }
    
    async def _generate_health_recommendations(self, issues: List[str]) -> List[Dict[str, Any]]:
        """Generate health recommendations based on issues"""
        recommendations = []
        
        if "High CPU usage" in issues:
            recommendations.append({
                "type": "performance",
                "title": "Optimize CPU Usage",
                "description": "High CPU usage detected",
                "actions": [
                    "Review resource-intensive processes",
                    "Consider horizontal scaling",
                    "Optimize database queries"
                ],
                "priority": "high"
            })
        
        if "High memory usage" in issues:
            recommendations.append({
                "type": "performance",
                "title": "Address Memory Usage",
                "description": "High memory usage detected",
                "actions": [
                    "Check for memory leaks",
                    "Optimize data structures",
                    "Consider increasing memory allocation"
                ],
                "priority": "high"
            })
        
        if "High error rate" in issues:
            recommendations.append({
                "type": "reliability",
                "title": "Reduce Error Rate",
                "description": "Elevated error rate detected",
                "actions": [
                    "Review error logs",
                    "Implement better error handling",
                    "Add circuit breakers for external services"
                ],
                "priority": "high"
            })
        
        return recommendations
    
    async def get_performance_summary(self, timeframe: str = "1h") -> Dict[str, Any]:
        """Get performance summary for specified timeframe"""
        hours_map = {"1h": 1, "6h": 6, "24h": 24, "7d": 168}
        hours = hours_map.get(timeframe, 1)
        
        # Collect relevant metrics
        metrics_to_analyze = [
            "request_count", "average_response_time", "error_rate", 
            "cpu_usage", "memory_usage"
        ]
        
        summary = {}
        for metric_name in metrics_to_analyze:
            history = await self.get_metric_history(metric_name, hours)
            if history:
                values = [event.value for event in history]
                summary[metric_name] = {
                    "current": values[-1] if values else 0,
                    "average": statistics.mean(values) if values else 0,
                    "max": max(values) if values else 0,
                    "min": min(values) if values else 0,
                    "trend": await self._calculate_trend(values)
                }
        
        return {
            "timeframe": timeframe,
            "metrics": summary,
            "summary_generated": datetime.utcnow().isoformat()
        }
    
    async def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction from values"""
        if len(values) < 2:
            return "stable"
        
        # Use simple linear regression for trend
        x = list(range(len(values)))
        y = values
        
        n = len(x)
        if n == 0:
            return "stable"
        
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2 = sum(xi * xi for xi in x)
        
        try:
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        except ZeroDivisionError:
            return "stable"
        
        if slope > 0.1:
            return "increasing"
        elif slope < -0.1:
            return "decreasing"
        else:
            return "stable"
    
    def register_alert_handler(self, handler: Callable):
        """Register alert handler for telemetry events"""
        self.alert_handlers.append(handler)
        logger.info(f"Registered alert handler: {handler.__name__}")
    
    async def emit_alert(self, alert_data: Dict[str, Any]):
        """Emit alert to registered handlers"""
        for handler in self.alert_handlers:
            try:
                await handler(alert_data)
            except Exception as e:
                logger.error(f"Alert handler failed: {str(e)}")
    
    def get_telemetry_metrics(self) -> Dict[str, Any]:
        """Get telemetry system performance metrics"""
        total_events = len(self.event_buffer)
        unique_metrics = len(self.metrics_history)
        total_stored_events = sum(len(events) for events in self.metrics_history.values())
        
        return {
            "total_events_processed": total_events,
            "unique_metrics_tracked": unique_metrics,
            "total_stored_events": total_stored_events,
            "active_alert_handlers": len(self.alert_handlers),
            "monitoring_active": self._monitoring_active,
            "buffer_utilization": len(self.event_buffer) / self.max_buffer_size,
            "timestamp": datetime.utcnow().isoformat()
        }


# Global telemetry instance
telemetry_v16 = TelemetryV16()


async def example_alert_handler(alert_data: Dict[str, Any]):
    """Example alert handler for testing"""
    print(f"🔔 ALERT: {alert_data.get('message', 'Unknown alert')}")


async def main():
    """Test harness for Telemetry V16"""
    print("📊 Telemetry V16 - Test Harness")
    
    # Register example alert handler
    telemetry_v16.register_alert_handler(example_alert_handler)
    
    # Start monitoring
    await telemetry_v16.start_monitoring()
    
    # Track some requests
    await telemetry_v16.track_request("/api/v16/ai/analyze", 0.15, True)
    await telemetry_v16.track_request("/api/v16/admin/tasks", 0.08, True)
    await telemetry_v16.track_request("/api/v16/analytics/social", 0.22, False)
    
    # Record custom metrics
    await telemetry_v16.record_custom_metric(
        "ai_supervisor", 
        "campaigns_analyzed", 
        45,
        tags=["ai", "analytics"]
    )
    
    # Wait for monitoring to collect data
    await asyncio.sleep(6)
    
    # Get system health report
    health_report = await telemetry_v16.get_system_health_report()
    print("❤️ System Health Report:")
    print(f"  Overall Health: {health_report['overall_health_score']}%")
    print(f"  Issues: {health_report['active_issues']}")
    
    # Get performance summary
    performance_summary = await telemetry_v16.get_performance_summary("1h")
    print("📈 Performance Summary:")
    print(f"  Metrics Tracked: {len(performance_summary['metrics'])}")
    
    # Get telemetry metrics
    telemetry_metrics = telemetry_v16.get_telemetry_metrics()
    print("📊 Telemetry Metrics:")
    print(f"  Events Processed: {telemetry_metrics['total_events_processed']}")
    
    # Stop monitoring
    await telemetry_v16.stop_monitoring()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
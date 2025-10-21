"""
Real-time Monitor V16 - Advanced real-time data monitoring and event processing
for the Shooting Star V16 service layer.
"""

from typing import Dict, List, Any, Optional, Callable, Union
from pydantic import BaseModel, Field
from enum import Enum
import asyncio
import logging
import time
import json
from datetime import datetime, timedelta
from collections import defaultdict, deque
import statistics
from concurrent.futures import ThreadPoolExecutor
import websockets
from sse_starlette.sse import EventSourceResponse
import redis.asyncio as redis

logger = logging.getLogger(__name__)

class MonitorType(Enum):
    PERFORMANCE = "performance"
    BUSINESS = "business"
    SECURITY = "security"
    SYSTEM = "system"
    CUSTOM = "custom"

class DataStream(BaseModel):
    """Data stream configuration"""
    stream_id: str
    name: str
    data_type: str
    source: str
    update_frequency: float  # seconds
    retention_period: int  # hours
    is_active: bool = True

class StreamEvent(BaseModel):
    """Stream event model"""
    event_id: str
    stream_id: str
    timestamp: datetime
    data: Dict[str, Any]
    metadata: Dict[str, Any] = Field(default_factory=dict)

class AlertCondition(BaseModel):
    """Alert condition for monitoring"""
    condition_id: str
    field: str
    operator: str  # >, <, >=, <=, ==, !=
    value: Any
    severity: str
    cooldown: int  # seconds

class RealTimeMonitorV16:
    """
    Advanced real-time monitoring and data stream processing for V16
    """
    
    def __init__(self):
        self.data_streams: Dict[str, DataStream] = {}
        self.stream_handlers: Dict[str, List[Callable]] = defaultdict(list)
        self.alert_conditions: Dict[str, List[AlertCondition]] = defaultdict(list)
        self.event_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.connected_clients: Dict[str, Any] = {}
        self.monitoring_tasks: Dict[str, asyncio.Task] = {}
        
        # Performance tracking
        self.performance_metrics = {
            "events_processed": 0,
            "alerts_triggered": 0,
            "active_streams": 0,
            "connected_clients": 0
        }
        
        # Redis connection for distributed monitoring
        self.redis_client: Optional[redis.Redis] = None
        self.is_distributed = False
        
        # Thread pool for CPU-intensive operations
        self.thread_pool = ThreadPoolExecutor(max_workers=8)
    
    async def initialize_redis(self, redis_url: str = "redis://localhost:6379"):
        """Initialize Redis for distributed monitoring"""
        try:
            self.redis_client = redis.from_url(redis_url)
            await self.redis_client.ping()
            self.is_distributed = True
            logger.info("Redis connected for distributed monitoring")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}. Using in-memory storage.")
            self.is_distributed = False
    
    async def create_data_stream(self, stream_config: Dict[str, Any]) -> DataStream:
        """Create a new data stream"""
        stream_id = f"stream_{int(time.time())}_{stream_config['name']}"
        
        stream = DataStream(
            stream_id=stream_id,
            name=stream_config["name"],
            data_type=stream_config["data_type"],
            source=stream_config["source"],
            update_frequency=stream_config.get("update_frequency", 1.0),
            retention_period=stream_config.get("retention_period", 24)
        )
        
        self.data_streams[stream_id] = stream
        self.performance_metrics["active_streams"] += 1
        
        # Start monitoring task if stream is active
        if stream.is_active:
            await self._start_stream_monitoring(stream_id)
        
        logger.info(f"Created data stream: {stream_id}")
        return stream
    
    async def _start_stream_monitoring(self, stream_id: str):
        """Start monitoring task for a data stream"""
        if stream_id in self.monitoring_tasks:
            logger.warning(f"Monitoring task already running for stream: {stream_id}")
            return
        
        stream = self.data_streams[stream_id]
        
        async def monitor_stream():
            while stream_id in self.data_streams and self.data_streams[stream_id].is_active:
                try:
                    # Simulate data collection - in production, this would connect to actual data sources
                    sample_data = await self._collect_stream_data(stream)
                    event = StreamEvent(
                        event_id=f"event_{int(time.time())}_{stream_id}",
                        stream_id=stream_id,
                        timestamp=datetime.utcnow(),
                        data=sample_data,
                        metadata={"source": stream.source, "collection_method": "simulated"}
                    )
                    
                    # Process the event
                    await self._process_stream_event(event)
                    
                    # Check alert conditions
                    await self._check_alert_conditions(stream_id, event)
                    
                    # Broadcast to connected clients
                    await self._broadcast_to_clients(stream_id, event)
                    
                except Exception as e:
                    logger.error(f"Stream monitoring error for {stream_id}: {str(e)}")
                
                await asyncio.sleep(stream.update_frequency)
        
        task = asyncio.create_task(monitor_stream())
        self.monitoring_tasks[stream_id] = task
        logger.info(f"Started monitoring for stream: {stream_id}")
    
    async def _collect_stream_data(self, stream: DataStream) -> Dict[str, Any]:
        """Collect data for a stream (simulated for demo)"""
        if stream.data_type == "performance":
            return {
                "cpu_usage": statistics.mean([psutil.cpu_percent() for _ in range(3)]),
                "memory_usage": psutil.virtual_memory().percent,
                "active_connections": len(self.connected_clients),
                "events_per_second": self.performance_metrics["events_processed"] / 60,
                "timestamp": datetime.utcnow().isoformat()
            }
        
        elif stream.data_type == "business":
            return {
                "active_campaigns": 15,
                "completed_tasks": 42,
                "pending_approvals": 3,
                "revenue_today": 12500.50,
                "conversion_rate": 0.045,
                "timestamp": datetime.utcnow().isoformat()
            }
        
        elif stream.data_type == "system":
            return {
                "service_status": "healthy",
                "database_connections": 25,
                "cache_hit_rate": 0.92,
                "queue_depth": 0,
                "uptime_hours": (time.time() - start_time) / 3600,
                "timestamp": datetime.utcnow().isoformat()
            }
        
        else:
            return {
                "value": time.time() % 100,
                "status": "active",
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def _process_stream_event(self, event: StreamEvent):
        """Process a stream event"""
        # Store in history
        self.event_history[event.stream_id].append(event)
        
        # Call stream handlers
        for handler in self.stream_handlers[event.stream_id]:
            try:
                await handler(event)
            except Exception as e:
                logger.error(f"Stream handler failed: {str(e)}")
        
        # Update performance metrics
        self.performance_metrics["events_processed"] += 1
        
        # Store in Redis if distributed
        if self.is_distributed and self.redis_client:
            try:
                key = f"stream:{event.stream_id}:events"
                await self.redis_client.lpush(key, event.json())
                await self.redis_client.ltrim(key, 0, 999)  # Keep last 1000 events
            except Exception as e:
                logger.error(f"Redis storage failed: {str(e)}")
    
    async def _check_alert_conditions(self, stream_id: str, event: StreamEvent):
        """Check alert conditions for a stream event"""
        conditions = self.alert_conditions.get(stream_id, [])
        
        for condition in conditions:
            try:
                field_value = event.data.get(condition.field)
                if field_value is None:
                    continue
                
                # Evaluate condition
                condition_met = False
                if condition.operator == ">":
                    condition_met = field_value > condition.value
                elif condition.operator == "<":
                    condition_met = field_value < condition.value
                elif condition.operator == ">=":
                    condition_met = field_value >= condition.value
                elif condition.operator == "<=":
                    condition_met = field_value <= condition.value
                elif condition.operator == "==":
                    condition_met = field_value == condition.value
                elif condition.operator == "!=":
                    condition_met = field_value != condition.value
                
                if condition_met:
                    await self._trigger_alert(stream_id, condition, event)
                    
            except Exception as e:
                logger.error(f"Alert condition evaluation failed: {str(e)}")
    
    async def _trigger_alert(self, stream_id: str, condition: AlertCondition, event: StreamEvent):
        """Trigger an alert"""
        from monitoring.alerts_handler import alerts_handler, AlertSeverity, AlertType
        
        alert_data = {
            "title": f"Alert: {condition.field} {condition.operator} {condition.value}",
            "description": f"Stream {stream_id} triggered alert condition",
            "severity": AlertSeverity(condition.severity),
            "alert_type": AlertType.SYSTEM,
            "source": f"realtime_monitor:{stream_id}",
            "metadata": {
                "stream_id": stream_id,
                "condition": condition.dict(),
                "event_data": event.data,
                "field": condition.field,
                "value": event.data.get(condition.field),
                "threshold": condition.value
            }
        }
        
        try:
            await alerts_handler.create_alert(**alert_data)
            self.performance_metrics["alerts_triggered"] += 1
            logger.info(f"Alert triggered for stream {stream_id}: {condition.field}")
        except Exception as e:
            logger.error(f"Alert creation failed: {str(e)}")
    
    async def _broadcast_to_clients(self, stream_id: str, event: StreamEvent):
        """Broadcast event to connected WebSocket clients"""
        if stream_id not in self.connected_clients:
            return
        
        disconnected = []
        message = {
            "type": "stream_update",
            "stream_id": stream_id,
            "event": event.dict(),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        for client_id, websocket in self.connected_clients[stream_id].items():
            try:
                await websocket.send(json.dumps(message, default=str))
            except Exception as e:
                logger.warning(f"Client {client_id} disconnected: {str(e)}")
                disconnected.append(client_id)
        
        # Remove disconnected clients
        for client_id in disconnected:
            del self.connected_clients[stream_id][client_id]
    
    def register_stream_handler(self, stream_id: str, handler: Callable):
        """Register handler for stream events"""
        self.stream_handlers[stream_id].append(handler)
        logger.info(f"Registered handler for stream: {stream_id}")
    
    async def add_alert_condition(self, stream_id: str, condition_config: Dict[str, Any]) -> AlertCondition:
        """Add alert condition to a stream"""
        condition_id = f"cond_{int(time.time())}_{stream_id}"
        
        condition = AlertCondition(
            condition_id=condition_id,
            field=condition_config["field"],
            operator=condition_config["operator"],
            value=condition_config["value"],
            severity=condition_config["severity"],
            cooldown=condition_config.get("cooldown", 60)
        )
        
        self.alert_conditions[stream_id].append(condition)
        logger.info(f"Added alert condition for stream {stream_id}: {condition.field}")
        
        return condition
    
    async def connect_client(self, stream_id: str, websocket, client_id: str):
        """Connect a client to a stream"""
        if stream_id not in self.connected_clients:
            self.connected_clients[stream_id] = {}
        
        self.connected_clients[stream_id][client_id] = websocket
        self.performance_metrics["connected_clients"] += 1
        
        logger.info(f"Client {client_id} connected to stream {stream_id}")
    
    async def disconnect_client(self, stream_id: str, client_id: str):
        """Disconnect a client from a stream"""
        if stream_id in self.connected_clients and client_id in self.connected_clients[stream_id]:
            del self.connected_clients[stream_id][client_id]
            self.performance_metrics["connected_clients"] -= 1
            
            # Remove stream if no clients
            if not self.connected_clients[stream_id]:
                del self.connected_clients[stream_id]
            
            logger.info(f"Client {client_id} disconnected from stream {stream_id}")
    
    async def get_stream_events(self, stream_id: str, limit: int = 100) -> List[StreamEvent]:
        """Get recent events for a stream"""
        if stream_id not in self.event_history:
            return []
        
        events = list(self.event_history[stream_id])
        return events[-limit:]
    
    async def get_stream_statistics(self, stream_id: str, hours: int = 1) -> Dict[str, Any]:
        """Get statistics for a stream"""
        events = await self.get_stream_events(stream_id, 1000)  # Get up to 1000 events
        
        if not events:
            return {
                "stream_id": stream_id,
                "event_count": 0,
                "message": "No events found"
            }
        
        # Filter by time range
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        recent_events = [e for e in events if e.timestamp >= cutoff_time]
        
        if not recent_events:
            return {
                "stream_id": stream_id,
                "event_count": 0,
                "message": "No recent events"
            }
        
        # Calculate basic statistics
        numeric_data = {}
        for event in recent_events:
            for key, value in event.data.items():
                if isinstance(value, (int, float)):
                    if key not in numeric_data:
                        numeric_data[key] = []
                    numeric_data[key].append(value)
        
        statistics_data = {}
        for key, values in numeric_data.items():
            if values:
                statistics_data[key] = {
                    "count": len(values),
                    "mean": statistics.mean(values),
                    "median": statistics.median(values),
                    "min": min(values),
                    "max": max(values),
                    "std_dev": statistics.stdev(values) if len(values) > 1 else 0
                }
        
        return {
            "stream_id": stream_id,
            "timeframe_hours": hours,
            "event_count": len(recent_events),
            "events_per_minute": len(recent_events) / (hours * 60),
            "last_event_time": recent_events[-1].timestamp.isoformat(),
            "field_statistics": statistics_data,
            "active_alert_conditions": len(self.alert_conditions.get(stream_id, [])),
            "connected_clients": len(self.connected_clients.get(stream_id, {})),
            "statistics_generated": datetime.utcnow().isoformat()
        }
    
    async def create_performance_dashboard(self) -> Dict[str, Any]:
        """Create performance dashboard data"""
        dashboard_data = {
            "overview": {
                "active_streams": len(self.data_streams),
                "total_events_processed": self.performance_metrics["events_processed"],
                "alerts_triggered": self.performance_metrics["alerts_triggered"],
                "connected_clients": self.performance_metrics["connected_clients"],
                "uptime_seconds": time.time() - start_time
            },
            "streams": [],
            "system_health": await self._get_system_health(),
            "recommendations": await self._generate_dashboard_recommendations()
        }
        
        # Add stream details
        for stream_id, stream in self.data_streams.items():
            stream_stats = await self.get_stream_statistics(stream_id, 1)
            dashboard_data["streams"].append({
                "stream_id": stream_id,
                "name": stream.name,
                "data_type": stream.data_type,
                "is_active": stream.is_active,
                "update_frequency": stream.update_frequency,
                "statistics": stream_stats
            })
        
        return dashboard_data
    
    async def _get_system_health(self) -> Dict[str, Any]:
        """Get system health information"""
        try:
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": memory.percent,
                "disk_percent": disk.percent,
                "active_tasks": len(self.monitoring_tasks),
                "thread_pool_workers": self.thread_pool._max_workers,
                "redis_connected": self.is_distributed,
                "health_status": "healthy" if len(self.monitoring_tasks) == len(self.data_streams) else "degraded"
            }
        except Exception as e:
            logger.error(f"System health check failed: {str(e)}")
            return {"health_status": "unknown", "error": str(e)}
    
    async def _generate_dashboard_recommendations(self) -> List[Dict[str, Any]]:
        """Generate dashboard recommendations"""
        recommendations = []
        
        # Check for inactive streams
        inactive_streams = [s for s in self.data_streams.values() if not s.is_active]
        if inactive_streams:
            recommendations.append({
                "type": "stream_optimization",
                "title": "Activate Inactive Streams",
                "description": f"{len(inactive_streams)} streams are inactive",
                "action": "Review and activate unused streams",
                "priority": "low"
            })
        
        # Check for streams with no alert conditions
        streams_without_alerts = [
            s for s in self.data_streams.values() 
            if s.stream_id not in self.alert_conditions or not self.alert_conditions[s.stream_id]
        ]
        if streams_without_alerts:
            recommendations.append({
                "type": "alert_optimization",
                "title": "Add Alert Conditions",
                "description": f"{len(streams_without_alerts)} streams have no alert conditions",
                "action": "Configure alert conditions for critical streams",
                "priority": "medium"
            })
        
        # Check performance
        events_per_second = self.performance_metrics["events_processed"] / max(time.time() - start_time, 1)
        if events_per_second > 1000:
            recommendations.append({
                "type": "performance",
                "title": "High Event Volume",
                "description": f"Processing {events_per_second:.1f} events per second",
                "action": "Consider scaling or optimizing event processing",
                "priority": "high"
            })
        
        return recommendations
    
    async def stop_stream(self, stream_id: str):
        """Stop a data stream"""
        if stream_id in self.monitoring_tasks:
            self.monitoring_tasks[stream_id].cancel()
            del self.monitoring_tasks[stream_id]
        
        if stream_id in self.data_streams:
            self.data_streams[stream_id].is_active = False
        
        logger.info(f"Stopped stream: {stream_id}")
    
    async def start_stream(self, stream_id: str):
        """Start a data stream"""
        if stream_id not in self.data_streams:
            raise ValueError(f"Stream {stream_id} not found")
        
        stream = self.data_streams[stream_id]
        stream.is_active = True
        
        await self._start_stream_monitoring(stream_id)
        logger.info(f"Started stream: {stream_id}")
    
    def get_monitor_metrics(self) -> Dict[str, Any]:
        """Get real-time monitor performance metrics"""
        active_tasks = len(self.monitoring_tasks)
        total_handlers = sum(len(handlers) for handlers in self.stream_handlers.values())
        total_conditions = sum(len(conditions) for conditions in self.alert_conditions.values())
        
        return {
            **self.performance_metrics,
            "data_streams_count": len(self.data_streams),
            "active_monitoring_tasks": active_tasks,
            "total_stream_handlers": total_handlers,
            "total_alert_conditions": total_conditions,
            "event_history_size": sum(len(events) for events in self.event_history.values()),
            "redis_connected": self.is_distributed,
            "timestamp": datetime.utcnow().isoformat()
        }


# Global start time for uptime calculation
start_time = time.time()

# Global real-time monitor instance
realtime_monitor = RealTimeMonitorV16()


async def example_stream_handler(event: StreamEvent):
    """Example stream handler for testing"""
    print(f"📊 Stream Event: {event.stream_id} - {event.data}")


async def main():
    """Test harness for Real-time Monitor"""
    print("🔍 Real-time Monitor V16 - Test Harness")
    
    # Initialize Redis (optional)
    await realtime_monitor.initialize_redis()
    
    # Create sample data streams
    performance_stream = await realtime_monitor.create_data_stream({
        "name": "System Performance",
        "data_type": "performance",
        "source": "system_monitor",
        "update_frequency": 2.0,
        "retention_period": 24
    })
    
    business_stream = await realtime_monitor.create_data_stream({
        "name": "Business Metrics",
        "data_type": "business",
        "source": "business_intelligence",
        "update_frequency": 5.0,
        "retention_period": 48
    })
    
    # Register handlers
    realtime_monitor.register_stream_handler(performance_stream.stream_id, example_stream_handler)
    
    # Add alert conditions
    await realtime_monitor.add_alert_condition(performance_stream.stream_id, {
        "field": "cpu_usage",
        "operator": ">",
        "value": 80.0,
        "severity": "high",
        "cooldown": 300
    })
    
    print(f"✅ Created {len(realtime_monitor.data_streams)} data streams")
    
    # Wait for some events to be processed
    await asyncio.sleep(6)
    
    # Get stream statistics
    stats = await realtime_monitor.get_stream_statistics(performance_stream.stream_id)
    print(f"📈 Stream Statistics: {stats['event_count']} events")
    
    # Get performance dashboard
    dashboard = await realtime_monitor.create_performance_dashboard()
    print(f"📊 Dashboard: {dashboard['overview']['active_streams']} active streams")
    
    # Get monitor metrics
    metrics = realtime_monitor.get_monitor_metrics()
    print("🔧 Monitor Metrics:", metrics)
    
    # Cleanup
    await realtime_monitor.stop_stream(performance_stream.stream_id)
    await realtime_monitor.stop_stream(business_stream.stream_id)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
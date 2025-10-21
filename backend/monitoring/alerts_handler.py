"""
Alerts Handler V16 - Advanced alert management and notification system
for the Shooting Star V16 Engine.
"""

from typing import Dict, List, Any, Optional, Callable
from pydantic import BaseModel, Field
from enum import Enum
import asyncio
import logging
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import json
import time
from datetime import datetime, timedelta
from collections import defaultdict, deque
import statistics

logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AlertType(Enum):
    SYSTEM = "system"
    PERFORMANCE = "performance"
    SECURITY = "security"
    BUSINESS = "business"
    CUSTOM = "custom"

class AlertStatus(Enum):
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"

class Alert(BaseModel):
    """Alert model"""
    alert_id: str
    title: str
    description: str
    severity: AlertSeverity
    alert_type: AlertType
    source: str
    status: AlertStatus
    created_at: datetime
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    assigned_to: Optional[str] = None

class NotificationChannel(BaseModel):
    """Notification channel configuration"""
    channel_id: str
    channel_type: str  # email, slack, webhook, etc.
    config: Dict[str, Any]
    enabled: bool = True
    severity_filter: List[AlertSeverity] = Field(default_factory=list)

class AlertRule(BaseModel):
    """Alert rule configuration"""
    rule_id: str
    name: str
    condition: str
    severity: AlertSeverity
    cooldown_minutes: int
    enabled: bool = True
    notification_channels: List[str] = Field(default_factory=list)

class AlertsHandlerV16:
    """
    Advanced alert management and notification system for V16
    """
    
    def __init__(self):
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.notification_channels: Dict[str, NotificationChannel] = {}
        self.alert_rules: Dict[str, AlertRule] = {}
        self.alert_cooldowns: Dict[str, datetime] = {}
        self.alert_handlers: Dict[AlertType, List[Callable]] = defaultdict(list)
        
        # Statistics
        self.alert_statistics = {
            "total_alerts_triggered": 0,
            "alerts_by_severity": defaultdict(int),
            "alerts_by_type": defaultdict(int),
            "average_resolution_time": 0.0
        }
        
        # Register default notification channels
        self._register_default_channels()
        
        # Register default alert rules
        self._register_default_rules()
    
    def _register_default_channels(self):
        """Register default notification channels"""
        # Email channel
        email_channel = NotificationChannel(
            channel_id="email_default",
            channel_type="email",
            config={
                "smtp_server": "localhost",
                "smtp_port": 587,
                "from_address": "alerts@shootingstar.com",
                "to_addresses": ["admin@shootingstar.com"]
            },
            severity_filter=[AlertSeverity.HIGH, AlertSeverity.CRITICAL]
        )
        self.notification_channels[email_channel.channel_id] = email_channel
        
        # Slack channel (example)
        slack_channel = NotificationChannel(
            channel_id="slack_default",
            channel_type="slack",
            config={
                "webhook_url": "https://hooks.slack.com/services/...",
                "channel": "#alerts"
            },
            severity_filter=[AlertSeverity.MEDIUM, AlertSeverity.HIGH, AlertSeverity.CRITICAL]
        )
        self.notification_channels[slack_channel.channel_id] = slack_channel
        
        logger.info("Registered default notification channels")
    
    def _register_default_rules(self):
        """Register default alert rules"""
        default_rules = [
            AlertRule(
                rule_id="high_cpu_usage",
                name="High CPU Usage",
                condition="cpu_usage > 90",
                severity=AlertSeverity.HIGH,
                cooldown_minutes=10,
                notification_channels=["email_default", "slack_default"]
            ),
            AlertRule(
                rule_id="high_memory_usage",
                name="High Memory Usage",
                condition="memory_usage > 90",
                severity=AlertSeverity.HIGH,
                cooldown_minutes=10,
                notification_channels=["email_default", "slack_default"]
            ),
            AlertRule(
                rule_id="high_disk_usage",
                name="High Disk Usage",
                condition="disk_usage > 95",
                severity=AlertSeverity.CRITICAL,
                cooldown_minutes=30,
                notification_channels=["email_default", "slack_default"]
            ),
            AlertRule(
                rule_id="high_error_rate",
                name="High Error Rate",
                condition="error_rate > 0.1",
                severity=AlertSeverity.HIGH,
                cooldown_minutes=5,
                notification_channels=["slack_default"]
            )
        ]
        
        for rule in default_rules:
            self.alert_rules[rule.rule_id] = rule
        
        logger.info(f"Registered {len(default_rules)} default alert rules")
    
    async def create_alert(self, title: str, description: str, severity: AlertSeverity,
                         alert_type: AlertType, source: str, metadata: Optional[Dict[str, Any]] = None) -> Alert:
        """
        Create and trigger a new alert
        """
        try:
            alert_id = f"alert_{int(time.time())}_{source}"
            
            alert = Alert(
                alert_id=alert_id,
                title=title,
                description=description,
                severity=severity,
                alert_type=alert_type,
                source=source,
                status=AlertStatus.ACTIVE,
                created_at=datetime.utcnow(),
                metadata=metadata or {}
            )
            
            # Check cooldown
            cooldown_key = f"{source}_{severity.value}"
            if cooldown_key in self.alert_cooldowns:
                cooldown_end = self.alert_cooldowns[cooldown_key]
                if datetime.utcnow() < cooldown_end:
                    logger.info(f"Alert suppressed due to cooldown: {alert_id}")
                    return alert
            
            # Store alert
            self.active_alerts[alert_id] = alert
            self.alert_history.append(alert)
            
            # Update statistics
            self.alert_statistics["total_alerts_triggered"] += 1
            self.alert_statistics["alerts_by_severity"][severity] += 1
            self.alert_statistics["alerts_by_type"][alert_type] += 1
            
            # Set cooldown
            matching_rules = [r for r in self.alert_rules.values() 
                            if r.enabled and self._matches_rule(alert, r)]
            
            if matching_rules:
                # Use the shortest cooldown from matching rules
                cooldown_minutes = min(r.cooldown_minutes for r in matching_rules)
                self.alert_cooldowns[cooldown_key] = datetime.utcnow() + timedelta(minutes=cooldown_minutes)
            
            # Send notifications
            await self._send_notifications(alert, matching_rules)
            
            # Call alert handlers
            await self._call_alert_handlers(alert)
            
            logger.info(f"Alert created: {alert_id} - {title} ({severity.value})")
            return alert
            
        except Exception as e:
            logger.error(f"Alert creation failed: {str(e)}")
            raise
    
    def _matches_rule(self, alert: Alert, rule: AlertRule) -> bool:
        """Check if alert matches a rule"""
        # Simple condition matching - in production, use a proper expression evaluator
        condition_lower = rule.condition.lower()
        
        if "cpu_usage" in condition_lower and alert.alert_type == AlertType.SYSTEM:
            cpu_usage = alert.metadata.get("cpu_usage", 0)
            if ">" in condition_lower:
                threshold = float(condition_lower.split(">")[1].strip())
                return cpu_usage > threshold
        
        elif "memory_usage" in condition_lower and alert.alert_type == AlertType.SYSTEM:
            memory_usage = alert.metadata.get("memory_usage", 0)
            if ">" in condition_lower:
                threshold = float(condition_lower.split(">")[1].strip())
                return memory_usage > threshold
        
        elif "disk_usage" in condition_lower and alert.alert_type == AlertType.SYSTEM:
            disk_usage = alert.metadata.get("disk_usage", 0)
            if ">" in condition_lower:
                threshold = float(condition_lower.split(">")[1].strip())
                return disk_usage > threshold
        
        elif "error_rate" in condition_lower and alert.alert_type == AlertType.PERFORMANCE:
            error_rate = alert.metadata.get("error_rate", 0)
            if ">" in condition_lower:
                threshold = float(condition_lower.split(">")[1].strip())
                return error_rate > threshold
        
        return False
    
    async def _send_notifications(self, alert: Alert, matching_rules: List[AlertRule]):
        """Send notifications for alert"""
        channels_to_notify = set()
        
        # Collect channels from matching rules
        for rule in matching_rules:
            channels_to_notify.update(rule.notification_channels)
        
        # Also include channels that match severity
        for channel in self.notification_channels.values():
            if channel.enabled and alert.severity in channel.severity_filter:
                channels_to_notify.add(channel.channel_id)
        
        # Send notifications
        for channel_id in channels_to_notify:
            channel = self.notification_channels.get(channel_id)
            if channel:
                try:
                    await self._send_notification(channel, alert)
                except Exception as e:
                    logger.error(f"Notification failed for channel {channel_id}: {str(e)}")
    
    async def _send_notification(self, channel: NotificationChannel, alert: Alert):
        """Send notification via specific channel"""
        if channel.channel_type == "email":
            await self._send_email_notification(channel, alert)
        elif channel.channel_type == "slack":
            await self._send_slack_notification(channel, alert)
        elif channel.channel_type == "webhook":
            await self._send_webhook_notification(channel, alert)
        else:
            logger.warning(f"Unsupported channel type: {channel.channel_type}")
    
    async def _send_email_notification(self, channel: NotificationChannel, alert: Alert):
        """Send email notification"""
        try:
            config = channel.config
            
            # Create message
            msg = MimeMultipart()
            msg['From'] = config['from_address']
            msg['To'] = ', '.join(config['to_addresses'])
            msg['Subject'] = f"🚨 {alert.severity.value.upper()} Alert: {alert.title}"
            
            # Create email body
            body = f"""
            Alert Details:
            
            Title: {alert.title}
            Description: {alert.description}
            Severity: {alert.severity.value}
            Type: {alert.alert_type.value}
            Source: {alert.source}
            Time: {alert.created_at.strftime('%Y-%m-%d %H:%M:%S UTC')}
            
            Metadata:
            {json.dumps(alert.metadata, indent=2)}
            
            Please take appropriate action.
            
            --
            Shooting Star V16 Monitoring System
            """
            
            msg.attach(MimeText(body, 'plain'))
            
            # Send email
            with smtplib.SMTP(config['smtp_server'], config['smtp_port']) as server:
                server.starttls()
                # server.login(config['username'], config['password'])  # If authentication required
                server.send_message(msg)
            
            logger.info(f"Email notification sent for alert {alert.alert_id}")
            
        except Exception as e:
            logger.error(f"Email notification failed: {str(e)}")
            raise
    
    async def _send_slack_notification(self, channel: NotificationChannel, alert: Alert):
        """Send Slack notification"""
        try:
            # This would integrate with Slack webhook API
            # For now, simulate the notification
            config = channel.config
            
            severity_emoji = {
                AlertSeverity.LOW: "ℹ️",
                AlertSeverity.MEDIUM: "⚠️",
                AlertSeverity.HIGH: "🚨",
                AlertSeverity.CRITICAL: "🔥"
            }
            
            message = {
                "channel": config.get("channel", "#alerts"),
                "username": "Shooting Star Alerts",
                "text": f"{severity_emoji.get(alert.severity, '⚠️')} *{alert.title}*",
                "attachments": [
                    {
                        "color": self._get_slack_color(alert.severity),
                        "fields": [
                            {"title": "Description", "value": alert.description, "short": False},
                            {"title": "Severity", "value": alert.severity.value, "short": True},
                            {"title": "Source", "value": alert.source, "short": True},
                            {"title": "Time", "value": alert.created_at.strftime('%H:%M:%S UTC'), "short": True}
                        ]
                    }
                ]
            }
            
            # In production: requests.post(config['webhook_url'], json=message)
            logger.info(f"Slack notification prepared for alert {alert.alert_id}")
            
        except Exception as e:
            logger.error(f"Slack notification failed: {str(e)}")
    
    async def _send_webhook_notification(self, channel: NotificationChannel, alert: Alert):
        """Send webhook notification"""
        try:
            config = channel.config
            
            webhook_data = {
                "alert_id": alert.alert_id,
                "title": alert.title,
                "description": alert.description,
                "severity": alert.severity.value,
                "type": alert.alert_type.value,
                "source": alert.source,
                "timestamp": alert.created_at.isoformat(),
                "metadata": alert.metadata
            }
            
            # In production: requests.post(config['webhook_url'], json=webhook_data)
            logger.info(f"Webhook notification prepared for alert {alert.alert_id}")
            
        except Exception as e:
            logger.error(f"Webhook notification failed: {str(e)}")
    
    def _get_slack_color(self, severity: AlertSeverity) -> str:
        """Get Slack attachment color based on severity"""
        colors = {
            AlertSeverity.LOW: "#36a64f",      # Green
            AlertSeverity.MEDIUM: "#f2c744",   # Yellow
            AlertSeverity.HIGH: "#e67e22",     # Orange
            AlertSeverity.CRITICAL: "#e74c3c"  # Red
        }
        return colors.get(severity, "#95a5a6")
    
    async def _call_alert_handlers(self, alert: Alert):
        """Call registered alert handlers"""
        handlers = self.alert_handlers.get(alert.alert_type, [])
        
        for handler in handlers:
            try:
                await handler(alert)
            except Exception as e:
                logger.error(f"Alert handler failed: {str(e)}")
    
    def register_alert_handler(self, alert_type: AlertType, handler: Callable):
        """Register alert handler for specific alert type"""
        self.alert_handlers[alert_type].append(handler)
        logger.info(f"Registered alert handler for {alert_type.value}")
    
    async def acknowledge_alert(self, alert_id: str, user_id: str) -> Optional[Alert]:
        """Acknowledge an alert"""
        if alert_id not in self.active_alerts:
            return None
        
        alert = self.active_alerts[alert_id]
        alert.status = AlertStatus.ACKNOWLEDGED
        alert.acknowledged_at = datetime.utcnow()
        alert.assigned_to = user_id
        
        logger.info(f"Alert {alert_id} acknowledged by {user_id}")
        return alert
    
    async def resolve_alert(self, alert_id: str, resolution_notes: str = "") -> Optional[Alert]:
        """Resolve an alert"""
        if alert_id not in self.active_alerts:
            return None
        
        alert = self.active_alerts[alert_id]
        alert.status = AlertStatus.RESOLVED
        alert.resolved_at = datetime.utcnow()
        
        if resolution_notes:
            alert.metadata["resolution_notes"] = resolution_notes
        
        # Calculate resolution time for statistics
        resolution_time = (alert.resolved_at - alert.created_at).total_seconds()
        
        # Update average resolution time
        total_alerts = self.alert_statistics["total_alerts_triggered"]
        current_avg = self.alert_statistics["average_resolution_time"]
        self.alert_statistics["average_resolution_time"] = (
            (current_avg * (total_alerts - 1) + resolution_time) / total_alerts
        )
        
        # Remove from active alerts
        del self.active_alerts[alert_id]
        
        logger.info(f"Alert {alert_id} resolved in {resolution_time:.1f} seconds")
        return alert
    
    async def get_active_alerts(self, severity: Optional[AlertSeverity] = None,
                              alert_type: Optional[AlertType] = None) -> List[Alert]:
        """Get active alerts with optional filtering"""
        alerts = list(self.active_alerts.values())
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        if alert_type:
            alerts = [a for a in alerts if a.alert_type == alert_type]
        
        return sorted(alerts, key=lambda x: x.created_at, reverse=True)
    
    async def get_alert_history(self, hours: int = 24, severity: Optional[AlertSeverity] = None,
                              alert_type: Optional[AlertType] = None) -> List[Alert]:
        """Get alert history with optional filtering"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        alerts = [a for a in self.alert_history if a.created_at >= cutoff_time]
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        if alert_type:
            alerts = [a for a in alerts if a.alert_type == alert_type]
        
        return sorted(alerts, key=lambda x: x.created_at, reverse=True)
    
    async def get_alert_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """Get alert statistics for specified time range"""
        recent_alerts = await self.get_alert_history(hours)
        
        if not recent_alerts:
            return {
                "timeframe_hours": hours,
                "total_alerts": 0,
                "alerts_by_severity": {},
                "alerts_by_type": {},
                "resolution_rate": 0.0,
                "average_resolution_time": 0.0
            }
        
        # Calculate statistics
        severity_counts = defaultdict(int)
        type_counts = defaultdict(int)
        resolved_alerts = 0
        total_resolution_time = 0
        
        for alert in recent_alerts:
            severity_counts[alert.severity] += 1
            type_counts[alert.alert_type] += 1
            
            if alert.status == AlertStatus.RESOLVED and alert.resolved_at:
                resolved_alerts += 1
                resolution_time = (alert.resolved_at - alert.created_at).total_seconds()
                total_resolution_time += resolution_time
        
        resolution_rate = resolved_alerts / len(recent_alerts)
        avg_resolution_time = total_resolution_time / max(resolved_alerts, 1)
        
        return {
            "timeframe_hours": hours,
            "total_alerts": len(recent_alerts),
            "alerts_by_severity": dict(severity_counts),
            "alerts_by_type": dict(type_counts),
            "resolution_rate": round(resolution_rate, 3),
            "average_resolution_time": round(avg_resolution_time, 1),
            "active_alerts_count": len(self.active_alerts),
            "statistics_generated": datetime.utcnow().isoformat()
        }
    
    async def create_notification_channel(self, channel_data: Dict[str, Any]) -> NotificationChannel:
        """Create a new notification channel"""
        channel_id = channel_data["channel_id"]
        
        channel = NotificationChannel(
            channel_id=channel_id,
            channel_type=channel_data["channel_type"],
            config=channel_data["config"],
            enabled=channel_data.get("enabled", True),
            severity_filter=[AlertSeverity(s) for s in channel_data.get("severity_filter", [])]
        )
        
        self.notification_channels[channel_id] = channel
        logger.info(f"Created notification channel: {channel_id}")
        
        return channel
    
    async def create_alert_rule(self, rule_data: Dict[str, Any]) -> AlertRule:
        """Create a new alert rule"""
        rule_id = rule_data["rule_id"]
        
        rule = AlertRule(
            rule_id=rule_id,
            name=rule_data["name"],
            condition=rule_data["condition"],
            severity=AlertSeverity(rule_data["severity"]),
            cooldown_minutes=rule_data["cooldown_minutes"],
            enabled=rule_data.get("enabled", True),
            notification_channels=rule_data.get("notification_channels", [])
        )
        
        self.alert_rules[rule_id] = rule
        logger.info(f"Created alert rule: {rule_id}")
        
        return rule
    
    def get_handler_metrics(self) -> Dict[str, Any]:
        """Get alerts handler performance metrics"""
        total_channels = len(self.notification_channels)
        enabled_channels = len([c for c in self.notification_channels.values() if c.enabled])
        total_rules = len(self.alert_rules)
        enabled_rules = len([r for r in self.alert_rules.values() if r.enabled])
        
        return {
            "active_alerts": len(self.active_alerts),
            "total_alert_history": len(self.alert_history),
            "notification_channels": {
                "total": total_channels,
                "enabled": enabled_channels,
                "types": list(set(c.channel_type for c in self.notification_channels.values()))
            },
            "alert_rules": {
                "total": total_rules,
                "enabled": enabled_rules
            },
            "alert_handlers_registered": sum(len(handlers) for handlers in self.alert_handlers.values()),
            "average_resolution_time": round(self.alert_statistics["average_resolution_time"], 1),
            "timestamp": datetime.utcnow().isoformat()
        }


# Global alerts handler instance
alerts_handler = AlertsHandlerV16()


async def example_alert_handler(alert: Alert):
    """Example alert handler for testing"""
    print(f"🔄 Alert Handler: Processing {alert.alert_id} - {alert.title}")


async def main():
    """Test harness for Alerts Handler V16"""
    print("🚨 Alerts Handler V16 - Test Harness")
    
    # Register example alert handler
    alerts_handler.register_alert_handler(AlertType.SYSTEM, example_alert_handler)
    
    # Create test alerts
    alert1 = await alerts_handler.create_alert(
        title="High CPU Usage Detected",
        description="CPU usage has exceeded 90% for more than 5 minutes",
        severity=AlertSeverity.HIGH,
        alert_type=AlertType.SYSTEM,
        source="system_monitor",
        metadata={"cpu_usage": 92.5, "duration_minutes": 7}
    )
    
    alert2 = await alerts_handler.create_alert(
        title="Database Connection Slow",
        description="Database response time has increased significantly",
        severity=AlertSeverity.MEDIUM,
        alert_type=AlertType.PERFORMANCE,
        source="performance_monitor",
        metadata={"response_time": 2.8, "threshold": 1.0}
    )
    
    print(f"✅ Created {len(alerts_handler.active_alerts)} alerts")
    
    # Get active alerts
    active_alerts = await alerts_handler.get_active_alerts()
    print(f"📋 Active Alerts: {len(active_alerts)}")
    
    # Acknowledge an alert
    if active_alerts:
        acknowledged = await alerts_handler.acknowledge_alert(active_alerts[0].alert_id, "user_001")
        print(f"✅ Acknowledged alert: {acknowledged.alert_id if acknowledged else 'None'}")
    
    # Get statistics
    stats = await alerts_handler.get_alert_statistics(1)
    print(f"📊 Alert Statistics: {stats['total_alerts']} alerts in last hour")
    
    # Get handler metrics
    metrics = alerts_handler.get_handler_metrics()
    print("🔧 Handler Metrics:", metrics)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
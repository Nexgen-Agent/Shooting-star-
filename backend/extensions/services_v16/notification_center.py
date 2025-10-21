"""
Notification Center V16 - Advanced notification management and multi-channel delivery
for the Shooting Star V16 service layer.
"""

from typing import Dict, List, Any, Optional, Callable, Union
from pydantic import BaseModel, Field
from enum import Enum
import asyncio
import logging
import smtplib
import aiohttp
import json
import time
from datetime import datetime, timedelta
from collections import defaultdict, deque
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
from email.mime.application import MimeApplication
import jinja2
import markdown

logger = logging.getLogger(__name__)

class NotificationType(Enum):
    ALERT = "alert"
    TASK = "task"
    SYSTEM = "system"
    BUSINESS = "business"
    PERSONAL = "personal"
    BROADCAST = "broadcast"

class NotificationPriority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"

class DeliveryChannel(Enum):
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    IN_APP = "in_app"
    SMS = "sms"
    PUSH = "push"

class NotificationStatus(Enum):
    PENDING = "pending"
    SENT = "sent"
    DELIVERED = "delivered"
    FAILED = "failed"
    READ = "read"

class NotificationTemplate(BaseModel):
    """Notification template model"""
    template_id: str
    name: str
    description: str
    notification_type: NotificationType
    channels: List[DeliveryChannel]
    subject_template: str
    body_template: str
    variables: List[str] = Field(default_factory=list)
    is_active: bool = True
    created_at: datetime
    updated_at: datetime

class Notification(BaseModel):
    """Notification model"""
    notification_id: str
    template_id: str
    recipient: str
    notification_type: NotificationType
    priority: NotificationPriority
    channels: List[DeliveryChannel]
    subject: str
    body: str
    data: Dict[str, Any] = Field(default_factory=dict)
    status: NotificationStatus
    delivery_attempts: int = 0
    created_at: datetime
    sent_at: Optional[datetime] = None
    read_at: Optional[datetime] = None

class ChannelConfig(BaseModel):
    """Channel configuration model"""
    channel: DeliveryChannel
    config: Dict[str, Any]
    enabled: bool = True
    rate_limit: Optional[int] = None  # messages per minute

class NotificationCenterV16:
    """
    Advanced notification management and multi-channel delivery for V16
    """
    
    def __init__(self):
        self.templates: Dict[str, NotificationTemplate] = {}
        self.notifications: Dict[str, Notification] = {}
        self.channel_configs: Dict[DeliveryChannel, ChannelConfig] = {}
        self.delivery_handlers: Dict[DeliveryChannel, Callable] = {}
        self.notification_history: deque = deque(maxlen=10000)
        
        # Rate limiting
        self.rate_limits: Dict[DeliveryChannel, deque] = defaultdict(lambda: deque(maxlen=100))
        self.delivery_queue: asyncio.Queue = asyncio.Queue()
        
        # Performance tracking
        self.performance_metrics = {
            "notifications_sent": 0,
            "notifications_failed": 0,
            "delivery_attempts": 0,
            "average_delivery_time": 0.0
        }
        
        # Template engine
        self.template_env = jinja2.Environment(
            loader=jinja2.DictLoader({}),
            autoescape=jinja2.select_autoescape(['html', 'xml'])
        )
        
        # Register default channel handlers
        self._register_default_handlers()
        
        # Start delivery worker
        self.delivery_worker_task = asyncio.create_task(self._delivery_worker())
    
    def _register_default_handlers(self):
        """Register default delivery channel handlers"""
        self.register_channel_handler(DeliveryChannel.EMAIL, self._deliver_email)
        self.register_channel_handler(DeliveryChannel.SLACK, self._deliver_slack)
        self.register_channel_handler(DeliveryChannel.WEBHOOK, self._deliver_webhook)
        self.register_channel_handler(DeliveryChannel.IN_APP, self._deliver_in_app)
        
        logger.info("Registered default channel handlers")
    
    def register_channel_handler(self, channel: DeliveryChannel, handler: Callable):
        """Register handler for delivery channel"""
        self.delivery_handlers[channel] = handler
        logger.info(f"Registered handler for channel: {channel.value}")
    
    async def configure_channel(self, channel: DeliveryChannel, config: Dict[str, Any]):
        """Configure a delivery channel"""
        channel_config = ChannelConfig(
            channel=channel,
            config=config,
            enabled=config.get("enabled", True),
            rate_limit=config.get("rate_limit")
        )
        
        self.channel_configs[channel] = channel_config
        logger.info(f"Configured channel: {channel.value}")
    
    async def create_template(self, template_data: Dict[str, Any]) -> NotificationTemplate:
        """Create a notification template"""
        template_id = f"template_{int(time.time())}_{template_data['name']}"
        now = datetime.utcnow()
        
        template = NotificationTemplate(
            template_id=template_id,
            name=template_data["name"],
            description=template_data["description"],
            notification_type=NotificationType(template_data["notification_type"]),
            channels=[DeliveryChannel(c) for c in template_data["channels"]],
            subject_template=template_data["subject_template"],
            body_template=template_data["body_template"],
            variables=template_data.get("variables", []),
            created_at=now,
            updated_at=now
        )
        
        self.templates[template_id] = template
        
        # Add to template engine
        self.template_env.loader.mapping[template_id] = template.body_template
        
        logger.info(f"Created template: {template_id} - {template.name}")
        return template
    
    async def send_notification(self, template_id: str, recipient: str,
                              data: Dict[str, Any], 
                              priority: NotificationPriority = NotificationPriority.MEDIUM,
                              channels: Optional[List[DeliveryChannel]] = None) -> Notification:
        """Send a notification using a template"""
        if template_id not in self.templates:
            raise ValueError(f"Template {template_id} not found")
        
        template = self.templates[template_id]
        
        # Render templates
        subject = await self._render_template(template.subject_template, data)
        body = await self._render_template(template.body_template, data)
        
        # Use template channels if not specified
        if channels is None:
            channels = template.channels
        
        # Filter enabled channels
        enabled_channels = []
        for channel in channels:
            channel_config = self.channel_configs.get(channel)
            if channel_config and channel_config.enabled:
                enabled_channels.append(channel)
        
        if not enabled_channels:
            raise ValueError("No enabled channels available for notification")
        
        notification_id = f"notif_{int(time.time())}_{recipient}"
        notification = Notification(
            notification_id=notification_id,
            template_id=template_id,
            recipient=recipient,
            notification_type=template.notification_type,
            priority=priority,
            channels=enabled_channels,
            subject=subject,
            body=body,
            data=data,
            status=NotificationStatus.PENDING,
            created_at=datetime.utcnow()
        )
        
        self.notifications[notification_id] = notification
        self.notification_history.append(notification)
        
        # Add to delivery queue
        await self.delivery_queue.put(notification)
        
        logger.info(f"Queued notification: {notification_id} for {recipient}")
        return notification
    
    async def _render_template(self, template: str, data: Dict[str, Any]) -> str:
        """Render a template with data"""
        try:
            jinja_template = self.template_env.from_string(template)
            return jinja_template.render(**data)
        except Exception as e:
            logger.error(f"Template rendering failed: {str(e)}")
            return template  # Return original template as fallback
    
    async def _delivery_worker(self):
        """Background worker for processing notification delivery"""
        while True:
            try:
                notification = await self.delivery_queue.get()
                
                # Process delivery for each channel
                delivery_tasks = []
                for channel in notification.channels:
                    if self._check_rate_limit(channel):
                        task = asyncio.create_task(
                            self._deliver_to_channel(notification, channel)
                        )
                        delivery_tasks.append(task)
                    else:
                        logger.warning(f"Rate limit exceeded for channel: {channel.value}")
                
                # Wait for all deliveries to complete
                if delivery_tasks:
                    await asyncio.gather(*delivery_tasks, return_exceptions=True)
                
                self.delivery_queue.task_done()
                
            except Exception as e:
                logger.error(f"Delivery worker error: {str(e)}")
                await asyncio.sleep(1)  # Prevent tight loop on errors
    
    def _check_rate_limit(self, channel: DeliveryChannel) -> bool:
        """Check if channel is within rate limit"""
        channel_config = self.channel_configs.get(channel)
        if not channel_config or not channel_config.rate_limit:
            return True
        
        now = time.time()
        one_minute_ago = now - 60
        
        # Remove old entries
        while (self.rate_limits[channel] and 
               self.rate_limits[channel][0] < one_minute_ago):
            self.rate_limits[channel].popleft()
        
        # Check if within limit
        if len(self.rate_limits[channel]) < channel_config.rate_limit:
            self.rate_limits[channel].append(now)
            return True
        
        return False
    
    async def _deliver_to_channel(self, notification: Notification, channel: DeliveryChannel):
        """Deliver notification to specific channel"""
        handler = self.delivery_handlers.get(channel)
        if not handler:
            logger.error(f"No handler for channel: {channel.value}")
            return
        
        try:
            start_time = time.time()
            
            await handler(notification, self.channel_configs[channel].config)
            
            # Update notification status
            notification.status = NotificationStatus.SENT
            notification.sent_at = datetime.utcnow()
            notification.delivery_attempts += 1
            
            # Update performance metrics
            self.performance_metrics["notifications_sent"] += 1
            self.performance_metrics["delivery_attempts"] += 1
            
            delivery_time = time.time() - start_time
            
            # Update average delivery time
            total_sent = self.performance_metrics["notifications_sent"]
            current_avg = self.performance_metrics["average_delivery_time"]
            self.performance_metrics["average_delivery_time"] = (
                (current_avg * (total_sent - 1) + delivery_time) / total_sent
            )
            
            logger.info(f"Notification {notification.notification_id} delivered via {channel.value}")
            
        except Exception as e:
            logger.error(f"Delivery failed for {notification.notification_id} via {channel.value}: {str(e)}")
            notification.delivery_attempts += 1
            self.performance_metrics["notifications_failed"] += 1
            self.performance_metrics["delivery_attempts"] += 1
    
    async def _deliver_email(self, notification: Notification, config: Dict[str, Any]):
        """Deliver notification via email"""
        try:
            # Create message
            msg = MimeMultipart()
            msg['From'] = config['from_address']
            msg['To'] = notification.recipient
            msg['Subject'] = notification.subject
            
            # Create HTML and plain text versions
            html_body = markdown.markdown(notification.body)
            
            # Add HTML version
            html_part = MimeText(html_body, 'html')
            msg.attach(html_part)
            
            # Add plain text version
            text_part = MimeText(notification.body, 'plain')
            msg.attach(text_part)
            
            # Add any attachments from data
            if 'attachments' in notification.data:
                for attachment in notification.data['attachments']:
                    attachment_part = MimeApplication(
                        attachment['content'],
                        Name=attachment['filename']
                    )
                    attachment_part['Content-Disposition'] = f'attachment; filename="{attachment["filename"]}"'
                    msg.attach(attachment_part)
            
            # Send email
            with smtplib.SMTP(config['smtp_server'], config.get('smtp_port', 587)) as server:
                server.starttls()
                if 'username' in config and 'password' in config:
                    server.login(config['username'], config['password'])
                server.send_message(msg)
            
        except Exception as e:
            raise Exception(f"Email delivery failed: {str(e)}")
    
    async def _deliver_slack(self, notification: Notification, config: Dict[str, Any]):
        """Deliver notification via Slack"""
        try:
            async with aiohttp.ClientSession() as session:
                # Create Slack message payload
                payload = {
                    "channel": config.get('channel', '#general'),
                    "username": config.get('username', 'Shooting Star Notifications'),
                    "text": f"*{notification.subject}*\n{notification.body}",
                    "icon_emoji": config.get('icon_emoji', ':rocket:'),
                    "attachments": []
                }
                
                # Add rich content if available
                if notification.data.get('fields'):
                    attachment = {
                        "color": self._get_slack_color(notification.priority),
                        "fields": [
                            {"title": key, "value": str(value), "short": True}
                            for key, value in notification.data['fields'].items()
                        ]
                    }
                    payload["attachments"].append(attachment)
                
                # Send to Slack webhook
                async with session.post(config['webhook_url'], json=payload) as response:
                    if response.status != 200:
                        raise Exception(f"Slack API returned {response.status}")
        
        except Exception as e:
            raise Exception(f"Slack delivery failed: {str(e)}")
    
    async def _deliver_webhook(self, notification: Notification, config: Dict[str, Any]):
        """Deliver notification via webhook"""
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "notification_id": notification.notification_id,
                    "recipient": notification.recipient,
                    "subject": notification.subject,
                    "body": notification.body,
                    "type": notification.notification_type.value,
                    "priority": notification.priority.value,
                    "data": notification.data,
                    "sent_at": notification.sent_at.isoformat() if notification.sent_at else None
                }
                
                headers = config.get('headers', {})
                timeout = aiohttp.ClientTimeout(total=30)
                
                async with session.post(
                    config['webhook_url'], 
                    json=payload, 
                    headers=headers,
                    timeout=timeout
                ) as response:
                    if response.status not in [200, 201, 202]:
                        raise Exception(f"Webhook returned {response.status}")
        
        except Exception as e:
            raise Exception(f"Webhook delivery failed: {str(e)}")
    
    async def _deliver_in_app(self, notification: Notification, config: Dict[str, Any]):
        """Deliver in-app notification"""
        try:
            # This would integrate with your in-app notification system
            # For now, simulate delivery
            logger.info(f"In-app notification for {notification.recipient}: {notification.subject}")
            
            # Store in user's notification feed
            user_notifications = config.get('user_notifications', {})
            if notification.recipient not in user_notifications:
                user_notifications[notification.recipient] = []
            
            user_notifications[notification.recipient].append({
                "id": notification.notification_id,
                "subject": notification.subject,
                "body": notification.body,
                "type": notification.notification_type.value,
                "priority": notification.priority.value,
                "sent_at": datetime.utcnow().isoformat(),
                "read": False
            })
        
        except Exception as e:
            raise Exception(f"In-app delivery failed: {str(e)}")
    
    def _get_slack_color(self, priority: NotificationPriority) -> str:
        """Get Slack attachment color based on priority"""
        colors = {
            NotificationPriority.LOW: "#36a64f",      # Green
            NotificationPriority.MEDIUM: "#f2c744",   # Yellow
            NotificationPriority.HIGH: "#e67e22",     # Orange
            NotificationPriority.URGENT: "#e74c3c"    # Red
        }
        return colors.get(priority, "#95a5a6")
    
    async def get_user_notifications(self, user_id: str, 
                                   unread_only: bool = False,
                                   limit: int = 50) -> List[Notification]:
        """Get notifications for a specific user"""
        user_notifications = [
            n for n in self.notification_history 
            if n.recipient == user_id
        ]
        
        if unread_only:
            user_notifications = [
                n for n in user_notifications 
                if n.status != NotificationStatus.READ
            ]
        
        return sorted(user_notifications, key=lambda x: x.created_at, reverse=True)[:limit]
    
    async def mark_as_read(self, notification_id: str, user_id: str) -> Optional[Notification]:
        """Mark a notification as read"""
        if (notification_id in self.notifications and 
            self.notifications[notification_id].recipient == user_id):
            
            notification = self.notifications[notification_id]
            notification.status = NotificationStatus.READ
            notification.read_at = datetime.utcnow()
            
            logger.info(f"Marked notification as read: {notification_id}")
            return notification
        
        return None
    
    async def get_delivery_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """Get delivery statistics for specified time range"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        recent_notifications = [
            n for n in self.notification_history 
            if n.created_at >= cutoff_time
        ]
        
        if not recent_notifications:
            return {
                "timeframe_hours": hours,
                "total_notifications": 0,
                "message": "No notifications in timeframe"
            }
        
        # Calculate statistics
        status_counts = defaultdict(int)
        type_counts = defaultdict(int)
        channel_counts = defaultdict(int)
        delivery_times = []
        
        for notification in recent_notifications:
            status_counts[notification.status] += 1
            type_counts[notification.notification_type] += 1
            
            for channel in notification.channels:
                channel_counts[channel] += 1
            
            if notification.sent_at and notification.created_at:
                delivery_time = (notification.sent_at - notification.created_at).total_seconds()
                delivery_times.append(delivery_time)
        
        delivery_rate = status_counts[NotificationStatus.SENT] / len(recent_notifications)
        avg_delivery_time = statistics.mean(delivery_times) if delivery_times else 0
        
        return {
            "timeframe_hours": hours,
            "total_notifications": len(recent_notifications),
            "delivery_rate": round(delivery_rate, 3),
            "status_distribution": {k.value: v for k, v in status_counts.items()},
            "type_distribution": {k.value: v for k, v in type_counts.items()},
            "channel_distribution": {k.value: v for k, v in channel_counts.items()},
            "average_delivery_time": round(avg_delivery_time, 2),
            "statistics_generated": datetime.utcnow().isoformat()
        }
    
    async def create_campaign_notification(self, campaign_id: str, recipient: str,
                                         notification_type: str, data: Dict[str, Any]) -> Notification:
        """Create a campaign-specific notification"""
        template_name = f"campaign_{notification_type}"
        
        if template_name not in [t.name for t in self.templates.values()]:
            # Create campaign template if it doesn't exist
            await self.create_template({
                "name": template_name,
                "description": f"Template for campaign {notification_type} notifications",
                "notification_type": NotificationType.BUSINESS,
                "channels": ["email", "in_app"],
                "subject_template": "Campaign Update: {{ campaign_name }}",
                "body_template": self._get_campaign_template_body(notification_type),
                "variables": ["campaign_name", "campaign_id", "update_type", "details"]
            })
        
        # Find template
        template_id = next(
            t.template_id for t in self.templates.values() 
            if t.name == template_name
        )
        
        return await self.send_notification(
            template_id=template_id,
            recipient=recipient,
            data=data,
            priority=NotificationPriority.MEDIUM
        )
    
    def _get_campaign_template_body(self, notification_type: str) -> str:
        """Get template body for campaign notification type"""
        templates = {
            "launch": """
# Campaign Launched: {{ campaign_name }}

Your campaign **{{ campaign_name }}** has been successfully launched.

**Details:**
- Campaign ID: {{ campaign_id }}
- Launch Time: {{ launch_time }}
- Expected Duration: {{ duration }}

We'll keep you updated on the campaign performance.
            """,
            "performance": """
# Campaign Performance Update: {{ campaign_name }}

Your campaign **{{ campaign_name }}** has new performance data.

**Performance Metrics:**
- Engagement Rate: {{ engagement_rate }}%
- Conversion Rate: {{ conversion_rate }}%
- ROI: {{ roi }}%

**Recommendations:**
{{ recommendations }}

[View Detailed Report]({{ report_url }})
            """,
            "completion": """
# Campaign Completed: {{ campaign_name }}

Your campaign **{{ campaign_name }}** has been completed.

**Final Results:**
- Total Reach: {{ total_reach }}
- Total Conversions: {{ total_conversions }}
- Final ROI: {{ final_roi }}%
- Total Spend: ${{ total_spend }}

**Summary:**
{{ campaign_summary }}

[View Final Report]({{ report_url }})
            """
        }
        
        return templates.get(notification_type, """
# Campaign Notification: {{ campaign_name }}

{{ details }}

[View Campaign]({{ campaign_url }})
        """)
    
    def get_center_metrics(self) -> Dict[str, Any]:
        """Get notification center performance metrics"""
        active_templates = len([t for t in self.templates.values() if t.is_active])
        pending_notifications = len([
            n for n in self.notifications.values() 
            if n.status == NotificationStatus.PENDING
        ])
        
        return {
            **self.performance_metrics,
            "total_templates": len(self.templates),
            "active_templates": active_templates,
            "total_notifications": len(self.notifications),
            "pending_notifications": pending_notifications,
            "delivery_queue_size": self.delivery_queue.qsize(),
            "channel_handlers_registered": len(self.delivery_handlers),
            "channels_configured": len(self.channel_configs),
            "timestamp": datetime.utcnow().isoformat()
        }


# Global notification center instance
notification_center = NotificationCenterV16()


async def main():
    """Test harness for Notification Center"""
    print("📢 Notification Center V16 - Test Harness")
    
    # Configure email channel
    await notification_center.configure_channel(DeliveryChannel.EMAIL, {
        "smtp_server": "localhost",
        "smtp_port": 587,
        "from_address": "notifications@shootingstar.com",
        "enabled": True,
        "rate_limit": 60  # 60 emails per minute
    })
    
    # Create a notification template
    template = await notification_center.create_template({
        "name": "system_alert",
        "description": "System alert notification template",
        "notification_type": "alert",
        "channels": ["email", "slack"],
        "subject_template": "🚨 Alert: {{ alert_title }}",
        "body_template": """
# Alert Notification

**Title:** {{ alert_title }}
**Description:** {{ alert_description }}
**Severity:** {{ alert_severity }}
**Time:** {{ alert_time }}

**Details:**
{{ alert_details }}

Please take appropriate action.
        """,
        "variables": ["alert_title", "alert_description", "alert_severity", "alert_time", "alert_details"]
    })
    
    print(f"✅ Created template: {template.template_id}")
    
    # Send a test notification
    notification = await notification_center.send_notification(
        template_id=template.template_id,
        recipient="admin@shootingstar.com",
        data={
            "alert_title": "High CPU Usage",
            "alert_description": "CPU usage has exceeded 90%",
            "alert_severity": "high",
            "alert_time": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
            "alert_details": "CPU usage: 92.5%\nDuration: 5 minutes\nAffected services: AI processing"
        },
        priority=NotificationPriority.HIGH
    )
    
    print(f"📤 Sent notification: {notification.notification_id}")
    
    # Wait for delivery
    await asyncio.sleep(2)
    
    # Get delivery statistics
    stats = await notification_center.get_delivery_statistics(1)
    print(f"📊 Delivery Statistics: {stats['total_notifications']} notifications")
    
    # Get center metrics
    metrics = notification_center.get_center_metrics()
    print("🔧 Center Metrics:", metrics)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
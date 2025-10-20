"""
Background job scheduler using Celery for async task processing.
"""

from celery import Celery
from celery.schedules import crontab
import asyncio
from typing import Any, Dict, List, Optional
import logging

from config.settings import settings
from core.utils import logger

# Configure Celery
celery_app = Celery(
    "shooting_star",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND
)

# Celery configuration
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=30 * 60,  # 30 minutes
    worker_max_tasks_per_child=1000,
    worker_prefetch_multiplier=1,
)


# Scheduled tasks configuration
celery_app.conf.beat_schedule = {
    # Daily performance analytics at 2 AM
    "daily-performance-analytics": {
        "task": "daily_performance_analytics",
        "schedule": crontab(hour=2, minute=0),
    },
    # Campaign performance sync every 30 minutes
    "sync-campaign-performance": {
        "task": "sync_campaign_performance",
        "schedule": crontab(minute="*/30"),
    },
    # Weekly financial reports on Monday at 3 AM
    "weekly-financial-reports": {
        "task": "generate_weekly_financial_reports",
        "schedule": crontab(day_of_week=1, hour=3, minute=0),
    },
    # AI tip generation daily at 4 AM
    "generate-daily-tips": {
        "task": "generate_daily_tips",
        "schedule": crontab(hour=4, minute=0),
    },
    # System optimization check every hour
    "system-optimization-check": {
        "task": "system_optimization_check",
        "schedule": crontab(minute=0),
    },
    # Clean up old logs weekly on Sunday at 1 AM
    "cleanup-old-logs": {
        "task": "cleanup_old_logs",
        "schedule": crontab(day_of_week=0, hour=1, minute=0),
    },
}


@celery_app.task(bind=True)
def daily_performance_analytics(self) -> Dict[str, Any]:
    """
    Generate daily performance analytics for all brands.
    
    Returns:
        Task execution results
    """
    from services.analytics_service import AnalyticsService
    from database.connection import AsyncSessionLocal
    
    logger.info("Starting daily performance analytics task")
    
    async def run_analytics():
        async with AsyncSessionLocal() as db:
            analytics_service = AnalyticsService(db)
            results = await analytics_service.generate_daily_analytics()
            logger.info(f"Daily analytics completed for {len(results)} brands")
            return results
    
    # Run async function in sync context
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(run_analytics())


@celery_app.task(bind=True)
def sync_campaign_performance(self) -> Dict[str, Any]:
    """
    Sync campaign performance data from various platforms.
    
    Returns:
        Sync results
    """
    from services.tracking_service import TrackingService
    from database.connection import AsyncSessionLocal
    
    logger.info("Starting campaign performance sync task")
    
    async def run_sync():
        async with AsyncSessionLocal() as db:
            tracking_service = TrackingService(db)
            results = await tracking_service.sync_all_campaigns_performance()
            logger.info(f"Campaign sync completed: {results}")
            return results
    
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(run_sync())


@celery_app.task(bind=True)
def generate_weekly_financial_reports(self) -> Dict[str, Any]:
    """
    Generate weekly financial reports for all brands.
    
    Returns:
        Report generation results
    """
    from services.budgeting_service import BudgetingService
    from database.connection import AsyncSessionLocal
    
    logger.info("Starting weekly financial reports task")
    
    async def run_reports():
        async with AsyncSessionLocal() as db:
            budgeting_service = BudgetingService(db)
            results = await budgeting_service.generate_weekly_reports()
            logger.info(f"Weekly reports generated for {len(results)} brands")
            return results
    
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(run_reports())


@celery_app.task(bind=True)
def generate_daily_tips(self) -> Dict[str, Any]:
    """
    Generate daily AI tips for all active brands.
    
    Returns:
        Tip generation results
    """
    from ai.tip_generator import TipGenerator
    from database.connection import AsyncSessionLocal
    
    logger.info("Starting daily tip generation task")
    
    async def run_tip_generation():
        async with AsyncSessionLocal() as db:
            tip_generator = TipGenerator(db)
            results = await tip_generator.generate_daily_tips_for_all_brands()
            logger.info(f"Daily tips generated: {results}")
            return results
    
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(run_tip_generation())


@celery_app.task(bind=True)
def system_optimization_check(self) -> Dict[str, Any]:
    """
    Run system optimization checks and apply improvements.
    
    Returns:
        Optimization results
    """
    from ai.system_optimizer import SystemOptimizer
    from database.connection import AsyncSessionLocal
    
    logger.info("Starting system optimization check task")
    
    async def run_optimization():
        async with AsyncSessionLocal() as db:
            optimizer = SystemOptimizer(db)
            results = await optimizer.optimize_all_brands()
            logger.info(f"System optimization completed: {results}")
            return results
    
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(run_optimization())


@celery_app.task(bind=True)
def cleanup_old_logs(self) -> Dict[str, Any]:
    """
    Clean up old system logs to manage database size.
    
    Returns:
        Cleanup results
    """
    from database.connection import AsyncSessionLocal
    from sqlalchemy import delete
    from datetime import datetime, timedelta
    from database.models.system_logs import SystemLog
    
    logger.info("Starting old logs cleanup task")
    
    async def run_cleanup():
        async with AsyncSessionLocal() as db:
            # Delete logs older than 90 days
            cutoff_date = datetime.utcnow() - timedelta(days=90)
            
            result = await db.execute(
                delete(SystemLog).where(SystemLog.created_at < cutoff_date)
            )
            await db.commit()
            
            deleted_count = result.rowcount
            logger.info(f"Cleaned up {deleted_count} old log entries")
            
            return {"deleted_count": deleted_count}
    
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(run_cleanup())


@celery_app.task(bind=True)
def process_campaign_analytics(self, campaign_id: str) -> Dict[str, Any]:
    """
    Process analytics for a specific campaign.
    
    Args:
        campaign_id: Campaign ID to process
        
    Returns:
        Analytics results
    """
    from services.analytics_service import AnalyticsService
    from database.connection import AsyncSessionLocal
    
    logger.info(f"Processing analytics for campaign: {campaign_id}")
    
    async def run_campaign_analytics():
        async with AsyncSessionLocal() as db:
            analytics_service = AnalyticsService(db)
            results = await analytics_service.analyze_campaign_performance(campaign_id)
            return results
    
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(run_campaign_analytics())


@celery_app.task(bind=True)
def send_bulk_messages(self, message_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Send bulk messages to multiple users.
    
    Args:
        message_data: Message data including recipients and content
        
    Returns:
        Sending results
    """
    from services.messaging_service import MessagingService
    from database.connection import AsyncSessionLocal
    
    logger.info("Starting bulk message sending task")
    
    async def run_bulk_messages():
        async with AsyncSessionLocal() as db:
            messaging_service = MessagingService(db)
            results = await messaging_service.send_bulk_messages(message_data)
            logger.info(f"Bulk messages sent: {results}")
            return results
    
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(run_bulk_messages())


def schedule_campaign_analysis(campaign_id: str) -> str:
    """
    Schedule campaign analysis task.
    
    Args:
        campaign_id: Campaign ID to analyze
        
    Returns:
        Task ID
    """
    task = process_campaign_analytics.delay(campaign_id)
    return task.id


def schedule_bulk_messages(message_data: Dict[str, Any]) -> str:
    """
    Schedule bulk message sending.
    
    Args:
        message_data: Message data
        
    Returns:
        Task ID
    """
    task = send_bulk_messages.delay(message_data)
    return task.id


def get_task_status(task_id: str) -> Dict[str, Any]:
    """
    Get status of a Celery task.
    
    Args:
        task_id: Task ID
        
    Returns:
        Task status information
    """
    task_result = celery_app.AsyncResult(task_id)
    
    return {
        "task_id": task_id,
        "status": task_result.status,
        "result": task_result.result if task_result.ready() else None,
        "successful": task_result.successful(),
        "failed": task_result.failed()
    }
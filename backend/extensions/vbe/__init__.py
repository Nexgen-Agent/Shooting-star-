# extensions/vbe/__init__.py
"""Virtual Business Engine (VBE) - AI-powered business automation system."""
from .config_vbe import get_vbe_settings
from .cheese_method import build_outreach_message
from .lead_hunter import hunt_once, continuous_hunt
from .outreach_queue import enqueue_draft, list_pending, approve_draft, reject_draft
from .schedule_manager import generate_daily_plan, stack_task, get_pending_tasks

__version__ = "0.1.0"
__all__ = [
    "get_vbe_settings",
    "build_outreach_message", 
    "hunt_once",
    "continuous_hunt",
    "enqueue_draft",
    "list_pending",
    "approve_draft", 
    "reject_draft",
    "generate_daily_plan",
    "stack_task",
    "get_pending_tasks",
]
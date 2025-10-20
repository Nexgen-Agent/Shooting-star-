"""
Routers package for API endpoints.
"""

from .auth_router import router as auth_router
from .brand_router import router as brand_router
from .campaign_router import router as campaign_router
from .dashboard_router import router as dashboard_router
from .finance_router import router as finance_router
from .message_router import router as message_router
from .employee_router import router as employee_router

__all__ = [
    "auth_router",
    "brand_router", 
    "campaign_router",
    "dashboard_router",
    "finance_router",
    "message_router",
    "employee_router"
]
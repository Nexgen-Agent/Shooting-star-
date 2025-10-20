"""
Uvicorn entry point for production deployment.
"""

import uvicorn
from config.settings import settings

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level="info",
        access_log=True,
        workers=4 if not settings.DEBUG else 1
    )
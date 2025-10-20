"""
Utility functions and helpers for the Shooting Star backend.
"""

import json
import logging
import asyncio
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta
from fastapi import HTTPException, status
import uuid

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("shooting_star")


class ResponseFormatter:
    """Standardized response formatting for API endpoints."""
    
    @staticmethod
    def success(
        data: Any = None,
        message: str = "Success",
        meta: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Format successful response.
        
        Args:
            data: Response data
            message: Success message
            meta: Additional metadata
            
        Returns:
            Formatted response dictionary
        """
        response = {
            "success": True,
            "message": message,
            "data": data,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if meta:
            response["meta"] = meta
            
        return response
    
    @staticmethod
    def error(
        message: str = "Error",
        error_code: str = "INTERNAL_ERROR",
        details: Optional[Any] = None,
        status_code: int = status.HTTP_400_BAD_REQUEST
    ) -> HTTPException:
        """
        Format error response.
        
        Args:
            message: Error message
            error_code: Machine-readable error code
            details: Additional error details
            status_code: HTTP status code
            
        Returns:
            HTTPException with formatted error
        """
        error_data = {
            "success": False,
            "error": {
                "code": error_code,
                "message": message,
                "details": details,
                "timestamp": datetime.utcnow().isoformat()
            }
        }
        
        return HTTPException(
            status_code=status_code,
            detail=error_data
        )


class Paginator:
    """Utility for paginating database queries."""
    
    def __init__(self, page: int = 1, per_page: int = 20):
        """
        Initialize paginator.
        
        Args:
            page: Page number (1-indexed)
            per_page: Items per page
        """
        self.page = max(1, page)
        self.per_page = min(100, max(1, per_page))
        self.offset = (self.page - 1) * self.per_page
    
    def paginate_query(self, query):
        """
        Apply pagination to SQLAlchemy query.
        
        Args:
            query: SQLAlchemy query object
            
        Returns:
            Paginated query
        """
        return query.offset(self.offset).limit(self.per_page)
    
    def create_metadata(self, total_count: int) -> Dict[str, Any]:
        """
        Create pagination metadata.
        
        Args:
            total_count: Total number of items
            
        Returns:
            Pagination metadata
        """
        total_pages = (total_count + self.per_page - 1) // self.per_page
        
        return {
            "page": self.page,
            "per_page": self.per_page,
            "total_count": total_count,
            "total_pages": total_pages,
            "has_previous": self.page > 1,
            "has_next": self.page < total_pages
        }


class DateTimeHelper:
    """Date and time utility functions."""
    
    @staticmethod
    def now() -> datetime:
        """Get current UTC datetime."""
        return datetime.utcnow()
    
    @staticmethod
    def format_date(dt: datetime, format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
        """
        Format datetime to string.
        
        Args:
            dt: Datetime object
            format_str: Format string
            
        Returns:
            Formatted date string
        """
        return dt.strftime(format_str)
    
    @staticmethod
    def parse_date(date_str: str, format_str: str = "%Y-%m-%d") -> datetime:
        """
        Parse string to datetime.
        
        Args:
            date_str: Date string
            format_str: Format string
            
        Returns:
            Datetime object
        """
        return datetime.strptime(date_str, format_str)
    
    @staticmethod
    def get_date_range(days: int = 30) -> tuple:
        """
        Get date range for last N days.
        
        Args:
            days: Number of days
            
        Returns:
            Tuple of (start_date, end_date)
        """
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        return start_date, end_date


class DataValidator:
    """Data validation utilities."""
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """
        Validate email format.
        
        Args:
            email: Email address to validate
            
        Returns:
            True if valid
        """
        import re
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))
    
    @staticmethod
    def validate_phone(phone: str) -> bool:
        """
        Validate phone number format.
        
        Args:
            phone: Phone number to validate
            
        Returns:
            True if valid
        """
        import re
        # Basic phone validation - adjust as needed
        pattern = r'^[\+]?[0-9\s\-\(\)]{10,}$'
        return bool(re.match(pattern, phone))
    
    @staticmethod
    def sanitize_string(text: str, max_length: int = 255) -> str:
        """
        Sanitize string input.
        
        Args:
            text: Input text
            max_length: Maximum allowed length
            
        Returns:
            Sanitized string
        """
        if not text:
            return ""
        
        # Remove extra whitespace
        text = " ".join(text.split())
        
        # Truncate if too long
        if len(text) > max_length:
            text = text[:max_length]
            
        return text


class PerformanceTracker:
    """Performance tracking and monitoring."""
    
    def __init__(self):
        """Initialize performance tracker."""
        self.metrics = {}
    
    async def track_execution_time(self, operation_name: str):
        """
        Track execution time of an operation.
        
        Args:
            operation_name: Name of the operation
            
        Returns:
            Async context manager
        """
        return ExecutionTimeTracker(operation_name, self)


class ExecutionTimeTracker:
    """Context manager for tracking execution time."""
    
    def __init__(self, operation_name: str, tracker: PerformanceTracker):
        """
        Initialize tracker.
        
        Args:
            operation_name: Name of the operation
            tracker: Performance tracker instance
        """
        self.operation_name = operation_name
        self.tracker = tracker
        self.start_time = None
    
    async def __aenter__(self):
        """Enter context."""
        self.start_time = asyncio.get_event_loop().time()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit context and record timing."""
        end_time = asyncio.get_event_loop().time()
        execution_time = end_time - self.start_time
        
        # Store metric
        self.tracker.metrics[self.operation_name] = execution_time
        
        # Log if operation is slow
        if execution_time > 1.0:  # More than 1 second
            logger.warning(
                f"Slow operation detected: {self.operation_name} "
                f"took {execution_time:.2f} seconds"
            )


# Global instances
response_formatter = ResponseFormatter()
paginator = Paginator()
datetime_helper = DateTimeHelper()
data_validator = DataValidator()
performance_tracker = PerformanceTracker()


def generate_uuid() -> str:
    """
    Generate UUID string.
    
    Returns:
        UUID string
    """
    return str(uuid.uuid4())


def safe_json_loads(json_str: str, default: Any = None) -> Any:
    """
    Safely parse JSON string.
    
    Args:
        json_str: JSON string
        default: Default value if parsing fails
        
    Returns:
        Parsed JSON or default
    """
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return default


async def async_retry(
    func,
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """
    Retry async function with exponential backoff.
    
    Args:
        func: Async function to retry
        max_attempts: Maximum number of attempts
        delay: Initial delay between attempts
        backoff: Backoff multiplier
        exceptions: Exceptions to catch and retry on
        
    Returns:
        Function result
        
    Raises:
        Last exception if all attempts fail
    """
    last_exception = None
    
    for attempt in range(max_attempts):
        try:
            return await func()
        except exceptions as e:
            last_exception = e
            if attempt == max_attempts - 1:
                break
                
            wait_time = delay * (backoff ** attempt)
            logger.warning(
                f"Attempt {attempt + 1} failed for {func.__name__}. "
                f"Retrying in {wait_time:.2f} seconds. Error: {str(e)}"
            )
            await asyncio.sleep(wait_time)
    
    raise last_exception
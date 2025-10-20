# gateway/advanced_api_gateway.py
import redis
from fastapi import Request, HTTPException
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

class AdvancedAPIGateway:
    def __init__(self):
        self.limiter = Limiter(key_func=get_remote_address)
        self.rate_limits = {
            "free": "100/hour",
            "basic": "1000/hour", 
            "premium": "10000/hour",
            "enterprise": "100000/hour"
        }
        
    async def validate_api_key(self, request: Request):
        """Advanced API key validation with rate limiting"""
        api_key = request.headers.get("X-API-Key")
        if not api_key:
            raise HTTPException(status_code=401, detail="API key required")
            
        # Check rate limits
        plan = await self._get_plan_from_api_key(api_key)
        rate_limit = self.rate_limits.get(plan, "100/hour")
        
        return await self.limiter.check_rate_limit(request, rate_limit)
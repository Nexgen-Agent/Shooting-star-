from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import Column, String, Integer, DateTime, Text, JSON
from datetime import datetime
import logging
import json

from database.base import Base

class SystemLog(Base):
    __tablename__ = "system_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    level = Column(String(50), index=True)  # INFO, WARNING, ERROR, CRITICAL
    module = Column(String(100), index=True)
    function = Column(String(100))
    message = Column(Text)
    data = Column(JSON)  # Additional structured data
    user_id = Column(Integer, nullable=True)
    session_id = Column(String(100), nullable=True)
    
    def __repr__(self):
        return f"<SystemLog {self.level} {self.module}.{self.function}>"

class FinancialLogger:
    """Enhanced logging for financial operations"""
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self.logger = logging.getLogger(__name__)

    async def log_profit_allocation(self, allocation_data: dict, user_id: int = None):
        """Log profit allocation operations"""
        log_entry = SystemLog(
            level="INFO",
            module="profit_allocator",
            function="allocate_profits",
            message=f"Profit allocation completed for {allocation_data.get('period')}",
            data=allocation_data,
            user_id=user_id
        )
        self.db.add(log_entry)
        await self.db.commit()

    async def log_financial_forecast(self, forecast_data: dict, user_id: int = None):
        """Log financial forecasting operations"""
        log_entry = SystemLog(
            level="INFO",
            module="finance_forecaster",
            function="generate_projection",
            message=f"Financial projection generated: {forecast_data.get('time_horizon')}",
            data={
                "projection_id": forecast_data.get("projection_id"),
                "time_horizon": forecast_data.get("time_horizon"),
                "confidence": forecast_data.get("overall_confidence")
            },
            user_id=user_id
        )
        self.db.add(log_entry)
        await self.db.commit()

    async def log_growth_decision(self, decision_data: dict, user_id: int = None):
        """Log AI growth decisions and recommendations"""
        log_entry = SystemLog(
            level="INFO",
            module="growth_predictor",
            function="generate_growth_actions",
            message=f"Growth decisions generated: {len(decision_data.get('actions', []))} actions",
            data=decision_data,
            user_id=user_id
        )
        self.db.add(log_entry)
        await self.db.commit()

    async def log_risk_event(self, risk_data: dict, user_id: int = None):
        """Log risk events and mitigation actions"""
        log_entry = SystemLog(
            level="WARNING",
            module="risk_monitor",
            function="detect_risk",
            message=f"Risk detected: {risk_data.get('risk_type')}",
            data=risk_data,
            user_id=user_id
        )
        self.db.add(log_entry)
        await self.db.commit()

    async def get_financial_audit_trail(self, 
                                      start_date: datetime, 
                                      end_date: datetime,
                                      module: str = None) -> list:
        """Get financial audit trail for specified period"""
        # This would query the system logs for financial operations
        # Implementation would depend on specific query requirements
        return []
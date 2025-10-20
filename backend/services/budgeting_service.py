"""
Budgeting service for financial management and budget allocation.
"""

from typing import Dict, Any, List, Optional, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_
from datetime import datetime, timedelta
import logging

from database.models.brand import Brand
from database.models.campaign import Campaign
from database.models.transaction import Transaction
from database.models.performance import Performance
from config.constants import BudgetStatus, TransactionType
from core.utils import response_formatter, datetime_helper

logger = logging.getLogger(__name__)


class BudgetingService:
    """Budgeting service for financial operations."""
    
    def __init__(self, db: AsyncSession):
        """
        Initialize budgeting service.
        
        Args:
            db: Database session
        """
        self.db = db
    
    async def allocate_budget(
        self,
        brand_id: str,
        amount: float,
        description: str = "Budget allocation",
        allocated_by: str = "system"
    ) -> Tuple[bool, Optional[Transaction], str]:
        """
        Allocate budget to a brand.
        
        Args:
            brand_id: Brand ID
            amount: Allocation amount
            description: Allocation description
            allocated_by: User who allocated the budget
            
        Returns:
            Tuple of (success, transaction, message)
        """
        try:
            # Get brand
            result = await self.db.execute(
                select(Brand).where(Brand.id == brand_id)
            )
            brand = result.scalar_one_or_none()
            
            if not brand:
                return False, None, "Brand not found"
            
            # Create transaction
            transaction = Transaction(
                transaction_type=TransactionType.BUDGET_ALLOCATION,
                description=description,
                amount=amount,
                brand_id=brand_id,
                payment_status=BudgetStatus.APPROVED,
                metadata={
                    "allocated_by": allocated_by,
                    "allocation_date": datetime.utcnow().isoformat()
                }
            )
            
            # Update brand balance
            brand.monthly_budget = float(brand.monthly_budget or 0) + amount
            brand.current_balance = float(brand.current_balance or 0) + amount
            
            self.db.add(transaction)
            await self.db.commit()
            await self.db.refresh(transaction)
            
            logger.info(f"Budget allocated to brand {brand_id}: ${amount}")
            return True, transaction, "Budget allocated successfully"
            
        except Exception as e:
            await self.db.rollback()
            logger.error(f"Error allocating budget to brand {brand_id}: {str(e)}")
            return False, None, f"Error allocating budget: {str(e)}"
    
    async def create_campaign_budget(
        self,
        campaign_id: str,
        amount: float,
        description: str = "Campaign budget allocation"
    ) -> Tuple[bool, Optional[Transaction], str]:
        """
        Allocate budget to a campaign.
        
        Args:
            campaign_id: Campaign ID
            amount: Allocation amount
            description: Allocation description
            
        Returns:
            Tuple of (success, transaction, message)
        """
        try:
            # Get campaign and brand
            from database.models.campaign import Campaign
            
            result = await self.db.execute(
                select(Campaign).where(Campaign.id == campaign_id)
            )
            campaign = result.scalar_one_or_none()
            
            if not campaign:
                return False, None, "Campaign not found"
            
            brand_id = campaign.brand_id
            
            # Check if brand has sufficient balance
            result = await self.db.execute(
                select(Brand).where(Brand.id == brand_id)
            )
            brand = result.scalar_one_or_none()
            
            if not brand:
                return False, None, "Brand not found"
            
            if float(brand.current_balance or 0) < amount:
                return False, None, "Insufficient brand balance"
            
            # Create transaction
            transaction = Transaction(
                transaction_type=TransactionType.CAMPAIGN_BUDGET,
                description=description,
                amount=amount,
                brand_id=brand_id,
                campaign_id=campaign_id,
                payment_status=BudgetStatus.APPROVED,
                metadata={
                    "allocation_date": datetime.utcnow().isoformat()
                }
            )
            
            # Update balances
            brand.current_balance = float(brand.current_balance or 0) - amount
            campaign.budget_allocated = float(campaign.budget_allocated or 0) + amount
            
            self.db.add(transaction)
            await self.db.commit()
            await self.db.refresh(transaction)
            
            logger.info(f"Budget allocated to campaign {campaign_id}: ${amount}")
            return True, transaction, "Campaign budget allocated successfully"
            
        except Exception as e:
            await self.db.rollback()
            logger.error(f"Error allocating campaign budget {campaign_id}: {str(e)}")
            return False, None, f"Error allocating campaign budget: {str(e)}"
    
    async def record_expense(
        self,
        brand_id: str,
        amount: float,
        description: str,
        campaign_id: Optional[str] = None,
        influencer_id: Optional[str] = None,
        expense_type: str = "marketing"
    ) -> Tuple[bool, Optional[Transaction], str]:
        """
        Record an expense for a brand.
        
        Args:
            brand_id: Brand ID
            amount: Expense amount
            description: Expense description
            campaign_id: Associated campaign ID
            influencer_id: Associated influencer ID
            expense_type: Type of expense
            
        Returns:
            Tuple of (success, transaction, message)
        """
        try:
            # Get brand
            result = await self.db.execute(
                select(Brand).where(Brand.id == brand_id)
            )
            brand = result.scalar_one_or_none()
            
            if not brand:
                return False, None, "Brand not found"
            
            # Check if brand has sufficient balance
            if float(brand.current_balance or 0) < amount:
                return False, None, "Insufficient brand balance"
            
            # Create transaction
            transaction = Transaction(
                transaction_type=TransactionType.EXPENSE,
                description=description,
                amount=-amount,  # Negative for expenses
                brand_id=brand_id,
                campaign_id=campaign_id,
                influencer_id=influencer_id,
                payment_status=BudgetStatus.APPROVED,
                metadata={
                    "expense_type": expense_type,
                    "expense_date": datetime.utcnow().isoformat()
                }
            )
            
            # Update brand balance
            brand.current_balance = float(brand.current_balance or 0) - amount
            
            # Update campaign budget if applicable
            if campaign_id:
                from database.models.campaign import Campaign
                campaign_result = await self.db.execute(
                    select(Campaign).where(Campaign.id == campaign_id)
                )
                campaign = campaign_result.scalar_one_or_none()
                
                if campaign:
                    campaign.budget_used = float(campaign.budget_used or 0) + amount
            
            self.db.add(transaction)
            await self.db.commit()
            await self.db.refresh(transaction)
            
            logger.info(f"Expense recorded for brand {brand_id}: ${amount} - {description}")
            return True, transaction, "Expense recorded successfully"
            
        except Exception as e:
            await self.db.rollback()
            logger.error(f"Error recording expense for brand {brand_id}: {str(e)}")
            return False, None, f"Error recording expense: {str(e)}"
    
    async def get_brand_financial_summary(self, brand_id: str) -> Optional[Dict[str, Any]]:
        """
        Get financial summary for a brand.
        
        Args:
            brand_id: Brand ID
            
        Returns:
            Financial summary data
        """
        try:
            # Get brand
            result = await self.db.execute(
                select(Brand).where(Brand.id == brand_id)
            )
            brand = result.scalar_one_or_none()
            
            if not brand:
                return None
            
            # Get recent transactions (last 30 days)
            thirty_days_ago = datetime.utcnow() - timedelta(days=30)
            
            transactions_result = await self.db.execute(
                select(Transaction)
                .where(Transaction.brand_id == brand_id)
                .where(Transaction.transaction_date >= thirty_days_ago)
                .order_by(Transaction.transaction_date.desc())
            )
            recent_transactions = transactions_result.scalars().all()
            
            # Calculate metrics
            total_income = sum(
                float(t.amount) for t in recent_transactions 
                if float(t.amount) > 0
            )
            total_expenses = abs(sum(
                float(t.amount) for t in recent_transactions 
                if float(t.amount) < 0
            ))
            
            # Get campaign spending
            from database.models.campaign import Campaign
            campaigns_result = await self.db.execute(
                select(Campaign).where(Campaign.brand_id == brand_id)
            )
            campaigns = campaigns_result.scalars().all()
            
            total_campaign_budget = sum(float(c.budget_allocated or 0) for c in campaigns)
            total_campaign_spent = sum(float(c.budget_used or 0) for c in campaigns)
            
            summary = {
                "brand": brand.to_dict(),
                "current_balance": float(brand.current_balance or 0),
                "monthly_budget": float(brand.monthly_budget or 0),
                "total_income_30d": total_income,
                "total_expenses_30d": total_expenses,
                "net_cash_flow_30d": total_income - total_expenses,
                "total_campaign_budget": total_campaign_budget,
                "total_campaign_spent": total_campaign_spent,
                "campaign_budget_utilization": (
                    (total_campaign_spent / total_campaign_budget * 100) 
                    if total_campaign_budget > 0 else 0
                ),
                "recent_transactions": [t.to_dict() for t in recent_transactions[:10]]
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting financial summary for brand {brand_id}: {str(e)}")
            return None
    
    async def generate_weekly_reports(self) -> Dict[str, Any]:
        """
        Generate weekly financial reports for all brands.
        
        Returns:
            Report generation results
        """
        try:
            # Get all active brands
            result = await self.db.execute(
                select(Brand).where(Brand.is_active == True)
            )
            brands = result.scalars().all()
            
            reports = {}
            
            for brand in brands:
                brand_id = str(brand.id)
                
                # Get financial summary
                summary = await self.get_brand_financial_summary(brand_id)
                if summary:
                    reports[brand_id] = summary
            
            logger.info(f"Weekly financial reports generated for {len(reports)} brands")
            return {
                "success": True,
                "reports_generated": len(reports),
                "reports": reports
            }
            
        except Exception as e:
            logger.error(f"Error generating weekly reports: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "reports_generated": 0
            }
    
    async def get_transaction_history(
        self,
        brand_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        transaction_type: Optional[str] = None,
        page: int = 1,
        per_page: int = 20
    ) -> Tuple[List[Transaction], int]:
        """
        Get transaction history with filtering.
        
        Args:
            brand_id: Brand ID
            start_date: Start date filter
            end_date: End date filter
            transaction_type: Transaction type filter
            page: Page number
            per_page: Items per page
            
        Returns:
            Tuple of (transactions, total_count)
        """
        try:
            from sqlalchemy import func
            
            # Build query
            query = select(Transaction).where(Transaction.brand_id == brand_id)
            
            if start_date:
                query = query.where(Transaction.transaction_date >= start_date)
            
            if end_date:
                query = query.where(Transaction.transaction_date <= end_date)
            
            if transaction_type:
                query = query.where(Transaction.transaction_type == transaction_type)
            
            # Get total count
            count_query = select(func.count()).select_from(query.subquery())
            total_result = await self.db.execute(count_query)
            total_count = total_result.scalar_one()
            
            # Apply pagination
            from core.utils import Paginator
            paginator = Paginator(page, per_page)
            query = paginator.paginate_query(query)
            
            # Order by date
            query = query.order_by(Transaction.transaction_date.desc())
            
            # Execute query
            result = await self.db.execute(query)
            transactions = result.scalars().all()
            
            return transactions, total_count
            
        except Exception as e:
            logger.error(f"Error getting transaction history for brand {brand_id}: {str(e)}")
            return [], 0
    
    async def create_recurring_budget(
        self,
        brand_id: str,
        amount: float,
        frequency: str,  # monthly, quarterly, etc.
        description: str = "Recurring budget allocation"
    ) -> Tuple[bool, Optional[Transaction], str]:
        """
        Create a recurring budget allocation.
        
        Args:
            brand_id: Brand ID
            amount: Allocation amount
            frequency: Recurrence frequency
            description: Allocation description
            
        Returns:
            Tuple of (success, transaction, message)
        """
        try:
            # Get brand
            result = await self.db.execute(
                select(Brand).where(Brand.id == brand_id)
            )
            brand = result.scalar_one_or_none()
            
            if not brand:
                return False, None, "Brand not found"
            
            # Create recurring transaction
            transaction = Transaction(
                transaction_type=TransactionType.BUDGET_ALLOCATION,
                description=description,
                amount=amount,
                brand_id=brand_id,
                payment_status=BudgetStatus.APPROVED,
                is_recurring=True,
                recurring_frequency=frequency,
                metadata={
                    "recurring_setup_date": datetime.utcnow().isoformat(),
                    "next_occurrence": self._calculate_next_occurrence(frequency)
                }
            )
            
            # Update brand balance
            brand.monthly_budget = float(brand.monthly_budget or 0) + amount
            brand.current_balance = float(brand.current_balance or 0) + amount
            
            self.db.add(transaction)
            await self.db.commit()
            await self.db.refresh(transaction)
            
            logger.info(f"Recurring budget created for brand {brand_id}: ${amount} {frequency}")
            return True, transaction, "Recurring budget created successfully"
            
        except Exception as e:
            await self.db.rollback()
            logger.error(f"Error creating recurring budget for brand {brand_id}: {str(e)}")
            return False, None, f"Error creating recurring budget: {str(e)}"
    
    def _calculate_next_occurrence(self, frequency: str) -> datetime:
        """
        Calculate next occurrence for recurring transaction.
        
        Args:
            frequency: Recurrence frequency
            
        Returns:
            Next occurrence datetime
        """
        now = datetime.utcnow()
        
        if frequency == "monthly":
            return now + timedelta(days=30)
        elif frequency == "quarterly":
            return now + timedelta(days=90)
        elif frequency == "yearly":
            return now + timedelta(days=365)
        else:  # weekly
            return now + timedelta(days=7)
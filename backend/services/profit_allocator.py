from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import logging
import json

from database.models.finance.transaction import Transaction, ProfitAllocation
from database.models.finance.performance import FinancialPerformance
from schemas.finance import ProfitAllocationCreate

logger = logging.getLogger(__name__)

class ProfitAllocator:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.allocation_rules = {
            "growth_fund": 0.30,      # 30% for growth & expansion
            "operations": 0.60,       # 60% for operations & creator payments
            "vault_reserves": 0.10    # 10% for emergency reserves
        }

    async def calculate_monthly_profit(self, year: int, month: int) -> Dict[str, Any]:
        """Calculate total monthly profit after all deductions"""
        try:
            period = f"{year}-{month:02d}"
            start_date = datetime(year, month, 1)
            if month == 12:
                end_date = datetime(year + 1, 1, 1) - timedelta(days=1)
            else:
                end_date = datetime(year, month + 1, 1) - timedelta(days=1)
            
            # Calculate total revenue
            revenue_result = await self.db.execute(
                select(func.sum(Transaction.amount)).where(
                    and_(
                        Transaction.transaction_type == "revenue",
                        Transaction.transaction_date >= start_date,
                        Transaction.transaction_date <= end_date,
                        Transaction.status == "completed"
                    )
                )
            )
            total_revenue = revenue_result.scalar() or 0.0
            
            # Calculate total expenses (operational costs + influencer payouts)
            expenses_result = await self.db.execute(
                select(func.sum(Transaction.amount)).where(
                    and_(
                        Transaction.transaction_type == "expense",
                        Transaction.transaction_date >= start_date,
                        Transaction.transaction_date <= end_date,
                        Transaction.status == "completed"
                    )
                )
            )
            total_expenses = expenses_result.scalar() or 0.0
            
            # Calculate tax liabilities (simplified - 25% of profit)
            gross_profit = total_revenue - total_expenses
            tax_liability = gross_profit * 0.25  # Simplified tax calculation
            
            # Net profit after taxes
            net_profit = gross_profit - tax_liability
            
            # Breakdown of expenses (for reporting)
            operational_costs = await self._get_expense_by_category("operational", start_date, end_date)
            influencer_payouts = await self._get_expense_by_category("influencer_payout", start_date, end_date)
            other_expenses = total_expenses - operational_costs - influencer_payouts
            
            return {
                "period": period,
                "total_revenue": total_revenue,
                "total_expenses": total_expenses,
                "operational_costs": operational_costs,
                "influencer_payouts": influencer_payouts,
                "other_expenses": other_expenses,
                "tax_liability": tax_liability,
                "gross_profit": gross_profit,
                "net_profit": net_profit,
                "profit_margin": (net_profit / total_revenue * 100) if total_revenue > 0 else 0,
                "calculation_timestamp": datetime.utcnow()
            }
            
        except Exception as e:
            logger.error(f"Error calculating monthly profit for {year}-{month}: {str(e)}")
            raise

    async def allocate_profits(self, year: int, month: int) -> ProfitAllocation:
        """Allocate profits according to the predefined rules"""
        try:
            # Calculate monthly profit
            profit_data = await self.calculate_monthly_profit(year, month)
            net_profit = profit_data["net_profit"]
            
            if net_profit <= 0:
                raise ValueError(f"No profit to allocate for {year}-{month}. Net profit: ${net_profit}")
            
            # Create allocation records
            allocation = ProfitAllocation(
                period=profit_data["period"],
                total_profit=net_profit,
                growth_fund_amount=net_profit * self.allocation_rules["growth_fund"],
                growth_fund_percentage=self.allocation_rules["growth_fund"] * 100,
                operations_amount=net_profit * self.allocation_rules["operations"],
                operations_percentage=self.allocation_rules["operations"] * 100,
                vault_reserves_amount=net_profit * self.allocation_rules["vault_reserves"],
                vault_reserves_percentage=self.allocation_rules["vault_reserves"] * 100,
                calculation_metadata=profit_data,
                allocation_rules=self.allocation_rules
            )
            
            self.db.add(allocation)
            await self.db.commit()
            await self.db.refresh(allocation)
            
            # Create transaction records for each allocation
            await self._create_allocation_transactions(allocation)
            
            # Update financial performance record
            await self._update_financial_performance(profit_data)
            
            logger.info(f"Successfully allocated ${net_profit} for {profit_data['period']}")
            return allocation
            
        except Exception as e:
            await self.db.rollback()
            logger.error(f"Error allocating profits for {year}-{month}: {str(e)}")
            raise

    async def get_allocation_history(self, months: int = 12) -> List[ProfitAllocation]:
        """Get profit allocation history for specified number of months"""
        try:
            result = await self.db.execute(
                select(ProfitAllocation)
                .order_by(ProfitAllocation.period.desc())
                .limit(months)
            )
            return result.scalars().all()
        except Exception as e:
            logger.error(f"Error fetching allocation history: {str(e)}")
            return []

    async def adjust_allocation_rules(self, new_rules: Dict[str, float]) -> bool:
        """Dynamically adjust allocation rules based on AI recommendations"""
        try:
            # Validate rules sum to 1.0 (100%)
            if sum(new_rules.values()) != 1.0:
                raise ValueError("Allocation rules must sum to 1.0 (100%)")
            
            self.allocation_rules = new_rules
            logger.info(f"Updated allocation rules: {new_rules}")
            return True
            
        except Exception as e:
            logger.error(f"Error adjusting allocation rules: {str(e)}")
            return False

    async def _get_expense_by_category(self, category: str, start_date: datetime, end_date: datetime) -> float:
        """Get total expenses for a specific category"""
        result = await self.db.execute(
            select(func.sum(Transaction.amount)).where(
                and_(
                    Transaction.transaction_type == "expense",
                    Transaction.category == category,
                    Transaction.transaction_date >= start_date,
                    Transaction.transaction_date <= end_date,
                    Transaction.status == "completed"
                )
            )
        )
        return result.scalar() or 0.0

    async def _create_allocation_transactions(self, allocation: ProfitAllocation):
        """Create transaction records for profit allocations"""
        transactions = [
            {
                "transaction_type": "allocation",
                "category": "growth_fund",
                "description": f"Growth fund allocation for {allocation.period}",
                "amount": allocation.growth_fund_amount,
                "allocation_category": "growth_fund",
                "allocation_percentage": allocation.growth_fund_percentage,
                "transaction_date": datetime.utcnow()
            },
            {
                "transaction_type": "allocation",
                "category": "operations",
                "description": f"Operations allocation for {allocation.period}",
                "amount": allocation.operations_amount,
                "allocation_category": "operations",
                "allocation_percentage": allocation.operations_percentage,
                "transaction_date": datetime.utcnow()
            },
            {
                "transaction_type": "allocation",
                "category": "vault_reserves",
                "description": f"Vault reserves allocation for {allocation.period}",
                "amount": allocation.vault_reserves_amount,
                "allocation_category": "vault_reserves",
                "allocation_percentage": allocation.vault_reserves_percentage,
                "transaction_date": datetime.utcnow()
            }
        ]
        
        for tx_data in transactions:
            transaction = Transaction(**tx_data)
            self.db.add(transaction)
        
        await self.db.commit()

    async def _update_financial_performance(self, profit_data: Dict[str, Any]):
        """Update or create financial performance record"""
        try:
            # Check if record exists
            result = await self.db.execute(
                select(FinancialPerformance)
                .where(FinancialPerformance.period == profit_data["period"])
            )
            existing_record = result.scalar_one_or_none()
            
            if existing_record:
                # Update existing record
                existing_record.total_revenue = profit_data["total_revenue"]
                existing_record.operational_costs = profit_data["operational_costs"]
                existing_record.influencer_payouts = profit_data["influencer_payouts"]
                existing_record.total_costs = profit_data["total_expenses"]
                existing_record.net_profit = profit_data["net_profit"]
                existing_record.profit_margin = profit_data["profit_margin"]
            else:
                # Create new record
                performance = FinancialPerformance(
                    period=profit_data["period"],
                    total_revenue=profit_data["total_revenue"],
                    operational_costs=profit_data["operational_costs"],
                    influencer_payouts=profit_data["influencer_payouts"],
                    total_costs=profit_data["total_expenses"],
                    net_profit=profit_data["net_profit"],
                    profit_margin=profit_data["profit_margin"]
                )
                self.db.add(performance)
            
            await self.db.commit()
            
        except Exception as e:
            logger.error(f"Error updating financial performance: {str(e)}")
            # Don't raise here - this is secondary to the main allocation
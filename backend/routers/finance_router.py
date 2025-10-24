"""
Finance router for budget management and financial operations.
"""
from services.profit_allocator import ProfitAllocator
from services.finance_forecaster import FinanceForecaster
from services.growth_predictor import GrowthPredictor
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
import logging

from database.connection import get_db
from database.models.user import User
from database.models.transaction import Transaction
from core.security import require_roles, get_current_user
from core.utils import response_formatter, paginator
from config.constants import UserRole, BudgetStatus, TransactionType
from services.budgeting_service import BudgetingService

# Create router
router = APIRouter(prefix="/finance", tags=["finance"])

logger = logging.getLogger(__name__)


@router.post("/budgets/allocate", response_model=Dict[str, Any])
async def allocate_budget(
    allocation_data: dict,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_roles([UserRole.SUPER_ADMIN]))
):
    """
    Allocate budget to a brand (super admin only).
    
    Args:
        allocation_data: Budget allocation data
        db: Database session
        current_user: Current authenticated user
        
    Returns:
        Allocation result
    """
    try:
        budgeting_service = BudgetingService(db)
        
        brand_id = allocation_data.get("brand_id")
        amount = allocation_data.get("amount")
        description = allocation_data.get("description", "Budget allocation")
        
        if not brand_id or not amount:
            raise response_formatter.error(
                message="Brand ID and amount are required",
                error_code="MISSING_REQUIRED_FIELDS",
                status_code=status.HTTP_400_BAD_REQUEST
            )
        
        success, transaction, message = await budgeting_service.allocate_budget(
            brand_id=brand_id,
            amount=amount,
            description=description,
            allocated_by=current_user.email
        )
        
        if not success:
            raise response_formatter.error(
                message=message,
                error_code="BUDGET_ALLOCATION_FAILED",
                status_code=status.HTTP_400_BAD_REQUEST
            )
        
        logger.info(f"Budget allocated by {current_user.email}: ${amount} to brand {brand_id}")
        return response_formatter.success(
            data=transaction.to_dict(),
            message=message
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error allocating budget: {str(e)}")
        raise response_formatter.error(
            message="Error allocating budget",
            error_code="BUDGET_ALLOCATION_ERROR",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@router.post("/campaigns/{campaign_id}/budget", response_model=Dict[str, Any])
async def allocate_campaign_budget(
    campaign_id: str,
    budget_data: dict,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Allocate budget to a campaign.
    
    Args:
        campaign_id: Campaign ID
        budget_data: Budget allocation data
        db: Database session
        current_user: Current authenticated user
        
    Returns:
        Allocation result
    """
    try:
        from sqlalchemy import select
        from database.models.campaign import Campaign
        
        # Get campaign
        result = await db.execute(
            select(Campaign).where(Campaign.id == campaign_id)
        )
        campaign = result.scalar_one_or_none()
        
        if not campaign:
            raise response_formatter.error(
                message="Campaign not found",
                error_code="CAMPAIGN_NOT_FOUND",
                status_code=status.HTTP_404_NOT_FOUND
            )
        
        # Check permissions
        if (current_user.role in [UserRole.BRAND_OWNER, UserRole.EMPLOYEE] and 
            str(current_user.brand_id) != str(campaign.brand_id)):
            raise response_formatter.error(
                message="Access denied to allocate budget for this campaign",
                error_code="ACCESS_DENIED",
                status_code=status.HTTP_403_FORBIDDEN
            )
        
        budgeting_service = BudgetingService(db)
        
        amount = budget_data.get("amount")
        description = budget_data.get("description", "Campaign budget allocation")
        
        if not amount:
            raise response_formatter.error(
                message="Amount is required",
                error_code="MISSING_AMOUNT",
                status_code=status.HTTP_400_BAD_REQUEST
            )
        
        success, transaction, message = await budgeting_service.create_campaign_budget(
            campaign_id=campaign_id,
            amount=amount,
            description=description
        )
        
        if not success:
            raise response_formatter.error(
                message=message,
                error_code="CAMPAIGN_BUDGET_FAILED",
                status_code=status.HTTP_400_BAD_REQUEST
            )
        
        logger.info(f"Campaign budget allocated by {current_user.email}: ${amount} to campaign {campaign_id}")
        return response_formatter.success(
            data=transaction.to_dict(),
            message=message
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error allocating campaign budget: {str(e)}")
        raise response_formatter.error(
            message="Error allocating campaign budget",
            error_code="CAMPAIGN_BUDGET_ERROR",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@router.post("/expenses", response_model=Dict[str, Any])
async def record_expense(
    expense_data: dict,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Record an expense.
    
    Args:
        expense_data: Expense data
        db: Database session
        current_user: Current authenticated user
        
    Returns:
        Expense recording result
    """
    try:
        budgeting_service = BudgetingService(db)
        
        brand_id = expense_data.get("brand_id")
        amount = expense_data.get("amount")
        description = expense_data.get("description")
        
        if not brand_id or not amount or not description:
            raise response_formatter.error(
                message="Brand ID, amount, and description are required",
                error_code="MISSING_REQUIRED_FIELDS",
                status_code=status.HTTP_400_BAD_REQUEST
            )
        
        # Check permissions
        if (current_user.role in [UserRole.BRAND_OWNER, UserRole.EMPLOYEE] and 
            str(current_user.brand_id) != brand_id):
            raise response_formatter.error(
                message="Access denied to record expenses for this brand",
                error_code="ACCESS_DENIED",
                status_code=status.HTTP_403_FORBIDDEN
            )
        
        success, transaction, message = await budgeting_service.record_expense(
            brand_id=brand_id,
            amount=amount,
            description=description,
            campaign_id=expense_data.get("campaign_id"),
            influencer_id=expense_data.get("influencer_id"),
            expense_type=expense_data.get("expense_type", "marketing")
        )
        
        if not success:
            raise response_formatter.error(
                message=message,
                error_code="EXPENSE_RECORDING_FAILED",
                status_code=status.HTTP_400_BAD_REQUEST
            )
        
        logger.info(f"Expense recorded by {current_user.email}: ${amount} for brand {brand_id}")
        return response_formatter.success(
            data=transaction.to_dict(),
            message=message
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error recording expense: {str(e)}")
        raise response_formatter.error(
            message="Error recording expense",
            error_code="EXPENSE_RECORDING_ERROR",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@router.get("/transactions", response_model=Dict[str, Any])
async def get_transactions(
    brand_id: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    transaction_type: Optional[str] = None,
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get transaction history with filtering.
    
    Args:
        brand_id: Filter by brand ID
        start_date: Start date filter (YYYY-MM-DD)
        end_date: End date filter (YYYY-MM-DD)
        transaction_type: Transaction type filter
        page: Page number
        per_page: Items per page
        db: Database session
        current_user: Current authenticated user
        
    Returns:
        Paginated transaction history
    """
    try:
        from datetime import datetime
        
        # Determine brand context
        if current_user.role == UserRole.SUPER_ADMIN:
            target_brand_id = brand_id
        else:
            target_brand_id = str(current_user.brand_id) if current_user.brand_id else None
        
        # Check permissions
        if (current_user.role in [UserRole.BRAND_OWNER, UserRole.EMPLOYEE] and 
            brand_id and str(current_user.brand_id) != brand_id):
            raise response_formatter.error(
                message="Access denied to other brands' transactions",
                error_code="ACCESS_DENIED",
                status_code=status.HTTP_403_FORBIDDEN
            )
        
        if not target_brand_id:
            raise response_formatter.error(
                message="No brand context available",
                error_code="NO_BRAND_CONTEXT",
                status_code=status.HTTP_400_BAD_REQUEST
            )
        
        # Parse dates
        start_date_obj = None
        end_date_obj = None
        
        if start_date:
            try:
                start_date_obj = datetime.strptime(start_date, "%Y-%m-%d")
            except ValueError:
                raise response_formatter.error(
                    message="Invalid start date format. Use YYYY-MM-DD",
                    error_code="INVALID_DATE_FORMAT",
                    status_code=status.HTTP_400_BAD_REQUEST
                )
        
        if end_date:
            try:
                end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")
            except ValueError:
                raise response_formatter.error(
                    message="Invalid end date format. Use YYYY-MM-DD",
                    error_code="INVALID_DATE_FORMAT",
                    status_code=status.HTTP_400_BAD_REQUEST
                )
        
        budgeting_service = BudgetingService(db)
        transactions, total_count = await budgeting_service.get_transaction_history(
            brand_id=target_brand_id,
            start_date=start_date_obj,
            end_date=end_date_obj,
            transaction_type=transaction_type,
            page=page,
            per_page=per_page
        )
        
        # Create metadata
        meta = paginator.create_metadata(total_count, page, per_page)
        
        return response_formatter.success(
            data=[transaction.to_dict() for transaction in transactions],
            meta=meta,
            message="Transactions retrieved successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting transactions: {str(e)}")
        raise response_formatter.error(
            message="Error retrieving transactions",
            error_code="TRANSACTIONS_RETRIEVAL_FAILED",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@router.get("/brands/{brand_id}/summary", response_model=Dict[str, Any])
async def get_brand_financial_summary(
    brand_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get financial summary for a brand.
    
    Args:
        brand_id: Brand ID
        db: Database session
        current_user: Current authenticated user
        
    Returns:
        Financial summary data
    """
    try:
        # Check permissions
        if (current_user.role in [UserRole.BRAND_OWNER, UserRole.EMPLOYEE] and 
            str(current_user.brand_id) != brand_id):
            raise response_formatter.error(
                message="Access denied to this brand's financial data",
                error_code="ACCESS_DENIED",
                status_code=status.HTTP_403_FORBIDDEN
            )
        
        budgeting_service = BudgetingService(db)
        financial_summary = await budgeting_service.get_brand_financial_summary(brand_id)
        
        if not financial_summary:
            raise response_formatter.error(
                message="Financial data not available for this brand",
                error_code="FINANCIAL_DATA_UNAVAILABLE",
                status_code=status.HTTP_404_NOT_FOUND
            )
        
        return response_formatter.success(
            data=financial_summary,
            message="Financial summary retrieved successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting financial summary for brand {brand_id}: {str(e)}")
        raise response_formatter.error(
            message="Error retrieving financial summary",
            error_code="FINANCIAL_SUMMARY_FAILED",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@router.post("/recurring-budgets", response_model=Dict[str, Any])
async def create_recurring_budget(
    recurring_data: dict,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_roles([UserRole.SUPER_ADMIN]))
):
    """
    Create a recurring budget allocation (super admin only).
    
    Args:
        recurring_data: Recurring budget data
        db: Database session
        current_user: Current authenticated user
        
    Returns:
        Recurring budget creation result
    """
    try:
        budgeting_service = BudgetingService(db)
        
        brand_id = recurring_data.get("brand_id")
        amount = recurring_data.get("amount")
        frequency = recurring_data.get("frequency")
        description = recurring_data.get("description", "Recurring budget allocation")
        
        if not brand_id or not amount or not frequency:
            raise response_formatter.error(
                message="Brand ID, amount, and frequency are required",
                error_code="MISSING_REQUIRED_FIELDS",
                status_code=status.HTTP_400_BAD_REQUEST
            )
        
        valid_frequencies = ["weekly", "monthly", "quarterly", "yearly"]
        if frequency not in valid_frequencies:
            raise response_formatter.error(
                message=f"Frequency must be one of: {', '.join(valid_frequencies)}",
                error_code="INVALID_FREQUENCY",
                status_code=status.HTTP_400_BAD_REQUEST
            )
        
        success, transaction, message = await budgeting_service.create_recurring_budget(
            brand_id=brand_id,
            amount=amount,
            frequency=frequency,
            description=description
        )
        
        if not success:
            raise response_formatter.error(
                message=message,
                error_code="RECURRING_BUDGET_FAILED",
                status_code=status.HTTP_400_BAD_REQUEST
            )
        
        logger.info(f"Recurring budget created by {current_user.email}: ${amount} {frequency} for brand {brand_id}")
        return response_formatter.success(
            data=transaction.to_dict(),
            message=message
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating recurring budget: {str(e)}")
        raise response_formatter.error(
            message="Error creating recurring budget",
            error_code="RECURRING_BUDGET_ERROR",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@router.get("/reports/monthly", response_model=Dict[str, Any])
async def generate_monthly_report(
    brand_id: Optional[str] = None,
    year: int = Query(None, ge=2020, le=2030),
    month: int = Query(None, ge=1, le=12),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Generate monthly financial report.
    
    Args:
        brand_id: Brand ID (super admin only)
        year: Year for report
        month: Month for report
        db: Database session
        current_user: Current authenticated user
        
    Returns:
        Monthly financial report
    """
    try:
        from datetime import datetime, timedelta
        from sqlalchemy import select, func
        from database.models.transaction import Transaction
        
        # Determine brand context
        if current_user.role == UserRole.SUPER_ADMIN:
            target_brand_id = brand_id
        else:
            target_brand_id = str(current_user.brand_id) if current_user.brand_id else None
        
        # Check permissions
        if (current_user.role in [UserRole.BRAND_OWNER, UserRole.EMPLOYEE] and 
            brand_id and str(current_user.brand_id) != brand_id):
            raise response_formatter.error(
                message="Access denied to other brands' reports",
                error_code="ACCESS_DENIED",
                status_code=status.HTTP_403_FORBIDDEN
            )
        
        if not target_brand_id:
            raise response_formatter.error(
                message="No brand context available",
                error_code="NO_BRAND_CONTEXT",
                status_code=status.HTTP_400_BAD_REQUEST
            )
        
        # Use current month/year if not specified
        if not year or not month:
            now = datetime.utcnow()
            year = now.year
            month = now.month
        
        # Calculate date range
        start_date = datetime(year, month, 1)
        if month == 12:
            end_date = datetime(year + 1, 1, 1) - timedelta(days=1)
        else:
            end_date = datetime(year, month + 1, 1) - timedelta(days=1)
        
        # Get transactions for the month
        result = await db.execute(
            select(Transaction)
            .where(Transaction.brand_id == target_brand_id)
            .where(Transaction.transaction_date >= start_date)
            .where(Transaction.transaction_date <= end_date)
            .order_by(Transaction.transaction_date.desc())
        )
        transactions = result.scalars().all()
        
        # Calculate summary
        income = sum(float(t.amount) for t in transactions if float(t.amount) > 0)
        expenses = abs(sum(float(t.amount) for t in transactions if float(t.amount) < 0))
        net_flow = income - expenses
        
        # Get budget utilization
        from database.models.brand import Brand
        brand_result = await db.execute(
            select(Brand).where(Brand.id == target_brand_id)
        )
        brand = brand_result.scalar_one_or_none()
        
        monthly_budget = float(brand.monthly_budget or 0) if brand else 0
        budget_utilization = (expenses / monthly_budget * 100) if monthly_budget > 0 else 0
        
        report_data = {
            "period": {
                "year": year,
                "month": month,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat()
            },
            "summary": {
                "total_income": income,
                "total_expenses": expenses,
                "net_cash_flow": net_flow,
                "monthly_budget": monthly_budget,
                "budget_utilization": budget_utilization
            },
            "transactions_by_type": {},
            "transactions": [t.to_dict() for t in transactions]
        }
        
        # Group transactions by type
        for transaction in transactions:
            t_type = transaction.transaction_type
            if t_type not in report_data["transactions_by_type"]:
                report_data["transactions_by_type"][t_type] = []
            report_data["transactions_by_type"][t_type].append(transaction.to_dict())
        
        return response_formatter.success(
            data=report_data,
            message="Monthly report generated successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating monthly report: {str(e)}")
        raise response_formatter.error(
            message="Error generating monthly report",
            error_code="MONTHLY_REPORT_FAILED",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

# =============================================================================
# PROJECT SKYROCKET - AUTONOMOUS GROWTH ENGINE ENDPOINTS
# =============================================================================

@router.post("/profit-allocation/calculate", response_model=Dict[str, Any])
async def calculate_profit_allocation(
    year: int = Query(..., description="Year for allocation"),
    month: int = Query(..., description="Month for allocation"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_roles([UserRole.SUPER_ADMIN]))
):
    """Calculate and allocate profits according to Project Skyrocket rules"""
    try:
        allocator = ProfitAllocator(db)
        allocation = await allocator.allocate_profits(year, month)
        
        logger.info(f"Profit allocation completed by {current_user.email} for {year}-{month}")
        return response_formatter.success(
            data=allocation.to_dict() if hasattr(allocation, 'to_dict') else allocation,
            message="Profit allocation completed successfully"
        )
    except ValueError as e:
        raise response_formatter.error(
            message=str(e),
            error_code="PROFIT_ALLOCATION_FAILED",
            status_code=status.HTTP_400_BAD_REQUEST
        )
    except Exception as e:
        logger.error(f"Error in profit allocation: {str(e)}")
        raise response_formatter.error(
            message="Error calculating profit allocation",
            error_code="PROFIT_ALLOCATION_ERROR",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@router.get("/profit-allocation/history", response_model=Dict[str, Any])
async def get_profit_allocation_history(
    months: int = Query(12, description="Number of months to retrieve"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_roles([UserRole.SUPER_ADMIN]))
):
    """Get profit allocation history"""
    try:
        allocator = ProfitAllocator(db)
        allocations = await allocator.get_allocation_history(months)
        
        return response_formatter.success(
            data=[alloc.to_dict() if hasattr(alloc, 'to_dict') else alloc for alloc in allocations],
            message="Profit allocation history retrieved successfully"
        )
    except Exception as e:
        logger.error(f"Error fetching allocation history: {str(e)}")
        raise response_formatter.error(
            message="Error retrieving allocation history",
            error_code="ALLOCATION_HISTORY_ERROR",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@router.get("/projection", response_model=Dict[str, Any])
async def get_financial_projection(
    projection_type: str = Query("five_year", description="Type of projection (five_year, monthly)"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get AI-powered financial projections"""
    try:
        forecaster = FinanceForecaster(db)
        
        if projection_type == "five_year":
            projection = await forecaster.generate_five_year_projection()
        elif projection_type == "monthly":
            projection = await forecaster.generate_monthly_projection(12)
        else:
            raise response_formatter.error(
                message="Invalid projection type. Use 'five_year' or 'monthly'",
                error_code="INVALID_PROJECTION_TYPE",
                status_code=status.HTTP_400_BAD_REQUEST
            )
        
        return response_formatter.success(
            data=projection,
            message="Financial projection generated successfully"
        )
    except Exception as e:
        logger.error(f"Error generating financial projection: {str(e)}")
        raise response_formatter.error(
            message="Error generating financial projection",
            error_code="PROJECTION_ERROR",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@router.get("/income-streams", response_model=Dict[str, Any])
async def get_income_streams_analysis(
    limit: int = Query(10, description="Number of streams to return"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get AI-analyzed income streams ranked by profit velocity"""
    try:
        predictor = GrowthPredictor(db)
        streams = await predictor.identify_fastest_income_streams(limit)
        
        return response_formatter.success(
            data={"income_streams": streams},
            message="Income streams analysis completed successfully"
        )
    except Exception as e:
        logger.error(f"Error analyzing income streams: {str(e)}")
        raise response_formatter.error(
            message="Error analyzing income streams",
            error_code="INCOME_STREAMS_ERROR",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@router.get("/growth-actions", response_model=Dict[str, Any])
async def get_growth_actions_recommendations(
    confidence_threshold: float = Query(0.7, description="Minimum confidence score"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get AI-recommended growth actions with ROI confidence scores"""
    try:
        predictor = GrowthPredictor(db)
        actions = await predictor.generate_growth_actions(confidence_threshold)
        
        return response_formatter.success(
            data={"growth_actions": actions},
            message="Growth actions recommendations generated successfully"
        )
    except Exception as e:
        logger.error(f"Error generating growth actions: {str(e)}")
        raise response_formatter.error(
            message="Error generating growth actions",
            error_code="GROWTH_ACTIONS_ERROR",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@router.post("/optimize-allocation", response_model=Dict[str, Any])
async def optimize_resource_allocation(
    available_budget: float = Query(..., description="Available budget for allocation"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Optimize resource allocation across income streams for maximum ROI"""
    try:
        predictor = GrowthPredictor(db)
        optimization = await predictor.optimize_resource_allocation(available_budget)
        
        return response_formatter.success(
            data=optimization,
            message="Resource allocation optimized successfully"
        )
    except Exception as e:
        logger.error(f"Error optimizing resource allocation: {str(e)}")
        raise response_formatter.error(
            message="Error optimizing resource allocation",
            error_code="ALLOCATION_OPTIMIZATION_ERROR",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@router.get("/dashboard-summary", response_model=Dict[str, Any])
async def get_financial_dashboard_summary(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get comprehensive financial dashboard summary for Project Skyrocket"""
    try:
        allocator = ProfitAllocator(db)
        forecaster = FinanceForecaster(db)
        predictor = GrowthPredictor(db)
        
        # Get current period
        from datetime import datetime
        current_date = datetime.utcnow()
        current_period = current_date.strftime("%Y-%m")
        
        # Get latest allocation
        allocations = await allocator.get_allocation_history(1)
        latest_allocation = allocations[0] if allocations else None
        
        # Get 5-year projection summary
        projection_summary = await forecaster.generate_five_year_projection()
        
        # Get top growth actions
        growth_actions = await predictor.generate_growth_actions(0.7)
        
        # Get top income streams
        income_streams = await predictor.identify_fastest_income_streams(5)
        
        summary_data = {
            "current_period": current_period,
            "latest_allocation": latest_allocation.to_dict() if latest_allocation and hasattr(latest_allocation, 'to_dict') else latest_allocation,
            "projection_summary": {
                "five_year_revenue": projection_summary.get("key_metrics", {}).get("estimated_5yr_revenue", 0),
                "five_year_profit": projection_summary.get("key_metrics", {}).get("estimated_5yr_profit", 0),
                "confidence": projection_summary.get("overall_confidence", 0)
            },
            "top_growth_actions": growth_actions[:3],
            "top_income_streams": income_streams,
            "last_updated": current_date.isoformat()
        }
        
        return response_formatter.success(
            data=summary_data,
            message="Financial dashboard summary retrieved successfully"
        )
        
    except Exception as e:
        logger.error(f"Error generating dashboard summary: {str(e)}")
        raise response_formatter.error(
            message="Error generating dashboard summary",
            error_code="DASHBOARD_SUMMARY_ERROR",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
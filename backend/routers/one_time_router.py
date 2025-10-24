from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional
import logging

from database.session import get_db
from services.one_time_service import OneTimeService
from services.retargeting_service import RetargetingService
from schemas.one_time import PurchaseCreate, PurchaseResponse, ProductTemplateCreate, ProductTemplateResponse

router = APIRouter(prefix="/api/one-time", tags=["one_time_purchases"])
logger = logging.getLogger(__name__)

@router.post("/purchases", response_model=PurchaseResponse, status_code=status.HTTP_201_CREATED)
async def create_purchase(
    purchase_data: PurchaseCreate,
    db: AsyncSession = Depends(get_db)
):
    """Create a new one-time purchase"""
    try:
        service = OneTimeService(db)
        purchase = await service.create_purchase(purchase_data)
        return purchase
    except Exception as e:
        logger.error(f"Error creating purchase: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create purchase"
        )

@router.get("/purchases/{purchase_id}", response_model=PurchaseResponse)
async def get_purchase(
    purchase_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Get specific purchase by ID"""
    try:
        service = OneTimeService(db)
        purchase = await service.get_purchase(purchase_id)
        
        if not purchase:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Purchase not found"
            )
            
        return purchase
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching purchase {purchase_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch purchase"
        )

@router.patch("/purchases/{purchase_id}/payment-status")
async def update_payment_status(
    purchase_id: int,
    status: str = Query(..., regex="^(paid|failed|refunded)$"),
    db: AsyncSession = Depends(get_db)
):
    """Update payment status for a purchase"""
    try:
        service = OneTimeService(db)
        success = await service.update_payment_status(purchase_id, status)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to update payment status"
            )
            
        return {"message": f"Payment status updated to {status}"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating payment status for purchase {purchase_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update payment status"
        )

@router.get("/purchases/{purchase_id}/progress")
async def get_delivery_progress(
    purchase_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Track delivery progress for a purchase"""
    try:
        service = OneTimeService(db)
        progress_data = await service.track_delivery_progress(purchase_id)
        return progress_data
    except Exception as e:
        logger.error(f"Error tracking progress for purchase {purchase_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to track delivery progress"
        )

@router.post("/templates", response_model=ProductTemplateResponse)
async def create_product_template(
    template_data: ProductTemplateCreate,
    db: AsyncSession = Depends(get_db)
):
    """Create a new product template"""
    try:
        service = OneTimeService(db)
        template = await service.create_product_template(template_data)
        return template
    except Exception as e:
        logger.error(f"Error creating product template: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create product template"
        )

@router.get("/templates", response_model=List[ProductTemplateResponse])
async def get_active_templates(
    db: AsyncSession = Depends(get_db)
):
    """Get all active product templates"""
    try:
        service = OneTimeService(db)
        templates = await service.get_active_templates()
        return templates
    except Exception as e:
        logger.error(f"Error fetching product templates: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch product templates"
        )

@router.get("/retargeting/candidates")
async def get_conversion_candidates(
    days: int = Query(90, description="Lookback period in days"),
    db: AsyncSession = Depends(get_db)
):
    """Get one-time buyers with high conversion potential"""
    try:
        service = RetargetingService(db)
        candidates = await service.find_conversion_candidates(days)
        return {"candidates": candidates}
    except Exception as e:
        logger.error(f"Error fetching conversion candidates: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch conversion candidates"
        )

@router.get("/retargeting/metrics")
async def get_conversion_metrics(
    db: AsyncSession = Depends(get_db)
):
    """Get conversion metrics from one-time to managed clients"""
    try:
        service = RetargetingService(db)
        metrics = await service.track_conversion_metrics()
        return metrics
    except Exception as e:
        logger.error(f"Error fetching conversion metrics: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch conversion metrics"
        )

@router.post("/retargeting/campaigns")
async def generate_retargeting_campaigns(
    candidate_ids: List[int],
    db: AsyncSession = Depends(get_db)
):
    """Generate retargeting campaigns for candidates"""
    try:
        service = RetargetingService(db)
        campaigns = await service.generate_retargeting_campaigns(candidate_ids)
        return {"campaigns": campaigns}
    except Exception as e:
        logger.error(f"Error generating retargeting campaigns: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate retargeting campaigns"
        )
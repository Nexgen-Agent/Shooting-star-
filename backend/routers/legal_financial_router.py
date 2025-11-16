# routers/legal_financial_router.py
"""
Legal-Financial Intelligence Router
API endpoints for the Hybrid Legal-Financial Counsel
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, List, Any
from pydantic import BaseModel
from datetime import datetime

from ai.hybrid_legal_financial_counsel import (
    HybridLegalFinancialCounsel, 
    LegalComplianceCheck, 
    FinancialDeal, 
    AssetValuation,
    LegalJurisdiction,
    AssetType,
    RiskLevel
)

router = APIRouter()

# Global instance
legal_financial_counsel = None

class ComplianceAnalysisRequest(BaseModel):
    operation_type: str
    jurisdictions: List[LegalJurisdiction]

class DealStructuringRequest(BaseModel):
    partner_profile: Dict[str, Any]
    deal_terms: Dict[str, Any]

class AssetEvaluationRequest(BaseModel):
    asset_type: AssetType
    quantity: float = 1.0

class PartnerRiskAssessmentRequest(BaseModel):
    partner_data: Dict[str, Any]

@router.on_event("startup")
async def startup_legal_financial_counsel():
    """Initialize the legal-financial counsel on startup"""
    global legal_financial_counsel
    legal_financial_counsel = HybridLegalFinancialCounsel()

@router.get("/legal-financial/status")
async def get_legal_financial_status():
    """Get overall status of the legal-financial counsel"""
    if not legal_financial_counsel:
        raise HTTPException(status_code=503, detail="Legal-Financial Counsel not initialized")
    
    return {
        "status": "active",
        "version": "1.0.0",
        "mission": "AI Legal Counsel + Financial Strategist + Asset Intelligence",
        "alerts_active": len(legal_financial_counsel.compliance_alerts),
        "weekly_reports_generated": len(legal_financial_counsel.weekly_reports),
        "assets_tracked": len(legal_financial_counsel.asset_ledger)
    }

@router.post("/legal-financial/analyze-compliance")
async def analyze_compliance(request: ComplianceAnalysisRequest) -> LegalComplianceCheck:
    """Analyze legal compliance for business operations"""
    if not legal_financial_counsel:
        raise HTTPException(status_code=503, detail="Legal-Financial Counsel not initialized")
    
    try:
        compliance_check = await legal_financial_counsel.analyze_legal_compliance(
            request.operation_type, 
            request.jurisdictions
        )
        return compliance_check
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Compliance analysis failed: {str(e)}")

@router.post("/legal-financial/structure-deal")
async def structure_financial_deal(request: DealStructuringRequest) -> FinancialDeal:
    """AI-optimized financial deal structuring"""
    if not legal_financial_counsel:
        raise HTTPException(status_code=503, detail="Legal-Financial Counsel not initialized")
    
    try:
        deal = await legal_financial_counsel.structure_financial_deal(
            request.partner_profile,
            request.deal_terms
        )
        return deal
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Deal structuring failed: {str(e)}")

@router.post("/legal-financial/evaluate-asset")
async def evaluate_asset(request: AssetEvaluationRequest) -> AssetValuation:
    """Comprehensive asset valuation and strategy analysis"""
    if not legal_financial_counsel:
        raise HTTPException(status_code=503, detail="Legal-Financial Counsel not initialized")
    
    try:
        valuation = await legal_financial_counsel.evaluate_asset_strategy(
            request.asset_type,
            request.quantity
        )
        return valuation
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Asset evaluation failed: {str(e)}")

@router.get("/legal-financial/weekly-report")
async def generate_weekly_report():
    """Generate autonomous weekly intelligence report"""
    if not legal_financial_counsel:
        raise HTTPException(status_code=503, detail="Legal-Financial Counsel not initialized")
    
    try:
        report = await legal_financial_counsel.generate_weekly_intelligence_report()
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Weekly report generation failed: {str(e)}")

@router.get("/legal-financial/alerts")
async def get_compliance_alerts(limit: int = 10):
    """Get recent compliance alerts"""
    if not legal_financial_counsel:
        raise HTTPException(status_code=503, detail="Legal-Financial Counsel not initialized")
    
    alerts = legal_financial_counsel.compliance_alerts[-limit:]
    return {
        "total_alerts": len(legal_financial_counsel.compliance_alerts),
        "recent_alerts": alerts
    }

@router.get("/legal-financial/asset-portfolio")
async def get_asset_portfolio():
    """Get current asset portfolio analysis"""
    if not legal_financial_counsel:
        raise HTTPException(status_code=503, detail="Legal-Financial Counsel not initialized")
    
    portfolio_analysis = await legal_financial_counsel._analyze_asset_portfolio()
    return portfolio_analysis

@router.post("/legal-financial/assess-partner-risk")
async def assess_partner_risk(request: PartnerRiskAssessmentRequest):
    """Assess partner risk profile"""
    if not legal_financial_counsel:
        raise HTTPException(status_code=503, detail="Legal-Financial Counsel not initialized")
    
    try:
        risk_level = await legal_financial_counsel._analyze_partner_risk(request.partner_data)
        return {
            "partner_name": request.partner_data.get('name', 'Unknown'),
            "risk_level": risk_level,
            "recommendations": await legal_financial_counsel._generate_risk_recommendations(risk_level)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Risk assessment failed: {str(e)}")

@router.get("/legal-financial/compensation-optimization")
async def get_compensation_optimization_guide():
    """Get AI-optimized compensation strategies"""
    if not legal_financial_counsel:
        raise HTTPException(status_code=503, detail="Legal-Financial Counsel not initialized")
    
    return {
        "optimization_strategies": {
            "high_risk_partner": "Higher cash allocation (60%), lower equity (20%), balanced assets (20%)",
            "strategic_partner": "Lower cash (20%), higher equity (50%), strategic assets (30%)",
            "influencer_partner": "Balanced cash (40%), performance equity (30%), visible assets (30%)",
            "standard_partner": "Optimized mix: Cash (30%), Equity (40%), Inflation-proof assets (30%)"
        },
        "asset_prioritization": [
            "Gold & precious metals (inflation protection)",
            "Strategic real estate (long-term appreciation)", 
            "Digital assets (high growth potential)",
            "Luxury assets (negotiation leverage)"
        ]
    }
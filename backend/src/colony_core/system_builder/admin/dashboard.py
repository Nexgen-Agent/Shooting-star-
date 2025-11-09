from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session
from typing import List, Dict, Any
import json

from ..database import get_db
from ..models.core import CoreColony, BrandColony
from ..system_builder.website_generator import WebsiteGenerator
from ..system_builder.admin_panel_generator import AdminPanelGenerator

router = APIRouter()
website_gen = WebsiteGenerator()
admin_gen = AdminPanelGenerator()

@router.get("/core/admin", response_class=HTMLResponse)
async def core_admin_dashboard(db: Session = Depends(get_db)):
    """Core Colony Admin Dashboard - Overview of all Brand Colonies"""
    
    core_colonies = db.query(CoreColony).all()
    brand_colonies = db.query(BrandColony).all()
    
    dashboard_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>The Colony - Core Admin Dashboard</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }}
            .header {{ background: #2c3e50; color: white; padding: 1rem 2rem; margin-bottom: 2rem; }}
            .stats-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-bottom: 2rem; }}
            .stat-card {{ background: white; padding: 1.5rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            .brands-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 1rem; }}
            .brand-card {{ background: white; padding: 1.5rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            .btn {{ background: #3498db; color: white; padding: 0.5rem 1rem; border: none; border-radius: 4px; cursor: pointer; }}
            .btn-success {{ background: #27ae60; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🏛️ The Colony - Core Administration</h1>
            <p>AI-Powered Business System Builder</p>
        </div>
        
        <div class="stats-grid">
            <div class="stat-card">
                <h3>Core Colonies</h3>
                <p class="stat-number">{len(core_colonies)}</p>
            </div>
            <div class="stat-card">
                <h3>Brand Colonies</h3>
                <p class="stat-number">{len(brand_colonies)}</p>
            </div>
            <div class="stat-card">
                <h3>Active Websites</h3>
                <p class="stat-number">{len([b for b in brand_colonies if b.website_url])}</p>
            </div>
            <div class="stat-card">
                <h3>System Health</h3>
                <p class="stat-number">100%</p>
            </div>
        </div>
        
        <div class="actions">
            <button class="btn btn-success" onclick="createNewBrand()">+ Create New Brand System</button>
        </div>
        
        <h2>Brand Colonies</h2>
        <div class="brands-grid">
            {"".join([generate_brand_card(brand) for brand in brand_colonies])}
        </div>
        
        <script>
            function createNewBrand() {{
                window.location.href = '/core/admin/brands/create';
            }}
            
            function viewBrandDetails(brandId) {{
                window.location.href = '/core/admin/brands/' + brandId;
            }}
        </script>
    </body>
    </html>
    """
    
    return dashboard_html

def generate_brand_card(brand: BrandColony) -> str:
    """Generate HTML card for a brand colony"""
    return f"""
    <div class="brand-card">
        <h3>{brand.name}</h3>
        <p><strong>Industry:</strong> {brand.industry}</p>
        <p><strong>Status:</strong> <span style="color: green;">Active</span></p>
        <p><strong>Website:</strong> {brand.website_url or "Not deployed"}</p>
        <div class="actions">
            <button class="btn" onclick="viewBrandDetails('{brand.id}')">Manage</button>
            <button class="btn" onclick="window.open('{brand.website_url}', '_blank')">Visit Website</button>
        </div>
    </div>
    """

@router.post("/core/admin/brands/create")
async def create_brand_system(brand_data: Dict[str, Any], db: Session = Depends(get_db)):
    """Create complete brand system (website + admin panel)"""
    
    try:
        # Generate website
        website_result = website_gen.generate_brand_website(brand_data)
        
        # Generate admin panel
        admin_result = admin_gen.generate_admin_panel(
            brand_data, 
            brand_data.get("features", [])
        )
        
        # Create brand colony record
        brand_colony = BrandColony(
            name=brand_data["name"],
            industry=brand_data.get("industry", "general"),
            website_url=website_result["website_url"],
            admin_panel_url=website_result["admin_panel_url"],
            config={
                "website_data": website_result,
                "admin_data": admin_result,
                "features": brand_data.get("features", [])
            }
        )
        
        db.add(brand_colony)
        db.commit()
        db.refresh(brand_colony)
        
        return {
            "status": "success",
            "message": f"Brand system created for {brand_data['name']}",
            "website_url": website_result["website_url"],
            "admin_panel_url": website_result["admin_panel_url"],
            "brand_id": brand_colony.id
        }
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to create brand system: {str(e)}")
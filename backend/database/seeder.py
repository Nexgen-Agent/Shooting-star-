"""
Database seeds for demo data and testing.
"""

from sqlalchemy.ext.asyncio import AsyncSession
from database.models.user import User
from database.models.brand import Brand
from database.models.campaign import Campaign
from core.security import get_password_hash
from config.constants import UserRole
import logging

logger = logging.getLogger(__name__)


async def seed_database(db: AsyncSession):
    """
    Seed the database with demo data.
    
    Args:
        db: Database session
    """
    try:
        logger.info("Starting database seeding...")
        
        # Create super admin
        super_admin = User(
            email="admin@shootingstar.com",
            hashed_password=get_password_hash("admin123"),
            first_name="System",
            last_name="Administrator",
            role=UserRole.SUPER_ADMIN,
            is_active=True,
            is_verified=True
        )
        db.add(super_admin)
        
        # Create demo brands
        brands_data = [
            {
                "name": "NexGen Tech",
                "industry": "Technology",
                "description": "Innovative tech solutions provider",
                "monthly_budget": 50000.00,
                "current_balance": 25000.00
            },
            {
                "name": "EcoLife Foods",
                "industry": "Food & Beverage", 
                "description": "Organic and sustainable food products",
                "monthly_budget": 30000.00,
                "current_balance": 15000.00
            }
        ]
        
        brands = []
        for brand_data in brands_data:
            brand = Brand(**brand_data)
            db.add(brand)
            brands.append(brand)
        
        await db.commit()
        
        # Refresh to get IDs
        for brand in brands:
            await db.refresh(brand)
        
        # Create brand owners
        brand_owners = [
            {
                "email": "owner@nexgentech.com",
                "first_name": "Alex",
                "last_name": "Johnson",
                "brand_id": brands[0].id,
                "role": UserRole.BRAND_OWNER
            },
            {
                "email": "owner@ecolife.com", 
                "first_name": "Sarah",
                "last_name": "Miller",
                "brand_id": brands[1].id,
                "role": UserRole.BRAND_OWNER
            }
        ]
        
        for owner_data in brand_owners:
            owner = User(
                email=owner_data["email"],
                hashed_password=get_password_hash("password123"),
                first_name=owner_data["first_name"],
                last_name=owner_data["last_name"],
                brand_id=owner_data["brand_id"],
                role=owner_data["role"],
                is_active=True,
                is_verified=True
            )
            db.add(owner)
        
        await db.commit()
        
        logger.info("Database seeding completed successfully")
        
    except Exception as e:
        await db.rollback()
        logger.error(f"Error seeding database: {str(e)}")
        raise
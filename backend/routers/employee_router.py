"""
Employee router for employee and department management.
"""

from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
import logging

from database.connection import get_db
from database.models.user import User
from database.models.department import Department
from core.security import require_roles, get_current_user
from core.utils import response_formatter, paginator
from config.constants import UserRole
from services.auth_service import AuthService

# Create router
router = APIRouter(prefix="/employees", tags=["employees"])

logger = logging.getLogger(__name__)


@router.post("/", response_model=Dict[str, Any])
async def create_employee(
    employee_data: dict,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_roles([UserRole.SUPER_ADMIN, UserRole.BRAND_OWNER]))
):
    """
    Create a new employee.
    
    Args:
        employee_data: Employee creation data
        db: Database session
        current_user: Current authenticated user
        
    Returns:
        Created employee data
    """
    try:
        auth_service = AuthService(db)
        
        # Check permissions
        brand_id = employee_data.get("brand_id")
        if (current_user.role == UserRole.BRAND_OWNER and 
            str(current_user.brand_id) != brand_id):
            raise response_formatter.error(
                message="Cannot create employees for other brands",
                error_code="PERMISSION_DENIED",
                status_code=status.HTTP_403_FORBIDDEN
            )
        
        # Use current user's brand if not specified for brand owners
        if current_user.role == UserRole.BRAND_OWNER and not brand_id:
            employee_data["brand_id"] = str(current_user.brand_id)
        
        # Set role to employee
        employee_data["role"] = UserRole.EMPLOYEE
        
        success, user, message = await auth_service.create_user(**employee_data)
        
        if not success:
            raise response_formatter.error(
                message=message,
                error_code="EMPLOYEE_CREATION_FAILED",
                status_code=status.HTTP_400_BAD_REQUEST
            )
        
        logger.info(f"Employee created by {current_user.email}: {employee_data['email']}")
        return response_formatter.success(
            data=user.to_dict(),
            message=message
        )
        
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        logger.error(f"Error creating employee: {str(e)}")
        raise response_formatter.error(
            message="Error creating employee",
            error_code="EMPLOYEE_CREATION_ERROR",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@router.get("/", response_model=Dict[str, Any])
async def list_employees(
    brand_id: Optional[str] = None,
    department_id: Optional[str] = None,
    role: Optional[UserRole] = None,
    is_active: Optional[bool] = None,
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    List employees with filtering and pagination.
    
    Args:
        brand_id: Filter by brand ID
        department_id: Filter by department ID
        role: Filter by role
        is_active: Filter by active status
        page: Page number
        per_page: Items per page
        db: Database session
        current_user: Current authenticated user
        
    Returns:
        Paginated list of employees
    """
    try:
        auth_service = AuthService(db)
        
        # Determine brand context
        if current_user.role == UserRole.SUPER_ADMIN:
            target_brand_id = brand_id
        else:
            target_brand_id = str(current_user.brand_id) if current_user.brand_id else None
        
        # Check permissions
        if (current_user.role in [UserRole.BRAND_OWNER, UserRole.EMPLOYEE] and 
            brand_id and str(current_user.brand_id) != brand_id):
            raise response_formatter.error(
                message="Access denied to other brands' employees",
                error_code="ACCESS_DENIED",
                status_code=status.HTTP_403_FORBIDDEN
            )
        
        # Filter by employee role by default
        if not role:
            role = UserRole.EMPLOYEE
        
        users, total_count = await auth_service.list_users(
            brand_id=target_brand_id,
            role=role,
            page=page,
            per_page=per_page
        )
        
        # Additional filtering by department
        if department_id:
            from sqlalchemy import select
            from database.models.user import User
            
            department_users_result = await db.execute(
                select(User).where(User.brand_id == target_brand_id)
            )
            all_brand_users = department_users_result.scalars().all()
            
            # This is a simplified filter - in a real app, you'd have a proper department association
            users = [user for user in users if str(user.id) in [u.id for u in all_brand_users]]
            total_count = len(users)
        
        if is_active is not None:
            users = [user for user in users if user.is_active == is_active]
            total_count = len(users)
        
        # Apply pagination after filtering
        from core.utils import Paginator
        paginator_obj = Paginator(page, per_page)
        start_idx = paginator_obj.offset
        end_idx = start_idx + paginator_obj.per_page
        paginated_users = users[start_idx:end_idx]
        
        # Create metadata
        meta = paginator.create_metadata(total_count, page, per_page)
        
        return response_formatter.success(
            data=[user.to_dict() for user in paginated_users],
            meta=meta,
            message="Employees retrieved successfully"
        )
        
    except Exception as e:
        logger.error(f"Error listing employees: {str(e)}")
        raise response_formatter.error(
            message="Error retrieving employees",
            error_code="EMPLOYEES_RETRIEVAL_FAILED",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@router.get("/departments", response_model=Dict[str, Any])
async def list_departments(
    brand_id: Optional[str] = None,
    is_active: Optional[bool] = None,
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    List departments with filtering and pagination.
    
    Args:
        brand_id: Filter by brand ID
        is_active: Filter by active status
        page: Page number
        per_page: Items per page
        db: Database session
        current_user: Current authenticated user
        
    Returns:
        Paginated list of departments
    """
    try:
        from sqlalchemy import select, func
        
        # Build query
        if current_user.role == UserRole.SUPER_ADMIN:
            query = select(Department)
            if brand_id:
                query = query.where(Department.brand_id == brand_id)
        else:
            query = select(Department).where(Department.brand_id == current_user.brand_id)
        
        if is_active is not None:
            query = query.where(Department.is_active == is_active)
        
        # Get total count
        count_query = select(func.count()).select_from(query.subquery())
        total_result = await db.execute(count_query)
        total_count = total_result.scalar_one()
        
        # Apply pagination
        query = paginator.paginate_query(query, page, per_page)
        
        # Order by name
        query = query.order_by(Department.name.asc())
        
        # Execute query
        result = await db.execute(query)
        departments = result.scalars().all()
        
        # Create metadata
        meta = paginator.create_metadata(total_count, page, per_page)
        
        return response_formatter.success(
            data=[department.to_dict() for department in departments],
            meta=meta,
            message="Departments retrieved successfully"
        )
        
    except Exception as e:
        logger.error(f"Error listing departments: {str(e)}")
        raise response_formatter.error(
            message="Error retrieving departments",
            error_code="DEPARTMENTS_RETRIEVAL_FAILED",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@router.post("/departments", response_model=Dict[str, Any])
async def create_department(
    department_data: dict,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_roles([UserRole.SUPER_ADMIN, UserRole.BRAND_OWNER]))
):
    """
    Create a new department.
    
    Args:
        department_data: Department creation data
        db: Database session
        current_user: Current authenticated user
        
    Returns:
        Created department data
    """
    try:
        from sqlalchemy import select
        
        # Check permissions
        brand_id = department_data.get("brand_id")
        if (current_user.role == UserRole.BRAND_OWNER and 
            str(current_user.brand_id) != brand_id):
            raise response_formatter.error(
                message="Cannot create departments for other brands",
                error_code="PERMISSION_DENIED",
                status_code=status.HTTP_403_FORBIDDEN
            )
        
        # Use current user's brand if not specified for brand owners
        if current_user.role == UserRole.BRAND_OWNER and not brand_id:
            department_data["brand_id"] = str(current_user.brand_id)
        
        # Verify brand exists
        from database.models.brand import Brand
        result = await db.execute(
            select(Brand).where(Brand.id == department_data["brand_id"])
        )
        brand = result.scalar_one_or_none()
        
        if not brand:
            raise response_formatter.error(
                message="Brand not found",
                error_code="BRAND_NOT_FOUND",
                status_code=status.HTTP_404_NOT_FOUND
            )
        
        # Create department
        department = Department(**department_data)
        db.add(department)
        await db.commit()
        await db.refresh(department)
        
        logger.info(f"Department created by {current_user.email}: {department_data['name']}")
        return response_formatter.success(
            data=department.to_dict(),
            message="Department created successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        logger.error(f"Error creating department: {str(e)}")
        raise response_formatter.error(
            message="Error creating department",
            error_code="DEPARTMENT_CREATION_FAILED",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@router.put("/departments/{department_id}", response_model=Dict[str, Any])
async def update_department(
    department_id: str,
    update_data: dict,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_roles([UserRole.SUPER_ADMIN, UserRole.BRAND_OWNER]))
):
    """
    Update department information.
    
    Args:
        department_id: Department ID
        update_data: Department update data
        db: Database session
        current_user: Current authenticated user
        
    Returns:
        Updated department data
    """
    try:
        from sqlalchemy import select, update
        
        # Get department
        result = await db.execute(
            select(Department).where(Department.id == department_id)
        )
        department = result.scalar_one_or_none()
        
        if not department:
            raise response_formatter.error(
                message="Department not found",
                error_code="DEPARTMENT_NOT_FOUND",
                status_code=status.HTTP_404_NOT_FOUND
            )
        
        # Check permissions
        if (current_user.role == UserRole.BRAND_OWNER and 
            str(current_user.brand_id) != str(department.brand_id)):
            raise response_formatter.error(
                message="Cannot update departments from other brands",
                error_code="PERMISSION_DENIED",
                status_code=status.HTTP_403_FORBIDDEN
            )
        
        # Filter allowed fields for update
        allowed_fields = {
            "name", "description", "department_type", "budget_allocated",
            "team_size", "manager_id", "permissions", "settings", "is_active"
        }
        
        filtered_data = {
            k: v for k, v in update_data.items() 
            if k in allowed_fields and v is not None
        }
        
        # Update department
        await db.execute(
            update(Department)
            .where(Department.id == department_id)
            .values(**filtered_data)
        )
        await db.commit()
        
        # Refresh department data
        await db.refresh(department)
        
        logger.info(f"Department updated by {current_user.email}: {department.name}")
        return response_formatter.success(
            data=department.to_dict(),
            message="Department updated successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        logger.error(f"Error updating department {department_id}: {str(e)}")
        raise response_formatter.error(
            message="Error updating department",
            error_code="DEPARTMENT_UPDATE_FAILED",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@router.put("/{user_id}/department", response_model=Dict[str, Any])
async def assign_employee_to_department(
    user_id: str,
    assignment_data: dict,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_roles([UserRole.SUPER_ADMIN, UserRole.BRAND_OWNER]))
):
    """
    Assign employee to department.
    
    Args:
        user_id: User ID
        assignment_data: Assignment data
        db: Database session
        current_user: Current authenticated user
        
    Returns:
        Assignment result
    """
    try:
        from sqlalchemy import select, update
        from database.models.user import User
        
        # Get user
        result = await db.execute(
            select(User).where(User.id == user_id)
        )
        user = result.scalar_one_or_none()
        
        if not user:
            raise response_formatter.error(
                message="User not found",
                error_code="USER_NOT_FOUND",
                status_code=status.HTTP_404_NOT_FOUND
            )
        
        # Check permissions
        if (current_user.role == UserRole.BRAND_OWNER and 
            str(current_user.brand_id) != str(user.brand_id)):
            raise response_formatter.error(
                message="Cannot assign employees from other brands",
                error_code="PERMISSION_DENIED",
                status_code=status.HTTP_403_FORBIDDEN
            )
        
        # For now, we'll store department info in user metadata
        # In a real app, you'd have a proper many-to-many relationship
        current_metadata = user.to_dict().get("metadata", {})
        current_metadata["department"] = assignment_data.get("department_id")
        
        await db.execute(
            update(User)
            .where(User.id == user_id)
            .values(metadata=current_metadata)
        )
        await db.commit()
        
        logger.info(f"Employee {user.email} assigned to department by {current_user.email}")
        return response_formatter.success(
            message="Employee assigned to department successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        logger.error(f"Error assigning employee to department: {str(e)}")
        raise response_formatter.error(
            message="Error assigning employee to department",
            error_code="DEPARTMENT_ASSIGNMENT_FAILED",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@router.get("/stats", response_model=Dict[str, Any])
async def get_employee_stats(
    brand_id: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get employee statistics.
    
    Args:
        brand_id: Brand ID (super admin only)
        db: Database session
        current_user: Current authenticated user
        
    Returns:
        Employee statistics
    """
    try:
        from sqlalchemy import select, func
        from database.models.user import User
        
        # Determine brand context
        if current_user.role == UserRole.SUPER_ADMIN:
            target_brand_id = brand_id
        else:
            target_brand_id = str(current_user.brand_id) if current_user.brand_id else None
        
        # Check permissions
        if (current_user.role in [UserRole.BRAND_OWNER, UserRole.EMPLOYEE] and 
            brand_id and str(current_user.brand_id) != brand_id):
            raise response_formatter.error(
                message="Access denied to other brands' statistics",
                error_code="ACCESS_DENIED",
                status_code=status.HTTP_403_FORBIDDEN
            )
        
        if not target_brand_id:
            raise response_formatter.error(
                message="No brand context available",
                error_code="NO_BRAND_CONTEXT",
                status_code=status.HTTP_400_BAD_REQUEST
            )
        
        # Get employee counts by role
        total_employees_result = await db.execute(
            select(func.count(User.id))
            .where(User.brand_id == target_brand_id)
            .where(User.role == UserRole.EMPLOYEE)
        )
        total_employees = total_employees_result.scalar_one()
        
        active_employees_result = await db.execute(
            select(func.count(User.id))
            .where(User.brand_id == target_brand_id)
            .where(User.role == UserRole.EMPLOYEE)
            .where(User.is_active == True)
        )
        active_employees = active_employees_result.scalar_one()
        
        # Get department stats
        from database.models.department import Department
        departments_result = await db.execute(
            select(Department)
            .where(Department.brand_id == target_brand_id)
            .where(Department.is_active == True)
        )
        departments = departments_result.scalars().all()
        
        stats = {
            "total_employees": total_employees,
            "active_employees": active_employees,
            "inactive_employees": total_employees - active_employees,
            "total_departments": len(departments),
            "departments": [
                {
                    "id": str(dept.id),
                    "name": dept.name,
                    "department_type": dept.department_type,
                    "team_size": dept.team_size,
                    "budget_allocated": float(dept.budget_allocated or 0)
                }
                for dept in departments
            ]
        }
        
        return response_formatter.success(
            data=stats,
            message="Employee statistics retrieved successfully"
        )
        
    except Exception as e:
        logger.error(f"Error getting employee stats: {str(e)}")
        raise response_formatter.error(
            message="Error retrieving employee statistics",
            error_code="EMPLOYEE_STATS_FAILED",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
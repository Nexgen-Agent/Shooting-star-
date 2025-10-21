"""
Workspace Builder V16 - Campaign workspace creation and management system
for the Shooting Star V16 admin platform.
"""

from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from enum import Enum
import asyncio
import logging
from datetime import datetime, timedelta
import uuid
from collections import defaultdict

logger = logging.getLogger(__name__)

class WorkspaceType(Enum):
    CAMPAIGN = "campaign"
    BRAND = "brand"
    TEAM = "team"
    PROJECT = "project"

class WorkspaceAccessLevel(Enum):
    OWNER = "owner"
    ADMIN = "admin"
    EDITOR = "editor"
    VIEWER = "viewer"

class WorkspaceMember(BaseModel):
    user_id: str
    access_level: WorkspaceAccessLevel
    joined_at: datetime
    role: Optional[str] = None

class WorkspaceResource(BaseModel):
    resource_id: str
    type: str  # file, link, integration, etc.
    name: str
    description: Optional[str] = None
    url: Optional[str] = None
    created_by: str
    created_at: datetime

class Workspace(BaseModel):
    workspace_id: str
    name: str
    description: str
    workspace_type: WorkspaceType
    campaign_id: Optional[str] = None
    brand_id: Optional[str] = None
    created_by: str
    members: List[WorkspaceMember]
    resources: List[WorkspaceResource]
    tags: List[str] = Field(default_factory=list)
    ai_recommendations: List[Dict[str, Any]] = Field(default_factory=list)
    is_active: bool = True
    created_at: datetime
    updated_at: datetime

class WorkspaceBuilderV16:
    """
    Advanced workspace builder for V16 campaign management
    """
    
    def __init__(self):
        self.workspaces: Dict[str, Workspace] = {}
        self.workspace_templates: Dict[str, Dict[str, Any]] = self._initialize_templates()
    
    def _initialize_templates(self) -> Dict[str, Dict[str, Any]]:
        """Initialize AI-powered workspace templates"""
        return {
            "social_media_campaign": {
                "name": "Social Media Campaign Template",
                "description": "Complete workspace for social media campaign management",
                "default_members": [
                    {"role": "campaign_manager", "access_level": "admin"},
                    {"role": "content_creator", "access_level": "editor"},
                    {"role": "analyst", "access_level": "editor"}
                ],
                "default_resources": [
                    {"type": "integration", "name": "Social Media Dashboard", "description": "Live social media analytics"},
                    {"type": "file", "name": "Campaign Brief", "description": "Campaign strategy and objectives"},
                    {"type": "link", "name": "Content Calendar", "description": "Content scheduling and planning"}
                ],
                "ai_suggestions": [
                    {
                        "type": "resource_optimization",
                        "title": "Recommended Integrations",
                        "description": "AI-suggested tools and integrations for campaign success",
                        "tools": ["Social listening tool", "Content performance tracker", "ROI calculator"]
                    }
                ]
            },
            "influencer_collaboration": {
                "name": "Influencer Collaboration Template", 
                "description": "Workspace for managing influencer partnerships",
                "default_members": [
                    {"role": "partnership_manager", "access_level": "admin"},
                    {"role": "influencer_coordinator", "access_level": "editor"},
                    {"role": "legal_review", "access_level": "viewer"}
                ],
                "default_resources": [
                    {"type": "file", "name": "Influencer Agreement", "description": "Partnership terms and conditions"},
                    {"type": "integration", "name": "Influencer Tracker", "description": "Performance and payment tracking"},
                    {"type": "link", "name": "Content Guidelines", "description": "Brand guidelines for influencers"}
                ]
            },
            "performance_analysis": {
                "name": "Performance Analysis Template",
                "description": "Workspace for campaign performance analysis and reporting",
                "default_members": [
                    {"role": "data_analyst", "access_level": "admin"},
                    {"role": "campaign_manager", "access_level": "editor"},
                    {"role": "stakeholder", "access_level": "viewer"}
                ],
                "default_resources": [
                    {"type": "integration", "name": "Analytics Dashboard", "description": "Real-time performance metrics"},
                    {"type": "file", "name": "Report Template", "description": "Standardized reporting format"},
                    {"type": "link", "name": "KPI Definitions", "description": "Key performance indicators"}
                ]
            }
        }
    
    async def create_workspace(self, workspace_data: Dict[str, Any]) -> Workspace:
        """
        Create a new workspace with AI-powered optimizations
        """
        try:
            workspace_id = f"workspace_{uuid.uuid4().hex[:8]}"
            now = datetime.utcnow()
            
            # Get template if specified
            template_name = workspace_data.get("template")
            template = self.workspace_templates.get(template_name, {})
            
            # Create members list
            members = []
            creator_member = WorkspaceMember(
                user_id=workspace_data["created_by"],
                access_level=WorkspaceAccessLevel.OWNER,
                joined_at=now,
                role="creator"
            )
            members.append(creator_member)
            
            # Add template members if any
            for member_template in template.get("default_members", []):
                # In real implementation, you'd resolve user_ids from roles
                mock_user_id = f"user_{uuid.uuid4().hex[:6]}"
                member = WorkspaceMember(
                    user_id=mock_user_id,
                    access_level=WorkspaceAccessLevel(member_template["access_level"]),
                    joined_at=now,
                    role=member_template["role"]
                )
                members.append(member)
            
            # Create resources from template
            resources = []
            for resource_template in template.get("default_resources", []):
                resource = WorkspaceResource(
                    resource_id=f"resource_{uuid.uuid4().hex[:6]}",
                    type=resource_template["type"],
                    name=resource_template["name"],
                    description=resource_template.get("description"),
                    created_by=workspace_data["created_by"],
                    created_at=now
                )
                resources.append(resource)
            
            # Generate AI recommendations
            ai_recommendations = await self._generate_workspace_recommendations(
                workspace_data, template
            )
            
            workspace = Workspace(
                workspace_id=workspace_id,
                name=workspace_data["name"],
                description=workspace_data["description"],
                workspace_type=WorkspaceType(workspace_data["workspace_type"]),
                campaign_id=workspace_data.get("campaign_id"),
                brand_id=workspace_data.get("brand_id"),
                created_by=workspace_data["created_by"],
                members=members,
                resources=resources,
                tags=workspace_data.get("tags", []),
                ai_recommendations=ai_recommendations,
                created_at=now,
                updated_at=now
            )
            
            self.workspaces[workspace_id] = workspace
            logger.info(f"Created workspace {workspace_id}: {workspace.name}")
            
            return workspace
            
        except Exception as e:
            logger.error(f"Workspace creation failed: {str(e)}")
            raise
    
    async def _generate_workspace_recommendations(self, workspace_data: Dict[str, Any], 
                                                template: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate AI-powered recommendations for workspace optimization
        """
        recommendations = []
        workspace_type = workspace_data.get("workspace_type")
        
        # Campaign workspace recommendations
        if workspace_type == "campaign":
            recommendations.append({
                "type": "campaign_structure",
                "title": "Optimal Campaign Setup",
                "description": "AI-recommended campaign workspace structure",
                "confidence": 0.82,
                "suggestions": [
                    "Set up separate channels for different social platforms",
                    "Create standardized reporting templates",
                    "Establish clear approval workflows"
                ]
            })
        
        # Team collaboration recommendations
        recommendations.append({
            "type": "collaboration_optimization",
            "title": "Team Collaboration Setup",
            "description": "Recommended tools and processes for team collaboration",
            "confidence": 0.78,
            "suggestions": [
                "Schedule weekly sync meetings",
                "Set up automated progress reporting",
                "Create knowledge sharing channels"
            ]
        })
        
        # Add template-specific recommendations
        if template.get("ai_suggestions"):
            recommendations.extend(template["ai_suggestions"])
        
        return recommendations
    
    async def add_member_to_workspace(self, workspace_id: str, user_id: str, 
                                    access_level: WorkspaceAccessLevel, role: Optional[str] = None) -> Workspace:
        """
        Add member to workspace with specified access level
        """
        if workspace_id not in self.workspaces:
            raise ValueError(f"Workspace {workspace_id} not found")
        
        workspace = self.workspaces[workspace_id]
        
        # Check if member already exists
        for member in workspace.members:
            if member.user_id == user_id:
                # Update existing member
                member.access_level = access_level
                member.role = role or member.role
                workspace.updated_at = datetime.utcnow()
                return workspace
        
        # Add new member
        new_member = WorkspaceMember(
            user_id=user_id,
            access_level=access_level,
            joined_at=datetime.utcnow(),
            role=role
        )
        workspace.members.append(new_member)
        workspace.updated_at = datetime.utcnow()
        
        logger.info(f"Added member {user_id} to workspace {workspace_id}")
        return workspace
    
    async def add_resource_to_workspace(self, workspace_id: str, resource_data: Dict[str, Any]) -> Workspace:
        """
        Add resource to workspace
        """
        if workspace_id not in self.workspaces:
            raise ValueError(f"Workspace {workspace_id} not found")
        
        workspace = self.workspaces[workspace_id]
        
        resource = WorkspaceResource(
            resource_id=f"resource_{uuid.uuid4().hex[:6]}",
            type=resource_data["type"],
            name=resource_data["name"],
            description=resource_data.get("description"),
            url=resource_data.get("url"),
            created_by=resource_data["created_by"],
            created_at=datetime.utcnow()
        )
        
        workspace.resources.append(resource)
        workspace.updated_at = datetime.utcnow()
        
        logger.info(f"Added resource {resource.name} to workspace {workspace_id}")
        return workspace
    
    async def get_workspace_by_campaign(self, campaign_id: str) -> Optional[Workspace]:
        """Get workspace associated with a campaign"""
        for workspace in self.workspaces.values():
            if workspace.campaign_id == campaign_id:
                return workspace
        return None
    
    async def get_user_workspaces(self, user_id: str) -> List[Workspace]:
        """Get all workspaces a user has access to"""
        user_workspaces = []
        
        for workspace in self.workspaces.values():
            if any(member.user_id == user_id for member in workspace.members):
                user_workspaces.append(workspace)
        
        return sorted(user_workspaces, key=lambda x: x.updated_at, reverse=True)
    
    async def generate_workspace_report(self, workspace_id: str) -> Dict[str, Any]:
        """Generate comprehensive workspace report"""
        if workspace_id not in self.workspaces:
            raise ValueError(f"Workspace {workspace_id} not found")
        
        workspace = self.workspaces[workspace_id]
        
        # Calculate workspace health metrics
        total_members = len(workspace.members)
        active_members = len([m for m in workspace.members 
                            if m.access_level in [WorkspaceAccessLevel.OWNER, WorkspaceAccessLevel.ADMIN, WorkspaceAccessLevel.EDITOR]])
        
        resource_distribution = defaultdict(int)
        for resource in workspace.resources:
            resource_distribution[resource.type] += 1
        
        workspace_age_days = (datetime.utcnow() - workspace.created_at).days
        
        return {
            "workspace_id": workspace_id,
            "workspace_name": workspace.name,
            "report_generated": datetime.utcnow().isoformat(),
            "membership_metrics": {
                "total_members": total_members,
                "active_members": active_members,
                "admin_count": len([m for m in workspace.members if m.access_level == WorkspaceAccessLevel.ADMIN]),
                "editor_count": len([m for m in workspace.members if m.access_level == WorkspaceAccessLevel.EDITOR]),
                "viewer_count": len([m for m in workspace.members if m.access_level == WorkspaceAccessLevel.VIEWER])
            },
            "resource_metrics": {
                "total_resources": len(workspace.resources),
                "resource_distribution": dict(resource_distribution),
                "recent_resources": len([r for r in workspace.resources 
                                       if (datetime.utcnow() - r.created_at).days <= 7])
            },
            "activity_metrics": {
                "workspace_age_days": workspace_age_days,
                "last_updated": workspace.updated_at.isoformat(),
                "ai_recommendations_count": len(workspace.ai_recommendations)
            },
            "health_score": await self._calculate_workspace_health(workspace)
        }
    
    async def _calculate_workspace_health(self, workspace: Workspace) -> float:
        """Calculate workspace health score (0-100)"""
        score = 0.0
        
        # Member diversity score
        access_levels = len(set(m.access_level for m in workspace.members))
        score += min(access_levels * 10, 20)  # Max 20 points
        
        # Resource diversity score
        resource_types = len(set(r.type for r in workspace.resources))
        score += min(resource_types * 5, 15)  # Max 15 points
        
        # Activity score (based on recent updates)
        days_since_update = (datetime.utcnow() - workspace.updated_at).days
        if days_since_update <= 1:
            score += 25
        elif days_since_update <= 7:
            score += 15
        elif days_since_update <= 30:
            score += 5
        
        # AI recommendations implementation score
        score += min(len(workspace.ai_recommendations) * 2, 10)  # Max 10 points
        
        # Basic completeness score
        if workspace.description and len(workspace.resources) > 0 and len(workspace.members) > 1:
            score += 30
        
        return min(score, 100)
    
    def get_available_templates(self) -> List[Dict[str, Any]]:
        """Get list of available workspace templates"""
        templates = []
        for template_id, template_data in self.workspace_templates.items():
            templates.append({
                "template_id": template_id,
                "name": template_data["name"],
                "description": template_data["description"],
                "suggested_use_cases": self._get_template_use_cases(template_id),
                "estimated_setup_time": "5-10 minutes"
            })
        return templates
    
    def _get_template_use_cases(self, template_id: str) -> List[str]:
        """Get suggested use cases for a template"""
        use_cases = {
            "social_media_campaign": [
                "Social media marketing campaigns",
                "Content marketing initiatives", 
                "Brand awareness campaigns"
            ],
            "influencer_collaboration": [
                "Influencer partnership management",
                "Brand ambassador programs",
                "Sponsored content coordination"
            ],
            "performance_analysis": [
                "Campaign performance tracking",
                "ROI analysis and reporting",
                "Marketing effectiveness studies"
            ]
        }
        return use_cases.get(template_id, [])
    
    def get_builder_metrics(self) -> Dict[str, Any]:
        """Get workspace builder performance metrics"""
        total_workspaces = len(self.workspaces)
        active_workspaces = len([w for w in self.workspaces.values() if w.is_active])
        
        total_members = sum(len(w.members) for w in self.workspaces.values())
        total_resources = sum(len(w.resources) for w in self.workspaces.values())
        
        return {
            "total_workspaces": total_workspaces,
            "active_workspaces": active_workspaces,
            "inactive_workspaces": total_workspaces - active_workspaces,
            "total_members": total_members,
            "total_resources": total_resources,
            "average_members_per_workspace": total_members / max(total_workspaces, 1),
            "average_resources_per_workspace": total_resources / max(total_workspaces, 1),
            "available_templates": len(self.workspace_templates),
            "timestamp": datetime.utcnow().isoformat()
        }


# Global workspace builder instance
workspace_builder = WorkspaceBuilderV16()


async def main():
    """Test harness for Workspace Builder"""
    print("🏢 Workspace Builder V16 - Test Harness")
    
    # Create test workspace
    workspace_data = {
        "name": "Q1 Social Media Blitz",
        "description": "Workspace for managing Q1 social media campaign",
        "workspace_type": "campaign",
        "campaign_id": "campaign_123",
        "brand_id": "brand_456",
        "created_by": "admin_001",
        "template": "social_media_campaign",
        "tags": ["q1", "social_media", "campaign"]
    }
    
    workspace = await workspace_builder.create_workspace(workspace_data)
    print("✅ Created Workspace:", workspace.name)
    print("👥 Members:", len(workspace.members))
    print("🔧 Resources:", len(workspace.resources))
    print("🤖 AI Recommendations:", len(workspace.ai_recommendations))
    
    # Add member
    workspace = await workspace_builder.add_member_to_workspace(
        workspace.workspace_id, "user_003", WorkspaceAccessLevel.EDITOR, "content_specialist"
    )
    print("➕ Added Member - Total:", len(workspace.members))
    
    # Generate report
    report = await workspace_builder.generate_workspace_report(workspace.workspace_id)
    print("📊 Workspace Health Score:", report["health_score"])
    
    # Get templates
    templates = workspace_builder.get_available_templates()
    print("📋 Available Templates:", len(templates))
    
    # Get builder metrics
    metrics = workspace_builder.get_builder_metrics()
    print("📈 Builder Metrics:", metrics)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
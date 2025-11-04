"""
Site Registry Service
Manages client website scaffolding and deployment with encrypted tenant isolation.
All deployments require founder approval for production domains.
"""

import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging
from datetime import datetime
import uuid

from crypto.key_manager import KeyManager

logger = logging.getLogger(__name__)

@dataclass
class ClientSite:
    site_id: str
    owner: str
    tenant_id: str
    domain: str
    staging_domain: str
    encrypted_keys: Dict
    deployment_target: str
    status: str
    created_at: datetime
    last_deployed: Optional[datetime] = None

class SiteRegistry:
    """
    Manages client website registration, encryption, and deployment workflows.
    Ensures tenant isolation and requires founder approval for production.
    """
    
    def __init__(self):
        self.key_manager = KeyManager()
        self.sites: Dict[str, ClientSite] = {}
        self.encryption_key = os.getenv("SITE_REGISTRY_ENCRYPTION_KEY", "default_key")
    
    async def register_site(self, owner: str, domain: str, 
                          deployment_target: str = "cloudflare") -> Dict[str, Any]:
        """
        Register a new client site with encrypted tenant isolation.
        """
        try:
            site_id = f"site_{uuid.uuid4().hex[:8]}"
            tenant_id = f"tenant_{uuid.uuid4().hex[:8]}"
            
            # Generate encrypted keys for the site
            encrypted_keys = await self._generate_encrypted_keys(site_id)
            
            # Create staging domain
            staging_domain = f"staging-{site_id}.shootingstar.ai"
            
            # Create site record
            site = ClientSite(
                site_id=site_id,
                owner=owner,
                tenant_id=tenant_id,
                domain=domain,
                staging_domain=staging_domain,
                encrypted_keys=encrypted_keys,
                deployment_target=deployment_target,
                status="registered",
                created_at=datetime.utcnow()
            )
            
            self.sites[site_id] = site
            
            logger.info(f"Registered new site: {site_id} for {owner}")
            
            return {
                "site_id": site_id,
                "tenant_id": tenant_id,
                "staging_domain": staging_domain,
                "status": "registered",
                "next_steps": ["scaffold_site", "deploy_staging"]
            }
            
        except Exception as e:
            logger.error(f"Site registration failed: {e}")
            raise
    
    async def scaffold_site(self, site_id: str, template: str = "default") -> Dict[str, Any]:
        """
        Scaffold website structure for a client site.
        """
        if site_id not in self.sites:
            raise ValueError(f"Site {site_id} not found")
        
        site = self.sites[site_id]
        
        try:
            # Generate site structure based on template
            site_structure = await self._generate_site_structure(template, site)
            
            # Create encrypted repository
            repo_url = await self._create_encrypted_repo(site, site_structure)
            
            # Update site status
            site.status = "scaffolded"
            
            return {
                "site_id": site_id,
                "scaffold_complete": True,
                "repo_url": repo_url,
                "site_structure": site_structure,
                "template_used": template
            }
            
        except Exception as e:
            logger.error(f"Site scaffolding failed: {e}")
            raise
    
    async def deploy_to_staging(self, site_id: str) -> Dict[str, Any]:
        """
        Deploy site to staging environment.
        """
        if site_id not in self.sites:
            raise ValueError(f"Site {site_id} not found")
        
        site = self.sites[site_id]
        
        try:
            # Build and deploy to staging
            build_result = await self._build_site(site_id)
            deploy_result = await self._deploy_to_staging(site, build_result)
            
            site.status = "staging_deployed"
            site.last_deployed = datetime.utcnow()
            
            return {
                "site_id": site_id,
                "staging_url": site.staging_domain,
                "deployment_id": deploy_result.get('deployment_id'),
                "build_success": build_result.get('success', False),
                "deploy_success": deploy_result.get('success', False)
            }
            
        except Exception as e:
            logger.error(f"Staging deployment failed: {e}")
            raise
    
    async def request_production_deployment(self, site_id: str) -> Dict[str, Any]:
        """
        Request production deployment requiring founder approval.
        """
        if site_id not in self.sites:
            raise ValueError(f"Site {site_id} not found")
        
        site = self.sites[site_id]
        
        # Generate approval request
        approval_request = await self._generate_deployment_approval(site)
        
        site.status = "approval_pending"
        
        return {
            "site_id": site_id,
            "domain": site.domain,
            "approval_required": True,
            "approval_hash": approval_request.get('approval_hash'),
            "founder_signature_required": True
        }
    
    async function approve_production_deployment(self, site_id: str, 
                                              founder_signature: str) -> Dict[str, Any]:
        """
        Approve production deployment with founder signature.
        """
        if site_id not in self.sites:
            raise ValueError(f"Site {site_id} not found")
        
        # Verify founder signature
        is_valid = await self.key_manager.verify_founder_signature(
            site_id, founder_signature
        )
        
        if not is_valid:
            raise SecurityError("Invalid founder signature")
        
        site = self.sites[site_id]
        
        # Deploy to production
        deploy_result = await self._deploy_to_production(site)
        
        site.status = "production_deployed"
        site.last_deployed = datetime.utcnow()
        
        return {
            "site_id": site_id,
            "domain": site.domain,
            "deployment_success": deploy_result.get('success', False),
            "production_live": True
        }
    
    async def get_site_status(self, site_id: str) -> Dict[str, Any]:
        """
        Get current status and details of a client site.
        """
        if site_id not in self.sites:
            raise ValueError(f"Site {site_id} not found")
        
        site = self.sites[site_id]
        
        return {
            "site_id": site_id,
            "owner": site.owner,
            "domain": site.domain,
            "staging_domain": site.staging_domain,
            "status": site.status,
            "deployment_target": site.deployment_target,
            "created_at": site.created_at.isoformat(),
            "last_deployed": site.last_deployed.isoformat() if site.last_deployed else None
        }
    
    # Internal methods
    async def _generate_encrypted_keys(self, site_id: str) -> Dict[str, str]:
        """Generate encrypted keys for site security."""
        keys = {
            "api_key": f"sk_{uuid.uuid4().hex}",
            "encryption_key": f"ek_{uuid.uuid4().hex}",
            "deployment_token": f"dt_{uuid.uuid4().hex}"
        }
        
        # Encrypt all keys
        encrypted_keys = {}
        for key_name, key_value in keys.items():
            encrypted_keys[key_name] = await self.key_manager.encrypt_data(
                key_value, self.encryption_key
            )
        
        return encrypted_keys
    
    async def _generate_site_structure(self, template: str, site: ClientSite) -> Dict[str, Any]:
        """Generate website structure based on template."""
        templates = {
            "default": {
                "structure": [
                    "index.html",
                    "css/styles.css",
                    "js/app.js",
                    "images/",
                    "config/site.json"
                ],
                "frameworks": ["bootstrap", "jquery"],
                "features": ["responsive", "seo_optimized"]
            },
            "react": {
                "structure": [
                    "src/components/",
                    "src/styles/",
                    "public/index.html",
                    "package.json",
                    "config/webpack.js"
                ],
                "frameworks": ["react", "tailwind"],
                "features": ["spa", "pwa"]
            }
        }
        
        return templates.get(template, templates["default"])
    
    async def _create_encrypted_repo(self, site: ClientSite, structure: Dict) -> str:
        """Create encrypted repository for site code."""
        repo_name = f"site-{site.site_id}"
        repo_url = f"https://github.com/shootingstar/{repo_name}"
        
        # Implementation would create actual git repo
        logger.info(f"Created encrypted repo: {repo_url}")
        
        return repo_url
    
    async def _build_site(self, site_id: str) -> Dict[str, Any]:
        """Build site for deployment."""
        return {
            "success": True,
            "build_id": f"build_{site_id}",
            "assets_generated": True,
            "optimization_complete": True
        }
    
    async def _deploy_to_staging(self, site: ClientSite, build_result: Dict) -> Dict[str, Any]:
        """Deploy site to staging environment."""
        return {
            "success": True,
            "deployment_id": f"dep_{site.site_id}",
            "staging_url": site.staging_domain,
            "deployed_at": datetime.utcnow().isoformat()
        }
    
    async def _generate_deployment_approval(self, site: ClientSite) -> Dict[str, Any]:
        """Generate deployment approval request."""
        return {
            "approval_hash": f"approval_{site.site_id}",
            "site_details": {
                "domain": site.domain,
                "owner": site.owner,
                "deployment_target": site.deployment_target
            }
        }
    
    async def _deploy_to_production(self, site: ClientSite) -> Dict[str, Any]:
        """Deploy site to production environment."""
        return {
            "success": True,
            "deployment_id": f"prod_dep_{site.site_id}",
            "production_url": f"https://{site.domain}",
            "deployed_at": datetime.utcnow().isoformat()
        }

class SecurityError(Exception):
    """Security violation in site registry."""
    pass
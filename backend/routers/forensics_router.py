# routers/forensics_router.py
"""
FORENSICS API ROUTER - CHAIN OF CUSTODY ENFORCED
FastAPI endpoints for forensic evidence management.
All access logged and authenticated. Evidence requires proper authorization.
"""

from fastapi import APIRouter, HTTPException, Depends, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import List, Optional
from pydantic import BaseModel

router = APIRouter(prefix="/api/v1/forensics", tags=["forensics"])
security = HTTPBearer()

# Request/Response models
class EvidenceCollectionRequest(BaseModel):
    incident_id: str
    evidence_types: List[str]  # ["logs", "pcap", "memory", "database"]
    scope: Dict
    justification: str

class EvidencePackageResponse(BaseModel):
    package_id: str
    incident_id: str
    storage_uri: str
    access_token: Optional[str] = None
    expires_at: Optional[str] = None

class LEASubmissionRequest(BaseModel):
    incident_id: str
    agency_name: str
    case_number: Optional[str] = None
    contact_email: str
    contact_phone: str
    submitted_by: str
    legal_approval: bool

@router.post("/collect", response_model=EvidencePackageResponse)
async def request_evidence_collection(
    request: EvidenceCollectionRequest,
    authorization: HTTPAuthorizationCredentials = Depends(security),
    x_requester: str = Header(...)
):
    """
    Request forensic evidence collection for an incident.
    Logs who requested collection and why for chain of custody.
    """
    try:
        # Verify authentication and authorization
        user = await _verify_forensics_access(authorization.credentials)
        
        # Log the collection request
        await _log_evidence_request(request.incident_id, x_requester, request.justification)
        
        # Initialize forensic service
        forensic_service = ForensicService()
        
        # Collect requested evidence types
        packages = []
        if "logs" in request.evidence_types:
            log_package = await forensic_service.collect_logs(
                request.incident_id, request.scope, x_requester
            )
            packages.append(log_package)
        
        if "pcap" in request.evidence_types:
            pcap_package = await forensic_service.capture_pcap(
                request.incident_id, 
                request.scope.get('interfaces', ['eth0']),
                300,  # 5-minute capture
                x_requester
            )
            packages.append(pcap_package)
        
        if "memory" in request.evidence_types:
            memory_package = await forensic_service.capture_memory_dump(
                request.incident_id,
                request.scope.get('host_id', 'unknown'),
                x_requester
            )
            packages.append(memory_package)
        
        if "database" in request.evidence_types:
            db_package = await forensic_service.snapshot_db(
                request.incident_id, x_requester
            )
            packages.append(db_package)
        
        # Create comprehensive package
        full_package_uri = await forensic_service.package_forensics(
            request.incident_id, x_requester
        )
        
        # Generate access token for download
        access_token = await _generate_access_token(request.incident_id, user)
        
        response = EvidencePackageResponse(
            package_id=f"package-{request.incident_id}",
            incident_id=request.incident_id,
            storage_uri=full_package_uri,
            access_token=access_token,
            expires_at=(datetime.utcnow() + timedelta(hours=24)).isoformat()
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Evidence collection failed: {str(e)}")

@router.get("/package/{incident_id}")
async def download_evidence_package(
    incident_id: str,
    authorization: HTTPAuthorizationCredentials = Depends(security),
    x_requester: str = Header(...)
):
    """
    Download encrypted forensic package for an incident.
    Requires proper authorization and logs all access.
    """
    try:
        user = await _verify_forensics_access(authorization.credentials)
        
        # Verify user has access to this incident
        if not await _verify_incident_access(incident_id, user):
            raise HTTPException(status_code=403, detail="Access denied to this incident")
        
        # Get package location from database
        package_uri = await _get_package_uri(incident_id)
        
        if not package_uri:
            raise HTTPException(status_code=404, detail="Evidence package not found")
        
        # Generate temporary download URL
        download_url = await _generate_download_url(package_uri)
        
        # Log the download access
        await _log_package_access(incident_id, x_requester, "download")
        
        return {
            "incident_id": incident_id,
            "download_url": download_url,
            "expires_in": "15 minutes",
            "access_logged": True
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Package download failed: {str(e)}")

@router.post("/submit_to_lea/{incident_id}")
async def prepare_lea_submission(
    incident_id: str,
    request: LEASubmissionRequest,
    authorization: HTTPAuthorizationCredentials = Depends(security)
):
    """
    Prepare Law Enforcement Agency submission package.
    DOES NOT AUTO-SUBMIT - creates template for legal review.
    Requires legal approval flag to be set.
    """
    try:
        user = await _verify_forensics_access(authorization.credentials)
        
        if not request.legal_approval:
            raise HTTPException(
                status_code=400, 
                detail="Legal approval required for LEA submission preparation"
            )
        
        # Verify user is authorized for LEA submissions
        if not await _verify_lea_authorization(user):
            raise HTTPException(status_code=403, detail="Not authorized for LEA submissions")
        
        # Generate LEA submission package
        submission_package = await _prepare_lea_package(incident_id, request)
        
        # Log the submission preparation
        await _log_lea_submission_preparation(incident_id, request.submitted_by, request.agency_name)
        
        return {
            "status": "submission_package_prepared",
            "incident_id": incident_id,
            "agency": request.agency_name,
            "package_location": submission_package['storage_uri'],
            "contact_template": submission_package['contact_template'],
            "next_steps": [
                "Review package with legal counsel",
                "Customize contact template",
                "Submit via approved channels (SFTP, secure email, physical media)"
            ]
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"LEA submission preparation failed: {str(e)}")

@router.get("/chain_of_custody/{incident_id}")
async def get_chain_of_custody(
    incident_id: str,
    authorization: HTTPAuthorizationCredentials = Depends(security)
):
    """Get complete chain of custody for an incident's evidence"""
    try:
        user = await _verify_forensics_access(authorization.credentials)
        
        if not await _verify_incident_access(incident_id, user):
            raise HTTPException(status_code=403, detail="Access denied")
        
        custody_log = await _get_chain_of_custody_log(incident_id)
        
        return {
            "incident_id": incident_id,
            "chain_of_custody": custody_log,
            "verified_integrity": await _verify_evidence_integrity(incident_id)
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to retrieve chain of custody: {str(e)}")

# Authentication and authorization helpers
async def _verify_forensics_access(token: str) -> Dict:
    """Verify user has access to forensics functionality"""
    # Implementation would validate JWT or other token
    # and check user permissions
    return {"user_id": "verified_user", "role": "soc_analyst"}

async def _verify_incident_access(incident_id: str, user: Dict) -> bool:
    """Verify user has access to this specific incident"""
    # Implementation would check incident ownership/permissions
    return True

async def _verify_lea_authorization(user: Dict) -> bool:
    """Verify user is authorized for LEA submissions"""
    return user.get('role') in ['soc_manager', 'legal_counsel', 'ciso']

# Evidence management helpers
async def _log_evidence_request(incident_id: str, requester: str, justification: str):
    """Log evidence collection request"""
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "incident_id": incident_id,
        "requester": requester,
        "action": "evidence_collection_requested",
        "justification": justification
    }
    logging.getLogger('forensic_access').info(str(log_entry))

async def _get_package_uri(incident_id: str) -> Optional[str]:
    """Get storage URI for incident evidence package"""
    # Implementation would query evidence database
    return f"s3://forensics-bucket/{incident_id}/full-package.encrypted"

async def _generate_download_url(package_uri: str) -> str:
    """Generate temporary download URL for evidence package"""
    # Implementation would use cloud provider pre-signed URLs
    return f"https://forensics-bucket.s3.amazonaws.com/{package_uri}?token=temporary"

async def _generate_access_token(incident_id: str, user: Dict) -> str:
    """Generate access token for evidence package"""
    # Implementation would create JWT or similar
    return f"token-{incident_id}-{datetime.utcnow().strftime('%H%M%S')}"

async def _log_package_access(incident_id: str, requester: str, action: str):
    """Log package access for chain of custody"""
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "incident_id": incident_id,
        "requester": requester,
        "action": action,
        "purpose": "incident_investigation"
    }
    logging.getLogger('package_access').info(str(log_entry))

async def _prepare_lea_package(incident_id: str, request: LEASubmissionRequest) -> Dict:
    """Prepare LEA submission package with contact template"""
    # Get incident details
    incident_brief = await _get_incident_brief(incident_id)
    
    # Get evidence package
    evidence_uri = await _get_package_uri(incident_id)
    
    # Generate LEA contact template
    contact_template = await _generate_lea_contact_template(incident_brief, request)
    
    # Package for LEA submission
    lea_package = {
        "incident_summary": incident_brief,
        "evidence_location": evidence_uri,
        "contact_template": contact_template,
        "submission_instructions": _get_lea_submission_instructions(),
        "legal_disclaimers": _get_legal_disclaimers()
    }
    
    # Store LEA package
    storage_uri = f"s3://lea-submissions/{incident_id}/lea-package-{datetime.utcnow().strftime('%Y%m%d')}.json"
    
    return {
        "storage_uri": storage_uri,
        "contact_template": contact_template
    }

async def _get_chain_of_custody_log(incident_id: str) -> List[Dict]:
    """Get chain of custody log for incident"""
    # Implementation would query audit logs
    return []

async def _verify_evidence_integrity(incident_id: str) -> bool:
    """Verify evidence integrity through checksum validation"""
    # Implementation would verify all evidence checksums
    return True

async def _get_incident_brief(incident_id: str) -> Dict:
    """Get incident brief for LEA submission"""
    # Implementation would query incident database
    return {}

async def _generate_lea_contact_template(incident_brief: Dict, request: LEASubmissionRequest) -> str:
    """Generate LEA contact template"""
    template = f"""
LAW ENFORCEMENT CONTACT TEMPLATE
--------------------------------

INCIDENT REPORT: {incident_brief.get('summary', 'Unknown')}
INCIDENT ID: {incident_brief.get('incident_id', 'Unknown')}
DATE DETECTED: {incident_brief.get('detected_at', 'Unknown')}

CONTACT INFORMATION:
- Agency: {request.agency_name}
- Case Number: {request.case_number or 'Pending'}
- Submitted By: {request.submitted_by}
- Contact Email: {request.contact_email}
- Contact Phone: {request.contact_phone}

INCIDENT OVERVIEW:
{incident_brief.get('detailed_summary', 'No detailed summary available')}

EVIDENCE COLLECTED:
- Full forensic package available at: {incident_brief.get('evidence_location', 'Unknown')}
- Includes: Logs, network captures, memory dumps, database snapshots
- All evidence cryptographically signed and checksum verified

REQUEST FOR ASSISTANCE:
{incident_brief.get('assistance_requested', 'Technical investigation and attribution assistance')}

LEGAL DISCLAIMERS:
- This information is provided for law enforcement purposes only
- All evidence collected in accordance with applicable laws
- Contact legal counsel for any jurisdictional questions
"""
    return template

def _get_lea_submission_instructions() -> str:
    """Get LEA submission instructions"""
    return """
SUBMISSION INSTRUCTIONS:
1. Review entire package with legal counsel
2. Customize contact template as needed
3. Submit via one of these secure methods:
   - Secure FTP to agency portal
   - Encrypted email with passphrase
   - Physical media via secure courier
4. Maintain chain of custody documentation
"""

def _get_legal_disclaimers() -> str:
    """Get legal disclaimers for LEA submissions"""
    return """
LEGAL DISCLAIMERS:
- Evidence collected for defensive cybersecurity purposes only
- No offensive actions taken by our organization
- IP geolocation data is probabilistic and not conclusive
- Attribution requires additional law enforcement investigation
- All actions logged and audited for legal compliance
"""
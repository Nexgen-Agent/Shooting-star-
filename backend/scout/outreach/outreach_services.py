# scout/outreach/outreach_service.py
import asyncio
from typing import List, Dict, Any
from datetime import datetime, timedelta
from ..models.candidate import CandidateProfile

class OutreachService:
    def __init__(self):
        self.rate_limits = {
            "email": 100,  # per hour
            "dm": 50,      # per hour
        }
        self.sent_messages = []
        
    async def send_initial_contact(self, candidate: CandidateProfile, template: str = "default") -> bool:
        """Send initial contact using Cheese Method"""
        if not candidate.contact_consent:
            return False
            
        message = self._create_cheese_message(candidate, template)
        
        # Check rate limits
        if not self._check_rate_limit("email"):
            await asyncio.sleep(3600)  # Wait an hour
            
        # Send message (implementation depends on channel)
        success = await self._dispatch_message(candidate, message)
        
        if success:
            self._log_outreach(candidate.id, message)
            
        return success
    
    def _create_cheese_message(self, candidate: CandidateProfile, template: str) -> str:
        """Create personalized message using Cheese Method"""
        templates = {
            "default": """Hey {name} — I loved your work on GitHub. You've got an uncommon eye for {skills}. We're building something that could use your exact energy. Interested in a quick 15-minute call to explore compensated work?""",
            "technical": """Hi {name} — Your contributions to {repo} show impressive technical depth. We're tackling similar challenges at ShootingStar and think you'd be perfect for our mission. Open to chatting about paid opportunities?"""
        }
        
        template_str = templates.get(template, templates["default"])
        return template_str.format(
            name=candidate.name or candidate.github_username,
            skills=', '.join(candidate.skills[:3]),
            repo=candidate.github_username
        )
    
    def _check_rate_limit(self, channel: str) -> bool:
        """Check if we're within rate limits"""
        hour_ago = datetime.utcnow() - timedelta(hours=1)
        recent_count = len([m for m in self.sent_messages 
                          if m['channel'] == channel and m['timestamp'] > hour_ago])
        return recent_count < self.rate_limits[channel]
    
    async def _dispatch_message(self, candidate: CandidateProfile, message: str) -> bool:
        """Actually send message via appropriate channel"""
        # Implementation for email/DM APIs
        return True
        
    def _log_outreach(self, candidate_id: str, message: str) -> None:
        """Log outreach attempt"""
        self.sent_messages.append({
            'candidate_id': candidate_id,
            'message': message,
            'channel': 'email',
            'timestamp': datetime.utcnow()
        })
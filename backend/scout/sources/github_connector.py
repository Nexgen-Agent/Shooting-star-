# scout/sources/github_connector.py
import aiohttp
import asyncio
from typing import List, Dict, Any, Optional
from ..models.candidate import CandidateProfile

class GitHubConnector:
    def __init__(self):
        self.base_url = "https://api.github.com"
        self.rate_limit_remaining = 5000
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def search_developers(self, skills: List[str]) -> List[CandidateProfile]:
        """Search GitHub for developers with specific skills"""
        candidates = []
        
        for skill in skills:
            users = await self._search_users(f"language:{skill}")
            for user in users[:10]:  # Limit per skill
                candidate = await self._user_to_candidate(user)
                if candidate:
                    candidates.append(candidate)
            
            await asyncio.sleep(0.1)  # Rate limiting
            
        return candidates
    
    async def _search_users(self, query: str) -> List[Dict[str, Any]]:
        """Search GitHub users with rate limit handling"""
        if not self.session:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/search/users",
                    params={"q": query, "per_page": 10}
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get('items', [])
        return []
    
    async def _user_to_candidate(self, user_data: Dict[str, Any]) -> Optional[CandidateProfile]:
        """Convert GitHub user to CandidateProfile"""
        try:
            return CandidateProfile(
                id=f"github_{user_data['id']}",
                source="github",
                name=user_data.get('login', ''),
                github_username=user_data.get('login'),
                portfolio_urls=[user_data.get('blog')] if user_data.get('blog') else []
            )
        except Exception as e:
            print(f"Error converting user: {e}")
            return None
# scout/enrichment/enricher.py
import asyncio
from typing import Dict, List, Optional, Tuple
import aiohttp
from datetime import datetime, timedelta
from ..models.candidate import CandidateProfile

class CandidateEnricher:
    def __init__(self):
        self.metrics_cache = {}
        
    async def enrich_candidate(self, candidate: CandidateProfile) -> Dict[str, float]:
        """Enrich candidate with detailed metrics from their profiles"""
        enrichment_data = {}
        
        if candidate.github_username:
            github_metrics = await self._get_github_metrics(candidate.github_username)
            enrichment_data.update(github_metrics)
            
        if candidate.portfolio_urls:
            portfolio_metrics = await self._analyze_portfolio(candidate.portfolio_urls)
            enrichment_data.update(portfolio_metrics)
            
        # Calculate scores based on enriched data
        scores = self._calculate_enrichment_scores(enrichment_data)
        candidate.technical_score = scores['technical']
        candidate.portfolio_score = scores['portfolio']
        
        return enrichment_data
    
    async def _get_github_metrics(self, username: str) -> Dict[str, any]:
        """Get detailed GitHub metrics for a user"""
        metrics = {
            'public_repos': 0,
            'total_stars': 0,
            'total_forks': 0,
            'contributions_last_year': 0,
            'repo_quality_score': 0.0,
            'followers': 0,
            'account_age_days': 0
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                # Get user basic info
                async with session.get(f"https://api.github.com/users/{username}") as resp:
                    if resp.status == 200:
                        user_data = await resp.json()
                        metrics.update({
                            'public_repos': user_data.get('public_repos', 0),
                            'followers': user_data.get('followers', 0),
                            'account_age_days': self._calculate_account_age(user_data.get('created_at'))
                        })
                
                # Get repository analysis
                repo_metrics = await self._analyze_repositories(username, session)
                metrics.update(repo_metrics)
                
        except Exception as e:
            print(f"Error fetching GitHub metrics for {username}: {e}")
            
        return metrics
    
    async def _analyze_repositories(self, username: str, session: aiohttp.ClientSession) -> Dict[str, any]:
        """Analyze user's repositories for quality signals"""
        repos = []
        async with session.get(f"https://api.github.com/users/{username}/repos?per_page=100") as resp:
            if resp.status == 200:
                repos = await resp.json()
        
        total_stars = sum(repo.get('stargazers_count', 0) for repo in repos)
        total_forks = sum(repo.get('forks_count', 0) for repo in repos)
        
        # Analyze individual repos for quality signals
        quality_signals = 0
        total_repos = len(repos)
        
        for repo in repos[:10]:  # Sample first 10 repos
            repo_quality = await self._analyze_single_repo(repo['name'], username, session)
            if repo_quality:
                quality_signals += repo_quality
        
        repo_quality_score = quality_signals / max(total_repos, 1)
        
        return {
            'total_stars': total_stars,
            'total_forks': total_forks,
            'repo_quality_score': repo_quality_score
        }
    
    async def _analyze_single_repo(self, repo_name: str, owner: str, session: aiohttp.ClientSession) -> float:
        """Analyze a single repository for quality indicators"""
        quality_score = 0.0
        
        try:
            # Check for README
            async with session.get(f"https://api.github.com/repos/{owner}/{repo_name}/readme") as resp:
                if resp.status == 200:
                    quality_score += 0.2
            
            # Check for CI configuration
            async with session.get(f"https://api.github.com/repos/{owner}/{repo_name}/contents/.github/workflows") as resp:
                if resp.status == 200:
                    quality_score += 0.3
            
            # Check for tests directory
            async with session.get(f"https://api.github.com/repos/{owner}/{repo_name}/contents/tests") as resp:
                if resp.status == 200:
                    quality_score += 0.3
                    
            # Check for recent activity
            async with session.get(f"https://api.github.com/repos/{owner}/{repo_name}/commits") as resp:
                if resp.status == 200:
                    commits = await resp.json()
                    if commits and len(commits) > 0:
                        quality_score += 0.2
                        
        except Exception:
            pass
            
        return quality_score
    
    def _calculate_enrichment_scores(self, metrics: Dict[str, any]) -> Dict[str, float]:
        """Calculate scores based on enriched metrics"""
        technical_score = min(
            (metrics.get('public_repos', 0) * 0.1 +
             metrics.get('repo_quality_score', 0) * 0.4 +
             metrics.get('total_stars', 0) * 0.01 +
             metrics.get('contributions_last_year', 0) * 0.05), 1.0
        )
        
        portfolio_score = min(
            metrics.get('repo_quality_score', 0) * 0.7 +
            (1 if metrics.get('has_live_demo', False) else 0) * 0.3, 1.0
        )
        
        return {
            'technical': technical_score,
            'portfolio': portfolio_score
        }
    
    def _calculate_account_age(self, created_at: str) -> int:
        """Calculate account age in days"""
        if not created_at:
            return 0
        created_date = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
        return (datetime.utcnow() - created_date).days
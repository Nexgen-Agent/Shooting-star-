# extensions/vbe/lead_hunter.py
"""
24/7 Lead Hunting System
Continuously discovers and qualifies potential leads from multiple sources
"""
import asyncio
import random
from typing import List, Dict, Optional
from datetime import datetime
import logging

logger = logging.getLogger("vbe.lead_hunter")


async def hunt_once() -> List[dict]:
    """
    Execute single lead hunting cycle
    
    Returns:
        List[dict]: List of discovered leads with profiles
        
    Example:
        >>> leads = await hunt_once()
        >>> len(leads) >= 0
        True
    """
    logger.info("Starting lead hunting cycle")
    
    # Mock connectors - replace with real API integrations
    connectors = [
        _mock_linkedin_connector,
        _mock_twitter_connector, 
        _mock_github_connector,
        _mock_product_hunt_connector
    ]
    
    all_leads = []
    
    for connector in connectors:
        try:
            leads = await connector()
            all_leads.extend(leads)
            logger.debug(f"Connector found {len(leads)} leads")
        except Exception as e:
            logger.warning(f"Connector failed: {e}")
    
    # Score and filter leads
    scored_leads = _score_leads(all_leads)
    qualified_leads = [lead for lead in scored_leads if lead["relevance_score"] >= 0.6]
    
    logger.info(f"Hunting cycle complete: {len(qualified_leads)} qualified leads found")
    return qualified_leads


async def continuous_hunt(stop_event: asyncio.Event):
    """
    Continuous lead hunting loop (24/7 operation)
    
    Args:
        stop_event: Event to signal when to stop hunting
    """
    logger.info("Starting continuous lead hunting")
    
    while not stop_event.is_set():
        try:
            leads = await hunt_once()
            
            if leads:
                logger.info(f"Discovered {len(leads)} new leads")
                # TODO: Automatically create outreach drafts for high-quality leads
                # await _create_auto_drafts(leads)
            else:
                logger.debug("No new leads discovered this cycle")
                
            # Wait between hunting cycles (configurable)
            await asyncio.sleep(300)  # 5 minutes
            
        except Exception as e:
            logger.error(f"Lead hunting error: {e}")
            await asyncio.sleep(60)  # Shorter delay on error
    
    logger.info("Lead hunting stopped")


async def _mock_linkedin_connector() -> List[dict]:
    """Mock LinkedIn API connector"""
    await asyncio.sleep(0.1)  # Simulate API call
    
    leads = [
        {
            "name": "Alex Johnson",
            "org": "ScaleUp Inc",
            "title": "CEO",
            "profile_url": "https://linkedin.com/in/alexjohnson",
            "source": "linkedin",
            "recent_activity": "Posted about company growth",
            "relevance_score": random.uniform(0.7, 0.9)
        },
        {
            "name": "Maria Garcia", 
            "org": "TechInnovate",
            "title": "Marketing Director",
            "profile_url": "https://linkedin.com/in/mariagarcia",
            "source": "linkedin", 
            "recent_activity": "Shared industry insights",
            "relevance_score": random.uniform(0.6, 0.8)
        }
    ]
    
    return leads


async def _mock_twitter_connector() -> List[dict]:
    """Mock Twitter API connector"""
    await asyncio.sleep(0.1)
    
    leads = [
        {
            "name": "David Chen",
            "org": "StartupGrid",
            "title": "Founder",
            "profile_url": "https://twitter.com/davidchen",
            "source": "twitter",
            "recent_activity": "Tweeted about product launch",
            "relevance_score": random.uniform(0.5, 0.85)
        }
    ]
    
    return leads


async def _mock_github_connector() -> List[dict]:
    """Mock GitHub API connector"""
    await asyncio.sleep(0.1)
    
    leads = [
        {
            "name": "Sarah Williams",
            "org": "DevOps Pro",
            "title": "CTO", 
            "profile_url": "https://github.com/sarahw",
            "source": "github",
            "recent_activity": "Created new repository",
            "relevance_score": random.uniform(0.4, 0.7)
        }
    ]
    
    return leads


async def _mock_product_hunt_connector() -> List[dict]:
    """Mock Product Hunt API connector"""
    await asyncio.sleep(0.1)
    
    leads = [
        {
            "name": "Mike Rodriguez",
            "org": "ProductLabs",
            "title": "Product Manager",
            "profile_url": "https://www.producthunt.com/@miker",
            "source": "product_hunt", 
            "recent_activity": "Launched new product",
            "relevance_score": random.uniform(0.8, 0.95)
        }
    ]
    
    return leads


def _score_leads(leads: List[dict]) -> List[dict]:
    """
    Score leads based on relevance criteria
    
    Args:
        leads: Raw discovered leads
        
    Returns:
        List[dict]: Leads with relevance scores
    """
    for lead in leads:
        # Base score from connector
        base_score = lead.get("relevance_score", 0.5)
        
        # Boost for certain titles
        title_boost = 0.0
        title = lead.get("title", "").lower()
        if any(role in title for role in ["ceo", "founder", "owner", "director"]):
            title_boost = 0.2
            
        # Boost for recent activity
        activity_boost = 0.1 if lead.get("recent_activity") else 0.0
        
        lead["relevance_score"] = min(1.0, base_score + title_boost + activity_boost)
    
    return sorted(leads, key=lambda x: x["relevance_score"], reverse=True)


if __name__ == "__main__":
    # Debug harness
    async def test_hunt():
        leads = await hunt_once()
        print("=== LEAD HUNTING RESULTS ===")
        for lead in leads[:3]:  # Show top 3
            print(f"Name: {lead['name']}")
            print(f"Org: {lead['org']} | Title: {lead['title']}")
            print(f"Score: {lead['relevance_score']:.2f}")
            print(f"Source: {lead['source']}")
            print("---")
    
    asyncio.run(test_hunt())
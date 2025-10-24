# services/ioc_enrich.py
"""
IOC ENRICHMENT WITH LEGAL CAVEATS
Indicator of Compromise enrichment service with attribution warnings.
Geolocation data is probabilistic only - never conclusive for physical location.
"""

import asyncio
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
import aiohttp
import maxminddb

class IPEnrichment(BaseModel):
    ip: str
    reverse_dns: Optional[str] = None
    asn: Optional[str] = None
    asn_org: Optional[str] = None
    country: Optional[str] = None
    city: Optional[str] = None
    isp: Optional[str] = None
    confidence: float = Field(..., ge=0, le=1)  # Confidence score 0-1
    caveats: List[str] = Field(default_factory=list)
    threat_feed_matches: List[str] = Field(default_factory=list)

class HashEnrichment(BaseModel):
    hash: str
    hash_type: str  # md5, sha1, sha256
    threat_feed_matches: List[str] = Field(default_factory=list)
    file_type: Optional[str] = None
    reputation: Optional[str] = None  # malicious/suspicious/clean/unknown

class IOCEnrichmentService:
    def __init__(self):
        self.geoip_reader = maxminddb.open_database('/var/lib/GeoIP/GeoLite2-City.mmdb')
        self.threat_feeds = [
            "https://otx.alienvault.com/api/v1/indicators/",
            # Add other threat intelligence feeds
        ]
        
    async def enrich_ip(self, ip: str) -> IPEnrichment:
        """
        Enrich IP address with threat intelligence and geolocation.
        LEGAL CAVEAT: IP geolocation is probabilistic and not conclusive for physical location.
        """
        enrichment = IPEnrichment(ip=ip, confidence=0.5)
        enrichment.caveats = [
            "IP geolocation is approximate and based on database records",
            "IP address may be using VPN, proxy, or Tor network",
            "Cannot definitively determine physical location from IP alone",
            "Attribution based solely on IP address is unreliable"
        ]
        
        try:
            # Reverse DNS lookup
            enrichment.reverse_dns = await self._reverse_dns_lookup(ip)
            
            # ASN information
            asn_info = await self._get_asn_info(ip)
            enrichment.asn = asn_info.get('asn')
            enrichment.asn_org = asn_info.get('org')
            
            # Geolocation (with low confidence for accuracy)
            geo_info = await self._get_geo_info(ip)
            enrichment.country = geo_info.get('country')
            enrichment.city = geo_info.get('city')
            enrichment.isp = geo_info.get('isp')
            
            # Adjust confidence based on data quality
            confidence_factors = []
            if enrichment.country:
                confidence_factors.append(0.3)
            if enrichment.asn_org:
                confidence_factors.append(0.2)
            if enrichment.reverse_dns:
                confidence_factors.append(0.1)
                
            enrichment.confidence = min(0.8, sum(confidence_factors))  # Cap at 0.8
            
            # Threat intelligence lookup
            threat_matches = await self._check_threat_feeds(ip, "ip")
            enrichment.threat_feed_matches = threat_matches
            
            if threat_matches:
                enrichment.confidence = max(enrichment.confidence, 0.7)
                
        except Exception as e:
            enrichment.caveats.append(f"Enrichment incomplete: {str(e)}")
            
        return enrichment
    
    async def enrich_hash(self, file_hash: str, hash_type: str = "sha256") -> HashEnrichment:
        """Enrich file hash with threat intelligence"""
        enrichment = HashEnrichment(hash=file_hash, hash_type=hash_type)
        
        try:
            # Determine file type if possible
            enrichment.file_type = await self._identify_file_type(file_hash, hash_type)
            
            # Check threat feeds
            threat_matches = await self._check_threat_feeds(file_hash, "hash")
            enrichment.threat_feed_matches = threat_matches
            
            # Set reputation based on threat feeds
            if threat_matches:
                enrichment.reputation = "malicious"
            else:
                enrichment.reputation = "unknown"
                
        except Exception as e:
            # Continue with basic information if enrichment fails
            pass
            
        return enrichment
    
    async def _reverse_dns_lookup(self, ip: str) -> Optional[str]:
        """Perform reverse DNS lookup"""
        try:
            import socket
            return socket.gethostbyaddr(ip)[0]
        except:
            return None
    
    async def _get_asn_info(self, ip: str) -> Dict:
        """Get ASN information for IP"""
        # Implementation would use MaxMind ASN or similar
        return {
            "asn": "AS15169",
            "org": "Google LLC"
        }
    
    async def _get_geo_info(self, ip: str) -> Dict:
        """Get geolocation information with legal caveats"""
        try:
            geo_data = self.geoip_reader.get(ip)
            return {
                "country": geo_data.get('country', {}).get('names', {}).get('en'),
                "city": geo_data.get('city', {}).get('names', {}).get('en'),
                "isp": "Unknown"  # Would need additional data source
            }
        except:
            return {}
    
    async def _check_threat_feeds(self, indicator: str, indicator_type: str) -> List[str]:
        """Check threat intelligence feeds"""
        matches = []
        
        for feed_url in self.threat_feeds:
            try:
                async with aiohttp.ClientSession() as session:
                    url = f"{feed_url}{indicator_type}/{indicator}"
                    async with session.get(url) as response:
                        if response.status == 200:
                            data = await response.json()
                            if self._is_threat_match(data):
                                matches.append(feed_url)
            except Exception as e:
                # Continue with other feeds if one fails
                continue
                
        return matches
    
    async def _identify_file_type(self, file_hash: str, hash_type: str) -> Optional[str]:
        """Identify file type from hash (if available in threat intelligence)"""
        # Implementation would query threat intelligence or internal databases
        return None
    
    def _is_threat_match(self, feed_data: Dict) -> bool:
        """Determine if threat feed data indicates malicious activity"""
        # Implementation would vary by threat feed API
        return feed_data.get('pulse_info', {}).get('count', 0) > 0

# Legal compliance warnings for all enrichment outputs
ENRICHMENT_DISCLAIMER = """
LEGAL DISCLAIMER: 
- IP geolocation data is probabilistic and approximate
- Cannot definitively determine physical location from IP address alone
- Threat intelligence data may contain false positives
- Attribution requires additional evidence beyond network indicators
- All data should be reviewed by legal counsel before external disclosure
"""
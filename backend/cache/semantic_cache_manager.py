"""
Advanced semantic caching system using vector embeddings for intelligent query matching.
Enables fast retrieval of semantically similar results without recomputation.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pydantic import BaseModel, Field
import logging
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json
import pickle
from sqlalchemy.ext.asyncio import AsyncSession

from core.system_logs import SystemLogs
from extensions.ai_v16.ai_governance_engine import AIGovernanceEngine

logger = logging.getLogger(__name__)

class CacheEntry(BaseModel):
    key: str
    vector_embedding: List[float]
    data: Any
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    ttl: Optional[timedelta] = None
    semantic_group: Optional[str] = None

class SemanticCacheConfig(BaseModel):
    max_size: int = 10000
    similarity_threshold: float = 0.85
    default_ttl: timedelta = timedelta(hours=24)
    eviction_policy: str = "lru"  # lru, lfu, semantic_clustering
    embedding_dimension: int = 384  # Typical for sentence transformers

class SemanticCacheManager:
    def __init__(self, config: SemanticCacheConfig = None):
        self.system_logs = SystemLogs()
        self.governance = AIGovernanceEngine()
        self.config = config or SemanticCacheConfig()
        self.cache_entries: Dict[str, CacheEntry] = {}
        self.vector_index: Dict[str, List[float]] = {}
        self.semantic_groups: Dict[str, List[str]] = {}
        self.model_version = "v2.3"
        
    async def get_semantic_match(self, 
                               query: str, 
                               query_vector: List[float],
                               context: Optional[Dict] = None) -> Optional[Any]:
        """Retrieve semantically similar cached result"""
        try:
            # Find similar entries using vector similarity
            similar_entries = await self._find_similar_entries(query_vector, context)
            
            if not similar_entries:
                return None
            
            best_match = similar_entries[0]
            
            if best_match['similarity'] >= self.config.similarity_threshold:
                # Update access statistics
                await self._update_access_stats(best_match['entry'].key)
                
                await self.system_logs.log_ai_activity(
                    module="semantic_cache_manager",
                    activity_type="cache_hit",
                    details={
                        "query": query[:100],  # Log first 100 chars
                        "similarity_score": best_match['similarity'],
                        "cache_key": best_match['entry'].key,
                        "semantic_group": best_match['entry'].semantic_group
                    }
                )
                
                return best_match['entry'].data
            else:
                await self.system_logs.log_ai_activity(
                    module="semantic_cache_manager",
                    activity_type="cache_miss",
                    details={
                        "query": query[:100],
                        "best_similarity": best_match['similarity'],
                        "threshold": self.config.similarity_threshold
                    }
                )
                return None
                
        except Exception as e:
            logger.error(f"Semantic cache lookup error: {str(e)}")
            await self.system_logs.log_error(
                module="semantic_cache_manager",
                error_type="lookup_failed",
                details={"query": query[:100], "error": str(e)}
            )
            return None
    
    async def store_semantic_result(self, 
                                  key: str,
                                  query: str,
                                  query_vector: List[float],
                                  data: Any,
                                  metadata: Optional[Dict] = None,
                                  ttl: Optional[timedelta] = None) -> bool:
        """Store result in semantic cache with vector embedding"""
        try:
            # Check cache size and evict if necessary
            await self._enforce_cache_limits()
            
            # Determine semantic group
            semantic_group = await self._determine_semantic_group(query_vector, query)
            
            cache_entry = CacheEntry(
                key=key,
                vector_embedding=query_vector,
                data=data,
                metadata=metadata or {},
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                access_count=1,
                ttl=ttl or self.config.default_ttl,
                semantic_group=semantic_group
            )
            
            # Store in cache
            self.cache_entries[key] = cache_entry
            self.vector_index[key] = query_vector
            
            # Update semantic groups
            if semantic_group not in self.semantic_groups:
                self.semantic_groups[semantic_group] = []
            self.semantic_groups[semantic_group].append(key)
            
            await self.system_logs.log_ai_activity(
                module="semantic_cache_manager",
                activity_type="cache_stored",
                details={
                    "key": key,
                    "semantic_group": semantic_group,
                    "vector_dimension": len(query_vector),
                    "ttl_seconds": cache_entry.ttl.total_seconds() if cache_entry.ttl else None
                }
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Semantic cache store error: {str(e)}")
            await self.system_logs.log_error(
                module="semantic_cache_manager",
                error_type="store_failed",
                details={"key": key, "error": str(e)}
            )
            return False
    
    async def _find_similar_entries(self, 
                                  query_vector: List[float], 
                                  context: Optional[Dict]) -> List[Dict[str, Any]]:
        """Find semantically similar cache entries"""
        similar_entries = []
        
        # Filter by context if provided
        candidate_keys = await self._filter_by_context(context) if context else list(self.vector_index.keys())
        
        if not candidate_keys:
            return []
        
        # Convert to numpy arrays for efficient computation
        query_vec = np.array(query_vector).reshape(1, -1)
        candidate_vectors = []
        valid_keys = []
        
        for key in candidate_keys:
            if key in self.vector_index and not await self._is_expired(key):
                candidate_vectors.append(self.vector_index[key])
                valid_keys.append(key)
        
        if not candidate_vectors:
            return []
        
        candidate_matrix = np.array(candidate_vectors)
        
        # Calculate cosine similarities
        similarities = cosine_similarity(query_vec, candidate_matrix)[0]
        
        # Create results with similarity scores
        for key, similarity in zip(valid_keys, similarities):
            entry = self.cache_entries[key]
            similar_entries.append({
                'entry': entry,
                'similarity': similarity,
                'key': key
            })
        
        # Sort by similarity (descending)
        similar_entries.sort(key=lambda x: x['similarity'], reverse=True)
        
        return similar_entries
    
    async def _determine_semantic_group(self, vector: List[float], query: str) -> str:
        """Determine semantic group for the query"""
        # Simple implementation - can be enhanced with clustering
        # For now, use the first few words or a hash of the vector
        
        if len(self.semantic_groups) == 0:
            return "group_0"
        
        # Find the most similar existing group
        best_group = None
        best_similarity = -1
        
        for group_name, group_keys in self.semantic_groups.items():
            if not group_keys:
                continue
                
            # Calculate average vector for the group
            group_vectors = [self.vector_index[key] for key in group_keys[:10]]  # Sample first 10
            avg_group_vector = np.mean(group_vectors, axis=0)
            
            similarity = cosine_similarity(
                np.array(vector).reshape(1, -1),
                avg_group_vector.reshape(1, -1)
            )[0][0]
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_group = group_name
        
        # Create new group if no good match found
        if best_similarity < 0.7 or best_group is None:
            new_group = f"group_{len(self.semantic_groups)}"
            return new_group
        
        return best_group
    
    async def _filter_by_context(self, context: Dict) -> List[str]:
        """Filter cache keys by context parameters"""
        filtered_keys = []
        
        for key, entry in self.cache_entries.items():
            if await self._matches_context(entry, context) and not await self._is_expired(key):
                filtered_keys.append(key)
        
        return filtered_keys
    
    async def _matches_context(self, entry: CacheEntry, context: Dict) -> bool:
        """Check if cache entry matches context"""
        entry_context = entry.metadata.get('context', {})
        
        for context_key, context_value in context.items():
            if context_key in entry_context:
                if entry_context[context_key] != context_value:
                    return False
            else:
                # If context key not in entry, it doesn't match
                return False
        
        return True
    
    async def _is_expired(self, key: str) -> bool:
        """Check if cache entry has expired"""
        if key not in self.cache_entries:
            return True
        
        entry = self.cache_entries[key]
        
        if entry.ttl is None:
            return False
        
        expiry_time = entry.created_at + entry.ttl
        return datetime.now() > expiry_time
    
    async def _update_access_stats(self, key: str):
        """Update access statistics for cache entry"""
        if key in self.cache_entries:
            entry = self.cache_entries[key]
            entry.last_accessed = datetime.now()
            entry.access_count += 1
    
    async def _enforce_cache_limits(self):
        """Enforce cache size limits using eviction policy"""
        if len(self.cache_entries) < self.config.max_size:
            return
        
        entries_to_evict = await self._select_entries_for_eviction()
        
        for key in entries_to_evict:
            await self._evict_entry(key)
    
    async def _select_entries_for_eviction(self) -> List[str]:
        """Select entries to evict based on eviction policy"""
        if self.config.eviction_policy == "lru":
            return await self._select_lru_entries()
        elif self.config.eviction_policy == "lfu":
            return await self._select_lfu_entries()
        elif self.config.eviction_policy == "semantic_clustering":
            return await self._select_semantic_entries()
        else:
            return await self._select_lru_entries()  # Default
    
    async def _select_lru_entries(self) -> List[str]:
        """Select least recently used entries"""
        sorted_entries = sorted(
            self.cache_entries.items(),
            key=lambda x: x[1].last_accessed
        )
        
        evict_count = len(self.cache_entries) - self.config.max_size + int(self.config.max_size * 0.1)
        return [key for key, _ in sorted_entries[:evict_count]]
    
    async def _select_lfu_entries(self) -> List[str]:
        """Select least frequently used entries"""
        sorted_entries = sorted(
            self.cache_entries.items(),
            key=lambda x: x[1].access_count
        )
        
        evict_count = len(self.cache_entries) - self.config.max_size + int(self.config.max_size * 0.1)
        return [key for key, _ in sorted_entries[:evict_count]]
    
    async def _select_semantic_entries(self) -> List[str]:
        """Select entries from largest semantic groups (spread risk)"""
        # Group entries by semantic group and select from largest groups
        group_sizes = {group: len(keys) for group, keys in self.semantic_groups.items()}
        sorted_groups = sorted(group_sizes.items(), key=lambda x: x[1], reverse=True)
        
        entries_to_evict = []
        evict_count = len(self.cache_entries) - self.config.max_size + int(self.config.max_size * 0.1)
        
        for group, size in sorted_groups:
            if len(entries_to_evict) >= evict_count:
                break
            
            group_entries = self.semantic_groups[group]
            # Take some entries from this group
            take_count = min(len(group_entries), max(1, evict_count // len(sorted_groups)))
            entries_to_evict.extend(group_entries[:take_count])
        
        return entries_to_evict[:evict_count]
    
    async def _evict_entry(self, key: str):
        """Evict a cache entry"""
        if key in self.cache_entries:
            entry = self.cache_entries[key]
            
            # Remove from semantic group
            if entry.semantic_group and entry.semantic_group in self.semantic_groups:
                if key in self.semantic_groups[entry.semantic_group]:
                    self.semantic_groups[entry.semantic_group].remove(key)
            
            # Remove from cache and index
            del self.cache_entries[key]
            if key in self.vector_index:
                del self.vector_index[key]
            
            await self.system_logs.log_ai_activity(
                module="semantic_cache_manager",
                activity_type="cache_evicted",
                details={"key": key, "reason": "cache_size_limit"}
            )
    
    async def clear_expired_entries(self) -> int:
        """Clear all expired cache entries"""
        expired_keys = []
        
        for key in list(self.cache_entries.keys()):
            if await self._is_expired(key):
                expired_keys.append(key)
        
        for key in expired_keys:
            await self._evict_entry(key)
        
        await self.system_logs.log_ai_activity(
            module="semantic_cache_manager",
            activity_type="expired_cleared",
            details={"count": len(expired_keys)}
        )
        
        return len(expired_keys)
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics and health metrics"""
        total_size = len(self.cache_entries)
        expired_count = len([key for key in self.cache_entries if await self._is_expired(key)])
        
        # Calculate hit rate (would need tracking)
        hit_rate = 0.0  # This would be calculated from actual usage tracking
        
        # Group statistics
        group_stats = {}
        for group, keys in self.semantic_groups.items():
            group_stats[group] = {
                "entry_count": len(keys),
                "avg_access_count": np.mean([self.cache_entries[key].access_count for key in keys]) if keys else 0
            }
        
        return {
            "total_entries": total_size,
            "expired_entries": expired_count,
            "cache_size_bytes": await self._calculate_cache_size(),
            "semantic_groups_count": len(self.semantic_groups),
            "group_statistics": group_stats,
            "hit_rate": hit_rate,
            "config": self.config.dict()
        }
    
    async def _calculate_cache_size(self) -> int:
        """Calculate approximate cache size in bytes"""
        total_size = 0
        
        for entry in self.cache_entries.values():
            # Approximate size calculation
            entry_size = len(pickle.dumps(entry.data)) if entry.data else 0
            vector_size = len(entry.vector_embedding) * 8  # 8 bytes per float
            total_size += entry_size + vector_size
        
        return total_size
    
    async def export_cache(self, file_path: str) -> bool:
        """Export cache to file for persistence"""
        try:
            cache_data = {
                'entries': {k: v.dict() for k, v in self.cache_entries.items()},
                'vector_index': self.vector_index,
                'semantic_groups': self.semantic_groups,
                'exported_at': datetime.now().isoformat()
            }
            
            with open(file_path, 'wb') as f:
                pickle.dump(cache_data, f)
            
            await self.system_logs.log_ai_activity(
                module="semantic_cache_manager",
                activity_type="cache_exported",
                details={"file_path": file_path, "entry_count": len(self.cache_entries)}
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Cache export error: {str(e)}")
            await self.system_logs.log_error(
                module="semantic_cache_manager",
                error_type="export_failed",
                details={"file_path": file_path, "error": str(e)}
            )
            return False
    
    async def import_cache(self, file_path: str) -> bool:
        """Import cache from file"""
        try:
            with open(file_path, 'rb') as f:
                cache_data = pickle.load(f)
            
            # Clear current cache
            self.cache_entries.clear()
            self.vector_index.clear()
            self.semantic_groups.clear()
            
            # Import entries
            for key, entry_data in cache_data['entries'].items():
                # Convert dict back to CacheEntry
                entry_data['created_at'] = datetime.fromisoformat(entry_data['created_at'])
                entry_data['last_accessed'] = datetime.fromisoformat(entry_data['last_accessed'])
                if entry_data['ttl']:
                    entry_data['ttl'] = timedelta(seconds=entry_data['ttl'])
                
                self.cache_entries[key] = CacheEntry(**entry_data)
            
            # Import index and groups
            self.vector_index = cache_data['vector_index']
            self.semantic_groups = cache_data['semantic_groups']
            
            await self.system_logs.log_ai_activity(
                module="semantic_cache_manager",
                activity_type="cache_imported",
                details={"file_path": file_path, "entry_count": len(self.cache_entries)}
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Cache import error: {str(e)}")
            await self.system_logs.log_error(
                module="semantic_cache_manager",
                error_type="import_failed",
                details={"file_path": file_path, "error": str(e)}
            )
            return False
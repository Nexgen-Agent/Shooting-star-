import asyncio
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import time
import hashlib
import json
import pickle
from collections import OrderedDict
import redis.asyncio as redis
from pydantic import BaseModel

@dataclass
class CacheEntry:
    vector: np.ndarray
    metadata: Dict[str, Any]
    created_at: float
    last_accessed: float
    access_count: int
    size: int

class SimilarityResult(BaseModel):
    key: str
    similarity: float
    metadata: Dict[str, Any]

class HighPerformanceVectorEmbeddingCache:
    """
    High-performance cache for vector embeddings with similarity search
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379", 
                 max_memory_mb: int = 1024, 
                 default_ttl: int = 3600):
        self.redis_url = redis_url
        self.redis_client = None
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.default_ttl = default_ttl
        
        # Local in-memory cache (LRU)
        self.local_cache = OrderedDict()
        self.local_cache_size = 0
        self.max_local_entries = 10000
        
        # Index for similarity search
        self.vector_index = {}  # key -> vector
        self.dimension = None
        
        # Metrics
        self.metrics = {
            "hits": 0,
            "misses": 0,
            "local_hits": 0,
            "redis_hits": 0,
            "evictions": 0,
            "total_size": 0
        }
        
        self.logger = logging.getLogger("VectorEmbeddingCache")
    
    async def initialize(self):
        """Initialize the cache"""
        self.redis_client = await redis.from_url(
            self.redis_url, 
            encoding="utf-8", 
            decode_responses=False
        )
        
        # Test connection
        await self.redis_client.ping()
        self.logger.info("Vector Embedding Cache initialized")
    
    def _generate_key(self, text: str, model: str = "default") -> str:
        """Generate cache key from text and model"""
        content = f"{model}:{text}"
        return f"embedding:{hashlib.md5(content.encode()).hexdigest()}"
    
    async def store_embedding(self, text: str, vector: np.ndarray, 
                            model: str = "default", 
                            metadata: Dict[str, Any] = None,
                            ttl: int = None) -> str:
        """Store vector embedding in cache"""
        key = self._generate_key(text, model)
        
        # Ensure vector is numpy array
        if not isinstance(vector, np.ndarray):
            vector = np.array(vector)
        
        # Set dimension if not set
        if self.dimension is None:
            self.dimension = vector.shape[0]
        
        entry = CacheEntry(
            vector=vector.copy(),
            metadata=metadata or {},
            created_at=time.time(),
            last_accessed=time.time(),
            access_count=1,
            size=vector.nbytes
        )
        
        # Store in local cache
        await self._store_local(key, entry)
        
        # Store in Redis
        await self._store_redis(key, entry, ttl)
        
        # Update index
        self.vector_index[key] = vector
        
        self.metrics["total_size"] += entry.size
        
        return key
    
    async def get_embedding(self, text: str, model: str = "default") -> Optional[CacheEntry]:
        """Retrieve vector embedding from cache"""
        key = self._generate_key(text, model)
        
        # Try local cache first
        entry = await self._get_local(key)
        if entry:
            self.metrics["hits"] += 1
            self.metrics["local_hits"] += 1
            return entry
        
        # Try Redis
        entry = await self._get_redis(key)
        if entry:
            self.metrics["hits"] += 1
            self.metrics["redis_hits"] += 1
            
            # Store in local cache for future access
            await self._store_local(key, entry)
            
            return entry
        
        self.metrics["misses"] += 1
        return None
    
    async def _store_local(self, key: str, entry: CacheEntry):
        """Store entry in local LRU cache"""
        if key in self.local_cache:
            # Update existing entry
            old_entry = self.local_cache[key]
            self.local_cache_size -= old_entry.size
            del self.local_cache[key]
        
        # Check if we need to evict
        while (self.local_cache_size + entry.size > self.max_memory_bytes and 
               len(self.local_cache) > 0):
            await self._evict_oldest_local()
        
        # Store new entry
        self.local_cache[key] = entry
        self.local_cache_size += entry.size
        
        # Move to end (most recently used)
        self.local_cache.move_to_end(key)
    
    async def _get_local(self, key: str) -> Optional[CacheEntry]:
        """Get entry from local cache"""
        if key in self.local_cache:
            entry = self.local_cache[key]
            entry.last_accessed = time.time()
            entry.access_count += 1
            
            # Move to end (most recently used)
            self.local_cache.move_to_end(key)
            
            return entry
        
        return None
    
    async def _store_redis(self, key: str, entry: CacheEntry, ttl: int = None):
        """Store entry in Redis"""
        try:
            # Serialize entry
            serialized = {
                'vector': pickle.dumps(entry.vector),
                'metadata': json.dumps(entry.metadata),
                'created_at': entry.created_at,
                'last_accessed': entry.last_accessed,
                'access_count': entry.access_count,
                'size': entry.size
            }
            
            redis_ttl = ttl or self.default_ttl
            await self.redis_client.hset(key, mapping=serialized)
            await self.redis_client.expire(key, redis_ttl)
            
        except Exception as e:
            self.logger.error(f"Redis store error for key {key}: {e}")
    
    async def _get_redis(self, key: str) -> Optional[CacheEntry]:
        """Get entry from Redis"""
        try:
            data = await self.redis_client.hgetall(key)
            if not data:
                return None
            
            # Deserialize entry
            vector = pickle.loads(data[b'vector'])
            metadata = json.loads(data[b'metadata'])
            
            entry = CacheEntry(
                vector=vector,
                metadata=metadata,
                created_at=float(data[b'created_at']),
                last_accessed=float(data[b'last_accessed']),
                access_count=int(data[b'access_count']),
                size=int(data[b'size'])
            )
            
            return entry
            
        except Exception as e:
            self.logger.error(f"Redis get error for key {key}: {e}")
            return None
    
    async def _evict_oldest_local(self):
        """Evict oldest entry from local cache"""
        if not self.local_cache:
            return
        
        key, entry = self.local_cache.popitem(last=False)
        self.local_cache_size -= entry.size
        self.metrics["evictions"] += 1
        
        self.logger.debug(f"Evicted key {key} from local cache")
    
    async def find_similar_vectors(self, query_vector: np.ndarray, 
                                 top_k: int = 10, 
                                 min_similarity: float = 0.7) -> List[SimilarityResult]:
        """Find similar vectors using approximate nearest neighbor search"""
        if not self.vector_index:
            return []
        
        similarities = []
        query_norm = np.linalg.norm(query_vector)
        
        for key, vector in self.vector_index.items():
            # Cosine similarity
            dot_product = np.dot(query_vector, vector)
            vector_norm = np.linalg.norm(vector)
            
            if query_norm > 0 and vector_norm > 0:
                similarity = dot_product / (query_norm * vector_norm)
                
                if similarity >= min_similarity:
                    # Get metadata for the entry
                    entry = await self._get_local(key) or await self._get_redis(key)
                    metadata = entry.metadata if entry else {}
                    
                    similarities.append(SimilarityResult(
                        key=key,
                        similarity=similarity,
                        metadata=metadata
                    ))
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x.similarity, reverse=True)
        return similarities[:top_k]
    
    async def batch_store_embeddings(self, texts: List[str], 
                                   vectors: List[np.ndarray],
                                   model: str = "default",
                                   metadata_list: List[Dict[str, Any]] = None):
        """Store multiple embeddings in batch"""
        if metadata_list is None:
            metadata_list = [{}] * len(texts)
        
        tasks = []
        for text, vector, metadata in zip(texts, vectors, metadata_list):
            task = self.store_embedding(text, vector, model, metadata)
            tasks.append(task)
        
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def batch_get_embeddings(self, texts: List[str], 
                                 model: str = "default") -> List[Optional[CacheEntry]]:
        """Get multiple embeddings in batch"""
        tasks = []
        for text in texts:
            task = self.get_embedding(text, model)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to None
        final_results = []
        for result in results:
            if isinstance(result, Exception):
                final_results.append(None)
            else:
                final_results.append(result)
        
        return final_results
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_entries = len(self.local_cache)
        
        hit_rate = (self.metrics["hits"] / 
                   (self.metrics["hits"] + self.metrics["misses"]) 
                   if (self.metrics["hits"] + self.metrics["misses"]) > 0 else 0)
        
        local_hit_rate = (self.metrics["local_hits"] / self.metrics["hits"] 
                         if self.metrics["hits"] > 0 else 0)
        
        return {
            "local_cache_entries": total_entries,
            "local_cache_size_mb": self.local_cache_size / (1024 * 1024),
            "vector_index_entries": len(self.vector_index),
            "hit_rate": hit_rate,
            "local_hit_rate": local_hit_rate,
            "total_hits": self.metrics["hits"],
            "total_misses": self.metrics["misses"],
            "evictions": self.metrics["evictions"],
            "memory_usage_percent": (self.local_cache_size / self.max_memory_bytes) * 100
        }
    
    async def clear_cache(self, pattern: str = None):
        """Clear cache entries (optionally by pattern)"""
        if pattern:
            # Clear matching keys from Redis
            keys = await self.redis_client.keys(pattern)
            if keys:
                await self.redis_client.delete(*keys)
            
            # Clear matching keys from local cache
            keys_to_remove = [k for k in self.local_cache.keys() if pattern in k]
            for key in keys_to_remove:
                entry = self.local_cache[key]
                self.local_cache_size -= entry.size
                del self.local_cache[key]
                del self.vector_index[key]
        else:
            # Clear everything
            self.local_cache.clear()
            self.local_cache_size = 0
            self.vector_index.clear()
            await self.redis_client.flushdb()
        
        self.logger.info(f"Cache cleared for pattern: {pattern or 'all'}")
    
    async def cleanup_expired(self):
        """Clean up expired entries from vector index"""
        current_time = time.time()
        expired_keys = []
        
        for key in list(self.vector_index.keys()):
            # Check if key exists in Redis (if not, it's expired)
            exists = await self.redis_client.exists(key)
            if not exists:
                expired_keys.append(key)
        
        # Remove expired keys
        for key in expired_keys:
            del self.vector_index[key]
            if key in self.local_cache:
                entry = self.local_cache[key]
                self.local_cache_size -= entry.size
                del self.local_cache[key]
        
        if expired_keys:
            self.logger.info(f"Cleaned up {len(expired_keys)} expired entries")
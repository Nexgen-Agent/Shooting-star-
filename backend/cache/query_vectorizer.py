"""
Advanced query vectorization using transformer models for semantic understanding.
Converts text queries into high-dimensional vector embeddings for semantic matching.
"""

import asyncio
from typing import Dict, List, Optional, Any, Tuple
from pydantic import BaseModel, Field
import logging
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoTokenizer, AutoModel
import hashlib

from core.system_logs import SystemLogs
from extensions.ai_v16.ai_governance_engine import AIGovernanceEngine

logger = logging.getLogger(__name__)

class VectorizationResult(BaseModel):
    vector: List[float]
    model_name: str
    dimension: int
    processing_time: float
    token_count: int
    hash: str

class VectorizerConfig(BaseModel):
    default_model: str = "all-MiniLM-L6-v2"  # Lightweight and effective
    backup_models: List[str] = Field(default_factory=list)
    max_sequence_length: int = 512
    normalize_embeddings: bool = True
    use_gpu: bool = True
    cache_embeddings: bool = True
    batch_size: int = 32

class QueryVectorizer:
    def __init__(self, config: VectorizerConfig = None):
        self.system_logs = SystemLogs()
        self.governance = AIGovernanceEngine()
        self.config = config or VectorizerConfig()
        self.models: Dict[str, Any] = {}
        self.tokenizers: Dict[str, Any] = {}
        self.embedding_cache: Dict[str, VectorizationResult] = {}
        self.model_version = "v3.2"
        
        # Initialize device
        self.device = torch.device("cuda" if torch.cuda.is_available() and self.config.use_gpu else "cpu")
        
    async def initialize_models(self):
        """Initialize vectorization models"""
        try:
            # Load default model
            await self._load_model(self.config.default_model)
            
            # Load backup models
            for model_name in self.config.backup_models:
                await self._load_model(model_name)
            
            await self.system_logs.log_ai_activity(
                module="query_vectorizer",
                activity_type="models_initialized",
                details={
                    "default_model": self.config.default_model,
                    "backup_models": self.config.backup_models,
                    "device": str(self.device),
                    "available_models": list(self.models.keys())
                }
            )
            
        except Exception as e:
            logger.error(f"Model initialization error: {str(e)}")
            await self.system_logs.log_error(
                module="query_vectorizer",
                error_type="initialization_failed",
                details={"error": str(e)}
            )
            raise
    
    async def _load_model(self, model_name: str):
        """Load a specific model"""
        try:
            if model_name.startswith("sentence-transformers/"):
                # Use sentence-transformers
                model = SentenceTransformer(model_name, device=str(self.device))
                self.models[model_name] = model
            else:
                # Use transformers with mean pooling
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModel.from_pretrained(model_name).to(self.device)
                
                self.models[model_name] = model
                self.tokenizers[model_name] = tokenizer
            
            logger.info(f"Loaded model: {model_name} on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {str(e)}")
            raise
    
    async def vectorize_query(self, 
                            query: str, 
                            model_name: Optional[str] = None,
                            context: Optional[Dict] = None) -> VectorizationResult:
        """Convert query text to vector embedding"""
        try:
            start_time = asyncio.get_event_loop().time()
            
            # Use specified model or default
            model_to_use = model_name or self.config.default_model
            
            if model_to_use not in self.models:
                await self._load_model(model_to_use)
            
            # Check cache first
            cache_key = await self._generate_cache_key(query, model_to_use, context)
            if self.config.cache_embeddings and cache_key in self.embedding_cache:
                cached_result = self.embedding_cache[cache_key]
                cached_result.processing_time = 0.001  # Near-zero for cache hits
                return cached_result
            
            # Preprocess query
            processed_query = await self._preprocess_query(query, context)
            
            # Generate embedding
            if model_to_use.startswith("sentence-transformers/"):
                vector = await self._vectorize_sentence_transformer(processed_query, model_to_use)
            else:
                vector = await self._vectorize_transformers(processed_query, model_to_use)
            
            processing_time = asyncio.get_event_loop().time() - start_time
            
            result = VectorizationResult(
                vector=vector.tolist() if hasattr(vector, 'tolist') else vector,
                model_name=model_to_use,
                dimension=len(vector),
                processing_time=processing_time,
                token_count=await self._count_tokens(processed_query, model_to_use),
                hash=cache_key
            )
            
            # Cache the result
            if self.config.cache_embeddings:
                self.embedding_cache[cache_key] = result
                await self._manage_cache_size()
            
            await self.system_logs.log_ai_activity(
                module="query_vectorizer",
                activity_type="query_vectorized",
                details={
                    "query_length": len(query),
                    "model_used": model_to_use,
                    "vector_dimension": result.dimension,
                    "processing_time": processing_time,
                    "cache_hit": False
                }
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Query vectorization error: {str(e)}")
            
            # Try fallback model
            if model_name != self.config.default_model:
                logger.info(f"Trying fallback model: {self.config.default_model}")
                return await self.vectorize_query(query, self.config.default_model, context)
            
            await self.system_logs.log_error(
                module="query_vectorizer",
                error_type="vectorization_failed",
                details={
                    "query": query[:100],
                    "model": model_name,
                    "error": str(e)
                }
            )
            raise
    
    async def vectorize_batch(self, 
                            queries: List[str], 
                            model_name: Optional[str] = None,
                            context: Optional[Dict] = None) -> List[VectorizationResult]:
        """Vectorize multiple queries in batch for efficiency"""
        try:
            model_to_use = model_name or self.config.default_model
            
            if model_to_use not in self.models:
                await self._load_model(model_to_use)
            
            # Process in batches
            batch_size = self.config.batch_size
            results = []
            
            for i in range(0, len(queries), batch_size):
                batch_queries = queries[i:i + batch_size]
                batch_results = await self._process_batch(batch_queries, model_to_use, context)
                results.extend(batch_results)
            
            await self.system_logs.log_ai_activity(
                module="query_vectorizer",
                activity_type="batch_vectorized",
                details={
                    "batch_size": len(queries),
                    "model_used": model_to_use,
                    "total_queries": len(queries)
                }
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Batch vectorization error: {str(e)}")
            await self.system_logs.log_error(
                module="query_vectorizer",
                error_type="batch_vectorization_failed",
                details={"batch_size": len(queries), "error": str(e)}
            )
            raise
    
    async def _process_batch(self, queries: List[str], model_name: str, context: Optional[Dict]) -> List[VectorizationResult]:
        """Process a batch of queries"""
        start_time = asyncio.get_event_loop().time()
        
        # Preprocess all queries
        processed_queries = [await self._preprocess_query(q, context) for q in queries]
        
        # Generate embeddings
        if model_name.startswith("sentence-transformers/"):
            embeddings = await self._vectorize_batch_sentence_transformer(processed_queries, model_name)
        else:
            embeddings = await self._vectorize_batch_transformers(processed_queries, model_name)
        
        processing_time = asyncio.get_event_loop().time() - start_time
        
        results = []
        for i, (query, embedding) in enumerate(zip(queries, embeddings)):
            cache_key = await self._generate_cache_key(query, model_name, context)
            
            result = VectorizationResult(
                vector=embedding.tolist() if hasattr(embedding, 'tolist') else embedding,
                model_name=model_name,
                dimension=len(embedding),
                processing_time=processing_time / len(queries),  # Average time per query
                token_count=await self._count_tokens(processed_queries[i], model_name),
                hash=cache_key
            )
            
            # Cache the result
            if self.config.cache_embeddings:
                self.embedding_cache[cache_key] = result
            
            results.append(result)
        
        await self._manage_cache_size()
        return results
    
    async def _vectorize_sentence_transformer(self, query: str, model_name: str) -> np.ndarray:
        """Vectorize using sentence-transformers"""
        model = self.models[model_name]
        
        # Sentence-transformers handle tokenization internally
        embedding = model.encode([query], convert_to_numpy=True, normalize_embeddings=self.config.normalize_embeddings)
        
        return embedding[0]  # Return first (and only) embedding
    
    async def _vectorize_transformers(self, query: str, model_name: str) -> np.ndarray:
        """Vectorize using transformers with mean pooling"""
        model = self.models[model_name]
        tokenizer = self.tokenizers[model_name]
        
        # Tokenize
        inputs = tokenizer(
            query,
            padding=True,
            truncation=True,
            max_length=self.config.max_sequence_length,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate embeddings
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state
        
        # Mean pooling
        attention_mask = inputs['attention_mask']
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
        sum_embeddings = torch.sum(embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        
        # Normalize if requested
        if self.config.normalize_embeddings:
            mean_embeddings = torch.nn.functional.normalize(mean_embeddings, p=2, dim=1)
        
        return mean_embeddings[0].cpu().numpy()
    
    async def _vectorize_batch_sentence_transformer(self, queries: List[str], model_name: str) -> np.ndarray:
        """Vectorize batch using sentence-transformers"""
        model = self.models[model_name]
        embeddings = model.encode(queries, convert_to_numpy=True, normalize_embeddings=self.config.normalize_embeddings)
        return embeddings
    
    async def _vectorize_batch_transformers(self, queries: List[str], model_name: str) -> np.ndarray:
        """Vectorize batch using transformers"""
        model = self.models[model_name]
        tokenizer = self.tokenizers[model_name]
        
        # Tokenize batch
        inputs = tokenizer(
            queries,
            padding=True,
            truncation=True,
            max_length=self.config.max_sequence_length,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate embeddings
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state
        
        # Mean pooling
        attention_mask = inputs['attention_mask']
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
        sum_embeddings = torch.sum(embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        
        # Normalize if requested
        if self.config.normalize_embeddings:
            mean_embeddings = torch.nn.functional.normalize(mean_embeddings, p=2, dim=1)
        
        return mean_embeddings.cpu().numpy()
    
    async def _preprocess_query(self, query: str, context: Optional[Dict]) -> str:
        """Preprocess query text before vectorization"""
        # Basic cleaning
        processed = query.strip()
        
        # Add context if provided
        if context:
            context_str = " ".join([f"{k}: {v}" for k, v in context.items()])
            processed = f"{processed} [CONTEXT: {context_str}]"
        
        # Truncate if too long (rough estimate)
        if len(processed) > self.config.max_sequence_length * 4:  # Approximate character to token ratio
            processed = processed[:self.config.max_sequence_length * 4]
        
        return processed
    
    async def _count_tokens(self, query: str, model_name: str) -> int:
        """Count tokens in query"""
        try:
            if model_name.startswith("sentence-transformers/"):
                # Sentence-transformers don't expose token count easily
                # Rough estimate: 4 characters per token
                return len(query) // 4
            else:
                tokenizer = self.tokenizers[model_name]
                tokens = tokenizer.encode(query, add_special_tokens=False)
                return len(tokens)
        except Exception:
            return len(query) // 4  # Fallback estimate
    
    async def _generate_cache_key(self, query: str, model_name: str, context: Optional[Dict]) -> str:
        """Generate cache key for query"""
        key_data = f"{query}_{model_name}"
        if context:
            key_data += f"_{hash(frozenset(context.items()))}"
        
        return hashlib.md5(key_data.encode()).hexdigest()
    
    async def _manage_cache_size(self):
        """Manage embedding cache size"""
        max_cache_size = 10000  # Maximum cache entries
        
        if len(self.embedding_cache) > max_cache_size:
            # Remove oldest entries (simple LRU)
            keys_to_remove = list(self.embedding_cache.keys())[:len(self.embedding_cache) - max_cache_size]
            for key in keys_to_remove:
                del self.embedding_cache[key]
    
    async def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a specific model"""
        if model_name not in self.models:
            return {"error": "Model not loaded"}
        
        model = self.models[model_name]
        
        info = {
            "model_name": model_name,
            "device": str(self.device),
            "vector_dimension": await self._get_model_dimension(model_name),
            "type": "sentence-transformers" if model_name.startswith("sentence-transformers/") else "transformers"
        }
        
        return info
    
    async def _get_model_dimension(self, model_name: str) -> int:
        """Get the output dimension of a model"""
        if model_name.startswith("sentence-transformers/"):
            model = self.models[model_name]
            return model.get_sentence_embedding_dimension()
        else:
            # For transformers, we need to check the model config
            model = self.models[model_name]
            return model.config.hidden_size
    
    async def clear_cache(self):
        """Clear the embedding cache"""
        cache_size = len(self.embedding_cache)
        self.embedding_cache.clear()
        
        await self.system_logs.log_ai_activity(
            module="query_vectorizer",
            activity_type="cache_cleared",
            details={"cleared_entries": cache_size}
        )
    
    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get vectorization performance statistics"""
        cache_hit_rate = 0.0  # Would need tracking
        
        model_stats = {}
        for model_name in self.models:
            model_stats[model_name] = {
                "loaded": True,
                "dimension": await self._get_model_dimension(model_name),
                "type": "sentence-transformers" if model_name.startswith("sentence-transformers/") else "transformers"
            }
        
        return {
            "cache_size": len(self.embedding_cache),
            "cache_hit_rate": cache_hit_rate,
            "loaded_models": list(self.models.keys()),
            "model_statistics": model_stats,
            "device": str(self.device),
            "config": self.config.dict()
        }
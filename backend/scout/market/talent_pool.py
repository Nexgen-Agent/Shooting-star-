# scout/market/talent_pool.py
import numpy as np
from typing import List, Dict, Optional, Any
from datetime import datetime
import json
from ..models.candidate import CandidateProfile

class TalentPool:
    def __init__(self, vector_dim: int = 512):
        self.vector_dim = vector_dim
        self.candidates = {}
        self.skill_vectors = {}
        self.skill_index = {}
        
    def add_candidate(self, candidate: CandidateProfile) -> str:
        """Add candidate to talent pool with vector embedding"""
        candidate_id = candidate.id
        
        # Generate skill vector
        skill_vector = self._generate_skill_vector(candidate.skills)
        
        # Store candidate data
        self.candidates[candidate_id] = {
            'profile': candidate,
            'skill_vector': skill_vector,
            'last_updated': datetime.utcnow(),
            'tags': self._generate_tags(candidate),
            'availability': True,
            'match_score': 0.0
        }
        
        # Update skill index
        self._update_skill_index(candidate_id, candidate.skills, skill_vector)
        
        return candidate_id
    
    def _generate_skill_vector(self, skills: List[str]) -> np.ndarray:
        """Generate vector embedding for skills"""
        # Simple one-hot encoding for demonstration
        # In production, use sentence transformers or similar
        vector = np.zeros(self.vector_dim)
        
        for i, skill in enumerate(skills):
            # Simple hash-based positioning
            hash_val = hash(skill) % (self.vector_dim - 10)
            vector[hash_val:hash_val + 10] += 1.0 / len(skills)
            
        return vector / np.linalg.norm(vector) if np.linalg.norm(vector) > 0 else vector
    
    def _generate_tags(self, candidate: CandidateProfile) -> List[str]:
        """Generate search tags for candidate"""
        tags = []
        
        # Skill-based tags
        tags.extend([f"skill:{skill}" for skill in candidate.skills])
        
        # Score-based tags
        if candidate.technical_score > 0.8:
            tags.append("level:senior")
        elif candidate.technical_score > 0.6:
            tags.append("level:mid")
        else:
            tags.append("level:junior")
            
        # Location tags
        if candidate.location:
            tags.append(f"location:{candidate.location.lower()}")
            
        # Availability tags
        tags.append("availability:active")
        
        return tags
    
    def search_candidates(self, 
                         required_skills: List[str],
                         min_score: float = 0.0,
                         location: Optional[str] = None,
                         limit: int = 20) -> List[Dict[str, Any]]:
        """Search for candidates matching criteria"""
        query_vector = self._generate_skill_vector(required_skills)
        results = []
        
        for candidate_id, candidate_data in self.candidates.items():
            profile = candidate_data['profile']
            
            # Apply filters
            if profile.overall_score < min_score:
                continue
                
            if location and profile.location and location.lower() not in profile.location.lower():
                continue
                
            # Calculate similarity
            similarity = self._cosine_similarity(query_vector, candidate_data['skill_vector'])
            candidate_data['match_score'] = similarity
            
            if similarity > 0.1:  # Minimum similarity threshold
                results.append({
                    'candidate_id': candidate_id,
                    'profile': profile,
                    'match_score': similarity,
                    'tags': candidate_data['tags']
                })
        
        # Sort by match score and return top results
        results.sort(key=lambda x: x['match_score'], reverse=True)
        return results[:limit]
    
    def find_similar_candidates(self, candidate_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Find candidates similar to given candidate"""
        if candidate_id not in self.candidates:
            return []
            
        target_vector = self.candidates[candidate_id]['skill_vector']
        results = []
        
        for other_id, candidate_data in self.candidates.items():
            if other_id == candidate_id:
                continue
                
            similarity = self._cosine_similarity(target_vector, candidate_data['skill_vector'])
            
            if similarity > 0.3:  # Higher threshold for similarity
                results.append({
                    'candidate_id': other_id,
                    'profile': candidate_data['profile'],
                    'similarity_score': similarity
                })
        
        results.sort(key=lambda x: x['similarity_score'], reverse=True)
        return results[:limit]
    
    def get_candidate_recommendations(self, project_requirements: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get candidate recommendations for project requirements"""
        required_skills = project_requirements.get('required_skills', [])
        preferred_skills = project_requirements.get('preferred_skills', [])
        role_level = project_requirements.get('role_level', 'mid')
        
        # Search for candidates with required skills
        base_results = self.search_candidates(required_skills, min_score=0.5)
        
        # Score candidates based on additional criteria
        for candidate in base_results:
            candidate['recommendation_score'] = self._calculate_recommendation_score(
                candidate, preferred_skills, role_level
            )
        
        # Sort by recommendation score
        base_results.sort(key=lambda x: x['recommendation_score'], reverse=True)
        return base_results[:10]
    
    def _calculate_recommendation_score(self, 
                                      candidate: Dict[str, Any], 
                                      preferred_skills: List[str],
                                      role_level: str) -> float:
        """Calculate recommendation score for candidate"""
        base_score = candidate['match_score']
        profile = candidate['profile']
        
        # Bonus for preferred skills
        preferred_skill_bonus = 0.0
        for skill in preferred_skills:
            if skill in profile.skills:
                preferred_skill_bonus += 0.1
                
        # Role level matching
        level_score = 1.0
        if role_level == "senior" and profile.technical_score < 0.7:
            level_score = 0.5
        elif role_level == "junior" and profile.technical_score > 0.8:
            level_score = 0.8
            
        # Communication bonus
        comm_bonus = profile.communication_score * 0.1
        
        return base_score * level_score + preferred_skill_bonus + comm_bonus
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return dot_product / (norm1 * norm2)
    
    def _update_skill_index(self, candidate_id: str, skills: List[str], vector: np.ndarray) -> None:
        """Update skill index for faster searching"""
        for skill in skills:
            if skill not in self.skill_index:
                self.skill_index[skill] = []
            self.skill_index[skill].append(candidate_id)
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """Get talent pool statistics"""
        total_candidates = len(self.candidates)
        
        skill_counts = {}
        location_counts = {}
        score_distribution = {
            'excellent': 0,  # > 0.8
            'good': 0,       # 0.6 - 0.8
            'average': 0,    # 0.4 - 0.6
            'poor': 0        # < 0.4
        }
        
        for candidate_data in self.candidates.values():
            profile = candidate_data['profile']
            
            # Count skills
            for skill in profile.skills:
                skill_counts[skill] = skill_counts.get(skill, 0) + 1
                
            # Count locations
            if profile.location:
                location_counts[profile.location] = location_counts.get(profile.location, 0) + 1
                
            # Score distribution
            if profile.overall_score > 0.8:
                score_distribution['excellent'] += 1
            elif profile.overall_score > 0.6:
                score_distribution['good'] += 1
            elif profile.overall_score > 0.4:
                score_distribution['average'] += 1
            else:
                score_distribution['poor'] += 1
        
        return {
            'total_candidates': total_candidates,
            'top_skills': dict(sorted(skill_counts.items(), key=lambda x: x[1], reverse=True)[:10]),
            'locations': location_counts,
            'score_distribution': score_distribution,
            'last_updated': datetime.utcnow()
        }
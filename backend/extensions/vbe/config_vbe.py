# extensions/vbe/config_vbe.py
"""
VBE Configuration Settings
Centralized configuration management for Virtual Business Engine
"""
import os
from typing import List, Optional
from functools import lru_cache

from pydantic import BaseSettings, Field


class VbeSettings(BaseSettings):
    """VBE configuration settings using Pydantic BaseSettings"""
    
    VBE_MODEL_DIR: str = Field(
        default="/tmp/vbe_models",
        description="Directory for VBE model files and data"
    )
    
    VBE_STREAM_BROKER: str = Field(
        default="redis://localhost:6379/0",
        description="Stream broker URL for real-time messaging"
    )
    
    VBE_CACHE_URL: str = Field(
        default="redis://localhost:6379/1", 
        description="Cache URL for temporary data storage"
    )
    
    VBE_APPROVAL_REQUIRED: bool = Field(
        default=True,
        description="Whether admin approval is required before sending outreach"
    )
    
    VBE_ADMIN_USER_IDS: List[str] = Field(
        default=["admin"],
        description="List of user IDs with admin privileges"
    )
    
    class Config:
        env_prefix = "VBE_"
        case_sensitive = False


@lru_cache()
def get_vbe_settings() -> VbeSettings:
    """
    Get cached VBE settings instance
    
    Returns:
        VbeSettings: Configuration instance
        
    Example:
        >>> settings = get_vbe_settings()
        >>> settings.VBE_APPROVAL_REQUIRED
        True
    """
    return VbeSettings()


if __name__ == "__main__":
    # Debug harness
    settings = get_vbe_settings()
    print("VBE Configuration:")
    for key, value in settings.dict().items():
        print(f"  {key}: {value}")
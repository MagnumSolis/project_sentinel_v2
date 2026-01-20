"""
Project Sentinel V2 - Configuration Management

Pydantic Settings for type-safe configuration with environment variable support.
All settings can be overridden via .env file or environment variables.
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional
from pathlib import Path
import os


class Settings(BaseSettings):
    """
    Centralized configuration for Project Sentinel V2.
    
    All settings are loaded from environment variables or .env file.
    Environment variables take precedence over .env file values.
    """
    
    # =========================================================================
    # QDRANT CONFIGURATION
    # =========================================================================
    QDRANT_HOST: str = Field(
        default="localhost",
        description="Qdrant server hostname"
    )
    QDRANT_PORT: int = Field(
        default=6333,
        description="Qdrant HTTP API port"
    )
    QDRANT_API_KEY: Optional[str] = Field(
        default=None,
        description="Optional API key for Qdrant authentication"
    )
    
    # =========================================================================
    # EMBEDDING MODELS
    # =========================================================================
    TEXT_EMBEDDING_MODEL: str = Field(
        default="BAAI/bge-small-en-v1.5",
        description="FastEmbed model for text embeddings (384 dimensions)"
    )
    VISION_EMBEDDING_MODEL: str = Field(
        default="Qdrant/clip-ViT-B-32-vision",
        description="FastEmbed model for image embeddings (512 dimensions)"
    )
    AUDIO_EMBEDDING_MODEL: str = Field(
        default="BAAI/bge-small-en-v1.5",
        description="Model for audio transcript embeddings"
    )
    
    # =========================================================================
    # PERPLEXITY CONFIGURATION
    # =========================================================================
    PERPLEXITY_API_KEY: Optional[str] = Field(
        default=None,
        description="Perplexity API key for LLM features"
    )
    PERPLEXITY_MODEL: str = Field(
        default="sonar-pro",
        description="Perplexity model for text generation (sonar-pro, sonar, etc.)"
    )
    OFFLINE_MODE: bool = Field(
        default=True,
        description="If True, skip Perplexity API calls and use local templates"
    )
    
    # =========================================================================
    # WHISPER CONFIGURATION
    # =========================================================================
    WHISPER_MODEL: str = Field(
        default="base",
        description="Whisper model size: tiny, base, small, medium, large"
    )
    
    # =========================================================================
    # SEARCH PARAMETERS
    # =========================================================================
    SEARCH_LIMIT: int = Field(
        default=5,
        description="Maximum number of results per collection"
    )
    SEARCH_SIMILARITY_THRESHOLD: float = Field(
        default=0.6,
        description="Minimum similarity score for results (0.0-1.0)"
    )
    TIME_DECAY_FACTOR: float = Field(
        default=0.95,
        description="Score multiplier per day old (for recency bias)"
    )
    
    # =========================================================================
    # DATA PATHS
    # =========================================================================
    DATA_DIR: str = Field(
        default="data",
        description="Root data directory"
    )
    RAW_DATASETS_DIR: str = Field(
        default="data/raw_datasets",
        description="Directory for downloaded datasets"
    )
    PROCESSED_DIR: str = Field(
        default="data/processed",
        description="Directory for processed files"
    )
    QDRANT_STORAGE: str = Field(
        default="data/qdrant_storage",
        description="Qdrant persistent storage path"
    )
    
    # =========================================================================
    # LOGGING
    # =========================================================================
    LOG_LEVEL: str = Field(
        default="INFO",
        description="Logging level: DEBUG, INFO, WARNING, ERROR"
    )
    
    # =========================================================================
    # COLLECTION NAMES (Internal)
    # =========================================================================
    COLLECTION_SEMANTIC: str = "sentinel_semantic"
    COLLECTION_EPISODIC: str = "sentinel_episodic"
    COLLECTION_AUDIO: str = "sentinel_audio"
    
    # =========================================================================
    # VECTOR DIMENSIONS (Internal)
    # =========================================================================
    TEXT_VECTOR_SIZE: int = 384
    VISION_VECTOR_SIZE: int = 512  # clip-ViT-B-32-vision produces 512-dim vectors
    AUDIO_VECTOR_SIZE: int = 384
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "ignore"
    
    @property
    def data_path(self) -> Path:
        """Get absolute path to data directory."""
        return Path(self.DATA_DIR).resolve()
    
    @property
    def raw_datasets_path(self) -> Path:
        """Get absolute path to raw datasets directory."""
        return Path(self.RAW_DATASETS_DIR).resolve()
    
    @property
    def processed_path(self) -> Path:
        """Get absolute path to processed files directory."""
        return Path(self.PROCESSED_DIR).resolve()
    
    @property
    def qdrant_url(self) -> str:
        """Get Qdrant connection URL."""
        return f"http://{self.QDRANT_HOST}:{self.QDRANT_PORT}"
    
    @property
    def perplexity_available(self) -> bool:
        """Check if Perplexity API is available."""
        return bool(self.PERPLEXITY_API_KEY) and not self.OFFLINE_MODE
    
    def ensure_directories(self) -> None:
        """Create all required data directories if they don't exist."""
        for path in [self.data_path, self.raw_datasets_path, self.processed_path]:
            path.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()

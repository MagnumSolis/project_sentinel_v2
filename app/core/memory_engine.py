"""
Project Sentinel V2 - Memory Engine

Four-tier memory architecture inspired by hippocampal memory encoding.
Manages Qdrant collections for semantic, episodic, audio, and video data.
"""

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    QuantizationConfig,
    BinaryQuantization,
    BinaryQuantizationConfig,
    PointStruct,
    CollectionInfo,
)
from typing import Dict, List, Optional, Any
import logging
import asyncio
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CollectionConfig:
    """Configuration for a Qdrant collection."""
    name: str
    vector_size: int
    distance: Distance
    use_binary_quantization: bool
    description: str


class MemoryEngine:
    """
    Initialize and manage Qdrant collections with three-tier memory architecture.
    
    Collections:
    - sentinel_semantic: Official documents (PDFs, reports) - high precision
    - sentinel_episodic: Field evidence (satellite images) - binary quantized
    - sentinel_audio: Emergency calls and transcripts - binary quantized
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        api_key: Optional[str] = None,
        text_vector_size: int = 384,
        vision_vector_size: int = 768,
        audio_vector_size: int = 384,
        video_vector_size: int = 384,
        storage_path: Optional[str] = None
    ):
        """
        Initialize the Memory Engine.
        
        Args:
            host: Qdrant server hostname
            port: Qdrant HTTP API port
            api_key: Optional API key for authentication
            text_vector_size: Dimension of text embeddings
            vision_vector_size: Dimension of vision embeddings
            audio_vector_size: Dimension of audio transcript embeddings
            storage_path: Optional path for local storage (Edge mode)
        """
        self.host = host
        self.port = port
        self.api_key = api_key
        self.storage_path = storage_path
        
        # Initialize Qdrant client
        try:
            if self.storage_path:
                # Local/Edge Mode
                logger.info(f"Initializing Qdrant in Local/Edge mode at {self.storage_path}")
                self.client = QdrantClient(path=self.storage_path)
            else:
                # Server Mode
                # Use URL format to ensure HTTP (not HTTPS)
                url = f"http://{host}:{port}"
                self.client = QdrantClient(
                    url=url,
                    api_key=api_key,
                    prefer_grpc=False,
                    timeout=30
                )
                logger.info(f"Connected to Qdrant at {url}")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise
        
        # Define collection configurations
        self.collections_config: Dict[str, CollectionConfig] = {
            "sentinel_semantic": CollectionConfig(
                name="sentinel_semantic",
                vector_size=text_vector_size,
                distance=Distance.COSINE,
                use_binary_quantization=False,
                description="Official documents (PDFs, reports)"
            ),
            "sentinel_episodic": CollectionConfig(
                name="sentinel_episodic",
                vector_size=vision_vector_size,
                distance=Distance.COSINE,
                use_binary_quantization=True,
                description="Field evidence (satellite images)"
            ),
            "sentinel_audio": CollectionConfig(
                name="sentinel_audio",
                vector_size=audio_vector_size,
                distance=Distance.COSINE,
                use_binary_quantization=True,
                description="Emergency calls and audio transcripts"
            ),
            "sentinel_video": CollectionConfig(
                name="sentinel_video",
                vector_size=video_vector_size,
                distance=Distance.COSINE,
                use_binary_quantization=True,
                description="Video recordings with transcripts"
            )
        }
    
    def _get_existing_collections(self) -> List[str]:
        """Get list of existing collection names."""
        try:
            collections = self.client.get_collections()
            return [c.name for c in collections.collections]
        except Exception as e:
            logger.error(f"Failed to get collections: {e}")
            return []
    
    def initialize_memory(self) -> Dict[str, bool]:
        """
        Create collections if they don't exist.
        
        Returns:
            Dict mapping collection name to success status
        """
        results = {}
        existing = self._get_existing_collections()
        
        for name, config in self.collections_config.items():
            try:
                if name in existing:
                    logger.info(f"✓ Collection '{name}' already exists")
                    results[name] = True
                    continue
                
                logger.info(f"Creating collection: {name}")
                
                # Build quantization config if needed
                quantization_config = None
                if config.use_binary_quantization:
                    quantization_config = BinaryQuantization(
                        binary=BinaryQuantizationConfig(
                            always_ram=True
                        )
                    )
                
                # Create the collection
                self.client.create_collection(
                    collection_name=name,
                    vectors_config=VectorParams(
                        size=config.vector_size,
                        distance=config.distance
                    ),
                    quantization_config=quantization_config
                )
                
                logger.info(f"✓ Collection '{name}' created successfully")
                results[name] = True
                
            except Exception as e:
                logger.error(f"✗ Failed to create '{name}': {e}")
                results[name] = False
        
        return results
    
    def get_collection_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics for all collections.
        
        Returns:
            Dict with collection stats including vector count, status, etc.
        """
        stats = {}
        
        for name in self.collections_config.keys():
            try:
                info = self.client.get_collection(collection_name=name)
                # Handle different qdrant-client versions
                # Newer versions use points_count, older use vectors_count
                points = getattr(info, 'points_count', None)
                if points is None:
                    points = getattr(info, 'vectors_count', 0)
                
                indexed = getattr(info, 'indexed_vectors_count', 0)
                status_val = getattr(info.status, 'value', 'unknown') if info.status else 'unknown'
                
                stats[name] = {
                    "vectors_count": points or 0,
                    "points_count": points or 0,
                    "status": status_val,
                    "indexed_vectors_count": indexed or 0,
                    "description": self.collections_config[name].description
                }
            except Exception as e:
                logger.warning(f"Could not get stats for '{name}': {e}")
                stats[name] = {
                    "vectors_count": 0,
                    "points_count": 0,
                    "status": "error",
                    "error": str(e),
                    "description": self.collections_config.get(name, CollectionConfig(
                        name=name, vector_size=0, distance=Distance.COSINE,
                        use_binary_quantization=False, description="Unknown"
                    )).description
                }
        
        return stats
    
    def upsert_vectors(
        self,
        collection_name: str,
        points: List[PointStruct]
    ) -> bool:
        """
        Upsert vectors into a collection.
        
        Args:
            collection_name: Target collection name
            points: List of PointStruct objects to upsert
        
        Returns:
            True if successful, False otherwise
        """
        if not points:
            logger.warning("No points to upsert")
            return True
        
        try:
            self.client.upsert(
                collection_name=collection_name,
                points=points,
                wait=True
            )
            logger.info(f"Upserted {len(points)} vectors to '{collection_name}'")
            return True
        except Exception as e:
            logger.error(f"Failed to upsert to '{collection_name}': {e}")
            return False
    
    def delete_collection(self, collection_name: str) -> bool:
        """
        Delete a collection.
        
        Args:
            collection_name: Name of collection to delete
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.client.delete_collection(collection_name=collection_name)
            logger.info(f"Deleted collection '{collection_name}'")
            return True
        except Exception as e:
            logger.error(f"Failed to delete '{collection_name}': {e}")
            return False
    
    def reset_all_collections(self) -> Dict[str, bool]:
        """
        Delete and recreate all collections.
        
        Returns:
            Dict mapping collection name to success status
        """
        results = {}
        
        for name in self.collections_config.keys():
            # Delete if exists
            try:
                self.client.delete_collection(collection_name=name)
                logger.info(f"Deleted collection '{name}'")
            except Exception:
                pass  # Collection might not exist
        
        # Recreate all
        return self.initialize_memory()
    
    def health_check(self) -> bool:
        """
        Check if Qdrant is healthy and responsive.
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            # Try to get collections - if this works, we're connected
            self.client.get_collections()
            return True
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    def get_client(self) -> QdrantClient:
        """Return the underlying Qdrant client for advanced operations."""
        return self.client

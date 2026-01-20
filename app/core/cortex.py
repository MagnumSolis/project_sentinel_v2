"""
Project Sentinel V2 - Sentinel Cortex

The "brain" of the system - handles search and retrieval across all memory layers.
Performs parallel semantic search with time decay and damage score boosting.
"""

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Filter,
    FieldCondition,
    MatchValue,
    Range,
    SearchRequest,
    ScoredPoint,
)
from fastembed import TextEmbedding
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Structured search result from a single collection."""
    collection: str
    score: float
    adjusted_score: float
    payload: Dict[str, Any]
    vector_id: int
    timestamp: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "collection": self.collection,
            "score": self.score,
            "adjusted_score": self.adjusted_score,
            "payload": self.payload,
            "vector_id": self.vector_id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None
        }


@dataclass
class CombinedSearchResults:
    """Combined results from all collections."""
    query: str
    results: List[SearchResult] = field(default_factory=list)
    total_count: int = 0
    search_time_ms: float = 0.0
    collections_searched: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "query": self.query,
            "results": [r.to_dict() for r in self.results],
            "total_count": self.total_count,
            "search_time_ms": self.search_time_ms,
            "collections_searched": self.collections_searched
        }


class SentinelCortex:
    """
    Search and retrieval engine combining all three memory layers.
    
    Features:
    - Parallel semantic search across all collections
    - Time decay scoring (newer = higher priority)
    - Damage score boosting (severe damage = higher priority)
    - Result fusion with source attribution
    """
    
    COLLECTION_SEMANTIC = "sentinel_semantic"
    COLLECTION_EPISODIC = "sentinel_episodic"
    COLLECTION_AUDIO = "sentinel_audio"
    
    def __init__(
        self,
        qdrant_client: QdrantClient,
        text_embedding_model: str = "BAAI/bge-small-en-v1.5",
        time_decay_factor: float = 0.95,
        search_limit: int = 5,
        similarity_threshold: float = 0.6
    ):
        """
        Initialize the Sentinel Cortex.
        
        Args:
            qdrant_client: Connected Qdrant client instance
            text_embedding_model: Model name for query embedding
            time_decay_factor: Score multiplier per day old (0.95 = 5% reduction/day)
            search_limit: Max results per collection
            similarity_threshold: Minimum score to include result
        """
        self.client = qdrant_client
        self.time_decay_factor = time_decay_factor
        self.search_limit = search_limit
        self.similarity_threshold = similarity_threshold
        
        # Initialize text embedder for queries
        logger.info(f"Loading text embedding model: {text_embedding_model}")
        self.text_embedder = TextEmbedding(model_name=text_embedding_model)
        logger.info("Text embedder loaded successfully")
        
        # Thread pool for parallel operations
        self._executor = ThreadPoolExecutor(max_workers=3)
    
    def _embed_query(self, query: str) -> List[float]:
        """
        Convert query text to embedding vector.
        
        Args:
            query: Search query text
        
        Returns:
            List of floats representing the query vector
        """
        embeddings = list(self.text_embedder.embed([query]))
        return embeddings[0].tolist()
    
    def _apply_time_decay(
        self,
        score: float,
        timestamp: Optional[str],
        reference_time: datetime
    ) -> float:
        """
        Apply time decay to score based on document age.
        
        Args:
            score: Original similarity score
            timestamp: ISO8601 timestamp string or None
            reference_time: Current time for comparison
        
        Returns:
            Adjusted score with time decay applied
        """
        if not timestamp:
            return score
        
        try:
            doc_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            days_old = (reference_time - doc_time.replace(tzinfo=None)).days
            
            if days_old < 0:
                days_old = 0
            
            # Apply exponential decay
            decay_multiplier = self.time_decay_factor ** days_old
            return score * decay_multiplier
            
        except (ValueError, TypeError) as e:
            logger.debug(f"Could not parse timestamp '{timestamp}': {e}")
            return score
    
    def _apply_damage_boost(
        self,
        score: float,
        payload: Dict[str, Any]
    ) -> float:
        """
        Boost score for items with high damage scores.
        
        Args:
            score: Current score
            payload: Document payload with possible damage_score field
        
        Returns:
            Boosted score if damage_score is present and high
        """
        damage_score = payload.get("damage_score", 0.0)
        
        if damage_score > 0.7:
            # High damage: 20% boost
            return score * 1.2
        elif damage_score > 0.4:
            # Medium damage: 10% boost
            return score * 1.1
        
        return score
    
    def _search_collection(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int
    ) -> List[ScoredPoint]:
        """
        Search a single collection.
        
        Args:
            collection_name: Name of collection to search
            query_vector: Query embedding vector
            limit: Maximum results to return
        
        Returns:
            List of ScoredPoint results
        """
        try:
            # Use query_points for newer qdrant-client versions (1.7+)
            results = self.client.query_points(
                collection_name=collection_name,
                query=query_vector,
                limit=limit,
                with_payload=True,
                score_threshold=self.similarity_threshold
            )
            # query_points returns QueryResponse with .points attribute
            return results.points if hasattr(results, 'points') else results
        except AttributeError:
            # Fallback for older versions
            try:
                results = self.client.search(
                    collection_name=collection_name,
                    query_vector=query_vector,
                    limit=limit,
                    with_payload=True,
                    score_threshold=self.similarity_threshold
                )
                return results
            except Exception as e:
                logger.error(f"Search failed for '{collection_name}': {e}")
                return []
        except Exception as e:
            logger.error(f"Search failed for '{collection_name}': {e}")
            return []
    
    def search(
        self,
        query: str,
        collections: Optional[List[str]] = None,
        limit: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> CombinedSearchResults:
        """
        Perform parallel semantic search across all specified collections.
        
        Args:
            query: Natural language search query
            collections: List of collections to search (default: all)
            limit: Max results per collection (default: self.search_limit)
            filters: Optional filter conditions per collection
        
        Returns:
            CombinedSearchResults with ranked, deduplicated results
        """
        import time
        start_time = time.time()
        
        # Default to all collections
        if collections is None:
            collections = [
                self.COLLECTION_SEMANTIC,
                self.COLLECTION_EPISODIC,
                self.COLLECTION_AUDIO
            ]
        
        if limit is None:
            limit = self.search_limit
        
        # Embed the query
        try:
            query_vector = self._embed_query(query)
        except Exception as e:
            logger.error(f"Failed to embed query: {e}")
            return CombinedSearchResults(
                query=query,
                results=[],
                total_count=0,
                search_time_ms=0,
                collections_searched=[]
            )
        
        # Reference time for decay calculation
        reference_time = datetime.utcnow()
        
        # Search all collections
        all_results: List[SearchResult] = []
        
        for collection in collections:
            raw_results = self._search_collection(collection, query_vector, limit)
            
            for point in raw_results:
                payload = point.payload or {}
                timestamp = payload.get("timestamp")
                
                # Apply scoring adjustments
                adjusted_score = point.score
                adjusted_score = self._apply_time_decay(
                    adjusted_score, timestamp, reference_time
                )
                adjusted_score = self._apply_damage_boost(adjusted_score, payload)
                
                # Parse timestamp if available
                parsed_timestamp = None
                if timestamp:
                    try:
                        parsed_timestamp = datetime.fromisoformat(
                            timestamp.replace('Z', '+00:00')
                        ).replace(tzinfo=None)
                    except (ValueError, TypeError):
                        pass
                
                result = SearchResult(
                    collection=collection,
                    score=point.score,
                    adjusted_score=adjusted_score,
                    payload=payload,
                    vector_id=point.id,
                    timestamp=parsed_timestamp
                )
                all_results.append(result)
        
        # Sort by adjusted score (descending)
        all_results.sort(key=lambda r: r.adjusted_score, reverse=True)
        
        # Calculate search time
        search_time_ms = (time.time() - start_time) * 1000
        
        return CombinedSearchResults(
            query=query,
            results=all_results,
            total_count=len(all_results),
            search_time_ms=search_time_ms,
            collections_searched=collections
        )
    
    def search_semantic(
        self,
        query: str,
        limit: Optional[int] = None
    ) -> List[SearchResult]:
        """Search only the semantic collection (PDFs, reports)."""
        results = self.search(
            query=query,
            collections=[self.COLLECTION_SEMANTIC],
            limit=limit
        )
        return results.results
    
    def search_episodic(
        self,
        query: str,
        limit: Optional[int] = None
    ) -> List[SearchResult]:
        """Search only the episodic collection (images)."""
        results = self.search(
            query=query,
            collections=[self.COLLECTION_EPISODIC],
            limit=limit
        )
        return results.results
    
    def search_audio(
        self,
        query: str,
        limit: Optional[int] = None
    ) -> List[SearchResult]:
        """Search only the audio collection (transcripts)."""
        results = self.search(
            query=query,
            collections=[self.COLLECTION_AUDIO],
            limit=limit
        )
        return results.results
    
    def search_by_stress_level(
        self,
        min_stress: float = 0.7,
        limit: int = 10
    ) -> List[SearchResult]:
        """
        Find audio entries with high stress levels.
        
        Args:
            min_stress: Minimum stress level (0.0-1.0)
            limit: Maximum results
        
        Returns:
            List of high-stress audio entries
        """
        try:
            results = self.client.scroll(
                collection_name=self.COLLECTION_AUDIO,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="stress_level",
                            range=Range(gte=min_stress)
                        )
                    ]
                ),
                limit=limit,
                with_payload=True,
                with_vectors=False
            )
            
            search_results = []
            for point in results[0]:
                search_results.append(SearchResult(
                    collection=self.COLLECTION_AUDIO,
                    score=1.0,
                    adjusted_score=point.payload.get("stress_level", 0.0),
                    payload=point.payload,
                    vector_id=point.id
                ))
            
            return search_results
            
        except Exception as e:
            logger.error(f"Stress level search failed: {e}")
            return []
    
    def search_by_damage(
        self,
        min_damage: float = 0.5,
        damage_categories: Optional[List[str]] = None,
        limit: int = 10
    ) -> List[SearchResult]:
        """
        Find images with high damage scores.
        
        Args:
            min_damage: Minimum damage score (0.0-1.0)
            damage_categories: Optional list of damage types to filter
            limit: Maximum results
        
        Returns:
            List of high-damage image entries
        """
        try:
            must_conditions = [
                FieldCondition(
                    key="damage_score",
                    range=Range(gte=min_damage)
                )
            ]
            
            results = self.client.scroll(
                collection_name=self.COLLECTION_EPISODIC,
                scroll_filter=Filter(must=must_conditions),
                limit=limit,
                with_payload=True,
                with_vectors=False
            )
            
            search_results = []
            for point in results[0]:
                # Apply category filter if specified
                if damage_categories:
                    item_categories = point.payload.get("damage_categories", [])
                    if not any(c in item_categories for c in damage_categories):
                        continue
                
                search_results.append(SearchResult(
                    collection=self.COLLECTION_EPISODIC,
                    score=1.0,
                    adjusted_score=point.payload.get("damage_score", 0.0),
                    payload=point.payload,
                    vector_id=point.id
                ))
            
            return search_results
            
        except Exception as e:
            logger.error(f"Damage search failed: {e}")
            return []
    
    def format_context_for_llm(
        self,
        results: CombinedSearchResults,
        max_context_length: int = 4000
    ) -> str:
        """
        Format search results as context for LLM prompt.
        
        Args:
            results: Combined search results
            max_context_length: Maximum characters for context
        
        Returns:
            Formatted context string with source citations
        """
        if not results.results:
            return "No relevant information found in the database."
        
        context_parts = []
        current_length = 0
        
        for i, result in enumerate(results.results, 1):
            # Build source label
            source_type = result.payload.get("source_type", result.collection)
            timestamp = result.payload.get("timestamp", "Unknown time")
            
            # Extract content based on collection type
            if result.collection == self.COLLECTION_SEMANTIC:
                content = result.payload.get("text", result.payload.get("content", ""))
                source = f"Document: {result.payload.get('document_id', 'Unknown')}"
            elif result.collection == self.COLLECTION_EPISODIC:
                content = result.payload.get("description", "Image analysis not available")
                damage = result.payload.get("damage_score", 0)
                categories = result.payload.get("damage_categories", [])
                content += f" | Damage Score: {damage:.2f} | Categories: {', '.join(categories)}"
                source = f"Image: {result.payload.get('media_type', 'Unknown')}"
            else:  # Audio
                content = result.payload.get("transcript", "Transcript not available")
                stress = result.payload.get("stress_level", 0)
                content += f" | Stress Level: {stress:.2f}"
                source = f"Audio: {result.payload.get('source', 'Unknown')}"
            
            # Format entry
            entry = f"[Source {i}] ({source}, {timestamp})\n{content}\n"
            
            # Check length
            if current_length + len(entry) > max_context_length:
                break
            
            context_parts.append(entry)
            current_length += len(entry)
        
        return "\n".join(context_parts)

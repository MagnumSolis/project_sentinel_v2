"""
Project Sentinel V2 - Core Tests

Basic unit tests for the core components.
Run with: pytest tests/ -v
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestConfig:
    """Tests for configuration module."""
    
    def test_settings_load(self):
        """Test that settings load correctly."""
        from app.config import settings
        
        assert settings.QDRANT_HOST == "localhost"
        assert settings.QDRANT_PORT == 6333
        assert settings.TEXT_EMBEDDING_MODEL == "BAAI/bge-small-en-v1.5"
        assert settings.TEXT_VECTOR_SIZE == 384
    
    def test_settings_properties(self):
        """Test settings helper properties."""
        from app.config import settings
        
        assert settings.qdrant_url == "http://localhost:6333"
        assert isinstance(settings.data_path, Path)


class TestMemoryEngine:
    """Tests for Memory Engine."""
    
    @patch('app.core.memory_engine.QdrantClient')
    def test_memory_engine_init(self, mock_client):
        """Test memory engine initialization."""
        from app.core.memory_engine import MemoryEngine
        
        engine = MemoryEngine(host="localhost", port=6333)
        
        assert engine.host == "localhost"
        assert engine.port == 6333
        assert len(engine.collections_config) == 3
    
    @patch('app.core.memory_engine.QdrantClient')
    def test_collection_configs(self, mock_client):
        """Test collection configurations are correct."""
        from app.core.memory_engine import MemoryEngine
        from qdrant_client.models import Distance
        
        engine = MemoryEngine(host="localhost", port=6333)
        
        # Semantic collection
        semantic = engine.collections_config["sentinel_semantic"]
        assert semantic.vector_size == 384
        assert semantic.distance == Distance.COSINE
        assert semantic.use_binary_quantization is False
        
        # Episodic collection
        episodic = engine.collections_config["sentinel_episodic"]
        assert episodic.vector_size == 768
        assert episodic.use_binary_quantization is True
        
        # Audio collection
        audio = engine.collections_config["sentinel_audio"]
        assert audio.vector_size == 384
        assert audio.use_binary_quantization is True


class TestCortex:
    """Tests for Sentinel Cortex."""
    
    def test_time_decay(self):
        """Test time decay calculation."""
        from app.core.cortex import SentinelCortex
        from datetime import datetime, timedelta
        
        # Create cortex with mocked client
        with patch('app.core.cortex.TextEmbedding'):
            mock_client = Mock()
            cortex = SentinelCortex(
                qdrant_client=mock_client,
                time_decay_factor=0.95
            )
        
        reference_time = datetime(2024, 1, 15)
        
        # Same day - no decay
        score = cortex._apply_time_decay(1.0, "2024-01-15T12:00:00", reference_time)
        assert score == 1.0
        
        # 1 day old - 5% decay
        score = cortex._apply_time_decay(1.0, "2024-01-14T12:00:00", reference_time)
        assert abs(score - 0.95) < 0.01
        
        # 7 days old
        score = cortex._apply_time_decay(1.0, "2024-01-08T12:00:00", reference_time)
        assert score < 0.75
    
    def test_damage_boost(self):
        """Test damage score boosting."""
        from app.core.cortex import SentinelCortex
        
        with patch('app.core.cortex.TextEmbedding'):
            mock_client = Mock()
            cortex = SentinelCortex(qdrant_client=mock_client)
        
        # No damage - no boost
        score = cortex._apply_damage_boost(1.0, {"damage_score": 0.0})
        assert score == 1.0
        
        # Medium damage - 10% boost
        score = cortex._apply_damage_boost(1.0, {"damage_score": 0.5})
        assert score == 1.1
        
        # High damage - 20% boost
        score = cortex._apply_damage_boost(1.0, {"damage_score": 0.8})
        assert score == 1.2


class TestUniversalIngestor:
    """Tests for Universal Ingestor."""
    
    def test_chunk_text(self):
        """Test text chunking."""
        from app.ingestion.universal_ingestor import UniversalIngestor
        
        with patch('app.ingestion.universal_ingestor.TextEmbedding'), \
             patch('app.ingestion.universal_ingestor.ImageEmbedding'):
            mock_client = Mock()
            ingestor = UniversalIngestor(
                qdrant_client=mock_client,
                text_embedding_model="test",
                vision_embedding_model="test"
            )
        
        # Test basic chunking
        text = "This is a test. " * 100  # ~1600 chars
        chunks = ingestor._chunk_text(text, chunk_size=500, overlap=50)
        
        assert len(chunks) >= 3
        assert all(len(chunk[0]) <= 550 for chunk in chunks)  # Allow some overflow
        assert all(isinstance(chunk[1], int) for chunk in chunks)  # Check indices
    
    def test_empty_text_chunking(self):
        """Test chunking with empty text."""
        from app.ingestion.universal_ingestor import UniversalIngestor
        
        with patch('app.ingestion.universal_ingestor.TextEmbedding'), \
             patch('app.ingestion.universal_ingestor.ImageEmbedding'):
            mock_client = Mock()
            ingestor = UniversalIngestor(
                qdrant_client=mock_client,
                text_embedding_model="test",
                vision_embedding_model="test"
            )
        
        assert ingestor._chunk_text("") == []
        assert ingestor._chunk_text("   ") == []
    
    def test_file_hash(self):
        """Test file hash computation."""
        from app.ingestion.universal_ingestor import UniversalIngestor
        
        with patch('app.ingestion.universal_ingestor.TextEmbedding'), \
             patch('app.ingestion.universal_ingestor.ImageEmbedding'):
            mock_client = Mock()
            ingestor = UniversalIngestor(
                qdrant_client=mock_client,
                text_embedding_model="test",
                vision_embedding_model="test"
            )
        
        # Create temp file
        with tempfile.NamedTemporaryFile(delete=False, mode='w') as f:
            f.write("test content")
            temp_path = Path(f.name)
        
        try:
            hash1 = ingestor._compute_file_hash(temp_path)
            hash2 = ingestor._compute_file_hash(temp_path)
            
            assert hash1 == hash2  # Same file = same hash
            assert len(hash1) == 64  # SHA256 hex length
        finally:
            os.unlink(temp_path)


class TestGeminiIntegration:
    """Tests for Gemini Integration."""
    
    def test_offline_mode(self):
        """Test offline mode initialization."""
        from app.llm.gemini_integration import GeminiIntegration
        
        gemini = GeminiIntegration(offline_mode=True)
        
        assert gemini.available is False
        assert gemini.is_available() is False
    
    def test_offline_response(self):
        """Test offline response generation."""
        from app.llm.gemini_integration import GeminiIntegration
        
        gemini = GeminiIntegration(offline_mode=True)
        
        response = gemini.generate_response(
            query="Test query",
            context="[Source 1] Test context"
        )
        
        assert "Test query" in response
        assert "Offline Mode" in response
    
    def test_json_parsing(self):
        """Test JSON parsing from LLM responses."""
        from app.llm.gemini_integration import GeminiIntegration
        
        gemini = GeminiIntegration(offline_mode=True)
        
        # Direct JSON
        result = gemini._parse_json_response('{"key": "value"}')
        assert result == {"key": "value"}
        
        # JSON in code block
        result = gemini._parse_json_response('```json\n{"key": "value"}\n```')
        assert result == {"key": "value"}
        
        # Invalid JSON
        result = gemini._parse_json_response('not json at all')
        assert result == {}


class TestSearchResults:
    """Tests for search result data classes."""
    
    def test_search_result_to_dict(self):
        """Test SearchResult serialization."""
        from app.core.cortex import SearchResult
        from datetime import datetime
        
        result = SearchResult(
            collection="sentinel_semantic",
            score=0.9,
            adjusted_score=0.85,
            payload={"text": "test"},
            vector_id=123,
            timestamp=datetime(2024, 1, 15, 12, 0, 0)
        )
        
        d = result.to_dict()
        
        assert d["collection"] == "sentinel_semantic"
        assert d["score"] == 0.9
        assert d["adjusted_score"] == 0.85
        assert d["payload"] == {"text": "test"}
        assert d["vector_id"] == 123
        assert "2024-01-15" in d["timestamp"]
    
    def test_combined_results_to_dict(self):
        """Test CombinedSearchResults serialization."""
        from app.core.cortex import CombinedSearchResults, SearchResult
        
        results = CombinedSearchResults(
            query="test query",
            results=[
                SearchResult(
                    collection="test",
                    score=0.9,
                    adjusted_score=0.9,
                    payload={},
                    vector_id=1
                )
            ],
            total_count=1,
            search_time_ms=50.0,
            collections_searched=["test"]
        )
        
        d = results.to_dict()
        
        assert d["query"] == "test query"
        assert len(d["results"]) == 1
        assert d["total_count"] == 1
        assert d["search_time_ms"] == 50.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

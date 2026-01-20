"""
Project Sentinel V2 - Universal Ingestor

Unified ingestion pipeline for all modalities:
- PDF documents (text extraction + chunking)
- Images (visual embedding + optional Gemini analysis)
- Audio files (Whisper transcription + stress analysis)
"""

from fastembed import TextEmbedding, ImageEmbedding
from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from PIL import Image
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Callable
from datetime import datetime
from dataclasses import dataclass
import logging
import base64
import io
import hashlib
import subprocess
import tempfile
import os
import uuid

logger = logging.getLogger(__name__)


@dataclass
class IngestionResult:
    """Result of a file ingestion operation."""
    file_path: str
    success: bool
    vectors_added: int
    collection: str
    errors: List[str]
    metadata: Dict[str, Any]


class UniversalIngestor:
    """
    Ingest PDFs, images, and audio files into Qdrant with multimodal embeddings.
    
    This class handles:
    - PDF text extraction and semantic chunking
    - Image embedding and optional AI analysis
    - Audio transcription and prosodic feature extraction
    """
    
    COLLECTION_SEMANTIC = "sentinel_semantic"
    COLLECTION_EPISODIC = "sentinel_episodic"
    COLLECTION_AUDIO = "sentinel_audio"
    
    def __init__(
        self,
        qdrant_client: QdrantClient,
        text_embedding_model: str = "BAAI/bge-small-en-v1.5",
        vision_embedding_model: str = "jinaai/jina-clip-v1",
        whisper_model_name: str = "base",
        gemini_client: Optional[Any] = None
    ):
        """
        Initialize the Universal Ingestor.
        
        Args:
            qdrant_client: Connected Qdrant client
            text_embedding_model: FastEmbed model for text
            vision_embedding_model: FastEmbed model for images
            whisper_model_name: Whisper model size (tiny/base/small/medium/large)
            gemini_client: Optional GeminiIntegration for enhanced image analysis
        """
        self.client = qdrant_client
        self.gemini_client = gemini_client
        
        # Load text embedding model
        logger.info(f"Loading text embedding model: {text_embedding_model}")
        self.text_embedder = TextEmbedding(model_name=text_embedding_model)
        
        # Load vision embedding model
        logger.info(f"Loading vision embedding model: {vision_embedding_model}")
        self.vision_embedder = ImageEmbedding(model_name=vision_embedding_model)
        
        # Load Whisper model (lazy loading to save memory)
        self.whisper_model_name = whisper_model_name
        self._whisper_model = None
        
        # Track ingested files
        self._ingested_hashes: set = set()
    
    @property
    def whisper_model(self):
        """Lazy load Whisper model on first use."""
        if self._whisper_model is None:
            logger.info(f"Loading Whisper {self.whisper_model_name} model...")
            import whisper
            self._whisper_model = whisper.load_model(self.whisper_model_name)
            logger.info("Whisper model loaded")
        return self._whisper_model
    
    def _generate_id(self) -> int:
        """Generate a unique integer ID for vectors."""
        return int(uuid.uuid4().int & (1 << 63) - 1)
    
    def _compute_file_hash(self, file_path: Path) -> str:
        """Compute SHA256 hash of file for deduplication."""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    def _chunk_text(
        self,
        text: str,
        chunk_size: int = 500,
        overlap: int = 50
    ) -> List[Tuple[str, int]]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Full text to chunk
            chunk_size: Target chunk size in characters
            overlap: Overlap between chunks
        
        Returns:
            List of (chunk_text, chunk_index) tuples
        """
        if not text or len(text.strip()) == 0:
            return []
        
        text = text.strip()
        chunks = []
        start = 0
        chunk_index = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings
                for punct in ['. ', '! ', '? ', '\n\n', '\n']:
                    last_punct = text[start:end].rfind(punct)
                    if last_punct > chunk_size // 2:
                        end = start + last_punct + len(punct)
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append((chunk, chunk_index))
                chunk_index += 1
            
            start = end - overlap
            if start < 0:
                start = 0
        
        return chunks
    
    def _image_to_base64_thumbnail(
        self,
        image_path: Path,
        max_size: Tuple[int, int] = (150, 150)
    ) -> str:
        """
        Create a small base64-encoded thumbnail.
        
        Args:
            image_path: Path to image file
            max_size: Maximum thumbnail dimensions
        
        Returns:
            Base64-encoded JPEG thumbnail
        """
        try:
            with Image.open(image_path) as img:
                img.thumbnail(max_size, Image.Resampling.LANCZOS)
                
                # Convert to RGB if needed
                if img.mode in ('RGBA', 'P'):
                    img = img.convert('RGB')
                
                buffer = io.BytesIO()
                img.save(buffer, format='JPEG', quality=70)
                return base64.b64encode(buffer.getvalue()).decode('utf-8')
        except Exception as e:
            logger.warning(f"Could not create thumbnail: {e}")
            return ""
    
    def ingest_pdf(
        self,
        file_path: str,
        document_id: Optional[str] = None,
        source_type: str = "pdf_report",
        extra_metadata: Optional[Dict[str, Any]] = None
    ) -> IngestionResult:
        """
        Ingest a PDF document into the semantic collection.
        
        Args:
            file_path: Path to PDF file
            document_id: Optional document identifier
            source_type: Type of document
            extra_metadata: Additional metadata to store
        
        Returns:
            IngestionResult with status and stats
        """
        path = Path(file_path)
        errors = []
        vectors_added = 0
        
        if not path.exists():
            return IngestionResult(
                file_path=str(path),
                success=False,
                vectors_added=0,
                collection=self.COLLECTION_SEMANTIC,
                errors=[f"File not found: {path}"],
                metadata={}
            )
        
        # Generate document ID if not provided
        if document_id is None:
            document_id = path.stem
        
        # Check for duplicates
        file_hash = self._compute_file_hash(path)
        if file_hash in self._ingested_hashes:
            logger.info(f"Skipping duplicate file: {path.name}")
            return IngestionResult(
                file_path=str(path),
                success=True,
                vectors_added=0,
                collection=self.COLLECTION_SEMANTIC,
                errors=["Duplicate file skipped"],
                metadata={"document_id": document_id}
            )
        
        try:
            # Extract text from PDF
            reader = PdfReader(path)
            full_text = ""
            
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    full_text += page_text + "\n\n"
            
            if not full_text.strip():
                errors.append("No text content extracted from PDF")
                return IngestionResult(
                    file_path=str(path),
                    success=False,
                    vectors_added=0,
                    collection=self.COLLECTION_SEMANTIC,
                    errors=errors,
                    metadata={"document_id": document_id}
                )
            
            # Chunk the text
            chunks = self._chunk_text(full_text)
            logger.info(f"Extracted {len(chunks)} chunks from {path.name}")
            
            # Embed and upsert each chunk
            points = []
            timestamp = datetime.utcnow().isoformat()
            
            for chunk_text, chunk_index in chunks:
                # Generate embedding
                embedding = list(self.text_embedder.embed([chunk_text]))[0].tolist()
                
                # Build payload
                payload = {
                    "document_id": document_id,
                    "chunk_index": chunk_index,
                    "text": chunk_text,
                    "source_type": source_type,
                    "timestamp": timestamp,
                    "file_name": path.name,
                    "confidence": 1.0
                }
                
                if extra_metadata:
                    payload.update(extra_metadata)
                
                points.append(PointStruct(
                    id=self._generate_id(),
                    vector=embedding,
                    payload=payload
                ))
            
            # Batch upsert
            if points:
                self.client.upsert(
                    collection_name=self.COLLECTION_SEMANTIC,
                    points=points,
                    wait=True
                )
                vectors_added = len(points)
                self._ingested_hashes.add(file_hash)
            
            logger.info(f"✓ Ingested {path.name}: {vectors_added} vectors")
            
            return IngestionResult(
                file_path=str(path),
                success=True,
                vectors_added=vectors_added,
                collection=self.COLLECTION_SEMANTIC,
                errors=errors,
                metadata={
                    "document_id": document_id,
                    "total_chunks": len(chunks),
                    "text_length": len(full_text)
                }
            )
            
        except Exception as e:
            # Try reading as plain text (fallback for text files with .pdf extension)
            logger.warning(f"PDF parsing failed for {path.name}, trying as text: {e}")
            try:
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    full_text = f.read()
                
                if full_text.strip():
                    # Chunk the text
                    chunks = self._chunk_text(full_text)
                    logger.info(f"Extracted {len(chunks)} chunks from {path.name} (as text)")
                    
                    # Embed and upsert each chunk
                    points = []
                    timestamp = datetime.utcnow().isoformat()
                    
                    for chunk_text, chunk_index in chunks:
                        embedding = list(self.text_embedder.embed([chunk_text]))[0].tolist()
                        
                        payload = {
                            "document_id": document_id,
                            "chunk_index": chunk_index,
                            "text": chunk_text,
                            "source_type": source_type,
                            "timestamp": timestamp,
                            "file_name": path.name,
                            "confidence": 0.9  # Lower confidence for text fallback
                        }
                        
                        if extra_metadata:
                            payload.update(extra_metadata)
                        
                        points.append(PointStruct(
                            id=self._generate_id(),
                            vector=embedding,
                            payload=payload
                        ))
                    
                    if points:
                        self.client.upsert(
                            collection_name=self.COLLECTION_SEMANTIC,
                            points=points,
                            wait=True
                        )
                        vectors_added = len(points)
                        self._ingested_hashes.add(file_hash)
                    
                    logger.info(f"✓ Ingested {path.name} (as text): {vectors_added} vectors")
                    
                    return IngestionResult(
                        file_path=str(path),
                        success=True,
                        vectors_added=vectors_added,
                        collection=self.COLLECTION_SEMANTIC,
                        errors=["Used text fallback instead of PDF"],
                        metadata={
                            "document_id": document_id,
                            "total_chunks": len(chunks),
                            "text_length": len(full_text)
                        }
                    )
            except Exception as text_error:
                logger.error(f"Text fallback also failed: {text_error}")
            
            errors.append(str(e))
            return IngestionResult(
                file_path=str(path),
                success=False,
                vectors_added=0,
                collection=self.COLLECTION_SEMANTIC,
                errors=errors,
                metadata={"document_id": document_id}
            )
    
    def ingest_image(
        self,
        file_path: str,
        media_type: str = "satellite",
        geohash: Optional[str] = None,
        extra_metadata: Optional[Dict[str, Any]] = None
    ) -> IngestionResult:
        """
        Ingest an image into the episodic collection.
        
        Args:
            file_path: Path to image file
            media_type: Type of image (satellite, uav, street_view)
            geohash: Optional location geohash
            extra_metadata: Additional metadata
        
        Returns:
            IngestionResult with status and stats
        """
        path = Path(file_path)
        errors = []
        
        if not path.exists():
            return IngestionResult(
                file_path=str(path),
                success=False,
                vectors_added=0,
                collection=self.COLLECTION_EPISODIC,
                errors=[f"File not found: {path}"],
                metadata={}
            )
        
        # Check for duplicates
        file_hash = self._compute_file_hash(path)
        if file_hash in self._ingested_hashes:
            logger.info(f"Skipping duplicate image: {path.name}")
            return IngestionResult(
                file_path=str(path),
                success=True,
                vectors_added=0,
                collection=self.COLLECTION_EPISODIC,
                errors=["Duplicate file skipped"],
                metadata={}
            )
        
        try:
            # Generate visual embedding
            embeddings = list(self.vision_embedder.embed([str(path)]))
            embedding = embeddings[0].tolist()
            
            # Create thumbnail
            thumbnail_b64 = self._image_to_base64_thumbnail(path)
            
            # Default damage analysis (will be enhanced by Gemini if available)
            damage_score = 0.0
            damage_categories = []
            description = "Image analysis not available (offline mode)"
            
            # Enhanced analysis with Gemini if available
            if self.gemini_client:
                try:
                    analysis = self.gemini_client.analyze_image(str(path))
                    if analysis:
                        damage_score = analysis.get("damage_score", 0.0)
                        damage_categories = analysis.get("damage_categories", [])
                        description = analysis.get("description", description)
                except Exception as e:
                    logger.warning(f"Gemini analysis failed: {e}")
            
            # Build payload
            timestamp = datetime.utcnow().isoformat()
            payload = {
                "media_type": media_type,
                "timestamp": timestamp,
                "geohash": geohash or "",
                "damage_score": damage_score,
                "damage_categories": damage_categories,
                "description": description,
                "thumbnail_b64": thumbnail_b64,
                "file_name": path.name,
                "source_url": f"file://{path.absolute()}"
            }
            
            if extra_metadata:
                payload.update(extra_metadata)
            
            # Upsert
            point = PointStruct(
                id=self._generate_id(),
                vector=embedding,
                payload=payload
            )
            
            self.client.upsert(
                collection_name=self.COLLECTION_EPISODIC,
                points=[point],
                wait=True
            )
            
            self._ingested_hashes.add(file_hash)
            logger.info(f"✓ Ingested image {path.name}")
            
            return IngestionResult(
                file_path=str(path),
                success=True,
                vectors_added=1,
                collection=self.COLLECTION_EPISODIC,
                errors=errors,
                metadata={
                    "damage_score": damage_score,
                    "damage_categories": damage_categories
                }
            )
            
        except Exception as e:
            logger.error(f"Image ingestion failed for {path.name}: {e}")
            errors.append(str(e))
            return IngestionResult(
                file_path=str(path),
                success=False,
                vectors_added=0,
                collection=self.COLLECTION_EPISODIC,
                errors=errors,
                metadata={}
            )
    
    def _convert_audio_to_wav(self, input_path: Path) -> Optional[Path]:
        """
        Convert audio file to WAV format using ffmpeg.
        
        Args:
            input_path: Path to input audio file
        
        Returns:
            Path to temporary WAV file, or None if conversion failed
        """
        try:
            # Create temp file for output
            temp_wav = tempfile.NamedTemporaryFile(
                suffix='.wav',
                delete=False
            )
            temp_wav.close()
            
            # Run ffmpeg conversion
            cmd = [
                'ffmpeg', '-y', '-i', str(input_path),
                '-ar', '16000',  # 16kHz sample rate for Whisper
                '-ac', '1',      # Mono
                '-f', 'wav',
                temp_wav.name
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=60
            )
            
            if result.returncode == 0:
                return Path(temp_wav.name)
            else:
                logger.error(f"FFmpeg failed: {result.stderr.decode()}")
                os.unlink(temp_wav.name)
                return None
                
        except subprocess.TimeoutExpired:
            logger.error("Audio conversion timed out")
            return None
        except FileNotFoundError:
            logger.error("FFmpeg not found. Install with: sudo apt install ffmpeg")
            return None
        except Exception as e:
            logger.error(f"Audio conversion failed: {e}")
            return None
    
    def _estimate_stress_level(self, audio_path: Path) -> float:
        """
        Estimate stress level from audio prosody.
        
        This is a simplified heuristic. A production system would use
        a proper prosodic analysis library or ML model.
        
        Args:
            audio_path: Path to WAV audio file
        
        Returns:
            Estimated stress level (0.0-1.0)
        """
        try:
            import soundfile as sf
            import numpy as np
            
            # Load audio
            data, sample_rate = sf.read(audio_path)
            
            if len(data) == 0:
                return 0.5
            
            # Simple heuristics based on audio characteristics
            # Higher volume variance and energy = higher stress
            
            # Normalize
            data = data / (np.max(np.abs(data)) + 1e-8)
            
            # Calculate RMS energy
            rms = np.sqrt(np.mean(data ** 2))
            
            # Calculate zero crossing rate (higher = more stressed speech)
            zero_crossings = np.sum(np.abs(np.diff(np.signbit(data)))) / len(data)
            
            # Combine metrics (simplified)
            stress = min(1.0, (rms * 2) + (zero_crossings * 10))
            
            return float(stress)
            
        except Exception as e:
            logger.warning(f"Stress estimation failed: {e}")
            return 0.5  # Default moderate stress
    
    def ingest_audio(
        self,
        file_path: str,
        source: str = "emergency_call",
        extra_metadata: Optional[Dict[str, Any]] = None
    ) -> IngestionResult:
        """
        Ingest an audio file into the audio collection.
        
        Args:
            file_path: Path to audio file
            source: Source type (emergency_call, rescuer_radio)
            extra_metadata: Additional metadata
        
        Returns:
            IngestionResult with status and stats
        """
        path = Path(file_path)
        errors = []
        
        if not path.exists():
            return IngestionResult(
                file_path=str(path),
                success=False,
                vectors_added=0,
                collection=self.COLLECTION_AUDIO,
                errors=[f"File not found: {path}"],
                metadata={}
            )
        
        # Check for duplicates
        file_hash = self._compute_file_hash(path)
        if file_hash in self._ingested_hashes:
            logger.info(f"Skipping duplicate audio: {path.name}")
            return IngestionResult(
                file_path=str(path),
                success=True,
                vectors_added=0,
                collection=self.COLLECTION_AUDIO,
                errors=["Duplicate file skipped"],
                metadata={}
            )
        
        temp_wav_path = None
        
        try:
            # Convert to WAV if needed
            if path.suffix.lower() != '.wav':
                temp_wav_path = self._convert_audio_to_wav(path)
                if temp_wav_path is None:
                    errors.append("Audio conversion failed")
                    return IngestionResult(
                        file_path=str(path),
                        success=False,
                        vectors_added=0,
                        collection=self.COLLECTION_AUDIO,
                        errors=errors,
                        metadata={}
                    )
                wav_path = temp_wav_path
            else:
                wav_path = path
            
            # Transcribe with Whisper
            logger.info(f"Transcribing {path.name}...")
            result = self.whisper_model.transcribe(str(wav_path))
            transcript = result.get("text", "").strip()
            language = result.get("language", "unknown")
            
            if not transcript:
                errors.append("No transcript generated")
                return IngestionResult(
                    file_path=str(path),
                    success=False,
                    vectors_added=0,
                    collection=self.COLLECTION_AUDIO,
                    errors=errors,
                    metadata={}
                )
            
            # Estimate stress level
            stress_level = self._estimate_stress_level(wav_path)
            
            # Get audio duration
            try:
                import soundfile as sf
                info = sf.info(wav_path)
                duration_seconds = info.duration
            except Exception:
                duration_seconds = 0.0
            
            # Generate text embedding of transcript
            embedding = list(self.text_embedder.embed([transcript]))[0].tolist()
            
            # Build payload
            timestamp = datetime.utcnow().isoformat()
            payload = {
                "transcript": transcript,
                "duration_seconds": duration_seconds,
                "stress_level": stress_level,
                "language": language,
                "source": source,
                "timestamp": timestamp,
                "file_name": path.name
            }
            
            if extra_metadata:
                payload.update(extra_metadata)
            
            # Upsert
            point = PointStruct(
                id=self._generate_id(),
                vector=embedding,
                payload=payload
            )
            
            self.client.upsert(
                collection_name=self.COLLECTION_AUDIO,
                points=[point],
                wait=True
            )
            
            self._ingested_hashes.add(file_hash)
            logger.info(f"✓ Ingested audio {path.name}: {len(transcript)} chars")
            
            return IngestionResult(
                file_path=str(path),
                success=True,
                vectors_added=1,
                collection=self.COLLECTION_AUDIO,
                errors=errors,
                metadata={
                    "transcript_length": len(transcript),
                    "stress_level": stress_level,
                    "duration_seconds": duration_seconds,
                    "language": language
                }
            )
            
        except Exception as e:
            logger.error(f"Audio ingestion failed for {path.name}: {e}")
            errors.append(str(e))
            return IngestionResult(
                file_path=str(path),
                success=False,
                vectors_added=0,
                collection=self.COLLECTION_AUDIO,
                errors=errors,
                metadata={}
            )
        finally:
            # Cleanup temp file
            if temp_wav_path and temp_wav_path.exists():
                try:
                    os.unlink(temp_wav_path)
                except Exception:
                    pass
    
    def ingest_file(
        self,
        file_path: str,
        **kwargs
    ) -> IngestionResult:
        """
        Detect file type and route to appropriate processor.
        
        Args:
            file_path: Path to file
            **kwargs: Additional arguments passed to specific ingestor
        
        Returns:
            IngestionResult with status and stats
        """
        path = Path(file_path)
        suffix = path.suffix.lower()
        
        if suffix == '.pdf':
            return self.ingest_pdf(file_path, **kwargs)
        elif suffix in ['.jpg', '.jpeg', '.png', '.webp', '.bmp']:
            return self.ingest_image(file_path, **kwargs)
        elif suffix in ['.mp3', '.wav', '.m4a', '.ogg', '.flac']:
            return self.ingest_audio(file_path, **kwargs)
        else:
            return IngestionResult(
                file_path=str(path),
                success=False,
                vectors_added=0,
                collection="unknown",
                errors=[f"Unsupported file type: {suffix}"],
                metadata={}
            )
    
    def ingest_directory(
        self,
        directory_path: str,
        recursive: bool = True,
        progress_callback: Optional[Callable[[str, int, int], None]] = None
    ) -> Dict[str, Any]:
        """
        Ingest all supported files from a directory.
        
        Args:
            directory_path: Path to directory
            recursive: Whether to process subdirectories
            progress_callback: Optional callback(file_name, current, total)
        
        Returns:
            Summary of ingestion results
        """
        path = Path(directory_path)
        
        if not path.is_dir():
            return {
                "success": False,
                "error": f"Not a directory: {directory_path}",
                "files_processed": 0
            }
        
        # Collect all supported files
        extensions = {'.pdf', '.jpg', '.jpeg', '.png', '.webp', '.bmp',
                      '.mp3', '.wav', '.m4a', '.ogg', '.flac'}
        
        if recursive:
            files = [f for f in path.rglob('*') if f.suffix.lower() in extensions]
        else:
            files = [f for f in path.iterdir() if f.suffix.lower() in extensions]
        
        total = len(files)
        results = {
            "total_files": total,
            "successful": 0,
            "failed": 0,
            "vectors_added": 0,
            "by_collection": {
                self.COLLECTION_SEMANTIC: 0,
                self.COLLECTION_EPISODIC: 0,
                self.COLLECTION_AUDIO: 0
            },
            "errors": []
        }
        
        for i, file_path in enumerate(files):
            if progress_callback:
                progress_callback(file_path.name, i + 1, total)
            
            result = self.ingest_file(str(file_path))
            
            if result.success:
                results["successful"] += 1
                results["vectors_added"] += result.vectors_added
                if result.collection in results["by_collection"]:
                    results["by_collection"][result.collection] += result.vectors_added
            else:
                results["failed"] += 1
                for error in result.errors:
                    results["errors"].append(f"{file_path.name}: {error}")
        
        return results

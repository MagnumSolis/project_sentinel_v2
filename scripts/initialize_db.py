#!/usr/bin/env python3
"""
Project Sentinel V2 - Database Initialization

Populates Qdrant with sample data from the generated datasets.
Run this after download_datasets.py to have a working demo.
"""

import os
import sys
from pathlib import Path
import logging
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Change to project directory for relative paths
os.chdir(PROJECT_ROOT)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def wait_for_qdrant(host: str, port: int, timeout: int = 30) -> bool:
    """Wait for Qdrant to be available."""
    import time
    import requests
    
    # Use /collections endpoint as it's available on all Qdrant versions
    url = f"http://{host}:{port}/collections"
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                logger.info("✓ Qdrant is healthy")
                return True
        except requests.exceptions.RequestException:
            pass
        
        logger.info("Waiting for Qdrant...")
        time.sleep(2)
    
    logger.error("Qdrant did not become available within timeout")
    return False


def main():
    """Initialize database with sample data."""
    logger.info("=" * 60)
    logger.info("Project Sentinel V2 - Database Initialization")
    logger.info("=" * 60)
    
    # Import after path setup
    from app.config import settings
    from app.core.memory_engine import MemoryEngine
    from app.ingestion.universal_ingestor import UniversalIngestor
    from app.llm.perplexity_integration import PerplexityIntegration
    
    # Ensure data directories exist
    settings.ensure_directories()
    
    # Check for Qdrant
    if not wait_for_qdrant(settings.QDRANT_HOST, settings.QDRANT_PORT):
        logger.error("Cannot connect to Qdrant. Please start it with: docker compose up -d")
        sys.exit(1)
    
    # Initialize memory engine
    logger.info("\n--- Initializing Memory Engine ---")
    try:
        memory = MemoryEngine(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
            api_key=settings.QDRANT_API_KEY,
            text_vector_size=settings.TEXT_VECTOR_SIZE,
            vision_vector_size=settings.VISION_VECTOR_SIZE,
            audio_vector_size=settings.AUDIO_VECTOR_SIZE
        )
        
        # Create collections
        results = memory.initialize_memory()
        for name, success in results.items():
            status = "✓" if success else "✗"
            logger.info(f"{status} Collection: {name}")
        
        if not all(results.values()):
            logger.error("Failed to create all collections")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Memory engine initialization failed: {e}")
        sys.exit(1)
    
    # Initialize Perplexity (optional)
    logger.info("\n--- Checking Perplexity Integration ---")
    llm_client = None
    if settings.PERPLEXITY_API_KEY and not settings.OFFLINE_MODE:
        try:
            llm_client = PerplexityIntegration(
                api_key=settings.PERPLEXITY_API_KEY,
                model=settings.PERPLEXITY_MODEL,
                offline_mode=settings.OFFLINE_MODE
            )
            if llm_client.is_available():
                logger.info("✓ Perplexity API connected")
            else:
                logger.info("○ Perplexity API not available, using offline mode")
                llm_client = None
        except Exception as e:
            logger.warning(f"Perplexity initialization failed: {e}")
    else:
        logger.info("○ Running in offline mode (OFFLINE_MODE=True)")
    
    # Initialize ingestor
    logger.info("\n--- Initializing Universal Ingestor ---")
    try:
        ingestor = UniversalIngestor(
            qdrant_client=memory.get_client(),
            text_embedding_model=settings.TEXT_EMBEDDING_MODEL,
            vision_embedding_model=settings.VISION_EMBEDDING_MODEL,
            whisper_model_name=settings.WHISPER_MODEL,
            gemini_client=llm_client
        )
        logger.info("✓ Ingestor ready")
    except Exception as e:
        logger.error(f"Ingestor initialization failed: {e}")
        sys.exit(1)
    
    # Ingest sample data
    data_dir = settings.raw_datasets_path
    
    # Check if sample data exists
    pdf_dir = data_dir / "reports"
    image_dir = data_dir / "images"
    audio_dir = data_dir / "audio"
    
    if not pdf_dir.exists() or not any(pdf_dir.iterdir()):
        logger.warning("No sample data found. Run 'python scripts/download_datasets.py' first.")
        logger.info("Attempting to create sample data now...")
        
        # Try to run download_datasets
        import subprocess
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "download_datasets.py")],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            logger.error("Failed to create sample data")
            logger.error(result.stderr)
            sys.exit(1)
    
    # Ingest PDFs
    logger.info("\n--- Ingesting PDF Reports ---")
    pdf_count = 0
    pdf_vectors = 0
    for pdf_file in pdf_dir.glob("*.pdf"):
        try:
            result = ingestor.ingest_pdf(
                str(pdf_file),
                document_id=pdf_file.stem,
                source_type="government_document"
            )
            if result.success:
                pdf_count += 1
                pdf_vectors += result.vectors_added
                logger.info(f"  ✓ {pdf_file.name}: {result.vectors_added} vectors")
            else:
                logger.warning(f"  ○ {pdf_file.name}: {result.errors}")
        except Exception as e:
            logger.error(f"  ✗ {pdf_file.name}: {e}")
    
    logger.info(f"PDF Summary: {pdf_count} files, {pdf_vectors} vectors")
    
    # Ingest images
    logger.info("\n--- Ingesting Images ---")
    image_count = 0
    image_vectors = 0
    for image_file in image_dir.glob("*.jpg"):
        try:
            # Extract metadata from filename if possible
            media_type = "satellite"
            if "uav" in image_file.name.lower():
                media_type = "uav"
            elif "street" in image_file.name.lower():
                media_type = "street_view"
            
            result = ingestor.ingest_image(
                str(image_file),
                media_type=media_type
            )
            if result.success:
                image_count += 1
                image_vectors += result.vectors_added
                logger.info(f"  ✓ {image_file.name}")
            else:
                logger.warning(f"  ○ {image_file.name}: {result.errors}")
        except Exception as e:
            logger.error(f"  ✗ {image_file.name}: {e}")
    
    logger.info(f"Image Summary: {image_count} files, {image_vectors} vectors")
    
    # Ingest audio - use transcript metadata if Whisper fails on synthetic audio
    logger.info("\n--- Ingesting Audio Files ---")
    audio_count = 0
    audio_vectors = 0
    
    # Load transcript metadata if available
    transcript_metadata = {}
    transcript_json = audio_dir / "transcripts_metadata.json"
    if transcript_json.exists():
        import json
        with open(transcript_json) as f:
            transcripts = json.load(f)
            # Map by audio file number
            for i, t in enumerate(transcripts):
                transcript_metadata[f"emergency_audio_{i+1:02d}"] = t
    
    for audio_file in list(audio_dir.glob("*.wav")) + list(audio_dir.glob("*.mp3")):
        try:
            source = "emergency_call" if "emergency" in audio_file.name.lower() else "rescuer_radio"
            
            result = ingestor.ingest_audio(
                str(audio_file),
                source=source
            )
            
            # If transcription failed, try using metadata
            if not result.success and audio_file.stem in transcript_metadata:
                meta = transcript_metadata[audio_file.stem]
                logger.info(f"  Using transcript metadata for {audio_file.name}")
                
                # Directly embed the transcript text
                from qdrant_client.models import PointStruct
                from datetime import datetime
                import uuid
                
                transcript = meta.get("text", "")
                if transcript:
                    embedding = list(ingestor.text_embedder.embed([transcript]))[0].tolist()
                    
                    payload = {
                        "transcript": transcript,
                        "duration_seconds": meta.get("duration", 30),
                        "stress_level": meta.get("stress_level", 0.5),
                        "language": "en",
                        "source": meta.get("source", source),
                        "timestamp": datetime.utcnow().isoformat(),
                        "file_name": audio_file.name
                    }
                    
                    point = PointStruct(
                        id=int(uuid.uuid4().int & (1 << 63) - 1),
                        vector=embedding,
                        payload=payload
                    )
                    
                    memory.get_client().upsert(
                        collection_name="sentinel_audio",
                        points=[point],
                        wait=True
                    )
                    
                    audio_count += 1
                    audio_vectors += 1
                    logger.info(f"  ✓ {audio_file.name} (from metadata)")
                    continue
            
            if result.success:
                audio_count += 1
                audio_vectors += result.vectors_added
                logger.info(f"  ✓ {audio_file.name}")
            else:
                logger.warning(f"  ○ {audio_file.name}: {result.errors}")
        except Exception as e:
            logger.error(f"  ✗ {audio_file.name}: {e}")
    
    logger.info(f"Audio Summary: {audio_count} files, {audio_vectors} vectors")
    
    # Get final stats
    logger.info("\n--- Collection Statistics ---")
    stats = memory.get_collection_stats()
    total_vectors = 0
    for name, info in stats.items():
        count = info.get("vectors_count", 0)
        total_vectors += count
        status = info.get("status", "unknown")
        logger.info(f"  {name}: {count} vectors ({status})")
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Database Initialization Complete!")
    logger.info("=" * 60)
    logger.info(f"Total Vectors: {total_vectors}")
    logger.info(f"  - Semantic: {stats.get('sentinel_semantic', {}).get('vectors_count', 0)}")
    logger.info(f"  - Episodic: {stats.get('sentinel_episodic', {}).get('vectors_count', 0)}")
    logger.info(f"  - Audio:    {stats.get('sentinel_audio', {}).get('vectors_count', 0)}")
    logger.info("\nNext step: Run 'streamlit run app/ui/dashboard.py'")


if __name__ == "__main__":
    main()

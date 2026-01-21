#!/usr/bin/env python3
"""
Project Sentinel V2 - Real Data Ingestor

Ingests the downloaded real-world disaster data into Qdrant.
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import settings
from app.core.memory_engine import MemoryEngine
from app.ingestion.universal_ingestor import UniversalIngestor
from app.llm.perplexity_integration import PerplexityIntegration

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    logger.info("=" * 60)
    logger.info("Project Sentinel V2 - Real Data Ingestor")
    logger.info("=" * 60)
    
    # Initialize components
    logger.info("Initializing system components...")
    
    try:
        memory = MemoryEngine(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
            api_key=settings.QDRANT_API_KEY,
            text_vector_size=settings.TEXT_VECTOR_SIZE,
            vision_vector_size=settings.VISION_VECTOR_SIZE,
            audio_vector_size=settings.AUDIO_VECTOR_SIZE
        )
        # Ensure collections exist
        memory.initialize_memory()
        
        # Initialize LLM (optional, for better image analysis)
        llm = PerplexityIntegration(
            api_key=settings.PERPLEXITY_API_KEY,
            model=settings.PERPLEXITY_MODEL,
            offline_mode=settings.OFFLINE_MODE
        )
        
        ingestor = UniversalIngestor(
            qdrant_client=memory.get_client(),
            text_embedding_model=settings.TEXT_EMBEDDING_MODEL,
            vision_embedding_model=settings.VISION_EMBEDDING_MODEL,
            whisper_model_name=settings.WHISPER_MODEL,
            gemini_client=llm
        )
        
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        sys.exit(1)
        
    # Directories to process
    data_dir = PROJECT_ROOT / "data" / "raw_datasets"
    dirs_to_ingest = [
        data_dir / "images",
        data_dir / "reports"
    ]
    
    total_success = 0
    total_failed = 0
    
    for directory in dirs_to_ingest:
        if not directory.exists():
            logger.warning(f"Directory not found: {directory}")
            continue
            
        logger.info(f"\nProcessing directory: {directory.name}")
        results = ingestor.ingest_directory(
            str(directory),
            recursive=True,
            progress_callback=lambda name, cur, tot: print(f"  [{cur}/{tot}] {name}...", end='\r')
        )
        
        print("") # Newline after progress
        
        logger.info(f"Results for {directory.name}:")
        logger.info(f"  Success: {results['successful']}")
        logger.info(f"  Failed: {results['failed']}")
        
        if results['errors']:
            logger.warning("  Errors:")
            for err in results['errors'][:5]: # Show first 5 errors
                logger.warning(f"    - {err}")
                
        total_success += results['successful']
        total_failed += results['failed']
        
    logger.info("\n" + "=" * 60)
    logger.info(f"Ingestion Complete. Total Processed: {total_success + total_failed}")
    logger.info(f"Success: {total_success}")
    logger.info(f"Failed: {total_failed}")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()

"""
Script to initialize Local Qdrant Storage and seed it with sample data.
"""
import sys
import os
from pathlib import Path
import logging

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Force OFFLINE_MODE to ensure we target local storage
os.environ["OFFLINE_MODE"] = "True"

from app.config import settings
from app.core.memory_engine import MemoryEngine
from app.ingestion.universal_ingestor import UniversalIngestor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def seed_db():
    logger.info(f"Seeding Local DB at: {settings.QDRANT_STORAGE}")
    
    # 1. Initialize Memory Engine (Creates collections)
    # We must ensure dimensions match the Embedding Model (512 for CLIP)
    engine = MemoryEngine(
        host=settings.QDRANT_HOST,
        port=settings.QDRANT_PORT,
        storage_path=settings.QDRANT_STORAGE,
        vision_vector_size=settings.VISION_VECTOR_SIZE # Fix: 512
    )
    
    # FORCE RESET to ensure correct dimensions
    logger.info("Resetting all collections to fix dimensions...")
    engine.reset_all_collections()
    
    results = engine.initialize_memory()
    logger.info(f"Collections initialized: {results}")

    # 2. Ingest Sample Image
    # Look for the generated image in artifacts or current dir
    # The agent might save it to artifacts, let's look for known path or argument
    # For now, we'll try to find any jpg/png in current dir or specific path
    
    sample_img_path = Path("flood_damage_sample.png") 
    if not sample_img_path.exists():
         # Fallback search
         files = list(Path(".").glob("*.png"))
         if files:
             sample_img_path = files[0]
    
    if sample_img_path.exists():
        logger.info(f"Ingesting image: {sample_img_path}")
        
        # We need a client. UniversalIngestor takes a client.
        client = engine.get_client()
        
        ingestor = UniversalIngestor(
            qdrant_client=client,
            text_embedding_model=settings.TEXT_EMBEDDING_MODEL,
            vision_embedding_model=settings.VISION_EMBEDDING_MODEL
        )
        
        res = ingestor.ingest_image(
            file_path=str(sample_img_path),
            media_type="uav_drone",
            extra_metadata={"description": "Aerial view of severe flooding in Sector 7. Stranded vehicle detected.", "damage_score": 0.85}
        )
        logger.info(f"Ingestion result: {res.success}")
    else:
        logger.warning(f"No sample image found at {sample_img_path}")

if __name__ == "__main__":
    seed_db()

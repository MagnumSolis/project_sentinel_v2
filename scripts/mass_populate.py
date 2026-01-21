"""
Script to populate the Qdrant database with MASSIVE diverse multi-modal data.
- Generates 10+ PDFs
- Ingests 8+ Images
- Mocks 20+ Audio files
"""
import sys
import os
import shutil
import uuid
import logging
from pathlib import Path
from datetime import datetime, timedelta
import random

# Force Offline Mode
os.environ["OFFLINE_MODE"] = "True"

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Ensure FPDF is available
try:
    from fpdf import FPDF
except ImportError:
    # If not installed, we can skip PDF gen or fail
    # We'll assume it's installed from previous step
    pass

from app.config import settings
from app.core.memory_engine import MemoryEngine
from app.ingestion.universal_ingestor import UniversalIngestor
from qdrant_client.models import PointStruct

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Mock Data Generation Helpers ---

LOCATIONS = ["Sector 1 (Downtown)", "Sector 2 (Industrial)", "Sector 3 (Residential)", "Sector 4 (Riverside)", "Sector 7 (Hilltop)", "Highway 9"]
STATUSES = ["Critical", "Stable", "Urgent", "Cleared", "Unknown"]
RESOURCES = ["Water", "Food", "Medical Kits", "Blankets", "Fuel", "Generators"]

def generate_audio_mock(idx):
    """Generate a random audio transcript."""
    types = ["emergency_call", "radio_chatter", "voice_memo", "news_broadcast"]
    selected_type = random.choice(types)
    location = random.choice(LOCATIONS)
    
    if selected_type == "emergency_call":
        transcript = f"Dispatch, we have a situation at {location}. {random.choice(['Flooding is severe.', 'Building collapse confirmed.', 'Fire spreading.', 'Vehicle trapped.'])} Send help immediately."
        stress = random.uniform(0.7, 1.0)
    elif selected_type == "radio_chatter":
        transcript = f"Unit {random.randint(1, 99)} reporting from {location}. Route is {random.choice(['clear', 'blocked', 'flooded'])}. Proceeding to next waypoint."
        stress = random.uniform(0.1, 0.5)
    elif selected_type == "voice_memo":
        transcript = f"Recording log. We are stuck near {location}. Supplies are low. We need {random.choice(RESOURCES)}."
        stress = random.uniform(0.5, 0.9)
    else:
        transcript = f"BREAKING NEWS: Flash floods reported in {location}. Authorities advise evacuation."
        stress = random.uniform(0.4, 0.7)
        
    return {
        "file_name": f"{selected_type}_{idx:03d}.wav",
        "transcript": transcript,
        "stress_level": stress,
        "language": "en",
        "duration_seconds": random.randint(5, 60),
        "source": selected_type
    }

def generate_pdf_mock(idx):
    """Generate mock PDF content."""
    doc_type = random.choice(["Situation Report", "Supply Manifest", "Casualty List", "Medical Log"])
    date_str = (datetime.now() - timedelta(hours=random.randint(0, 48))).strftime("%Y-%m-%d %H:%M")
    
    content = f"""
    {doc_type.upper()}
    Date: {date_str}
    Location: {random.choice(LOCATIONS)}
    
    Summary:
    - Status: {random.choice(STATUSES)}
    - Active Units: {random.randint(1, 20)}
    
    Details:
    This document serves as an official record for {doc_type}. 
    Resources requested: {', '.join(random.sample(RESOURCES, 2))}.
    
    Notes:
    {random.choice(['Power grid offline.', 'Roads accessible.', 'Heavy rain continuing.', 'Shelter capacity full.'])}
    """
    
    return {
        "filename": f"{doc_type.lower().replace(' ', '_')}_{idx:03d}.pdf",
        "title": f"{doc_type} #{idx}",
        "content": content
    }

def make_pdf(doc_data, output_dir):
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        for line in doc_data["content"].split('\n'):
            # simple sanitization
            line = line.encode('latin-1', 'replace').decode('latin-1')
            pdf.cell(200, 10, txt=line.strip(), ln=1, align='L')
        output_path = output_dir / doc_data["filename"]
        pdf.output(str(output_path))
        return output_path
    except Exception as e:
        logger.error(f"Failed to generate PDF: {e}")
        return None

def run_mass_population():
    logger.info("Starting MASS Database Population...")
    
    # Init Engine
    engine = MemoryEngine(
        host=settings.QDRANT_HOST,
        port=settings.QDRANT_PORT,
        storage_path=settings.QDRANT_STORAGE,
        vision_vector_size=settings.VISION_VECTOR_SIZE 
    )
    client = engine.get_client()
    
    ingestor = UniversalIngestor(
        qdrant_client=client,
        text_embedding_model=settings.TEXT_EMBEDDING_MODEL,
        vision_embedding_model=settings.VISION_EMBEDDING_MODEL
    )
    
    # 1. Ingest ALL Images in data/raw_datasets/images
    img_dir = PROJECT_ROOT / "data" / "raw_datasets" / "images"
    if img_dir.exists():
        images = list(img_dir.glob("*.png"))
        logger.info(f"Found {len(images)} images to ingest.")
        for img_file in images:
            logger.info(f"Ingesting Image: {img_file.name}")
            try:
                ingestor.ingest_image(
                    file_path=str(img_file),
                    media_type="uav_drone",
                    extra_metadata={"description": f"Scenario: {img_file.stem.replace('_', ' ')}"}
                )
            except Exception as e:
                logger.error(f"Failed to ingest image {img_file.name}: {e}")

    # 2. Ingest Audio Mocks (20 items)
    logger.info("Injecting 20 Mock Audio Records...")
    for i in range(20):
        audio = generate_audio_mock(i)
        embedding = list(ingestor.text_embedder.embed([audio["transcript"]]))[0].tolist()
        payload = {
            "transcript": audio["transcript"],
            "duration_seconds": audio["duration_seconds"],
            "stress_level": audio["stress_level"],
            "language": audio["language"],
            "source": audio["source"],
            "timestamp": datetime.utcnow().isoformat(),
            "file_name": audio["file_name"],
            "file_path": f"/mock/audio/{audio['file_name']}" 
        }
        point = PointStruct(
            id=ingestor._generate_id(),
            vector=embedding,
            payload=payload
        )
        client.upsert(collection_name="sentinel_audio", points=[point])
        
    # 3. Ingest Docs (10 items)
    logger.info("Generating and Ingesting 10 PDF Reports...")
    doc_dir = PROJECT_ROOT / "data" / "raw_datasets" / "documents"
    doc_dir.mkdir(parents=True, exist_ok=True)
    
    for i in range(10):
        doc_data = generate_pdf_mock(i)
        pdf_path = make_pdf(doc_data, doc_dir)
        if pdf_path:
            logger.info(f"Ingesting PDF: {pdf_path.name}")
            ingestor.ingest_pdf(str(pdf_path))
            
    logger.info("MASS Population Complete!")

if __name__ == "__main__":
    run_mass_population()

"""
Script to populate the Qdrant database with diverse multi-modal data.
- Generates PDFs
- Ingests Images
- Mocks Audio ingestion (since we can't generate real audio files easily)
"""
import sys
import os
import shutil
import uuid
import logging
from pathlib import Path
from datetime import datetime, timedelta

# Force Offline Mode
os.environ["OFFLINE_MODE"] = "True"

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import settings
from app.core.memory_engine import MemoryEngine
from app.ingestion.universal_ingestor import UniversalIngestor
from qdrant_client.models import PointStruct

# Import fpdf for PDF generation
try:
    from fpdf import FPDF
except ImportError:
    print("Please install fpdf: pip install fpdf")
    sys.exit(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Mock Data Definitions ---

AUDIO_MOCKS = [
    {
        "file_name": "emergency_call_001.wav",
        "transcript": "Please help! We are trapped on the second floor of the apartment complex on 5th Street. The water is rising fast!",
        "stress_level": 0.95,
        "language": "en",
        "duration_seconds": 15.0,
        "source": "911_call"
    },
    {
        "file_name": "radio_chatter_alpha.wav",
        "transcript": "Base, this is Alpha Team. We have cleared Sector 4. No casualties found. Proceeding to Sector 5.",
        "stress_level": 0.3,
        "language": "en",
        "duration_seconds": 8.0,
        "source": "radio"
    },
    {
        "file_name": "survivor_voice_memo.wav",
        "transcript": "I don't know if anyone can hear this. My leg is broken. I'm under the rubble near the old library.",
        "stress_level": 0.98,
        "language": "en",
        "duration_seconds": 22.0,
        "source": "upload"
    }
]

DOCUMENTS = [
    {
        "filename": "situation_report_day3.pdf",
        "title": "Situation Report - Day 3",
        "content": """
        EMERGENCY SITUATION REPORT - DAY 3
        
        Summary:
        Flood waters have begun to recede in the northern districts, but the downtown area remains critical.
        Power outages affect 40% of the grid. 
        
        Casualties:
        - Confirmed D.O.A: 12
        - Missing: 45
        - Injured: 120
        
        Critical Infrastructure:
        - General Hospital: Running on generators.
        - Bridge 4: Collapsed.
        - Highway 9: Blocked by debris.
        
        Priorities:
        1. Search and Rescue in Sector 7.
        2. Food and Water distribution at Stadium.
        """
    },
    {
        "filename": "medical_triage_log.pdf",
        "title": "Field Hospital Triage Log",
        "content": """
        FIELD HOSPITAL TRIAGE LOG
        Date: 2026-01-21
        
        Patient A1: Male, 45. Broken Tibia. Stable.
        Patient A2: Female, 8. Hypothermia. Critical.
        Patient A3: Male, 70. Diabetic Shock. Stable after insulin.
        
        Supply Status:
        - Antibiotics: Low
        - Bandages: Moderate
        - Water: Critical
        """
    }
]

def generate_pdf(doc_data, output_dir):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # Simple clean text formatting
    for line in doc_data["content"].split('\n'):
        pdf.cell(200, 10, txt=line.strip(), ln=1, align='L')
        
    output_path = output_dir / doc_data["filename"]
    pdf.output(str(output_path))
    return output_path

def run_population():
    logger.info("Starting Database Population...")
    
    # Initialize Engine
    engine = MemoryEngine(
        host=settings.QDRANT_HOST,
        port=settings.QDRANT_PORT,
        storage_path=settings.QDRANT_STORAGE,
        vision_vector_size=settings.VISION_VECTOR_SIZE # 512
    )
    
    # Ensure collections exist
    engine.initialize_memory()
    client = engine.get_client()
    
    # Initialize Ingestor
    ingestor = UniversalIngestor(
        qdrant_client=client,
        text_embedding_model=settings.TEXT_EMBEDDING_MODEL,
        vision_embedding_model=settings.VISION_EMBEDDING_MODEL
    )
    
    # 1. Process Images
    img_dir = PROJECT_ROOT / "data" / "raw_datasets" / "images"
    if img_dir.exists():
        for img_file in img_dir.glob("*.png"):
            logger.info(f"Ingesting Image: {img_file.name}")
            ingestor.ingest_image(
                file_path=str(img_file),
                media_type="uav_drone",
                extra_metadata={"description": f"Auto-generated scenario: {img_file.stem}"}
            )
            
    # 2. Process Documents
    doc_dir = PROJECT_ROOT / "data" / "raw_datasets" / "documents"
    doc_dir.mkdir(parents=True, exist_ok=True)
    
    for doc in DOCUMENTS:
        logger.info(f"Generating PDF: {doc['filename']}")
        pdf_path = generate_pdf(doc, doc_dir)
        
        logger.info(f"Ingesting PDF: {doc['filename']}")
        # We prefer using the PDF ingestor to test chunking
        ingestor.ingest_pdf(str(pdf_path))
        
    # 3. Process Audio (Mock)
    # Since we don't have real audio files + whisper, we manually inject vectors
    logger.info("Injecting Mock Audio Data...")
    
    for audio in AUDIO_MOCKS:
        # Generate embedding for transcript
        embedding = list(ingestor.text_embedder.embed([audio["transcript"]]))[0].tolist()
        
        payload = {
            "transcript": audio["transcript"],
            "duration_seconds": audio["duration_seconds"],
            "stress_level": audio["stress_level"],
            "language": audio["language"],
            "source": audio["source"],
            "timestamp": datetime.utcnow().isoformat(),
            "file_name": audio["file_name"],
            # No actual file path as it's a mock
            "file_path": f"/mock/audio/{audio['file_name']}" 
        }
        
        point = PointStruct(
            id=ingestor._generate_id(),
            vector=embedding,
            payload=payload
        )
        
        client.upsert(
            collection_name="sentinel_audio",
            points=[point]
        )
        logger.info(f"✓ Injected audio mock: {audio['file_name']}")
        
    logger.info("Database Population Complete!")

if __name__ == "__main__":
    run_population()

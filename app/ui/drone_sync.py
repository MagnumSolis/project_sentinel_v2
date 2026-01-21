"""
Drone Sync Simulation Handler
-----------------------------
Manages the simulation of connecting to a drone network and downloading "off-grid" data.
"""

import time
import os
import streamlit as st
from pathlib import Path

# We assume this path is populated with "secret" files
DRONE_CACHE_DIR = Path("data/drone_cache")
PROCESSED_HISTORY = Path("data/processed/drone_sync_history.txt")

def check_drone_status():
    """Simulate checking for available drones."""
    # Always pretend a drone is nearby if we have data in the cache that hasn't been synced
    if not DRONE_CACHE_DIR.exists():
        return {"status": "scanning", "message": "Scanning frequencies..."}
    
    files = list(DRONE_CACHE_DIR.glob("*.png"))
    if not files:
        return {"status": "scanning", "message": "No active signals."}
        
    return {
        "status": "available",
        "drone_id": "BRAVO-7-QUARANTINE",
        "signal_strength": "86%",
        "packet_size": f"{len(files) * 2.4:.1f} MB",
        "file_count": len(files)
    }

def perform_uplink(components):
    """
    Connects to the drone and ingests data.
    """
    # Create ingestor on the fly
    from app.ingestion.universal_ingestor import UniversalIngestor
    from app.config import settings
    
    ingestor = UniversalIngestor(
        components['memory'].get_client(),
        settings.TEXT_EMBEDDING_MODEL,
        settings.VISION_EMBEDDING_MODEL,
        settings.WHISPER_MODEL,
        components['llm']
    )
    
    files = list(DRONE_CACHE_DIR.glob("*.png"))
    progress_bar = st.progress(0, text="Establishing Handshake...")
    status_text = st.empty()
    
    for i, file_path in enumerate(files):
        # Simulate transfer time
        time.sleep(0.8) 
        
        status_text.text(f"Downloading & Decrypting: {file_path.name}...")
        progress_bar.progress((i / len(files)) * 0.5, text="Downloading...")
        
        # Ingest
        status_text.text(f"Ingesting Vector: {file_path.name}...")
        try:
            # We copy to raw first to simulate "download"
            target_path = Path(settings.RAW_DATASETS_DIR) / "images" / file_path.name
            with open(file_path, "rb") as src, open(target_path, "wb") as dst:
                dst.write(src.read())
            
            # Ingest from the new location
            ingestor.ingest_image(
                str(target_path), 
                media_type="drone_recon",
                extra_metadata={"source": "Drone Bravo-7", "classification": "RESTRICTED", "sector": "Sector 9"}
            )
        except Exception as e:
            st.error(f"Failed to ingest {file_path.name}: {e}")
            
        progress_bar.progress(0.5 + ((i + 1) / len(files)) * 0.5, text="Indexing...")

    # Clear cache or mark as done (we'll just move them to a 'synced' folder in cache to hide them next time?)
    # For repeated demos, we might NOT want to delete them. 
    # But to make it "demonstrable change", we should ensure we don't re-ingest the same thing 100 times.
    # Let's simple move them to a 'synced' subfolder
    synced_dir = DRONE_CACHE_DIR / "synced"
    synced_dir.mkdir(exist_ok=True)
    for f in files:
        try:
            os.rename(str(f), str(synced_dir / f.name))
        except:
            pass
            
    progress_bar.progress(1.0, text="Sync Complete.")
    status_text.success("All Intelligence Vectors Integrated.")
    time.sleep(1)
    st.rerun()

def reset_simulation():
    """
    Resets the simulation by moving synced files back to the cache directory.
    This allows the demo to be run again.
    """
    synced_dir = DRONE_CACHE_DIR / "synced"
    if not synced_dir.exists():
        return False
        
    files = list(synced_dir.glob("*.png"))
    if not files:
        return False
        
    count = 0
    for f in files:
        try:
            # Move back to parent directory (drone_cache)
            os.rename(str(f), str(DRONE_CACHE_DIR / f.name))
            count += 1
        except Exception as e:
            print(f"Error resetting file {f.name}: {e}")
            
    return count > 0

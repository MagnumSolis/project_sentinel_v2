"""
Script to wipe all collections and re-seed fresh.
"""
import sys
import os
from pathlib import Path
import shutil

# Force Offline Mode - target local storage cleanly
os.environ["OFFLINE_MODE"] = "True"

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import settings

def clean_and_seed():
    print(f"Target Storage: {settings.QDRANT_STORAGE}")
    
    # Nuclear Option: Delete the storage folder physically to remove ANY zombie data
    # (Qdrant Local is file based)
    storage_path = Path(settings.QDRANT_STORAGE)
    if storage_path.exists():
        print(f"Removing physical directory: {storage_path}")
        shutil.rmtree(storage_path)
    
    # Re-run seed script logic (importing it)
    from scripts.seed_local_db import seed_db
    seed_db()

if __name__ == "__main__":
    clean_and_seed()

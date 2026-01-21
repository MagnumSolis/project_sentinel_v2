"""
Script to inspect Qdrant data in Local Mode.
"""
import sys
import os
from pathlib import Path
from qdrant_client import QdrantClient

# Force Offline Mode
os.environ["OFFLINE_MODE"] = "True"

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import settings

def inspect_data():
    client = QdrantClient(path=settings.QDRANT_STORAGE)
    
    print(f"Connected to {settings.QDRANT_STORAGE}")
    
    try:
        # Check count
        count = client.get_collection("sentinel_episodic").points_count
        print(f"sentinel_episodic count: {count}")
        
        if count > 0:
            # Scroll to see payload
            points, _ = client.scroll(
                collection_name="sentinel_episodic",
                limit=5,
                with_payload=True,
                with_vectors=False
            )
            
            for p in points:
                print(f"--- Point {p.id} ---")
                keys = list(p.payload.keys())
                print(f"Payload Keys: {keys}")
                if "thumbnail_b64" in keys:
                    print(f"Thumbnail: Present (len={len(p.payload['thumbnail_b64'])})")
                else:
                    print("Thumbnail: MISSING")
                print(f"File Path: {p.payload.get('file_path')}")
                
        else:
            print("Collection is empty!")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    inspect_data()

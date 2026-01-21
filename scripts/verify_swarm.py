"""
Script to verify Swarm-Sync Protocol.
Creates two local Qdrant instances (nodes) and simulates a sync.
"""

import os
import sys
import shutil
from pathlib import Path
import logging
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.core.swarm import SwarmNode

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_swarm():
    # Setup temp directories for two nodes
    node1_dir = PROJECT_ROOT / "data" / "node1_storage"
    node2_dir = PROJECT_ROOT / "data" / "node2_storage"
    
    # Clean up old runs
    if node1_dir.exists(): shutil.rmtree(node1_dir)
    if node2_dir.exists(): shutil.rmtree(node2_dir)
    
    node1_dir.mkdir(parents=True)
    node2_dir.mkdir(parents=True)
    
    logger.info("Initializing Node 1 and Node 2...")
    
    # Init Clients
    client1 = QdrantClient(path=str(node1_dir))
    client2 = QdrantClient(path=str(node2_dir))
    
    collection = "test_swarm"
    
    # Create Collections
    for client in [client1, client2]:
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=4, distance=Distance.COSINE)
        )
    
    # Add Data to Node 1 ONLY
    logger.info("Adding data to Node 1...")
    points = [
        PointStruct(id=1, vector=[0.1, 0.1, 0.1, 0.1], payload={"info": "drone1_data"}),
        PointStruct(id=2, vector=[0.2, 0.2, 0.2, 0.2], payload={"info": "drone1_data"})
    ]
    client1.upsert(collection_name=collection, points=points)
    
    # Init Swarm Nodes
    node1 = SwarmNode("node_1", client1, collection)
    node2 = SwarmNode("node_2", client2, collection)
    
    # 1. Node 1 generates Sync Packet (State advertising)
    packet1 = node1.generate_sync_packet()
    logger.info(f"Node 1 Packet Merkle Root: {packet1.merkle_root}")
    
    # 2. Node 2 receives Packet from Node 1
    # It should realize it is missing IDs 1 and 2
    missing_ids = node2.receive_sync_packet(packet1)
    logger.info(f"Node 2 missing IDs: {missing_ids}")
    
    if not missing_ids:
        logger.error("❌ Node 2 failed to detect missing vectors!")
        sys.exit(1)
        
    # 3. Simulate Request/Response (Fetch data from Node 1)
    # Node 2 asks Node 1 for specific IDs
    missing_data = node2.get_missing_vectors_from_peer(client1, missing_ids)
    logger.info(f"Fetched {len(missing_data)} vectors from Node 1")
    
    # 4. Merge into Node 2
    count = node2.merge_vectors(missing_data)
    
    # Verify Node 2 now has the data
    info2 = client2.get_collection(collection)
    logger.info(f"Node 2 Vector Count: {info2.points_count}")
    
    if info2.points_count == 2:
        logger.info("✅ SUCCESS: Swarm Sync Verified!")
    else:
        logger.error("❌ FAILURE: Node 2 data count mismatch")

if __name__ == "__main__":
    verify_swarm()

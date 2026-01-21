"""
Project Sentinel V2 - Swarm-Sync Protocol

Handles decentralized synchronization of vector indices between drones (nodes).
Uses Merkle Trees for efficient difference detection.
"""

import hashlib
import json
import logging
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
from datetime import datetime

# Attempt to import merkle_json, fallback if not installed yet
try:
    from merkle_json import MerkleJson
except ImportError:
    MerkleJson = None

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchValue

logger = logging.getLogger(__name__)

@dataclass
class SyncPacket:
    """Data packet exchanged during synchronization."""
    node_id: str
    timestamp: str
    merkle_root: str
    vector_ids: List[int]  # List of IDs this node has (simplified for hackathon)
    # In a real system, we'd exchange Merkle branches, not full ID lists

class SwarmNode:
    """
    Represents a single node in the swarm (a drone).
    """
    
    def __init__(self, node_id: str, client: QdrantClient, collection_name: str):
        self.node_id = node_id
        self.client = client
        self.collection_name = collection_name
        self.known_peers: Set[str] = set()
        self.last_sync: Dict[str, datetime] = {}
        
        if MerkleJson is None:
            logger.warning("merkle-json not installed. Swarm sync will be limited.")
            self.mj = None
        else:
            self.mj = MerkleJson()

    def _get_local_vector_ids(self) -> List[int]:
        """Fetch all vector IDs from the local Qdrant collection."""
        try:
            # Scroll through all points to get IDs
            # Note: In production, we'd maintain this list separately to avoid full scan
            points, _ = self.client.scroll(
                collection_name=self.collection_name,
                limit=10000,  # Hackathon limit
                with_payload=False,
                with_vectors=False
            )
            return sorted([p.id for p in points])
        except Exception as e:
            logger.error(f"Failed to fetch local vector IDs: {e}")
            return []

    def generate_sync_packet(self) -> SyncPacket:
        """Generate a synchronization packet to broadcast."""
        ids = self._get_local_vector_ids()
        
        # Calculate Merkle Root
        merkle_root = ""
        if self.mj:
            # Structure data for Merkle tree (hashing IDs)
            data = {"ids": ids}
            merkle_root = self.mj.hash(data)
        else:
            # Fallback simple hash
            merkle_root = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
            
        return SyncPacket(
            node_id=self.node_id,
            timestamp=datetime.utcnow().isoformat(),
            merkle_root=merkle_root,
            vector_ids=ids
        )

    def receive_sync_packet(self, packet: SyncPacket) -> List[int]:
        """
        Process a received packet.
        Returns a list of Vector IDs that this node is MISSING (needs to request).
        """
        if packet.node_id == self.node_id:
            return []
            
        local_ids = set(self._get_local_vector_ids())
        remote_ids = set(packet.vector_ids)
        
        missing_ids = list(remote_ids - local_ids)
        
        logger.info(f"Sync with {packet.node_id}: Found {len(missing_ids)} missing vectors.")
        self.last_sync[packet.node_id] = datetime.utcnow()
        
        return missing_ids

    def merge_vectors(self, vectors: List[PointStruct]) -> int:
        """
        Merge received vectors into local store.
        Returns count of successfully merged vectors.
        """
        if not vectors:
            return 0
            
        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=vectors
            )
            logger.info(f"Merged {len(vectors)} vectors from swarm.")
            return len(vectors)
        except Exception as e:
            logger.error(f"Merge failed: {e}")
            return 0

    def get_missing_vectors_from_peer(self, peer_client: QdrantClient, missing_ids: List[int]) -> List[PointStruct]:
        """
        Simulate fetching actual vector data from a peer for specific IDs.
        In reality, this would be a network request.
        """
        if not missing_ids:
            return []
            
        try:
            vectors = peer_client.retrieve(
                collection_name=self.collection_name,
                ids=missing_ids,
                with_payload=True,
                with_vectors=True
            )
            return vectors
        except Exception as e:
            logger.error(f"Failed to fetch vectors from peer: {e}")
            return []

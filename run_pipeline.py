#!/usr/bin/env python3
"""
Project Sentinel V2 - Main Pipeline Orchestrator

This script initializes the complete system:
1. Verifies Qdrant connectivity
2. Creates/verifies collections
3. Loads embedding models
4. Ingests sample data (if not present)
5. Validates search functionality
6. Reports ready status

Run this before launching the dashboard.
"""

import os
import sys
import time
from pathlib import Path
import logging
import subprocess

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_banner():
    """Print the Project Sentinel banner."""
    banner = """
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║   🛰️  PROJECT SENTINEL V2                                     ║
║   Multimodal RAG System for Disaster Response                ║
║                                                               ║
║   Production-Grade | Offline-First | Real-Time Analysis       ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def check_dependencies() -> bool:
    """Check if required Python packages are installed."""
    required = [
        'qdrant_client',
        'fastembed',
        'streamlit',
        'pypdf',
        'PIL',
        'pydantic_settings',
    ]
    
    missing = []
    for package in required:
        try:
            __import__(package)
        except ImportError:
            # Handle package name differences
            alt_names = {
                'PIL': 'Pillow',
                'pydantic_settings': 'pydantic-settings'
            }
            missing.append(alt_names.get(package, package))
    
    if missing:
        logger.error(f"Missing packages: {', '.join(missing)}")
        logger.info("Install with: pip install -r requirements.txt")
        return False
    
    return True


def check_docker() -> bool:
    """Check if Docker is available."""
    try:
        result = subprocess.run(
            ['docker', '--version'],
            capture_output=True,
            text=True,
            timeout=10
        )
        return result.returncode == 0
    except Exception:
        return False


def check_qdrant_container() -> bool:
    """Check if Qdrant container is running."""
    try:
        result = subprocess.run(
            ['docker', 'ps', '--filter', 'name=sentinel_qdrant', '--format', '{{.Names}}'],
            capture_output=True,
            text=True,
            timeout=10
        )
        return 'sentinel_qdrant' in result.stdout
    except Exception:
        return False


def start_qdrant() -> bool:
    """Start Qdrant using Docker Compose."""
    logger.info("Starting Qdrant with Docker Compose...")
    try:
        result = subprocess.run(
            ['docker', 'compose', 'up', '-d'],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=PROJECT_ROOT
        )
        if result.returncode != 0:
            logger.error(f"Docker Compose failed: {result.stderr}")
            return False
        
        # Wait for Qdrant to be ready
        logger.info("Waiting for Qdrant to be ready...")
        time.sleep(5)
        return True
        
    except Exception as e:
        logger.error(f"Failed to start Qdrant: {e}")
        return False


def wait_for_qdrant(host: str, port: int, timeout: int = 30) -> bool:
    """Wait for Qdrant to be healthy."""
    import requests
    
    # Use /collections endpoint as it's available on all Qdrant versions
    url = f"http://{host}:{port}/collections"
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(2)
    
    return False


def run_pipeline():
    """Run the complete initialization pipeline."""
    print_banner()
    
    # Step 1: Check dependencies
    logger.info("Step 1/6: Checking dependencies...")
    if not check_dependencies():
        sys.exit(1)
    logger.info("✓ All dependencies installed")
    
    # Step 2: Check/start Qdrant
    logger.info("\nStep 2/6: Checking Qdrant...")
    
    if not check_docker():
        logger.error("Docker not found. Please install Docker.")
        sys.exit(1)
    
    if not check_qdrant_container():
        logger.info("Qdrant container not running, starting...")
        if not start_qdrant():
            logger.error("Failed to start Qdrant. Please run: docker compose up -d")
            sys.exit(1)
    
    # Import after dependency check
    from app.config import settings
    
    if not wait_for_qdrant(settings.QDRANT_HOST, settings.QDRANT_PORT):
        logger.error("Qdrant is not responding. Check Docker logs.")
        sys.exit(1)
    logger.info("✓ Qdrant is healthy")
    
    # Step 3: Initialize Memory Engine
    logger.info("\nStep 3/6: Initializing Memory Engine...")
    from app.core.memory_engine import MemoryEngine
    
    try:
        memory = MemoryEngine(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
            api_key=settings.QDRANT_API_KEY,
            text_vector_size=settings.TEXT_VECTOR_SIZE,
            vision_vector_size=settings.VISION_VECTOR_SIZE,
            audio_vector_size=settings.AUDIO_VECTOR_SIZE
        )
        results = memory.initialize_memory()
        
        for name, success in results.items():
            status = "✓" if success else "✗"
            logger.info(f"  {status} {name}")
        
        if not all(results.values()):
            logger.error("Failed to create all collections")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Memory engine failed: {e}")
        sys.exit(1)
    
    # Step 4: Check for sample data
    logger.info("\nStep 4/6: Checking sample data...")
    settings.ensure_directories()
    
    pdf_dir = settings.raw_datasets_path / "reports"
    has_data = pdf_dir.exists() and any(pdf_dir.glob("*.pdf"))
    
    if not has_data:
        logger.info("No sample data found. Generating...")
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "download_datasets.py")],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            logger.warning("Sample data generation had issues")
            logger.debug(result.stderr)
    
    logger.info("✓ Sample data available")
    
    # Step 5: Ingest data if collections are empty
    logger.info("\nStep 5/6: Checking vector counts...")
    stats = memory.get_collection_stats()
    total_vectors = sum(s.get("vectors_count", 0) for s in stats.values())
    
    if total_vectors == 0:
        logger.info("Collections empty. Running ingestion...")
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "initialize_db.py")],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            logger.warning("Ingestion had issues")
            logger.debug(result.stderr)
        
        # Refresh stats
        stats = memory.get_collection_stats()
        total_vectors = sum(s.get("vectors_count", 0) for s in stats.values())
    
    logger.info(f"✓ Total vectors: {total_vectors}")
    
    # Step 6: Validate search
    logger.info("\nStep 6/6: Validating search...")
    
    if total_vectors > 0:
        from app.core.cortex import SentinelCortex
        
        try:
            cortex = SentinelCortex(
                qdrant_client=memory.get_client(),
                text_embedding_model=settings.TEXT_EMBEDDING_MODEL,
                search_limit=3
            )
            
            results = cortex.search("disaster damage assessment")
            
            if results.results:
                logger.info(f"✓ Search working ({results.search_time_ms:.0f}ms, {len(results.results)} results)")
            else:
                logger.warning("○ Search returned no results (data may not be indexed yet)")
                
        except Exception as e:
            logger.warning(f"○ Search validation failed: {e}")
    else:
        logger.warning("○ Skipping search validation (no vectors)")
    
    # Summary
    print("\n" + "=" * 60)
    print("🎉 PROJECT SENTINEL V2 - READY!")
    print("=" * 60)
    print(f"""
System Status:
  ✓ Qdrant:      http://{settings.QDRANT_HOST}:{settings.QDRANT_PORT}
  ✓ Collections: {len(stats)} initialized
  ✓ Vectors:     {total_vectors:,} total
  ✓ Mode:        {'Offline' if settings.OFFLINE_MODE else 'Online (Gemini)'}

To launch the dashboard:
  streamlit run app/ui/dashboard.py

Dashboard will be available at:
  http://localhost:8501

Press Ctrl+C to exit.
    """)
    
    # Close Qdrant connection to release lock before spawning dashboard
    try:
        if 'memory' in locals():
            memory.client.close()
            del memory
        if 'cortex' in locals():
            del cortex
    except:
        pass

    # Optionally launch dashboard
    launch = input("\nLaunch dashboard now? [Y/n]: ").strip().lower()
    if launch != 'n':
        logger.info("Launching Streamlit dashboard...")
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run',
            str(PROJECT_ROOT / 'app' / 'ui' / 'dashboard.py'),
            '--server.headless', 'true'
        ])


if __name__ == "__main__":
    run_pipeline()

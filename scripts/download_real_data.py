#!/usr/bin/env python3
"""
Project Sentinel V2 - Real Data Downloader

Downloads real-world disaster data from public sources (Wikimedia Commons, NASA, etc.)
to replace synthetic samples.
"""

import os
import sys
import requests
import logging
from pathlib import Path
from urllib.parse import urlparse

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Real-world data sources (Wikimedia Commons Filenames)
WIKI_FILES = [
    {"filename": "File:Kerala flood 2018.jpg", "name": "kerala_flood_2018_real.jpg"},
    {"filename": "File:2010 Haiti earthquake damage2.jpg", "name": "haiti_earthquake_damage_real.jpg"},
    {"filename": "File:Camp Fire - California 2018.jpg", "name": "camp_fire_california_real.jpg"},
    {"filename": "File:Hurricane Harvey flood rescue 2017.jpg", "name": "harvey_flood_rescue_real.jpg"},
    {"filename": "File:2011 Tohoku earthquake and tsunami damage in Oirase.jpg", "name": "japan_tsunami_damage_real.jpg"},
    {"filename": "File:Cyclone Idai damage in Beira, Mozambique.jpg", "name": "cyclone_idai_damage_real.jpg"}
]

def get_wiki_image_url(filename: str) -> str:
    """Get direct image URL from Wikimedia API."""
    try:
        api_url = "https://commons.wikimedia.org/w/api.php"
        params = {
            "action": "query",
            "titles": filename,
            "prop": "imageinfo",
            "iiprop": "url",
            "format": "json"
        }
        headers = {
            'User-Agent': 'ProjectSentinelBot/1.0 (https://github.com/project-sentinel; contact@example.com)'
        }
        response = requests.get(api_url, params=params, headers=headers, timeout=10)
        data = response.json()
        
        pages = data.get("query", {}).get("pages", {})
        for _, page in pages.items():
            if "imageinfo" in page:
                return page["imageinfo"][0]["url"]
    except Exception as e:
        logger.warning(f"Failed to resolve URL for {filename}: {e}")
    return None

def download_file(url: str, output_path: Path) -> bool:
    """Download a file with simple retry logic."""
    try:
        logger.info(f"Downloading {url}...")
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        response = requests.get(url, stream=True, timeout=30, headers=headers)
        
        if response.status_code != 200:
            logger.warning(f"Failed to download {url}: Status {response.status_code}")
            return False
            
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                
        return True
    except Exception as e:
        logger.error(f"Error downloading {url}: {e}")
        return False

def main():
    logger.info("=" * 60)
    logger.info("Project Sentinel V2 - Real Data Downloader")
    logger.info("=" * 60)
    
    data_dir = PROJECT_ROOT / "data" / "raw_datasets"
    
    # Process Images
    image_dir = data_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("\n--- Downloading Real Imagery (via Wikimedia API) ---")
    for item in WIKI_FILES:
        output_path = image_dir / item["name"]
        if output_path.exists():
            logger.info(f"File already exists: {item['name']}")
            continue
            
        url = get_wiki_image_url(item["filename"])
        if url:
            if download_file(url, output_path):
                logger.info(f"✓ Saved: {item['name']}")
            else:
                logger.error(f"✗ Failed download: {item['name']}")
        else:
            logger.error(f"✗ Could not resolve URL for: {item['filename']}")

    # Process Reports
    report_dir = data_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("\n--- Downloading Real Reports ---")
    # Using specific stable URLs
    real_reports = [
         ("https://www.fema.gov/sites/default/files/2020-07/fema_hurricane-maria_after-action-report.pdf", "fema_maria_aar_real.pdf"),
         ("https://reliefweb.int/attachments/3ccba97e-131c-3e3c-9149-16675276550b/Situation%20Report%20-%20Turkey%20Syria%20Earthquake%20-%2016%20Feb%202023.pdf", "turkey_syria_report.pdf")
    ]
    
    for url, filename in real_reports:
        output_path = report_dir / filename
        if output_path.exists():
            continue
            
        if download_file(url, output_path):
            logger.info(f"✓ Saved: {filename}")
        else:
             logger.warning(f"Could not download real report {filename}")

if __name__ == "__main__":
    main()

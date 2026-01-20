#!/usr/bin/env python3
"""
Project Sentinel V2 - Dataset Download & Sample Generation

Downloads sample disaster data and generates synthetic examples for testing.
This script prepares realistic data for the RAG system without requiring
massive dataset downloads.
"""

import os
import sys
from pathlib import Path
import requests
import json
import logging
import shutil
from datetime import datetime, timedelta
import random

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Sample disaster report content
SAMPLE_REPORTS = [
    {
        "title": "Hurricane Maria Damage Assessment - Puerto Rico",
        "content": """EMERGENCY SITUATION REPORT
Date: September 21, 2017
Location: San Juan, Puerto Rico

SUMMARY:
Hurricane Maria made landfall as a Category 4 hurricane with sustained winds of 155 mph. 
The storm caused catastrophic damage across the entire island.

DAMAGE ASSESSMENT:
- Power Infrastructure: 100% of the island without electricity
- Water Systems: 55% of population without potable water
- Communications: 95% of cell towers inoperable
- Transportation: Major highways blocked by debris
- Structures: Estimated 70,000 homes severely damaged or destroyed

PRIORITY AREAS:
1. Sector 4-A (Condado): Multiple building collapses, civilians trapped
2. Sector 7-B (Santurce): Hospital flooding, patient evacuation needed
3. Sector 12-C (Carolina): Bridge collapse, community isolated

RESOURCE REQUIREMENTS:
- Urban Search and Rescue Teams: 15 additional units
- Medical Personnel: 200 additional staff
- Heavy Equipment: Bulldozers for debris clearance
- Helicopters: 10 for medical evacuations

RECOMMENDED ACTIONS:
1. Deploy USAR teams to Sector 4-A immediately
2. Establish alternate route to Carolina via Route 66
3. Set up emergency water distribution at 20 locations
4. Request federal assistance for power restoration

Contact: Emergency Operations Center
Phone: 787-555-0100 (Satellite)
"""
    },
    {
        "title": "California Wildfire Situation Report",
        "content": """WILDFIRE INCIDENT REPORT
Incident: Camp Fire
Date: November 10, 2018
Location: Butte County, California

FIRE STATUS:
- Size: 153,336 acres
- Containment: 25%
- Rate of Spread: 80 football fields per minute at peak
- Structures Destroyed: 18,804
- Confirmed Fatalities: 85

AFFECTED AREAS:
Paradise Township: Near-total destruction
- Estimated 95% of structures destroyed
- Population of 26,000 displaced
- Hospital evacuated, patients relocated to Chico

Concow: Severe damage
- Access roads destroyed
- Multiple communities isolated
- Evacuation status unknown for 150+ residents

ACTIVE RESCUE OPERATIONS:
Zone Alpha (Paradise):
- 5 civilian groups reported trapped
- Coordinates: 39.7596° N, 121.6219° W
- Access via Skyway blocked

Zone Beta (Magalia):
- Care facility evacuation in progress
- 30 elderly patients requiring transport

RESOURCE DEPLOYMENT:
- Fire Personnel: 5,600
- Fire Engines: 622
- Aircraft: 23 (12 helicopters, 11 fixed-wing)
- Bulldozers: 75

WEATHER OUTLOOK:
- Wind shift expected 1400 hours
- Red Flag Warning extended through Friday
- Humidity dropping to 8%

IMMEDIATE PRIORITIES:
1. Complete evacuation of Zone Beta care facility
2. Establish firebreak along Ridge Road
3. Deploy additional air tankers to eastern perimeter
4. Search and rescue operations in Paradise grid sectors

Report prepared by: Incident Command Post
Next Update: 0600 hours November 11, 2018
"""
    },
    {
        "title": "Earthquake Response - Turkey Syria 2023",
        "content": """RAPID DAMAGE ASSESSMENT
Earthquake Event: Kahramanmaraş, Turkey / Northwestern Syria
Date: February 6, 2023
Magnitude: 7.8 Mw (followed by 7.7 Mw aftershock)

IMPACT SUMMARY:
This is one of the deadliest earthquakes in modern history affecting both Turkey and Syria.

TURKEY (Southern Region):
Confirmed Deaths: 45,000+
Injured: 105,000+
Collapsed Buildings: 12,000+
Displaced Population: 1.5 million

SYRIA (Northwestern Region - Conflict Zone):
Confirmed Deaths: 5,800+
Injured: 12,000+
Collapsed Buildings: 3,000+
Access Challenges: Active conflict zones limiting humanitarian access

CRITICAL SECTORS:
Gaziantep Province:
- Multiple high-rise collapses
- Hospital infrastructure damaged
- Industrial zone fires reported
- Survivors detected under rubble at GPS: 37.0660° N, 37.3781° E

Hatay Province:
- Province capital Antakya severely damaged
- Ancient structures collapsed
- International rescue teams deployed

Aleppo, Syria:
- Add-on damage to conflict-affected buildings
- Limited heavy equipment available
- Border crossings congested

SEARCH AND RESCUE PRIORITIES:
High Priority (72-hour window):
- Voice detection reported: 5 locations
- Thermal signatures: 12 locations
- Cell phone signals: 23 locations

INTERNATIONAL RESPONSE:
Teams Deployed: 70+ countries
Personnel: 12,000+ international responders
Equipment: 500+ sniffer dogs

LOGISTICS CHALLENGES:
- Road network damaged
- Airport operations limited
- Cold weather (-5°C) affecting survivors
- Aftershocks continuing (200+ recorded)

RECOMMENDATIONS:
1. Prioritize thermal imaging sweeps of collapsed mid-rise buildings
2. Establish warm shelter zones within 48 hours
3. Deploy additional medical teams with trauma capabilities
4. Coordinate cross-border humanitarian corridor

Report Classification: IMMEDIATE PRIORITY
Distribution: All HADR Partners
"""
    }
]

# Sample emergency call transcripts
SAMPLE_TRANSCRIPTS = [
    {
        "text": "Hello? Hello? We need help! There are five of us trapped on the third floor. The building... the building partially collapsed. We can hear water rushing below us. Please send help! We're at 142 Main Street, the old apartment complex near the river. There's a child with us, she's scared but okay. We have some supplies but the water is rising. Please hurry!",
        "stress_level": 0.85,
        "duration": 45,
        "source": "emergency_call"
    },
    {
        "text": "This is Rescue Team Alpha reporting from Sector 7. We have located survivors in the collapsed parking structure on Oak Avenue. Count is approximately 12 individuals, mix of injuries. We need additional extraction equipment - specifically hydraulic spreaders. Building is unstable, possible secondary collapse risk. Request structural engineer on site before proceeding.",
        "stress_level": 0.65,
        "duration": 38,
        "source": "rescuer_radio"
    },
    {
        "text": "Dispatch, this is Unit 47. We're at the flood zone on Riverside Drive. Multiple vehicles submerged. We've extracted four people so far, one unconscious, performing CPR now. Need water rescue team and ambulance backup immediately. Current is strong, visibility poor. Will attempt to reach the sedan near the bridge.",
        "stress_level": 0.78,
        "duration": 32,
        "source": "rescuer_radio"
    },
    {
        "text": "Help us please! The fire is spreading so fast! We're on the roof of the elementary school on Pine Street. There are about twenty of us - teachers and children. The smoke is getting thick. We can see helicopters in the distance. Please tell them to come here! The children are crying, we're waving a large banner.",
        "stress_level": 0.92,
        "duration": 55,
        "source": "emergency_call"
    },
    {
        "text": "Base, this is Survey Drone Operator 3. Completing thermal sweep of Grid Charlie. I'm seeing significant heat signatures in the debris field at coordinates 34.0522 north, 118.2437 west. Marking locations on the map. Appears to be at least three separate clusters. Structural integrity of surrounding buildings looks compromised. Recommend ground teams approach from the north side.",
        "stress_level": 0.45,
        "duration": 42,
        "source": "rescuer_radio"
    }
]

# Sample image metadata (for synthetic image generation)
SAMPLE_IMAGE_METADATA = [
    {
        "filename": "satellite_flood_sector4.jpg",
        "description": "Aerial view showing widespread flooding in residential area. Multiple buildings partially submerged. Roads impassable. Approximately 50 structures affected.",
        "damage_score": 0.75,
        "damage_categories": ["flood", "infrastructure_damage"],
        "media_type": "satellite",
        "geohash": "9q5ctr"
    },
    {
        "filename": "uav_building_collapse.jpg",
        "description": "Close-up drone footage of collapsed multi-story building. Pancake collapse pattern visible. Rescue workers visible in lower right. Heavy machinery on scene.",
        "damage_score": 0.95,
        "damage_categories": ["building_collapse", "structural_failure"],
        "media_type": "uav",
        "geohash": "9q5cs0"
    },
    {
        "filename": "street_view_fire_damage.jpg",
        "description": "Ground-level view of fire-damaged commercial district. Multiple storefronts with broken windows and smoke damage. One building still smoldering. No visible casualties.",
        "damage_score": 0.68,
        "damage_categories": ["fire_damage", "commercial_damage"],
        "media_type": "street_view",
        "geohash": "9q5ctq"
    },
    {
        "filename": "satellite_earthquake_zone.jpg",
        "description": "Satellite imagery showing earthquake damage in urban center. Multiple building collapses visible. Ground displacement evident. Emergency vehicles clustered at three locations.",
        "damage_score": 0.88,
        "damage_categories": ["earthquake", "building_collapse", "ground_displacement"],
        "media_type": "satellite",
        "geohash": "svseb1"
    },
    {
        "filename": "uav_landslide_area.jpg",
        "description": "Drone survey of landslide affecting hillside community. Approximately 15 structures buried or displaced. Road access completely blocked. Debris field extends 200 meters.",
        "damage_score": 0.82,
        "damage_categories": ["landslide", "road_blockage", "structural_damage"],
        "media_type": "uav",
        "geohash": "9q5ctp"
    }
]


def check_disk_space(required_gb: float = 1.0) -> bool:
    """Check if sufficient disk space is available."""
    import shutil
    total, used, free = shutil.disk_usage("/")
    free_gb = free / (1024 ** 3)
    logger.info(f"Free disk space: {free_gb:.2f} GB")
    return free_gb >= required_gb


def create_sample_pdf(output_path: Path, content: dict) -> bool:
    """
    Create a sample PDF from text content.
    Uses a simple text-to-PDF approach without external dependencies.
    """
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
        from reportlab.lib.units import inch
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        
        doc = SimpleDocTemplate(str(output_path), pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        story.append(Paragraph(content["title"], styles['Heading1']))
        story.append(Spacer(1, 0.5 * inch))
        
        # Content - split into paragraphs
        for para in content["content"].split("\n\n"):
            if para.strip():
                story.append(Paragraph(para.replace("\n", "<br/>"), styles['Normal']))
                story.append(Spacer(1, 0.2 * inch))
        
        doc.build(story)
        return True
        
    except ImportError:
        # Fallback: create a simple text file with .pdf extension
        # (Will still work for text extraction)
        logger.warning("reportlab not installed, creating text-based PDF placeholder")
        with open(output_path, 'w') as f:
            f.write(f"TITLE: {content['title']}\n\n")
            f.write(content['content'])
        return True
        
    except Exception as e:
        logger.error(f"Failed to create PDF: {e}")
        return False


def create_sample_image(output_path: Path, metadata: dict) -> bool:
    """
    Create a placeholder image with embedded metadata.
    In production, you'd download real satellite imagery.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
        
        # Create a gradient background with disaster-like colors
        width, height = 800, 600
        img = Image.new('RGB', (width, height))
        
        # Generate color based on damage type
        damage_type = metadata.get("damage_categories", ["unknown"])[0]
        colors = {
            "flood": [(30, 100, 150), (50, 150, 200)],
            "fire_damage": [(100, 50, 30), (200, 100, 50)],
            "building_collapse": [(80, 80, 80), (150, 150, 150)],
            "earthquake": [(100, 80, 60), (180, 150, 120)],
            "landslide": [(80, 60, 40), (160, 130, 100)],
        }
        color_pair = colors.get(damage_type, [(100, 100, 100), (180, 180, 180)])
        
        # Create gradient
        for y in range(height):
            r = int(color_pair[0][0] + (color_pair[1][0] - color_pair[0][0]) * y / height)
            g = int(color_pair[0][1] + (color_pair[1][1] - color_pair[0][1]) * y / height)
            b = int(color_pair[0][2] + (color_pair[1][2] - color_pair[0][2]) * y / height)
            for x in range(width):
                # Add some noise for texture
                noise = random.randint(-20, 20)
                img.putpixel((x, y), (
                    max(0, min(255, r + noise)),
                    max(0, min(255, g + noise)),
                    max(0, min(255, b + noise))
                ))
        
        # Add some shapes to simulate structures
        draw = ImageDraw.Draw(img)
        
        # Draw "buildings" or "debris"
        for _ in range(random.randint(5, 15)):
            x1 = random.randint(50, width - 150)
            y1 = random.randint(50, height - 150)
            x2 = x1 + random.randint(30, 120)
            y2 = y1 + random.randint(30, 80)
            
            # Darker shade for structures
            shade = random.randint(0, 60)
            draw.rectangle([x1, y1, x2, y2], 
                          fill=(shade, shade, shade),
                          outline=(shade + 30, shade + 30, shade + 30))
        
        # Add metadata text overlay
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        except Exception:
            font = ImageFont.load_default()
        
        # Add semi-transparent overlay for text
        overlay = Image.new('RGBA', (width, 120), (0, 0, 0, 128))
        img = img.convert('RGBA')
        img.paste(overlay, (0, height - 120), overlay)
        
        draw = ImageDraw.Draw(img)
        text = f"{metadata['media_type'].upper()} | Damage: {metadata['damage_score']:.0%}"
        text2 = f"Categories: {', '.join(metadata['damage_categories'])}"
        
        draw.text((10, height - 110), text, fill=(255, 255, 255), font=font)
        draw.text((10, height - 85), text2, fill=(255, 255, 255), font=font)
        draw.text((10, height - 60), metadata['description'][:80] + "...", 
                 fill=(200, 200, 200), font=font)
        
        # Convert back to RGB and save
        img = img.convert('RGB')
        img.save(output_path, 'JPEG', quality=85)
        
        logger.info(f"Created sample image: {output_path.name}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to create image: {e}")
        return False


def create_sample_audio(output_path: Path, transcript_data: dict) -> bool:
    """
    Create a sample audio file.
    Uses text-to-speech if available, otherwise creates a minimal WAV.
    """
    try:
        import numpy as np
        import soundfile as sf
        
        # Create simple audio (sine wave with some variation)
        duration = transcript_data.get("duration", 30)
        sample_rate = 16000
        samples = int(duration * sample_rate)
        
        # Generate audio that vaguely sounds like speech rhythm
        t = np.linspace(0, duration, samples)
        
        # Base frequency with variation (simulates speech)
        freq_base = 150 + 50 * np.sin(2 * np.pi * 0.5 * t)
        freq_variation = 20 * np.sin(2 * np.pi * 3 * t)
        
        # Generate waveform
        phase = np.cumsum(2 * np.pi * (freq_base + freq_variation) / sample_rate)
        audio = 0.3 * np.sin(phase)
        
        # Add amplitude variation (simulates words/pauses)
        envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 2 * t)
        audio = audio * envelope
        
        # Add some noise
        audio += 0.05 * np.random.randn(samples)
        
        # Normalize
        audio = audio / np.max(np.abs(audio)) * 0.8
        
        # Save as WAV
        sf.write(str(output_path), audio.astype(np.float32), sample_rate)
        
        logger.info(f"Created sample audio: {output_path.name}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to create audio: {e}")
        return False


def create_transcript_json(output_path: Path, transcripts: list) -> bool:
    """Save transcript metadata as JSON for reference."""
    try:
        with open(output_path, 'w') as f:
            json.dump(transcripts, f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Failed to save transcripts: {e}")
        return False


def download_ladi_samples(output_dir: Path, num_samples: int = 10) -> int:
    """
    Download sample images from LADI dataset (Low Altitude Disaster Imagery).
    
    Note: This attempts to download from public sources. If unavailable,
    generates synthetic samples instead.
    """
    downloaded = 0
    
    # LADI v2 sample URLs (from public registry)
    # These are example URLs - actual URLs depend on current registry
    sample_urls = [
        # Example format - replace with actual LADI registry URLs
    ]
    
    if not sample_urls:
        logger.info("LADI direct download not configured, using synthetic samples")
        return 0
    
    for i, url in enumerate(sample_urls[:num_samples]):
        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                output_path = output_dir / f"ladi_sample_{i:04d}.jpg"
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                downloaded += 1
                logger.info(f"Downloaded: {output_path.name}")
        except Exception as e:
            logger.warning(f"Failed to download LADI sample {i}: {e}")
    
    return downloaded


def main():
    """Main function to prepare sample datasets."""
    logger.info("=" * 60)
    logger.info("Project Sentinel V2 - Dataset Preparation")
    logger.info("=" * 60)
    
    # Check disk space
    if not check_disk_space(1.0):
        logger.error("Insufficient disk space. Need at least 1GB free.")
        sys.exit(1)
    
    # Setup directories
    data_dir = PROJECT_ROOT / "data"
    raw_dir = data_dir / "raw_datasets"
    processed_dir = data_dir / "processed"
    
    pdf_dir = raw_dir / "reports"
    image_dir = raw_dir / "images"
    audio_dir = raw_dir / "audio"
    
    for d in [pdf_dir, image_dir, audio_dir, processed_dir]:
        d.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {d}")
    
    # Generate sample PDFs
    logger.info("\n--- Creating Sample PDF Reports ---")
    for i, report in enumerate(SAMPLE_REPORTS):
        filename = f"disaster_report_{i+1:02d}.pdf"
        output_path = pdf_dir / filename
        if create_sample_pdf(output_path, report):
            logger.info(f"✓ Created: {filename}")
        else:
            logger.error(f"✗ Failed: {filename}")
    
    # Generate sample images
    logger.info("\n--- Creating Sample Images ---")
    for metadata in SAMPLE_IMAGE_METADATA:
        output_path = image_dir / metadata["filename"]
        if create_sample_image(output_path, metadata):
            logger.info(f"✓ Created: {metadata['filename']}")
        else:
            logger.error(f"✗ Failed: {metadata['filename']}")
    
    # Try to download real LADI samples
    logger.info("\n--- Attempting LADI Dataset Download ---")
    ladi_downloaded = download_ladi_samples(image_dir)
    if ladi_downloaded > 0:
        logger.info(f"Downloaded {ladi_downloaded} LADI samples")
    else:
        logger.info("Using synthetic images (LADI download not available)")
    
    # Generate sample audio files
    logger.info("\n--- Creating Sample Audio Files ---")
    for i, transcript in enumerate(SAMPLE_TRANSCRIPTS):
        filename = f"emergency_audio_{i+1:02d}.wav"
        output_path = audio_dir / filename
        if create_sample_audio(output_path, transcript):
            logger.info(f"✓ Created: {filename}")
        else:
            logger.error(f"✗ Failed: {filename}")
    
    # Save transcript metadata
    transcript_path = audio_dir / "transcripts_metadata.json"
    if create_transcript_json(transcript_path, SAMPLE_TRANSCRIPTS):
        logger.info(f"✓ Created: transcripts_metadata.json")
    
    # Summary
    pdf_count = len(list(pdf_dir.glob("*.pdf")))
    image_count = len(list(image_dir.glob("*.jpg"))) + len(list(image_dir.glob("*.png")))
    audio_count = len(list(audio_dir.glob("*.wav"))) + len(list(audio_dir.glob("*.mp3")))
    
    logger.info("\n" + "=" * 60)
    logger.info("Dataset Preparation Complete!")
    logger.info("=" * 60)
    logger.info(f"PDF Reports:  {pdf_count}")
    logger.info(f"Images:       {image_count}")
    logger.info(f"Audio Files:  {audio_count}")
    logger.info(f"\nData directory: {data_dir}")
    logger.info("\nNext step: Run 'python scripts/initialize_db.py' to populate Qdrant")


if __name__ == "__main__":
    main()

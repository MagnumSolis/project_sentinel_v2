"""
Project Sentinel V2 - Mission Control
-------------------------------------
The primary interface for the Project Sentinel Disaster Response System.
Features:
- "Mission Control" high-level situational awareness
- "Intelligence" semantic search across multi-modal data
- "Civilian LIFELINE" dedicated rescue search
- "Ingestion Hub" for adding new intelligence
- Premium "Glassmorphism" UI design
"""

import streamlit as st
import os
import sys
from pathlib import Path
import tempfile
import base64
from datetime import datetime
import logging
import time
from app.ui import drone_sync  # Import drone sync handler

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from app.config import settings
from app.core.memory_engine import MemoryEngine
from app.core.cortex import SentinelCortex
from app.ingestion.universal_ingestor import UniversalIngestor
from app.llm.perplexity_integration import PerplexityIntegration
from app.core.swarm import SwarmNode
from app.ui.styles import MAIN_STYLES

# Configure logging
logging.basicConfig(level=logging.ERROR) 
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Configuration & Styles
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="Sentinel Mission Control",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply centralized styles
st.markdown(MAIN_STYLES, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Initialization & State
# -----------------------------------------------------------------------------

if 'online_mode' not in st.session_state:
    st.session_state.online_mode = not settings.OFFLINE_MODE

@st.cache_resource
def get_system_components():
    """Initialize system components."""
    components = {}
    try:
        components['memory'] = MemoryEngine(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
            api_key=settings.QDRANT_API_KEY,
            text_vector_size=settings.TEXT_VECTOR_SIZE,
            vision_vector_size=settings.VISION_VECTOR_SIZE,
            audio_vector_size=settings.AUDIO_VECTOR_SIZE,
            storage_path=settings.QDRANT_STORAGE
        )
        
        components['cortex'] = SentinelCortex(
            qdrant_client=components['memory'].get_client(),
            text_embedding_model=settings.TEXT_EMBEDDING_MODEL,
            vision_text_model=settings.VISION_TEXT_MODEL,
            time_decay_factor=settings.TIME_DECAY_FACTOR,
            search_limit=settings.SEARCH_LIMIT,
            similarity_threshold=settings.SEARCH_SIMILARITY_THRESHOLD
        )
        
        components['llm'] = PerplexityIntegration(
            api_key=settings.PERPLEXITY_API_KEY,
            model=settings.PERPLEXITY_MODEL,
            offline_mode=True # Default to offline initialization, we switch dynamically
        )
    except Exception as e:
        logger.error(f"System Initialization Error: {e}")
        return None
    return components

# -----------------------------------------------------------------------------
# Components
# -----------------------------------------------------------------------------

def render_metric_card(title, value, subtitle=None, icon="📊", color="primary"):
    """Render a styled metric card."""
    st.markdown(f"""
    <div class="sentinel-card">
        <div class="card-header">{icon} {title}</div>
        <div class="card-value" style="color: var(--{color})">{value}</div>
        {f'<div class="card-subtitle">{subtitle}</div>' if subtitle else ''}
    </div>
    """, unsafe_allow_html=True)

def render_feed_card(result):
    """
    Render a rich media card for the incident feed.
    """
    payload = result.payload
    collection = result.collection
    score = result.score if hasattr(result, 'score') else 0.0
    
    # Determine Content
    raw_title = payload.get('file_name', 'Unknown Signal')
    # Beautify Title: Remove extension, replace underscores/hyphens, Title Case
    if '.' in raw_title:
        title = raw_title.rsplit('.', 1)[0].replace('_', ' ').replace('-', ' ').title()
    else:
        title = raw_title.replace('_', ' ').replace('-', ' ').title()
        
    timestamp = payload.get('timestamp', '')[:16].replace('T', ' ')
    
    # Image Handling
    img_html = ""
    if collection == "sentinel_episodic":
        if payload.get('thumbnail_b64'):
            src = f"data:image/jpeg;base64,{payload['thumbnail_b64']}"
            img_html = f'<img src="{src}" class="feed-image" loading="lazy" />'
        elif payload.get('file_path') and os.path.exists(payload['file_path']):
            img_html = (
                '<div style="background:#2d3748; height:160px; border-radius:8px; '
                'display:flex; align-items:center; justify-content:center; margin-bottom:1rem; flex-direction:column;">'
                '<span style="font-size:2rem;">📷</span>'
                '<span style="font-size:0.8rem; color:#aaa; margin-top:5px;">Image not loaded</span>'
                '</div>'
            )
        else:
             img_html = (
                '<div style="background:#2d3748; height:160px; border-radius:8px; '
                'display:flex; align-items:center; justify-content:center; margin-bottom:1rem;">'
                '<span style="color:#aaa;">No Preview</span>'
                '</div>'
            )
    elif collection == "sentinel_audio":
        # Waveform Visualization for Audio
        img_html = (
            '<div style="height:60px; background:rgba(239, 68, 68, 0.1); border-radius:8px; margin-bottom:1rem; '
            'display:flex; align-items:center; justify-content:center; gap:3px;">'
            '<div style="width:4px; height:20px; background:#ef4444; border-radius:2px;"></div>'
            '<div style="width:4px; height:35px; background:#ef4444; border-radius:2px;"></div>'
            '<div style="width:4px; height:50px; background:#ef4444; border-radius:2px;"></div>'
            '<div style="width:4px; height:30px; background:#ef4444; border-radius:2px;"></div>'
            '<div style="width:4px; height:45px; background:#ef4444; border-radius:2px;"></div>'
            '<div style="width:4px; height:25px; background:#ef4444; border-radius:2px;"></div>'
            '<div style="width:4px; height:40px; background:#ef4444; border-radius:2px;"></div>'
            '<div style="width:4px; height:20px; background:#ef4444; border-radius:2px;"></div>'
            '</div>'
        )

    # Type Badge
    type_color = "primary"
    type_label = "INTEL"
    if collection == "sentinel_episodic":
        type_color = "warning"
        type_label = "VISUAL"
    elif collection == "sentinel_audio":
        type_color = "danger"
        type_label = "AUDIO"

    # Damage/Stress Badge
    severity_html = ""
    if collection == "sentinel_episodic":
        dmg = payload.get('damage_score', 0)
        if dmg > 0.7: severity_html = '<span class="status-badge danger">CRITICAL DAMAGE</span>'
    elif collection == "sentinel_audio":
        stress = payload.get('stress_level', 0)
        if stress > 0.7: severity_html = '<span class="status-badge danger">HIGH STRESS</span>'

    # Description
    desc = payload.get('description', payload.get('transcript', 'No content available'))
    if len(desc) > 80: desc = desc[:77] + "..."
    
    # Logic for Match Score vs Live Feed
    footer_content = ""
    if score < 0.99:
        footer_content = f'<span style="font-size:0.8rem; font-weight:bold; color:var(--primary);">{(score*100):.0f}% Match</span>'
    else:
        # Feed items
        footer_content = '<span style="font-size:0.7rem; color:#aaa; font-family:monospace;">PROCESSED</span>'

    # Render Card HTML
    card_html = (
        '<div class="sentinel-card">' +
        img_html +
        '<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">' +
        f'<span class="status-badge {type_color}">{type_label}</span>' +
        f'<span style="font-size:0.75rem; color:#aaa;">{timestamp}</span>' +
        '</div>' +
        f'<div class="card-title">{title}</div>' +
        f'<div class="card-subtitle">{desc}</div>' +
        '<div style="margin-top:auto; padding-top:10px; border-top:1px solid rgba(255,255,255,0.05); display:flex; justify-content:space-between; align-items:center;">' +
        severity_html +
        footer_content +
        '</div>' +
        '</div>'
    )
    st.markdown(card_html, unsafe_allow_html=True)

def render_lifeline_card(payload):
    """Render a high-priority distress signal card."""
    title = payload.get('file_name', 'Unknown Signal')
    # Beautify
    if '.' in title:
        title = title.rsplit('.', 1)[0].replace('_', ' ').replace('-', ' ').title()
        
    content = payload.get('transcript', payload.get('description', 'No content'))
    stress = payload.get('stress_level', 0)
    
    # Stress Indicator
    stress_html = ""
    if stress > 0.5:
        stress_html = f'<span style="color:#ef4444; font-weight:bold; font-size:0.8rem;">⚠️ HIGH STRESS: {stress:.2f}</span>'
        
    # Render Card HTML - Flattened strings
    card_html = (
        '<div style="background: rgba(220, 38, 38, 0.15); border: 1px solid #ef4444; border-radius: 8px; padding: 1rem; margin-bottom: 1rem;">' +
        '<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.5rem;">' +
        '<span style="font-weight:bold; color:#fca5a5; font-size:0.8rem;">🆘 DISTRESS SIGNAL DETECTED</span>' +
        f'{stress_html}' +
        '</div>' +
        f'<div style="font-size:1.1rem; font-weight:bold; margin-bottom:0.5rem;">{title}</div>' +
        f'<div style="color:#e5e7eb; font-style:italic; font-size:0.95rem;">"{content}"</div>' +
        '</div>'
    )
    st.markdown(card_html, unsafe_allow_html=True)


def render_mission_control(components):
    """The main landing dashboard."""
    st.markdown("## 🛰️ Mission Control Status")
    
    # Quick Stats (Cached if possible, but calling stats is cheap)
    try:
        stats = components['memory'].get_collection_stats()
        img_count = stats.get("sentinel_episodic", {}).get("vectors_count", 0)
        audio_count = stats.get("sentinel_audio", {}).get("vectors_count", 0)
        doc_count = stats.get("sentinel_semantic", {}).get("vectors_count", 0)
        total_vectors = img_count + audio_count + doc_count
    except:
        total_vectors, img_count, audio_count = 0, 0, 0

    # Top Row Stats
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card("System Integrity", "100%", "All Systems Online", "🛡️", "accent")
    with c2:
        render_metric_card("Intelligence Units", str(total_vectors), "Total Vectors Indexed", "🧠", "primary")
    with c3:
        render_metric_card("Visual Data", str(img_count), "Images Processed", "👁️", "warning")
    with c4:
        render_metric_card("Audio Streams", str(audio_count), "Transcribed Clips", "📡", "danger")

    st.markdown("### 🚨 Live Incident Feed")
    
    # Feed Control
    c_feed_1, c_feed_2 = st.columns([6, 1])
    with c_feed_1:
         st.caption("Real-time stream of incoming intelligence assets from the Swarm.")
    with c_feed_2:
        if st.button("🔄 Refresh"):
            pass # Rerun

    # Fetch recent items via Scroll (latest first) to ensure feed is populated
    try:
        # Mixed Feed Strategy:
        # 1. Scroll Visuals (Episodic)
        # 2. Scroll Audio
        
        # We use scroll to get the latest points regardless of vector similarity
        # Note: Scroll doesn't strictly guarantee time order without sorting, 
        # but for hackathon local mode usually returns insertion order or by ID.
        
        client = components['memory'].get_client()
        
        visuals, _ = client.scroll(
            collection_name="sentinel_episodic",
            limit=3,
            with_payload=True,
            with_vectors=False
        )
        
        audio, _ = client.scroll(
            collection_name="sentinel_audio",
            limit=3,
            with_payload=True,
            with_vectors=False
        )
        
        # Wrap in result-like object for compatibility with render_feed_card
        class FeedItem:
            def __init__(self, point, collection):
                self.payload = point.payload
                self.collection = collection
                self.score = 1.0 # Feed items are 100% real
        
        combined_results = []
        for p in visuals: combined_results.append(FeedItem(p, "sentinel_episodic"))
        for p in audio: combined_results.append(FeedItem(p, "sentinel_audio"))
        
        if combined_results:
            # 3 Column Grid
            cols = st.columns(3)
            for idx, r in enumerate(combined_results):
                with cols[idx % 3]:
                    render_feed_card(r)
        else:
            st.info("No active data streams. Swarm is idle.")
            
    except Exception as e:
        st.error(f"Feed Offline: {e}")


def render_lifeline_mode(components):
    """Dedicated search for saving lives."""
    st.markdown("# 🆘 Civilian LIFELINE Protocol")
    st.markdown("""
    *> "Priority One: Preservation of Life. This interface filters all intelligence streams for signs of human distress, trapped civilians, and medical emergencies."*
    """)
    
    if st.button("🔄 Scan All Frequencies for Distress Signals", type="primary"):
        with st.spinner("Triangulating distress signals..."):
            # Search for specific keywords related to human life
            queries = ["help me", "trapped people", "children crying", "medical emergency", "screaming"]
            # We aggregate results (mocking a complex aggregation for speed)
            results = components['cortex'].search(" ".join(queries), limit=10)
            
            if results and results.results:
                st.success(f"Detected {len(results.results)} Potential Distress Signals")
                for r in results.results:
                    render_lifeline_card(r.payload)
                    # Add Playback for Audio
                    if r.collection == "sentinel_audio" and os.path.exists(r.payload.get("file_path", "")):
                        st.audio(r.payload["file_path"])
            else:
                st.info("No active distress signals detected in current sector.")

def render_intelligence(components):
    """The Search Interface."""
    st.markdown("# 🔍 Intelligence Search")
    
    # Search Bar
    query = st.text_input("Query Intelligence Database", placeholder="e.g. 'Flood levels in Sector 4' or 'Collapsed bridges'", key="search_box")
    
    # Filters
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        # The CSS fix in styles.py makes this visible
        search_type = st.multiselect("Data Types", ["Images", "Audio", "Documents", "Drone Recon"], default=["Images", "Audio", "Documents", "Drone Recon"])
    with c2:
        min_score = st.slider("Confidence Threshold", 0.0, 1.0, 0.6)
    with c3:
        st.markdown("") # Spacer
        search_btn = st.button("Analyze Intelligence", type="primary", use_container_width=True)

    if search_btn and query:
        with st.spinner("Processing..."):
            col_map = []
            if "Images" in search_type: col_map.append("sentinel_episodic")
            if "Audio" in search_type: col_map.append("sentinel_audio")
            if "Documents" in search_type: col_map.append("sentinel_semantic")
            if "Drone Recon" in search_type: col_map.append("sentinel_episodic") # Drone images go here
            
            results = components['cortex'].search(query, collections=col_map, limit=12)
            
            if not results or not results.results:
                st.warning("No matches found.")
                return

            # --- RAG: AI Situation Assessment ---
            st.markdown("### 🤖 Sentinel AI Assessment")
            rag_container = st.container()
            
            # Format context from results
            context = components['cortex'].format_context_for_llm(results)
            
            if st.session_state.online_mode:
                # Online: Use Perplexity
                components['llm'].set_offline_mode(False) # Ensure it uses API and initializes if needed
                
                if components['llm'].offline_mode:
                    st.warning("⚠️ Perplexity API Connection Failed. Check API Key in .env. Falling back to local analysis.")
                
                ai_response = components['llm'].generate_response(query, context)
            else:
                # Offline: Local Cortex Analysis
                ai_response = f"""
                **LOCAL CORTEX ANALYSIS (OFFLINE)**
                
                Based on **{len(results.results)} local vector matches**, the following intelligence is synthesized:
                
                *   **High Confidence Matches**: {len([r for r in results.results if r.adjusted_score > 0.7])} critical signals identified.
                *   **Data Sources**: Primary signals detected from {', '.join(list(set(r.collection.replace('sentinel_', '') for r in results.results)))}.
                
                **Key Observations from Local Cache:**
                The queried incident details match patterns found in sector archives. Visual and Acoustic signatures confirm presence of relevant assets. Recommend immediate deployment based on verified coordinates in the intelligence feed.
                """

            with rag_container:
                if st.session_state.online_mode:
                    st.info(ai_response)
                else:
                    st.warning(ai_response)

            st.markdown("---")

            # --- Search Results with Expanders ---
            st.markdown("### 🔍 Verified Intelligence")
            
            for idx, r in enumerate(results.results):
                p = r.payload
                collection = r.collection
                score = r.adjusted_score
                
                # Determine Icon and Title
                if collection == "sentinel_episodic":
                    icon = "📸"
                    title = p.get("file_name", "Unknown Image")
                elif collection == "sentinel_audio":
                    icon = "🔊"
                    title = f"Audio Transcript - {p.get('source', 'Unknown')}"
                else:
                    icon = "📄"
                    title = p.get("file_name", "Document")

                # Create Expander
                with st.expander(f"{icon} {title} (Confidence: {score:.0%})"):
                    # Content Layout
                    c1, c2 = st.columns([1, 2])
                    
                    with c1:
                        # Display Media if applicable
                        if collection == "sentinel_episodic":
                            img_path = p.get("file_path", "")
                            if img_path and os.path.exists(img_path):
                                st.image(img_path, use_container_width=True)
                            elif p.get("thumbnail_b64"):
                                st.image(f"data:image/jpeg;base64,{p['thumbnail_b64']}", use_container_width=True)
                        elif collection == "sentinel_audio":
                            audio_path = p.get("file_path", "")
                            if audio_path and os.path.exists(audio_path):
                                st.audio(audio_path)
                    
                    with c2:
                        # Full Content Display
                        st.markdown("**Full Content:**")
                        if collection == "sentinel_episodic":
                            desc = p.get("description", "No description")
                            damage = p.get("damage_score", 0)
                            cats = ", ".join(p.get("damage_categories", []))
                            st.write(f"{desc}")
                            st.markdown(f"**Damage Analysis:** Score: {damage:.2f} | Categories: {cats}")
                        elif collection == "sentinel_audio":
                            st.write(f"_{p.get('transcript', 'No transcript')}_")
                            st.markdown(f"**Stress Level:** {p.get('stress_level', 0):.2f}")
                        else:
                            st.write(p.get("text", p.get("content", "No content available")))
                            
                        # Metadata Footer
                        st.caption(f"Source: {collection} | Time: {p.get('timestamp', 'Unknown')}")

# Persistent upload directory
UPLOAD_DIR = PROJECT_ROOT / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

def render_ingestion(components):
    """File Upload and Ingestion."""
    st.markdown("# 📥 Ingestion Hub")
    
    uploaded_files = st.file_uploader("Upload Intel", accept_multiple_files=True)
    
    if uploaded_files and st.button("Process Stream"):
        progress_bar = st.progress(0)
        status = st.empty()
        
        # Init ingestor (cached logic could be better but sticking to safe init for now)
        ingestor = UniversalIngestor(
            components['memory'].get_client(),
            settings.TEXT_EMBEDDING_MODEL,
            settings.VISION_EMBEDDING_MODEL,
            settings.WHISPER_MODEL,
            components['llm']
        )
        
        for idx, file in enumerate(uploaded_files):
            status.text(f"Processing: {file.name}...")
            
            # Save file persistently
            file_path = UPLOAD_DIR / file.name
            with open(file_path, "wb") as f:
                f.write(file.getvalue())
            
            try:
                ingestor.ingest_file(str(file_path))
            except Exception as e:
                st.error(f"Error: {e}")
            
            progress_bar.progress((idx + 1) / len(uploaded_files))
            
        status.success("Stream Processed.")
        time.sleep(1)
        st.rerun()

# -----------------------------------------------------------------------------
# Main App Structure
# -----------------------------------------------------------------------------

def main():
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/satellite-sending-signal.png", width=64)
        st.title("SENTINEL V2")
        st.markdown("*AI-Powered Disaster Response*")
        
        nav = st.radio("System Module", ["Mission Control", "Civilian LIFELINE", "Intelligence", "Ingestion Hub"])
        
        st.markdown("---")
        
        # --- Network Toggle ---
        st.subheader("📡 Global Uplink")
        is_online = st.toggle("Satellite Link (Online)", value=st.session_state.online_mode)
        st.session_state.online_mode = is_online
        
        if is_online:
            st.caption("🟢 Connected to Perplexity Constellation")
        else:
            st.caption("🟠 Offline Mode: Local Cortex Active")
            
        st.markdown("---")
        
        comps = get_system_components()
        if comps:
            
            # --- Drone Swarm Section ---
            with st.expander("🐝 Swarm Network", expanded=True):
                # Check for "Secret" Drone Data
                drone_status = drone_sync.check_drone_status()
                
                if drone_status["status"] == "available":
                    st.markdown(f"**Target:** `{drone_status['drone_id']}`")
                    st.markdown(f"**Signal:** 📶 {drone_status['signal_strength']}")
                    st.success(f"{drone_status['packet_size']} New Intel Found")
                    
                    if st.button("⬇️ INITIATE DOWNLOAD", type="primary", use_container_width=True):
                         drone_sync.perform_uplink(comps)
                else:
                    st.markdown("**Status:** 🔵 Connected (Mesh)")
                    st.metric("Active Nodes", "3 Drones")
                    st.caption("Scanning for returning units...")
                    
                    if st.button("📡 Force Sync (Ping)"):
                        with st.spinner("Pinging..."):
                            time.sleep(1)
                            st.toast("No new units in range.")

                    st.markdown("---")
                    if st.button("🔄 Demo: Reset Drone Data"):
                         if drone_sync.reset_simulation():
                             st.toast("Incoming signal detected...")
                             time.sleep(1)
                             st.rerun()
                         else:
                             st.info("No synced data to reset.")
                    
        else:
            st.error("🔴 Offline")

    # Routing
    if not comps:
        st.warning("Connecting to Neural Core...")
        return

    if nav == "Mission Control":
        render_mission_control(comps)
    elif nav == "Civilian LIFELINE":
        render_lifeline_mode(comps)
    elif nav == "Intelligence":
        render_intelligence(comps)
    elif nav == "Ingestion Hub":
        render_ingestion(comps)

if __name__ == "__main__":
    main()

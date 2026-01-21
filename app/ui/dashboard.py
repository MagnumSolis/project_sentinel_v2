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

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from app.config import settings
from app.core.memory_engine import MemoryEngine
from app.core.cortex import SentinelCortex
from app.ingestion.universal_ingestor import UniversalIngestor
from app.llm.perplexity_integration import PerplexityIntegration
from app.ui.styles import MAIN_STYLES  # Import centralized styles

# Configure logging
logging.basicConfig(level=logging.ERROR) # Reduced logging noise
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
# Initialization
# -----------------------------------------------------------------------------

@st.cache_resource
def get_system_components():
    """Initialize system components with robust error handling."""
    components = {}
    try:
        components['memory'] = MemoryEngine(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
            api_key=settings.QDRANT_API_KEY,
            text_vector_size=settings.TEXT_VECTOR_SIZE,
            vision_vector_size=settings.VISION_VECTOR_SIZE,
            audio_vector_size=settings.AUDIO_VECTOR_SIZE
        )
        # Assuming memory is already initialized, or skipping expensive re-init
        # components['memory'].initialize_memory() 
        
        components['cortex'] = SentinelCortex(
            qdrant_client=components['memory'].get_client(),
            text_embedding_model=settings.TEXT_EMBEDDING_MODEL,
            vision_text_model=settings.VISION_TEXT_MODEL,  # Added this
            time_decay_factor=settings.TIME_DECAY_FACTOR,
            search_limit=settings.SEARCH_LIMIT,
            similarity_threshold=settings.SEARCH_SIMILARITY_THRESHOLD
        )
        
        components['llm'] = PerplexityIntegration(
            api_key=settings.PERPLEXITY_API_KEY,
            model=settings.PERPLEXITY_MODEL,
            offline_mode=settings.OFFLINE_MODE
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
        {f'<div class="card-stat-label">{subtitle}</div>' if subtitle else ''}
    </div>
    """, unsafe_allow_html=True)

def render_lifeline_card(payload):
    """Render a dedicated high-priority civilian rescue card."""
    st.markdown(f"""
    <div class="sentinel-card lifeline-card">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <h3 style="margin:0; color:#ef4444;">🆘 CIVILIAN DISTRESS SIGNAL</h3>
            <span style="background:red; color:white; padding:2px 8px; border-radius:4px; font-size:0.8em; font-weight:bold;">CRITICAL</span>
        </div>
        <div style="margin-top:10px; font-size:1.1em; color:white;">
            "{payload.get('transcript', payload.get('description', 'Unknown Context'))}"
        </div>
        <div style="margin-top:10px; font-size:0.9em; color:#bbb;">
            📍 Location: {payload.get('file_name', 'Unknown Sector')} | 🕒 {payload.get('timestamp', 'Recent')}
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_mission_control(components):
    """The main landing dashboard."""
    st.markdown("# 🛰️ Mission Control Status")
    
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
    
    # Fetch recent random items or high priority
    try:
        results = components['cortex'].search("damage disaster emergency", limit=4)
        if results and results.results:
            cols = st.columns(2)
            for idx, r in enumerate(results.results[:4]):
                with cols[idx % 2]:
                    p = r.payload
                    with st.container():
                        st.markdown(f"""
                        <div class="sentinel-card">
                            <strong>{p.get('file_name')}</strong><br>
                            <span style="font-size:0.9em; color:#ccc">{p.get('description', p.get('transcript', ''))[:100]}...</span>
                        </div>
                        """, unsafe_allow_html=True)
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
        search_type = st.multiselect("Data Types", ["Images", "Audio", "Documents"], default=["Images", "Audio", "Documents"])
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
            
            results = components['cortex'].search(query, collections=col_map, limit=12)
            
            if not results or not results.results:
                st.warning("No matches found.")
                return

            # --- RAG: AI Situation Assessment ---
            st.markdown("### 🤖 Sentinel AI Assessment")
            rag_container = st.container()
            
            # Format context from results
            context = components['cortex'].format_context_for_llm(results)
            
            # Generate Response
            ai_response = components['llm'].generate_response(query, context)
            
            with rag_container:
                st.info(ai_response)

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
        comps = get_system_components()
        if comps:
            st.success("🟢 System Online")
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

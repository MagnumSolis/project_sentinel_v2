"""
Project Sentinel V2 - Streamlit Dashboard

Main user interface for the disaster response RAG system.
Features:
- Semantic search across all modalities
- File upload for new data ingestion
- Real-time collection statistics
- AI-powered response generation
"""

import streamlit as st
import os
import sys
from pathlib import Path
import tempfile
import base64
from datetime import datetime
import logging

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from app.config import settings
from app.core.memory_engine import MemoryEngine
from app.core.cortex import SentinelCortex
from app.ingestion.universal_ingestor import UniversalIngestor
from app.llm.perplexity_integration import PerplexityIntegration

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Project Sentinel V2",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #1e3a5f;
        --secondary-color: #2d5a87;
        --accent-color: #4a9eff;
        --danger-color: #ff4a4a;
        --success-color: #4aff4a;
        --warning-color: #ffaa4a;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        color: white;
        text-align: center;
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
        font-size: 1.1rem;
    }
    
    /* Stats cards */
    .stat-card {
        background: linear-gradient(135deg, #2d5a87 0%, #1e3a5f 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .stat-card h3 {
        margin: 0;
        font-size: 2rem;
        font-weight: 700;
    }
    
    .stat-card p {
        margin: 0.25rem 0 0 0;
        opacity: 0.8;
        font-size: 0.9rem;
    }
    
    /* Result cards */
    .result-card {
        background: #f8f9fa;
        border-left: 4px solid #4a9eff;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
    }
    
    .result-card.high-priority {
        border-left-color: #ff4a4a;
        background: #fff5f5;
    }
    
    .result-card.medium-priority {
        border-left-color: #ffaa4a;
        background: #fffaf5;
    }
    
    /* Thumbnail styling */
    .thumbnail-container {
        display: inline-block;
        margin-right: 1rem;
        vertical-align: top;
    }
    
    .thumbnail-container img {
        border-radius: 4px;
        max-width: 150px;
        max-height: 100px;
        object-fit: cover;
    }
    
    /* Status indicators */
    .status-online {
        color: #4aff4a;
        font-weight: bold;
    }
    
    .status-offline {
        color: #ff4a4a;
        font-weight: bold;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Improve button styling */
    .stButton > button {
        background: linear-gradient(135deg, #4a9eff 0%, #2d5a87 100%);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(74, 158, 255, 0.4);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def init_memory_engine():
    """Initialize and cache the memory engine."""
    try:
        engine = MemoryEngine(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
            api_key=settings.QDRANT_API_KEY,
            text_vector_size=settings.TEXT_VECTOR_SIZE,
            vision_vector_size=settings.VISION_VECTOR_SIZE,
            audio_vector_size=settings.AUDIO_VECTOR_SIZE
        )
        engine.initialize_memory()
        return engine
    except Exception as e:
        st.error(f"Failed to connect to Qdrant: {e}")
        return None


@st.cache_resource
def init_cortex(_memory_engine):
    """Initialize and cache the search cortex."""
    if _memory_engine is None:
        return None
    try:
        return SentinelCortex(
            qdrant_client=_memory_engine.get_client(),
            text_embedding_model=settings.TEXT_EMBEDDING_MODEL,
            time_decay_factor=settings.TIME_DECAY_FACTOR,
            search_limit=settings.SEARCH_LIMIT,
            similarity_threshold=settings.SEARCH_SIMILARITY_THRESHOLD
        )
    except Exception as e:
        st.error(f"Failed to initialize search cortex: {e}")
        return None


@st.cache_resource
def init_perplexity():
    """Initialize and cache Perplexity integration."""
    try:
        return PerplexityIntegration(
            api_key=settings.PERPLEXITY_API_KEY,
            model=settings.PERPLEXITY_MODEL,
            offline_mode=settings.OFFLINE_MODE
        )
    except Exception as e:
        logger.warning(f"Perplexity initialization failed: {e}")
        return PerplexityIntegration(offline_mode=True)


def init_ingestor(memory_engine, llm_client):
    """Initialize the universal ingestor (not cached due to state)."""
    if memory_engine is None:
        return None
    try:
        return UniversalIngestor(
            qdrant_client=memory_engine.get_client(),
            text_embedding_model=settings.TEXT_EMBEDDING_MODEL,
            vision_embedding_model=settings.VISION_EMBEDDING_MODEL,
            whisper_model_name=settings.WHISPER_MODEL,
            llm_client=llm_client
        )
    except Exception as e:
        st.error(f"Failed to initialize ingestor: {e}")
        return None


def render_header():
    """Render the main header."""
    st.markdown("""
    <div class="main-header">
        <h1>🛰️ Project Sentinel V2</h1>
        <p>Multimodal RAG System for Disaster Response</p>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar(memory_engine, gemini_client):
    """Render the sidebar with stats and controls."""
    with st.sidebar:
        st.markdown("### 📊 System Status")
        
        # Connection status
        qdrant_status = "🟢 Connected" if memory_engine and memory_engine.health_check() else "🔴 Disconnected"
        llm_status = "🟢 Online" if gemini_client and gemini_client.is_available() else "🟡 Offline"
        
        st.markdown(f"**Qdrant:** {qdrant_status}")
        st.markdown(f"**Perplexity:** {llm_status}")
        
        st.markdown("---")
        
        # Collection statistics
        st.markdown("### 📁 Collections")
        
        if memory_engine:
            stats = memory_engine.get_collection_stats()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                semantic_count = stats.get("sentinel_semantic", {}).get("vectors_count", 0)
                st.metric("📄 Docs", semantic_count)
            
            with col2:
                episodic_count = stats.get("sentinel_episodic", {}).get("vectors_count", 0)
                st.metric("🖼️ Images", episodic_count)
            
            with col3:
                audio_count = stats.get("sentinel_audio", {}).get("vectors_count", 0)
                st.metric("🎙️ Audio", audio_count)
            
            total = semantic_count + episodic_count + audio_count
            st.markdown(f"**Total Vectors:** {total:,}")
        
        st.markdown("---")
        
        # Search settings
        st.markdown("### ⚙️ Search Settings")
        
        search_limit = st.slider(
            "Results per collection",
            min_value=1,
            max_value=20,
            value=settings.SEARCH_LIMIT,
            key="search_limit"
        )
        
        similarity_threshold = st.slider(
            "Similarity threshold",
            min_value=0.0,
            max_value=1.0,
            value=settings.SEARCH_SIMILARITY_THRESHOLD,
            step=0.05,
            key="similarity_threshold"
        )
        
        st.markdown("---")
        
        # Mode toggle
        st.markdown("### 🔧 Mode")
        offline_mode = st.checkbox(
            "Offline Mode",
            value=settings.OFFLINE_MODE,
            help="When enabled, Perplexity API calls are skipped"
        )
        
        return search_limit, similarity_threshold, offline_mode


def render_result_card(result, index):
    """Render a single search result card."""
    payload = result.payload
    collection = result.collection
    score = result.adjusted_score
    
    # Determine priority based on score and content
    priority_class = ""
    if score > 0.85 or payload.get("stress_level", 0) > 0.8 or payload.get("damage_score", 0) > 0.8:
        priority_class = "high-priority"
    elif score > 0.7:
        priority_class = "medium-priority"
    
    # Collection icon
    icons = {
        "sentinel_semantic": "📄",
        "sentinel_episodic": "🖼️",
        "sentinel_audio": "🎙️"
    }
    icon = icons.get(collection, "📌")
    
    # Format content based on type
    if collection == "sentinel_semantic":
        title = f"{icon} Document: {payload.get('document_id', 'Unknown')}"
        content = payload.get("text", "No text available")[:500]
        if len(payload.get("text", "")) > 500:
            content += "..."
        extra_info = f"Chunk {payload.get('chunk_index', 0)} | {payload.get('source_type', 'Unknown')}"
        
    elif collection == "sentinel_episodic":
        title = f"{icon} Image: {payload.get('file_name', 'Unknown')}"
        content = payload.get("description", "No description available")
        damage = payload.get("damage_score", 0)
        categories = ", ".join(payload.get("damage_categories", []))
        extra_info = f"Damage: {damage:.0%} | Type: {categories}"
        
        # Show thumbnail if available
        thumbnail = payload.get("thumbnail_b64", "")
        if thumbnail:
            st.markdown(f"""
            <div class="thumbnail-container">
                <img src="data:image/jpeg;base64,{thumbnail}" alt="Thumbnail"/>
            </div>
            """, unsafe_allow_html=True)
            
    else:  # Audio
        title = f"{icon} Audio: {payload.get('file_name', 'Unknown')}"
        content = payload.get("transcript", "No transcript available")
        stress = payload.get("stress_level", 0)
        duration = payload.get("duration_seconds", 0)
        extra_info = f"Stress: {stress:.0%} | Duration: {duration:.1f}s | {payload.get('source', 'Unknown')}"
    
    # Render card
    with st.container():
        st.markdown(f"""
        <div class="result-card {priority_class}">
            <strong>{title}</strong>
            <br><small style="color: #666;">{extra_info} | Score: {score:.2%}</small>
            <p style="margin-top: 0.5rem;">{content}</p>
        </div>
        """, unsafe_allow_html=True)


def render_search_section(cortex, gemini_client, search_limit, similarity_threshold):
    """Render the main search interface."""
    st.markdown("### 🔍 Semantic Search")
    
    # Search input
    query = st.text_input(
        "Enter your search query",
        placeholder="e.g., Find areas with trapped civilians in flood zones",
        key="search_query"
    )
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        search_semantic = st.checkbox("📄 Documents", value=True)
    with col2:
        search_episodic = st.checkbox("🖼️ Images", value=True)
    with col3:
        search_audio = st.checkbox("🎙️ Audio", value=True)
    with col4:
        generate_response = st.checkbox("🤖 AI Response", value=True)
    
    if st.button("🔎 Search", type="primary", use_container_width=True):
        if not query:
            st.warning("Please enter a search query")
            return
        
        if cortex is None:
            st.error("Search system not initialized. Check Qdrant connection.")
            return
        
        # Build collection list
        collections = []
        if search_semantic:
            collections.append("sentinel_semantic")
        if search_episodic:
            collections.append("sentinel_episodic")
        if search_audio:
            collections.append("sentinel_audio")
        
        if not collections:
            st.warning("Please select at least one collection to search")
            return
        
        # Perform search
        with st.spinner("Searching..."):
            results = cortex.search(
                query=query,
                collections=collections,
                limit=search_limit
            )
        
        # Display results
        st.markdown(f"### 📋 Results ({results.total_count} found in {results.search_time_ms:.0f}ms)")
        
        if not results.results:
            st.info("No results found. Try adjusting your query or lowering the similarity threshold.")
            return
        
        # Show results
        for i, result in enumerate(results.results[:15]):  # Limit display
            render_result_card(result, i)
        
        # Generate AI response if requested
        if generate_response and gemini_client:
            st.markdown("---")
            st.markdown("### 🤖 AI Analysis")
            
            with st.spinner("Generating response..."):
                context = cortex.format_context_for_llm(results)
                response = gemini_client.generate_response(query, context)
            
            st.markdown(response)


def render_upload_section(ingestor, memory_engine):
    """Render the file upload interface."""
    st.markdown("### 📤 Upload Data")
    
    uploaded_files = st.file_uploader(
        "Upload disaster data files",
        type=["pdf", "jpg", "jpeg", "png", "mp3", "wav"],
        accept_multiple_files=True,
        help="Supported: PDF reports, images (JPG/PNG), audio (MP3/WAV)"
    )
    
    if uploaded_files and st.button("📥 Ingest Files", type="primary"):
        if ingestor is None:
            st.error("Ingestor not initialized")
            return
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        results = []
        for i, uploaded_file in enumerate(uploaded_files):
            progress = (i + 1) / len(uploaded_files)
            progress_bar.progress(progress)
            status_text.text(f"Processing: {uploaded_file.name}")
            
            # Save to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
            
            try:
                result = ingestor.ingest_file(tmp_path)
                results.append((uploaded_file.name, result))
            finally:
                os.unlink(tmp_path)
        
        progress_bar.empty()
        status_text.empty()
        
        # Show results
        st.markdown("#### Ingestion Results")
        for filename, result in results:
            if result.success:
                st.success(f"✓ {filename}: {result.vectors_added} vectors added to {result.collection}")
            else:
                st.error(f"✗ {filename}: {', '.join(result.errors)}")
        
        # Clear cache to refresh stats
        st.cache_resource.clear()
        st.rerun()


def render_quick_actions(cortex, gemini_client):
    """Render quick action buttons."""
    st.markdown("### ⚡ Quick Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🚨 High Priority Cases", use_container_width=True):
            if cortex:
                with st.spinner("Finding high priority cases..."):
                    results = cortex.search_by_stress_level(min_stress=0.7, limit=5)
                    if results:
                        st.markdown("#### High Stress Audio")
                        for r in results:
                            st.warning(f"🎙️ Stress: {r.payload.get('stress_level', 0):.0%} - {r.payload.get('transcript', '')[:200]}...")
                    else:
                        st.info("No high-stress audio found")
    
    with col2:
        if st.button("🏚️ Severe Damage", use_container_width=True):
            if cortex:
                with st.spinner("Finding severe damage..."):
                    results = cortex.search_by_damage(min_damage=0.7, limit=5)
                    if results:
                        st.markdown("#### Severe Damage Areas")
                        for r in results:
                            damage = r.payload.get('damage_score', 0)
                            cats = ", ".join(r.payload.get('damage_categories', []))
                            st.error(f"🖼️ Damage: {damage:.0%} - {cats}")
                    else:
                        st.info("No severe damage images found")
    
    with col3:
        if st.button("📊 Situation Report", use_container_width=True):
            if cortex and gemini_client:
                with st.spinner("Generating situation report..."):
                    # Get recent data from all collections
                    results = cortex.search("damage assessment rescue operations", limit=10)
                    summary = gemini_client.summarize_situation([r.to_dict() for r in results.results])
                    st.markdown("#### Situation Summary")
                    st.markdown(summary)


def main():
    """Main application entry point."""
    # Initialize components
    memory_engine = init_memory_engine()
    cortex = init_cortex(memory_engine)
    llm_client = init_perplexity()
    
    # Render header
    render_header()
    
    # Render sidebar
    search_limit, similarity_threshold, offline_mode = render_sidebar(memory_engine, llm_client)
    
    # Update settings if changed
    if llm_client and offline_mode != settings.OFFLINE_MODE:
        llm_client.offline_mode = offline_mode
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["🔍 Search", "📤 Upload", "⚡ Quick Actions"])
    
    with tab1:
        render_search_section(cortex, llm_client, search_limit, similarity_threshold)
    
    with tab2:
        ingestor = init_ingestor(memory_engine, llm_client)
        render_upload_section(ingestor, memory_engine)
    
    with tab3:
        render_quick_actions(cortex, llm_client)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666; font-size: 0.8rem;'>"
        "Project Sentinel V2 | Multimodal RAG for Disaster Response | "
        f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()

# 🛰️ Project Sentinel V2
### *AI-Powered Multimodal Disaster Response System*

![Sentinel Banner](https://img.icons8.com/color/480/satellite-sending-signal.png)

> **"Preservation of Life through Intelligent Intelligence."**

Project Sentinel V2 is a production-grade, offline-capable **RAG (Retrieval-Augmented Generation)** system designed for rapid disaster response. It ingests, analyzes, and synthesizes multimodal data (images, audio, and documents) to provide actionable intelligence for rescue teams.

---

## 🌟 Key Features

### 🧠 Multimodal "Cortex" Engine
*   **Visual Recon**: Ingests and indexes satellite imagery and drone footage using **CLIP (ViT-B/32)** embeddings.
*   **Audio Intelligence**: Transcribes emergency calls and radio feeds using **Whisper** and analyzes them for stress levels.
*   **Semantic Archive**: Indexes disaster reports, PDFs, and field manuals for semantic search.
*   **Dual-Vector Search**: Fixes dimension mismatches by intelligently switching between 384-dim (text) and 512-dim (visual) vectors.

### 🛡️ Mission Control Dashboard
A premium, "Glassmorphism" UI built with Streamlit:
*   **🤖 AI Situation Assessment**: Uses **Perplexity (Sonar Pro)** to generate real-time situation reports based on retrieved evidence.
*   **🔍 Verified Intelligence**: Deep-dive into search results with expandable cards showing full transcripts, high-res images, and confidence scores.
*   **📸 Visual First**: Images are treated as first-class citizens, displayed natively in search results.
*   **🆘 Civilian LIFELINE**: A dedicated protocol to instantly filter for human distress signals ("trapped", "help me").

### ⚡ Production Ready
*   **Offline-First Architecture**: Core search and analysis works without internet access using local embeddings.
*   **Universal Ingestor**: Drag-and-drop ingestion for images, audio, and documents via the UI.
*   **Persistent Memory**: Powered by **Qdrant**, enabling long-term storage and retrieval of intelligence.

---

## 🚀 Quick Start Guide

### Prerequisites
*   Python 3.10+
*   Docker (for Qdrant)
*   Top-tier bravery 🫡

### 1. Installation
Clone the repository and install dependencies:

```bash
git clone https://github.com/MagnumSolis/project_sentinel_v2.git
cd project_sentinel_v2

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 2. Configuration
Create a `.env` file (see `.env.example`). For full AI features, add your Perplexity API key:

```ini
QDRANT_HOST=localhost
QDRANT_PORT=6333
PERPLEXITY_API_KEY=your_key_here  # Optional: Set OFFLINE_MODE=True if missing
OFFLINE_MODE=False
```

### 3. Launch System
Use the all-in-one pipeline script to start Qdrant, check dependencies, and validate the system:

```bash
python3 run_pipeline.py
```

*This script will automatically verify Qdrant connectivity, check vector dimensions, and offer to launch the dashboard.*

### 4. Manual Dashboard Launch
If the system is already running:

```bash
streamlit run app/ui/dashboard.py
```

---

## 🎮 user Manual

### The Modules

1.  **Mission Control**: 
    *   High-level metrics: System integrity, total indexed vectors, and media counts.
    *   Live Feed: Recent incoming data streams.

2.  **Intelligence (Search)**:
    *   **Query**: Ask natural language questions like *"Where is the flooding most severe?"*
    *   **AI Assessment**: Read the auto-generated summary at the top.
    *   **Deep Dive**: Click the `>` arrow on any result to see the full document text or full-size image.

3.  **Civilian LIFELINE**:
    *   Click **"Scan All Frequencies"** to prioritize saving lives.
    *   Filters specifically for high-stress audio and keywords like "trapped" or "medical emergency".

4.  **Ingestion Hub**:
    *   Drag and drop files (Images, WAV/MP3, PDF, TXT) to add them to the knowledge base.
    *   They are instantly indexed and searchable.

---

## 🛠️ Architecture

```mermaid
graph TD
    User[User] --> UI[Streamlit Dashboard]
    UI --> Ingest[Universal Ingestor]
    UI --> Cortex[Sentinel Cortex]
    
    Ingest --> TextEmb[FastEmbed (Text)]
    Ingest --> VisionEmb[FastEmbed (Vision)]
    Ingest --> Whisper[Whisper ASR]
    
    Cortex --> Qdrant[(Qdrant Vector DB)]
    Cortex --> Perplexity[Perplexity LLM]
    
    Qdrant -- Semantic/Audio Nodes --> Cortex
    Qdrant -- Visual Nodes --> Cortex
    
    Perplexity -- RAG Summary --> UI
```

## ⚠️ Troubleshooting

**"Vector Dimension Error (512 vs 384)"**:
*   *Cause*: Mismatch between text query model and image embedding model.
*   *Fix*: This is **SOLVED** in V2. The `SentinelCortex` automatically uses `clip-ViT-B-32-text` (512-dim) for image searches and `bge-small-en-v1.5` (384-dim) for everything else.

**"Perplexity API Error"**:
*   Ensure `OFFLINE_MODE=False` in `.env` and your API key is valid. The system falls back to template responses gracefully if the API fails.

---

**Project Sentinel V2** - *Because every second counts.*

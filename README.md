# 🛰️ Project Sentinel V2
### *AI-Powered Multimodal Disaster Response System*

![Sentinel Banner](file:///home/magnum-solis/.gemini/antigravity/brain/0ca3a36b-2a11-44c3-b0d6-d81c583078bb/project_sentinel_banner_1769025081332.png)

> **"Preservation of Life through Intelligent Intelligence."**

**Project Sentinel V2** is a production-grade, offline-capable **RAG (Retrieval-Augmented Generation)** system designed for rapid disaster response. It ingests, analyzes, and synthesizes multimodal data (images, audio, and documents) to provide actionable intelligence for rescue teams in real-time.

---

## 🌟 Key Features

### 🧠 Multimodal "Cortex" Engine
*   **Visual Recon**: Ingests and indexes satellite imagery and drone footage using **CLIP (ViT-B/32)** embeddings.
*   **Audio Intelligence**: Transcribes emergency calls and radio feeds using **Whisper**, analyzing them for stress levels and keywords.
*   **Semantic Archive**: Indexes disaster reports, PDFs, and field manuals for deep semantic search.
*   **Dual-Vector Architecture**: Seamlessly handles diverse vector spaces (384-dim text vs 512-dim vision) for accurate cross-modal retrieval.

### 🛡️ Mission Control Dashboard
A premium, **"Glassmorphism" UI** built for high-stakes environments:
*   **🤖 Hybrid AI Assessment**: Dynamically switches between **Perplexity (Sonar Pro)** for online grounded insights and a **Local Cortex** for offline template-based analysis.
*   **🐝 Interactive Drone Mesh**: Simulates real-time data uplink from field drones, complete with "New Intel" alerts and progressive download visualization.
*   **🔍 Verified Intelligence**: Deep-dive into search results with expandable cards, showing full transcripts, high-res images, and confidence scores.
*   **🆘 Civilian LIFELINE**: A dedicated "Red Button" protocol to instantly filter all data streams for signs of human distress ("help me", "screaming", "trapped").

### ⚡ Operational Resilience
*   **Offline-First Design**: The core vector search and local analysis engine functions entirely without internet access.
*   **Messy Data Ready**: Handles varied inputs—distorted audio, grainy images, and scanned PDFs.
*   **Mass Population Tool**: Includes `scripts/mass_populate.py` to generate hundreds of mock scenarios (Floods, Earthquakes, Biohazards) for robust testing and demos.

---

## 🚀 Quick Start Guide

### Prerequisites
*   **Python 3.10+**
*   **Docker** (for Qdrant vector database)
*   **FFmpeg** (for audio processing)

### 1. Installation
Clone the repository and set up your environment:

```bash
git clone https://github.com/MagnumSolis/project_sentinel_v2.git
cd project_sentinel_v2

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration
Create a `.env` file for your API keys (optional for basic usage):

```ini
QDRANT_HOST=localhost
QDRANT_PORT=6333
PERPLEXITY_API_KEY=your_key_here  # Required for "Online" mode AI summaries
OFFLINE_MODE=True                 # Set to False to enable Perplexity by default
```

### 3. Initialize & Populate
Use the all-in-one pipeline to start the database and seed it with rich data:

```bash
python3 run_pipeline.py
```

> **Pro Tip:** To create a massive dataset for demos, stop the pipeline and run:
> ```bash
> python3 scripts/mass_populate.py
> ```
> Then restart `run_pipeline.py`.

### 4. Launch Mission Control
If the pipeline script didn't auto-launch the UI:

```bash
streamlit run app/ui/dashboard.py
```

---

## 🎮 Demo Guide: The "Drone Sync"
Project Sentinel V2 features a simulated drone uplink for presentations.

1.  **Status**: In the sidebar, check the **"🐝 Swarm Network"** section. Initially, it may say "Scanning...".
2.  **Trigger Event**: Click the **"🔄 Demo: Reset Drone Data"** button. This simulates a drone returning to range.
3.  **The Alert**: A distinctive **"🚨 New Intel Found"** alert will appear.
4.  **Uplink**: Click **"⬇️ INITIATE DOWNLOAD"**.
5.  **Watch It Happen**: Progress bars will visualize the data transfer and real-time ingestion into the vector database.
6.  **Verify**: Go to the **Intelligence** tab and search for *"quarantine"* to see the newly ingested drone imagery.

---

## 🛠️ System Architecture

```mermaid
graph TD
    User[Mission Commander] --> UI[Streamlit Dashboard]
    
    subgraph "Ingestion Layer"
        UI --> Ingest[Universal Ingestor]
        Drone[Drone Swarm Mesh] --> Ingest
        Mass[Mass Populator] --> Ingest
    end
    
    subgraph "Processing Core"
        Ingest --> TextEmb["FastEmbed (Text 384d)"]
        Ingest --> VisionEmb["FastEmbed (Vision 512d)"]
        Ingest --> Whisper[Whisper ASR]
    end
    
    subgraph "Memory & Intelligence"
        TextEmb --> Qdrant[("Qdrant Vector DB")]
        VisionEmb --> Qdrant
        
        Qdrant <--> Cortex[Sentinel Cortex Engine]
        
        Cortex --> Perplexity[Perplexity LLM (Online)]
        Cortex --> Local[Template Engine (Offline)]
    end
    
    Perplexity --> UI
    Local --> UI
```

## 📂 Project Structure

| Directory | Description |
|-----------|-------------|
| `app/ui/` | **Streamlit Dashboard**: The glassmorphic frontend (`dashboard.py`) and drone sync logic (`drone_sync.py`). |
| `app/core/` | **Backend Logic**: `memory_engine.py` (Qdrant), `cortex.py` (Search Logic), and `swarm.py`. |
| `app/ingestion/` | **Data Pipeline**: `universal_ingestor.py` handles PDF parsing, audio transcription, and embeddings. |
| `app/llm/` | **Intelligence**: `perplexity_integration.py` manages the Online/Offline AI toggle. |
| `scripts/` | **Utilities**: `mass_populate.py` (Data Gen), `run_pipeline.py` (Orchestrator). |
| `data/` | **Storage**: Local Qdrant storage, raw datasets, and drone cache. |

---

## ⚠️ Troubleshooting

*   **"Qdrant Connection Refused"**: Ensure Docker is running (`docker ps`). If not, run `docker-compose up -d`.
*   **"Perplexity Connection Failed"**: Check your `.env` key. The UI will show a yellow warning and fall back to local mode automatically.
*   **"No Drone Data"**: Click the "Reset Drone Data" button in the sidebar to re-stage the demo files.

---

**Project Sentinel V2** — *Because every second counts.*

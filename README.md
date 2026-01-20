# Project Sentinel V2

> **Production-Grade Multimodal RAG System for Offline-First Disaster Response**

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![Qdrant](https://img.shields.io/badge/vector_db-qdrant-green.svg)
![Streamlit](https://img.shields.io/badge/ui-streamlit-red.svg)
![License](https://img.shields.io/badge/license-MIT-purple.svg)

---

## 🎯 Overview

Project Sentinel V2 is a multimodal Retrieval-Augmented Generation (RAG) system designed for disaster response operations. It ingests and semantically indexes:

- **📄 PDF Documents** - Government reports, damage assessments, situation updates
- **🖼️ Satellite/UAV Images** - Aerial imagery with AI-powered damage analysis
- **🎙️ Emergency Audio** - Rescue team communications with stress level detection

All data is embedded into a unified vector space, enabling rescue teams to find correlated information across modalities - for example, linking a distress call to satellite imagery of the same location.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **Multimodal Search** | Query across documents, images, and audio simultaneously |
| **Offline-First** | All AI models run locally; Gemini API is optional |
| **Time-Decay Scoring** | Recent data is prioritized in search results |
| **Damage Boosting** | Severe damage findings are ranked higher |
| **Stress Detection** | Audio transcripts include prosodic stress analysis |
| **Real-Time Dashboard** | Streamlit UI with file upload and visualization |
| **Binary Quantization** | 32x compression for image vectors |

---

## 🚀 Quick Start (< 10 minutes)

### Prerequisites

- **Linux** (Ubuntu 20.04+ recommended)
- **Python 3.10+**
- **Docker & Docker Compose v2**
- **FFmpeg** (for audio processing)
- **16GB RAM minimum**
- **20GB free disk space**

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-org/project-sentinel-v2.git
cd project-sentinel-v2

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install FFmpeg (if not already installed)
sudo apt update && sudo apt install -y ffmpeg

# 5. Configure environment
cp .env.example .env
# Edit .env to add your GEMINI_API_KEY (optional)
```

### Start the System

```bash
# Run the initialization pipeline (starts Qdrant, creates collections, ingests sample data)
python run_pipeline.py
```

The script will:
1. ✅ Start Qdrant via Docker
2. ✅ Create vector collections
3. ✅ Generate sample disaster data
4. ✅ Ingest data into Qdrant
5. ✅ Validate search functionality
6. ✅ Launch the Streamlit dashboard

### Access the Dashboard

Open your browser to: **http://localhost:8501**

---

## 📁 Project Structure

```
project_sentinel_v2/
├── app/
│   ├── config.py              # Pydantic settings management
│   ├── core/
│   │   ├── memory_engine.py   # Qdrant collection management
│   │   └── cortex.py          # Search and retrieval engine
│   ├── ingestion/
│   │   └── universal_ingestor.py  # PDF/Image/Audio ingestion
│   ├── llm/
│   │   └── gemini_integration.py  # Gemini API with offline fallback
│   └── ui/
│       └── dashboard.py       # Streamlit interface
├── scripts/
│   ├── download_datasets.py   # Sample data generation
│   └── initialize_db.py       # Database population
├── data/
│   ├── raw_datasets/          # Source files (PDFs, images, audio)
│   ├── processed/             # Processed outputs
│   └── qdrant_storage/        # Qdrant persistence
├── tests/
│   └── test_core.py           # Unit tests
├── docker-compose.yml         # Qdrant deployment
├── requirements.txt           # Python dependencies
├── .env.example               # Environment template
├── run_pipeline.py            # Main orchestrator
└── README.md                  # This file
```

---

## 🧠 Architecture

### Three-Tier Memory System

Inspired by hippocampal memory encoding:

| Collection | Purpose | Embedding Model | Quantization |
|------------|---------|-----------------|--------------|
| `sentinel_semantic` | Official documents | bge-small-en-v1.5 (384d) | None |
| `sentinel_episodic` | Satellite/UAV images | jina-clip-v1 (768d) | Binary |
| `sentinel_audio` | Emergency calls | bge-small-en-v1.5 (384d) | Binary |

### Data Flow

```
Raw Data → Type Detection → Embedding → Qdrant Upsert
    ↓
PDF → pypdf → Chunking (500 chars) → Text Embedding → sentinel_semantic
    ↓
Image → PIL → Optional Gemini Vision → Visual Embedding → sentinel_episodic
    ↓
Audio → FFmpeg → Whisper → Stress Analysis → Text Embedding → sentinel_audio
```

### Search Pipeline

```
User Query → Text Embedding → Parallel Search (3 collections)
    ↓
Results ← Time Decay ← Damage Boost ← Score Fusion
    ↓
Context Formatting → Gemini Response (or offline template)
```

---

## ⚙️ Configuration

All settings are in `.env`:

```bash
# Qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6333

# Gemini (optional - leave empty for offline mode)
GEMINI_API_KEY=your-api-key-here
OFFLINE_MODE=True  # Set to False to use Gemini

# Models
TEXT_EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
VISION_EMBEDDING_MODEL=jinaai/jina-clip-v1
WHISPER_MODEL=base

# Search
SEARCH_LIMIT=5
SEARCH_SIMILARITY_THRESHOLD=0.6
TIME_DECAY_FACTOR=0.95
```

---

## 📊 Dashboard Features

### Search Tab
- Natural language queries across all modalities
- Collection toggles (Documents, Images, Audio)
- AI-powered response generation with citations
- Relevance scores and source attribution

### Upload Tab
- Drag-and-drop file upload
- Supports: PDF, JPG, PNG, MP3, WAV
- Real-time ingestion progress
- Automatic collection routing

### Quick Actions Tab
- **High Priority Cases**: Find high-stress audio transcripts
- **Severe Damage**: Locate images with damage scores > 70%
- **Situation Report**: Generate AI summary of current data

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=app --cov-report=html
```

---

## 🔧 Troubleshooting

### Qdrant Connection Failed
```bash
# Check if container is running
docker ps | grep qdrant

# Restart Qdrant
docker compose down && docker compose up -d

# Check logs
docker logs sentinel_qdrant
```

### FFmpeg Not Found
```bash
sudo apt update && sudo apt install -y ffmpeg
```

### Out of Memory (Whisper)
```bash
# Use smaller model in .env
WHISPER_MODEL=tiny
```

### Slow First Search
First search loads embedding models into memory (~30 seconds). Subsequent searches are <500ms.

---

## 📈 Performance

| Metric | Target | Typical |
|--------|--------|---------|
| Search Latency (p95) | <500ms | ~200ms |
| Ingestion (PDF, 10 pages) | <30s | ~15s |
| Ingestion (Image) | <5s | ~2s |
| Ingestion (Audio, 1 min) | <60s | ~45s |
| Memory Usage | <4GB | ~2.5GB |

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

---

## 📜 License

MIT License - see [LICENSE](LICENSE) file.

---

## 🙏 Acknowledgments

- **Qdrant** - High-performance vector database
- **FastEmbed** - Lightweight ONNX embeddings
- **OpenAI Whisper** - Speech recognition
- **Google Gemini** - Multimodal LLM
- **LADI Dataset** - Low-altitude disaster imagery

---

<div align="center">
  <strong>Built for disaster responders, by developers who care.</strong>
  <br><br>
  🛰️ Project Sentinel V2 | Save Lives with Data
</div>

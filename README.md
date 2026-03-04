# RAG System

A Retrieval-Augmented Generation (RAG) system for document question-answering. Process documents (PDF, TXT, MD, DOCX, images) and query them using natural language with an LLM-powered answering system.

## Overview

RAG System combines document processing, vector embeddings, and large language models to enable semantic search and Q&A over your documents. It supports:

- **Multi-format document ingestion**: PDF, TXT, MD, DOCX, PNG, JPG, JPEG, GIF, HEIC, HEIF
- **OCR support**: Extract text from images using PaddleOCR
- **Vector embeddings**: Semantic search using BAAI/bge-small-en-v1.5
- **LLM integration**: Powered by Ollama for natural language answers

## Features

- **Document Processing**: Automatically extracts text from various document formats
- **OCR for Images**: PaddleOCR handles scanned PDFs and images
- **Chunking**: Smart text chunking for optimal retrieval
- **Vector Storage**: Qdrant-powered vector database for semantic search
- **LLM Answers**: Generate contextual answers using local LLM (Ollama)
- **CLI Interface**: Easy command-line usage with ingest/query modes
- **Interactive Mode**: Chat with your documents in an interactive session
- **REST API**: Expose RAG functionality via FastAPI

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        RAG System                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐    ┌──────────────┐    ┌─────────────────────┐    │
│  │ Document │───▶│   Processor  │───▶│  Text Extraction    │    │
│  │  Input   │    │  (PDF/OCR)   │    │  & Cleaning         │    │
│  └──────────┘    └──────────────┘    └──────────┬──────────┘    │
│                                                 │               │
│                                                 ▼               │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                   Chunking & Embedding                   │   │
│  │              (Sentence Transformers - bge-small)         │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                  │              │
│                                                  ▼              │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                   Vector Store (Qdrant)                  │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                  │              │
│                                                  ▼              │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                      Query Flow                          │   │
│  │  User Query ──▶ Embed ──▶ Vector Search ──▶ Context      │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                  │              │
│                                                  ▼              │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                 LLM Generation (Ollama)                  │   │
│  │         Context + Question ──▶ Answer Generation         │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
rag-system/
├── api.py                 # FastAPI server for REST endpoints
├── cli.py                 # Command-line interface
├── config.py              # Configuration settings
├── rag_engine.py          # Core RAG engine implementation
├── manage_ollama.py       # Ollama management utilities
├── requirements.txt      # Python dependencies
├── .gitignore            # Git ignore rules
├── README.md             # This file
│
├── data/                 # Input documents (user-managed)
│   ├── vector_db/        # Chroma vector database storage
│   ├── test_doc.txt     # Sample document
│   ├── python_guide.md  # Sample document
│   ├── javascript_tutorial.docx
│   ├── typescript_info.jpg
│   └── react_info.png
│
└── qdrant_data/          # Qdrant vector database storage
```

## Prerequisites

Before installing the RAG System, ensure you have the following:

### Required External Services

1. **Ollama** - Local LLM runtime
   - Install: https://github.com/ollama/ollama
   - Model: `qwen3:4b` (configured by default)
   - Start: `ollama serve`

2. **Qdrant** - Vector database (embedded mode, runs automatically)
   - No separate installation needed
   - Uses local storage in `qdrant_data/`

### System Dependencies

- **macOS**: `brew install poppler tesseract` (for PDF/OCR processing)
- **Linux**: `apt-get install poppler-utils tesseract-ocr`

## Installation

```bash
pip install rag-system
```

### System Dependencies

Before running, install system dependencies for PDF/OCR processing:

- **macOS**: `brew install poppler tesseract`
- **Linux**: `apt-get install poppler-utils tesseract-ocr`

### Start Ollama

The RAG system uses Ollama for LLM-powered answers. Start it before use:

```bash
ollama serve
ollama pull qwen3:4b  # First time only
```

### From Source (Development)

```bash
pip install -e /path/to/rag-system
```

## Usage

### CLI Commands

#### Ingest Documents

Add documents to the knowledge base:

```bash
# Single file
python cli.py ingest path/to/document.pdf

# Multiple files
python cli.py ingest file1.pdf file2.txt file3.md

# Directory (recursively finds all supported files)
python cli.py ingest path/to/folder/

# Glob pattern
python cli.py ingest "*.pdf"
```

Supported file types: `.pdf`, `.txt`, `.md`, `.docx`, `.png`, `.jpg`, `.jpeg`, `.gif`, `.heic`, `.heif`

#### Query the System

Ask questions about your documents:

```bash
python cli.py query "What is the main topic of the document?"
python cli.py query "How do I configure the application?" -s  # Show sources
```

#### Interactive Mode

Chat with your documents:

```bash
python cli.py interactive
# or
python cli.py i
```

Commands in interactive mode:
- `:ingest <file>` - Add a document
- `:query <question>` - Ask a question
- `:sources on/off` - Toggle source display
- `:quit` or `:q` - Exit

#### Start API Server

```bash
python cli.py server
# Then run:
python api.py
```

The API will be available at `http://localhost:8000`

### API Endpoints

When running the API server:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/ingest` | POST | Ingest a document |
| `/query` | POST | Query the RAG system |
| `/health` | GET | Health check |

#### Example API Usage

```bash
# Ingest document
curl -X POST -F "file=@document.pdf" http://localhost:8000/ingest

# Query
curl -X POST -H "Content-Type: application/json" \
  -d '{"question": "What is this about?"}' \
  http://localhost:8000/query
```

## Configuration

Edit `config.py` to customize settings:

```python
class Config:
    # Ollama settings
    OLLAMA_MODEL = "qwen3:4b"          # LLM model
    OLLAMA_TIMEOUT = 300               # LLM unload after 5 min
    OLLAMA_STOP_TIMEOUT = 900          # Stop Ollama after 15 min

    # Embedding settings
    EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
    EMBEDDING_PERSIST = False          # Keep model in memory

    # Chunking settings
    CHUNK_SIZE = 512
    CHUNK_OVERLAP = 50

    # Retrieval settings
    TOP_K_DOCS = 3
    TOP_K_SECTIONS = 2
    TOP_K_CHUNKS = 2

    # API settings
    API_HOST = "0.0.0.0"
    API_PORT = 8000
```

## Dependencies

### Core

- `sentence-transformers>=2.2.0` - Text embeddings
- `qdrant-client>=1.7.0` - Vector database
- `paddlepaddle>=2.5.0` - Deep learning framework
- `paddleocr>=2.7.0` - OCR for images/scanned docs
- `ftfy>=6.1.0` - Text encoding fixes
- `pdf2image>=1.16.0` - PDF to image conversion
- `python-docx>=0.8.0` - Word document parsing
- `PyPDF2>=3.0.0` - PDF text extraction

### API & Server

- `fastapi>=0.100.0` - Web framework
- `uvicorn>=0.23.0` - ASGI server

### Utilities

- `tqdm>=4.65.0` - Progress bars
- `pyyaml>=6.0` - YAML config
- `psutil>=5.9.0` - System monitoring

## Troubleshooting

### Ollama Not Running

```
Error: Ollama service not available
```

**Solution**: Start Ollama with `ollama serve`

### Qdrant Connection Error

```
Error: Could not connect to Qdrant
```

**Solution**: Ensure Qdrant is accessible or check `QDRANT_HOST`/`QDRANT_PORT` in config

### Out of Memory

If you encounter memory issues:
1. Set `EMBEDDING_PERSIST = False` in config.py
2. Use a smaller LLM model
3. Reduce `CHUNK_SIZE`

### OCR Not Working

Ensure system dependencies are installed:
- macOS: `brew install poppler tesseract`
- Linux: `apt-get install poppler-utils tesseract-ocr`

## Development

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run tests
pytest

# With coverage
pytest --cov=. --cov-report=term-missing
```

### Linting

```bash
# Install linting tools
pip install ruff black mypy

# Run linter
ruff check .

# Format code
black .

# Type checking
mypy .
```

## License

MIT License

"""RAG System Configuration"""

import os
from pathlib import Path


class Config:
    # Paths
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    VECTOR_DB_DIR = DATA_DIR / "vector_db"
    
    # Create directories if not exist
    VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)

    # Ollama settings
    OLLAMA_MODEL = "qwen3:4b"
    OLLAMA_TIMEOUT = 300  # 5 minutes - LLM unload after inactivity
    OLLAMA_STOP_TIMEOUT = 900  # 15 minutes - stop Ollama after inactivity

    # Embedding settings
    EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
    EMBEDDING_PERSIST = False  # Keep embedding model loaded after ingestion

    # Chunking settings
    CHUNK_SIZE = 512
    CHUNK_OVERLAP = 50

    # Retrieval settings
    TOP_K_DOCS = 3
    TOP_K_SECTIONS = 2
    TOP_K_CHUNKS = 2

    # Qdrant settings
    QDRANT_HOST = "localhost"
    QDRANT_PORT = 6333
    COLLECTION_NAME = "rag_documents"

    # API settings
    API_HOST = "0.0.0.0"
    API_PORT = 8000


config = Config()

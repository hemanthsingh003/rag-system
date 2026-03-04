"""FastAPI Interface for RAG System"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import uvicorn

from rag_engine import get_engine


app = FastAPI(
    title="RAG System API",
    description="API for document ingestion and querying",
    version="1.0.0"
)


class IngestRequest(BaseModel):
    file_path: str


class QueryRequest(BaseModel):
    question: str
    show_sources: bool = False


class Source(BaseModel):
    title: str
    text: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]


@app.on_event("startup")
async def startup_event():
    """Initialize RAG engine on startup"""
    print("[API] Initializing RAG Engine...")
    engine = get_engine()
    print("[API] Ready")


@app.post("/ingest", summary="Ingest a document")
async def ingest_document(request: IngestRequest):
    """Ingest a document into the RAG system"""
    try:
        engine = get_engine()
        result = engine.ingest(request.file_path)
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class BatchIngestRequest(BaseModel):
    file_paths: List[str]


@app.post("/ingest/batch", summary="Ingest multiple documents")
async def batch_ingest_documents(request: BatchIngestRequest):
    """Ingest multiple documents into the RAG system"""
    try:
        engine = get_engine()
        result = engine.ingest_batch(request.file_paths)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", summary="Query the RAG system")
async def query_document(request: QueryRequest):
    """Query the RAG system with a question"""
    try:
        engine = get_engine()
        result = engine.query(request.question)
        
        return QueryResponse(
            answer=result["answer"],
            sources=[
                Source(
                    title=src["title"],
                    text=src["text"],
                    score=src["score"]
                )
                for src in result.get("sources", [])
            ]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health", summary="Health check")
async def health_check():
    """Check if API is running"""
    return {"status": "healthy"}


def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the API server"""
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()

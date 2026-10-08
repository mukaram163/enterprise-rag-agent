import os
import re
import uuid
import json
import logging
from typing import List, Dict, Optional
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import psycopg2

from src.ingestion.schema import create_tables
from src.ingestion.indexer import DocumentIndexer
from src.retrieval.dense import DenseRetriever
from src.retrieval.sparse import SparseRetriever
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import CrossEncoderReranker
from src.generation.generator import GeminiGenerator
from src.ingestion.embeddings import EmbeddingGenerator
from src.ingestion.drive_watcher import DriveWatcher
from src.models.document import DocumentChunk

load_dotenv()

logger = logging.getLogger(__name__)

db_connection_string = os.getenv(
    "DATABASE_URL", 
    "postgresql://rag_user:rag_password@localhost:5432/enterprise_rag"
)

indexer = None
embedder = None
hybrid_retriever = None
generator = None
drive_watcher = None
bm25 = None
rerank = None


def fetch_document_chunks() -> List[DocumentChunk]:
    raw_chunks = indexer.get_all_chunks()
    chunks = []
    for c in raw_chunks:
        meta = c.get("metadata") if isinstance(c, dict) else getattr(c, "metadata", {})
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except Exception:
                meta = {}
        if not isinstance(meta, dict):
            meta = {}

        file_name = (
            meta.get("file_name") 
            or meta.get("source") 
            or (c.get("file_name") if isinstance(c, dict) else getattr(c, "file_name", None))
            or "unknown.pdf"
        )
        
        raw_page = meta.get("page_number")
        if raw_page is None:
            raw_page = c.get("page_number") if isinstance(c, dict) else getattr(c, "page_number", 0)
            
        try:
            page_number = int(raw_page) if raw_page is not None else 0
        except (ValueError, TypeError):
            page_number = 0

        chunk_id = c.get("id") if isinstance(c, dict) else getattr(c, "id", "")
        doc_id = (c.get("doc_id") if isinstance(c, dict) else getattr(c, "doc_id", None)) or file_name

        chunk_obj = DocumentChunk(
            id=chunk_id,
            doc_id=doc_id,
            chunk_id=chunk_id,
            file_id=file_name,
            file_name=file_name,
            page_number=page_number,
            text=c.get("text", "") if isinstance(c, dict) else getattr(c, "text", ""),
            embedding=c.get("embedding", []) if isinstance(c, dict) else getattr(c, "embedding", []),
            allowed_users=c.get("allowed_users", []) if isinstance(c, dict) else getattr(c, "allowed_users", []),
            metadata=meta
        )
        chunks.append(chunk_obj)
    return chunks


def extract_used_citations(generated_text: str, chunks: List[DocumentChunk]) -> List[Dict[str, int]]:
    """
    Parses single and grouped page markers (e.g., [p. 21] or [p. 21, p. 25]) 
    from generated text and maps them to retrieved chunk file names.
    """
    # Extract all digits following 'p.' within brackets
    raw_matches = re.findall(r'p\.\s*(\d+)', generated_text)
    if not raw_matches:
        return []

    page_to_file = {}
    for chunk in chunks:
        fn = getattr(chunk, "file_name", "unknown.pdf")
        pn = getattr(chunk, "page_number", None)
        if pn is not None:
            page_to_file[int(pn)] = fn

    used_citations = []
    seen_pages = set()

    for page_str in raw_matches:
        page_num = int(page_str)
        if page_num in page_to_file and page_num not in seen_pages:
            seen_pages.add(page_num)
            used_citations.append({
                "file_name": page_to_file[page_num],
                "page_number": page_num
            })

    return used_citations


@asynccontextmanager
async def lifespan(app: FastAPI):
    global indexer, embedder, hybrid_retriever, generator, drive_watcher, bm25, rerank
    try:
        logger.info("Initializing database schema...")
        create_tables(db_connection_string)

        logger.info("Initializing indexer, embedder, generator, and retrieval components...")
        indexer = DocumentIndexer(db_connection_string=db_connection_string)
        embedder = EmbeddingGenerator()
        
        all_chunks = fetch_document_chunks()
        bm25 = SparseRetriever(chunks=all_chunks)
        logger.info(f"BM25 initialized with {len(all_chunks)} chunks from database.")

        try:
            rerank = CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-6-v2")
            logger.info("Cross-Encoder Reranker successfully loaded.")
        except Exception as e:
            logger.error(f"Failed to initialize CrossEncoderReranker: {e}")
            rerank = None

        dense_retriever = DenseRetriever(db_connection_string=db_connection_string)
        hybrid_retriever = HybridRetriever(
            dense_retriever=dense_retriever, 
            sparse_retriever=bm25,
            reranker=rerank
        )
        
        api_key = os.getenv("GEMINI_API_KEY", "")
        generator = GeminiGenerator(api_key=api_key)

        def sync_bm25():
            global bm25
            if bm25 is not None:
                updated_chunks = fetch_document_chunks()
                if hasattr(bm25, "fit"):
                    bm25.fit(updated_chunks)
                elif hasattr(bm25, "chunks"):
                    bm25.chunks = updated_chunks
                logger.info(f"BM25 index re-fitted dynamically with {len(updated_chunks)} total chunks.")

        try:
            drive_service = getattr(indexer, "drive_service", None)
            drive_watcher = DriveWatcher(
                drive_service=drive_service,
                indexer=indexer,
                on_new_file_callback=sync_bm25,
                poll_interval=15
            )
            if hasattr(drive_watcher, "start"):
                drive_watcher.start()
            logger.info("DriveWatcher initialized successfully.")
        except Exception as e:
            logger.warning(f"DriveWatcher failed to start: {e}")

        logger.info("All dependencies successfully initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize dependencies on startup: {e}")

    yield

    if drive_watcher and hasattr(drive_watcher, "stop"):
        try:
            drive_watcher.stop()
        except Exception as e:
            logger.warning(f"Error stopping DriveWatcher: {e}")
    logger.info("Shutting down application...")


app = FastAPI(title="Enterprise RAG API", lifespan=lifespan)

app.mount("/static", StaticFiles(directory="src/ui"), name="static")


@app.get("/")
async def serve_index():
    return FileResponse("src/ui/index.html")


class ChatRequest(BaseModel):
    query: str
    user_id: Optional[str] = None


class QueryRequest(BaseModel):
    query: str
    query_vector: List[float]
    user_id: Optional[str] = None


class FeedbackRequest(BaseModel):
    query: str
    response: str
    rating: int = Field(..., ge=1, le=5)
    comments: Optional[str] = None


class IngestChunk(BaseModel):
    id: str
    text: str
    embedding: List[float]
    allowed_users: Optional[List[str]] = []
    metadata: Optional[dict] = {}


class IngestRequest(BaseModel):
    doc_id: str
    chunks: List[IngestChunk]
    reindex: bool = False


@app.post("/query")
def query_endpoint(req: QueryRequest):
    missing = []
    if not hybrid_retriever: missing.append("hybrid_retriever")
    if not generator: missing.append("generator")

    if missing:
        raise HTTPException(
            status_code=503, 
            detail=f"Dependencies not initialized: {', '.join(missing)}"
        )
    try:
        chunks = hybrid_retriever.search(
            query=req.query, 
            query_vector=req.query_vector, 
            top_k=5, 
            user_id=req.user_id
        )
        response_text = generator.generate_response(query=req.query, context_chunks=chunks) if generator else ""
        return {"query": req.query, "response": response_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    missing = []
    if not embedder: missing.append("embedder")
    if not hybrid_retriever: missing.append("hybrid_retriever")
    if not generator: missing.append("generator")

    if missing:
        raise HTTPException(
            status_code=503, 
            detail=f"Dependencies not initialized: {', '.join(missing)}"
        )
    
    query = request.query

    try:
        query_embeddings = embedder.generate_embeddings([query])
        query_vector = query_embeddings[0] if query_embeddings else [0.0] * 384
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        query_vector = [0.0] * 384

    try:
        chunks = hybrid_retriever.search(
            query=query, 
            query_vector=query_vector, 
            top_k=5, 
            user_id=request.user_id
        )
    except Exception as e:
        logger.error(f"Hybrid retrieval failed: {e}")
        chunks = []

    async def event_generator():
        accumulated_response = ""
        try:
            async for token in generator.generate_stream(query, chunks):
                accumulated_response += token
                payload = json.dumps({"event": "token", "data": token})
                yield f"data: {payload}\n\n"

            # Parse page markers from full text and emit citations
            final_citations = extract_used_citations(accumulated_response, chunks)
            citation_payload = json.dumps({"event": "citations", "data": final_citations})
            yield f"data: {citation_payload}\n\n"

        except Exception as err:
            logger.error(f"Streaming error occurred: {err}")
            error_payload = json.dumps({
                "event": "error", 
                "data": "Something went wrong, try again"
            })
            yield f"data: {error_payload}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/ingest")
def ingest_endpoint(req: IngestRequest):
    if not indexer:
        raise HTTPException(
            status_code=503, 
            detail="Dependencies not initialized: indexer"
        )
    try:
        chunks_data = [chunk.dict() for chunk in req.chunks]
        count = indexer.reindex_document(doc_id=req.doc_id, chunks_data=chunks_data)

        if bm25 is not None:
            all_chunks = fetch_document_chunks()
            if hasattr(bm25, "fit"):
                bm25.fit(all_chunks)
            else:
                bm25.chunks = all_chunks
            logger.info(f"BM25 re-indexed with {len(all_chunks)} chunks.")

        return {"status": "success", "processed_chunks": count, "doc_id": req.doc_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/feedback")
def feedback_endpoint(req: FeedbackRequest):
    feedback_id = str(uuid.uuid4())
    sql = """
        INSERT INTO feedback (id, query, response, rating, comments)
        VALUES (%s, %s, %s, %s, %s);
    """
    try:
        with psycopg2.connect(db_connection_string) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (feedback_id, req.query, req.response, req.rating, req.comments))
        return {"status": "success", "feedback_id": feedback_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

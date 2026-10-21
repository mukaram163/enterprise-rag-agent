import os
import re
import uuid
import json
import time
import logging
import asyncio
from typing import List, Dict, Optional
from contextlib import asynccontextmanager

from src.utils.request_logger import log_request
from src.retrieval.acl_guard import verify_access
from src.utils.router import classify
from src.utils.cost_guard import check_ceiling
from src.utils.rate_limiter import RateLimiter
from src.utils.injection_guard import check_injection
from src.utils.pii_filter import redact
from src.utils.faithfulness_guard import check_faithfulness
from src.utils.block_log import log_block

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
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

session_limiter = RateLimiter(int(os.getenv("RATE_LIMIT_SESSION_PER_HOUR", "20")))
ip_limiter = RateLimiter(int(os.getenv("RATE_LIMIT_IP_PER_HOUR", "60")))

indexer = None
embedder = None
hybrid_retriever = None
generator = None
drive_watcher = None
bm25 = None
rerank = None


def build_pipeline():
    """
    Factory function to initialize indexer, embeddings, and retrievers.
    Does NOT start background processes like DriveWatcher.
    Raises an error immediately if EVAL_MODE=1 and any initialization fails.
    """
    eval_mode = os.getenv("EVAL_MODE", "0").lower() in ("1", "true", "yes")
    try:
        global indexer, embedder, hybrid_retriever, generator, bm25, rerank

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
            if eval_mode:
                raise RuntimeError(f"EVAL_MODE=1: Reranker initialization failed: {e}") from e
            logger.warning(f"Failed to initialize CrossEncoderReranker: {e}")
            rerank = None

        dense_retriever = DenseRetriever(db_connection_string=db_connection_string)
        hybrid_retriever = HybridRetriever(
            dense_retriever=dense_retriever, 
            sparse_retriever=bm25,
            reranker=rerank,
            access_checker=lambda chunks, user_id: verify_access(db_connection_string, chunks, user_id),
        )

        api_key = os.getenv("GEMINI_API_KEY", "")
        generator = GeminiGenerator(api_key=api_key)

        return hybrid_retriever
    except Exception as e:
        if eval_mode:
            raise RuntimeError(f"EVAL_MODE=1: Pipeline initialization failed: {e}") from e
        logger.error(f"Pipeline initialization failed: {e}")
        return None


def fetch_document_chunks() -> List[DocumentChunk]:
    if indexer is None:
        return []
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


def extract_used_citations(generated_text, chunks):
    valid = {}
    for chunk in chunks:
        fn = getattr(chunk, "file_name", "unknown.pdf")
        pn = getattr(chunk, "page_number", None)
        if pn is not None:
            valid[(fn, int(pn))] = True
    by_page = {}
    for (fn, pn) in valid:
        by_page.setdefault(pn, fn)

    used, seen = [], set()
    # [file.pdf p.21]  (file + page)
    for fn, pg in re.findall(r'\[([^\]\[]+?\.pdf)\s+p\.\s*(\d+)', generated_text):
        key = (fn.strip(), int(pg))
        if key in valid and key not in seen:
            seen.add(key)
            used.append({"file_name": key[0], "page_number": key[1]})
    # [p.21] or [p.21, p.25]  (page only: first file with that page)
    stripped = re.sub(r'\[[^\]\[]+?\.pdf\s+p\.\s*\d+[^\]]*\]', '', generated_text)
    for pg in re.findall(r'p\.\s*(\d+)', stripped):
        pn = int(pg)
        if pn in by_page and (by_page[pn], pn) not in seen:
            seen.add((by_page[pn], pn))
            used.append({"file_name": by_page[pn], "page_number": pn})
    return used


@asynccontextmanager
async def lifespan(app: FastAPI):
    global drive_watcher
    eval_mode = os.getenv("EVAL_MODE", "0").lower() in ("1", "true", "yes")

    try:
        logger.info("Initializing database schema...")
        create_tables(db_connection_string)

        # Initialize pipeline using build_pipeline
        build_pipeline()

        def sync_bm25():
            global bm25
            if bm25 is not None:
                updated_chunks = fetch_document_chunks()
                if hasattr(bm25, "fit"):
                    bm25.fit(updated_chunks)
                elif hasattr(bm25, "chunks"):
                    bm25.chunks = updated_chunks
                logger.info(f"BM25 index re-fitted dynamically with {len(updated_chunks)} total chunks.")

        # DriveWatcher is exclusively managed inside lifespan context
        try:
            from src.connectors.google_drive import GoogleDriveConnector
            drive_conn = GoogleDriveConnector()
            drive_service = drive_conn.service
            if indexer:
                indexer.drive_connector = drive_conn
                indexer.embedder = embedder
            drive_watcher = DriveWatcher(
                drive_service=drive_service,
                indexer=indexer,
                on_new_file_callback=sync_bm25,
                poll_interval=15
            )
            
            # Preload already processed document IDs from database before starting watcher
            with psycopg2.connect(db_connection_string) as conn, conn.cursor() as cur:
                cur.execute("SELECT DISTINCT doc_id FROM document_chunks")
                drive_watcher._processed_file_ids = {r[0] for r in cur.fetchall()}

            if hasattr(drive_watcher, "start"):
                drive_watcher.start()
            logger.info("DriveWatcher initialized successfully.")
        except Exception as e:
            logger.warning(f"DriveWatcher failed to start: {e}")

        logger.info("All dependencies successfully initialized.")
    except Exception as e:
        if eval_mode:
            raise RuntimeError(f"EVAL_MODE=1: Startup failed: {e}") from e
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
    session_id: Optional[str] = None


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
async def chat_stream(request: ChatRequest, http_request: Request):
    missing = []
    if not embedder: missing.append("embedder")
    if not hybrid_retriever: missing.append("hybrid_retriever")
    if not generator: missing.append("generator")

    if missing:
        raise HTTPException(
            status_code=503, 
            detail=f"Dependencies not initialized: {', '.join(missing)}"
        )

    ip = http_request.client.host if http_request.client else "unknown"
    for limiter, key, kind in (
        (session_limiter, request.session_id, "session"),
        (ip_limiter, ip, "ip"),
    ):
        if not key:
            continue
        ok, retry = limiter.check(key)
        if not ok:
            logger.warning(f"RATE LIMIT: {kind}={key} blocked, retry in {retry}s")
            log_block("rate_limit", f"{kind} limit hit", session=request.session_id, ip=ip)
            raise HTTPException(
                status_code=429,
                detail=f"Too many requests. Try again in {retry // 60 + 1} minutes.",
                headers={"Retry-After": str(retry)},
            )

    flagged, rules = check_injection(request.query)
    if flagged:
        log_block("injection", f"rules={rules} text={request.query[:100]!r}", session=request.session_id, ip=ip)
        raise HTTPException(
            status_code=400,
            detail="That request looks like an attempt to change my instructions, so I can't process it. Please ask a question about the documents.",
        )

    start_time = time.perf_counter()
    query = request.query
    tier, route_reasons = classify(query)
    tier, ceiling_note, blocked = check_ceiling(tier)
    if ceiling_note:
        route_reasons = route_reasons + [ceiling_note]
    if blocked:
        log_block("cost_ceiling", "hard limit reached", session=request.session_id, ip=ip)
        raise HTTPException(status_code=429, detail="Spending limit reached")

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

            safe_text, pii_counts = redact(accumulated_response)
            if pii_counts:
                log_block("pii", f"redacted {pii_counts}", session=request.session_id, ip=ip)
            
            if os.getenv("FAITHFULNESS_GUARD", "0") == "1":
                threshold = float(os.getenv("FAITHFULNESS_MIN", "0.5"))
                score = await check_faithfulness(query, safe_text, [c.text for c in chunks])
                if score is not None and score < threshold:
                    logger.warning(f"FAITHFULNESS BLOCK: score={score:.2f} < {threshold} question={query[:80]!r}")
                    log_block("faithfulness", f"score={score:.2f} question={query[:80]!r}", session=request.session_id, ip=ip)
                    safe_text = "I couldn't verify that answer against the documents, so I'm not showing it. Please rephrase or check the source pages."

            for i in range(0, len(safe_text), 40):
                piece = {"event": "token", "data": safe_text[i:i + 40]}
                yield f"data: {json.dumps(piece)}\n\n"
                await asyncio.sleep(0.02)

            final_citations = extract_used_citations(safe_text, chunks)
            citation_payload = json.dumps({"event": "citations", "data": final_citations})
            yield f"data: {citation_payload}\n\n"

            if getattr(generator, "last_provider", None) not in (None, generator.model_name):
                notice = {"event": "notice", "data": "Answered by a backup model, so page citations may be missing."}
                yield f"data: {json.dumps(notice)}\n\n"

            latency_ms = (time.perf_counter() - start_time) * 1000
            usage = getattr(generator, "last_usage", {}) or {}
            pages_cited = [c["page_number"] for c in final_citations if isinstance(c, dict) and "page_number" in c]
            log_request(
                question=query,
                model_used=getattr(generator, "last_provider", None) or generator.model_name,
                tokens_in=usage.get("tokens_in"),
                tokens_out=usage.get("tokens_out"),
                tokens_thinking=usage.get("tokens_thinking"),
                latency_ms=latency_ms,
                pages_cited=pages_cited,
                estimated_cost=None,
                tier=tier,
                route_reasons=route_reasons,
            )

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
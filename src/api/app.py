import json
import logging
from dotenv import load_dotenv
from typing import AsyncGenerator
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from src.generation.rag_chain import GeminiRAGGenerator
from src.retrieval.pipeline import RetrievalPipeline

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag-api")

app = FastAPI(
    title="Enterprise RAG Agent API",
    version="2.0.0",
    description="Streaming RAG API backed by PostgreSQL/pgvector and Gemini",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

generator = GeminiRAGGenerator()
retrieval_pipeline = RetrievalPipeline()


class ChatQueryRequest(BaseModel):
    query: str
    top_k: int = 3
    user_id: str = "default_user"


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "database": "postgres_pgvector",
        "stage": "Stage 2 - SSE Streaming"
    }


async def stream_rag_response(
    query: str, top_k: int, user_id: str
) -> AsyncGenerator[str, None]:
    try:
        retrieved_items = retrieval_pipeline.retrieve(
            query=query, candidate_k=top_k * 3, top_k=top_k, user_id=user_id
        )

        citations = [
            {
                "file_name": item.get("file_name", ""),
                "page_number": item.get("page_number", 1),
                "chunk_index": item.get("chunk_index", 0),
                "score": item.get("rerank_score", item.get("score", 0.0)),
            }
            for item in retrieved_items
        ]
        yield json.dumps({"event": "metadata", "citations": citations})

        async for chunk in generator.generate_response_stream(
            query=query, retrieved_items=retrieved_items
        ):
            yield json.dumps({"event": "delta", "text": chunk})

    except Exception as e:
        logger.error(f"Error during RAG streaming: {str(e)}", exc_info=True)
        yield json.dumps({"event": "error", "message": str(e)})


@app.post("/api/v1/chat/stream")
async def chat_stream_endpoint(request: ChatQueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    return EventSourceResponse(
        stream_rag_response(
            query=request.query, top_k=request.top_k, user_id=request.user_id
        )
    )
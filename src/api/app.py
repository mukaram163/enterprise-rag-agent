import json
import logging
from typing import AsyncGenerator
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from src.generation.rag_chain import GeminiRAGGenerator
from src.retrieval.hybrid_pipeline import HybridPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag-api")

app = FastAPI(
    title="Enterprise RAG Agent API",
    version="2.0.0",
    description="Streaming RAG API with pgvector and Gemini async generation",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instantiate models and pipeline components
generator = GeminiRAGGenerator()
retrieval_pipeline = HybridPipeline()


class ChatQueryRequest(BaseModel):
    query: str
    top_k: int = 5
    user_id: str = "default_user"


@app.get("/health")
async def health_check():
    return {"status": "ok", "database": "pgvector", "stage": "Stage 2 - SSE Streaming"}


async def stream_rag_response(
    query: str, top_k: int, user_id: str
) -> AsyncGenerator[str, None]:
    try:
        # 1. Retrieve top context items asynchronously
        retrieved_items = retrieval_pipeline.search(query=query, top_k=top_k)

        # Send retrieved metadata first (sources & citations)
        citations = [
            {
                "file_name": item.get("file_name"),
                "page_number": item.get("page_number"),
                "score": item.get("rerank_score", 0.0),
            }
            for item in retrieved_items
        ]
        yield json.dumps({"event": "metadata", "citations": citations})

        # 2. Stream generated answer chunks via Gemini Async Client
        async for chunk in generator.generate_response_stream(
            query=query, retrieved_items=retrieved_items
        ):
            yield json.dumps({"event": "delta", "text": chunk})

    except Exception as e:
        logger.error(f"Error during RAG streaming: {str(e)}")
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
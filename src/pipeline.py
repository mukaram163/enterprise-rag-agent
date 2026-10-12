import time
import logging
from typing import Any, Dict

from src.api import app as api
from src.utils.request_logger import log_request
from src.utils.router import classify

_ready = False


class _FallbackDetector(logging.Handler):
    """Flags when the generator silently fell back from Gemini streaming."""
    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.fell_back = False

    def emit(self, record):
        if "Falling back to sync response" in record.getMessage():
            self.fell_back = True


def init() -> None:
    global _ready
    if _ready:
        return
    if api.build_pipeline() is None:
        raise RuntimeError("Pipeline failed to initialize (see logs).")
    if api.rerank is None:
        raise RuntimeError("Reranker not loaded; refusing to evaluate without reranking.")
    _ready = True


async def run_rag(query: str, top_k: int = 5) -> Dict[str, Any]:
    init()
    start_time = time.perf_counter()
    tier, route_reasons = classify(query)

    embeddings = api.embedder.generate_embeddings([query])
    if not embeddings:
        raise RuntimeError("Embedding generation returned nothing.")

    chunks = api.hybrid_retriever.search(
        query=query, query_vector=embeddings[0], top_k=top_k, user_id=None
    )

    detector = _FallbackDetector()
    gen_logger = logging.getLogger("src.generation.generator")
    gen_logger.addHandler(detector)
    try:
        answer = ""
        async for token in api.generator.generate_stream(query, chunks):
            answer += token
    finally:
        gen_logger.removeHandler(detector)

    if answer.startswith("Error: All API providers"):
        raise RuntimeError(answer)

    citations = api.extract_used_citations(answer, chunks)
    provider = "fallback (not Gemini streaming)" if detector.fell_back else api.generator.model_name
    latency_ms = (time.perf_counter() - start_time) * 1000

    usage = getattr(api.generator, "last_usage", {}) or {}
    pages_cited = [c["page_number"] for c in citations if isinstance(c, dict) and "page_number" in c]

    log_request(
        question=query,
        model_used=provider,
        tokens_in=usage.get("tokens_in"),
        tokens_out=usage.get("tokens_out"),
        tokens_thinking=usage.get("tokens_thinking"),
        latency_ms=latency_ms,
        pages_cited=pages_cited,
        estimated_cost=None,
        source="eval",
        tier=tier,
        route_reasons=route_reasons,
    )

    return {
        "answer": answer,
        "reranked_chunks": [
            {"text": c.text, "file_name": c.file_name, "page_number": c.page_number}
            for c in chunks
        ],
        "contexts": [c.text for c in chunks],
        "citations": citations,
        "provider": provider,
    }
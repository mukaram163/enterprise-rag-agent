import os
from typing import Any, Dict, List

from mcp.server.fastmcp import FastMCP

from src import pipeline
from src.utils.chunk_guard import filter_chunks
from src.utils.chunk_guard import filter_chunks

mcp = FastMCP("rag-documents")

# Who the agents work for. Set by the app that starts this server, never by the model.
USER_ID = os.getenv("RAG_USER_ID") or None


@mcp.tool()
def search_documents(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Hybrid search (BM25 + vectors, reranked) over the indexed documents.
    Returns the best chunks with file name and page number for citation."""
    pipeline.init()
    api = pipeline.api
    top_k = max(1, min(int(top_k), 10))
    emb = api.embedder.generate_embeddings([query])
    chunks = api.hybrid_retriever.search(
        query=query, query_vector=emb[0], top_k=top_k, user_id=USER_ID
    )
    chunks, _removed = filter_chunks(chunks)  # drop chunks that try to instruct the AI
    chunks, _removed = filter_chunks(chunks)  # drop chunks that try to instruct the AI
    return [
        {
            "text": c.text,
            "file_name": c.file_name,
            "page_number": c.page_number,
            "citation": f"[{c.file_name} p.{c.page_number}]",
        }
        for c in chunks
    ]


if __name__ == "__main__":
    mcp.run()  # stdio transport
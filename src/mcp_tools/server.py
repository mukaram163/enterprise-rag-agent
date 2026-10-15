from typing import Any, Dict, List

from mcp.server.fastmcp import FastMCP

from src import pipeline

mcp = FastMCP("rag-documents")


@mcp.tool()
def search_documents(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Hybrid search (BM25 + vectors, reranked) over the indexed documents.
    Returns the best chunks with file name and page number for citation."""
    pipeline.init()
    api = pipeline.api
    top_k = max(1, min(int(top_k), 10))
    emb = api.embedder.generate_embeddings([query])
    chunks = api.hybrid_retriever.search(
        query=query, query_vector=emb[0], top_k=top_k, user_id=None
    )
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
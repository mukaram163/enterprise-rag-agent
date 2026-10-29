import os
import re
from typing import Any, Dict, List

from mcp.server.fastmcp import FastMCP

from src import pipeline
from src.utils.chunk_guard import filter_chunks

mcp = FastMCP("rag-documents")

# Who the agents work for. Set by the app that starts this server, never by the model.
USER_ID = os.getenv("RAG_USER_ID") or None

# Page expansion: when a page is found, the agent also gets that page's other chunks, so lists and tables
# are not cut into fragments. 0 = off; 3 = add the whole pages of the first 3 result pages.
# Set with the RAG_EXPAND_PAGES environment variable; RAG_EXPAND_MAX_CHARS caps the added text.
EXPAND_PAGES = int(os.getenv("RAG_EXPAND_PAGES", "3"))
EXPAND_MAX_CHARS = int(os.getenv("RAG_EXPAND_MAX_CHARS", "6000"))

_POS = re.compile(r"_p(\d+)_c(\d+)$")
_by_page = None


def _chunk_no(c):
    m = _POS.search(str(c.id))
    return int(m.group(2)) if m else 0


def _pages_index(api):
    global _by_page
    if _by_page is None:
        _by_page = {}
        for c in api.bm25.chunks:
            _by_page.setdefault((c.file_name, c.page_number), []).append(c)
        for chunks in _by_page.values():
            chunks.sort(key=_chunk_no)
    return _by_page


def _expand(api, chunks):
    """Add the other chunks of the best pages. Every added chunk must pass the permission check."""
    checker = api.hybrid_retriever.access_checker
    if EXPAND_PAGES <= 0 or checker is None or not chunks:  # no checker -> do not expand (fail closed)
        return chunks
    pages = []
    for c in chunks:
        key = (c.file_name, c.page_number)
        if key not in pages:
            pages.append(key)
    index = _pages_index(api)
    seen = {str(c.id) for c in chunks}
    extra = []
    for key in pages[:EXPAND_PAGES]:
        for c in index.get(key, []):
            if str(c.id) not in seen:
                seen.add(str(c.id))
                extra.append(c)
    extra = checker(extra, USER_ID)  # drop anything this user may not read
    out = list(chunks)
    total = sum(len(c.text) for c in out)
    for c in extra:
        if total + len(c.text) > EXPAND_MAX_CHARS:
            break
        out.append(c)
        total += len(c.text)
    return out


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
    chunks = _expand(api, chunks)  # whole-page context (does nothing unless expansion is switched on)
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

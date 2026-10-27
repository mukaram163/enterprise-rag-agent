import os
import re
from typing import Any, Dict, List

from mcp.server.fastmcp import FastMCP

from src import pipeline
from src.retrieval.acl_guard import verify_access
from src.retrieval.expand import PageExpander
from src.utils.chunk_guard import filter_chunks

mcp = FastMCP("rag-documents")

# Who the agents work for. Set by the app that starts this server, never by the model.
USER_ID = os.getenv("RAG_USER_ID") or None

# Whole-page context: when a page is found, the agent also gets that page's other chunks, so lists and
# tables are not cut into fragments. 0 = off (default). Set RAG_EXPAND_PAGES=3 to turn it on.
EXPAND_PAGES = int(os.getenv("RAG_EXPAND_PAGES", "0"))
_expander = None


def _expand(api, hits):
    global _expander
    if _expander is None:
        _expander = PageExpander(api.bm25.chunks)
    expanded = _expander.expand(hits, n_pages=EXPAND_PAGES)
    # the extra chunks did not come through the search, so they get the same final permission check
    return verify_access(api.indexer.db_connection_string, expanded, USER_ID)

# Page expansion (off by default). 0 = off; 3 = add the whole pages of the first 3 result pages.
EXPAND_PAGES = int(os.getenv("RAG_EXPAND_PAGES", "0"))
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
    if EXPAND_PAGES > 0:
        chunks = _expand(api, chunks)
    chunks = _expand(api, chunks)
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

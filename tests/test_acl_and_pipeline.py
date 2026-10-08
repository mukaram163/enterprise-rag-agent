import pytest
from src.retrieval.document import DocumentChunk
from src.retrieval.sparse import SparseRetriever

def test_sparse_retriever_acl_filtering():
    chunks = [
        DocumentChunk(id="c1", doc_id="d1", text="Public info on AI", embedding=[0.1]*384, allowed_users=[]),
        DocumentChunk(id="c2", doc_id="d2", text="Restricted info for user_a", embedding=[0.1]*384, allowed_users=["user_a"]),
        DocumentChunk(id="c3", doc_id="d3", text="Restricted info for user_b", embedding=[0.1]*384, allowed_users=["user_b"]),
    ]

    retriever = SparseRetriever(chunks=chunks)

    # Anonymous user request -> Should only see c1
    anon_results = retriever.search(query="info", user_id=None)
    assert len(anon_results) == 1
    assert anon_results[0].id == "c1"

    # Authorized user_a request -> Should see c1 and c2, but NOT c3
    user_a_results = retriever.search(query="info", user_id="user_a")
    retrieved_ids = [c.id for c in user_a_results]
    assert "c1" in retrieved_ids
    assert "c2" in retrieved_ids
    assert "c3" not in retrieved_ids
from src.models.document import DocumentChunk
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.sparse import SparseRetriever


def make_chunks():
    return [
        DocumentChunk(id="c1", doc_id="d1", text="Public info on AI", embedding=[0.1] * 384, allowed_users=[]),
        DocumentChunk(id="c2", doc_id="d2", text="Restricted info for user_a", embedding=[0.1] * 384, allowed_users=["user_a"]),
        DocumentChunk(id="c3", doc_id="d3", text="Restricted info for user_b", embedding=[0.1] * 384, allowed_users=["user_b"]),
    ]


def test_sparse_retriever_acl_filtering():
    retriever = SparseRetriever(chunks=make_chunks())

    anon = [c.id for c in retriever.search(query="info", user_id=None)]
    assert anon == ["c1"]  # anonymous users only see public chunks

    user_a = [c.id for c in retriever.search(query="info", user_id="user_a")]
    assert "c1" in user_a and "c2" in user_a
    assert "c3" not in user_a  # never another user's private chunk


class RecordingRetriever:
    """Stand-in retriever that remembers which user it was asked to search for."""

    def __init__(self):
        self.seen = []

    def search(self, *args, user_id=None, **kwargs):
        self.seen.append(user_id)
        return []


class PassThroughReranker:
    def rerank(self, query, chunks, top_k):
        return chunks[:top_k]


def test_hybrid_passes_user_id_to_both_retrievers():
    dense, sparse = RecordingRetriever(), RecordingRetriever()
    hybrid = HybridRetriever(dense_retriever=dense, sparse_retriever=sparse, reranker=PassThroughReranker())
    hybrid.search(query="info", query_vector=[0.1], top_k=3, user_id="alice@example.com")
    assert dense.seen == ["alice@example.com"]
    assert sparse.seen == ["alice@example.com"]  # was None before the fix: keyword search ignored the user


def test_hybrid_defaults_to_anonymous():
    dense, sparse = RecordingRetriever(), RecordingRetriever()
    hybrid = HybridRetriever(dense_retriever=dense, sparse_retriever=sparse, reranker=PassThroughReranker())
    hybrid.search(query="info", query_vector=[0.1], top_k=3)
    assert dense.seen == [None] and sparse.seen == [None]

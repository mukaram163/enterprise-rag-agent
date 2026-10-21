from src.models.document import DocumentChunk
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.sparse import SparseRetriever

# Each chunk has its own unique word on purpose: BM25 gives no weight to words found in most chunks.
STORED = [
    ("c1", "alpha public handbook page", {}, []),
    ("c2", "beta private memo for alice", {}, ["alice@example.com"]),
    ("c3", "gamma private memo for carol", {}, ["carol@example.com"]),
    ("c4", "delta filler text one", {}, []),
    ("c5", "epsilon filler text two", {}, []),
    ("c6", "zeta filler text three", {}, []),
]
QUERY = "alpha beta gamma"


def ids(results):
    return [c.id for c in results]


def make_chunks():
    return [DocumentChunk(id=i, doc_id=i, text=t, embedding=[0.1] * 384, allowed_users=u)
            for i, t, _, u in STORED]


def test_sparse_retriever_acl_filtering():
    retriever = SparseRetriever(chunks=make_chunks())

    anon = ids(retriever.search(query=QUERY, user_id=None))
    assert "c1" in anon  # public
    assert "c2" not in anon and "c3" not in anon  # anonymous never sees private chunks

    alice = ids(retriever.search(query=QUERY, user_id="alice@example.com"))
    assert "c1" in alice and "c2" in alice  # her own private chunk
    assert "c3" not in alice  # never someone else's


class FakeCursor:
    def __init__(self, table):
        self.table = table
        self.rows = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        if sql.strip().upper().startswith("SELECT"):
            cols = [c.strip() for c in sql.split("SELECT", 1)[1].split("FROM", 1)[0].split(",")]
            self.rows = [tuple(row[c] for c in cols) for row in self.table]

    def fetchall(self):
        return self.rows


class FakeConnection:
    def __init__(self, table):
        self.table = table

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def cursor(self):
        return FakeCursor(self.table)


def test_bm25_index_built_from_database_keeps_permissions(monkeypatch):
    """Regression: get_all_chunks() used to skip allowed_users, so keyword search saw every chunk as public."""
    import src.ingestion.indexer as indexer_module

    table = [{"id": i, "text": t, "metadata": m, "allowed_users": u} for i, t, m, u in STORED]
    monkeypatch.setattr(indexer_module.psycopg2, "connect", lambda *a, **k: FakeConnection(table))
    indexer = indexer_module.DocumentIndexer.__new__(indexer_module.DocumentIndexer)
    indexer.db_connection_string = "fake"

    rows = indexer.get_all_chunks()
    chunks = [DocumentChunk(id=r["id"], doc_id=r["id"], text=r["text"], allowed_users=r["allowed_users"])
              for r in rows]
    retriever = SparseRetriever(chunks=chunks)

    anon = ids(retriever.search(query=QUERY, user_id=None))
    assert "c2" not in anon and "c3" not in anon
    alice = ids(retriever.search(query=QUERY, user_id="alice@example.com"))
    assert "c2" in alice and "c3" not in alice


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
    assert sparse.seen == ["alice@example.com"]


def test_hybrid_defaults_to_anonymous():
    dense, sparse = RecordingRetriever(), RecordingRetriever()
    hybrid = HybridRetriever(dense_retriever=dense, sparse_retriever=sparse, reranker=PassThroughReranker())
    hybrid.search(query="info", query_vector=[0.1], top_k=3)
    assert dense.seen == [None] and sparse.seen == [None]

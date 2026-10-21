"""Checks that a permission change made AFTER startup takes effect immediately.

Builds the pipeline (the keyword index is loaded while everything is public), then restricts the
chunks an anonymous user receives, and checks that the SAME running process stops returning them.
Always restores the permissions. Run from the project root:
    python3 scripts/acl_stale_index_test.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import psycopg2  # noqa: E402

from src import pipeline  # noqa: E402

QUESTION = "What attendance is required for the final exam?"
OWNER = "alice@example.com"


def main() -> int:
    pipeline.init()  # the keyword index is built NOW, while everything is public
    api = pipeline.api
    dsn = api.indexer.db_connection_string
    emb = api.embedder.generate_embeddings([QUESTION])

    def search(user):
        return api.hybrid_retriever.search(query=QUESTION, query_vector=emb[0], top_k=5, user_id=user)

    ids = [str(c.id) for c in search(None)]
    print(f"anonymous receives {len(ids)} chunks before any change")

    with psycopg2.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM document_chunks WHERE allowed_users = ARRAY[%s]::text[]", (OWNER,))
        if cur.fetchone()[0] != 0:
            print(f"ABORT: {OWNER} already owns restricted chunks; not touching real permissions.")
            return 2

    ok = False
    try:
        with psycopg2.connect(dsn) as conn, conn.cursor() as cur:
            cur.execute("UPDATE document_chunks SET allowed_users = ARRAY[%s]::text[] WHERE id::text = ANY(%s)",
                        (OWNER, ids))
            print(f"permission change AFTER startup: {cur.rowcount} chunks restricted to {OWNER}")
        counts = {}
        for user in (None, "bob@example.com", OWNER):
            counts[user] = sum(1 for c in search(user) if str(c.id) in ids)
            print(f"  {user}: {counts[user]} of those chunks returned")
        ok = counts[None] == 0 and counts["bob@example.com"] == 0 and counts[OWNER] > 0
        print("PASS: the change took effect without a restart" if ok
              else "FAIL: a stale index still returns chunks the user may no longer read")
    finally:
        with psycopg2.connect(dsn) as conn, conn.cursor() as cur:
            cur.execute("UPDATE document_chunks SET allowed_users = '{}' "
                        "WHERE id::text = ANY(%s) AND allowed_users = ARRAY[%s]::text[]", (ids, OWNER))
            print(f"permissions restored ({cur.rowcount} chunks)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

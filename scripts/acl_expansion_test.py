"""Page expansion must not bypass permissions.

Restricts the chunk(s) containing a marker word to one owner, then checks (through the real search tool function)
that nobody else receives them via page expansion and that the owner does. ALWAYS restores the permissions.
Run from the project root:  python3 scripts/acl_expansion_test.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import psycopg2  # noqa: E402

from src import pipeline  # noqa: E402
from src.mcp_tools import server  # noqa: E402

QUESTION = "What professional qualities and capabilities are stated as general learning objectives?"
MARKER = "stamina"
OWNER = "alice@example.com"


def found(user):
    server.USER_ID = user
    out = server.search_documents(QUESTION, 6)
    return any(MARKER in c["text"].lower() for c in out)


def main() -> int:
    server.EXPAND_PAGES = 3
    pipeline.init()
    dsn = pipeline.api.indexer.db_connection_string
    with psycopg2.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM document_chunks WHERE allowed_users = ARRAY[%s]::text[]", (OWNER,))
        if cur.fetchone()[0] != 0:
            print(f"ABORT: {OWNER} already owns restricted chunks; not touching real permissions.")
            return 2
        cur.execute("SELECT id::text FROM document_chunks WHERE text ILIKE %s AND cardinality(allowed_users) = 0",
                    (f"%{MARKER}%",))
        ids = [r[0] for r in cur.fetchall()]
    if not ids:
        print(f"ABORT: no public chunk contains '{MARKER}'")
        return 2
    if not found(None):
        print("INCONCLUSIVE: the marker is not returned even before restricting anything")
        return 2
    print(f"{len(ids)} chunk(s) contain '{MARKER}'; before any change an anonymous search returns it")
    ok = False
    try:
        with psycopg2.connect(dsn) as conn, conn.cursor() as cur:
            cur.execute("UPDATE document_chunks SET allowed_users = ARRAY[%s]::text[] WHERE id::text = ANY(%s)",
                        (OWNER, ids))
        anon, bob, owner = found(None), found("bob@example.com"), found(OWNER)
        print(f"after restricting: anonymous sees it: {anon}, bob: {bob}, owner: {owner}")
        ok = (not anon) and (not bob) and owner
        print("PASS: page expansion respects permissions" if ok
              else "FAIL: expansion leaked a restricted chunk, or hid it from its owner")
    finally:
        with psycopg2.connect(dsn) as conn, conn.cursor() as cur:
            cur.execute("UPDATE document_chunks SET allowed_users = '{}' "
                        "WHERE id::text = ANY(%s) AND allowed_users = ARRAY[%s]::text[]", (ids, OWNER))
            print(f"permissions restored ({cur.rowcount} chunks)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

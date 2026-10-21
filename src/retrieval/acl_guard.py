import logging

import psycopg2

logger = logging.getLogger(__name__)


def verify_access(db_connection_string, chunks, user_id):
    """Final check: keep only chunks this user may read according to the database right now.

    Protects against stale in-memory indexes. Fails closed: with no user, only public chunks pass.
    """
    if not chunks:
        return chunks
    ids = [str(c.id) for c in chunks]
    sql = (
        "SELECT id::text FROM document_chunks WHERE id::text = ANY(%s) "
        "AND (allowed_users IS NULL OR cardinality(allowed_users) = 0 OR %s = ANY(allowed_users))"
    )
    with psycopg2.connect(db_connection_string) as conn, conn.cursor() as cur:
        cur.execute(sql, (ids, user_id))
        allowed = {row[0] for row in cur.fetchall()}
    kept = [c for c in chunks if str(c.id) in allowed]
    if len(kept) < len(chunks):
        logger.warning(f"ACL: removed {len(chunks) - len(kept)} chunk(s) this user may not read")
    return kept

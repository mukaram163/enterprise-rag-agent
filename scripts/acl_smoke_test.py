"""End-to-end permission check against the real database.

Restricts the chunks an anonymous user receives for a test question to one owner, checks that nobody
else receives them, then ALWAYS puts the permissions back. Run from the project root:
    python3 scripts/acl_smoke_test.py
"""
import subprocess
import sys

QUESTION = "What attendance is required for the final exam?"
OWNER = "alice@example.com"
CONTAINER = "postgres_pgvector"

FIND = '''
from src import pipeline
pipeline.init()
api = pipeline.api
emb = api.embedder.generate_embeddings([{q!r}])
chunks = api.hybrid_retriever.search(query={q!r}, query_vector=emb[0], top_k=5, user_id=None)
print("RESULT=" + ",".join(str(c.id) for c in chunks))
'''

CHECK = '''
import sys
from src import pipeline
restricted = set(sys.argv[1].split(","))
pipeline.init()
api = pipeline.api
emb = api.embedder.generate_embeddings([{q!r}])
for user in [None, "bob@example.com", {owner!r}]:
    chunks = api.hybrid_retriever.search(query={q!r}, query_vector=emb[0], top_k=5, user_id=user)
    n = sum(1 for c in chunks if str(c.id) in restricted)
    print("RESULT=" + str(user) + "=" + str(n))
'''


def run_python(code, *args):
    r = subprocess.run([sys.executable, "-c", code, *args], capture_output=True, text=True)
    lines = [l[len("RESULT="):] for l in r.stdout.splitlines() if l.startswith("RESULT=")]
    if r.returncode != 0 or not lines:
        raise RuntimeError("subprocess failed:\n" + r.stderr[-800:])
    return lines


def psql(sql):
    r = subprocess.run(
        ["docker", "exec", "-i", CONTAINER, "sh", "-c", 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -At'],
        input=sql, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("psql failed: " + r.stderr)
    return r.stdout.strip()


def main() -> int:
    if psql(f"SELECT count(*) FROM document_chunks WHERE allowed_users = ARRAY['{OWNER}'];") != "0":
        print(f"ABORT: {OWNER} already owns restricted chunks; not touching real permissions.")
        return 2
    ids = run_python(FIND.format(q=QUESTION))[0].split(",")
    print(f"anonymous receives {len(ids)} chunks for the test question")
    quoted = ",".join("'" + i.replace("'", "''") + "'" for i in ids)
    ok = False
    try:
        print(psql(f"UPDATE document_chunks SET allowed_users = ARRAY['{OWNER}'] WHERE id::text IN ({quoted});"),
              "(restricted to the owner)")
        counts = dict(r.rsplit("=", 1) for r in run_python(CHECK.format(q=QUESTION, owner=OWNER), ",".join(ids)))
        for user, n in counts.items():
            print(f"  {user}: {n} restricted chunks returned")
        ok = counts["None"] == "0" and counts["bob@example.com"] == "0" and int(counts[OWNER]) > 0
        print("PASS: only the owner receives the restricted chunks" if ok
              else "FAIL: restricted chunks leaked, or the owner lost access")
    finally:
        print(psql(f"UPDATE document_chunks SET allowed_users = '{{}}' WHERE allowed_users = ARRAY['{OWNER}'];"),
              "(permissions restored)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

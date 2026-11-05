import requests

vec = [0.01] * 384

# Test /query
res = requests.post(
    "http://localhost:8000/query",
    json={
        "query": "machine learning",
        "query_vector": vec,
        "user_id": "user_a",
    },
)
print("Query Response:", res.json())

# Test /ingest
res = requests.post(
    "http://localhost:8000/ingest",
    json={
        "doc_id": "doc_101",
        "reindex": True,
        "chunks": [{
            "id": "c101",
            "text": "Updated content",
            "embedding": [0.02] * 384,
            "allowed_users": ["user_a"],
        }],
    },
)
print("Ingest Response:", res.json())

"""How much of the evidence can the retriever reach at different numbers of chunks (top_k)?

No language model is called, so this spends no tokens. Run from the project root:
    python3 scripts/retrieval_ceiling.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import pipeline  # noqa: E402

DATASET = "data/golden_multi.json"
KS = (6, 10, 15, 20)
# words that must appear in the retrieved text for an answer to be able to state these facts
PROBES = {
    3: ["Histopathology", "Cytopathology", "Parasitology", "Immunology", "Serology"],
    5: ["meticulous", "stamina", "leadership", "judgment", "legal", "continuing education",
        "quality control", "instrumentation"],
}


def main() -> None:
    pipeline.init()
    api = pipeline.api
    with open(DATASET, encoding="utf-8") as f:
        data = json.load(f)
    print(f"{'Q':<4}{'what':<22}" + "".join(f"top_k={k:<6}" for k in KS))
    totals = {k: [0, 0] for k in KS}
    for i, item in enumerate(data, 1):
        expected = item["expected_page"] if isinstance(item["expected_page"], list) else [item["expected_page"]]
        emb = api.embedder.generate_embeddings([item["question"]])
        pages_row, probe_row = [], []
        for k in KS:
            chunks = api.hybrid_retriever.search(query=item["question"], query_vector=emb[0], top_k=k, user_id=None)
            pages = {c.page_number for c in chunks}
            got = sum(1 for p in expected if p in pages)
            totals[k][0] += got
            totals[k][1] += len(expected)
            pages_row.append(f"{got}/{len(expected)}")
            if i in PROBES:
                text = " ".join(c.text for c in chunks).lower()
                found = sum(1 for w in PROBES[i] if w.lower() in text)
                probe_row.append(f"{found}/{len(PROBES[i])}")
        print(f"{i:<4}{'expected pages hit':<22}" + "".join(f"{c:<12}" for c in pages_row))
        if probe_row:
            print(f"{'':<4}{'key terms found':<22}" + "".join(f"{c:<12}" for c in probe_row))
    print(f"{'all':<4}{'expected pages hit':<22}" + "".join(f"{totals[k][0]}/{totals[k][1]:<10}" for k in KS))


if __name__ == "__main__":
    main()

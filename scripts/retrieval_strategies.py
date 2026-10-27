"""Free retrieval experiment: which way of fetching text finds the most of the facts in the expected answers?

No language model is called, so this spends no tokens. Run from the project root:
    python3 scripts/retrieval_strategies.py
"Recall" = share of the distinctive words of the expected answer that appear in the retrieved text.
"chars" = how much text the model would have to read (about 4 characters per token).
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import pipeline  # noqa: E402

DATASET = sys.argv[1] if len(sys.argv) > 1 else "data/golden_multi.json"
WORD = re.compile(r"[a-z][a-z\-]{4,}")
POS = re.compile(r"_p(\d+)_c(\d+)$")


def words(text):
    return set(WORD.findall(text.lower()))


def where(c):
    m = POS.search(str(c.id))
    return (c.file_name, c.page_number, int(m.group(2)) if m else 0)


def main():
    pipeline.init()
    api = pipeline.api
    corpus = api.bm25.chunks
    index = {where(c): c for c in corpus}
    by_page = {}
    for c in corpus:
        by_page.setdefault((c.file_name, c.page_number), []).append(c)
    for chunks in by_page.values():
        chunks.sort(key=lambda c: where(c)[2])
    df = {}
    for c in corpus:
        for w in words(c.text):
            df[w] = df.get(w, 0) + 1

    def search(question, vector, k):
        return api.hybrid_retriever.search(query=question, query_vector=vector, top_k=k, user_id=None)

    def with_neighbors(chunks):
        out, seen = [], set()
        for c in chunks:
            f, p, i = where(c)
            for j in (i - 1, i, i + 1):
                n = index.get((f, p, j))
                if n is not None and n.id not in seen:
                    seen.add(n.id)
                    out.append(n)
        return out

    def whole_pages(chunks, n_pages):
        pages = []
        for c in chunks:
            key = (c.file_name, c.page_number)
            if key not in pages:
                pages.append(key)
        return [c for key in pages[:n_pages] for c in by_page[key]]

    def hits_plus_pages(chunks, n_pages):
        seen = {c.id for c in chunks}
        return list(chunks) + [c for c in whole_pages(chunks, n_pages) if c.id not in seen]

    strategies = {
        "top 6 (current)": lambda q, v: search(q, v, 6),
        "top 10": lambda q, v: search(q, v, 10),
        "top 15": lambda q, v: search(q, v, 15),
        "top 6 + neighbors": lambda q, v: with_neighbors(search(q, v, 6)),
        "top 6 -> 3 whole pages": lambda q, v: whole_pages(search(q, v, 6), 3),
        "top 6 -> 4 whole pages": lambda q, v: whole_pages(search(q, v, 6), 4),
        "top 6 + 3 whole pages": lambda q, v: hits_plus_pages(search(q, v, 6), 3),
    }
    with open(DATASET, encoding="utf-8") as f:
        data = json.load(f)
    vectors = [api.embedder.generate_embeddings([item["question"]])[0] for item in data]
    probes = [{w for w in words(item.get("expected_answer") or "") if 1 <= df.get(w, 0) <= 20} for item in data]
    keep = [(item, vector, terms) for item, vector, terms in zip(data, vectors, probes) if terms]
    print(f"{len(keep)} of {len(data)} questions have distinctive words to look for")

    print("\n" + f"{'strategy':<26}" + "".join(f"{'Q' + str(i):>6}" for i in range(1, len(keep) + 1) if len(keep) <= 8)
          + f"{'avg':>7}{'chars':>8}")
    for name, fetch in strategies.items():
        recalls, sizes = [], []
        for item, vector, terms in keep:
            chunks = fetch(item["question"], vector)
            text = " ".join(c.text for c in chunks).lower()
            recalls.append(sum(1 for t in terms if t in text) / len(terms) if terms else 0.0)
            sizes.append(len(text))
        print(f"{name:<26}" + "".join(f"{r:>6.2f}" for r in recalls if len(keep) <= 8)
              + f"{sum(recalls) / len(recalls):>7.2f}{sum(sizes) // len(sizes):>8}")


if __name__ == "__main__":
    main()

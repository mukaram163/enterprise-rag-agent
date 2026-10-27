"""Free test of page expansion in the MCP search tool (no model calls, no database changes).
    python3 scripts/expand_pages_test.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.mcp_tools import server  # noqa: E402

Q = "What learning objectives and practical competencies does the curriculum list?"


def run():
    res = server.search_documents(Q, top_k=6)
    return res, sum(len(r["text"]) for r in res)


def main():
    server.EXPAND_PAGES = 0
    base, base_chars = run()
    server.EXPAND_PAGES = 3
    exp, exp_chars = run()
    print(f"off: {len(base)} chunks, {base_chars} chars | on: {len(exp)} chunks, {exp_chars} chars")
    assert len(exp) > len(base), "expansion added nothing"
    assert exp_chars <= max(base_chars, server.EXPAND_MAX_CHARS), "character cap not respected"

    base_pages = {r["page_number"] for r in base}
    new_pages = {r["page_number"] for r in exp} - base_pages
    added = [r for r in exp if r["text"] not in {b["text"] for b in base}]
    assert added, "no added chunks to test with"
    blocked = added[0]["page_number"]

    api = server.pipeline.api
    real = api.hybrid_retriever.access_checker
    assert real is not None, "no access_checker is wired in pipeline.init(): expansion stays off"

    # 1) a checker that removes one page must also remove it from the expanded results
    api.hybrid_retriever.access_checker = lambda chunks, user: [c for c in chunks if c.page_number != blocked]
    res, _ = run()
    assert all(r["page_number"] != blocked for r in res if r["page_number"] not in base_pages), "blocked page leaked"
    print(f"PASS: page {blocked} removed by the permission check did not come back through expansion")

    # 2) no checker -> no expansion
    api.hybrid_retriever.access_checker = None
    res, chars = run()
    assert chars == base_chars or chars <= base_chars + 1, "expanded without a permission checker"
    print("PASS: without an access_checker nothing is expanded")

    api.hybrid_retriever.access_checker = real
    print("ALL PASS")


if __name__ == "__main__":
    main()

import os
import re
import sys
import json
import time
import types
import asyncio
import warnings
from datetime import datetime
from dotenv import load_dotenv

load_dotenv(override=True)
os.environ["EVAL_MODE"] = "1"
warnings.filterwarnings("ignore")

if "langchain_community.chat_models.vertexai" not in sys.modules:
    dummy = types.ModuleType("langchain_community.chat_models.vertexai")
    dummy.ChatVertexAI = type("ChatVertexAI", (object,), {})
    sys.modules["langchain_community.chat_models.vertexai"] = dummy

from ragas import SingleTurnSample
from ragas.metrics import Faithfulness
from ragas.llms import LangchainLLMWrapper
from langchain_groq import ChatGroq
from src.pipeline import run_rag

DATASET = "data/golden_dataset.json"
RESULTS_DIR = "data/eval_runs"
SLEEP = 7  # seconds between questions (rate limits)
NOT_STATED = re.compile(r"no (information|mention|reference|details?)|not (stated|mentioned|specified|provided|available|found)|does not (state|mention|specify|contain)|cannot (find|answer)", re.I)


def strip_markers(text):
    text = re.sub(r"\[\s*p\.[^\]]*\]", "", text)
    return re.sub(r"\s{2,}", " ", text).strip()


def load_latest_run():
    if not os.path.exists(RESULTS_DIR):
        return {}
    files = sorted([f for f in os.listdir(RESULTS_DIR) if f.startswith("run_") and f.endswith(".json")])
    if not files:
        return {}
    latest_file = os.path.join(RESULTS_DIR, files[-1])
    try:
        with open(latest_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {item["question"]: item for item in data if isinstance(item, dict) and "question" in item}
    except Exception:
        return {}


async def score_item(item, metric):
    q, exp_page = item["question"], item["expected_page"]
    exp_pages = exp_page if isinstance(exp_page, list) else ([exp_page] if exp_page is not None else None)
    out = await run_rag(q)
    pages = [c["page_number"] for c in out["reranked_chunks"]]
    cited = [c["page_number"] for c in out["citations"]]
    answer = out["answer"]

    if exp_page is None:
        # Unanswerable: bot must say it isn't stated
        says_not_stated = bool(NOT_STATED.search(answer))
        retrieval = None
        citation = None
        passed = says_not_stated
        faith = None
    else:
        retrieval = any(p in pages for p in exp_pages)
        citation = any(p in cited for p in exp_pages)
        sample = SingleTurnSample(
            user_input=q,
            response=strip_markers(answer),
            retrieved_contexts=out["contexts"],
        )
        faith = await metric.single_turn_ascore(sample)
        passed = retrieval and citation and faith >= 0.8

    return {
        "question": q, "expected_page": exp_page, "answer": answer,
        "chunk_pages": pages, "cited_pages": cited, "provider": out["provider"],
        "retrieval_hit": retrieval, "citation_correct": citation,
        "faithfulness": faith, "passed": passed, "error": None,
    }


async def main():
    resume_mode = "--resume" in sys.argv
    prev_results = load_latest_run() if resume_mode else {}

    with open(DATASET, "r", encoding="utf-8") as f:
        data = json.load(f)

    judge = ChatGroq(model="openai/gpt-oss-120b", groq_api_key=os.environ["GROQ_API_KEY"], temperature=0.0)
    metric = Faithfulness(llm=LangchainLLMWrapper(judge))

    results = []
    reused = 0
    api_errors = 0

    for i, item in enumerate(data, 1):
        q = item["question"]
        exp_p = item["expected_page"]

        if resume_mode and q in prev_results:
            prev = prev_results[q]
            if prev.get("error") is None and prev.get("expected_page") == exp_p:
                print(f"[{i}/{len(data)}] [REUSED] {q[:60]}...")
                results.append(prev)
                reused += 1
                continue

        print(f"[{i}/{len(data)}] {q[:60]}...")
        try:
            res = await score_item(item, metric)
        except Exception as e:
            res = {
                "question": q,
                "expected_page": exp_p,
                "answer": None,
                "chunk_pages": [],
                "cited_pages": [],
                "provider": None,
                "retrieval_hit": None,
                "citation_correct": None,
                "faithfulness": None,
                "passed": False,
                "error": str(e)[:200],
            }

        if res.get("error") is not None:
            api_errors += 1

        results.append(res)
        time.sleep(SLEEP)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"{RESULTS_DIR}/run_{stamp}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    passed_count = sum(1 for r in results if r.get("passed"))
    print(f"\nSaved: {path}")
    print(f"Summary: Total: {len(results)} | Passed: {passed_count} | Reused: {reused} | Items with API errors: {api_errors}")


if __name__ == "__main__":
    asyncio.run(main())
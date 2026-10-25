"""Compare the single agent and the multi-agent system on the golden dataset, scored like run_eval.py.

    python3 -m src.evaluation.run_agent_eval --system single --limit 3
    python3 -m src.evaluation.run_agent_eval --system multi --limit 3
    python3 -m src.evaluation.run_agent_eval --compare
Results go to data/eval_agents/, separate from the stage 3 runs.
"""
import argparse
import asyncio
import glob
import json
import os
import re
from datetime import datetime

# run_eval must be imported first: it loads .env and prepares the ragas imports
from src.evaluation.run_eval import (
    ChatGroq, Faithfulness, LangchainLLMWrapper, NOT_STATED, SingleTurnSample, strip_markers,
)
from src.evaluation.report import summarize
from src.agents.pipeline_agents import _retry_wait, research_parallel
from src.agents.single_agent import CITE_FIND, ask
from src.agents.synthesizer import synthesize
from src.agents.usage import reset_run, run_totals
from src.utils.pricing import estimate_cost

import time

DEFAULT_DATASET = "data/golden_dataset.json"
OUT_DIR = "data/eval_agents"
STATE = {"phase": ""}  # which step is running, so an error says where it happened
BASELINE = "data/eval_baseline.json"
PAGE = re.compile(r"p\.\s*(\d+)", re.I)


def pages_of(text):
    return sorted({int(PAGE.search(c).group(1)) for c in CITE_FIND.findall(text or "")})


async def run_system(system, q):
    reset_run()
    trace = []
    if system == "single":
        for attempt in (1, 2, 3):
            try:
                answer = await ask(q, trace)
                break
            except Exception as e:
                if attempt == 3:
                    raise
                print(f"[retry] {str(e)[:100]}")
                await asyncio.sleep(_retry_wait(e))
                trace.clear()
                reset_run()
    else:
        _, results = await research_parallel(q)
        answer = await synthesize(q, results)
        trace = [c for r in results for c in r.get("tool_calls", [])]
    return answer, trace


async def judge_with_retry(metric, sample, attempts=3):
    """Returns (score, error_text). Waits and retries when the judge is rate limited."""
    err = None
    for attempt in range(1, attempts + 1):
        try:
            return await metric.single_turn_ascore(sample), None
        except Exception as e:
            err = str(e)[:300]
            limited = any(w in err.lower() for w in ("429", "rate limit", "quota", "resource_exhausted"))
            if not limited:
                break
            if attempt < attempts:
                await asyncio.sleep(max(_retry_wait(e), 15))
    return None, err


def unique_chunks(trace):
    """Distinct retrieved chunks across all searches, so repeated results are not counted twice."""
    seen, out = set(), []
    for call in trace:
        text = call["result_text"]
        try:
            data = json.loads(text)
            items = data.get("result", [data]) if isinstance(data, dict) else data
        except ValueError:
            items = []
            for line in text.splitlines():
                try:
                    items.append(json.loads(line))
                except ValueError:
                    pass
        texts = [i["text"] for i in items if isinstance(i, dict) and i.get("text")] or [text]
        for t in texts:
            if t not in seen:
                seen.add(t)
                out.append(t)
    return out


async def score(system, item, metric):
    q, exp = item["question"], item["expected_page"]
    exp_pages = exp if isinstance(exp, list) else ([exp] if exp is not None else [])
    t0 = time.perf_counter()
    STATE["phase"] = "agents"
    answer, trace = await run_system(system, q)
    seconds = round(time.perf_counter() - t0, 1)
    totals = run_totals()
    model = os.getenv("AGENT_MODEL", "openai/gpt-oss-120b")
    retrieved = sorted({p for c in trace for p in pages_of(c["result_text"])})
    cited = pages_of(answer)
    row = {
        "question": q, "expected_page": exp, "answer": answer, "retrieved_pages": retrieved,
        "cited_pages": cited, "model_calls": totals["calls"],
        "tokens": totals["tokens_in"] + totals["tokens_out"], "seconds": seconds,
        "cost": estimate_cost(model, totals["tokens_in"], totals["tokens_out"]),
        "retrieval_hit": None, "citation_correct": None, "citation_all": None,
        "faithfulness": None, "passed": False, "error": None,
    }
    if exp is None:  # unanswerable question: the system must say it is not stated
        row["passed"] = bool(NOT_STATED.search(answer))
        return row
    row["retrieval_hit"] = any(p in retrieved for p in exp_pages)
    row["citation_correct"] = any(p in cited for p in exp_pages)
    row["citation_all"] = all(p in cited for p in exp_pages)
    contexts = unique_chunks(trace)
    row["contexts"] = contexts  # saved so answers can be re-judged later without re-running the agents
    sample = SingleTurnSample(
        user_input=q,
        response=strip_markers(CITE_FIND.sub("", answer)),
        retrieved_contexts=contexts or ["(no documents were retrieved)"],
    )
    STATE["phase"] = "judge"
    faith, err = await judge_with_retry(metric, sample)
    if err:  # keep everything the agents produced, so only the judging step has to be repeated later
        row["error"] = f"[judge] {err}"
        return row
    row["faithfulness"] = None if faith != faith else faith  # NaN means the judge gave no score
    row["passed"] = bool(row["retrieval_hit"] and row["citation_correct"]
                         and row["faithfulness"] is not None and row["faithfulness"] >= 0.8)
    return row


async def score_with_retry(system, item, metric, attempts=3):
    for attempt in range(1, attempts + 1):
        try:
            return await score(system, item, metric)
        except Exception as e:
            limited = "429" in str(e) or "rate limit" in str(e).lower()
            if attempt == attempts or not limited:
                raise
            wait = _retry_wait(e)
            print(f"[rate limit] waiting {wait:.0f}s, then retrying this question")
            await asyncio.sleep(wait)


def avg(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def stats(rows):
    s = summarize(rows)
    s["tokens"] = avg([r.get("tokens") for r in rows])
    s["seconds"] = avg([r.get("seconds") for r in rows])
    s["cost"] = avg([r.get("cost") for r in rows])
    multi_page = [r for r in rows if isinstance(r.get("expected_page"), list) and r.get("citation_all") is not None]
    s["all_pages"] = avg([1 if r["citation_all"] else 0 for r in multi_page])
    return s


COLS = [("items", "n"), ("errors", "err"), ("pass_rate", "pass"), ("retrieval_rate", "retrieval"), ("citation_rate", "citation"),
        ("faithfulness_avg", "faithful"), ("all_pages", "all-pages"), ("tokens", "tokens/q"),
        ("seconds", "sec/q"), ("cost", "$/q")]


def show(named_rows):
    print(f"\n{'system':<26}" + "".join(f"{label:>11}" for _, label in COLS))
    for name, rows in named_rows.items():
        if not rows:
            continue
        s = stats(rows)
        cells = []
        for key, _ in COLS:
            v = s.get(key)
            if v is None:
                cells.append("n/a")
            elif key in ("items", "errors"):
                cells.append(str(v))
            elif key == "tokens":
                cells.append(f"{v:,.0f}")
            elif key == "cost":
                cells.append(f"${v:.4f}")
            elif key == "seconds":
                cells.append(f"{v:.1f}s")
            else:
                cells.append(f"{v:.2f}")
        print(f"{name:<26}" + "".join(f"{c:>11}" for c in cells))


def make_judge(kind):
    if kind == "groq":
        return ChatGroq(model="openai/gpt-oss-120b", groq_api_key=os.environ["GROQ_API_KEY"], temperature=0.0)
    # a different model family than the one that wrote the answers, so the judge is not grading its own work
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(model=os.getenv("JUDGE_MODEL", "gemini-3.8-flash"),
                                  google_api_key=os.environ["GEMINI_API_KEY"], temperature=0.0,
                                  max_output_tokens=8192)


async def run(system, dataset, limit, sleep, retry_errors=False, judge_kind="gemini"):
    with open(dataset, encoding="utf-8") as f:
        data = json.load(f)
    if limit:
        data = data[:limit]
    judge = make_judge(judge_kind)
    metric = Faithfulness(llm=LangchainLLMWrapper(judge))
    prev = {r["question"]: r for r in load_latest(system)} if retry_errors else {}
    rows = []
    for i, item in enumerate(data, 1):
        old = prev.get(item["question"])
        if old is not None and not old.get("error") and old.get("judge") == judge_kind:
            print(f"[{i}/{len(data)}] (reused) {item['question'][:60]}...")
            rows.append(old)
            continue
        print(f"[{i}/{len(data)}] ({system}) {item['question'][:60]}...")
        try:
            rows.append(await score_with_retry(system, item, metric))
        except Exception as e:
            rows.append({"question": item["question"], "expected_page": item["expected_page"], "answer": None,
                         "passed": False, "error": f"[{STATE['phase']}] {str(e)[:300]}", "retrieval_hit": None,
                         "citation_correct": None, "citation_all": None, "faithfulness": None,
                         "tokens": None, "seconds": None, "cost": None})
        await asyncio.sleep(sleep)
    for r in rows:
        r.setdefault("judge", judge_kind)
    os.makedirs(OUT_DIR, exist_ok=True)
    path = f"{OUT_DIR}/{system}_{datetime.now():%Y%m%d_%H%M%S}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {path}")
    show({system: rows})


async def rejudge(system, judge_kind, sleep):
    """Re-run ONLY the judging step on saved agent answers (no agent tokens are spent)."""
    rows = load_latest(system)
    metric = Faithfulness(llm=LangchainLLMWrapper(make_judge(judge_kind)))
    for r in rows:
        if not (r.get("error") or "").startswith("[judge]") or not r.get("contexts"):
            continue
        print(f"(rejudge) {r['question'][:60]}...")
        sample = SingleTurnSample(
            user_input=r["question"],
            response=strip_markers(CITE_FIND.sub("", r["answer"])),
            retrieved_contexts=r["contexts"],
        )
        faith, err = await judge_with_retry(metric, sample)
        if err:
            r["error"] = f"[judge] {err}"
        else:
            r["error"] = None
            r["faithfulness"] = None if faith != faith else faith
            r["passed"] = bool(r["retrieval_hit"] and r["citation_correct"]
                               and r["faithfulness"] is not None and r["faithfulness"] >= 0.8)
        r["judge"] = judge_kind
        await asyncio.sleep(sleep)
    os.makedirs(OUT_DIR, exist_ok=True)
    path = f"{OUT_DIR}/{system}_{datetime.now():%Y%m%d_%H%M%S}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {path}")
    show({system: rows})


def load_latest(system):
    files = sorted(glob.glob(f"{OUT_DIR}/{system}_*.json"))
    if not files:
        return []
    with open(files[-1], encoding="utf-8") as f:
        return json.load(f)


def label(r):
    return "ERROR" if r.get("error") else ("PASS" if r["passed"] else "FAIL")


def compare():
    single, multi = load_latest("single"), load_latest("multi")
    common = {r["question"] for r in single} & {r["question"] for r in multi}
    if not common:
        print("Run both systems first, with the same --limit.")
        return
    named = {"single agent": [r for r in single if r["question"] in common],
             "multi-agent": [r for r in multi if r["question"] in common]}
    try:
        with open(BASELINE, encoding="utf-8") as f:
            base = json.load(f)["questions"]
        named["stage-1 RAG (saved baseline)"] = [{**base[q], "question": q} for q in common if q in base]
    except Exception:
        pass
    judges = {r.get("judge") for r in single + multi}
    if judges - {None, "groq"}:  # a different judge than the saved baseline: not comparable
        named.pop("stage-1 RAG (saved baseline)", None)
    print(f"Same {len(common)} questions for every system. Judge: {', '.join(sorted(str(j) for j in judges))}")
    show(named)
    by_multi = {r["question"]: r for r in multi}
    print("\nper question:")
    for r in single:
        if r["question"] in common:
            m = by_multi[r["question"]]
            print(f"  single:{label(r):<5}  multi:{label(m):<5}  {r['question'][:70]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", choices=["single", "multi"])
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sleep", type=float, default=15)
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--retry-errors", action="store_true")
    ap.add_argument("--judge", choices=["gemini", "groq"], default="gemini")
    ap.add_argument("--rejudge", action="store_true")
    a = ap.parse_args()
    if a.compare:
        compare()
    elif a.system and a.rejudge:
        asyncio.run(rejudge(a.system, a.judge, a.sleep))
    elif a.system:
        asyncio.run(run(a.system, a.dataset, a.limit, a.sleep, a.retry_errors, a.judge))
    else:
        ap.print_help()


if __name__ == "__main__":
    main()

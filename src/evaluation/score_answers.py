"""Score saved answers against the reference answer (expected_answer) with the Gemini judge.

For every question: how many of the reference's key facts does the answer state correctly (coverage),
how many does it state wrongly, and does it contain clear errors of its own (for example totals that do not add up).
Uses the saved answers of the latest single_* and multi_* runs, so it spends no agent tokens:
    python3 -m src.evaluation.score_answers --dataset data/golden_multi.json
"""
import argparse
import asyncio
import glob
import json
from datetime import datetime

from src.evaluation.run_agent_eval import OUT_DIR, load_latest, make_judge

PROMPT = """You are grading a candidate answer against a reference answer.

Question: {question}

Reference answer: {reference}

Candidate answer: {answer}

Step 1: split the reference answer into its separate key facts (every name, number or list item is its own fact).
Step 2: for each fact decide: "yes" = the candidate states it correctly, "no" = the candidate does not mention it, "wrong" = the candidate states it differently or contradicts it.
Step 3: list any "extra_errors": statements in the candidate that clearly contradict the reference or are inconsistent with themselves (for example totals that do not add up).

Reply with JSON only, no other text:
{{"facts": [{{"fact": "...", "status": "yes"}}], "extra_errors": ["..."]}}"""


def as_text(content):
    if isinstance(content, list):
        return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    return str(content)


async def grade(llm, item, answer):
    prompt = PROMPT.format(question=item["question"], reference=item["expected_answer"], answer=answer)
    for attempt in range(1, 5):
        try:
            reply = await llm.ainvoke(prompt)
            text = as_text(reply.content)
            data = json.loads(text[text.find("{"):text.rfind("}") + 1])
            facts = data.get("facts", [])
            yes = sum(1 for f in facts if f.get("status") == "yes")
            wrong = sum(1 for f in facts if f.get("status") == "wrong")
            return {"facts": facts, "extra_errors": data.get("extra_errors", []),
                    "coverage": yes / len(facts) if facts else None, "wrong": wrong}
        except Exception as e:
            msg = str(e)
            print(f"  [attempt {attempt}] {msg[:120]}")
            if attempt == 4:
                return {"error": msg[:200]}
            await asyncio.sleep(30 if ("429" in msg or "RESOURCE_EXHAUSTED" in msg) else 5)


def average(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def fmt(v):
    return "n/a" if v is None else f"{v:.2f}"


def report(out):
    print(f"{'system':<8}{'coverage':>10}{'wrong facts':>13}{'extra errors':>14}{'graded':>9}")
    for system, graded in out.items():
        ok = [g for g in graded if "error" not in g]
        print(f"{system:<8}{fmt(average([g['coverage'] for g in ok])):>10}{sum(g['wrong'] for g in ok):>13}"
              f"{sum(len(g['extra_errors']) for g in ok):>14}{len(ok):>5}/{len(graded)}")
    print("\nper question (single / multi):")
    s = {g["question"]: g for g in out["single"]}
    m = {g["question"]: g for g in out["multi"]}
    for i, q in enumerate(s, 1):
        a, b = s[q], m.get(q, {})
        cov = lambda g: fmt(g.get("coverage")) if g and "error" not in g else "n/a"
        err = lambda g: str(len(g.get("extra_errors", []))) if g and "error" not in g else "n/a"
        print(f"Q{i}: coverage {cov(a)} / {cov(b)}    extra errors {err(a)} / {err(b)}")


def load_previous():
    files = sorted(glob.glob(f"{OUT_DIR}/answers_*.json"))
    if not files:
        return {}
    with open(files[-1], encoding="utf-8") as f:
        return json.load(f)


async def main(dataset, sleep, resume=False):
    with open(dataset, encoding="utf-8") as f:
        items = {d["question"]: d for d in json.load(f)}
    llm = make_judge("gemini")
    prev = load_previous() if resume else {}
    out = {}
    for system in ("single", "multi"):
        done = {g["question"]: g for g in prev.get(system, []) if "error" not in g}
        graded = []
        for r in load_latest(system):
            if r["question"] in done:  # already graded in an earlier run
                graded.append(done[r["question"]])
                continue
            item = items.get(r["question"])
            if item is None or not r.get("answer"):
                continue
            print(f"({system}) {r['question'][:60]}...")
            g = await grade(llm, item, r["answer"])
            g["question"] = r["question"]
            graded.append(g)
            await asyncio.sleep(sleep)
        out[system] = graded
    path = f"{OUT_DIR}/answers_{datetime.now():%Y%m%d_%H%M%S}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {path}\n")
    report(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/golden_multi.json")
    ap.add_argument("--sleep", type=float, default=8)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    asyncio.run(main(a.dataset, a.sleep, a.resume))

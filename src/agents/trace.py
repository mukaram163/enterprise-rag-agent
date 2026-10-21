import json
import os
import sys

from src.agents.usage import run_totals
from src.utils.pricing import estimate_cost

TRACE_FILE = "data/logs/agent_runs.jsonl"


def write_trace(question, status, started_at, wall_seconds, subs=None, results=None, answer="", reason=None, user=None):
    """One line per research run: what was asked, what each agent did, what it cost. Never raises."""
    try:
        t = run_totals()
        model = os.getenv("AGENT_MODEL", "openai/gpt-oss-120b")
        entry = {
            "timestamp": started_at,
            "question": question,
            "user": user or "anonymous",
            "status": status,  # ok | partial | blocked
            "reason": reason,
            "wall_seconds": round(wall_seconds, 1),
            "sub_questions": subs or [],
            "agents": [
                {
                    "sub_question": r["sub_question"],
                    "seconds": r["seconds"],
                    "ok": not r["answer"].startswith("ERROR:"),
                    "error": r["answer"][:150] if r["answer"].startswith("ERROR:") else None,
                    "tool_calls": r.get("tool_calls", []),
                }
                for r in (results or [])
            ],
            "model": model,
            "model_calls": t["calls"],
            "tokens_in": t["tokens_in"],
            "tokens_out": t["tokens_out"],
            "estimated_cost": estimate_cost(model, t["tokens_in"], t["tokens_out"]),
            "answer_chars": len(answer),
        }
        os.makedirs(os.path.dirname(TRACE_FILE), exist_ok=True)
        with open(TRACE_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[trace] write failed (run not affected): {e}")


def show(n: int = 5) -> None:
    try:
        with open(TRACE_FILE, encoding="utf-8") as f:
            lines = f.readlines()[-n:]
    except FileNotFoundError:
        print("No runs logged yet.")
        return
    for line in lines:
        r = json.loads(line)
        ok = sum(a["ok"] for a in r["agents"])
        cost = f"${r['estimated_cost']:.5f}" if r["estimated_cost"] is not None else "unknown"
        print(f"\n{r['timestamp'][:19]}  {r['status'].upper():8} {r['wall_seconds']}s  "
              f"{r['model_calls']} calls, {r['tokens_in'] + r['tokens_out']:,} tokens, {cost}  "
              f"agents ok: {ok}/{len(r['agents'])}")
        print(f"  user: {r.get('user', 'anonymous')}  Q: {r['question'][:90]}")
        if r["reason"]:
            print(f"  blocked: {r['reason']}")
        for a in r["agents"]:
            flag = "ok  " if a["ok"] else "FAIL"
            queries = "; ".join(str(c["query"]) for c in a["tool_calls"])[:90]
            print(f"  [{flag}] {a['seconds']:>5}s  {a['sub_question'][:50]}  <- {queries}")


if __name__ == "__main__":
    show(int(sys.argv[1]) if len(sys.argv) > 1 else 5)

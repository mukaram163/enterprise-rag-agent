import os
import sys
import json
import glob

RUNS_DIR = "data/eval_runs"
BASELINE = "data/eval_baseline.json"
FAITH_DROP = 0.10
RATE_KEYS = ["pass_rate", "retrieval_rate", "citation_rate", "faithfulness_avg"]


def latest_run():
    files = sorted(glob.glob(f"{RUNS_DIR}/run_*.json"))
    if not files:
        sys.exit("No run files in data/eval_runs/. Run run_eval first.")
    return files[-1]


def rate(xs):
    return sum(xs) / len(xs) if xs else None


def fmt(x):
    return "n/a" if x is None else f"{x:.2f}"


def yn(v):
    return "-" if v is None else ("yes" if v else "NO")


def summarize(results):
    ok = [r for r in results if not r.get("error")]
    return {
        "items": len(results),
        "errors": len(results) - len(ok),
        "pass_rate": rate([1 if r["passed"] else 0 for r in ok]),
        "retrieval_rate": rate([1 if r["retrieval_hit"] else 0 for r in ok if r.get("retrieval_hit") is not None]),
        "citation_rate": rate([1 if r["citation_correct"] else 0 for r in ok if r.get("citation_correct") is not None]),
        "faithfulness_avg": rate([r["faithfulness"] for r in ok if r.get("faithfulness") is not None]),
    }


def print_report(results, s):
    print(f"\n{'#':>2} {'RESULT':<6} {'EXPECTED':<9} {'CITED':<12} RETR CITE FAITH  QUESTION")
    for i, r in enumerate(results, 1):
        status = "ERROR" if r.get("error") else ("PASS" if r["passed"] else "FAIL")
        print(f"{i:>2} {status:<6} {str(r.get('expected_page')):<9} {str(r.get('cited_pages')):<12} "
              f"{yn(r.get('retrieval_hit')):<4} {yn(r.get('citation_correct')):<4} "
              f"{fmt(r.get('faithfulness')):<6} {r['question'][:55]}")
    print("\n--- OVERALL ---")
    print(f"Items: {s['items']}  API errors: {s['errors']}")
    print(f"Pass rate:        {fmt(s['pass_rate'])}")
    print(f"Retrieval hit:    {fmt(s['retrieval_rate'])}  (answerable questions)")
    print(f"Citation correct: {fmt(s['citation_rate'])}  (answerable questions)")
    print(f"Faithfulness avg: {fmt(s['faithfulness_avg'])}")


def compare(results, summary, base):
    regressions, improvements = [], []
    for r in results:
        b = base["questions"].get(r["question"])
        if b is None:
            continue
        q = r["question"][:60]
        if b["passed"] and not r["passed"]:
            regressions.append(f"{q}: PASS -> FAIL")
        elif not b["passed"] and r["passed"]:
            improvements.append(f"{q}: FAIL -> PASS")
        bf, nf = b.get("faithfulness"), r.get("faithfulness")
        if bf is not None and nf is not None and nf < bf - FAITH_DROP:
            regressions.append(f"{q}: faithfulness {bf:.2f} -> {nf:.2f}")
    for k in RATE_KEYS:
        b, n = base["summary"].get(k), summary.get(k)
        if b is not None and n is not None and n < b - 0.001:
            regressions.append(f"overall {k}: {b:.2f} -> {n:.2f}")
    return regressions, improvements


def main():
    path = latest_run()
    with open(path, "r", encoding="utf-8") as f:
        results = json.load(f)
    summary = summarize(results)
    print(f"Run file: {path}")
    print_report(results, summary)

    if summary["errors"]:
        print(f"\nWARNING: {summary['errors']} items had API errors. "
              "Rerun with --resume. This run can't be used as a baseline or compared.")
        sys.exit(2)

    if "--save-baseline" in sys.argv:
        data = {
            "run_file": path,
            "summary": summary,
            "questions": {
                r["question"]: {k: r.get(k) for k in ("passed", "retrieval_hit", "citation_correct", "faithfulness")}
                for r in results
            },
        }
        with open(BASELINE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"\nBaseline saved to {BASELINE}")
        return

    if not os.path.exists(BASELINE):
        print("\nNo baseline yet. Save one with: python3 -m src.evaluation.report --save-baseline")
        return

    with open(BASELINE, "r", encoding="utf-8") as f:
        base = json.load(f)
    regressions, improvements = compare(results, summary, base)

    print(f"\n--- VS BASELINE ({base['run_file']}) ---")
    for x in improvements:
        print("IMPROVED :", x)
    for x in regressions:
        print("REGRESSED:", x)
    if not regressions:
        print("No regressions.")
    sys.exit(1 if regressions else 0)


if __name__ == "__main__":
    main()

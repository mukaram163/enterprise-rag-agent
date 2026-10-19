import asyncio
import sys
import time
from datetime import datetime, timezone

from src.agents.pipeline_agents import research_parallel
from src.agents.synthesizer import synthesize
from src.agents.trace import write_trace
from src.agents.usage import reset_run, run_summary
from src.utils.block_log import log_block
from src.utils.cost_guard import check_ceiling
from src.utils.injection_guard import check_injection
from src.utils.pii_filter import redact

BLOCKED_MSG = "Sorry, I can't help with that request."
CEILING_MSG = "The spending limit has been reached, so I can't run new research right now."
DEFAULT_Q = "What are the attempt limits for each professional exam, and what attendance is required for the final?"


async def guarded_research(question: str, session: str = "agent-cli") -> str:
    reset_run()
    started = datetime.now(timezone.utc).isoformat()
    t0 = time.perf_counter()

    # 1. input guardrail: check BEFORE any agent runs, so a blocked request costs nothing
    flagged, rules = check_injection(question)
    if flagged:
        log_block("injection", f"rules={rules} text={question[:100]!r}", session=session, ip="local")
        write_trace(question, "blocked", started, time.perf_counter() - t0, reason=f"injection {rules}")
        return BLOCKED_MSG

    # 2. cost ceiling: the same one the API uses
    _, note, blocked = check_ceiling("flash")
    if blocked:
        log_block("cost_ceiling", note, session=session, ip="local")
        write_trace(question, "blocked", started, time.perf_counter() - t0, reason=note)
        return CEILING_MSG

    # 3. plan -> research in parallel -> synthesize
    subs, results = await research_parallel(question)
    answer = await synthesize(question, results)

    # 4. output guardrail: redact personal information
    safe, counts = redact(answer)
    if counts:
        log_block("pii", f"redacted {counts}", session=session, ip="local")

    all_ok = all(not r["answer"].startswith("ERROR:") for r in results)
    write_trace(question, "ok" if all_ok else "partial", started, time.perf_counter() - t0,
                subs=subs, results=results, answer=safe)
    return safe


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or DEFAULT_Q
    print(asyncio.run(guarded_research(q)))
    print(f"\n[usage] {run_summary()}")

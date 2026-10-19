import asyncio
import os
import re
import sys
import time

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from src.agents.planner import plan
from src.agents.single_agent import ask_with_session
from src.agents.usage import reset_run, run_tokens

AGENT_TIMEOUT = 150  # seconds per research agent (rate-limit waits count)
MAX_SUBQUESTIONS = 5  # the planner can never create more agents than this
MAX_RUN_TOKENS = int(os.getenv("MAX_RUN_TOKENS", "60000"))  # no new agent starts once a run used this many tokens
DEFAULT_Q = "What are the attempt limits for each professional exam, and what attendance is required for the final?"


def _retry_wait(e) -> float:
    # the API says "Please try again in 25.1s": wait that long plus a little (default 10s)
    m = re.search(r"try again in ([\d.]+)s", str(e))
    return float(m.group(1)) + 1 if m else 10


async def _one(session, tools, sq, sem):
    async with sem:
        t0 = time.perf_counter()  # timer starts after waiting for a free slot
        ans = None
        for attempt in (1, 2, 3):
            if run_tokens() >= MAX_RUN_TOKENS:
                ans = "ERROR: token budget reached for this run"
                print(f"[agent skipped] {sq!r}: {ans}")
                break
            try:
                ans = await asyncio.wait_for(ask_with_session(session, tools, sq), AGENT_TIMEOUT)
                break
            except asyncio.TimeoutError:
                ans = f"ERROR: timed out after {AGENT_TIMEOUT}s"
                print(f"[agent failed] {sq!r}: {ans}")
                break  # a slow agent is not retried
            except Exception as e:
                ans = f"ERROR: {e!r}"
                print(f"[agent failed] attempt {attempt} for {sq!r}: {str(e)[:150]}")
                if attempt < 3:
                    await asyncio.sleep(_retry_wait(e))  # wait as long as the API asks, then retry
    return {"sub_question": sq, "answer": ans, "seconds": round(time.perf_counter() - t0, 1)}


async def research_parallel(question: str, concurrency: int = 2, subs=None, verbose: bool = False):
    reset_run()  # token counters start fresh for every research run
    subs = subs or await plan(question)
    if len(subs) > MAX_SUBQUESTIONS:
        print(f"[agents] planner made {len(subs)} sub-questions, keeping the first {MAX_SUBQUESTIONS}")
        subs = subs[:MAX_SUBQUESTIONS]
    sem = asyncio.Semaphore(concurrency)
    params = StdioServerParameters(command=sys.executable, args=["-m", "src.mcp_tools.server"])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = [{"type": "function", "function": {
                "name": t.name, "description": t.description or "", "parameters": t.inputSchema}}
                for t in (await session.list_tools()).tools]
            if verbose:  # warm up the server so model loading isn't timed
                await session.call_tool("search_documents", {"query": "warm up", "top_k": 1})
            t0 = time.perf_counter()
            results = list(await asyncio.gather(*[_one(session, tools, sq, sem) for sq in subs]))
            if verbose:
                print(f"[agents] {len(subs)} sub-questions, concurrency={concurrency}, "
                      f"wall time {time.perf_counter() - t0:.1f}s")
    return subs, results


research_sequential = research_parallel  # old name kept so nothing else breaks


if __name__ == "__main__":
    from src.agents.synthesizer import synthesize, CITE
    from src.agents.single_agent import ask

    bench = "--bench" in sys.argv
    q = " ".join(a for a in sys.argv[1:] if a != "--bench") or DEFAULT_Q

    async def run():
        single = await ask(q)
        subs = await plan(q)
        if bench:  # compare one-at-a-time vs two-at-a-time (doubles the API calls)
            await research_parallel(q, concurrency=1, subs=subs, verbose=True)
        _, res = await research_parallel(q, concurrency=2, subs=subs, verbose=True)
        multi = await synthesize(q, res)
        return single, subs, res, multi

    single, subs, res, multi = asyncio.run(run())
    print("\n=== SINGLE AGENT ===\n" + single)
    print("\nSUB-QUESTIONS:", subs)
    for r in res:
        flag = "  [FAILED]" if r["answer"].startswith("ERROR:") else ""
        print(f"  agent took {r['seconds']}s{flag}: {r['sub_question']}")
    print("\n=== MULTI AGENT ===\n" + multi)
    print("\nSINGLE CITED:", sorted(set(CITE.findall(single))))
    print("MULTI CITED: ", sorted(set(CITE.findall(multi))))

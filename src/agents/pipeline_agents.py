import asyncio
import sys
import time

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from src.agents.planner import plan
from src.agents.single_agent import ask_with_session

AGENT_TIMEOUT = 60  # seconds per research agent
DEFAULT_Q = "What are the attempt limits for each professional exam, and what attendance is required for the final?"


async def _one(session, tools, sq, sem):
    async with sem:
        t0 = time.perf_counter()  # timer starts after waiting for a free slot
        try:
            ans = await asyncio.wait_for(ask_with_session(session, tools, sq), AGENT_TIMEOUT)
        except asyncio.TimeoutError:
            ans = f"ERROR: timed out after {AGENT_TIMEOUT}s"
        except Exception as e:
            ans = f"ERROR: {e}"
    return {"sub_question": sq, "answer": ans, "seconds": round(time.perf_counter() - t0, 1)}


async def research_parallel(question: str, concurrency: int = 2, subs=None, verbose: bool = False):
    subs = subs or await plan(question)
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

    q = " ".join(sys.argv[1:]) or DEFAULT_Q

    async def run():
        single = await ask(q)
        subs = await plan(q)
        await research_parallel(q, concurrency=1, subs=subs, verbose=True)  # one at a time
        _, res = await research_parallel(q, concurrency=2, subs=subs, verbose=True)  # two at a time
        multi = await synthesize(q, res)
        return single, subs, res, multi

    single, subs, res, multi = asyncio.run(run())
    print("\n=== SINGLE AGENT ===\n" + single)
    print("\nSUB-QUESTIONS:", subs)
    for r in res:
        print(f"  agent took {r['seconds']}s: {r['sub_question']}")
    print("\n=== MULTI AGENT ===\n" + multi)
    print("\nSINGLE CITED:", sorted(set(CITE.findall(single))))
    print("MULTI CITED: ", sorted(set(CITE.findall(multi))))

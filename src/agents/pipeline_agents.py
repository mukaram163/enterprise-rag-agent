import asyncio
import sys
import time

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from src.agents.planner import plan
from src.agents.single_agent import ask_with_session


async def research_sequential(question: str):
    subs = await plan(question)
    params = StdioServerParameters(command=sys.executable, args=["-m", "src.mcp_tools.server"])
    results = []
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = [{"type": "function", "function": {
                "name": t.name, "description": t.description or "", "parameters": t.inputSchema}}
                for t in (await session.list_tools()).tools]
            for sq in subs:
                t0 = time.perf_counter()
                try:
                    ans = await ask_with_session(session, tools, sq)
                except Exception as e:
                    ans = f"ERROR: {e}"
                results.append({"sub_question": sq, "answer": ans,
                                "seconds": round(time.perf_counter() - t0, 1)})
                await asyncio.sleep(2)  # stay under Groq per-minute limits
    return subs, results


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "Compare the passing rules and total marks of the Second and Final Professional exams."
    subs, res = asyncio.run(research_sequential(q))
    for r in res:
        print(f"\n## {r['sub_question']}  ({r['seconds']}s)\n{r['answer']}")
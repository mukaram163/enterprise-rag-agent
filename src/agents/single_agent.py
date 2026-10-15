import asyncio
import re
import sys

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from src.agents.llm import get_llm

SYSTEM = (
    "You answer questions using ONLY the search_documents tool. "
    "Call it, then answer from the returned chunks. After each fact add its citation "
    "marker exactly as given in the chunk's 'citation' field using plain square brackets only (e.g. [file.pdf p.21]). "
    "Never use full-width brackets. "
    "If the chunks do not contain the answer, say so and cite nothing."
)
MAX_TOOL_CALLS = 2


def normalize_citations(text: str) -> str:
    text = text.replace("【", "[").replace("】", "]")
    text = re.sub(r"[\u200b\u200c\u200d\u2060\ufeff]", "", text)  # zero-width characters
    text = text.replace("\u00a0", " ")                            # non-breaking space
    return text


async def ask_with_session(session, tools, question: str) -> str:
    llm = get_llm().bind_tools(tools)
    messages = [SystemMessage(content=SYSTEM), HumanMessage(content=question)]
    calls = 0
    while True:
        reply = await llm.ainvoke(messages)
        messages.append(reply)
        if not reply.tool_calls or calls >= MAX_TOOL_CALLS:
            return normalize_citations(reply.content)
        for tc in reply.tool_calls:
            calls += 1
            result = await session.call_tool(tc["name"], tc["args"])
            if result.isError:
                raise RuntimeError(f"Tool error: {result.content}")
            text = "\n".join(c.text for c in result.content if hasattr(c, "text"))
            print(f"[tool call {calls}] {tc['name']}({tc['args']}) -> {len(text)} chars")
            messages.append(ToolMessage(content=text, tool_call_id=tc["id"]))


async def ask(question: str) -> str:
    params = StdioServerParameters(command=sys.executable, args=["-m", "src.mcp_tools.server"])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            mcp_tools = (await session.list_tools()).tools
            tools = [
                {"type": "function", "function": {
                    "name": t.name, "description": t.description or "",
                    "parameters": t.inputSchema}}
                for t in mcp_tools
            ]
            return await ask_with_session(session, tools, question)


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "What is the passing criteria for the BSc MLT examination?"
    print(asyncio.run(ask(q)))

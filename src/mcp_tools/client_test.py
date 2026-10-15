import asyncio
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main():
    params = StdioServerParameters(command=sys.executable, args=["-m", "src.mcp_tools.server"])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as s:
            await s.initialize()
            print("TOOLS:", [t.name for t in (await s.list_tools()).tools])
            res = await s.call_tool(
                "search_documents", {"query": "passing criteria for the BSc MLT examination"}
            )
            print(res.content[0].text[:700])


asyncio.run(main())
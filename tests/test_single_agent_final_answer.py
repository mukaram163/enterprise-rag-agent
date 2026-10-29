import asyncio
from types import SimpleNamespace as NS

from langchain_core.messages import AIMessage

import src.agents.single_agent as sa


class FakeLLM:
    """Stand-in model: with tools it keeps asking for searches; without tools it answers."""

    def __init__(self, log, bound=False):
        self.log, self.bound = log, bound

    def bind_tools(self, tools):
        return FakeLLM(self.log, bound=True)

    async def ainvoke(self, messages):
        self.log.append((self.bound, list(messages)))
        if self.bound:
            return AIMessage(content="", tool_calls=[
                {"name": "search_documents", "args": {"query": "q"}, "id": f"id{len(self.log)}"}])
        return AIMessage(content="Final answer [a.pdf p.1]")


class FakeSession:
    async def call_tool(self, name, args):
        return NS(isError=False, content=[NS(text="[a.pdf p.1] some retrieved text")])


def test_final_answer_after_the_search_budget_has_no_tool_call_history(monkeypatch):
    log = []
    monkeypatch.setattr(sa, "get_llm", lambda: FakeLLM(log))
    answer = asyncio.run(sa.ask_with_session(FakeSession(), [], "What is X?"))
    bound, messages = log[-1]
    assert not bound
    assert [type(m).__name__ for m in messages] == ["SystemMessage", "HumanMessage"]
    assert "some retrieved text" in messages[1].content
    assert "Final answer" in answer

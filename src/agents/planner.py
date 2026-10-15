import asyncio
import json
import re
import sys
from typing import List

from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.llm import get_llm

MAX_SUBQUESTIONS = 5

SYSTEM = (
    "You split a user question about a university curriculum document into the smallest set of "
    "independent sub-questions that can each be answered by one document search. "
    f"Return at most {MAX_SUBQUESTIONS}. If the question is already simple, return it unchanged as a single item. "
    "Each sub-question must be self-contained (name the exam or subject explicitly, no pronouns). "
    'Respond with JSON only, exactly: {"sub_questions": ["...", "..."]}'
)


def parse_plan(text: str, original: str) -> List[str]:
    match = re.search(r"\{.*\}", text, re.S)
    try:
        subs = json.loads(match.group(0))["sub_questions"] if match else []
        subs = [s.strip() for s in subs if isinstance(s, str) and s.strip()]
    except (ValueError, KeyError, TypeError):
        subs = []
    if not subs:
        return [original]  # fall back to the original question, never fail empty
    return subs[:MAX_SUBQUESTIONS]


async def plan(question: str) -> List[str]:
    reply = await get_llm().ainvoke([SystemMessage(content=SYSTEM), HumanMessage(content=question)])
    return parse_plan(reply.content, question)


async def _demo(qs):
    for q in qs:
        print("Q:", q)
        for i, s in enumerate(await plan(q), 1):
            print(f"  {i}. {s}")


if __name__ == "__main__":
    qs = sys.argv[1:] or [
        "Compare the passing rules and total marks of the Second and Final Professional exams.",
        "What are the attempt limits for each professional exam, and what attendance is required for the final?",
        "Which subjects are in the third year, and how are the marks split across them?",
        "What is the course duration?",
    ]
    asyncio.run(_demo(qs))

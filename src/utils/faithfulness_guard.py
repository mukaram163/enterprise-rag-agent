import os
import re
import sys
import types
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

# Same polyfill as run_eval.py (RAGAS 0.2.x import issue)
if "langchain_community.chat_models.vertexai" not in sys.modules:
    _dummy = types.ModuleType("langchain_community.chat_models.vertexai")
    _dummy.ChatVertexAI = type("ChatVertexAI", (object,), {})
    sys.modules["langchain_community.chat_models.vertexai"] = _dummy

_metric = None


def strip_markers(text: str) -> str:
    text = re.sub(r"\[\s*p\.[^\]]*\]", "", text)
    return re.sub(r"\s{2,}", " ", text).strip()


def _get_metric():
    global _metric
    if _metric is None:
        from ragas.metrics import Faithfulness
        from ragas.llms import LangchainLLMWrapper
        from langchain_groq import ChatGroq
        judge = ChatGroq(
            model="openai/gpt-oss-120b",
            groq_api_key=os.environ["GROQ_API_KEY"],
            temperature=0.0,
        )
        _metric = Faithfulness(llm=LangchainLLMWrapper(judge))
    return _metric


async def check_faithfulness(question: str, answer: str, contexts: List[str]) -> Optional[float]:
    """Faithfulness score 0-1, or None if the judge failed. Never raises."""
    try:
        from ragas import SingleTurnSample
        sample = SingleTurnSample(
            user_input=question,
            response=strip_markers(answer),
            retrieved_contexts=contexts,
        )
        return await _get_metric().single_turn_ascore(sample)
    except Exception as e:
        logger.warning(f"Faithfulness check failed (answer not blocked): {e}")
        return None
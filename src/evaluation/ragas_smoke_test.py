import sys
import types
import warnings
import os
import json
import re
import asyncio
from dotenv import load_dotenv

load_dotenv(override=True)

# Make sure the eval never silently falls back to another provider.
os.environ["EVAL_MODE"] = "1"

warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*coroutine 'ClientResponse.json' was never awaited.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

# Polyfill ChatVertexAI module for RAGAS 0.2.10 compatibility
if "langchain_community.chat_models.vertexai" not in sys.modules:
    dummy_module = types.ModuleType("langchain_community.chat_models.vertexai")
    dummy_module.ChatVertexAI = type("ChatVertexAI", (object,), {})
    sys.modules["langchain_community.chat_models.vertexai"] = dummy_module

from ragas import SingleTurnSample
from ragas.metrics import Faithfulness
from ragas.llms import LangchainLLMWrapper
from langchain_groq import ChatGroq

from src.pipeline import run_rag


def get_judge_llm():
    """Judge must differ from the generator (Gemini), so Groq only."""
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise ValueError("GROQ_API_KEY missing: judge must not be the same model as the generator.")
    print("RAGAS judge: Groq (openai/gpt-oss-120b)")
    return ChatGroq(model="openai/gpt-oss-120b", groq_api_key=key, temperature=0.0)


def strip_markers(text: str) -> str:
    """Remove [p.21] / [p.25, p.28] markers so they aren't judged as claims."""
    text = re.sub(r"\[\s*p\.[^\]]*\]", "", text)
    return re.sub(r"\s{2,}", " ", text).strip()


async def main():
    with open("data/golden_dataset.json", "r", encoding="utf-8") as f:
        golden_data = json.load(f)

    item = golden_data[3]
    question = item["question"]

    print("\n[1/2] Running real pipeline (run_rag)...")
    out = await run_rag(question)
    answer = out["answer"]
    contexts = out["contexts"]

    print("[2/2] Running RAGAS faithfulness...")
    evaluator_llm = LangchainLLMWrapper(get_judge_llm())
    metric = Faithfulness(llm=evaluator_llm)

    sample = SingleTurnSample(
        user_input=question,
        response=strip_markers(answer),
        retrieved_contexts=contexts,
    )
    score = await metric.single_turn_ascore(sample)

    print("\n" + "=" * 60)
    print(f"QUESTION:      {question}")
    print(f"ANSWER:        {answer}")
    print(f"PROVIDER:      {out['provider']}")
    print(f"CHUNK PAGES:   {[c['page_number'] for c in out['reranked_chunks']]}")
    print(f"CITATIONS:     {out['citations']}")
    print(f"EXPECTED PAGE: {item['expected_page']}")
    print(f"FAITHFULNESS:  {score}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
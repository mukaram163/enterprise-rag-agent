import sys
import types
import warnings
import os
import json
import time
import asyncio
from dotenv import load_dotenv

load_dotenv(override=True)

warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*coroutine 'ClientResponse.json' was never awaited.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

# Polyfill ChatVertexAI module for RAGAS 0.2.10 compatibility
if "langchain_community.chat_models.vertexai" not in sys.modules:
    dummy_module = types.ModuleType("langchain_community.chat_models.vertexai")
    dummy_module.ChatVertexAI = type("ChatVertexAI", (object,), {})
    sys.modules["langchain_community.chat_models.vertexai"] = dummy_module

# RAGAS & LangChain LLM imports
from ragas import SingleTurnSample
from ragas.metrics import Faithfulness
from ragas.llms import LangchainLLMWrapper
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from src.generation.rag_chain import run_rag_pipeline_standalone


def get_judge_llm():
    """
    Selects the best available judge LLM for RAGAS evaluation.
    Prioritizes Groq -> Gemini -> OpenAI based on available API keys.
    """
    # 1. Try Groq (Llama 3.3 / Llama 3.1)
    if os.getenv("GROQ_API_KEY"):
        try:
            print("Configuring RAGAS Judge: Groq (openai/gpt-oss-120b)...")
            return ChatGroq(
                model="openai/gpt-oss-120b",
                groq_api_key=os.getenv("GROQ_API_KEY"),
                temperature=0.0
            )
        except Exception as e:
            print(f"Groq setup failed ({e}), falling back to Gemini...")

    # 2. Try Gemini
    if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
        print("Configuring RAGAS Judge: Gemini (gemini-2.5-flash)...")
        return ChatGoogleGenerativeAI(
            model="gemini-3.8-flash",
            google_api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"),
            temperature=0.0
        )

    # 3. Try OpenAI
    if os.getenv("OPENAI_API_KEY"):
        print("Configuring RAGAS Judge: OpenAI (gpt-4o-mini)...")
        return ChatOpenAI(
            model="gpt-4o-mini",
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.0
        )

    raise ValueError("No valid API keys found in environment variables!")


async def main():
    # 1. Load Golden Dataset Item #4
    dataset_path = "data/golden_dataset.json"
    with open(dataset_path, "r", encoding="utf-8") as f:
        golden_data = json.load(f)
    
    golden_item = golden_data[3]
    question = golden_item["question"]
    
    # 2. Run real pipeline
    print("\n[1/2] Running RAG Pipeline...")
    pipeline_out = run_rag_pipeline_standalone(question)
    answer = pipeline_out["answer"]
    contexts = pipeline_out["contexts"]
    
    # 3. Configure Judge LLM
    print("\n[2/2] Running RAGAS Faithfulness Evaluation...")
    judge_llm = get_judge_llm()
    evaluator_llm = LangchainLLMWrapper(judge_llm)
    faithfulness_metric = Faithfulness(llm=evaluator_llm)
    
    # 4. Input shape for RAGAS
    sample = SingleTurnSample(
        user_input=question,
        response=answer,
        retrieved_contexts=contexts
    )
    
    # Calculate Faithfulness score
    score = await faithfulness_metric.single_turn_ascore(sample)
    
    # 5. Output Results
    print("\n" + "=" * 60)
    print(f"QUESTION:           {question}")
    print(f"ANSWER:             {answer}")
    print(f"RETRIEVED CHUNKS:   {len(contexts)}")
    print(f"FAITHFULNESS SCORE: {score}")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
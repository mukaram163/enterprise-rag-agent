import os

from dotenv import load_dotenv

load_dotenv()


def get_llm():
    provider = os.getenv("AGENT_PROVIDER", "groq").lower()
    model = os.getenv("AGENT_MODEL", "openai/gpt-oss-120b")
    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=model, groq_api_key=os.environ["GROQ_API_KEY"], temperature=0.0)
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model, google_api_key=os.environ["GEMINI_API_KEY"], temperature=0.0)
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, api_key=os.environ["OPENAI_API_KEY"], temperature=0.0)
    raise ValueError(f"Unknown AGENT_PROVIDER: {provider}")
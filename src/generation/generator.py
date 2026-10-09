import os
import logging
from typing import List, AsyncGenerator, Any

logger = logging.getLogger(__name__)


class GeminiGenerator:
    """
    Multi-provider generator that attempts Groq first, falls back to Gemini,
    and then OpenAI while keeping backwards compatibility.
    """
    def __init__(self, api_key: str = None, model_name: str = None):
        self.groq_key = os.getenv("GROQ_API_KEY")
        self.gemini_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.model_name = model_name or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    def _build_prompt(self, query: str, context_chunks: List[Any]) -> str:
        formatted_context = ""
        for i, chunk in enumerate(context_chunks, 1):
            if isinstance(chunk, dict):
                text = chunk.get("text", "")
                meta = chunk.get("metadata", {}) or {}
                fname = chunk.get("file_name") or meta.get("file_name") or "document.pdf"
                page = chunk.get("page_number") or meta.get("page_number") or 1
            else:
                text = getattr(chunk, "text", "")
                fname = getattr(chunk, "file_name", "document.pdf")
                page = getattr(chunk, "page_number", 1)

            formatted_context += f"\n--- Chunk {i} [File: {fname}] [page: {page}] ---\n{text}\n"

        return (
            "You are a strict QA assistant. Answer the query based ONLY on the provided context chunks.\n\n"
            f"Context:\n{formatted_context}\n\n"
            "INSTRUCTIONS FOR CITATIONS:\n"
            "1. After stating any fact, requirement, or point, append an inline page marker in the format '[p.N]' "
            "where N is the exact page number from the corresponding '[page: N]' marker.\n"
            "2. NEVER cite a page number that is not explicitly present in the provided context.\n"
            "3. Do not invent page numbers or citations.\n\n"
            f"Query: {query}\n"
            "Answer:"
        )

    def generate_response(self, query: str, context_chunks: List[Any]) -> str:
        prompt = self._build_prompt(query, context_chunks)

        # 1. Primary Provider: Groq
        if self.groq_key:
            try:
                from langchain_groq import ChatGroq
                llm = ChatGroq(
                    model="openai/gpt-oss-120b",
                    groq_api_key=self.groq_key,
                    temperature=0.0
                )
                res = llm.invoke(prompt)
                if res and res.content:
                    return str(res.content)
            except Exception as e:
                logger.warning(f"Groq generation failed/rate limited: {e}. Falling back to Gemini...")

        # 2. Secondary Provider: Gemini
        if self.gemini_key:
            try:
                from google import genai
                client = genai.Client(api_key=self.gemini_key)
                response = client.models.generate_content(
                    model=self.model_name,
                    contents=prompt
                )
                if response and response.text:
                    return response.text
            except Exception as e:
                logger.warning(f"Gemini generation failed/rate limited: {e}. Falling back to OpenAI...")

        # 3. Tertiary Provider: OpenAI
        if self.openai_key:
            try:
                from langchain_openai import ChatOpenAI
                llm = ChatOpenAI(
                    model="gpt-4o-mini",
                    api_key=self.openai_key,
                    temperature=0.0
                )
                res = llm.invoke(prompt)
                if res and res.content:
                    return str(res.content)
            except Exception as e:
                logger.error(f"OpenAI generation failed: {e}")

        return "Error: All API providers (Groq, Gemini, OpenAI) failed or rate limits were exhausted."

    async def generate_stream(self, query: str, context_chunks: List[Any]) -> AsyncGenerator[str, None]:
        import asyncio
        prompt = self._build_prompt(query, context_chunks)
        eval_mode = os.getenv("EVAL_MODE", "0").lower() in ("1", "true", "yes")

        if self.gemini_key:
            from google import genai
            client = genai.Client(api_key=self.gemini_key)
            last_err = None
            for attempt in range(4):
                has_yielded = False
                try:
                    stream_result = client.aio.models.generate_content_stream(
                        model=self.model_name,
                        contents=prompt,
                    )
                    if hasattr(stream_result, "__await__"):
                        stream_result = await stream_result

                    async for chunk in stream_result:
                        if hasattr(chunk, "text") and chunk.text:
                            has_yielded = True
                            yield chunk.text

                    if has_yielded:
                        return
                except Exception as e:
                    last_err = e
                    logger.warning(f"Gemini streaming failed (attempt {attempt + 1}/4): {e}")
                    if has_yielded:
                        raise  # tokens already sent: surface an error, never append a second answer
                    await asyncio.sleep(2 ** attempt)

            if eval_mode:
                raise RuntimeError(f"Gemini unavailable after retries: {last_err}")

        logger.warning("Falling back to sync response.")
        yield self.generate_response(query, context_chunks)


MultiProviderGenerator = GeminiGenerator
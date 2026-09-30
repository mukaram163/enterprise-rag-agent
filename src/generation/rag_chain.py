from typing import List, Dict, Any, AsyncGenerator

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()


class GeminiRAGGenerator:
    def __init__(self, model_name: str = "gemini-3.8-flash"):
        self.model_name = model_name
        self.client = genai.Client()

    def format_context(self, retrieved_items: List[Dict[str, Any]]) -> str:
        formatted_blocks = []

        for i, item in enumerate(retrieved_items, 1):
            page = item.get("page_number", "N/A")
            file_name = item.get("file_name", "Document")
            text = item.get("text", "").strip()
            score = item.get("rerank_score", 0.0)

            block = (
                f"[Source {i} | File: {file_name} | Page: {page} | "
                f"Relevance Score: {score:.2f}]\n"
                f"{text}"
            )

            formatted_blocks.append(block)

        return "\n\n---\n\n".join(formatted_blocks)

    async def generate_response_stream(
        self,
        query: str,
        retrieved_items: List[Dict[str, Any]]
    ) -> AsyncGenerator[str, None]:

        formatted_context = self.format_context(retrieved_items)

        system_instruction = (
            "You are an expert AI assistant answering questions based solely "
            "on retrieved internal documents.\n"
            "Guidelines:\n"
            "1. Rely strictly on the provided context to answer the prompt.\n"
            "2. If the context does not contain sufficient information, "
            "state that clearly.\n"
            "3. Cite relevant page numbers when referencing facts."
        )

        user_prompt = f"""
Retrieved Document Context:
{formatted_context}

----------------
User Query:
{query}

Answer:
"""

        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.2,
        )

        response_stream = await self.client.aio.models.generate_content_stream(
            model=self.model_name,
            contents=user_prompt,
            config=config,
        )

        async for chunk in response_stream:
            if chunk.text:
                yield chunk.text
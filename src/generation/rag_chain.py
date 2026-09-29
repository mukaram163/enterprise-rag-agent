import os
import time
from typing import List, Dict, Any
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import ServerError

load_dotenv()

class GeminiRAGGenerator:
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        self.client = genai.Client()
        self.model_name = model_name

    def format_context(self, retrieved_items: List[Dict[str, Any]]) -> str:
        formatted_blocks = []
        for i, item in enumerate(retrieved_items, 1):
            page = item.get("page_number", "N/A")
            file_name = item.get("file_name", "Document")
            text = item.get("text", "").strip()
            score = item.get("rerank_score", 0.0)
            
            block = (
                f"[Source {i} | File: {file_name} | Page: {page} | Relevance Score: {score:.2f}]\n"
                f"{text}"
            )
            formatted_blocks.append(block)
        
        return "\n\n---\n\n".join(formatted_blocks)

    def generate_response(self, query: str, retrieved_items: List[Dict[str, Any]], max_retries: int = 3) -> str:
        formatted_context = self.format_context(retrieved_items)

        system_instruction = (
            "You are an expert AI assistant answering questions based solely on retrieved internal documents. "
            "Guidelines:\n"
            "1. Rely strictly on the provided context to answer the prompt.\n"
            "2. If the context does not contain sufficient information, state that clearly.\n"
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

        # Retry loop for handling temporary 503 capacity spikes
        for attempt in range(1, max_retries + 1):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=user_prompt,
                    config=config,
                )
                return response.text
            except ServerError as e:
                if attempt == max_retries:
                    raise e
                print(f"[Warning] Gemini API 503 high demand spike. Retrying in 2 seconds (Attempt {attempt}/{max_retries})...")
                time.sleep(2)
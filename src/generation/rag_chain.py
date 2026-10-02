import os
import time
import logging
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

logger = logging.getLogger(__name__)

class GeminiRAGGenerator:
    def __init__(self, model_name: Optional[str] = None):
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.model_name = model_name or os.getenv("GEMINI_MODEL")

        if not self.model_name:
            raise RuntimeError("GEMINI_MODEL is not configured in .env")
        
        if not self.api_key:
            logger.warning("GEMINI_API_KEY not found in environment variables.")

        self.client = genai.Client(api_key=self.api_key) if self.api_key else None

    def _format_context(self, retrieved_items: List[Dict[str, Any]]) -> str:
        if not retrieved_items:
            return "No relevant context found in documents."

        context_blocks = []
        for idx, item in enumerate(retrieved_items, 1):
            file_name = item.get("file_name", "Unknown File")
            page_number = item.get("page_number", 1)
            text = item.get("text", "")
            context_blocks.append(f"[Document {idx}: {file_name} (Page {page_number})]\n{text}")

        return "\n\n".join(context_blocks)

    def generate_response(self, query: str, retrieved_items: List[Dict[str, Any]]) -> str:
        if not self.client:
            raise RuntimeError("Gemini Client is not initialized. Check GEMINI_API_KEY.")

        context_str = self._format_context(retrieved_items)
        prompt = f"""You are a helpful Enterprise AI Assistant. Use the provided context to answer the question accurately.

Context:
{context_str}

Question: {query}

Answer:"""

        max_retries = 3
        backoff_delay = 2.0

        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.2,
                        max_output_tokens=1024,
                    )
                )
                return response.text
            except Exception as e:
                if attempt < max_retries - 1 and ("503" in str(e) or "UNAVAILABLE" in str(e)):
                    logger.warning(f"Gemini API 503 spike encountered. Retrying in {backoff_delay}s... (Attempt {attempt+1}/{max_retries})")
                    time.sleep(backoff_delay)
                    backoff_delay *= 2
                else:
                    logger.error(f"Gemini generation error: {e}")
                    raise e
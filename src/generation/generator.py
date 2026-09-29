import os
import time
from typing import List, Dict, Any
from dotenv import load_dotenv
from google import genai
from google.genai import errors

# Load environment variables from .env
load_dotenv()

class ResponseGenerator:
    def __init__(self, model_name: str = "gemini-3.8-flash", fallback_model: str = "gemini-3.8-pro"):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set in .env or environment.")
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.fallback_model = fallback_model

    def build_prompt(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        """Constructs a grounded RAG prompt with retrieved document context."""
        context_str = ""
        for i, chunk in enumerate(context_chunks, 1):
            context_str += f"\n--- Context Document [{i}] ---\n"
            context_str += f"Source: {chunk.get('file_name', 'Unknown')} (Page {chunk.get('page_number', 'N/A')})\n"
            context_str += f"Content: {chunk.get('text', '')}\n"

        return f"""You are an enterprise AI assistant answering questions based strictly on the provided context documents.

[CONTEXT DOCUMENTS]
{context_str}

[USER QUESTION]
{query}

[INSTRUCTIONS]
1. Answer the question accurately using ONLY the information from the provided context documents.
2. Cite the source document name and page number for facts stated in your answer.
3. If the answer cannot be determined from the context documents, state clearly: "I cannot answer this question based on the provided documents."
"""

    def generate_response(self, query: str, context_chunks: List[Dict[str, Any]], retries: int = 3) -> str:
        """Generates a grounded answer using Gemini API with retry and fallback mechanism."""
        prompt = self.build_prompt(query, context_chunks)
        
        # Try primary model (gemini-3.8-flash) with retries
        for attempt in range(retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                )
                return response.text
            except errors.ServerError:
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff (1s, 2s, 4s)
                    continue
                print(f"Primary model {self.model_name} failed with 503. Switching to fallback model {self.fallback_model}...")

        # Fallback model attempt (gemini-3.8-pro)
        try:
            response = self.client.models.generate_content(
                model=self.fallback_model,
                contents=prompt,
            )
            return response.text
        except Exception as e:
            raise RuntimeError(f"Failed to generate response after retries and fallback: {e}")
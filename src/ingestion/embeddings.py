from typing import List
from sentence_transformers import SentenceTransformer

class EmbeddingGenerator:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generates embedding vectors locally using Hugging Face models."""
        if not texts:
            return []

        cleaned_texts = [text.replace("\n", " ") if text.strip() else " " for text in texts]
        embeddings = self.model.encode(cleaned_texts, convert_to_numpy=True)
        return embeddings.tolist()
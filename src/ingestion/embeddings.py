import os

# Suppress Hugging Face tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from typing import List
from transformers import logging as tf_logging
from sentence_transformers import SentenceTransformer

# Suppress transformer logging output
tf_logging.set_verbosity_error()


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


def get_embedding(text: str, model_name: str = "all-MiniLM-L6-v2") -> List[float]:
    """Convenience helper function to retrieve a single embedding vector."""
    generator = EmbeddingGenerator(model_name=model_name)
    embeddings = generator.generate_embeddings([text])
    return embeddings[0] if embeddings else []
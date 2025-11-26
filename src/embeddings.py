"""
Embedding generation using sentence transformers
"""
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List

class EmbeddingGenerator:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text"""
        return self.model.encode([text])[0]
    
    def generate_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for multiple texts"""
        if not texts:
            return np.array([])
        return self.model.encode(texts)
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings"""
        return self.model.get_sentence_embedding_dimension()
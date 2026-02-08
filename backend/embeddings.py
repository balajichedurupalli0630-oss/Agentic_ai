import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List


class Embeddings:
    """Embedding generation using SentenceTransformer"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the embeddings model"""
        self.model_name = model_name
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the Sentence Transformer model"""
        try:
            print(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            print(f"Model loaded successfully. Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
        except Exception as e:
            print(f"Error loading model {self.model_name}: {e}")
            raise
    
    def generate_embedding(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for a list of texts
        
        Args:
            texts: List of strings to embed
            batch_size: Batch size for encoding
            
        Returns:
            numpy array of embeddings with shape (len(texts), embedding_dim)
        """
        if not self.model:
            raise ValueError("Model not loaded")
        
        if not texts:
            raise ValueError("No texts provided for embeddings")
        
        show_progress = len(texts) >= 10

        print(f"Generating embeddings for {len(texts)} texts...")
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True
            )
            print(f"Generated embeddings with shape: {embeddings.shape}")
            return embeddings
        except Exception as e:
            print(f"[ERROR] in generating embeddings: {e}")
            raise

embedding_loader = Embeddings()
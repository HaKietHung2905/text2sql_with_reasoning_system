"""
Embedding generation utilities.
Handles text-to-vector conversion using sentence transformers.
"""

from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer
from utils.logging_utils import get_logger

logger = get_logger(__name__)


class EmbeddingGenerator:
    """Generate embeddings using sentence transformers"""
    
    DEFAULT_MODEL = 'all-MiniLM-L6-v2'
    
    def __init__(self, model_name: str = None):
        """
        Initialize embedding generator
        
        Args:
            model_name: Name of sentence transformer model
        """
        self.model_name = model_name or self.DEFAULT_MODEL
        logger.info(f"Loading sentence transformer model: {self.model_name}")
        
        try:
            self.model = SentenceTransformer(self.model_name)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def encode(self, text: Union[str, List[str]]) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Generate embeddings for text
        
        Args:
            text: Single text or list of texts
            
        Returns:
            Embedding vector(s)
        """
        try:
            return self.model.encode(text)
        except Exception as e:
            logger.error(f"Encoding failed: {e}")
            raise
    
    def encode_batch(self, texts: List[str], batch_size: int = 32) -> List[np.ndarray]:
        """
        Generate embeddings in batches
        
        Args:
            texts: List of texts
            batch_size: Batch size for encoding
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.encode(batch)
            embeddings.extend(batch_embeddings)
            
            if (i // batch_size + 1) % 10 == 0:
                logger.info(f"Encoded {i + len(batch)}/{len(texts)} texts")
        
        return embeddings
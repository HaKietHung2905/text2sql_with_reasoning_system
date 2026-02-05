"""
ChromaDB operations for Spider dataset.
Handles vector database operations for Text-to-SQL retrieval.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import chromadb

from utils.logging_utils import get_logger

logger = get_logger(__name__)


class ChromaDBHandler:
    """Handle ChromaDB operations"""
    
    def __init__(self, persist_dir: str = "./data/embeddings/chroma_db"):
        """
        Initialize ChromaDB handler
        
        Args:
            persist_dir: Directory for persistent storage
        """
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize client
        self.client = self._initialize_client()
        
        # Collections
        self.schema_collection = None
        self.question_collection = None
        self.sql_collection = None
    
    def _initialize_client(self) -> chromadb.Client:
        """
        Initialize ChromaDB client with fallback
        
        Returns:
            ChromaDB client instance
        """
        try:
            client = chromadb.PersistentClient(path=str(self.persist_dir))
            logger.info("Using persistent ChromaDB client")
            return client
        except Exception as e:
            logger.warning(f"Persistent client failed: {e}")
            logger.info("Using in-memory ChromaDB client")
            return chromadb.Client()
    
    def delete_collection(self, name: str) -> bool:
        """
        Delete a collection if it exists
        
        Args:
            name: Collection name
            
        Returns:
            True if deleted, False otherwise
        """
        try:
            self.client.delete_collection(name)
            logger.info(f"Deleted collection: {name}")
            return True
        except Exception as e:
            logger.debug(f"Could not delete collection {name}: {e}")
            return False
    
    def create_collection(self, name: str) -> chromadb.Collection:
        """
        Create or get a collection
        
        Args:
            name: Collection name
            
        Returns:
            Collection instance
        """
        collection = self.client.get_or_create_collection(name)
        logger.info(f"Created/retrieved collection: {name}")
        return collection
    
    def setup_collections(self, reset: bool = True) -> None:
        """
        Setup standard Spider collections
        
        Args:
            reset: If True, delete existing collections first
        """
        collection_names = ["spider_schemas", "spider_questions", "spider_sql"]
        
        if reset:
            for name in collection_names:
                self.delete_collection(name)
        
        self.schema_collection = self.create_collection("spider_schemas")
        self.question_collection = self.create_collection("spider_questions")
        self.sql_collection = self.create_collection("spider_sql")
        
        logger.info("All collections setup complete")
    
    def add_batch(
        self,
        collection: chromadb.Collection,
        documents: List[str],
        embeddings: List[List[float]],
        ids: List[str],
        metadatas: List[Dict[str, Any]],
        batch_size: int = 1000
    ) -> int:
        """
        Add documents to collection in batches
        
        Args:
            collection: Target collection
            documents: Document texts
            embeddings: Document embeddings
            ids: Document IDs
            metadatas: Document metadata
            batch_size: Batch size for adding
            
        Returns:
            Number of documents successfully added
        """
        total_items = len(documents)
        added_count = 0
        
        logger.info(f"Adding {total_items} items in batches of {batch_size}...")
        
        for i in range(0, total_items, batch_size):
            end_idx = min(i + batch_size, total_items)
            
            batch_docs = documents[i:end_idx]
            batch_embeddings = embeddings[i:end_idx]
            batch_ids = ids[i:end_idx]
            batch_metadatas = metadatas[i:end_idx]
            
            batch_num = i // batch_size + 1
            total_batches = (total_items - 1) // batch_size + 1
            
            try:
                collection.add(
                    documents=batch_docs,
                    embeddings=batch_embeddings,
                    ids=batch_ids,
                    metadatas=batch_metadatas
                )
                added_count += len(batch_docs)
                logger.info(f"  ✓ Batch {batch_num}/{total_batches} ({len(batch_docs)} items)")
                
            except Exception as e:
                logger.error(f"  ✗ Error adding batch {batch_num}: {e}")
                
                # Try smaller batch size
                if batch_size > 100:
                    logger.info(f"  Retrying batch {batch_num} with smaller size...")
                    added_count += self.add_batch(
                        collection,
                        batch_docs,
                        batch_embeddings,
                        batch_ids,
                        batch_metadatas,
                        batch_size=100
                    )
                else:
                    logger.warning(f"  Failed to add batch {batch_num}, skipping...")
        
        return added_count
    
    def query_collection(
        self,
        collection: chromadb.Collection,
        query_embeddings: List[List[float]],
        n_results: int = 5,
        where: Optional[Dict] = None
    ) -> Dict:
        """
        Query a collection
        
        Args:
            collection: Collection to query
            query_embeddings: Query embedding vectors
            n_results: Number of results to return
            where: Metadata filter
            
        Returns:
            Query results
        """
        try:
            results = collection.query(
                query_embeddings=query_embeddings,
                n_results=n_results,
                where=where
            )
            return results
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return {}
    
    def get_collection_count(self, collection: chromadb.Collection) -> int:
        """
        Get number of items in collection
        
        Args:
            collection: Collection to count
            
        Returns:
            Number of items
        """
        try:
            return collection.count()
        except Exception as e:
            logger.error(f"Count failed: {e}")
            return 0
    
    def get_statistics(self) -> Dict[str, int]:
        """
        Get statistics for all collections
        
        Returns:
            Dictionary with collection counts
        """
        stats = {}
        
        if self.schema_collection:
            stats['schemas'] = self.get_collection_count(self.schema_collection)
        
        if self.question_collection:
            stats['questions'] = self.get_collection_count(self.question_collection)
        
        if self.sql_collection:
            stats['sql_queries'] = self.get_collection_count(self.sql_collection)
        
        return stats
"""
Retrieval system for finding similar examples.
Handles semantic search across questions, SQL, and schemas.
"""

from typing import List, Dict, Any, Optional
import numpy as np

from src.retrieval.chromadb_handler import ChromaDBHandler
from utils.embedding_utils import EmbeddingGenerator
from utils.logging_utils import get_logger

logger = get_logger(__name__)


class SpiderRetriever:
    """Retrieve similar examples from Spider dataset"""
    
    def __init__(
        self,
        persist_dir: str = "./data/embeddings/chroma_db",
        model_name: str = None
    ):
        """
        Initialize retriever
        
        Args:
            persist_dir: ChromaDB persist directory
            model_name: Embedding model name
        """
        self.db_handler = ChromaDBHandler(persist_dir)
        self.embedding_gen = EmbeddingGenerator(model_name)
        
        # Load collections
        self._load_collections()
    
    def _load_collections(self) -> bool:
        """
        Load existing collections
        
        Returns:
            True if collections loaded successfully
        """
        try:
            self.schema_collection = self.db_handler.client.get_collection("spider_schemas")
            self.question_collection = self.db_handler.client.get_collection("spider_questions")
            self.sql_collection = self.db_handler.client.get_collection("spider_sql")
            
            # Check if collections have data
            schema_count = self.db_handler.get_collection_count(self.schema_collection)
            question_count = self.db_handler.get_collection_count(self.question_collection)
            sql_count = self.db_handler.get_collection_count(self.sql_collection)
            
            logger.info(f"Loaded collections: Schemas={schema_count}, Questions={question_count}, SQL={sql_count}")
            
            if schema_count == 0 or question_count == 0:
                logger.warning("Collections are empty. Need to populate with data.")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load collections: {e}")
            logger.info("Run build_chromadb.py first to populate ChromaDB")
            return False
    
    def _distance_to_similarity(self, distance: float) -> float:
        """
        Convert distance to similarity score
        
        Args:
            distance: Distance value
            
        Returns:
            Similarity score (0-1)
        """
        return 1 - distance
    
    def retrieve_similar_questions(
        self,
        query: str,
        n_results: int = 5,
        min_similarity: float = 0.3,
        db_filter: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Find similar questions from the dataset
        
        Args:
            query: User question
            n_results: Number of results to return
            min_similarity: Minimum similarity threshold
            db_filter: Filter by database ID
            
        Returns:
            Dictionary with results
        """
        if not self.question_collection:
            return {"error": "Question collection not available"}
        
        try:
            # Generate embedding
            query_embedding = self.embedding_gen.encode(query)
            
            # Prepare filter
            where = {"db_id": db_filter} if db_filter else None
            
            # Query
            results = self.db_handler.query_collection(
                self.question_collection,
                query_embeddings=[query_embedding.tolist()],
                n_results=n_results,
                where=where
            )
            
            if not results.get('metadatas', [[]])[0]:
                return {"error": "No similar questions found"}
            
            # Process results
            similar_queries = []
            for i, (metadata, distance) in enumerate(zip(
                results['metadatas'][0],
                results['distances'][0]
            )):
                similarity = self._distance_to_similarity(distance)
                
                if similarity >= min_similarity:
                    similar_queries.append({
                        'rank': i + 1,
                        'question': metadata.get('question', ''),
                        'sql_query': metadata.get('sql', ''),
                        'database': metadata.get('db_id', 'unknown'),
                        'similarity_score': round(similarity, 4),
                        'split': metadata.get('split', 'unknown')
                    })
            
            return {
                "query": query,
                "total_results": len(similar_queries),
                "results": similar_queries
            }
            
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return {"error": str(e)}
    
    def retrieve_similar_sql(
        self,
        query: str,
        n_results: int = 5,
        min_similarity: float = 0.3,
        db_filter: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Find similar SQL queries
        
        Args:
            query: User question
            n_results: Number of results
            min_similarity: Minimum similarity threshold
            db_filter: Filter by database ID
            
        Returns:
            Dictionary with results
        """
        if not self.sql_collection:
            return {"error": "SQL collection not available"}
        
        try:
            # Generate embedding
            query_embedding = self.embedding_gen.encode(query)
            
            # Prepare filter
            where = {"db_id": db_filter} if db_filter else None
            
            # Query
            results = self.db_handler.query_collection(
                self.sql_collection,
                query_embeddings=[query_embedding.tolist()],
                n_results=n_results,
                where=where
            )
            
            if not results.get('metadatas', [[]])[0]:
                return {"error": "No similar SQL queries found"}
            
            # Process results
            similar_sql = []
            for i, (document, metadata, distance) in enumerate(zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            )):
                similarity = self._distance_to_similarity(distance)
                
                if similarity >= min_similarity:
                    similar_sql.append({
                        'rank': i + 1,
                        'sql_query': document,
                        'original_question': metadata.get('question', ''),
                        'database': metadata.get('db_id', 'unknown'),
                        'similarity_score': round(similarity, 4),
                        'split': metadata.get('split', 'unknown')
                    })
            
            return {
                "query": query,
                "total_results": len(similar_sql),
                "results": similar_sql
            }
            
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return {"error": str(e)}
    
    def retrieve_relevant_schemas(
        self,
        query: str,
        n_results: int = 3
    ) -> Dict[str, Any]:
        """
        Find relevant database schemas
        
        Args:
            query: User question
            n_results: Number of schemas to return
            
        Returns:
            Dictionary with schema results
        """
        if not self.schema_collection:
            return {"error": "Schema collection not available"}
        
        try:
            # Generate embedding
            query_embedding = self.embedding_gen.encode(query)
            
            # Query
            results = self.db_handler.query_collection(
                self.schema_collection,
                query_embeddings=[query_embedding.tolist()],
                n_results=n_results
            )
            
            if not results.get('metadatas', [[]])[0]:
                return {"error": "No relevant schemas found"}
            
            # Process results
            relevant_schemas = []
            for i, (document, metadata, distance) in enumerate(zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            )):
                similarity = self._distance_to_similarity(distance)
                relevant_schemas.append({
                    'rank': i + 1,
                    'database': metadata.get('db_id', 'unknown'),
                    'schema': document,
                    'tables': metadata.get('num_tables', 0),
                    'columns': metadata.get('num_columns', 0),
                    'similarity_score': round(similarity, 4)
                })
            
            return {
                "query": query,
                "relevant_schemas": relevant_schemas
            }
            
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return {"error": str(e)}
    
    def retrieve_comprehensive(
        self,
        query: str,
        n_results: int = 3,
        min_similarity: float = 0.3
    ) -> Dict[str, Any]:
        """
        Get comprehensive retrieval results
        
        Args:
            query: User question
            n_results: Number of results per category
            min_similarity: Minimum similarity threshold
            
        Returns:
            Dictionary with all retrieval results
        """
        return {
            "query": query,
            "similar_questions": self.retrieve_similar_questions(
                query, n_results, min_similarity
            ),
            "similar_sql": self.retrieve_similar_sql(
                query, n_results, min_similarity
            ),
            "relevant_schemas": self.retrieve_relevant_schemas(
                query, n_results
            )
        }
    
    def get_all_schemas(self) -> List[Dict[str, Any]]:
        """
        Get all available database schemas
        
        Returns:
            List of schema dictionaries
        """
        if not self.schema_collection:
            logger.error("Schema collection not available")
            return []
        
        try:
            all_schemas = self.schema_collection.get(
                include=['documents', 'metadatas']
            )
            
            schemas = []
            for document, metadata in zip(
                all_schemas.get('documents', []),
                all_schemas.get('metadatas', [])
            ):
                schemas.append({
                    'database': metadata.get('db_id', 'unknown'),
                    'schema': document,
                    'tables': metadata.get('num_tables', 0),
                    'columns': metadata.get('num_columns', 0)
                })
            
            return schemas
            
        except Exception as e:
            logger.error(f"Failed to retrieve schemas: {e}")
            return []
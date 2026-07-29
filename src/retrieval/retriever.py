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
    """Retrieve similar examples from a Text-to-SQL dataset (Spider, WikiSQL, ...)"""

    def __init__(
        self,
        persist_dir: str = "./data/embeddings/chroma_db",
        model_name: str = None,
        collection_prefix: str = "spider"
    ):
        """
        Initialize retriever

        Args:
            persist_dir: ChromaDB persist directory
            model_name: Embedding model name
            collection_prefix: Which dataset's collections to load, e.g.
                "spider" or "wikisql". Must match the prefix used when the
                collections were built (see ChromaDBHandler.setup_collections
                and WikiSQLChromaDBIntegration / SpiderChromaDBIntegration).
        """
        self.db_handler = ChromaDBHandler(persist_dir)
        self.embedding_gen = EmbeddingGenerator(model_name)
        self.collection_prefix = collection_prefix

        # Load collections
        self._load_collections()

    def _load_collections(self) -> bool:
        """
        Load existing collections

        Returns:
            True if collections loaded successfully
        """
        try:
            p = self.collection_prefix
            self.schema_collection = self.db_handler.client.get_collection(f"{p}_schemas")
            self.question_collection = self.db_handler.client.get_collection(f"{p}_questions")
            self.sql_collection = self.db_handler.client.get_collection(f"{p}_sql")

            # Check if collections have data
            schema_count = self.db_handler.get_collection_count(self.schema_collection)
            question_count = self.db_handler.get_collection_count(self.question_collection)
            sql_count = self.db_handler.get_collection_count(self.sql_collection)

            logger.info(f"Loaded '{p}' collections: Schemas={schema_count}, Questions={question_count}, SQL={sql_count}")

            if schema_count == 0 or question_count == 0:
                logger.warning("Collections are empty. Need to populate with data.")
                return False

            return True

        except Exception as e:
            logger.error(f"Failed to load collections (prefix='{self.collection_prefix}'): {e}")
            logger.info(
                "Run build_chromadb.py (for 'spider') or "
                "build_chromadb_wikisql.py (for 'wikisql') first to populate ChromaDB"
            )
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
        db_filter: Optional[str] = None,
        split_filter: Optional[str] = "train",
        exclude_exact_match: bool = True
    ) -> Dict[str, Any]:
        """
        Find similar questions from the dataset

        Args:
            query: User question
            n_results: Number of results to return
            min_similarity: Minimum similarity threshold
            db_filter: Filter by database ID. Leave None for Spider (cross-domain:
                dev databases never appear in train, so filtering by db_id here
                always returns empty). Also leave None for WikiSQL (each
                question has its own unique one-off db_id, so filtering by it
                would never match anything either).
            split_filter: Restrict candidate pool to this split. Defaults to
                "train" — this collection also stores dev-split questions, so
                without this filter, retrieval could leak the dev question
                (and its gold SQL) back into its own prompt.
            exclude_exact_match: Extra safety net — drop any candidate whose
                question text exactly matches the query.

        Returns:
            Dictionary with results
        """
        if not self.question_collection:
            return {"error": "Question collection not available"}

        try:
            # Generate embedding
            query_embedding = self.embedding_gen.encode(query)

            # Build metadata filter
            conditions = []
            if split_filter:
                conditions.append({"split": split_filter})
            if db_filter:
                conditions.append({"db_id": db_filter})

            if not conditions:
                where = None
            elif len(conditions) == 1:
                where = conditions[0]
            else:
                where = {"$and": conditions}

            # Over-fetch so filtering (exact-match / min_similarity) doesn't
            # leave us with fewer than n_results usable examples.
            fetch_n = n_results * 3 if exclude_exact_match else n_results

            results = self.db_handler.query_collection(
                self.question_collection,
                query_embeddings=[query_embedding.tolist()],
                n_results=fetch_n,
                where=where
            )

            if not results.get('metadatas', [[]])[0]:
                return {"error": "No similar questions found"}

            similar_queries = []
            for metadata, distance in zip(
                results['metadatas'][0],
                results['distances'][0]
            ):
                similarity = self._distance_to_similarity(distance)
                if similarity < min_similarity:
                    continue

                candidate_question = metadata.get('question', '')
                if exclude_exact_match and candidate_question.strip() == query.strip():
                    continue

                similar_queries.append({
                    'question': candidate_question,
                    'sql_query': metadata.get('sql', ''),
                    'database': metadata.get('db_id', 'unknown'),
                    'similarity_score': round(similarity, 4),
                    'split': metadata.get('split', 'unknown')
                })

                if len(similar_queries) >= n_results:
                    break

            for i, item in enumerate(similar_queries, 1):
                item['rank'] = i

            if not similar_queries:
                return {"error": "No similar questions found after filtering"}

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
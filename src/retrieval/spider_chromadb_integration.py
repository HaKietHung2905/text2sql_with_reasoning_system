"""
Spider dataset integration with ChromaDB.
Handles loading Spider data and storing in vector database.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional

from src.data.spider_loader import SpiderDataLoader
from src.data.spider_downloader import SpiderDatasetDownloader
from src.retrieval.chromadb_handler import ChromaDBHandler
from utils.embedding_utils import EmbeddingGenerator
from utils.schema_utils import format_schema_for_embedding, format_question_with_schema
from utils.logging_utils import get_logger

logger = get_logger(__name__)


class SpiderChromaDBIntegration:
    """Integrate Spider dataset with ChromaDB for retrieval"""
    
    def __init__(
        self,
        data_dir: str = "./data/raw/spider",
        persist_dir: str = "./data/embeddings/chroma_db",
        model_name: str = None
    ):
        """
        Initialize Spider-ChromaDB integration
        
        Args:
            data_dir: Directory containing Spider dataset
            persist_dir: Directory for ChromaDB storage
            model_name: Embedding model name
        """
        self.data_dir = Path(data_dir)
        self.persist_dir = Path(persist_dir)
        
        # Initialize components
        self.loader = SpiderDataLoader(str(self.data_dir))
        self.db_handler = ChromaDBHandler(str(self.persist_dir))
        self.embedding_gen = EmbeddingGenerator(model_name)
        
        # Data storage
        self.train_data = None
        self.dev_data = None
        self.tables_data = None
    
    def load_dataset(self, download_if_missing: bool = True) -> bool:
        """
        Load Spider dataset
        
        Args:
            download_if_missing: Download if not found locally
            
        Returns:
            True if successful
        """
        logger.info("Loading Spider dataset...")
        
        # Try loading from local files
        self.train_data = self.loader.load_split("train")
        self.dev_data = self.loader.load_split("dev")
        self.tables_data = self.loader.load_tables()
        
        # Check if data loaded successfully
        if not self.train_data and not self.dev_data:
            if download_if_missing:
                logger.info("Dataset not found locally, downloading...")
                downloader = SpiderDatasetDownloader(str(self.data_dir.parent))
                
                if downloader.download_from_huggingface():
                    # Retry loading
                    self.train_data = self.loader.load_split("train")
                    self.dev_data = self.loader.load_split("dev")
                    self.tables_data = self.loader.load_tables()
                else:
                    logger.error("Failed to download dataset")
                    return False
            else:
                logger.error("Dataset not found and download disabled")
                return False
        
        # Use sample data if still nothing loaded
        if not self.train_data and not self.dev_data:
            logger.warning("Using sample data")
            self._load_sample_data()
        
        logger.info(f"Dataset loaded:")
        logger.info(f"  Train: {len(self.train_data)} examples")
        logger.info(f"  Dev: {len(self.dev_data)} examples")
        logger.info(f"  Tables: {len(self.tables_data)} schemas")
        
        return True
    
    def _load_sample_data(self):
        """Load sample data as fallback"""
        self.train_data = [
            {
                "db_id": "restaurant_1",
                "question": "How many restaurants are there?",
                "query": "SELECT count(*) FROM restaurant"
            },
            {
                "db_id": "restaurant_1",
                "question": "What are the names of all restaurants?",
                "query": "SELECT name FROM restaurant"
            }
        ]
        
        self.dev_data = self.train_data[:1]
        
        self.tables_data = [
            {
                "db_id": "restaurant_1",
                "table_names_original": ["restaurant"],
                "table_names": ["restaurant"],
                "column_names_original": [[-1, "*"], [0, "id"], [0, "name"]],
                "column_names": [[-1, "*"], [0, "id"], [0, "name"]],
                "column_types": ["text", "number", "text"],
                "foreign_keys": [],
                "primary_keys": [1]
            }
        ]
        
        logger.info("Loaded sample data")
    
    def setup_collections(self, reset: bool = True) -> None:
        """
        Setup ChromaDB collections
        
        Args:
            reset: Reset existing collections
        """
        logger.info("Setting up ChromaDB collections...")
        self.db_handler.setup_collections(reset=reset)
    
    def store_schemas(self) -> int:
        """
        Store database schemas in ChromaDB
        
        Returns:
            Number of schemas stored
        """
        if not self.tables_data:
            logger.warning("No schema data to store")
            return 0
        
        logger.info("Storing database schemas...")
        
        documents = []
        embeddings = []
        ids = []
        metadatas = []
        
        for table_info in self.tables_data:
            schema_text = format_schema_for_embedding(table_info)
            embedding = self.embedding_gen.encode(schema_text)
            
            documents.append(schema_text)
            embeddings.append(embedding.tolist())
            ids.append(table_info['db_id'])
            metadatas.append({
                "db_id": table_info['db_id'],
                "num_tables": len(table_info.get('table_names', [])),
                "num_columns": len(table_info.get('column_names', [])),
                "type": "schema"
            })
        
        count = self.db_handler.add_batch(
            self.db_handler.schema_collection,
            documents,
            embeddings,
            ids,
            metadatas,
            batch_size=500
        )
        
        logger.info(f"Stored {count} database schemas")
        return count
    
    def store_questions(self, examples: List[Dict], split: str = "train") -> int:
        """
        Store questions in ChromaDB
        
        Args:
            examples: List of examples
            split: Dataset split name
            
        Returns:
            Number of questions stored
        """
        if not examples:
            logger.warning(f"No {split} data to store")
            return 0
        
        logger.info(f"Storing {split} questions...")
        
        documents = []
        embeddings = []
        ids = []
        metadatas = []
        
        for i, example in enumerate(examples):
            question_text = format_question_with_schema(
                example.get('question', ''),
                example.get('db_id', 'unknown')
            )
            embedding = self.embedding_gen.encode(question_text)
            
            documents.append(question_text)
            embeddings.append(embedding.tolist())
            ids.append(f"{split}_{i}")
            metadatas.append({
                "db_id": example.get('db_id', 'unknown'),
                "question": example.get('question', ''),
                "sql": example.get('query', ''),
                "split": split,
                "type": "question"
            })
        
        count = self.db_handler.add_batch(
            self.db_handler.question_collection,
            documents,
            embeddings,
            ids,
            metadatas,
            batch_size=1000
        )
        
        logger.info(f"Stored {count} questions from {split} split")
        return count
    
    def store_sql_queries(self, examples: List[Dict], split: str = "train") -> int:
        """
        Store SQL queries in ChromaDB
        
        Args:
            examples: List of examples
            split: Dataset split name
            
        Returns:
            Number of queries stored
        """
        if not examples:
            logger.warning(f"No {split} SQL data to store")
            return 0
        
        logger.info(f"Storing {split} SQL queries...")
        
        documents = []
        embeddings = []
        ids = []
        metadatas = []
        
        for i, example in enumerate(examples):
            sql_query = example.get('query', '')
            if not sql_query:
                continue
            
            embedding = self.embedding_gen.encode(sql_query)
            
            documents.append(sql_query)
            embeddings.append(embedding.tolist())
            ids.append(f"{split}_sql_{i}")
            metadatas.append({
                "db_id": example.get('db_id', 'unknown'),
                "question": example.get('question', ''),
                "sql": sql_query,
                "split": split,
                "type": "sql"
            })
        
        count = self.db_handler.add_batch(
            self.db_handler.sql_collection,
            documents,
            embeddings,
            ids,
            metadatas,
            batch_size=1000
        )
        
        logger.info(f"Stored {count} SQL queries from {split} split")
        return count
    
    def store_all_data(self) -> Dict[str, int]:
        """
        Store all data in ChromaDB
        
        Returns:
            Dictionary with counts of stored items
        """
        counts = {}
        
        counts['schemas'] = self.store_schemas()
        counts['train_questions'] = self.store_questions(self.train_data, "train")
        counts['dev_questions'] = self.store_questions(self.dev_data, "dev")
        counts['train_sql'] = self.store_sql_queries(self.train_data, "train")
        counts['dev_sql'] = self.store_sql_queries(self.dev_data, "dev")
        
        return counts
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get integration statistics
        
        Returns:
            Statistics dictionary
        """
        stats = self.db_handler.get_statistics()
        
        stats['train_examples'] = len(self.train_data) if self.train_data else 0
        stats['dev_examples'] = len(self.dev_data) if self.dev_data else 0
        stats['table_schemas'] = len(self.tables_data) if self.tables_data else 0
        
        return stats
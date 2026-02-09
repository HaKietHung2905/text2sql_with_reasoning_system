"""Spider dataset download and setup"""

from pathlib import Path
from typing import Dict, List, Optional
import sqlite3

from utils.file_io import (
    read_json, 
    write_json, 
    download_file, 
    extract_zip,
    ensure_directory
)
from utils.schema_utils import get_sqlite_schema
from utils.logging_utils import get_logger
from src.data.spider_loader import SpiderSchemaExtractor

logger = get_logger(__name__)


class SpiderDatasetDownloader:
    """Download and setup Spider dataset"""
    
    # Download sources
    HUGGINGFACE_SOURCE = "huggingface"
    FULL_DATASET_SOURCES = [
        "https://drive.google.com/uc?export=download&id=1TqleXec_OykOYFREKKtschzY29dUcVAQ",
        "https://yale-lily.github.io/spider/spider.zip"
    ]
    
    def __init__(self, data_dir: str = "./data/raw"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.spider_dir = self.data_dir / "spider"
        self.schema_extractor = SpiderSchemaExtractor()
    
    def download_from_huggingface(self) -> bool:
        """
        Download Spider dataset from HuggingFace
        
        Returns:
            True if successful, False otherwise
        """
        try:
            from datasets import load_dataset
            
            logger.info("📦 Downloading Spider dataset from HuggingFace...")
            dataset = load_dataset("xlangai/spider")
            
            # Create directory
            self.spider_dir.mkdir(parents=True, exist_ok=True)
            
            # Collect data for schema extraction
            all_sql_queries = []
            all_db_ids = []
            
            # Save train data
            train_data = self._format_huggingface_data(dataset['train'])
            write_json(train_data, str(self.spider_dir / 'train_spider.json'))
            
            # Collect for schema extraction
            all_sql_queries.extend([item['query'] for item in train_data])
            all_db_ids.extend([item['db_id'] for item in train_data])
            
            # Save dev data
            dev_data = self._format_huggingface_data(dataset['validation'])
            write_json(dev_data, str(self.spider_dir / 'dev.json'))
            
            # Collect for schema extraction
            all_sql_queries.extend([item['query'] for item in dev_data])
            all_db_ids.extend([item['db_id'] for item in dev_data])
            
            # Extract and save schema information
            logger.info("🔍 Extracting table information from SQL queries...")
            schemas = self.schema_extractor.extract_schema_from_queries(
                all_sql_queries, 
                all_db_ids
            )
            
            formatted_schemas = self.schema_extractor.format_for_spider(schemas)
            write_json(formatted_schemas, str(self.spider_dir / 'tables.json'))
            
            logger.info(f"✅ Downloaded {len(train_data)} training examples")
            logger.info(f"✅ Downloaded {len(dev_data)} validation examples")
            logger.info(f"✅ Extracted schema for {len(schemas)} databases")
            logger.info(f"📁 Saved to: {self.spider_dir}")
            logger.warning("⚠️  Note: Database .sqlite files not included")
            logger.warning("⚠️  Schema extracted from SQL queries (basic only)")
            
            return True
            
        except ImportError:
            logger.error("❌ HuggingFace datasets library not installed")
            logger.info("💡 Install with: pip install datasets")
            return False
        except Exception as e:
            logger.error(f"❌ Error downloading from HuggingFace: {e}")
            return False
    
    def _format_huggingface_data(self, dataset) -> List[Dict]:
        """Format HuggingFace dataset to Spider format"""
        data = []
        for item in dataset:
            data.append({
                'db_id': item['db_id'],
                'question': item['question'],
                'sql': item['query'],
                'query': item['query'],
                'query_toks': item['query_toks'],
                'query_toks_no_value': item['query_toks_no_value'],
                'question_toks': item['question_toks']
            })
        return data
    
    def download_full_dataset(self) -> bool:
        """
        Download full Spider dataset with SQLite files
        
        Returns:
            True if successful, False otherwise
        """
        logger.info("🕷️ Attempting to download full Spider dataset...")
        
        for url in self.FULL_DATASET_SOURCES:
            try:
                logger.info(f"Trying {url}...")
                
                zip_path = self.data_dir / "spider_full.zip"
                
                if download_file(url, zip_path):
                    if extract_zip(zip_path, self.data_dir):
                        zip_path.unlink()  # Remove zip file
                        logger.info("✅ Full Spider dataset downloaded successfully!")
                        return True
                    
            except Exception as e:
                logger.warning(f"Failed with {url}: {e}")
                continue
        
        logger.warning("❌ Could not download full dataset")
        return False
    
    def download_dataset(self, prefer_full: bool = True) -> bool:
        """
        Download Spider dataset (tries full first, fallback to HuggingFace)
        
        Args:
            prefer_full: If True, try to download full dataset first
            
        Returns:
            True if successful, False otherwise
        """
        if prefer_full:
            if self.download_full_dataset():
                return True
            logger.info("Falling back to HuggingFace version...")
        
        return self.download_from_huggingface()
    
    def verify_dataset(self) -> Dict:
        """
        Verify downloaded dataset and return statistics
        
        Returns:
            Dictionary with dataset statistics
        """
        if not self.spider_dir.exists():
            logger.error("Spider dataset not found")
            return {}
        
        stats = {
            "train_exists": (self.spider_dir / "train_spider.json").exists(),
            "dev_exists": (self.spider_dir / "dev.json").exists(),
            "tables_exists": (self.spider_dir / "tables.json").exists(),
            "database_dir_exists": (self.spider_dir / "database").exists()
        }
        
        if stats["train_exists"]:
            train_data = read_json(str(self.spider_dir / "train_spider.json"))
            stats["train_count"] = len(train_data)
        
        if stats["dev_exists"]:
            dev_data = read_json(str(self.spider_dir / "dev.json"))
            stats["dev_count"] = len(dev_data)
        
        if stats["tables_exists"]:
            tables_data = read_json(str(self.spider_dir / "tables.json"))
            stats["tables_count"] = len(tables_data)
        
        return stats


class SpiderDatabaseManager:
    """Manage Spider database connections and schema"""
    
    def __init__(self, data_dir: str = "./data/raw/spider"):
        self.data_dir = Path(data_dir)
        self.db_dir = self.data_dir / "database"
    
    def connect_to_database(self, db_name: str) -> Optional[sqlite3.Connection]:
        """
        Connect to a specific database
        
        Args:
            db_name: Database name
            
        Returns:
            SQLite connection or None if not found
        """
        db_path = self.db_dir / db_name / f"{db_name}.sqlite"
        
        if not db_path.exists():
            logger.warning(f"Database {db_name} not found at {db_path}")
            logger.info("💡 Physical database files need full Spider dataset")
            return None
        
        try:
            conn = sqlite3.connect(str(db_path))
            logger.info(f"Connected to database: {db_name}")
            return conn
        except Exception as e:
            logger.error(f"Error connecting to database {db_name}: {e}")
            return None
    
    def get_database_schema(self, db_name: str) -> Dict:
        """
        Get schema from physical database or fallback to extracted schema
        
        Args:
            db_name: Database name
            
        Returns:
            Schema dictionary
        """
        # Try physical database first
        db_path = self.db_dir / db_name / f"{db_name}.sqlite"
        
        if db_path.exists():
            from utils.schema_utils import get_sqlite_schema
            schema = get_sqlite_schema(db_path)
            if schema:
                logger.info(f"Retrieved schema from SQLite: {db_name}")
                return schema
        
        # Fallback to extracted schema
        from src.data.spider_loader import SpiderDataLoader
        loader = SpiderDataLoader(str(self.data_dir))
        schema_info = loader.get_database_schema(db_name)
        
        if schema_info:
            logger.info(f"Retrieved extracted schema: {db_name}")
            return schema_info
        
        logger.warning(f"No schema found for: {db_name}")
        return {}
    
    def list_available_databases(self) -> List[str]:
        """
        List all available databases
        
        Returns:
            List of database names
        """
        if not self.db_dir.exists():
            return []
        
        databases = [
            d.name for d in self.db_dir.iterdir() 
            if d.is_dir()
        ]
        
        return sorted(databases)
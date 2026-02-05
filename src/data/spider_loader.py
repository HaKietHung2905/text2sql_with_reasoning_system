"""Spider dataset loading and processing"""

from pathlib import Path
from typing import Dict, List, Optional, Set
import pandas as pd

from utils.file_io import read_json, write_json, ensure_directory
from utils.sql_utils import extract_tables_from_sql, extract_columns_from_select
from utils.logging_utils import get_logger

logger = get_logger(__name__)


class SpiderDataLoader:
    """Load and process Spider dataset files"""
    
    def __init__(self, data_dir: str = "./data/raw/spider"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def load_split(self, split: str = "train") -> List[Dict]:
        """
        Load a specific split of the dataset
        
        Args:
            split: Dataset split ('train', 'dev', 'test')
            
        Returns:
            List of examples
        """
        if split == "train":
            filepath = self.data_dir / "train_spider.json"
        elif split == "dev":
            filepath = self.data_dir / "dev.json"
        elif split == "test":
            filepath = self.data_dir / "test.json"
        else:
            raise ValueError(f"Invalid split: {split}")
        
        if not filepath.exists():
            logger.warning(f"File not found: {filepath}")
            return []
        
        data = read_json(str(filepath))
        logger.info(f"Loaded {len(data)} examples from {split} split")
        return data
    
    def load_tables(self) -> List[Dict]:
        """
        Load table schema information
        
        Returns:
            List of table schemas
        """
        filepath = self.data_dir / "tables.json"
        
        if not filepath.exists():
            logger.warning(f"Tables file not found: {filepath}")
            return []
        
        tables = read_json(str(filepath))
        logger.info(f"Loaded {len(tables)} table schemas")
        return tables
    
    def get_database_schema(self, db_id: str) -> Optional[Dict]:
        """
        Get schema for a specific database
        
        Args:
            db_id: Database identifier
            
        Returns:
            Schema dictionary or None if not found
        """
        tables = self.load_tables()
        
        for table_info in tables:
            if table_info.get("db_id") == db_id:
                return table_info
        
        logger.warning(f"Schema not found for database: {db_id}")
        return None
    
    def get_unique_databases(self) -> Set[str]:
        """
        Get set of unique database IDs in dataset
        
        Returns:
            Set of database IDs
        """
        db_ids = set()
        
        for split in ["train", "dev"]:
            data = self.load_split(split)
            db_ids.update(item["db_id"] for item in data)
        
        return db_ids
    
    def create_dataframe(self, split: str = "train") -> pd.DataFrame:
        """
        Create pandas DataFrame from dataset split
        
        Args:
            split: Dataset split name
            
        Returns:
            DataFrame with dataset
        """
        data = self.load_split(split)
        
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        logger.info(
            f"Created DataFrame with {len(df)} rows and {len(df.columns)} columns"
        )
        
        return df
    
    def get_samples(self, split: str = "train", n: int = 5) -> List[Dict]:
        """
        Get sample examples from dataset
        
        Args:
            split: Dataset split
            n: Number of samples
            
        Returns:
            List of sample examples
        """
        data = self.load_split(split)
        return data[:n] if data else []


class SpiderSchemaExtractor:
    """Extract schema information from SQL queries"""
    
    def __init__(self):
        pass
    
    def extract_schema_from_queries(
        self, 
        sql_queries: List[str], 
        db_ids: List[str]
    ) -> Dict[str, Dict]:
        """
        Extract table and column info from SQL queries
        
        Args:
            sql_queries: List of SQL queries
            db_ids: Corresponding database IDs
            
        Returns:
            Dictionary mapping db_id to schema information
        """
        schemas = {}
        
        for sql, db_id in zip(sql_queries, db_ids):
            if db_id not in schemas:
                schemas[db_id] = {
                    "db_id": db_id,
                    "table_names": set(),
                    "table_names_original": set(),
                    "column_names": set(),
                    "column_names_original": set()
                }
            
            # Extract tables
            tables = extract_tables_from_sql(sql)
            for table in tables:
                schemas[db_id]["table_names"].add(table.lower())
                schemas[db_id]["table_names_original"].add(table)
            
            # Extract columns
            columns = extract_columns_from_select(sql)
            for col in columns:
                schemas[db_id]["column_names"].add(col.lower())
                schemas[db_id]["column_names_original"].add(col)
        
        # Convert sets to lists for JSON serialization
        for db_id in schemas:
            for key in ["table_names", "table_names_original", 
                       "column_names", "column_names_original"]:
                schemas[db_id][key] = list(schemas[db_id][key])
        
        return schemas
    
    def format_for_spider(self, schemas: Dict[str, Dict]) -> List[Dict]:
        """
        Format extracted schemas to Spider's expected format
        
        Args:
            schemas: Dictionary of extracted schemas
            
        Returns:
            List of schemas in Spider format
        """
        formatted = []
        
        for db_id, info in schemas.items():
            # Create column_names in Spider format: [[table_idx, col_name]]
            column_names = [[0, "*"]]  # Spider always starts with this
            column_names.extend([
                [i, col] 
                for i, col in enumerate(info["column_names"], 1)
            ])
            
            column_names_original = [["*"]]
            column_names_original.extend([
                [col] 
                for col in info["column_names_original"]
            ])
            
            formatted.append({
                "db_id": db_id,
                "table_names": info["table_names"],
                "table_names_original": info["table_names_original"],
                "column_names": column_names,
                "column_names_original": column_names_original,
                "column_types": ["text"] * len(column_names),
                "foreign_keys": [],
                "primary_keys": []
            })
        
        return formatted
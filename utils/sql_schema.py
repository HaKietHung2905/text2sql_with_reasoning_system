"""
Schema utilities for SQL parsing.
Handles database schema representation and mapping.
"""

import json
import sqlite3
from typing import Dict, List
from pathlib import Path

from utils.logging_utils import get_logger

logger = get_logger(__name__)

class Schema:
    """
    Schema representation for SQL parsing.
    Maps tables and columns to unique identifiers.
    """
    
    def __init__(self, schema: Dict[str, List[str]]):
        """
        Initialize schema
        
        Args:
            schema: Dictionary mapping table names to column lists
        """
        self._schema = schema
        self._idMap = self._map(self._schema)
    
    @property
    def schema(self) -> Dict[str, List[str]]:
        """Get raw schema dictionary"""
        return self._schema
    
    @property
    def idMap(self) -> Dict[str, str]:
        """Get ID mapping dictionary"""
        return self._idMap
    
    def _map(self, schema: Dict[str, List[str]]) -> Dict[str, str]:
        """
        Create ID mapping for tables and columns
        
        Args:
            schema: Schema dictionary
            
        Returns:
            ID mapping dictionary
        """
        idMap = {'*': "__all__"}
        id_counter = 1
        
        # Map columns
        for table, columns in schema.items():
            for column in columns:
                key = f"{table.lower()}.{column.lower()}"
                idMap[key] = f"__{key}__"
                id_counter += 1
        
        # Map tables
        for table in schema:
            idMap[table.lower()] = f"__{table.lower()}__"
            id_counter += 1
        
        return idMap
    
    def get_tables_with_alias(self, toks: List[str]) -> Dict[str, str]:
        """
        Get tables with their aliases from tokens
        
        Args:
            toks: List of tokens
            
        Returns:
            Dictionary mapping aliases to table names
        """
        from utils.sql_parser import scan_alias
        
        tables = scan_alias(toks)
        
        # Add original table names
        for table in self.schema:
            if table not in tables:
                tables[table] = table
        
        return tables


def get_schema_from_sqlite(db_path: str) -> Dict[str, List[str]]:
    """
    Get database schema from SQLite file
    
    Args:
        db_path: Path to SQLite database
        
    Returns:
        Schema dictionary
    """
    schema = {}
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Fetch table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [str(table[0].lower()) for table in cursor.fetchall()]
        
        # Fetch table info
        for table in tables:
            cursor.execute(f"PRAGMA table_info({table})")
            schema[table] = [str(col[1].lower()) for col in cursor.fetchall()]
        
        conn.close()
        #logger.info(f"Loaded schema with {len(schema)} tables from {db_path}")
        
    except Exception as e:
        logger.error(f"Failed to load schema from {db_path}: {e}")
        raise
    
    return schema


def get_schema_from_json(fpath: str) -> Dict[str, List[str]]:
    """
    Get database schema from JSON file
    
    Args:
        fpath: Path to JSON file
        
    Returns:
        Schema dictionary
    """
    try:
        with open(fpath) as f:
            data = json.load(f)
        
        schema = {}
        for entry in data:
            table = str(entry['table'].lower())
            cols = [str(col['column_name'].lower()) for col in entry['col_data']]
            schema[table] = cols
        
        #logger.info(f"Loaded schema with {len(schema)} tables from {fpath}")
        return schema
        
    except Exception as e:
        logger.error(f"Failed to load schema from {fpath}: {e}")
        raise


def load_schema(source: str) -> Schema:
    """
    Load schema from SQLite database or JSON file
    
    Args:
        source: Path to database or JSON file
        
    Returns:
        Schema object
    """
    source_path = Path(source)
    
    if source_path.suffix == '.sqlite':
        schema_dict = get_schema_from_sqlite(source)
    elif source_path.suffix == '.json':
        schema_dict = get_schema_from_json(source)
    else:
        raise ValueError(f"Unsupported schema source: {source}")
    
    return Schema(schema_dict)
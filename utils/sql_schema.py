"""
Schema utilities for SQL parsing.
Handles database schema representation and mapping.
"""

import json
import sqlite3
from typing import Dict, List, Optional, Tuple, Any
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
        Create ID mapping for tables and columns.
        Registers both original and lowercase keys so the parser handles
        quoted (mixed-case) identifiers from WikiSQL gold SQL correctly.
        """
        idMap = {'*': "__all__"}

        for table, columns in schema.items():
            for column in columns:
                key = f"{table}.{column}"
                mapped = f"__{key.lower()}__"
                idMap[key] = mapped
                idMap[key.lower()] = mapped         
                idMap[column] = mapped
                idMap[column.lower()] = mapped

        for table in schema:
            mapped = f"__{table.lower()}__"
            idMap[table] = mapped
            idMap[table.lower()] = mapped

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

# utils/sql_schema.py

"""
Schema utilities for SQL parsing.
Handles database schema representation and mapping.
"""

import json
import sqlite3
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

from utils.logging_utils import get_logger

logger = get_logger(__name__)

# ... existing Schema class and functions ...

# ============================================================================
# ADD THESE NEW FUNCTIONS AT THE END OF THE FILE
# ============================================================================

def load_schema_for_db(db_id: str, db_dir: str) -> Dict:
    """
    Load schema from database
    
    Args:
        db_id: Database identifier
        db_dir: Directory containing databases
        
    Returns:
        Schema dictionary with enhanced metadata
    """
    db_path = Path(db_dir) / db_id / f"{db_id}.sqlite"
    
    if not db_path.exists():
        logger.warning(f"Database not found: {db_path}")
        return {}
    
    try:
        schema = get_schema_from_sqlite(str(db_path))
        return {
            'db_id': db_id,
            'tables': schema,
            'table_names': list(schema.keys()),
            'db_path': str(db_path)
        }
    except Exception as e:
        logger.error(f"Failed to load schema for {db_id}: {e}")
        return {}


def load_foreign_keys(db_id: str, db_dir: str) -> List[Dict]:
    """
    Load foreign key relationships from database
    
    Args:
        db_id: Database identifier
        db_dir: Directory containing databases
        
    Returns:
        List of foreign key relationships
    """
    db_path = Path(db_dir) / db_id / f"{db_id}.sqlite"
    
    if not db_path.exists():
        return []
    
    foreign_keys = []
    
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        
        # For each table, get foreign keys
        for table in tables:
            cursor.execute(f'PRAGMA foreign_key_list("{table}")')
            fks = cursor.fetchall()
            
            for fk in fks:
                # fk format: (id, seq, table, from, to, on_update, on_delete, match)
                foreign_keys.append({
                    'from_table': table,
                    'from_column': fk[3],
                    'to_table': fk[2],
                    'to_column': fk[4],
                    'constraint_name': f"fk_{table}_{fk[0]}"
                })
        
        conn.close()
        logger.debug(f"Loaded {len(foreign_keys)} foreign keys for {db_id}")
        
    except Exception as e:
        logger.error(f"Failed to load foreign keys for {db_id}: {e}")
    
    return foreign_keys


def load_sample_rows(db_id: str, db_dir: str, limit: int = 3) -> Dict[str, List[Dict]]:
    """
    Load sample rows from each table
    
    Args:
        db_id: Database identifier
        db_dir: Directory containing databases
        limit: Number of sample rows per table
        
    Returns:
        Dictionary mapping table names to sample rows
    """
    db_path = Path(db_dir) / db_id / f"{db_id}.sqlite"
    
    if not db_path.exists():
        return {}
    
    samples = {}
    
    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row  # Enable column access by name
        cursor = conn.cursor()
        
        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        
        # For each table, get sample rows
        for table in tables:
            try:
                cursor.execute(f'SELECT * FROM "{table}" LIMIT {limit}')
                rows = cursor.fetchall()
                
                # Convert to list of dictionaries
                samples[table] = [dict(row) for row in rows]
                
            except Exception as e:
                logger.warning(f"Could not load samples from {table}: {e}")
                samples[table] = []
        
        conn.close()
        logger.debug(f"Loaded samples for {len(samples)} tables in {db_id}")
        
    except Exception as e:
        logger.error(f"Failed to load sample data for {db_id}: {e}")
    
    return samples


def load_full_db_context(db_id: str, db_dir: str) -> Dict:
    """
    Load complete database context including schema, foreign keys, and samples
    
    Args:
        db_id: Database identifier
        db_dir: Directory containing databases
        
    Returns:
        Complete database context dictionary
    """
    context = {
        'db_id': db_id,
        'schema': load_schema_for_db(db_id, db_dir),
        'foreign_keys': load_foreign_keys(db_id, db_dir),
        'sample_data': load_sample_rows(db_id, db_dir)
    }
    
    # Add summary statistics
    context['stats'] = {
        'num_tables': len(context['schema'].get('tables', {})),
        'num_foreign_keys': len(context['foreign_keys']),
        'num_tables_with_samples': len(context['sample_data'])
    }
    
    logger.debug(f"Loaded full context for {db_id}: {context['stats']}")
    
    return context


def format_db_context_for_prompt(context: Dict) -> str:
    """
    Format database context for LLM prompt
    
    Args:
        context: Database context from load_full_db_context()
        
    Returns:
        Formatted string for prompt
    """
    lines = []
    
    # Database header
    lines.append(f"Database: {context['db_id']}")
    lines.append("=" * 60)
    
    # Schema
    schema = context.get('schema', {})
    tables = schema.get('tables', {})
    
    lines.append("\nSchema:")
    for table_name, columns in tables.items():
        lines.append(f"\n  Table: {table_name}")
        lines.append(f"  Columns: {', '.join(columns)}")
    
    # Foreign keys
    fks = context.get('foreign_keys', [])
    if fks:
        lines.append("\nForeign Keys:")
        for fk in fks:
            lines.append(f"  - {fk['from_table']}.{fk['from_column']} "
                        f"-> {fk['to_table']}.{fk['to_column']}")
    
    # Sample data (optional, can be verbose)
    samples = context.get('sample_data', {})
    if samples:
        lines.append("\nSample Data (first 3 rows):")
        for table_name, rows in samples.items():
            if rows:
                lines.append(f"\n  {table_name}:")
                for i, row in enumerate(rows[:3], 1):
                    lines.append(f"    Row {i}: {row}")
    
    return "\n".join(lines)
    
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
        tables = [str(row[0].lower()) for row in cursor.fetchall()]
        
        # Fetch table info
        for table in tables:
            cursor.execute(f'PRAGMA table_info("{table}")')
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
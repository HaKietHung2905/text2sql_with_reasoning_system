"""Schema processing utilities"""

import sqlite3
from typing import Dict, List, Optional, Any
from pathlib import Path
from typing import Dict, List, Any

def get_sqlite_schema(db_path: Path) -> Dict[str, List[Dict[str, Any]]]:
    """
    Extract schema from SQLite database
    
    Args:
        db_path: Path to SQLite database file
        
    Returns:
        Dictionary mapping table names to column information
    """
    schema = {}
    
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        
        # Get columns for each table
        for table in tables:
            cursor.execute(f'PRAGMA table_info("{table}");')
            columns = cursor.fetchall()
            
            schema[table] = [
                {
                    "name": col[1],
                    "type": col[2],
                    "nullable": not col[3],
                    "primary_key": bool(col[5])
                }
                for col in columns
            ]
        
        conn.close()
        return schema
        
    except Exception as e:
        print(f"Error extracting schema: {e}")
        return {}


def format_schema_for_spider(schema: Dict[str, List[Dict]]) -> Dict[str, Any]:
    """
    Format schema dictionary to Spider's expected format
    
    Args:
        schema: Schema dictionary from get_sqlite_schema
        
    Returns:
        Spider-format schema dictionary
    """
    table_names = list(schema.keys())
    column_names = [[-1, "*"]]  # Spider format: [table_idx, column_name]
    column_types = ["text"]
    primary_keys = []
    
    for table_idx, (table_name, columns) in enumerate(schema.items()):
        for col in columns:
            column_names.append([table_idx, col["name"]])
            column_types.append(col["type"].lower() if col["type"] else "text")
            
            if col.get("primary_key"):
                primary_keys.append(len(column_names) - 1)
    
    return {
        "table_names": [t.lower() for t in table_names],
        "table_names_original": table_names,
        "column_names": column_names,
        "column_types": column_types,
        "primary_keys": primary_keys,
        "foreign_keys": []  # Would need additional query to extract
    }

"""Schema processing utilities"""

def format_schema_for_prompt(schema: Dict) -> str:
    """
    Format schema dictionary for prompt inclusion
    
    Args:
        schema: Schema dictionary
        
    Returns:
        Formatted schema string
    """
    lines = []
    
    db_id = schema.get('db_id', 'unknown')
    lines.append(f"Database: {db_id}")
    
    table_names = schema.get('table_names_original', schema.get('table_names', []))
    column_names = schema.get('column_names_original', schema.get('column_names', []))
    column_types = schema.get('column_types', [])
    foreign_keys = schema.get('foreign_keys', [])
    primary_keys = schema.get('primary_keys', [])
    
    # Format tables and columns
    for i, table_name in enumerate(table_names):
        lines.append(f"\nTable: {table_name}")
        
        # Get columns for this table
        table_columns = []
        for j, col in enumerate(column_names):
            if isinstance(col, list) and len(col) >= 2:
                if col[0] == i:  # Column belongs to this table
                    col_type = column_types[j] if j < len(column_types) else 'unknown'
                    is_pk = j in primary_keys
                    table_columns.append((j, col[1], col_type, is_pk))
        
        for col_idx, col_name, col_type, is_pk in table_columns:
            pk_marker = " [PRIMARY KEY]" if is_pk else ""
            lines.append(f"  - {col_name} ({col_type}){pk_marker}")
    
    # Format foreign keys
    if foreign_keys:
        lines.append("\nForeign Keys:")
        for fk in foreign_keys:
            if len(fk) >= 2:
                try:
                    col1_idx, col2_idx = fk[0], fk[1]
                    if col1_idx < len(column_names) and col2_idx < len(column_names):
                        col1 = column_names[col1_idx]
                        col2 = column_names[col2_idx]
                        
                        if (isinstance(col1, list) and isinstance(col2, list) and
                            len(col1) >= 2 and len(col2) >= 2 and
                            col1[0] < len(table_names) and col2[0] < len(table_names)):
                            
                            table1 = table_names[col1[0]]
                            table2 = table_names[col2[0]]
                            lines.append(f"  - {table1}.{col1[1]} -> {table2}.{col2[1]}")
                except (IndexError, TypeError):
                    continue
    
    return "\n".join(lines)


def format_schema_for_embedding(schema: Dict) -> str:
    """
    Format schema for embedding generation
    Similar to format_schema_for_prompt but optimized for semantic search
    
    Args:
        schema: Schema dictionary
        
    Returns:
        Formatted schema string for embedding
    """
    return format_schema_for_prompt(schema)


def format_question_with_schema(question: str, db_id: str) -> str:
    """
    Format question with database context
    
    Args:
        question: Natural language question
        db_id: Database identifier
        
    Returns:
        Formatted question string
    """
    return f"Database: {db_id}\nQuestion: {question}"
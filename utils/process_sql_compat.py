"""
Compatibility layer for old process_sql.py imports.
Provides backward compatibility for code expecting process_sql functions.
"""

from utils.sql_schema import Schema, get_schema, get_schema_from_json
from src.data.sql_parser import parse_sql

# Alias for backward compatibility
def get_sql(schema: Schema, query: str):
    """
    Parse SQL query using schema (backward compatible)
    
    Args:
        schema: Schema object
        query: SQL query string
        
    Returns:
        Parsed SQL structure
    """
    return parse_sql(query, schema)


__all__ = [
    'Schema',
    'get_schema',
    'get_schema_from_json', 
    'get_sql',
]
"""SQL parsing and formatting utilities"""

import re
from typing import List, Set, Tuple


def extract_tables_from_sql(sql: str) -> Set[str]:
    """
    Extract table names from SQL query
    
    Args:
        sql: SQL query string
        
    Returns:
        Set of table names
    """
    tables = set()
    
    # Extract FROM clause tables
    from_matches = re.findall(
        r'\bFROM\s+([a-zA-Z_][a-zA-Z0-9_]*)', 
        sql, 
        re.IGNORECASE
    )
    
    # Extract JOIN clause tables
    join_matches = re.findall(
        r'\bJOIN\s+([a-zA-Z_][a-zA-Z0-9_]*)', 
        sql, 
        re.IGNORECASE
    )
    
    tables.update(from_matches)
    tables.update(join_matches)
    
    return tables


def extract_columns_from_select(sql: str) -> List[str]:
    """
    Extract column names from SELECT clause
    
    Args:
        sql: SQL query string
        
    Returns:
        List of column names
    """
    columns = []
    
    # Extract SELECT clause
    select_match = re.search(
        r'SELECT\s+(.*?)\s+FROM', 
        sql, 
        re.IGNORECASE | re.DOTALL
    )
    
    if select_match:
        select_part = select_match.group(1)
        
        # Split by comma and clean
        raw_columns = [col.strip() for col in select_part.split(',')]
        
        for col in raw_columns:
            # Handle table.column notation
            if '.' in col:
                col = col.split('.')[-1]
            
            # Remove AS aliases
            if ' AS ' in col.upper():
                col = col.split(' AS ')[0].strip()
            
            # Remove functions
            col = re.sub(r'\w+\((.*?)\)', r'\1', col)
            
            col = col.strip()
            if col and col != '*':
                columns.append(col)
    
    return columns


def normalize_sql(sql: str) -> str:
    """
    Normalize SQL query formatting
    
    Args:
        sql: SQL query string
        
    Returns:
        Normalized SQL
    """
    sql = sql.strip()
    sql = re.sub(r'\s+', ' ', sql)
    return sql
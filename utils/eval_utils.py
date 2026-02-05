"""
Evaluation Utilities - ENHANCED VERSION
Replace the content of utils/eval_utils.py with this
"""
import re
from typing import Optional, Tuple


def normalize_sql_for_evaluation(sql: Optional[str]) -> Optional[str]:
    """
    Normalize SQL query for fair comparison - ENHANCED VERSION
    
    Handles:
    - Newlines and whitespace collapse
    - Keyword case normalization (lowercase)
    - Aggregate function case (uppercase per Spider convention)
    - Spacing around punctuation
    - Trailing semicolons
    
    Args:
        sql: SQL query string
        
    Returns:
        Normalized SQL query
        
    Example:
        >>> sql = "SELECT\\n  name\\nFROM users"
        >>> normalize_sql_for_evaluation(sql)
        'select name from users'
    """
    if not sql:
        return sql
    
    # Step 1: Remove newlines and normalize whitespace
    sql = re.sub(r'\s+', ' ', sql.strip())
    
    # Step 2: Normalize keywords to lowercase
    keywords = [
        'SELECT', 'FROM', 'WHERE', 'GROUP', 'BY', 'HAVING', 
        'ORDER', 'LIMIT', 'JOIN', 'ON', 'AND', 'OR', 'IN',
        'NOT', 'LIKE', 'BETWEEN', 'DISTINCT', 'AS', 'UNION',
        'INTERSECT', 'EXCEPT', 'LEFT', 'RIGHT', 'INNER', 'OUTER',
        'CROSS', 'NATURAL', 'EXISTS', 'ALL', 'ANY', 'CASE', 'WHEN',
        'THEN', 'ELSE', 'END'
    ]
    for kw in keywords:
        sql = re.sub(rf'\b{kw}\b', kw.lower(), sql, flags=re.IGNORECASE)
    
    # Step 3: Keep aggregate functions uppercase (Spider convention)
    functions = ['COUNT', 'SUM', 'AVG', 'MIN', 'MAX']
    for fn in functions:
        sql = re.sub(rf'\b{fn}\b', fn, sql, flags=re.IGNORECASE)
    
    # Step 4: Normalize spacing around punctuation
    sql = re.sub(r'\s*,\s*', ' , ', sql)
    sql = re.sub(r'\s*\(\s*', ' ( ', sql)
    sql = re.sub(r'\s*\)\s*', ' ) ', sql)
    sql = re.sub(r'\s*=\s*', ' = ', sql)
    sql = re.sub(r'\s*<\s*', ' < ', sql)
    sql = re.sub(r'\s*>\s*', ' > ', sql)
    sql = re.sub(r'\s*!\s*=\s*', ' != ', sql)
    
    # Step 5: Remove trailing semicolon
    sql = sql.rstrip(';').strip()
    
    # Step 6: Final whitespace cleanup
    sql = re.sub(r'\s+', ' ', sql)
    
    return sql.strip()


def extract_db_name_from_question(question: str) -> Optional[str]:
    """
    Extract database name from question if formatted as 'question [db_name]'
    
    Args:
        question: Question string potentially containing [db_name]
        
    Returns:
        Database name or None
    """
    if not question:
        return None
        
    match = re.search(r'\[(.*?)\]$', question.strip())
    if match:
        return match.group(1)
    return None


def clean_sql_string(sql: str) -> str:
    """
    Clean SQL string for processing - removes markdown code blocks
    
    Args:
        sql: SQL string potentially with markdown
        
    Returns:
        Cleaned SQL string
    """
    if not sql:
        return ""
    
    # Remove markdown code blocks
    sql = re.sub(r'```sql\n?', '', sql)
    sql = re.sub(r'```\n?', '', sql)
    sql = re.sub(r'`', '', sql)
    
    return normalize_sql_for_evaluation(sql) or ""


def compare_sql_normalized(sql1: str, sql2: str) -> bool:
    """
    Compare two SQL queries after normalization
    
    Args:
        sql1: First SQL query
        sql2: Second SQL query
        
    Returns:
        True if queries match after normalization
    """
    norm1 = normalize_sql_for_evaluation(sql1)
    norm2 = normalize_sql_for_evaluation(sql2)
    return norm1 == norm2


__all__ = [
    'normalize_sql_for_evaluation', 
    'extract_db_name_from_question', 
    'clean_sql_string',
    'compare_sql_normalized'
]
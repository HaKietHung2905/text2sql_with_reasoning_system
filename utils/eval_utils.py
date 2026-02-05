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
    - JOIN syntax normalization (INNER/LEFT/RIGHT JOIN -> JOIN for Spider parser)
    - Column name case normalization
    
    Args:
        sql: SQL query string
        
    Returns:
        Normalized SQL query
    """
    if not sql:
        return sql
    
    # Step 0: Handle "SELECT 1" placeholder - return as-is to fail gracefully
    if sql.strip().upper() == "SELECT 1":
        return sql.strip().lower()
    
    # Step 1: Remove newlines and normalize whitespace
    sql = re.sub(r'\s+', ' ', sql.strip())
    
    # Step 2: Normalize JOIN syntax FIRST (before case changes)
    # Spider parser has very limited JOIN support - convert all to simple JOIN
    # CRITICAL: Spider parser doesn't support LEFT/RIGHT/FULL joins in many cases
    sql = re.sub(r'\bINNER\s+JOIN\b', 'JOIN', sql, flags=re.IGNORECASE)
    sql = re.sub(r'\bLEFT\s+OUTER\s+JOIN\b', 'JOIN', sql, flags=re.IGNORECASE)
    sql = re.sub(r'\bLEFT\s+JOIN\b', 'JOIN', sql, flags=re.IGNORECASE)  # Add this
    sql = re.sub(r'\bRIGHT\s+OUTER\s+JOIN\b', 'JOIN', sql, flags=re.IGNORECASE)
    sql = re.sub(r'\bRIGHT\s+JOIN\b', 'JOIN', sql, flags=re.IGNORECASE)  # Add this
    sql = re.sub(r'\bFULL\s+OUTER\s+JOIN\b', 'JOIN', sql, flags=re.IGNORECASE)
    sql = re.sub(r'\bFULL\s+JOIN\b', 'JOIN', sql, flags=re.IGNORECASE)  # Add this
    
    # Step 3: Normalize keywords to lowercase
    keywords = [
        'SELECT', 'FROM', 'WHERE', 'GROUP', 'BY', 'HAVING', 
        'ORDER', 'LIMIT', 'JOIN', 'ON', 'AND', 'OR', 'IN',
        'NOT', 'LIKE', 'BETWEEN', 'DISTINCT', 'AS', 'UNION',
        'INTERSECT', 'EXCEPT', 'LEFT', 'RIGHT', 'INNER', 'OUTER',
        'CROSS', 'NATURAL', 'EXISTS', 'ALL', 'ANY', 'CASE', 'WHEN',
        'THEN', 'ELSE', 'END', 'DESC', 'ASC', 'NULL', 'IS'
    ]
    for kw in keywords:
        sql = re.sub(rf'\b{kw}\b', kw.lower(), sql, flags=re.IGNORECASE)
    
    # Step 4: Keep aggregate functions uppercase (Spider convention)
    functions = ['COUNT', 'SUM', 'AVG', 'MIN', 'MAX']
    for fn in functions:
        sql = re.sub(rf'\b{fn}\b', fn, sql, flags=re.IGNORECASE)
    
    # Step 5: Normalize table/column aliases and identifiers to lowercase
    def lowercase_identifier(match):
        return match.group(0).lower()
    
    # Pattern: table.column (e.g., T1.fname, student.age)
    sql = re.sub(r'\b[a-zA-Z_][a-zA-Z0-9_]*\.[a-zA-Z_][a-zA-Z0-9_]*\b', 
                 lowercase_identifier, sql)
    
    # Step 6: Normalize spacing around punctuation
    sql = re.sub(r'\s*,\s*', ' , ', sql)
    sql = re.sub(r'\s*\(\s*', ' ( ', sql)
    sql = re.sub(r'\s*\)\s*', ' ) ', sql)
    sql = re.sub(r'\s*=\s*', ' = ', sql)
    sql = re.sub(r'\s*<\s*', ' < ', sql)
    sql = re.sub(r'\s*>\s*', ' > ', sql)
    sql = re.sub(r'\s*!\s*=\s*', ' != ', sql)
    
    # Step 7: Remove trailing semicolon
    sql = sql.rstrip(';').strip()
    
    # Step 8: Final whitespace cleanup
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
"""
Evaluation Utilities - ENHANCED VERSION
Replace the content of utils/eval_utils.py with this
"""
import re
from typing import Optional, Tuple


def normalize_sql_for_evaluation(sql: Optional[str]) -> Optional[str]:
    """
    Normalize SQL query for fair comparison.
    Handles Spider + WikiSQL quirks.
    """
    if not sql:
        return sql

    # Step 0: Handle placeholder
    if sql.strip().upper() == "SELECT 1":
        return sql.strip().lower()

    # Step 1: Collapse whitespace
    sql = re.sub(r'\s+', ' ', sql.strip())

    # Step 2: Normalize JOIN variants → simple JOIN
    sql = re.sub(r'\bINNER\s+JOIN\b',       'JOIN', sql, flags=re.IGNORECASE)
    sql = re.sub(r'\bLEFT\s+OUTER\s+JOIN\b','JOIN', sql, flags=re.IGNORECASE)
    sql = re.sub(r'\bLEFT\s+JOIN\b',        'JOIN', sql, flags=re.IGNORECASE)
    sql = re.sub(r'\bRIGHT\s+OUTER\s+JOIN\b','JOIN', sql, flags=re.IGNORECASE)
    sql = re.sub(r'\bRIGHT\s+JOIN\b',       'JOIN', sql, flags=re.IGNORECASE)
    sql = re.sub(r'\bFULL\s+OUTER\s+JOIN\b','JOIN', sql, flags=re.IGNORECASE)
    sql = re.sub(r'\bFULL\s+JOIN\b',        'JOIN', sql, flags=re.IGNORECASE)

    # Step 3: Lowercase all keywords
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

    # Step 4: Uppercase aggregate functions (Spider convention)
    for fn in ['COUNT', 'SUM', 'AVG', 'MIN', 'MAX']:
        sql = re.sub(rf'\b{fn}\b', fn, sql, flags=re.IGNORECASE)

    # Step 5: Lowercase table.column identifiers
    sql = re.sub(
        r'\b[a-zA-Z_][a-zA-Z0-9_]*\.[a-zA-Z_][a-zA-Z0-9_]*\b',
        lambda m: m.group(0).lower(), sql
    )

    # Step 6: Normalize spacing around punctuation
    sql = re.sub(r'\s*,\s*',  ' , ', sql)
    sql = re.sub(r'\s*\(\s*', ' ( ', sql)
    sql = re.sub(r'\s*\)\s*', ' ) ', sql)
    sql = re.sub(r'\s*=\s*',  ' = ', sql)
    sql = re.sub(r'\s*<\s*',  ' < ', sql)
    sql = re.sub(r'\s*>\s*',  ' > ', sql)
    sql = re.sub(r'\s*!=\s*', ' != ', sql)

    # Step 7: Remove trailing semicolon
    sql = sql.rstrip(';').strip()

    # ── WikiSQL-specific normalizations ──────────────────────────────────────

    # Step 8: Lowercase string literals so value casing doesn't affect exact match
    # e.g. 'Butler CC (KS)' → 'butler cc (ks)'
    def lowercase_string_literal(m):
        quote = m.group(1)
        value = m.group(2).lower()
        return f"{quote}{value}{quote}"

    sql = re.sub(r"(['\"])([^'\"]*)\1", lowercase_string_literal, sql)

    # Step 9: Normalize numeric string literals → bare integers
    sql = re.sub(r"= '(\d+)'", r"= \1", sql)
    sql = re.sub(r"= \"(\d+)\"", r"= \1", sql)

    # Step 10: Remove DISTINCT inside COUNT for WikiSQL exact match
    sql = re.sub(r'\bCOUNT\s*\(\s*distinct\s+', 'COUNT ( ', sql, flags=re.IGNORECASE)

    # Step 11: Final whitespace cleanup
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
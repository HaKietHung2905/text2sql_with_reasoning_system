"""
Evaluation Utilities - ENHANCED VERSION
Replace the content of utils/eval_utils.py with this
"""
import re
from typing import Optional, Tuple


def _repair_unbalanced_apostrophes(sql: str) -> str:
    """
    Fix SQL strings where apostrophes inside string literals were never
    escaped (e.g. 'St. John's'  →  'St. John''s').

    This occurs in WikiSQL gold SQL where values like "St. John's" were
    stored without SQL-standard escaping.  A simple count-based heuristic:
    scan for single-quoted regions; if we hit an apostrophe that would
    close the literal mid-word (i.e. the very next char is a letter or
    digit), treat it as an embedded apostrophe and double it.
    """
    result = []
    i = 0
    n = len(sql)

    while i < n:
        ch = sql[i]

        if ch == "'":
            # Opening quote — collect literal content
            result.append("'")
            i += 1
            while i < n:
                c = sql[i]
                if c == "'":
                    # Peek: is this closing the literal or an embedded apostrophe?
                    # Heuristic: if the next char is alphanumeric/space-then-alpha,
                    # it's an embedded apostrophe that needs escaping.
                    next_i = i + 1
                    if (next_i < n
                            and sql[next_i].isalpha()):
                        # Embedded apostrophe → escape as ''
                        result.append("''")
                        i += 1
                    else:
                        # Real closing quote
                        result.append("'")
                        i += 1
                        break
                else:
                    result.append(c)
                    i += 1
        else:
            result.append(ch)
            i += 1

    return "".join(result)


def _lowercase_sql_string_literals(sql: str) -> str:
    """
    Lowercase the content of every SQL string literal in *sql* without
    breaking SQL-standard escaped apostrophes ('' inside a literal).

    Also repairs unbalanced apostrophes from WikiSQL gold SQL (e.g.
    'St. John's' → 'St. John''s') before processing.
    """
    # Repair any unescaped apostrophes in string literals first.
    sql = _repair_unbalanced_apostrophes(sql)

    result = []
    i = 0
    n = len(sql)

    while i < n:
        ch = sql[i]

        if ch in ("'", '"'):
            quote = ch
            result.append(quote)
            i += 1
            literal_chars = []

            while i < n:
                c = sql[i]
                if c == quote:
                    # '' = escaped quote inside literal
                    if i + 1 < n and sql[i + 1] == quote:
                        literal_chars.append(quote)
                        literal_chars.append(quote)
                        i += 2
                    else:
                        break  # closing quote
                else:
                    literal_chars.append(c)
                    i += 1

            result.append("".join(literal_chars).lower())
            result.append(quote)
            i += 1  # skip closing quote
        else:
            result.append(ch)
            i += 1

    return "".join(result)

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
    sql = re.sub(r'\bINNER\s+JOIN\b',        'JOIN', sql, flags=re.IGNORECASE)
    sql = re.sub(r'\bLEFT\s+OUTER\s+JOIN\b', 'JOIN', sql, flags=re.IGNORECASE)
    sql = re.sub(r'\bLEFT\s+JOIN\b',         'JOIN', sql, flags=re.IGNORECASE)
    sql = re.sub(r'\bRIGHT\s+OUTER\s+JOIN\b','JOIN', sql, flags=re.IGNORECASE)
    sql = re.sub(r'\bRIGHT\s+JOIN\b',        'JOIN', sql, flags=re.IGNORECASE)
    sql = re.sub(r'\bFULL\s+OUTER\s+JOIN\b', 'JOIN', sql, flags=re.IGNORECASE)
    sql = re.sub(r'\bFULL\s+JOIN\b',         'JOIN', sql, flags=re.IGNORECASE)

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

    # Step 8: Lowercase string literals so value casing doesn't affect exact match.
    # Uses a character-by-character scanner that correctly handles SQL-escaped
    # apostrophes ('') inside single-quoted literals (e.g. 'st. john''s').
    # The old regex  r"(['\"])([^'\"]*)\1"  broke such literals by treating the
    # first '' as open+close delimiters, mangling the rest of the SQL.
    sql = _lowercase_sql_string_literals(sql)

    # Step 9: Normalize numeric string literals → bare integers
    # Use a scanner-safe approach: only replace = '<digits>' patterns
    sql = re.sub(r"= '(\d+)'", r"= \1", sql)
    sql = re.sub(r'= "(\d+)"',  r"= \1", sql)

    # Step 10: Remove DISTINCT inside COUNT for WikiSQL exact match
    sql = re.sub(r'\bCOUNT\s*\(\s*distinct\s+', 'COUNT ( ', sql, flags=re.IGNORECASE)

    # Step 11: Final whitespace cleanup
    sql = re.sub(r'\s+', ' ', sql)

    return sql.strip()

def extract_db_name_from_question(question: str) -> Optional[str]:
    """
    Extract database name from question if formatted as 'question [db_name]'
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
    """
    if not sql:
        return ""
    sql = re.sub(r'```sql\n?', '', sql)
    sql = re.sub(r'```\n?', '', sql)
    sql = re.sub(r'`', '', sql)
    return normalize_sql_for_evaluation(sql) or ""


def compare_sql_normalized(sql1: str, sql2: str) -> bool:
    """
    Compare two SQL queries after normalization
    """
    norm1 = normalize_sql_for_evaluation(sql1)
    norm2 = normalize_sql_for_evaluation(sql2)
    return norm1 == norm2


__all__ = [
    'normalize_sql_for_evaluation',
    'extract_db_name_from_question',
    'clean_sql_string',
    'compare_sql_normalized',
]
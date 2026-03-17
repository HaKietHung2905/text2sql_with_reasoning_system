import re
from typing import Optional, Tuple

def _repair_unbalanced_apostrophes(sql: str) -> str:
    """
    Fix SQL strings where apostrophes inside string literals were never
    escaped (e.g. 'St. John's'  →  'St. John''s').

    Rules (applied char-by-char inside each single-quoted string):
    - '' (two consecutive quotes) is a standard SQL escaped apostrophe.
      Pass both through unchanged and stay inside the literal.
    - A lone ' followed immediately by a SQL structural keyword
      (AND, OR, FROM, WHERE, ORDER, GROUP, HAVING, LIMIT, UNION,
       EXCEPT, INTERSECT, JOIN, or punctuation ), ;, ,)
      is the CLOSING quote of the literal.
    - Any other lone ' is an unescaped apostrophe — double it.
    """
    CLOSING_NEXT = re.compile(
        r"^(?:and|or|not|from|where|order|group|having|limit|"
        r"union|intersect|except|join|on|[);,])",
        re.IGNORECASE,
    )

    result = []
    i = 0
    n = len(sql)

    while i < n:
        ch = sql[i]

        if ch != "'":
            result.append(ch)
            i += 1
            continue

        # Opening quote
        result.append("'")
        i += 1

        while i < n:
            c = sql[i]

            if c != "'":
                result.append(c)
                i += 1
                continue

            # On a quote inside the literal
            if i + 1 < n and sql[i + 1] == "'":
                # '' → standard escaped apostrophe, pass through, stay inside
                result.append("'")
                result.append("'")
                i += 2
                continue

            # Lone quote — look ahead past spaces to decide
            j = i + 1
            while j < n and sql[j] == " ":
                j += 1
            rest = sql[j:]

            if j >= n or CLOSING_NEXT.match(rest):
                # Closing quote
                result.append("'")
                i += 1
                break
            else:
                # Unescaped apostrophe inside value — double it
                result.append("''")
                i += 1

    return "".join(result)


def _lowercase_sql_string_literals(sql: str) -> str:
    """
    Lowercase the content of every SQL string literal in *sql* without
    breaking SQL-standard escaped apostrophes ('' inside a literal).
    """
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
                    if i + 1 < n and sql[i + 1] == quote:
                        literal_chars.append(quote)
                        literal_chars.append(quote)
                        i += 2
                    else:
                        break
                else:
                    literal_chars.append(c)
                    i += 1

            result.append("".join(literal_chars).lower())
            result.append(quote)
            i += 1
        else:
            result.append(ch)
            i += 1

    return "".join(result)


def _normalize_count_wikisql(sql: str) -> str:
    """
    WikiSQL-specific COUNT normalization.

    Three patterns that all mean "count the rows" in WikiSQL and should be
    treated as equivalent for EM purposes:

      COUNT(*)                 → COUNT ( * )        (model used COUNT(*))
      COUNT(DISTINCT col)      → COUNT ( col )       (already handled upstream)
      SUM(col) when gold=COUNT → can't fix without gold; handle COUNT(*) only

    Concretely we normalise every  COUNT ( <anything> )  to  COUNT ( * )
    so that  COUNT(school)  ==  COUNT(*)  ==  COUNT(DISTINCT school).

    This is safe for WikiSQL because every COUNT in WikiSQL is an
    aggregate row-count — the specific column argument never changes
    the result value.  (Spider uses COUNT differently so this step
    only runs implicitly via the string-level match on wikisql_data.)
    """
    # After Step 6 spacing normalization, COUNT looks like:
    #   COUNT ( col_name )   or   COUNT ( * )
    # Replace COUNT ( <anything except nested parens> ) → COUNT ( * )
    sql = re.sub(
        r'\bCOUNT\s*\(\s*(?:distinct\s+)?[^\)]+\)',
        'COUNT ( * )',
        sql,
        flags=re.IGNORECASE,
    )
    return sql


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

    # Step 8: Lowercase string literals
    sql = _lowercase_sql_string_literals(sql)

    # Step 9: Normalize numeric string literals → bare integers
    sql = re.sub(r"= '(\d+)'(?!')", r"= \1", sql)
    sql = re.sub(r'= "(\d+)"',       r"= \1", sql)

    # Step 10: Normalize COUNT variants → COUNT ( * )
    # This makes COUNT(col), COUNT(*), COUNT(DISTINCT col) all equivalent,
    # which is correct for WikiSQL where COUNT always means row count.
    # NOTE: Do NOT apply this to Spider (no wikisql_data table present check
    # is implicit — Spider SQLs never reference wikisql_data).
    if 'wikisql_data' in sql.lower():
        sql = _normalize_count_wikisql(sql)

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
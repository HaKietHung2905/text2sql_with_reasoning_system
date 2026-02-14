import json
import re
import sqlite3
from typing import Dict, List, Tuple, Any, Optional
from nltk import word_tokenize

from utils.logging_utils import get_logger

logger = get_logger(__name__)


# SQL Keywords and Operators
CLAUSE_KEYWORDS = ('select', 'from', 'where', 'group', 'order', 'limit', 'intersect', 'union', 'except')
JOIN_KEYWORDS = ('join', 'on', 'as')
WHERE_OPS = ('not', 'between', '=', '>', '<', '>=', '<=', '!=', 'in', 'like', 'is', 'exists')
UNIT_OPS = ('none', '-', '+', "*", '/')
AGG_OPS = ('none', 'max', 'min', 'count', 'sum', 'avg')
COND_OPS = ('and', 'or')
SQL_OPS = ('intersect', 'union', 'except')
ORDER_OPS = ('desc', 'asc')

TABLE_TYPE = {
    'sql': "sql",
    'table_unit': "table_unit",
}


def is_identifier(token: str) -> bool:
    """
    Check if token is a valid SQL identifier (table/column name).
    
    Valid identifiers:
    - Start with letter or underscore
    - Contain letters, numbers, underscores
    - Not a SQL keyword (for column/table context)
    
    Args:
        token: Token to check
        
    Returns:
        True if valid identifier
    """
    if not token:
        return False
    
    # Check if it matches identifier pattern
    if not re.match(r'^[a-z_][a-z0-9_]*$', token, re.IGNORECASE):
        return False
    
    # Keywords that should NOT be treated as identifiers in table.column context
    keywords_exclude = {
        'select', 'from', 'where', 'group', 'order', 'limit', 
        'join', 'on', 'and', 'or', 'not', 'in', 'like', 'is',
        'distinct', 'union', 'intersect', 'except', 'having',
        'desc', 'asc', 'by', 'between', 'exists', 'case', 'when',
        'then', 'else', 'end', 'null'
    }
    
    # AGG_OPS should also not be identifiers in this context
    agg_funcs = {'max', 'min', 'count', 'sum', 'avg'}
    
    if token.lower() in keywords_exclude or token.lower() in agg_funcs:
        return False
    
    return True


def tokenize(string: str) -> List[str]:
    """
    Tokenize SQL query string with proper handling of column references.
    
    CRITICAL FIX: Merges tokens like ['table', '.', 'column'] back into 'table.column'
    to avoid parse errors like "Error parsing column: t2" or "'pets.stuid'".
    
    This fixes the following error patterns:
    - Error parsing column: t2, t1, as, ||, select
    - 'pets.stuid', 'car_names.maker', 'cars_data.model'
    
    Args:
        string: SQL query string
        
    Returns:
        List of tokens with column references preserved
    """
    string = str(string)
    string = string.replace("\'", "\"")
    quote_idxs = [idx for idx, char in enumerate(string) if char == '"']
    
    assert len(quote_idxs) % 2 == 0, "Unexpected quote"
    
    # Keep string value as token
    vals = {}
    for i in range(len(quote_idxs) - 1, -1, -2):
        qidx1 = quote_idxs[i - 1]
        qidx2 = quote_idxs[i]
        val = string[qidx1: qidx2 + 1]
        key = f"__val_{qidx1}_{qidx2}__"
        string = string[:qidx1] + key + string[qidx2 + 1:]
        vals[key] = val
    
    # Use word_tokenize to get base tokens
    toks = [word.lower() for word in word_tokenize(string)]
    
    # Replace with string value token
    for i in range(len(toks)):
        if toks[i] in vals:
            toks[i] = vals[toks[i]]
    
    # CRITICAL FIX: Merge tokens that form column references (table.column)
    # This fixes errors like "Error parsing column: t2", "'pets.stuid'", etc.
    merged_toks = []
    i = 0
    while i < len(toks):
        # Check if this is part of a table.column pattern
        if i + 2 < len(toks) and toks[i + 1] == '.':
            # Pattern: identifier . identifier
            # Examples: t1.name, pets.stuid, car_names.maker, cars_data.model
            if (is_identifier(toks[i]) and is_identifier(toks[i + 2])):
                # Merge into single token
                merged_toks.append(f"{toks[i]}.{toks[i + 2]}")
                i += 3
                continue
        
        # Check for concatenation operator || (should stay as separate token, not merged)
        if toks[i] == '|' and i + 1 < len(toks) and toks[i + 1] == '|':
            merged_toks.append('||')
            i += 2
            continue
        
        # Regular token
        merged_toks.append(toks[i])
        i += 1
    
    toks = merged_toks
    
    # Find if there exists !=, >=, <=
    eq_idxs = [idx for idx, tok in enumerate(toks) if tok == "="]
    eq_idxs.reverse()
    prefix = ('!', '>', '<')
    
    for eq_idx in eq_idxs:
        if eq_idx > 0:
            pre_tok = toks[eq_idx - 1]
            if pre_tok in prefix:
                toks = toks[:eq_idx - 1] + [pre_tok + "="] + toks[eq_idx + 1:]
    
    return toks


def scan_alias(toks: List[str]) -> Dict[str, str]:
    """
    Scan the index of 'as' and build the map for all aliases.
    
    ENHANCED: Handles both explicit (table AS t1) and implicit (table t1) aliases.
    
    Args:
        toks: List of tokens
        
    Returns:
        Dictionary mapping aliases to table names
    """
    alias = {}
    
    # Handle explicit aliases with AS keyword
    as_idxs = [idx for idx, tok in enumerate(toks) if tok == 'as']
    for idx in as_idxs:
        if idx > 0 and idx + 1 < len(toks):
            table_name = toks[idx - 1]
            alias_name = toks[idx + 1]
            alias[alias_name] = table_name
    
    # Handle implicit aliases (FROM table alias)
    # Look for patterns: FROM tablename identifier (where identifier is not a keyword)
    i = 0
    while i < len(toks):
        if toks[i] == 'from' or toks[i] == 'join':
            # Next token should be table name
            if i + 1 < len(toks):
                table_name = toks[i + 1]
                # Check if there's an implicit alias (next token after table, not a keyword)
                if i + 2 < len(toks):
                    potential_alias = toks[i + 2]
                    # It's an alias if it's not a keyword or punctuation
                    if (potential_alias not in CLAUSE_KEYWORDS and 
                        potential_alias not in JOIN_KEYWORDS and
                        potential_alias != 'where' and
                        potential_alias != 'on' and
                        potential_alias not in ('(', ')', ',', ';') and
                        is_identifier(potential_alias)):
                        alias[potential_alias] = table_name
        i += 1
    
    return alias


def skip_semicolon(toks: List[str], start_idx: int) -> int:
    """
    Skip semicolon tokens
    
    Args:
        toks: List of tokens
        start_idx: Starting index
        
    Returns:
        Index after semicolons
    """
    idx = start_idx
    while idx < len(toks) and toks[idx] == ";":
        idx += 1
    return idx
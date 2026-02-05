"""
SQL parsing utilities for Spider dataset.
Parses SQL queries into structured representations.

Based on Spider dataset parsing logic.
"""

import json
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


def tokenize(string: str) -> List[str]:
    """
    Tokenize SQL query string
    
    Args:
        string: SQL query string
        
    Returns:
        List of tokens
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
    
    toks = [word.lower() for word in word_tokenize(string)]
    
    # Replace with string value token
    for i in range(len(toks)):
        if toks[i] in vals:
            toks[i] = vals[toks[i]]
    
    # Find if there exists !=, >=, <=
    eq_idxs = [idx for idx, tok in enumerate(toks) if tok == "="]
    eq_idxs.reverse()
    prefix = ('!', '>', '<')
    
    for eq_idx in eq_idxs:
        pre_tok = toks[eq_idx - 1]
        if pre_tok in prefix:
            toks = toks[:eq_idx - 1] + [pre_tok + "="] + toks[eq_idx + 1:]
    
    return toks


def scan_alias(toks: List[str]) -> Dict[str, str]:
    """
    Scan the index of 'as' and build the map for all aliases
    
    Args:
        toks: List of tokens
        
    Returns:
        Dictionary mapping aliases to table names
    """
    as_idxs = [idx for idx, tok in enumerate(toks) if tok == 'as']
    alias = {}
    
    for idx in as_idxs:
        alias[toks[idx + 1]] = toks[idx - 1]
    
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
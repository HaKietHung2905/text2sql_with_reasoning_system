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

    CRITICAL FIX: Merges tokens like ['table', '.', 'column'] back into
    'table.column' to avoid parse errors like "Error parsing column: t2".

    Also handles apostrophes inside string literals (e.g. "St. John's")
    that would otherwise cause an "Unexpected quote" assertion error.
    """
    string = str(string)

    # ------------------------------------------------------------------
    # STEP 1: Normalize quotes.
    # The rest of this function works entirely with double-quote delimiters.
    # We need to convert single-quoted SQL strings to double-quoted ones,
    # but we must preserve apostrophes *inside* string literals first.
    #
    # Strategy: scan for single-quoted regions manually, then escape any
    # apostrophe that sits inside a quoted value.
    # ------------------------------------------------------------------

    # Replace SQL-standard escaped apostrophes '' and \' with a safe placeholder
    # BEFORE we start scanning for quote pairs.
    APOS = "\x00APOS\x00"
    string = string.replace("''", APOS)   # SQL standard:  'St. John''s' → 'St. John\x00APOS\x00s'
    string = string.replace("\\'", APOS)  # C-style escape: 'O\'Brien'   → 'O\x00APOS\x00Brien'

    # Now every remaining single-quote is a proper open/close delimiter.
    # Convert them all to double-quotes for uniform processing below.
    string = string.replace("'", '"')

    quote_idxs = [idx for idx, char in enumerate(string) if char == '"']

    if len(quote_idxs) % 2 != 0:
        # Graceful recovery: an odd number of double-quotes means bad input
        # (e.g. the model generated unbalanced quotes).  Remove the last
        # unpaired quote so parsing can continue rather than crashing.
        logger.warning(
            f"Unexpected number of quotes in SQL: {string!r}. "
            "Attempting recovery by removing the unpaired quote."
        )
        # Pair greedily from the left; the leftover is the unpaired one.
        paired: set = set()
        for i in range(0, len(quote_idxs) - 1, 2):
            paired.add(quote_idxs[i])
            paired.add(quote_idxs[i + 1])
        unpaired = [idx for idx in quote_idxs if idx not in paired]
        s_list = list(string)
        for idx in sorted(unpaired, reverse=True):
            s_list[idx] = " "
        string = "".join(s_list)
        quote_idxs = [idx for idx, char in enumerate(string) if char == '"']

    # ------------------------------------------------------------------
    # STEP 2: Extract quoted string literals and replace them with stable
    # placeholder tokens BEFORE word_tokenize runs.
    #
    # word_tokenize would mangle things like  "St. John's"  or long values
    # with punctuation.  We replace each "..." region with a simple token
    # __STR_N__ that tokenize won't split, then restore afterwards.
    # ------------------------------------------------------------------
    vals: dict = {}
    # Process right-to-left so replacements don't shift earlier indices.
    str_idx = 0
    for i in range(len(quote_idxs) - 1, -1, -2):
        qidx1 = quote_idxs[i - 1]
        qidx2 = quote_idxs[i]
        # The raw value still has the APOS placeholder — restore it now.
        raw_val = string[qidx1: qidx2 + 1].replace(APOS, "'")
        placeholder = f"__STR{str_idx}__"
        str_idx += 1
        string = string[:qidx1] + placeholder + string[qidx2 + 1:]
        vals[placeholder.lower()] = raw_val  # key in lowercase to match word_tokenize output

    # ------------------------------------------------------------------
    # STEP 3: Tokenize the placeholder-substituted string.
    # ------------------------------------------------------------------
    toks = [word.lower() for word in word_tokenize(string)]

    # Restore string literal tokens.
    toks = [vals.get(tok, tok) for tok in toks]

    # ------------------------------------------------------------------
    # STEP 4: Merge  table . column  patterns into single tokens.
    # Fixes "Error parsing column: t2", "'pets.stuid'", etc.
    # ------------------------------------------------------------------
    merged_toks = []
    i = 0
    while i < len(toks):
        # table.column pattern
        if (i + 2 < len(toks)
                and toks[i + 1] == '.'
                and is_identifier(toks[i])
                and is_identifier(toks[i + 2])):
            merged_toks.append(f"{toks[i]}.{toks[i + 2]}")
            i += 3
            continue

        # Concatenation operator ||
        if toks[i] == '|' and i + 1 < len(toks) and toks[i + 1] == '|':
            merged_toks.append('||')
            i += 2
            continue

        merged_toks.append(toks[i])
        i += 1

    toks = merged_toks

    # ------------------------------------------------------------------
    # STEP 5: Merge  !=  >=  <=  into single tokens.
    # ------------------------------------------------------------------
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
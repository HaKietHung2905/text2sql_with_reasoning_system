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
    """Check if token is a valid SQL identifier (table/column name)."""
    if not token:
        return False

    if not re.match(r'^[a-z_][a-z0-9_]*$', token, re.IGNORECASE):
        return False

    keywords_exclude = {
        'select', 'from', 'where', 'group', 'order', 'limit',
        'join', 'on', 'and', 'or', 'not', 'in', 'like', 'is',
        'distinct', 'union', 'intersect', 'except', 'having',
        'desc', 'asc', 'by', 'between', 'exists', 'case', 'when',
        'then', 'else', 'end', 'null'
    }
    agg_funcs = {'max', 'min', 'count', 'sum', 'avg'}

    if token.lower() in keywords_exclude or token.lower() in agg_funcs:
        return False

    return True


def tokenize(string: str) -> List[str]:
    """
    Tokenize SQL query string with proper handling of column references.

    Key fixes vs original:
    - APOS placeholder is added to the __STR_N__ restoration map so
      word_tokenize lowercasing can never leave __escapedapos__ as a
      raw token that hits _parse_col and raises ValueError.
    - WikiSQL gold SQL uses outer double-apostrophes around titles like
      ''Don''t Stop Believin'' — these are now stripped to a clean
      quoted string before further processing.
    """
    string = str(string)

    # ------------------------------------------------------------------
    # STEP 0: Strip WikiSQL-style double-apostrophe wrappers.
    #
    # WikiSQL gold SQL sometimes wraps string values in doubled quotes:
    #   WHERE title = ''Don''t Stop Believin''
    # After normalize_sql_for_evaluation these become:
    #   where title = ''don''t stop believin''
    #
    # Convert:  ''...''  →  '...'  (collapsing the outer double-apos pairs)
    # This must happen BEFORE any other quote processing.
    # ------------------------------------------------------------------
    # Replace leading/trailing doubled single-quotes around a value:
    # ''value''  →  'value'
    # We do this with a regex that matches = '' ... '' patterns.
    string = re.sub(
        r"= ''((?:[^']|'')*?)''",
        lambda m: "= '" + m.group(1).replace("''", "\x00INNERAPOS\x00") + "'",
        string
    )
    # Restore any inner '' that were inside the value as SQL-escaped apos
    string = string.replace("\x00INNERAPOS\x00", "''")

    # ------------------------------------------------------------------
    # STEP 1: Pre-escape apostrophes inside double-quoted regions.
    # ------------------------------------------------------------------
    # Use a unique placeholder that will NEVER appear as a real token.
    # Must be something word_tokenize cannot split AND we can map back.
    APOS_PLACEHOLDER = "__STR_APOS__"

    result = []
    in_dquote = False
    i = 0
    while i < len(string):
        ch = string[i]
        if ch == '"':
            in_dquote = not in_dquote
            result.append(ch)
        elif ch == "'" and in_dquote:
            result.append(APOS_PLACEHOLDER)
        else:
            result.append(ch)
        i += 1
    string = "".join(result)

    # ------------------------------------------------------------------
    # STEP 2: Normalize single-quoted SQL strings → double-quoted.
    #
    # Replace SQL-standard '' and C-style \' with APOS_PLACEHOLDER first
    # so every remaining ' is a genuine open/close delimiter.
    # ------------------------------------------------------------------
    string = string.replace("''", APOS_PLACEHOLDER)
    string = string.replace("\\'", APOS_PLACEHOLDER)
    string = string.replace("'", '"')

    quote_idxs = [idx for idx, char in enumerate(string) if char == '"']

    if len(quote_idxs) % 2 != 0:
        logger.warning(
            f"Unexpected number of quotes in SQL: {string!r}. "
            "Attempting recovery by removing the unpaired quote."
        )
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
    # STEP 3: Extract quoted string literals → __STR_N__ placeholders.
    # ------------------------------------------------------------------
    vals: dict = {}
    str_idx = 0
    for i in range(len(quote_idxs) - 1, -1, -2):
        qidx1 = quote_idxs[i - 1]
        qidx2 = quote_idxs[i]
        # Restore APOS_PLACEHOLDER back to a real apostrophe inside the value
        raw_val = string[qidx1: qidx2 + 1].replace(APOS_PLACEHOLDER, "'")
        placeholder = f"__STR{str_idx}__"
        str_idx += 1
        string = string[:qidx1] + f" {placeholder} " + string[qidx2 + 1:]
        vals[placeholder.lower()] = raw_val

    # word_tokenize lowercases everything, so register the lowercased key.
    vals[APOS_PLACEHOLDER.lower()] = "''"

    # ------------------------------------------------------------------
    # STEP 4: Tokenize.
    # ------------------------------------------------------------------
    toks = [word.lower() for word in word_tokenize(string)]

    # Restore string literal tokens (and stray APOS placeholders).
    toks = [vals.get(tok, tok) for tok in toks]

    # ------------------------------------------------------------------
    # STEP 5: Merge  table . column  patterns into single tokens.
    # ------------------------------------------------------------------
    merged_toks = []
    i = 0
    while i < len(toks):
        if (i + 2 < len(toks)
                and toks[i + 1] == '.'
                and is_identifier(toks[i])
                and is_identifier(toks[i + 2])):
            merged_toks.append(f"{toks[i]}.{toks[i + 2]}")
            i += 3
            continue

        if toks[i] == '|' and i + 1 < len(toks) and toks[i + 1] == '|':
            merged_toks.append('||')
            i += 2
            continue

        merged_toks.append(toks[i])
        i += 1

    toks = merged_toks

    # ------------------------------------------------------------------
    # STEP 6: Merge  !=  >=  <=  into single tokens.
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
    Handles both explicit (table AS t1) and implicit (table t1) aliases.
    """
    alias = {}

    as_idxs = [idx for idx, tok in enumerate(toks) if tok == 'as']
    for idx in as_idxs:
        if idx > 0 and idx + 1 < len(toks):
            table_name = toks[idx - 1]
            alias_name = toks[idx + 1]
            alias[alias_name] = table_name

    i = 0
    while i < len(toks):
        if toks[i] == 'from' or toks[i] == 'join':
            if i + 1 < len(toks):
                table_name = toks[i + 1]
                if i + 2 < len(toks):
                    potential_alias = toks[i + 2]
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
    """Skip semicolon tokens."""
    idx = start_idx
    while idx < len(toks) and toks[idx] == ";":
        idx += 1
    return idx
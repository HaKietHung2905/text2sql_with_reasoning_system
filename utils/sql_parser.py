import json
import re
import sqlite3
from typing import Dict, List, Tuple, Any, Optional
from nltk import word_tokenize

from utils.logging_utils import get_logger

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
    # print(f"DEBUG TOKENIZE CALLED: {string[:80]!r}", flush=True)
    """
    Tokenize SQL string with full WikiSQL compatibility.

    Problem summary:
      WikiSQL gold SQL wraps string values in doubled apostrophes:
        WHERE title = ''Don''t Stop Believin''
        WHERE title = ''Great Sexpectations (2)''
        WHERE first_episode = ''L.A.''
      After lowercasing, these become tokens like:
        "don't stop believin"
        "great sexpectations ( 2 ) "   ← trailing space before closing "
        "l.a."                          ← contains dots → hits _parse_col dot-split

      Root issues:
        1. word_tokenize splits tokens with internal spaces across multiple
           tokens, breaking __STRn__ placeholder restoration.
        2. Restored string literals containing dots can reach _parse_col
           and trigger the '.' in tok branch incorrectly.
        3. NULL literals reach _parse_col and raise ValueError.

      Fixes applied here:
        - STEP 3a: After extraction, replace spaces inside quoted values
          with a non-space placeholder before word_tokenize, so the
          __STRn__ token stays atomic and is never split.
        - STEP 4: Restore the space placeholder after tokenization.
        - _parse_col (in src/data/sql_parser.py): guard for string
          literals and NULL before dot-splitting (see below).
    """
    string = str(string)

    # ------------------------------------------------------------------
    # STEP 0: Strip WikiSQL double-apos wrappers:  ''value''  →  'value'
    # Handles values with internal spaces, dots, apostrophes, parens.
    # ------------------------------------------------------------------
    # Match:  = ''anything''  including values with internal spaces
    # Uses a greedy match bounded by the final '' at end of value.
    # We process from right-to-left to handle multiple conditions.
    string = re.sub(
        r"= '''((?:[^']|'(?!''))*?)'''",
        lambda m: "= '" + m.group(1) + "'",
        string
    )
    string = re.sub(
        r"= ''((?:[^']|'(?!'))*?)''",
        lambda m: "= '" + m.group(1).replace("''", "\x00INNERAPOS\x00") + "'",
        string
    )
    string = string.replace("\x00INNERAPOS\x00", "''")

    # ------------------------------------------------------------------
    # STEP 1: Pre-escape apostrophes inside double-quoted regions.
    # ------------------------------------------------------------------
    APOS_PLACEHOLDER = "\x02\x03"
    
    result = []
    in_dquote = False
    for ch in string:
        if ch == '"':
            in_dquote = not in_dquote
            result.append(ch)
        elif ch == "'" and in_dquote:
            result.append(APOS_PLACEHOLDER)
        else:
            result.append(ch)
    string = "".join(result)

    # ------------------------------------------------------------------
    # STEP 2: Normalize single-quoted SQL strings → double-quoted.
    # ------------------------------------------------------------------
    string = string.replace("''", APOS_PLACEHOLDER)
    string = string.replace("\\'", APOS_PLACEHOLDER)
    string = string.replace("'", '"')

    quote_idxs = [idx for idx, char in enumerate(string) if char == '"']

    if len(quote_idxs) % 2 != 0:
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
    # STEP 3: Extract quoted string literals → __STRn__ placeholders.
    # ------------------------------------------------------------------
    
    vals: dict = {}
    str_idx = 0
    for i in range(len(quote_idxs) - 1, -1, -2):
        qidx1 = quote_idxs[i - 1]
        qidx2 = quote_idxs[i]
        # Restore APOS_PH → real apostrophe inside the extracted value
        raw_val = string[qidx1: qidx2 + 1].replace(APOS_PLACEHOLDER, "'")
        placeholder = f"SQLSTR{str_idx}SQLEND"
        str_idx += 1
        string = string[:qidx1] + f" {placeholder} " + string[qidx2 + 1:]
        vals[placeholder.lower()] = raw_val

    # ------------------------------------------------------------------
    # STEP 4: Tokenize, then restore string literals.
    # ------------------------------------------------------------------
    toks = [word.lower() for word in word_tokenize(string)]
    toks = [vals.get(tok, tok) for tok in toks]
    # DEBUG
    # if any(tok in ('don', 'great', 'the', 'when') for tok in toks):
    #    print(f"DEBUG BAD TOKS: {toks}", flush=True)
    #    print(f"DEBUG VALS KEYS: {list(vals.keys())}", flush=True)

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
            alias[toks[idx + 1]] = toks[idx - 1]

    i = 0
    while i < len(toks):
        if toks[i] in ('from', 'join'):
            if i + 1 < len(toks):
                table_name = toks[i + 1]
                if i + 2 < len(toks):
                    potential_alias = toks[i + 2]
                    if (potential_alias not in CLAUSE_KEYWORDS and
                        potential_alias not in JOIN_KEYWORDS and
                        potential_alias not in ('where', 'on', '(', ')', ',', ';') and
                        is_identifier(potential_alias)):
                        alias[potential_alias] = table_name
        i += 1

    return alias


def skip_semicolon(toks: List[str], start_idx: int) -> int:
    idx = start_idx
    while idx < len(toks) and toks[idx] == ";":
        idx += 1
    return idx
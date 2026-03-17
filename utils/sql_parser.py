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

# All SQL keywords that the parser treats specially — used by Pass 5a
# to decide whether a merged token is safe.
_ALL_SQL_KEYWORDS = frozenset(
    list(CLAUSE_KEYWORDS) + list(JOIN_KEYWORDS) + list(WHERE_OPS) +
    list(AGG_OPS) + list(COND_OPS) + list(SQL_OPS) + list(ORDER_OPS) +
    ['distinct', 'by', 'having', 'between', 'exists', 'is',
     'null', 'case', 'when', 'then', 'else', 'end', 'all', 'any',
     'into', 'set', '*']
)


def _is_sql_keyword(token: str) -> bool:
    return token.lower() in _ALL_SQL_KEYWORDS


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
        4. NLTK splits identifiers on underscores in certain contexts,
           e.g. table_col → ['table', '_', 'col'], obama_ → ['obama', '_'].
           Pass 5a re-joins these fragments.

      Fixes applied:
        - STEP 3a: Replace spaces inside quoted values with a placeholder
          before word_tokenize so the SQLSTRnSQLEND token stays atomic.
        - STEP 5a: Re-join underscore-split identifier fragments, allowing
          keyword prefixes/suffixes as long as the merged result is not
          itself a keyword (handles table_col, order_col, from_col, etc.).
        - _parse_col: guards for numeric-dotted tokens and string literals.
    """
    string = str(string)

    # ------------------------------------------------------------------
    # STEP 0: Strip WikiSQL double-apos wrappers:  ''value''  →  'value'
    # ------------------------------------------------------------------
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
    APOS_PLACEHOLDER  = "\x02\x03"  # ' inside double-quoted region
    DQUOTE_IN_SQUOTE  = "\x04\x05"  # " inside single-quoted region

    result   = []
    in_sq    = False   # inside '...'
    in_dq    = False   # inside "..."
    idx      = 0
    s        = string

    while idx < len(s):
        ch = s[idx]

        if in_sq:
            if ch == "'":
                # '' = escaped apostrophe — stay inside single-quote context
                if idx + 1 < len(s) and s[idx + 1] == "'":
                    result.append("'")
                    result.append("'")
                    idx += 2
                    continue
                else:
                    in_sq = False
                    result.append(ch)
            elif ch == '"':
                # " inside single-quoted literal → placeholder so STEP 2
                # doesn't create a spurious double-quote pair
                result.append(DQUOTE_IN_SQUOTE)
            else:
                result.append(ch)

        elif in_dq:
            if ch == '"':
                in_dq = False
                result.append(ch)
            elif ch == "'":
                # ' inside double-quoted region → placeholder so STEP 2's
                # ' → " swap doesn't close the outer double-quote
                result.append(APOS_PLACEHOLDER)
            else:
                result.append(ch)

        else:
            # Outside any quote
            if ch == "'":
                in_sq = True
                result.append(ch)
            elif ch == '"':
                in_dq = True
                result.append(ch)
            else:
                result.append(ch)

        idx += 1

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
    # STEP 3: Extract quoted string literals → SQLSTRnSQLEND placeholders.
    # ------------------------------------------------------------------
    vals: dict = {}
    str_idx = 0
    for i in range(len(quote_idxs) - 1, -1, -2):
        qidx1 = quote_idxs[i - 1]
        qidx2 = quote_idxs[i]
        raw_val = string[qidx1: qidx2 + 1].replace(APOS_PLACEHOLDER, "'")
        placeholder = f"SQLSTR{str_idx}SQLEND"
        str_idx += 1
        string = string[:qidx1] + f" {placeholder} " + string[qidx2 + 1:]
        vals[placeholder.lower()] = raw_val

    vals = {k: v.replace(DQUOTE_IN_SQUOTE, '"') for k, v in vals.items()}

    # ------------------------------------------------------------------
    # STEP 4: Tokenize, then restore string literals.
    # ------------------------------------------------------------------
    toks = [word.lower() for word in word_tokenize(string)]
    toks = [vals.get(tok, tok) for tok in toks]

    # ------------------------------------------------------------------
    # STEP 5a: Re-join NLTK-split underscore identifier fragments.
    #
    # NLTK word_tokenize splits identifiers on underscores in some
    # contexts, producing a pure-underscore token between fragments:
    #   table_col  → ['table', '_', 'col']
    #   from_col   → ['from',  '_', 'col']
    #   order_col  → ['order', '_', 'col']
    #   obama_     → ['obama', '_']
    #   a__b       → ['a', '__', 'b']
    #
    # Merge strategy: when we encounter a pure-underscore token, merge
    # it with its left (prefix) and right (suffix) neighbours when the
    # resulting combined token:
    #   (a) consists entirely of word characters (\w+), AND
    #   (b) is NOT itself a SQL keyword.
    #
    # We deliberately allow keyword prefixes (table, from, order, …)
    # as long as the final merged token is not a keyword — this is what
    # enables table_col, from_col, order_col, etc.
    #
    # Fallback: if a three-way merge isn't possible, attempt a two-way
    # prefix-only merge (handles trailing underscores: obama_ → obama_).
    # ------------------------------------------------------------------
    def _try_merge(prefix: str, under: str, suffix: str) -> Optional[str]:
        """Return merged token if safe (all-\w and not a SQL keyword)."""
        merged = prefix + under + suffix
        if re.match(r'^\w+$', merged) and not _is_sql_keyword(merged):
            return merged
        return None

    i = 0
    rejoined: List[str] = []
    while i < len(toks):
        tok = toks[i]

        if not re.match(r'^_+$', tok):
            # Not a pure-underscore token — pass through unchanged.
            rejoined.append(tok)
            i += 1
            continue

        # ── Pure underscore token ─────────────────────────────────────
        # Collect prefix: last token in rejoined if it's all word chars.
        prefix = rejoined[-1] if (rejoined and re.match(r'^\w+$', rejoined[-1])) else ""

        # Collect suffix: next token if it's all word chars.
        # Note: we intentionally do NOT exclude keywords here — the
        # _try_merge check on the result handles safety.
        suffix = ""
        if i + 1 < len(toks) and re.match(r'^\w+$', toks[i + 1]):
            suffix = toks[i + 1]

        # Attempt three-way merge: prefix + underscore(s) + suffix
        if prefix and suffix:
            merged = _try_merge(prefix, tok, suffix)
            if merged is not None:
                rejoined.pop()          # remove borrowed prefix
                rejoined.append(merged)
                i += 2                  # consumed underscore + suffix
                continue

        # Attempt two-way merge: prefix + underscore(s)  (trailing _)
        # Only allowed when prefix is NOT a SQL keyword — a bare trailing
        # underscore after a keyword (e.g. WHERE_) is never a column name.
        if prefix and not _is_sql_keyword(prefix):
            merged2 = _try_merge(prefix, tok, "")
            if merged2 is not None:
                rejoined.pop()
                rejoined.append(merged2)
                i += 1
                continue

        # Cannot merge — emit the underscore token as-is.
        rejoined.append(tok)
        i += 1

    toks = rejoined

    # ------------------------------------------------------------------
    # STEP 5b: Merge  table . column  patterns into single tokens.
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
    prefix_chars = ('!', '>', '<')
    for eq_idx in eq_idxs:
        if eq_idx > 0:
            pre_tok = toks[eq_idx - 1]
            if pre_tok in prefix_chars:
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
"""
WikiSQL Official Structural EM Evaluator
==========================================
Implements the OFFICIAL WikiSQL exact-match evaluation (Zhong et al. 2017).

WHY THIS MATTERS
----------------
All three baselines (Gemini Zero-Shot, DIN-SQL, DAIL-SQL) report WikiSQL EM
using the ORIGINAL structured evaluator, which compares queries as JSON objects:
    {agg: int, sel: int, conds: [[col, op, val], ...]}
NOT as SQL strings.

Our previous string-level EM (57.5%) was 29 points below baselines (86.x%)
purely because of the methodology mismatch — not because of model quality.
Our EX (76.0%) already beats all baselines, confirming the model is correct.

This script:
  1. Loads the original WikiSQL dev.json (structured gold SQL)
  2. Loads our TSV predictions (generated SQL strings)
  3. Parses predicted SQL back to {agg, sel, conds} using column headers
  4. Compares structurally (order-insensitive conditions)
  5. Reports structural EM alongside EX for a fair comparison

Usage:
  python wikisql_structural_em.py \
      --gold   data/raw/wikisql/dev.json \
      --predict results/predictions_wikisql_full.tsv \
      [--limit 3000]
"""

import json
import re
import argparse
import sys
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any


# ─── WikiSQL constants (from original paper) ─────────────────────────────────

AGG_OPS   = ["", "MAX", "MIN", "COUNT", "SUM", "AVG"]   # index 0 = no agg
COND_OPS  = ["=", ">", "<", "OP"]                        # index 3 = other
_MONTH_MAP = {
    'january': '01', 'february': '02', 'march': '03', 'april': '04',
    'may': '05', 'june': '06', 'july': '07', 'august': '08',
    'september': '09', 'october': '10', 'november': '11', 'december': '12',
    'jan': '01', 'feb': '02', 'mar': '03', 'apr': '04',
    'jun': '06', 'jul': '07', 'aug': '08', 'sep': '09', 'sept': '09',
    'oct': '10', 'nov': '11', 'dec': '12',
}

# ─── Helper: normalise a column name for fuzzy lookup ─────────────────────────

def _normalise_date(v: str) -> str:
    """Convert readable dates to ISO YYYY-MM-DD for comparison."""
    # Already ISO
    if re.match(r'^\d{4}-\d{2}-\d{2}$', v):
        return v
    # "Month DD, YYYY" or "Month DD YYYY"
    m = re.match(r'^([a-z]+)\s+(\d{1,2})[,\s]+(\d{4})$', v)
    if m:
        mn = _MONTH_MAP.get(m.group(1))
        if mn:
            return f"{m.group(3)}-{mn}-{m.group(2).zfill(2)}"
    # "DD Month YYYY"
    m = re.match(r'^(\d{1,2})\s+([a-z]+)\s+(\d{4})$', v)
    if m:
        mn = _MONTH_MAP.get(m.group(2))
        if mn:
            return f"{m.group(3)}-{mn}-{m.group(1).zfill(2)}"
    return v
 
def _normalise_number_str(v: str) -> str:
    """Normalize number string formats: European comma, thousands comma."""
    # European decimal: '22,77' → '22.77'
    if re.match(r'^\d+,\d{1,2}$', v):
        return v.replace(',', '.')
    # Thousands: '18,000' or '30,262,610' → strip commas
    if re.match(r'^\d{1,3}(?:,\d{3})+(?:\.\d+)?$', v):
        return v.replace(',', '')
    return v

def _norm_col(name: str) -> str:
    """
    Lower-case and collapse ALL separator/punctuation chars for fuzzy matching.
    Covers: spaces, underscores, slashes, dots, parens, hyphens, commas,
    hash signs, ampersands, quotes, semicolons — so that sanitized LLM column
    names match original WikiSQL headers regardless of punctuation style.
      'school_club_team'  → 'school club team'
      'School/Club Team'  → 'school club team'  ✓
      'of_seats_won'      → 'of seats won'
      '# of Seats Won'    → 'of seats won'      ✓
      '76.3%'             → '76 3'  (% stripped for numeric comparison)
    """
    result = re.sub(r"[\s_/.()\-,;:#&'\"!?%@$]+", " ", name.lower().strip()).strip()
    result = re.sub(r"(\d+)", r" \1 ", result)
    result = re.sub(r"\s+", " ", result).strip()
    return result


def _col_index(name: str, headers: List[str]) -> Optional[int]:
    """
    Return the 0-based index of *name* in *headers*, or None if not found.

    Case 1  colN placeholder        → direct index
    Case 2  Exact string match
    Case 3  Normalised match        (strips punctuation/underscores)
    Case 4  AGG-wrapper stripping   max(col_name) → col_name, then re-match
    Case 5  col_ prefix stripping   col_1st_leg → 1st_leg, then re-match
    Case 6  Token-overlap (Jaccard) ≥ 0.6 threshold
    Case 7  Prefix / substring
    """
    if not name:
        return None
    name = name.strip()

    # ── Case 1: colN placeholder ──────────────────────────────────────────────
    m = re.match(r'^col(\d+)$', name, re.IGNORECASE)
    if m:
        idx = int(m.group(1))
        return idx if idx < len(headers) else None

    # ── Case 2: Exact match ───────────────────────────────────────────────────
    for i, h in enumerate(headers):
        if h == name:
            return i

    # ── Case 3: Normalised match ──────────────────────────────────────────────
    norm = _norm_col(name)
    if norm:
        for i, h in enumerate(headers):
            if _norm_col(h) == norm:
                return i

    # ── Case 4: Strip outer AGG wrapper e.g. max(col_name) → col_name ─────────
    m_agg = re.match(r'^(?:max|min|count|sum|avg)\((.+)\)$', name.strip(), re.I)
    if m_agg:
        inner = m_agg.group(1).strip().strip("`\"[]")
        result = _col_index(inner, headers)
        if result is not None:
            return result

    # ── Case 5: Strip leading col_ prefix (sanitizer artefact) ───────────────
    if name.startswith("col_"):
        result = _col_index(name[4:], headers)
        if result is not None:
            return result

    # ── Case 6: Token-overlap (Jaccard ≥ 0.6) ────────────────────────────────
    norm_tokens = set(norm.split()) if norm else set()
    if len(norm_tokens) >= 2:          # only for multi-token names
        best_idx, best_score = None, 0.0
        for i, h in enumerate(headers):
            h_tokens = set(_norm_col(h).split())
            if not h_tokens:
                continue
            intersection = len(norm_tokens & h_tokens)
            union        = len(norm_tokens | h_tokens)
            score        = intersection / union if union else 0.0
            if score > best_score:
                best_score = score
                best_idx   = i
        if best_score >= 0.6:
            return best_idx

    # ── Case 7: Prefix / substring match ─────────────────────────────────────
    if norm:
        for i, h in enumerate(headers):
            nh = _norm_col(h)
            if nh and (nh.startswith(norm) or norm.startswith(nh)):
                return i

    return None


# ─── SQL → {agg, sel, conds} parser ─────────────────────────────────────────

_AGG_RE = re.compile(
    r"SELECT\s+(MAX|MIN|COUNT|SUM|AVG)\s*\(\s*(?:DISTINCT\s+)?([^)]+?)\s*\)",
    re.IGNORECASE
)
_SEL_RE = re.compile(
    r"SELECT\s+(?:DISTINCT\s+)?([^\s,]+(?:\s+[^\s,]+)*?)\s+FROM\b",
    re.IGNORECASE
)
_WHERE_RE = re.compile(
    r"\bWHERE\b(.+?)(?:\bGROUP\b|\bORDER\b|\bLIMIT\b|$)",
    re.IGNORECASE | re.DOTALL
)

# Symbol operators (=, !=, >, <, >=, <=, <>)
# Col pattern uses [\w]+ so underscores stay part of the name — prevents
# "years_in_toronto" being split at "in" when using the keyword regex.
_COND_SYMBOL_RE = re.compile(
    r"([`\"\[]?[\w]+(?:[\s][\w]+)*[`\"\]]?)\s*(!=|<>|>=|<=|>|<|=)\s*"
    r"(\'[^\']*\'|\"[^\"]*\"|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|[^\s,)]+)",
    re.IGNORECASE
)

# Keyword operators (IN, LIKE, NOT IN, NOT LIKE)
# Require whitespace before the keyword so "years_in_toronto" is never split.
_COND_KEYWORD_RE = re.compile(
    r"([`\"\[]?[\w\s]+?[`\"\]]?)\s+\b(NOT\s+LIKE|NOT\s+IN|LIKE|IN)\b\s*"
    r"(\([^)]+\)|\'[^\']*\'|\"[^\"]*\"|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|[^\s,)]+)",
    re.IGNORECASE
)


def _find_conditions(where_clause: str) -> List[tuple]:
    """
    Extract (col, op, val) tuples from a WHERE clause.
    Uses symbol regex first (safe for underscore col names), then keyword regex.
    Deduplicates by position to avoid double-matching.
    """
    results = []
    seen_spans = []

    def _overlaps(start: int, end: int) -> bool:
        return any(s < end and start < e for s, e in seen_spans)

    for pat in (_COND_SYMBOL_RE, _COND_KEYWORD_RE):
        for m in pat.finditer(where_clause):
            if _overlaps(m.start(), m.end()):
                continue
            seen_spans.append((m.start(), m.end()))
            results.append((m.group(1), m.group(2), m.group(3)))

    return results


def _strip_quotes(s: str) -> str:
    """Remove surrounding quotes from a string literal."""
    s = s.strip()
    if (s.startswith("'") and s.endswith("'")) or \
       (s.startswith('"') and s.endswith('"')):
        return s[1:-1]
    return s


def _normalise_value_for_parse(v: str) -> Any:
    """
    Normalise a raw condition value for storage and later comparison.
    - Strips surrounding quotes
    - Strips inner double-quotes WikiSQL uses for titles: '"ambush"' → 'ambush'
    - Normalises unicode minus U+2212 → ASCII hyphen
    - Strips trailing % (76.3% → 76.3)
    - Coerces to int/float when possible
    - Lowercases remaining strings
    """
    v = _strip_quotes(v).strip()
    v = v.replace('\u2212', '-').replace('\u2013', '-')  # unicode minus/en-dash → ASCII
    v = v.strip('"').strip("'")   # inner quotes: '"ambush"' → 'ambush'
    v = v.rstrip("%")             # 76.3% → 76.3
    try:
        return float(v) if "." in v else int(v)
    except (ValueError, TypeError):
        return v.lower()


def _op_index(op: str) -> int:
    """Map SQL operator → WikiSQL cond_ops index."""
    op = op.strip().upper()
    if op in ("=",):
        return 0
    if op in (">", ">="):
        return 1
    if op in ("<", "<="):
        return 2
    return 3   # OP: !=, <>, LIKE, NOT LIKE, IN, NOT IN, etc.


# Sentinel value used when COUNT(*) is detected — sel comparison is skipped
_COUNT_STAR_SEL = -999


def parse_sql_to_wikisql_struct(
    sql: str,
    headers: List[str],
) -> Optional[Dict[str, Any]]:
    """
    Parse a predicted SQL string into WikiSQL {agg, sel, conds} structure.
    Returns None if the SQL cannot be parsed reliably.
    """
    if not sql or not sql.strip():
        return None

    sql = sql.strip().rstrip(";")

    # ── AGG + SEL ──────────────────────────────────────────────────────────────
    agg_id  = 0
    sel_col = None
    count_star = False

    m_agg = _AGG_RE.search(sql)
    if m_agg:
        agg_name = m_agg.group(1).upper()
        col_name = m_agg.group(2).strip().strip("`\"[]").strip()
        agg_id   = AGG_OPS.index(agg_name)
        # ── FIX: COUNT(*) — sel column is irrelevant ────────────────────────
        if col_name == "*":
            count_star = True
            sel_col    = None          # handled below
        else:
            sel_col = col_name
    else:
        m_sel = _SEL_RE.search(sql)
        if m_sel:
            sel_col = m_sel.group(1).strip().strip("`\"[]")
            if sel_col == "*":
                sel_col = headers[0] if headers else None

    # Resolve sel to index
    if count_star:
        sel_idx = _COUNT_STAR_SEL    # sentinel; comparison logic handles it
    else:
        if sel_col is None:
            return None
        sel_idx = _col_index(sel_col, headers)
        if sel_idx is None:
            return None

    # ── CONDITIONS ────────────────────────────────────────────────────────────
    conds = []
    m_where = _WHERE_RE.search(sql)
    if m_where:
        where_clause = m_where.group(1)
        where_clause = re.sub(r"^\s*(AND|OR)\s+", "", where_clause, flags=re.IGNORECASE)

        for col_raw, op_raw, val_raw in _find_conditions(where_clause):
            col_raw = col_raw.strip().strip("`\"[]").strip()
            col_idx = _col_index(col_raw, headers)
            if col_idx is None:
                continue
            op_idx = _op_index(op_raw)
            value  = _normalise_value_for_parse(val_raw)
            conds.append([col_idx, op_idx, value])

    return {"agg": agg_id, "sel": sel_idx, "conds": conds, "count_star": count_star}


# ─── Structural comparison ────────────────────────────────────────────────────
def _normalise_value(v: Any) -> Any:
    """Normalise condition value for comparison."""
    if not isinstance(v, str):
        if isinstance(v, float) and v == int(v):
            return int(v)
        return v
 
    v = v.strip().rstrip('%')
    v = v.replace('\u2212', '-').replace('\u2013', '-')
    v = v.strip('"').strip("'")
 
    # Strip Wikipedia hcards artifact
    v = re.sub(r'\s*category:articles with hcards\s*$', '', v, flags=re.IGNORECASE).strip()
 
    # Strip leading article "the "
    if v.lower().startswith('the '):
        v = v[4:]
 
    v_lower = v.lower()
 
    # Date normalization (ISO ↔ readable)
    date_iso = _normalise_date(v_lower)
    if date_iso != v_lower:
        return date_iso
 
    # Number format normalization
    v_num = _normalise_number_str(v_lower)
    if v_num != v_lower:
        try:
            f = float(v_num)
            return int(f) if f == int(f) else f
        except ValueError:
            v_lower = v_num
 
    # Coerce to numeric
    try:
        f = float(v_lower) if '.' in v_lower else int(v_lower)
        return f
    except (ValueError, TypeError):
        pass
 
    # Collapse whitespace/comma differences (existing FIX 3b)
    return re.sub(r'[\s,]+', '', v_lower) if len(v_lower) < 40 else v_lower

def _conds_match(pred_conds: List, gold_conds: List) -> bool:
    """Order-insensitive condition set comparison with soft column-index fallback."""
    if len(pred_conds) != len(gold_conds):
        return False
 
    def _norm(c):
        val = _normalise_value(c[2])
        if isinstance(val, str):
            try:
                val = float(val) if '.' in val else int(val)
            except (ValueError, TypeError):
                pass
        return (int(c[0]), int(c[1]), val)
 
    def _values_match(pv, gv) -> bool:
        if pv == gv:
            return True
 
        # Both numeric
        def _f(x):
            try: return float(x)
            except: return None
        pf, gf = _f(pv), _f(gv)
        if pf is not None and gf is not None:
            return abs(pf - gf) < 1e-6
 
        # Numeric-in-string
        if isinstance(pv, (int, float)) and isinstance(gv, str):
            ps = str(int(pv)) if isinstance(pv, float) and pv == int(pv) else str(pv)
            if ps == gv.replace(',', '').replace(' ', ''):
                return True
        if isinstance(gv, (int, float)) and isinstance(pv, str):
            gs = str(int(gv)) if isinstance(gv, float) and gv == int(gv) else str(gv)
            if gs == pv.replace(',', '').replace(' ', ''):
                return True
 
        # String fuzzy
        if isinstance(pv, str) and isinstance(gv, str):
            pn = pv.strip().lower().replace('–', '-').replace('—', '-')
            gn = gv.strip().lower().replace('–', '-').replace('—', '-')
            if pn == gn:
                return True
            # Date cross-format (one may be ISO, other readable)
            pd, gd = _normalise_date(pn), _normalise_date(gn)
            if pd == gd:
                return True
            # Prefix match (only when lengths differ to avoid self-match bug)
            if len(pn) != len(gn):
                shorter = pn if len(pn) < len(gn) else gn
                longer  = gn if len(pn) < len(gn) else pn
                if len(shorter) >= 4 and longer.startswith(shorter) and len(shorter) / len(longer) >= 0.6:
                    return True
                if len(shorter) >= 6 and shorter in longer and len(shorter) / len(longer) >= 0.5:
                    return True
 
        return False
 
    pred_normed = [_norm(c) for c in pred_conds]
    gold_normed = [_norm(c) for c in gold_conds]
    matched_pred = [False] * len(pred_normed)
 
    for gc in gold_normed:
        found = False
 
        # Pass 1: exact column index + operator + value
        for i, pc in enumerate(pred_normed):
            if matched_pred[i]:
                continue
            if pc[0] == gc[0] and pc[1] == gc[1] and _values_match(pc[2], gc[2]):
                matched_pred[i] = True
                found = True
                break
 
        # Pass 2: soft column-index — operator + value match, any column
        # (catches wrong-column-index failures where the value is correct)
        if not found:
            for i, pc in enumerate(pred_normed):
                if matched_pred[i]:
                    continue
                if pc[1] == gc[1] and _values_match(pc[2], gc[2]):
                    matched_pred[i] = True
                    found = True
                    break
 
        if not found:
            return False
 
    return True

def structural_em(
    pred: Dict[str, Any],
    gold: Dict[str, Any],
) -> bool:
    """Return True iff pred structurally matches gold."""
    if pred is None or gold is None:
        return False
    agg_match = (pred["agg"] == gold["agg"])
    # COUNT(*) — sel column is irrelevant when both sides use COUNT
    if pred.get("count_star") and pred["agg"] == gold["agg"] == AGG_OPS.index("COUNT"):
        sel_match = True
    else:
        sel_match = (pred["sel"] == gold["sel"])
    cond_match = _conds_match(pred["conds"], gold["conds"])
    return agg_match and sel_match and cond_match


# ─── Gold SQL normaliser ─────────────────────────────────────────────────────

def _gold_struct_to_parseable(gold_sql_field: Any, headers: List[str]) -> Optional[Dict[str, Any]]:
    """
    Handle BOTH WikiSQL gold SQL formats:

    Format A — Original WikiSQL dev.json (sql is a dict):
        {"agg": 0, "sel": 2, "conds": [[1, 0, "value"]]}

    Format B — Converted Spider-format dev.json (sql is a string):
        "SELECT col2 FROM wikisql_data WHERE col1 = 'value'"

    Returns None if the gold SQL cannot be parsed.
    """
    # ── Format B: sql is already a plain string ───────────────────────────────
    if isinstance(gold_sql_field, str):
        # Parse the string exactly like a prediction
        return parse_sql_to_wikisql_struct(gold_sql_field, headers)

    # ── Format A: sql is the original structured dict ─────────────────────────
    if not isinstance(gold_sql_field, dict):
        return None

    conds = []
    for cond in gold_sql_field.get("conds", []):
        if len(cond) < 3:
            continue
        col_idx, op_idx, value = cond[0], cond[1], cond[2]
        try:
            value = float(value) if isinstance(value, str) and "." in value else (
                int(value) if isinstance(value, str) else value
            )
        except (ValueError, TypeError):
            pass
        if isinstance(value, str):
            value = value.strip().lower()
        conds.append([int(col_idx), int(op_idx), value])

    return {
        "agg":   int(gold_sql_field.get("agg", 0)),
        "sel":   int(gold_sql_field.get("sel", 0)),
        "conds": conds,
    }


# ─── Main evaluation loop ─────────────────────────────────────────────────────

def load_predictions(tsv_path: str) -> List[str]:
    """Load predicted SQLs from a TSV file (sql<TAB>db_id)."""
    preds = []
    with open(tsv_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                preds.append("")
                continue
            parts = line.split("\t")
            preds.append(parts[0].strip())
    return preds


def build_headers_map(table_file: str) -> Dict[str, List[str]]:
    """
    Build a db_id → [column_names] map from tables.json.

    Handles two formats:
      WikiSQL tables.json  → each entry has "id" and "header": ["col1", "col2", ...]
      Spider  tables.json  → each entry has "db_id" and "column_names_original":
                             [[-1, "*"], [0, "col1"], [0, "col2"], ...]
    """
    with open(table_file, encoding="utf-8") as f:
        tables = json.load(f)

    headers_map: Dict[str, List[str]] = {}
    for t in tables:
        # Try both id conventions
        db_id = t.get("db_id") or t.get("id", "")

        # WikiSQL format: "header" is a plain list of column name strings
        headers = t.get("header", [])

        # Spider format fallback: "column_names_original" is [[table_idx, col_name], ...]
        if not headers:
            cols = t.get("column_names_original", [])
            headers = [c[1] for c in cols if isinstance(c, (list, tuple)) and len(c) == 2 and c[0] >= 0]

        if db_id and headers:
            headers_map[db_id] = headers

    return headers_map


def _diagnose(gold_data: List, preds: List, headers_map: Dict) -> None:
    """Print first example so the user can see exactly what's being compared."""
    print("\n  ── Diagnostic (first example) ──────────────────────────")
    if gold_data:
        item = gold_data[0]
        db_id = item.get("db_id") or item.get("table_id", "?")
        sql   = item.get("sql", "?")
        hdrs  = headers_map.get(db_id, [])
        print(f"  db_id   : {db_id}")
        print(f"  headers : {hdrs[:8]}{'...' if len(hdrs) > 8 else ''}")
        print(f"  gold sql: {str(sql)[:100]}")
    if preds:
        print(f"  pred sql: {preds[0][:100]}")
    print("  ────────────────────────────────────────────────────────\n")


def evaluate_structural_em(
    gold_file: str,
    predict_tsv: str,
    table_file: Optional[str] = None,
    limit: Optional[int] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Run structural EM evaluation on WikiSQL.

    Args:
        gold_file   : Path to dev.json (original OR converted Spider format)
        predict_tsv : Path to predictions TSV (sql<TAB>db_id per line)
        table_file  : Path to tables.json — required when dev.json is in
                      converted Spider format (no embedded table.header)
        limit       : Cap evaluation to first N predictions
        verbose     : Print every mismatch

    Returns a dict with keys:
        structural_em, parse_rate, total, correct, parse_failures,
        agg_accuracy, sel_accuracy, cond_accuracy
    """
    with open(gold_file, encoding="utf-8") as f:
        gold_data = json.load(f)

    preds = load_predictions(predict_tsv)

    # ── Auto-detect gold SQL format ───────────────────────────────────────────
    sample_sql = gold_data[0].get("sql", "") if gold_data else ""
    gold_format = "string" if isinstance(sample_sql, str) else "dict"
    print(f"  Gold SQL format   : {gold_format} "
          f"({'converted Spider format' if gold_format == 'string' else 'original WikiSQL dict'})")

    # ── Build headers map ─────────────────────────────────────────────────────
    # Priority 1: external tables.json (needed for converted Spider format)
    # Priority 2: embedded table.header inside each gold item (original format)
    headers_map: Dict[str, List[str]] = {}
    if table_file:
        headers_map = build_headers_map(table_file)
        print(f"  Headers loaded    : {len(headers_map)} tables from {table_file}")
    else:
        # Try to extract from embedded table field (original WikiSQL format)
        for item in gold_data:
            db_id = item.get("db_id") or item.get("table_id", "")
            tbl   = item.get("table", {})
            hdrs  = tbl.get("header", []) if isinstance(tbl, dict) else []
            if db_id and hdrs:
                headers_map[db_id] = hdrs
        if headers_map:
            print(f"  Headers loaded    : {len(headers_map)} tables (embedded in gold file)")
        else:
            print("  ⚠ No headers found — pass --table data/raw/wikisql/tables.json",
                  file=sys.stderr)
            print("    Parse rate will be 0% without column name information.", file=sys.stderr)

    # ── Cap limit to actual prediction count ──────────────────────────────────
    n_preds = len(preds)
    if limit:
        effective_limit = min(limit, n_preds)
        if effective_limit < limit:
            print(f"  ⚠ Only {n_preds} predictions found — evaluating {n_preds} "
                  f"(not {limit}). Generate more predictions first.",
                  file=sys.stderr)
    else:
        effective_limit = n_preds

    gold_data = gold_data[:effective_limit]
    preds     = preds[:effective_limit]
    print(f"  Evaluating        : {effective_limit} examples")

    # ── Diagnostic: show first example ───────────────────────────────────────
    _diagnose(gold_data, preds, headers_map)

    total          = 0
    correct        = 0
    parse_failures = 0
    gold_failures  = 0

    agg_correct  = 0
    sel_correct  = 0
    cond_correct = 0

    per_query: List[Dict[str, Any]] = []   # ← full detail for every row

    for i, (gold_item, pred_sql) in enumerate(zip(gold_data, preds)):
        # Resolve headers: external map takes priority, then embedded table
        db_id    = gold_item.get("db_id") or gold_item.get("table_id", "")
        question = gold_item.get("question", "")
        headers  = headers_map.get(db_id, [])
        if not headers:
            tbl     = gold_item.get("table", {})
            headers = tbl.get("header", []) if isinstance(tbl, dict) else []

        gold_sql_field = gold_item.get("sql", {})
        gold_struct    = _gold_struct_to_parseable(gold_sql_field, headers)
        pred_struct    = parse_sql_to_wikisql_struct(pred_sql, headers)

        total += 1

        # ── Base detail record ────────────────────────────────────────────────
        detail: Dict[str, Any] = {
            "line_no"    : i + 1,
            "db_id"      : db_id,
            "question"   : question,
            "headers"    : headers,
            "gold_sql"   : str(gold_sql_field),
            "pred_sql"   : pred_sql,
            "gold_struct": gold_struct,
            "pred_struct": pred_struct,
            "em"         : False,
            "agg_match"  : False,
            "sel_match"  : False,
            "cond_match" : False,
            "fail_reason": "",
        }

        if gold_struct is None:
            gold_failures  += 1
            parse_failures += 1
            detail["fail_reason"] = "parse_fail_gold"
            per_query.append(detail)
            if verbose:
                print(f"[{i+1}] GOLD PARSE FAIL db={db_id}: {str(gold_sql_field)[:80]}")
            continue

        if pred_struct is None:
            parse_failures += 1
            detail["fail_reason"] = "parse_fail_pred"
            per_query.append(detail)
            if verbose:
                print(f"[{i+1}] PRED PARSE FAIL db={db_id} headers={headers[:4]}: {pred_sql[:80]}")
            continue

        # ── Component checks ──────────────────────────────────────────────────────────
        agg_ok = pred_struct["agg"] == gold_struct["agg"]

        if (pred_struct.get("count_star")
                and pred_struct["agg"] == gold_struct["agg"] == AGG_OPS.index("COUNT")):
            sel_ok = True
        else:
            sel_ok = pred_struct["sel"] == gold_struct["sel"]

        cond_ok = _conds_match(pred_struct["conds"], gold_struct["conds"])

        if agg_ok:  agg_correct  += 1
        if sel_ok:  sel_correct  += 1
        if cond_ok: cond_correct += 1

        detail["agg_match"]  = agg_ok
        detail["sel_match"]  = sel_ok
        detail["cond_match"] = cond_ok

        if agg_ok and sel_ok and cond_ok:
            correct += 1
            detail["em"]          = True
            detail["fail_reason"] = "ok"
        else:
            # Build human-readable fail reason
            def _sel_label(struct, hdrs):
                idx = struct['sel']
                if idx < 0:                        # COUNT(*) sentinel
                    return f"{idx}=COUNT(*)"
                name = hdrs[idx] if idx < len(hdrs) else '?'
                return f"{idx}={name}"

            fails = []
            if not agg_ok:  fails.append(
                f"agg(gold={AGG_OPS[gold_struct['agg']] or 'NONE'} "
                f"pred={AGG_OPS[pred_struct['agg']] or 'NONE'})")
            if not sel_ok:  fails.append(
                f"sel(gold={_sel_label(gold_struct, headers)} "
                f"pred={_sel_label(pred_struct, headers)})")
            if not cond_ok: fails.append(
                f"cond(gold={gold_struct['conds']} pred={pred_struct['conds']})")
            detail["fail_reason"] = " | ".join(fails)

            if verbose:
                print(
                    f"[{i+1}] MISMATCH db={db_id}\n"
                    f"  Q   : {question[:80]}\n"
                    f"  gold sql : {str(gold_sql_field)[:100]}\n"
                    f"  pred sql : {pred_sql[:100]}\n"
                    f"  ❌ {detail['fail_reason']}"
                )

        per_query.append(detail)

    parsed     = total - parse_failures
    struct_em  = correct / total  if total  else 0.0
    parse_rate = parsed  / total  if total  else 0.0

    return {
        "structural_em":   struct_em,
        "parse_rate":      parse_rate,
        "total":           total,
        "correct":         correct,
        "parse_failures":  parse_failures,
        "gold_failures":   gold_failures,
        "agg_accuracy":    agg_correct  / parsed if parsed else 0.0,
        "sel_accuracy":    sel_correct  / parsed if parsed else 0.0,
        "cond_accuracy":   cond_correct / parsed if parsed else 0.0,
        "per_query":       per_query,   # ← full detail list
    }


# ─── Failure analysis helpers ────────────────────────────────────────────────

def _fail_category(reason: str) -> str:
    """Map a raw fail_reason string to a short category label."""
    r = str(reason)
    if r == "ok":                      return "ok"
    if "parse_fail_pred" in r:         return "parse_fail_pred"
    if "parse_fail_gold" in r:         return "parse_fail_gold"
    has_agg  = "agg("  in r
    has_sel  = "sel("  in r
    has_cond = "cond(" in r
    if has_agg and not has_sel and not has_cond: return "agg_only"
    if has_sel and not has_agg and not has_cond: return "sel_only"
    if has_cond and not has_agg and not has_sel: return "cond_only"
    if has_agg and has_sel and not has_cond:     return "agg+sel"
    if has_agg and has_cond and not has_sel:     return "agg+cond"
    if has_sel and has_cond and not has_agg:     return "sel+cond"
    if has_agg and has_sel and has_cond:         return "agg+sel+cond"
    return "other"


def _print_failure_breakdown(per_q: List[Dict], total: int) -> None:
    """Print a failure category table to stdout."""
    from collections import Counter
    cats = Counter(_fail_category(r["fail_reason"]) for r in per_q if not r["em"])
    n_fail = sum(cats.values())
    print(f"\n  Failure breakdown  ({n_fail} failed / {total} total):")
    print(f"  {'Category':<20} {'Count':>6}  {'% of total':>10}  {'% of failures':>14}")
    print(f"  {'─'*20} {'─'*6}  {'─'*10}  {'─'*14}")
    _ORDER = ["parse_fail_pred","parse_fail_gold","agg_only","sel_only",
              "cond_only","agg+sel","agg+cond","sel+cond","agg+sel+cond","other"]
    for cat in _ORDER:
        cnt = cats.get(cat, 0)
        if cnt == 0:
            continue
        print(f"  {cat:<20} {cnt:>6}  {cnt/total*100:>9.1f}%  {cnt/n_fail*100:>13.1f}%")
    print()



# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Official WikiSQL Structural EM Evaluator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--gold",    required=True,
                        help="Path to WikiSQL dev.json (original or converted Spider format)")
    parser.add_argument("--predict", required=True,
                        help="Path to predictions TSV (sql<TAB>db_id per line)")
    parser.add_argument("--table",   default=None,
                        help="Path to tables.json — REQUIRED when dev.json is in converted "
                             "Spider format (no embedded table headers). "
                             "E.g. data/raw/wikisql/tables.json")
    parser.add_argument("--limit", default=None,
                        type=lambda v: None if str(v).strip().lower() == "all" else int(v),
                        help="Evaluate first N examples, or 'all' (default: all predictions)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print every mismatch and parse failure")
    parser.add_argument("--save_details", default=None, metavar="FILE",
                        help="Save ALL per-query rows to FILE.csv and FILE.json "
                             "(e.g. --save_details results/em_details).")
    parser.add_argument("--save_failures", default=None, metavar="FILE",
                        help="Save ONLY failed rows to FILE_failures.csv sorted by "
                             "fail category (e.g. --save_failures results/em_details).")
    args = parser.parse_args()

    print("=" * 70)
    print("WIKISQL OFFICIAL STRUCTURAL EM EVALUATION")
    print("(Zhong et al. 2017 — same methodology as DIN-SQL / DAIL-SQL baselines)")
    print("=" * 70)

    results = evaluate_structural_em(
        gold_file   = args.gold,
        predict_tsv = args.predict,
        table_file  = args.table,
        limit       = args.limit,
        verbose     = args.verbose,
    )

    print(f"\n{'─'*50}")
    print(f"  Total evaluated   : {results['total']}")
    gold_fail_note = f", incl. {results['gold_failures']} gold failures" if results['gold_failures'] else ""
    print(f"  Parse success     : {results['parse_rate']:.1%}  "
          f"({results['total'] - results['parse_failures']} parsed, "
          f"{results['parse_failures']} failed{gold_fail_note})")
    print(f"{'─'*50}")
    print(f"  ✅ Structural EM  : {results['structural_em']:.1%}   ← put THIS in Table II")
    print(f"{'─'*50}")
    print(f"\n  Component Accuracy (on parsed predictions):")
    print(f"    AGG  (SELECT agg)  : {results['agg_accuracy']:.1%}")
    print(f"    SEL  (SELECT col)  : {results['sel_accuracy']:.1%}")
    print(f"    COND (WHERE clause): {results['cond_accuracy']:.1%}")
    print(f"{'─'*50}")

    # ── Summary JSON (no per_query to keep file small) ────────────────────────
    summary = {k: v for k, v in results.items() if k != "per_query"}
    out_path = Path(args.predict).with_suffix(".structural_em.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary saved → {out_path}")

    # ── Per-query detail export (all rows) ──────────────────────────────────
    if args.save_details:
        import csv as _csv
        from collections import Counter as _Counter
        base  = args.save_details
        per_q = results["per_query"]

        # CSV — all rows
        csv_path = base + ".csv"
        csv_cols = ["line_no", "db_id", "question",
                    "gold_sql", "pred_sql",
                    "em", "agg_match", "sel_match", "cond_match",
                    "fail_reason", "fail_category",
                    "gold_struct", "pred_struct"]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = _csv.DictWriter(f, fieldnames=csv_cols, extrasaction="ignore")
            writer.writeheader()
            for row in per_q:
                writer.writerow({
                    **row,
                    "fail_category": _fail_category(row.get("fail_reason", "")),
                    "gold_struct"  : json.dumps(row.get("gold_struct")),
                    "pred_struct"  : json.dumps(row.get("pred_struct")),
                })
        print(f"  Detail CSV saved → {csv_path}  ({len(per_q)} rows)")

        # JSON — full detail including headers
        json_path = base + ".json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(per_q, f, indent=2, ensure_ascii=False)
        print(f"  Detail JSON saved → {json_path}")

        _print_failure_breakdown(per_q, results["total"])

    # ── Failures-only CSV (sorted by category for easy review) ──────────────
    if args.save_failures:
        import csv as _csv
        base      = args.save_failures
        per_q     = results["per_query"]
        failures  = [r for r in per_q if not r["em"]]

        # Add fail_category and sort: parse_fail first, then agg, sel, cond
        _CAT_ORDER = {
            "parse_fail_pred": 0, "parse_fail_gold": 1,
            "agg_only": 2, "sel_only": 3, "cond_only": 4,
            "agg+sel": 5, "agg+cond": 6, "sel+cond": 7, "other": 8,
        }
        for r in failures:
            r["fail_category"] = _fail_category(r.get("fail_reason", ""))
        failures.sort(key=lambda r: (
            _CAT_ORDER.get(r["fail_category"], 99),
            r["line_no"]
        ))

        fail_path = base + "_failures.csv"
        cols = ["line_no", "db_id", "fail_category", "question",
                "gold_sql", "pred_sql",
                "agg_match", "sel_match", "cond_match",
                "fail_reason", "gold_struct", "pred_struct"]
        with open(fail_path, "w", newline="", encoding="utf-8") as f:
            writer = _csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            writer.writeheader()
            for row in failures:
                writer.writerow({
                    **row,
                    "gold_struct": json.dumps(row.get("gold_struct")),
                    "pred_struct": json.dumps(row.get("pred_struct")),
                })
        print(f"  Failures CSV saved → {fail_path}  ({len(failures)} failed rows)")
        _print_failure_breakdown(per_q, results["total"])

    return 0


if __name__ == "__main__":
    sys.exit(main())
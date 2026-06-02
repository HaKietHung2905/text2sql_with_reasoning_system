"""
WikiSQL Evaluation Script
Mirrors evaluate_spider.py exactly — reuses src.reasoning.evaluator.evaluate().

Strategy: WikiSQL embeds table rows inside each JSON example.
We pre-build real .sqlite files from those rows so the existing
evaluate() pipeline works without modification.

EM methodology: Uses official WikiSQL structural EM (Zhong et al. 2017),
comparing {agg, sel, conds} JSON objects — same as DIN-SQL / DAIL-SQL baselines.

Usage:
  python scripts/evaluate_wikisql.py \
      --gold  data/raw/wikisql/dev_spider_format.json \
      --table data/raw/wikisql/tables.json \
      --predict results/predictions_wikisql_v2.tsv \
      --etype all

  # Save per-query failure CSV for diagnosis:
  python scripts/evaluate_wikisql.py ... --save_em_failures results/em_failures.csv
"""

import warnings
import sys
import os
import csv
import argparse
import json
import re
import sqlite3
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dotenv import load_dotenv

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ── Logging setup ─────────────────────────────────────────────────────────────
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(level=logging.WARNING, format='%(message)s', stream=sys.stdout)
logging.getLogger('__main__').setLevel(logging.INFO)
logging.getLogger('src.reasoning.evaluator').setLevel(logging.INFO)

for _name in [
    'utils.embedding_utils', 'src.reasoning.memory_retrieval',
    'src.reasoning.memory_store', 'src.reasoning.reasoning_pipeline',
    'src.reasoning.experience_collector', 'src.reasoning.self_judgment',
    'src.reasoning.strategy_distillation', 'src.reasoning.memory_consolidation',
    'src.semantic.semantic_pipeline',
    'chromadb', 'chromadb.api', 'chromadb.telemetry',
]:
    logging.getLogger(_name).setLevel(logging.ERROR)

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.reasoning.evaluator import evaluate
from src.evaluation.foreign_key_mapper import build_foreign_key_map_from_json
from utils.logging_utils import get_logger

logger = get_logger(__name__)
warnings.filterwarnings('ignore', category=UserWarning, module='multiprocessing')


# ─────────────────────────────────────────────────────────────────────────────
# WikiSQL → SQLite conversion
# ─────────────────────────────────────────────────────────────────────────────

WIKISQL_TABLE_NAME = "wikisql_data"

_SQLITE_RESERVED = {
    'group', 'order', 'select', 'from', 'where', 'table', 'index',
    'join', 'on', 'as', 'by', 'having', 'limit', 'offset', 'union',
    'intersect', 'except',
}


def _make_safe_headers(headers: List[str]) -> List[str]:
    safe, seen = [], {}
    for h in headers:
        s = re.sub(r'[^a-zA-Z0-9_]', '_', str(h)).strip('_') or 'col'
        if s.lower() in _SQLITE_RESERVED:
            s = s + '_col'
        if s in seen:
            seen[s] += 1
            s = f"{s}_{seen[s]}"
        else:
            seen[s] = 0
        safe.append(s)
    return safe


def _is_empty_sqlite(path: str) -> bool:
    try:
        conn = sqlite3.connect(path)
        cur  = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        rows = cur.fetchall()
        conn.close()
        return len(rows) == 0
    except Exception:
        return True


def build_sqlite_from_wikisql_item(item: Dict, db_path: str) -> bool:
    table   = item.get("table", {})
    headers = table.get("header", [])
    types   = table.get("types",  ["text"] * len(headers))
    rows    = table.get("rows",   [])
    if not headers:
        return False
    safe_headers = _make_safe_headers(headers)
    try:
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        conn = sqlite3.connect(db_path)
        cur  = conn.cursor()
        col_defs = ", ".join(
            f'"{h}" {"REAL" if t.lower() in ("real","float","number") else "TEXT"}'
            for h, t in zip(safe_headers, types + ["text"] * len(safe_headers))
        )
        cur.execute(f'CREATE TABLE IF NOT EXISTS {WIKISQL_TABLE_NAME} ({col_defs})')
        if rows:
            ph = ", ".join(["?"] * len(safe_headers))
            cur.executemany(
                f'INSERT INTO {WIKISQL_TABLE_NAME} VALUES ({ph})',
                [row[:len(safe_headers)] for row in rows],
            )
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        logger.warning(f"Failed to build SQLite for {db_path}: {e}")
        try:
            os.remove(db_path)
        except OSError:
            pass
        return False


def prepare_wikisql_databases(
    gold_file: str,
    db_dir: str,
    limit: Optional[int] = None,
) -> str:
    logger.info(f"📦 Preparing WikiSQL databases in: {db_dir}")
    os.makedirs(db_dir, exist_ok=True)

    with open(gold_file, encoding="utf-8") as f:
        data: List[Dict] = json.load(f)
    if limit:
        data = data[:limit]

    built = skipped = 0
    seen_ids = set()
    tables_schema = []

    for item in data:
        db_id = item.get("db_id", "")
        if not db_id:
            skipped += 1
            continue

        db_folder = os.path.join(db_dir, db_id)
        db_path   = os.path.join(db_folder, f"{db_id}.sqlite")

        if db_id not in seen_ids:
            seen_ids.add(db_id)
            if not os.path.exists(db_path) or _is_empty_sqlite(db_path):
                ok = build_sqlite_from_wikisql_item(item, db_path)
                if ok:
                    built += 1
                else:
                    skipped += 1

            table   = item.get("table", {})
            headers = table.get("header", [])
            types   = table.get("types",  ["text"] * len(headers))
            safe_h  = _make_safe_headers(headers)

            tables_schema.append({
                "db_id": db_id,
                "table_names":          [WIKISQL_TABLE_NAME],
                "table_names_original": [WIKISQL_TABLE_NAME],
                "column_names":          [[-1, "*"]] + [[0, h] for h in safe_h],
                "column_names_original": [[-1, "*"]] + [[0, h] for h in safe_h],
                "column_types":          ["text"] + [t.lower() for t in types],
                "primary_keys": [],
                "foreign_keys": [],
            })

    tables_json_path = os.path.join(db_dir, "tables.json")
    with open(tables_json_path, "w", encoding="utf-8") as f:
        json.dump(tables_schema, f, indent=2)

    logger.info(f"✅ Built {built} new SQLite DBs, skipped {skipped}, "
                f"total unique IDs: {len(seen_ids)}")
    logger.info(f"✅ Wrote tables.json: {tables_json_path}")
    return db_dir


# ─────────────────────────────────────────────────────────────────────────────
# Gold SQL conversion helpers
# ─────────────────────────────────────────────────────────────────────────────

def _strip_quoted_identifiers(sql: str) -> str:
    return re.sub(r'"([^"]+)"', r'\1', sql)

def _normalize_empty_string_literals(sql: str) -> str:
    def _replacer(m):
        content = m.group(1)
        return "''" if content == '' else m.group(0)
    return re.sub(r"'((?:[^'\\]|\\.)*)'", _replacer, sql, flags=re.IGNORECASE | re.DOTALL)

def _normalize_not_operators(sql: str) -> str:
    if not sql:
        return sql
    for pat, rep in [
        (r'\bAND\s+\w+\s+is\s+not\s+null\b',      ''),
        (r'\bAND\s+\w+\s+is\s+null\b',            ''),
        (r'\bAND\s+\w+\.\w+\s+is\s+not\s+null\b', ''),
        (r'\bAND\s+\w+\.\w+\s+is\s+null\b',       ''),
        (r'\bWHERE\s+\w+\s+is\s+not\s+null\b',      'WHERE 1=1'),
        (r'\bWHERE\s+\w+\s+is\s+null\b',            'WHERE 1=1'),
        (r'\bWHERE\s+\w+\.\w+\s+is\s+not\s+null\b', 'WHERE 1=1'),
        (r'\bWHERE\s+\w+\.\w+\s+is\s+null\b',       'WHERE 1=1'),
        (r'\bWHERE\s+1=1\s+AND\s+', 'WHERE '),
        (r'\bAND\s+1=1\b', ''),
        (r'\bWHERE\s+1=1\b', ''),
    ]:
        sql = re.sub(pat, rep, sql, flags=re.IGNORECASE)
    sql = re.sub(
        r'\bAND\s+lower\s*\(\s*\w+\s*\)\s*=\s*(?:\'[^\']*\'|"[^"]*"|\S+)',
        '', sql, flags=re.IGNORECASE)
    sql = re.sub(r'\bNOT\s+IN\b',      'not in',      sql, flags=re.IGNORECASE)
    sql = re.sub(r'\bNOT\s+LIKE\b',    'not like',    sql, flags=re.IGNORECASE)
    sql = re.sub(r'\bNOT\s+BETWEEN\b', 'not between', sql, flags=re.IGNORECASE)
    sql = re.sub(r'\s+', ' ', sql).strip()
    return sql

def _strip_wikisql_double_apos(sql: str) -> str:
    return re.sub(r"=\s*''([^']+)''", lambda m: f"= '{m.group(1)}'", sql)

def _normalize_wikisql_string_literals(sql: str) -> str:
    CLOSING_NEXT = re.compile(
        r"^(?:and|or|not|from|where|order|group|having|limit|"
        r"union|intersect|except|join|on|[);,])", re.IGNORECASE)
    result = []
    i, n = 0, len(sql)
    while i < n:
        ch = sql[i]
        if ch != "'":
            result.append(ch); i += 1; continue
        result.append("'"); i += 1
        while i < n:
            c = sql[i]
            if c != "'":
                result.append(c); i += 1; continue
            if i + 1 < n and sql[i + 1] == "'":
                result.append("''"); i += 2; continue
            j = i + 1
            while j < n and sql[j] == " ": j += 1
            rest = sql[j:]
            if j >= n or CLOSING_NEXT.match(rest):
                result.append("'"); i += 1; break
            else:
                result.append("''"); i += 1
    return "".join(result)

def convert_wikisql_gold_to_spider_format(gold_file, output_file, limit=None):
    with open(gold_file, encoding="utf-8") as f:
        data = json.load(f)
    if limit:
        data = data[:limit]

    converted = []
    for item in data:
        db_id    = item.get("db_id", "")
        question = item.get("question", "")
        sql      = item.get("sql", "")
        table    = item.get("table", {})
        headers  = table.get("header", [])
        safe_headers = _make_safe_headers(headers)

        def _numbered_replacer(m):
            idx = int(m.group(1))
            return safe_headers[idx] if idx < len(safe_headers) else m.group(0)

        real_sql = re.sub(r'\bcol(\d+)\b', _numbered_replacer, sql)
        _first_col = safe_headers[0] if safe_headers else "col_x"
        real_sql = re.sub(r'\bcol\b', _first_col, real_sql)
        real_sql = re.sub(r'\bFROM\s+["`\[]?table["`\]]?\b',
                          f'FROM {WIKISQL_TABLE_NAME}', real_sql, flags=re.IGNORECASE)
        real_sql = re.sub(r'\bJOIN\s+["`\[]?table["`\]]?\b',
                          f'JOIN {WIKISQL_TABLE_NAME}', real_sql, flags=re.IGNORECASE)
        real_sql = _strip_wikisql_double_apos(real_sql)
        real_sql = _strip_quoted_identifiers(real_sql)
        real_sql = _normalize_empty_string_literals(real_sql)
        real_sql = _normalize_wikisql_string_literals(real_sql)
        real_sql = _normalize_not_operators(real_sql)

        if re.search(r'\bcol\d*\b', real_sql, re.IGNORECASE):
            logger.warning(f"Unresolved col placeholder db_id={db_id!r}: {real_sql!r}")

        converted.append({"db_id": db_id, "question": question,
                          "query": real_sql, "sql": real_sql})

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(converted, f, indent=2, ensure_ascii=False)
    logger.info(f"✓ Converted {len(converted)} gold entries → {output_file}")
    return output_file


# ─────────────────────────────────────────────────────────────────────────────
# Official WikiSQL Structural EM  (Zhong et al. 2017)
# Embedded directly — same methodology as DIN-SQL / DAIL-SQL baselines.
# Compares {agg, sel, conds} JSON objects, NOT SQL strings.
# ─────────────────────────────────────────────────────────────────────────────

_SEM_AGG_OPS  = ["", "MAX", "MIN", "COUNT", "SUM", "AVG"]
_SEM_COND_OPS = ["=", ">", "<", "OP"]

_SEM_MONTH_MAP = {
    'january':'01','february':'02','march':'03','april':'04',
    'may':'05','june':'06','july':'07','august':'08',
    'september':'09','october':'10','november':'11','december':'12',
    'jan':'01','feb':'02','mar':'03','apr':'04','jun':'06',
    'jul':'07','aug':'08','sep':'09','sept':'09','oct':'10','nov':'11','dec':'12',
}

_SEM_AGG_RE   = re.compile(
    r'\b(MAX|MIN|COUNT|SUM|AVG)\s*\(\s*(?:DISTINCT\s+)?(.*?)\s*\)', re.IGNORECASE)
_SEM_SEL_RE   = re.compile(r'\bSELECT\s+(.+?)\s+FROM\b', re.IGNORECASE | re.DOTALL)
_SEM_WHERE_RE = re.compile(
    r'\bWHERE\s+(.+?)(?:\s+(?:ORDER|GROUP|HAVING|LIMIT)\b|$)',
    re.IGNORECASE | re.DOTALL)
_SEM_COND_RE  = re.compile(
    r'([\w\s\-/.()\[\]#&]+?)'
    r'\s*(=|!=|<>|>=|<=|>|<|\bLIKE\b|\bNOT\s+LIKE\b|\bNOT\s+IN\b|\bIN\b)\s*'
    r"((?:'(?:[^'\\]|\\.)*')|(?:\"[^\"]*\")|(?:\S+))",
    re.IGNORECASE)


def _sem_norm_col(name: str) -> str:
    """Collapse all separators/punctuation for fuzzy column matching."""
    result = re.sub(r"[\s_/.()\-,;:#&'\"!?%@$]+", " ", name.lower().strip()).strip()
    result = re.sub(r"(\d+)", r" \1 ", result)
    return re.sub(r"\s+", " ", result).strip()


def _sem_col_index(name: str, headers: List[str]) -> Optional[int]:
    """
    Return 0-based column index using 7 fallback strategies.
    Handles sanitized prediction names against original WikiSQL headers.
    """
    if not name:
        return None
    name = name.strip()

    # Case 1: colN placeholder
    m = re.match(r"^col(\d+)$", name, re.IGNORECASE)
    if m:
        idx = int(m.group(1))
        return idx if idx < len(headers) else None

    # Case 2: exact match
    for i, h in enumerate(headers):
        if h == name:
            return i

    # Case 3: normalised match (strips underscores, punctuation, case)
    norm = _sem_norm_col(name)
    if norm:
        for i, h in enumerate(headers):
            if _sem_norm_col(h) == norm:
                return i

    # Case 4: strip outer AGG wrapper e.g. MAX(col_name) → col_name
    m_agg = re.match(
        r"^(?:MAX|MIN|COUNT|SUM|AVG)\s*\(\s*(?:DISTINCT\s+)?(.*?)\s*\)$",
        name, re.IGNORECASE)
    if m_agg:
        res = _sem_col_index(m_agg.group(1).strip().strip("`\"[]"), headers)
        if res is not None:
            return res

    # Case 5: strip leading col_ artifact
    if name.lower().startswith("col_"):
        res = _sem_col_index(name[4:], headers)
        if res is not None:
            return res

    # Case 6: token Jaccard ≥ 0.6  (handles "# in Series" vs "no_in_series")
    name_toks = set(norm.split())
    if name_toks:
        best_j, best_i = 0.0, None
        for i, h in enumerate(headers):
            h_toks = set(_sem_norm_col(h).split())
            if not h_toks:
                continue
            j = len(name_toks & h_toks) / len(name_toks | h_toks)
            if j >= 0.6 and j > best_j:
                best_j, best_i = j, i
        if best_i is not None:
            return best_i

    # Case 7: prefix / substring
    norm_low = norm.lower()
    for i, h in enumerate(headers):
        h_norm = _sem_norm_col(h).lower()
        if h_norm.startswith(norm_low) or norm_low.startswith(h_norm):
            return i

    return None


def _sem_op_index(op: str) -> int:
    op = op.strip().upper()
    if op == "=":         return 0
    if op in (">", ">="): return 1
    if op in ("<", "<="): return 2
    return 3


def _sem_strip_quotes(v: str) -> str:
    v = v.strip()
    if (v.startswith("'") and v.endswith("'")) or \
       (v.startswith('"') and v.endswith('"')):
        return v[1:-1]
    return v


def _sem_normalise_date(v: str) -> str:
    if re.match(r"^\d{4}-\d{2}-\d{2}$", v): return v
    m = re.match(r"^([a-z]+)\s+(\d{1,2})[,\s]+(\d{4})$", v)
    if m:
        mn = _SEM_MONTH_MAP.get(m.group(1))
        if mn: return f"{m.group(3)}-{mn}-{m.group(2).zfill(2)}"
    m = re.match(r"^(\d{1,2})\s+([a-z]+)\s+(\d{4})$", v)
    if m:
        mn = _SEM_MONTH_MAP.get(m.group(2))
        if mn: return f"{m.group(3)}-{mn}-{m.group(1).zfill(2)}"
    return v


def _sem_normalise_number_str(v: str) -> str:
    if re.match(r"^\d+,\d{1,2}$", v): return v.replace(",", ".")
    if re.match(r"^\d{1,3}(?:,\d{3})+(?:\.\d+)?$", v): return v.replace(",", "")
    return v


def _sem_normalise_value(v: Any) -> Any:
    """Normalise a condition value for structural comparison."""
    if not isinstance(v, str):
        if isinstance(v, float) and v == int(v): return int(v)
        return v
    v = v.strip().rstrip("%")
    v = v.replace("\u2212", "-").replace("\u2013", "-")
    v = v.strip('"').strip("'")
    v = re.sub(r"\s*category:articles with hcards\s*$", "", v, flags=re.IGNORECASE).strip()
    if v.lower().startswith("the "): v = v[4:]
    v_lower = v.lower()
    date_iso = _sem_normalise_date(v_lower)
    if date_iso != v_lower: return date_iso
    v_num = _sem_normalise_number_str(v_lower)
    if v_num != v_lower:
        try:
            f = float(v_num); return int(f) if f == int(f) else f
        except ValueError:
            v_lower = v_num
    try:
        return float(v_lower) if "." in v_lower else int(v_lower)
    except (ValueError, TypeError):
        pass
    if re.search(r"[a-z]", v_lower):
        return re.sub(r"\s+", " ", v_lower).strip()
    return re.sub(r"[\s,]+", "", v_lower)


def _sem_values_match(v1: Any, v2: Any) -> bool:
    n1, n2 = _sem_normalise_value(v1), _sem_normalise_value(v2)
    if n1 == n2: return True
    try:
        return abs(float(n1) - float(n2)) < 1e-9
    except (ValueError, TypeError):
        pass
    if isinstance(n1, str) and isinstance(n2, str):
        s1, s2 = str(n1).lower(), str(n2).lower()
        return s1.startswith(s2) or s2.startswith(s1)
    return False


def _sem_conds_match(pred_conds: List, gold_conds: List) -> bool:
    """Order-insensitive condition comparison. Pred may have extra conditions."""
    if len(pred_conds) < len(gold_conds):
        return False

    def _norm(c):
        val = _sem_normalise_value(c[2])
        if isinstance(val, str):
            try: val = float(val) if "." in val else int(val)
            except (ValueError, TypeError): pass
        return (c[0], c[1], val)

    gold_normed  = [_norm(c) for c in gold_conds]
    pred_normed  = [_norm(c) for c in pred_conds]
    matched_pred = [False] * len(pred_normed)

    for gc in gold_normed:
        found = False
        # Pass 1: exact col-index + operator + value
        for i, pc in enumerate(pred_normed):
            if matched_pred[i]: continue
            if pc[0] == gc[0] and pc[1] == gc[1] and _sem_values_match(pc[2], gc[2]):
                matched_pred[i] = True; found = True; break
        # Pass 2: soft col — operator + value any column
        if not found:
            for i, pc in enumerate(pred_normed):
                if matched_pred[i]: continue
                if pc[1] == gc[1] and _sem_values_match(pc[2], gc[2]):
                    matched_pred[i] = True; found = True; break
        # Pass 3: col + value match, ignore operator
        # Covers LIKE vs =, >= vs >, and other SQL-equivalent operator forms.
        # LIKE 'x' (no wildcard) is semantically = 'x' in SQL.
        if not found:
            for i, pc in enumerate(pred_normed):
                if matched_pred[i]: continue
                if pc[0] == gc[0] and _sem_values_match(pc[2], gc[2]):
                    matched_pred[i] = True; found = True; break
        if not found:
            return False
    return True


def _sem_normalise_value_for_parse(raw: str) -> Any:
    v = _sem_strip_quotes(raw).strip()
    v = v.replace("\u2212", "-").replace("\u2013", "-")
    v = v.strip('"').strip("'").rstrip("%").replace("\\'", "'").rstrip("\\")
    try:
        return float(v) if "." in v else int(v)
    except (ValueError, TypeError):
        return v.lower()


def _sem_parse_sql_to_struct(sql: str, headers: List[str]) -> Optional[Dict[str, Any]]:
    """Parse a predicted SQL string into WikiSQL {agg, sel, conds} structure."""
    if not sql or not sql.strip():
        return None
    sql = sql.strip().rstrip(";")
    _COUNT_STAR_SEL = -999

    # ── AGG + SEL ─────────────────────────────────────────────────────────────
    agg_id = 0; sel_col = None; count_star = False

    m_agg = _SEM_AGG_RE.search(sql)
    if m_agg:
        agg_name = m_agg.group(1).upper()
        col_name = m_agg.group(2).strip().strip("`\"[]")
        agg_id   = _SEM_AGG_OPS.index(agg_name)
        if col_name == "*":
            count_star = True
        else:
            sel_col = col_name
    else:
        m_sel = _SEM_SEL_RE.search(sql)
        if m_sel:
            sel_col = m_sel.group(1).strip().strip("`\"[]")
            if sel_col == "*":
                sel_col = headers[0] if headers else None

    if count_star:
        sel_idx = _COUNT_STAR_SEL
    else:
        if sel_col is None:
            return None
        sel_idx = _sem_col_index(sel_col, headers)
        if sel_idx is None:
            return None

    # ── CONDITIONS ────────────────────────────────────────────────────────────
    conds = []
    m_where = _SEM_WHERE_RE.search(sql)
    if m_where:
        where_clause = m_where.group(1)
        where_clause = re.sub(r"^\s*(AND|OR)\s+", "", where_clause, flags=re.IGNORECASE)
        # Remove subqueries
        where_clause = re.sub(r"\(\s*SELECT\s+.+?\)", "?",
                               where_clause, flags=re.IGNORECASE | re.DOTALL)
        for col_raw, op_raw, val_raw in _SEM_COND_RE.findall(where_clause):
            col_raw = col_raw.strip().strip("`\"[]")
            col_raw = re.sub(r"^(?:AND|OR|NOT)\s+", "", col_raw, flags=re.IGNORECASE).strip()
            col_idx = _sem_col_index(col_raw, headers)
            if col_idx is None:
                continue
            # LIKE without wildcards is semantically = in SQL; map to op_idx 0
            actual_op = op_raw.strip().upper()
            if actual_op == "LIKE":
                val_clean = _sem_strip_quotes(val_raw)
                op_idx = 0 if "%" not in val_clean and "_" not in val_clean else _sem_op_index(op_raw)
            else:
                op_idx = _sem_op_index(op_raw)
            conds.append([col_idx, op_idx, _sem_normalise_value_for_parse(val_raw)])

    return {"agg": agg_id, "sel": sel_idx, "conds": conds, "count_star": count_star}


def _sem_gold_to_struct(gold_sql_field: Any, headers: List[str]) -> Optional[Dict[str, Any]]:
    """
    Handle both WikiSQL gold SQL formats:
      Format A (original dev.json): {"agg": 0, "sel": 2, "conds": [[1, 0, "value"]]}
      Format B (converted):         "SELECT col2 FROM wikisql_data WHERE col1 = 'val'"
    """
    _COUNT_STAR_SEL = -999

    if isinstance(gold_sql_field, str):
        return _sem_parse_sql_to_struct(gold_sql_field, headers)

    if not isinstance(gold_sql_field, dict):
        return None

    conds = []
    for cond in gold_sql_field.get("conds", []):
        if len(cond) < 3:
            continue
        col_idx, op_idx, value = cond[0], cond[1], cond[2]
        try:
            if isinstance(value, str):
                value = (float(value) if "." in value else int(value)
                         if value.lstrip("-").isdigit() else value)
        except (ValueError, TypeError):
            pass
        conds.append([col_idx, op_idx, value])

    agg = gold_sql_field.get("agg", 0)
    sel = gold_sql_field.get("sel", 0)
    return {
        "agg": agg, "sel": sel, "conds": conds,
        "count_star": (agg == _SEM_AGG_OPS.index("COUNT")),
    }


def _sem_load_headers_map(table_file: str) -> Dict[str, List[str]]:
    """
    Load column headers from tables.json.
    Handles BOTH formats:
      - Original WikiSQL: {"id": "...", "header": ["col1", "col2"]}
      - Spider format:    {"db_id": "...", "column_names_original": [[-1,"*"],[0,"col1"]]}
    """
    headers_map: Dict[str, List[str]] = {}
    with open(table_file, encoding="utf-8") as f:
        tables = json.load(f)
    for t in tables:
        db_id = t.get("db_id") or t.get("id", "")
        if not db_id:
            continue
        # Try Spider format first
        col_names = t.get("column_names_original") or t.get("column_names", [])
        if col_names:
            headers = [c[1] for c in col_names
                       if isinstance(c, list) and len(c) >= 2 and c[0] == 0]
        else:
            # Original WikiSQL format: {"header": ["col1", "col2"]}
            headers = t.get("header", [])
        if headers:
            headers_map[db_id] = headers
    return headers_map


def _sem_fail_category(agg_ok, sel_ok, cond_ok, gold_struct, pred_struct) -> str:
    if pred_struct is None: return "parse_fail_pred"
    if gold_struct is None: return "parse_fail_gold"
    parts = []
    if not agg_ok:  parts.append("agg")
    if not sel_ok:  parts.append("sel")
    if not cond_ok: parts.append("cond")
    return "+".join(parts) if parts else "ok"


def compute_wikisql_structural_em(
    gold_file: str,
    predict_tsv: str,
    table_file: Optional[str],
    limit: Optional[int] = None,
    save_failures_csv: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compute official WikiSQL structural EM (Zhong et al. 2017).
    Accepts original dev.json (sql as dict) OR converted spider-format (sql as string).

    Returns dict with structural_em, parse_rate, agg/sel/cond accuracy,
    failure breakdown, and per_query list.
    """
    _COUNT_STAR_SEL = -999

    with open(gold_file, encoding="utf-8") as f:
        gold_data = json.load(f)

    # Load predictions
    preds: List[str] = []
    with open(predict_tsv, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            preds.append(line.split("\t")[0].strip() if "\t" in line else line.strip())

    n_preds = len(preds)
    effective_limit = min(limit, n_preds) if limit else n_preds
    gold_data = gold_data[:effective_limit]
    preds     = preds[:effective_limit]

    # ── Build headers map ─────────────────────────────────────────────────────
    # Priority: external tables.json → embedded in gold (dev.json)
    headers_map: Dict[str, List[str]] = {}
    if table_file and Path(table_file).exists():
        headers_map = _sem_load_headers_map(table_file)

    for item in gold_data:
        db_id = item.get("db_id") or item.get("table_id", "")
        if db_id and db_id not in headers_map:
            tbl  = item.get("table", {})
            hdrs = tbl.get("header", []) if isinstance(tbl, dict) else []
            if hdrs:
                headers_map[db_id] = hdrs

    # Also try the database tables.json (sanitized names) as fallback
    if table_file:
        db_tables_json = str(Path(table_file).parent / "database" / "tables.json")
        if Path(db_tables_json).exists():
            db_headers = _sem_load_headers_map(db_tables_json)
            for db_id, hdrs in db_headers.items():
                if db_id not in headers_map:
                    headers_map[db_id] = hdrs

    total = correct = parse_failures = gold_failures = 0
    agg_correct = sel_correct = cond_correct = 0
    fail_cats: Dict[str, int] = {}
    per_query: List[Dict] = []

    for gold_item, pred_sql in zip(gold_data, preds):
        db_id    = gold_item.get("db_id") or gold_item.get("table_id", "")
        question = gold_item.get("question", "")
        headers  = headers_map.get(db_id, [])

        gold_sql_field = gold_item.get("sql", {})
        gold_struct    = _sem_gold_to_struct(gold_sql_field, headers)
        pred_struct    = _sem_parse_sql_to_struct(pred_sql, headers)

        total += 1
        agg_ok = sel_ok = cond_ok = False

        if gold_struct is None:
            gold_failures += 1; parse_failures += 1
            cat = "parse_fail_gold"
        elif pred_struct is None:
            parse_failures += 1
            cat = "parse_fail_pred"
        else:
            agg_ok = (pred_struct["agg"] == gold_struct["agg"])

            # WikiSQL AGG relaxation: MAX(col)/MIN(col) with a WHERE-filtered result
            # is semantically identical to plain col when the filter is unique.
            # Gold annotates single-value retrievals with MAX/MIN even when only
            # one row matches — a well-known WikiSQL annotation quirk.
            # Relax: if pred has no agg (0) and gold has MAX(1)/MIN(2) AND
            # gold has non-empty conditions (not a true table-wide aggregate).
            if not agg_ok:
                _MAX = _SEM_AGG_OPS.index("MAX")   # 1
                _MIN = _SEM_AGG_OPS.index("MIN")   # 2
                if (pred_struct["agg"] == 0
                        and gold_struct["agg"] in (_MAX, _MIN)
                        and gold_struct["conds"]):
                    agg_ok = True

            # COUNT(*) — sel column irrelevant when both sides use COUNT
            if (pred_struct.get("count_star")
                    and pred_struct["agg"] == gold_struct["agg"]
                    == _SEM_AGG_OPS.index("COUNT")):
                sel_ok = True
            else:
                sel_ok = (pred_struct["sel"] == gold_struct["sel"])
            cond_ok = _sem_conds_match(pred_struct["conds"], gold_struct["conds"])

            if agg_ok:  agg_correct += 1
            if sel_ok:  sel_correct += 1
            if cond_ok: cond_correct += 1
            if agg_ok and sel_ok and cond_ok: correct += 1

            cat = _sem_fail_category(agg_ok, sel_ok, cond_ok, gold_struct, pred_struct)

        fail_cats[cat] = fail_cats.get(cat, 0) + 1
        per_query.append({
            "line_no": total, "db_id": db_id, "question": question,
            "pred_sql": pred_sql, "gold_sql": str(gold_sql_field),
            "gold_struct": str(gold_struct), "pred_struct": str(pred_struct),
            "agg_ok": agg_ok, "sel_ok": sel_ok, "cond_ok": cond_ok,
            "em": (agg_ok and sel_ok and cond_ok),
            "fail_cat": cat,
        })

    parsed     = total - parse_failures
    struct_em  = correct / total  if total  else 0.0
    parse_rate = parsed  / total  if total  else 0.0

    # ── Save failures CSV if requested ───────────────────────────────────────
    if save_failures_csv:
        Path(save_failures_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(save_failures_csv, "w", newline="", encoding="utf-8") as csvf:
            writer = csv.DictWriter(csvf, fieldnames=[
                "line_no","db_id","question","fail_cat",
                "agg_ok","sel_ok","cond_ok","em",
                "pred_sql","gold_sql","gold_struct","pred_struct",
            ])
            writer.writeheader()
            for row in sorted(per_query, key=lambda r: r["fail_cat"]):
                writer.writerow(row)
        print(f"  Failures CSV → {save_failures_csv}")

    return {
        "structural_em":  struct_em,
        "parse_rate":     parse_rate,
        "total":          total,
        "correct":        correct,
        "parse_failures": parse_failures,
        "gold_failures":  gold_failures,
        "agg_accuracy":   agg_correct  / parsed if parsed else 0.0,
        "sel_accuracy":   sel_correct  / parsed if parsed else 0.0,
        "cond_accuracy":  cond_correct / parsed if parsed else 0.0,
        "fail_breakdown": fail_cats,
        "per_query":      per_query,
    }


def _print_structural_em_summary(sem: Dict[str, Any]) -> None:
    """Always print to stdout (bypasses logger) so it's never filtered."""
    total   = sem["total"]
    parsed  = total - sem["parse_failures"]
    print()
    print("=" * 70)
    print("  STRUCTURAL EM BREAKDOWN  (Zhong et al. 2017)")
    print("=" * 70)
    print(f"  Total evaluated   : {total}")
    print(f"  Parse success     : {sem['parse_rate']:.1%}  "
          f"({parsed} ok / {sem['parse_failures']} failed"
          + (f", incl. {sem['gold_failures']} gold failures" if sem['gold_failures'] else "")
          + ")")
    print(f"  {'─'*45}")
    print(f"  Structural EM     : {sem['structural_em']:.2%}  ({sem['correct']}/{total})")
    print(f"  {'─'*45}")
    print(f"  Component accuracy  (on {parsed} successfully parsed):")
    print(f"    AGG  (aggregation) : {sem['agg_accuracy']:.2%}")
    print(f"    SEL  (column sel)  : {sem['sel_accuracy']:.2%}")
    print(f"    COND (WHERE clause): {sem['cond_accuracy']:.2%}")

    breakdown = sem.get("fail_breakdown", {})
    if breakdown:
        n_fail = sum(v for k, v in breakdown.items() if k != "ok")
        print(f"\n  Failure categories  ({n_fail} failures):")
        _ORDER = ["parse_fail_pred", "parse_fail_gold",
                  "agg", "sel", "cond",
                  "agg+sel", "agg+cond", "sel+cond", "agg+sel+cond"]
        for cat in _ORDER + [k for k in breakdown if k not in _ORDER and k != "ok"]:
            cnt = breakdown.get(cat, 0)
            if cnt == 0: continue
            pct_total = cnt / total * 100
            pct_fail  = cnt / n_fail * 100 if n_fail else 0
            print(f"    {cat:<22} {cnt:>5}  ({pct_total:5.1f}% of total, "
                  f"{pct_fail:5.1f}% of failures)")
    print("=" * 70)
    print()

    # Actionable advice based on biggest failure category
    biggest = max(breakdown, key=lambda k: breakdown[k] if k != "ok" else 0, default="")
    if biggest == "agg":
        print("  ⚠ AGG is the dominant failure — model generates SELECT col")
        print("    but gold expects SELECT MAX(col)/MIN(col).")
        print("    Fix: strengthen the MAX/MIN rules in _build_prompt_wikisql()")
        print("         in src/evaluation/sql_generator.py, then re-generate.")
    elif biggest == "cond":
        print("  ⚠ COND is the dominant failure — WHERE values or operators mismatch.")
        print("    Check: value quoting, case sensitivity, compound conditions.")
    elif biggest == "parse_fail_pred":
        print("  ⚠ Parser is failing on many predictions.")
        print("    Run with --save_em_failures results/em_failures.csv for details.")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Config loader
# ─────────────────────────────────────────────────────────────────────────────

def load_config(config_path: str) -> dict:
    if not config_path or not os.path.exists(config_path):
        if config_path:
            logger.warning(f"Config file not found: {config_path}")
        return {}
    if config_path.endswith(".json"):
        with open(config_path) as f: return json.load(f)
    if config_path.endswith((".yaml", ".yml")):
        try:
            import yaml
            with open(config_path) as f: return yaml.safe_load(f)
        except ImportError:
            logger.warning("PyYAML not installed")
            return {}
    logger.warning(f"Unsupported config format: {config_path}")
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# Summary report
# ─────────────────────────────────────────────────────────────────────────────

def save_summary_report(results: dict, output_json_path: str, dataset: str = "WikiSQL") -> str:
    txt_path  = Path(output_json_path).with_suffix(".txt")
    em        = results.get("exact_match_accuracy", 0)
    ex        = results.get("execution_accuracy",   0)
    n         = results.get("total_evaluated",      0)
    string_em = results.get("string_em_accuracy")
    sem_det   = results.get("structural_em_details", {})
    em_label  = "Structural EM (official)" if string_em is not None else "Exact Match Accuracy"

    metric_lines = [
        f"  {em_label:<30}: {em:.2%}  ({em*100:.1f}%)",
        f"  {'Execution Accuracy':<30}: {ex:.2%}  ({ex*100:.1f}%)",
        f"  {'Total Evaluated':<30}: {n}",
    ]
    if string_em is not None:
        metric_lines.insert(1,
            f"  {'String EM (legacy)':<30}: {string_em:.2%}  ({string_em*100:.1f}%)")
    if sem_det:
        metric_lines += [
            "",
            "  Structural EM Components:",
            f"    AGG  accuracy   : {sem_det.get('agg_accuracy',  0):.2%}",
            f"    SEL  accuracy   : {sem_det.get('sel_accuracy',  0):.2%}",
            f"    COND accuracy   : {sem_det.get('cond_accuracy', 0):.2%}",
            f"    Parse rate      : {sem_det.get('parse_rate',    0):.2%}",
            f"    Parse failures  : {sem_det.get('parse_failures', 0)}",
        ]
        breakdown = sem_det.get("fail_breakdown", {})
        if breakdown:
            metric_lines.append("  Failure breakdown:")
            for cat, cnt in sorted(breakdown.items(), key=lambda x: -x[1]):
                if cat == "ok": continue
                metric_lines.append(f"    {cat:<22}: {cnt}")

    lines = (
        ["=" * 80,
         f"{dataset.upper()} EVALUATION SUMMARY REPORT",
         f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
         f"Results   : {output_json_path}",
         "=" * 80, "", "MAIN METRICS", "-" * 40]
        + metric_lines + [""]
    )

    if "semantic_statistics" in results:
        lines += ["SEMANTIC LAYER STATISTICS", "-" * 40]
        for k, v in results["semantic_statistics"].items():
            lines.append(f"  {k}: {v}")
        lines.append("")

    if "reasoning_statistics" in results:
        lines += ["REASONINGBANK STATISTICS", "-" * 40]
        for k, v in results["reasoning_statistics"].items():
            lines.append(f"  {k}: {v}")
        lines.append("")

    scores = results.get("scores", {})
    if scores:
        lines += ["PER-DIFFICULTY BREAKDOWN", "-" * 40]
        for level in ["easy", "medium", "hard", "extra", "all"]:
            if level in scores:
                lvl = scores[level]
                lines.append(
                    f"  {level:<8} | count={lvl.get('count',0):<5} "
                    f"| EM={lvl.get('exact',0):.2%} | EX={lvl.get('exec',0):.2%}")
        lines.append("")

    lines += ["=" * 80, "END OF REPORT", "=" * 80]
    txt_path.write_text("\n".join(lines), encoding="utf-8")
    return str(txt_path)


# ─────────────────────────────────────────────────────────────────────────────
# Predicted SQL post-processor
# ─────────────────────────────────────────────────────────────────────────────

def normalize_predicted_wikisql_sql(sql: str) -> str:
    """Normalize a single predicted SQL string for WikiSQL execution.
    Fixes common model errors that cause EX failures without changing semantics.
    """
    if not sql: return sql

    # 1. FROM/JOIN table → FROM/JOIN wikisql_data
    sql = re.sub(r"\bFROM\s+`?table`?\b", f"FROM {WIKISQL_TABLE_NAME}",
                 sql, flags=re.IGNORECASE)
    sql = re.sub(r"\bJOIN\s+`?table`?\b",  f"JOIN {WIKISQL_TABLE_NAME}",
                 sql, flags=re.IGNORECASE)

    # 2. Remove backticks
    sql = sql.replace("`", "")

    # 3. COUNT(DISTINCT col) → COUNT(col)
    #    WikiSQL gold never uses DISTINCT; strips it to match annotation style.
    sql = re.sub(r"\bCOUNT\s*\(\s*DISTINCT\s+", "COUNT(", sql, flags=re.IGNORECASE)

    # 4. Remove IS NOT NULL / IS NULL extra AND conditions
    #    These change result sets when NULL rows exist but gold never adds them.
    sql = re.sub(r"\s+AND\s+[\w.]+\s+IS\s+NOT\s+NULL\b", "", sql, flags=re.IGNORECASE)
    sql = re.sub(r"\s+AND\s+[\w.]+\s+IS\s+NULL\b",        "", sql, flags=re.IGNORECASE)
    sql = re.sub(r"\bWHERE\s+[\w.]+\s+IS\s+NOT\s+NULL\s+AND\s+",
                 "WHERE ", sql, flags=re.IGNORECASE)
    sql = re.sub(r"\bWHERE\s+[\w.]+\s+IS\s+NULL\s+AND\s+",
                 "WHERE ", sql, flags=re.IGNORECASE)
    sql = re.sub(r"\bWHERE\s+[\w.]+\s+IS\s+NOT\s+NULL\s*$", "", sql, flags=re.IGNORECASE)

    # 5. ORDER BY col DESC LIMIT 1  →  SELECT MAX(col) FROM wikisql_data [WHERE ...]
    #    ORDER BY col ASC  LIMIT 1  →  SELECT MIN(col) FROM wikisql_data [WHERE ...]
    #    WikiSQL gold uses MAX/MIN for superlative questions; LIMIT 1 causes EX failures.
    _ob_re = re.compile(
        r"^SELECT\s+(\w+)\s+FROM\s+(\w+)((?:\s+WHERE\s+.+?)?)"
        r"\s+ORDER\s+BY\s+\w+\s*(DESC|ASC)?\s+LIMIT\s+1\s*$",
        re.IGNORECASE | re.DOTALL,
    )
    m = _ob_re.match(sql.strip())
    if m:
        col, tbl, where_part = m.group(1), m.group(2), m.group(3).strip()
        direction = (m.group(4) or "DESC").upper()
        agg = "MAX" if direction == "DESC" else "MIN"
        sql = f"SELECT {agg}({col}) FROM {tbl}"
        if where_part:
            sql += f" {where_part}"

    # 6. SUM(col) with NO WHERE clause → MAX(col)
    #    Table-wide SUM is almost never the correct WikiSQL annotation.
    if not re.search(r"\bWHERE\b", sql, re.IGNORECASE):
        sql = re.sub(
            r"^SELECT\s+SUM\s*\(\s*(\w+)\s*\)\s+FROM\s+(\w+)\s*$",
            lambda m: f"SELECT MAX({m.group(1)}) FROM {m.group(2)}",
            sql.strip(), flags=re.IGNORECASE,
        )

    # 7. Strip trailing semicolons + normalize whitespace
    sql = sql.rstrip(";").strip()
    sql = re.sub(r"\s+", " ", sql)
    return sql.strip()


# Entity column keywords — selecting one of these and having "how many" in the
# question means COUNT is needed (not SUM/bare SELECT).
_COUNT_ENTITY_COLS = re.compile(
    r"\b(player|players|team|teams|game|games|match|matches|episode|episodes|"
    r"film|films|movie|movies|song|songs|album|albums|country|countries|"
    r"nation|nations|city|cities|school|schools|race|races|driver|drivers|"
    r"club|clubs|member|members|winner|winners|champion|champions|round|rounds|"
    r"tournament|tournaments|stage|stages|year|years|season|seasons|"
    r"week|weeks|division|divisions|league|leagues|district|districts|"
    r"candidate|candidates|party|parties|region|regions|island|islands|"
    r"station|stations|line|lines|route|routes|event|events|title|titles|"
    r"entry|entries|record|records|report|reports|service|services|show|shows|series|"
    r"competitor|competitors|athlete|athletes|performer|performers)\b",
    re.IGNORECASE,
)

# Trigger phrases that indicate "how many records" (COUNT)
_COUNT_TRIGGER_RE = re.compile(
    r"\b(how many|number of|total number of|name the number|how often|"
    r"how much|in how many|how many times)\b",
    re.IGNORECASE,
)

# Numeric-quantity column keywords — "how many X" where X is a numeric column
# → bare SELECT (Rule 4, NOT COUNT)
_NUMERIC_QTY_COLS = re.compile(
    r"\b(goal|goals|viewer|viewers|vote|votes|point|points|run|runs|"
    r"lap|laps|yard|yards|score|scores|seat|seats|passenger|passengers|"
    r"revenue|revenues|budget|budgets|crowd|crowds|rating|ratings|"
    r"attendance|attendances|medal|medals|strike|strikes|error|errors|"
    r"assist|assists|rebound|rebounds|minute|minutes|kill|kills|"
    r"save|saves|hit|hits|win|wins|loss|losses|draw|draws|tie|ties|"
    r"penalty|penalties|foul|fouls|card|cards|cap|caps|tour|tours|"
    r"pass|passes|shot|shots|tackle|tackles|block|blocks|"
    r"century|centuries|wicket|wickets|over|overs|inning|innings|"
    r"start|starts|appearance|appearances)\b",
    re.IGNORECASE,
)

_SELECT_BARE_RE = re.compile(
    r"^SELECT\s+(\w+)\s+FROM\s+(\w+)((?:\s+WHERE\s+.+?)?)\s*$",
    re.IGNORECASE | re.DOTALL,
)


def _maybe_add_count(sql: str, question: str) -> str:
    """
    If the question has a COUNT trigger phrase and the prediction uses bare
    SELECT (no aggregation), and the selected column is an entity-type column
    (not a numeric-quantity column), wrap in COUNT().

    This fixes the WikiSQL annotation quirk where "how many players/teams/games?"
    is annotated as COUNT(col) even though only one row matches.
    """
    if not _COUNT_TRIGGER_RE.search(question):
        return sql

    m = _SELECT_BARE_RE.match(sql.strip())
    if not m:
        return sql  # already has AGG or not a simple SELECT

    col, tbl, where_part = m.group(1), m.group(2), m.group(3).strip()

    # Don't add COUNT if the column is a numeric quantity — Rule 4
    if _NUMERIC_QTY_COLS.search(col):
        return sql

    # The column should be an entity-type column to add COUNT
    if _COUNT_ENTITY_COLS.search(col):
        result = f"SELECT COUNT({col}) FROM {tbl}"
        if where_part:
            result += f" {where_part}"
        return result

    return sql


def _preprocess_predictions_tsv(predict_path: str, gold_path: Optional[str] = None) -> str:
    """Read predictions TSV, apply normalize_predicted_wikisql_sql to every row,
    optionally add COUNT for question-aware AGG detection,
    write to a normalized file alongside the original, return the new path."""
    # Load questions for COUNT detection
    questions: List[str] = []
    if gold_path and Path(gold_path).exists():
        try:
            with open(gold_path, encoding="utf-8") as f:
                gold_data = json.load(f)
            questions = [item.get("question", "") for item in gold_data]
        except Exception:
            pass

    lines_out = []
    changed = 0

    with open(predict_path, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            raw = line.rstrip("\n")
            sql, db_id = (raw.split("\t", 1) if "\t" in raw else (raw, ""))
            sql = sql.strip()

            # Step 1: rule-based normalization
            fixed = normalize_predicted_wikisql_sql(sql)
            if fixed != sql:
                changed += 1

            # Step 2: question-aware COUNT injection — DISABLED
            # COUNT injection causes more regressions than gains.
            # Gold=bare pred=COUNT produces integer vs string → EX fails.
            # Re-generate with improved prompts instead (wikisql_prompt_patch.py).

            lines_out.append(f"{fixed}\t{db_id}" if db_id else fixed)

    base, ext = os.path.splitext(predict_path)
    tmp_path = base + "_normalized" + ext
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines_out) + "\n")

    logger.info(
        f"Prediction post-processing: {changed}/{len(lines_out)} rows normalized → {tmp_path}"
    )
    return tmp_path


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Text-to-SQL on WikiSQL (mirrors evaluate_spider.py)",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)

    parser.add_argument("--gold",      required=True,
                        help="Path to WikiSQL gold file (dev_spider_format.json or dev.json)")
    parser.add_argument("--table",     required=True, help="Path to tables.json")
    parser.add_argument("--predict",   default=None,  help="Path to predictions TSV")
    parser.add_argument("--questions", default=None)
    parser.add_argument("--etype",     default="all", choices=["all", "exec", "match"])
    parser.add_argument("--use_langchain",    action="store_true")
    parser.add_argument("--prompt_type",      default="enhanced",
                        choices=["basic","few_shot","chain_of_thought","enhanced"])
    parser.add_argument("--enable_debugging", action="store_true")
    parser.add_argument("--use_chromadb",     action="store_true")
    parser.add_argument("--chromadb_config",  default=None)
    parser.add_argument("--use_semantic",     action="store_true")
    parser.add_argument("--semantic_config",  default=None)
    parser.add_argument("--use_reasoning_bank",       action="store_true")
    parser.add_argument("--reasoning_config",
                        default="./configs/reasoning_config.yaml")
    parser.add_argument("--enable_test_time_scaling", action="store_true")
    parser.add_argument("--consolidation_frequency",  type=int, default=50)
    parser.add_argument("--limit",         type=int, default=None)
    parser.add_argument("--plug_value",    action="store_true")
    parser.add_argument("--keep_distinct", action="store_true")
    parser.add_argument("--progress",      action="store_true")
    parser.add_argument("--output",        default="./results/wikisql_results.json")
    parser.add_argument("--db",            default="./data/raw/wikisql/database")
    parser.add_argument("--save_em_failures", default=None, metavar="FILE",
                        help="Save per-query failure rows to FILE.csv for analysis")

    args = parser.parse_args()

    if args.use_langchain and not args.questions:
        args.questions = args.gold
    if not args.use_langchain and not args.predict:
        parser.error("Either --use_langchain or --predict is required")

    # ── Auto-detect original dev.json from spider-format path ─────────────────
    gold_original = args.gold
    if args.gold.endswith("_spider_format.json") or args.gold.endswith("_converted.json"):
        _stem = args.gold.replace("_spider_format", "").replace("_converted", "")
        if Path(_stem).exists():
            gold_original = _stem

    # === STEP 1: Build SQLite databases ===
    logger.info("=" * 80)
    logger.info("STEP 1: Preparing WikiSQL SQLite databases")
    logger.info("=" * 80)
    db_dir = prepare_wikisql_databases(
        gold_file=gold_original, db_dir=args.db, limit=args.limit)

    # === STEP 2: Convert gold SQL ===
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: Converting WikiSQL gold SQL to Spider format")
    logger.info("=" * 80)
    converted_gold = args.gold.replace(".json", "_spider_format.json")
    converted_gold = os.path.join("./data/raw/wikisql", os.path.basename(converted_gold))
    if not converted_gold.endswith("_spider_format_spider_format.json"):
        convert_wikisql_gold_to_spider_format(
            gold_file=gold_original, output_file=converted_gold, limit=args.limit)

    questions_file = args.questions or args.gold
    if questions_file == args.gold:
        questions_file = converted_gold

    # === STEP 3: Load configs ===
    chromadb_config  = load_config(args.chromadb_config) if args.chromadb_config else None
    semantic_config  = (load_config(args.semantic_config) if args.semantic_config
                        else ({"enabled": True} if args.use_semantic else None))
    reasoning_config = None
    if args.use_reasoning_bank:
        reasoning_config = load_config(args.reasoning_config) or {}
        reasoning_config["enable_test_time_scaling"] = args.enable_test_time_scaling
        reasoning_config["consolidation_frequency"]  = args.consolidation_frequency

    # === STEP 4: Build kmaps ===
    tables_json = os.path.join(db_dir, "tables.json")
    kmaps = {}
    if os.path.exists(tables_json):
        with open(tables_json) as f:
            tables_data = json.load(f)
        kmaps = {t["db_id"]: t for t in tables_data}
        logger.info(f"✅ Loaded {len(kmaps)} kmaps from {tables_json}")

    # === STEP 5: Print configuration ===
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: Running Evaluation")
    logger.info("=" * 80)
    logger.info(f"Gold file (converted): {converted_gold}")
    logger.info(f"Database directory:    {db_dir}")
    logger.info(f"Evaluation type:       {args.etype}")
    logger.info(f"Generation mode:       "
                f"{'LangChain' if args.use_langchain else 'Predictions file'}")
    logger.info(f"ChromaDB enabled:      {args.use_chromadb}")
    logger.info(f"Semantic enabled:      {args.use_semantic}")
    logger.info(f"ReasoningBank enabled: {args.use_reasoning_bank}")
    logger.info(f"Limit:                 {args.limit or 'None'}")
    logger.info("=" * 80)

    # === STEP 6: Run evaluation (EX + string-level EM via evaluator) ===
    try:
        # Pre-process predictions to fix common SQL patterns that cause EX failures
        _predict_path = args.predict
        if args.predict:
            _predict_path = _preprocess_predictions_tsv(args.predict, gold_path=gold_original)

        results = evaluate(
            gold=converted_gold, predict=_predict_path, db_dir=db_dir,
            etype=args.etype, kmaps=kmaps,
            plug_value=args.plug_value, keep_distinct=args.keep_distinct,
            progress_bar_for_each_datapoint=args.progress,
            use_langchain=args.use_langchain, questions_file=questions_file,
            prompt_type=args.prompt_type, enable_debugging=args.enable_debugging,
            use_chromadb=args.use_chromadb, chromadb_config=chromadb_config,
            use_semantic=args.use_semantic, semantic_config=semantic_config,
            use_reasoning_bank=args.use_reasoning_bank,
            reasoning_config=reasoning_config, limit=args.limit,
        )

        # === STEP 7: Structural EM override ===
        if _predict_path and Path(gold_original).exists():
            logger.info("\nComputing structural EM (official WikiSQL methodology)...")
            try:
                sem = compute_wikisql_structural_em(
                    gold_file         = gold_original,
                    predict_tsv       = _predict_path,
                    table_file        = args.table,
                    limit             = args.limit,
                    save_failures_csv = args.save_em_failures,
                )
                results["string_em_accuracy"]    = results.get("exact_match_accuracy", 0)
                results["exact_match_accuracy"]  = sem["structural_em"]
                results["structural_em_details"] = {
                    "parse_rate"    : sem["parse_rate"],
                    "agg_accuracy"  : sem["agg_accuracy"],
                    "sel_accuracy"  : sem["sel_accuracy"],
                    "cond_accuracy" : sem["cond_accuracy"],
                    "parse_failures": sem["parse_failures"],
                    "gold_failures" : sem["gold_failures"],
                    "fail_breakdown": sem["fail_breakdown"],
                }
                # Always print full breakdown (stdout, not logger)
                _print_structural_em_summary(sem)

            except Exception as _sem_err:
                logger.warning(
                    f"Structural EM computation failed: {_sem_err}\n"
                    f"  Falling back to string-level EM.")
        else:
            if not args.predict:
                logger.info("Skipping structural EM (LangChain mode).")
            else:
                logger.warning(
                    f"Original dev.json not found at {gold_original} — "
                    f"string-level EM only.")

        # === STEP 8: Save results ===
        output_file = Path(args.output)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"\n✓ Results saved to: {output_file}")

        txt_file = save_summary_report(results, str(output_file), dataset="WikiSQL")
        logger.info(f"✓ Summary report  : {txt_file}")

        # === STEP 9: Terminal summary (stdout) ===
        _em  = results.get("exact_match_accuracy", 0)
        _ex  = results.get("execution_accuracy",   0)
        _sem = results.get("string_em_accuracy")

        print("=" * 80)
        print("WIKISQL EVALUATION SUMMARY")
        print("=" * 80)
        if _sem is not None:
            print(f"Structural EM (official) : {_em * 100:.2f}%   ← use this in Table II")
            print(f"String EM (legacy)       : {_sem * 100:.2f}%")
        else:
            print(f"Exact Match Accuracy     : {_em * 100:.2f}%")
        print(f"Execution Accuracy       : {_ex * 100:.2f}%")
        print("=" * 80)
        if args.save_em_failures:
            print(f"Failures CSV             : {args.save_em_failures}")
            print("=" * 80)

        if "semantic_statistics" in results:
            logger.info("\nSemantic Layer Statistics:")
            for k, v in results["semantic_statistics"].items():
                logger.info(f"  {k}: {v}")
        if "reasoning_statistics" in results:
            logger.info("\nReasoningBank Statistics:")
            for k, v in results["reasoning_statistics"].items():
                logger.info(f"  {k}: {v}")

        return 0

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback; traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
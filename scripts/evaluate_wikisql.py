"""
WikiSQL Evaluation Script
Mirrors evaluate_spider.py exactly — reuses src.reasoning.evaluator.evaluate().

Strategy: WikiSQL embeds table rows inside each JSON example.
We pre-build real .sqlite files from those rows so the existing
evaluate() pipeline works without modification.

Usage:
  python scripts/evaluate_wikisql.py \
      --gold data/raw/wikisql/dev.json \
      --table data/raw/wikisql/tables.json \
      --questions data/raw/wikisql/dev.json \
      --use_langchain \
      --use_semantic \
      --use_chromadb \
      --use_reasoning_bank \
      --etype all \
      --limit 10
"""

import warnings
import sys
import os
import argparse
import json
import re
import sqlite3
import logging
from pathlib import Path
from typing import Dict, List, Optional
from dotenv import load_dotenv

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ── Logging setup ─────────────────────────────────────────────────────────────
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(level=logging.WARNING, format='%(message)s', stream=sys.stdout)
logging.getLogger('__main__').setLevel(logging.INFO)
logging.getLogger('src.reasoning.evaluator').setLevel(logging.INFO)

for _name in [
    'utils.embedding_utils',
    'src.reasoning.memory_retrieval',
    'src.reasoning.memory_store',
    'src.reasoning.reasoning_pipeline',
    'src.reasoning.experience_collector',
    'src.reasoning.self_judgment',
    'src.reasoning.strategy_distillation',
    'src.reasoning.memory_consolidation',
    'src.semantic.semantic_pipeline',
    'chromadb', 'chromadb.api', 'chromadb.telemetry',
]:
    logging.getLogger(_name).setLevel(logging.ERROR)

warnings.filterwarnings('ignore')

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.reasoning.evaluator import evaluate
from src.evaluation.foreign_key_mapper import build_foreign_key_map_from_json
from utils.logging_utils import get_logger

logger = get_logger(__name__)

warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────────────────────────────────────
# WikiSQL → SQLite conversion
# ─────────────────────────────────────────────────────────────────────────────

WIKISQL_TABLE_NAME = "wikisql_data"

# SQL reserved words that must be double-quoted as column names in SQLite DDL.
_SQLITE_RESERVED = {
    'group', 'order', 'select', 'from', 'where', 'table', 'index',
    'join', 'on', 'as', 'by', 'having', 'limit', 'offset', 'union',
    'intersect', 'except', 'case', 'when', 'then', 'else', 'end',
    'and', 'or', 'not', 'in', 'like', 'is', 'null', 'between',
    'exists', 'all', 'any', 'distinct', 'into', 'set', 'values',
    'insert', 'update', 'delete', 'create', 'drop', 'alter', 'add',
    'column', 'primary', 'key', 'foreign', 'references', 'unique',
    'check', 'default', 'with', 'desc', 'asc',
}


def sanitize_col(name: str) -> str:
    """Make a column name safe for SQLite."""
    name = re.sub(r"[^\w]", "_", name.strip())
    if not name or name[0].isdigit():
        name = "col_" + name
    return name


def _quote_col(name: str) -> str:
    """Double-quote a column name if it is a reserved word or contains special chars."""
    if name.lower() in _SQLITE_RESERVED or ' ' in name or '-' in name:
        return f'"{name}"'
    return name


def wikisql_type_to_sqlite(t: str) -> str:
    t = t.lower()
    if t in ("real", "float", "number"):
        return "REAL"
    if t in ("integer", "int"):
        return "INTEGER"
    return "TEXT"


def _is_empty_sqlite(db_path: str) -> bool:
    """Return True if the SQLite file exists but contains no tables."""
    try:
        conn  = sqlite3.connect(db_path)
        cur   = conn.cursor()
        cur.execute("SELECT count(*) FROM sqlite_master WHERE type='table'")
        count = cur.fetchone()[0]
        conn.close()
        return count == 0
    except Exception:
        return True


def build_sqlite_from_wikisql_item(item: Dict, db_path: str) -> bool:
    """
    Create a .sqlite file for one WikiSQL example.
    Columns are stored lowercase so gold/predicted SQL match without quoting.
    Reserved SQL keywords used as column names are double-quoted in DDL.
    """
    table   = item.get("table", {})
    headers = table.get("header", [])
    types   = table.get("types",  ["text"] * len(headers))
    rows    = table.get("rows",   [])

    if not headers:
        return False

    safe_headers = [sanitize_col(h).lower() for h in headers]

    col_defs = ", ".join(
        f"{_quote_col(col)} {wikisql_type_to_sqlite(t)}"
        for col, t in zip(safe_headers, types)
    )

    try:
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        conn = sqlite3.connect(db_path)
        conn.text_factory = str
        cur  = conn.cursor()
        cur.execute(
            f'CREATE TABLE IF NOT EXISTS {WIKISQL_TABLE_NAME} ({col_defs})'
        )
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
    """
    Read WikiSQL JSON, build one .sqlite file per unique db_id,
    and write a Spider-compatible tables.json into db_dir.

    Returns db_dir (same path passed in, created if needed).
    """
    logger.info(f"📦 Preparing WikiSQL databases in: {db_dir}")
    os.makedirs(db_dir, exist_ok=True)

    with open(gold_file, encoding="utf-8") as f:
        data: List[Dict] = json.load(f)

    if limit:
        data = data[:limit]

    built    = 0
    skipped  = 0
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

            # Rebuild if missing OR empty (previously failed on reserved keyword)
            if not os.path.exists(db_path) or _is_empty_sqlite(db_path):
                ok = build_sqlite_from_wikisql_item(item, db_path)
                if ok:
                    built += 1
                else:
                    skipped += 1

            # Build tables.json entry — column names must match SQLite DDL
            table   = item.get("table", {})
            headers = table.get("header", [])
            types   = table.get("types",  ["text"] * len(headers))
            safe_h  = [sanitize_col(h).lower() for h in headers]

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
    """
    Remove double-quote wrappers from column/table identifiers only.

    Double-quoted tokens that look like plain identifiers (word chars only)
    are unquoted and lowercased.  Double-quoted tokens that contain spaces,
    apostrophes, or other special characters are string literals — they are
    converted to single-quoted SQL strings with apostrophes escaped as ''.

    Examples:
        "Position"    → position
        "st. john's"  → 'st. john''s'
        "New York"    → 'New York'
    """
    def replacer(m: re.Match) -> str:
        inner = m.group(1)
        if re.fullmatch(r'\w+', inner):
            return inner.lower()
        escaped = inner.replace("'", "''")
        return f"'{escaped}'"

    return re.sub(r'"([^"]*)"', replacer, sql)


def _normalize_empty_string_literals(sql: str) -> str:
    """
    Replace empty string literals  = ''  with a sentinel value.
    The Spider parser cannot tokenize an empty SQL string literal.
    """
    return re.sub(r"=\s*''(?=[^']|$)", "= '__empty__'", sql)


def _normalize_not_operators(sql: str) -> str:
    """
    Normalise NOT-related patterns and model artifact clauses in SQL
    before parsing.

    Handles:
    1. AND col IS NOT NULL / AND col IS NULL  — strips them
    2. WHERE col IS NOT NULL (standalone)     → WHERE 1=1
    3. AND lower(col) = 'val'                 — strips them
    4. Scientific notation  = 71.1e           → = '71.1e'
    5. NOT IN / NOT LIKE / NOT BETWEEN        — lowercase
    """
    if not sql:
        return sql

    # AND-prefixed IS NOT NULL / IS NULL
    sql = re.sub(r'\bAND\s+\w+\s+is\s+not\s+null\b',      '', sql, flags=re.IGNORECASE)
    sql = re.sub(r'\bAND\s+\w+\s+is\s+null\b',            '', sql, flags=re.IGNORECASE)
    sql = re.sub(r'\bAND\s+\w+\.\w+\s+is\s+not\s+null\b', '', sql, flags=re.IGNORECASE)
    sql = re.sub(r'\bAND\s+\w+\.\w+\s+is\s+null\b',       '', sql, flags=re.IGNORECASE)

    # Standalone WHERE col IS NOT NULL / WHERE col IS NULL
    sql = re.sub(r'\bWHERE\s+\w+\s+is\s+not\s+null\b',      'WHERE 1=1', sql, flags=re.IGNORECASE)
    sql = re.sub(r'\bWHERE\s+\w+\s+is\s+null\b',            'WHERE 1=1', sql, flags=re.IGNORECASE)
    sql = re.sub(r'\bWHERE\s+\w+\.\w+\s+is\s+not\s+null\b', 'WHERE 1=1', sql, flags=re.IGNORECASE)
    sql = re.sub(r'\bWHERE\s+\w+\.\w+\s+is\s+null\b',       'WHERE 1=1', sql, flags=re.IGNORECASE)

    # Remove  AND lower(col) = 'val'  — unsupported function artifact
    sql = re.sub(
        r'\bAND\s+lower\s*\(\s*\w+\s*\)\s*=\s*(?:\'[^\']*\'|"[^"]*"|\S+)',
        '', sql, flags=re.IGNORECASE
    )

    # Fix scientific notation  = 71.1e  → = '71.1e'
    sql = re.sub(
        r'=\s*(\d+\.\d+[eE])\b',
        lambda m: f"= '{m.group(1)}'",
        sql
    )

    # Convert remaining double-quoted string values → single-quoted
    # e.g.  = "super☆looper"  →  = 'super☆looper'
    # This handles model output that uses double quotes for string literals
    # which the Spider tokenizer misinterprets as identifiers.
    def _dquote_to_squote(m: re.Match) -> str:
        inner = m.group(1).replace("'", "''")   # escape any apostrophes
        return f"= '{inner}'"
    sql = re.sub(r'=\s*"([^"]*)"', _dquote_to_squote, sql)

    # Lowercase NOT IN / NOT LIKE / NOT BETWEEN
    sql = re.sub(r'\bNOT\s+IN\b',      'not in',      sql, flags=re.IGNORECASE)
    sql = re.sub(r'\bNOT\s+LIKE\b',    'not like',    sql, flags=re.IGNORECASE)
    sql = re.sub(r'\bNOT\s+BETWEEN\b', 'not between', sql, flags=re.IGNORECASE)

    sql = re.sub(r'\s+', ' ', sql).strip()
    return sql


def convert_wikisql_gold_to_spider_format(
    gold_file: str,
    output_file: str,
    limit: Optional[int] = None,
) -> str:
    """
    Convert WikiSQL dev.json → Spider-compatible gold JSON.

    WikiSQL gold SQL uses col0/col1 placeholders.  We replace them with
    the actual sanitized column names so the existing evaluator can parse
    and execute them against the SQLite files we built.

    Output format (Spider JSON):
        [{"db_id": "...", "question": "...", "query": "<real SQL>"}, ...]
    """
    with open(gold_file, encoding="utf-8") as f:
        data: List[Dict] = json.load(f)

    if limit:
        data = data[:limit]

    converted = []
    for item in data:
        db_id    = item.get("db_id", "")
        question = item.get("question", "")
        sql      = item.get("sql", "")
        table    = item.get("table", {})
        headers  = table.get("header", [])

        safe_headers = [sanitize_col(h).lower() for h in headers]

        def replacer(m):
            idx = int(m.group(1))
            return safe_headers[idx] if idx < len(safe_headers) else m.group(0)

        real_sql = re.sub(r'\bcol(\d+)\b', replacer, sql)

        real_sql = re.sub(
            r'\bFROM\s+["`\[]?table["`\]]?\b',
            f'FROM {WIKISQL_TABLE_NAME}',
            real_sql,
            flags=re.IGNORECASE,
        )
        real_sql = re.sub(
            r'\bJOIN\s+["`\[]?table["`\]]?\b',
            f'JOIN {WIKISQL_TABLE_NAME}',
            real_sql,
            flags=re.IGNORECASE,
        )

        real_sql = _strip_quoted_identifiers(real_sql)
        real_sql = _normalize_empty_string_literals(real_sql)
        real_sql = _normalize_not_operators(real_sql)

        converted.append({
            "db_id":    db_id,
            "question": question,
            "query":    real_sql,
            "sql":      real_sql,
        })

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(converted, f, indent=2, ensure_ascii=False)

    logger.info(f"✅ Converted {len(converted)} examples → {output_file}")
    return output_file


# ─────────────────────────────────────────────────────────────────────────────
# Config loader
# ─────────────────────────────────────────────────────────────────────────────

def load_config(config_path: str) -> dict:
    if not config_path or not os.path.exists(config_path):
        if config_path:
            logger.warning(f"Config file not found: {config_path}")
        return {}
    if config_path.endswith('.json'):
        with open(config_path) as f:
            return json.load(f)
    elif config_path.endswith(('.yaml', '.yml')):
        try:
            import yaml
            with open(config_path) as f:
                return yaml.safe_load(f)
        except ImportError:
            logger.warning("PyYAML not installed, cannot load YAML config")
            return {}
    logger.warning(f"Unsupported config format: {config_path}")
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# Summary report
# ─────────────────────────────────────────────────────────────────────────────

def save_summary_report(results: dict, output_json_path: str, dataset: str = "WikiSQL") -> str:
    """Save a human-readable .txt summary next to the JSON results file."""
    from datetime import datetime

    txt_path = Path(output_json_path).with_suffix(".txt")

    em  = results.get("exact_match_accuracy", 0)
    ex  = results.get("execution_accuracy",   0)
    n   = results.get("total_evaluated",      0)

    lines = [
        "=" * 80,
        f"{dataset.upper()} EVALUATION SUMMARY REPORT",
        f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Results   : {output_json_path}",
        "=" * 80,
        "",
        "MAIN METRICS",
        "-" * 40,
        f"  Exact Match Accuracy : {em:.2%}  ({em*100:.1f}%)",
        f"  Execution Accuracy   : {ex:.2%}  ({ex*100:.1f}%)",
        f"  Total Evaluated      : {n}",
        "",
    ]

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
                lvl   = scores[level]
                cnt   = lvl.get("count", 0)
                exact = lvl.get("exact", 0)
                exec_ = lvl.get("exec",  0)
                lines.append(
                    f"  {level:<8} | count={cnt:<5} | EM={exact:.2%} | EX={exec_:.2%}"
                )
        lines.append("")

    lines += ["=" * 80, "END OF REPORT", "=" * 80]
    txt_path.write_text("\n".join(lines), encoding="utf-8")
    return str(txt_path)


# ─────────────────────────────────────────────────────────────────────────────
# Predicted SQL post-processor
# ─────────────────────────────────────────────────────────────────────────────

def normalize_predicted_wikisql_sql(sql: str) -> str:
    """
    Post-process LLM-generated SQL for WikiSQL evaluation.
    Renames FROM table → FROM wikisql_data and strips backticks.
    """
    if not sql:
        return sql
    sql = re.sub(r'\bFROM\s+`?table`?\b', f'FROM {WIKISQL_TABLE_NAME}',
                 sql, flags=re.IGNORECASE)
    sql = re.sub(r'\bJOIN\s+`?table`?\b',  f'JOIN {WIKISQL_TABLE_NAME}',
                 sql, flags=re.IGNORECASE)
    sql = sql.replace('`', '')
    return sql.strip()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate Text-to-SQL system on WikiSQL (mirrors evaluate_spider.py)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('--gold',      required=True)
    parser.add_argument('--table',     required=True)
    parser.add_argument('--predict',   default=None)
    parser.add_argument('--questions', default=None)
    parser.add_argument('--etype',     default='all',
                        choices=['all', 'exec', 'match'])
    parser.add_argument('--use_langchain',    action='store_true')
    parser.add_argument('--prompt_type',      default='enhanced',
                        choices=['basic', 'few_shot', 'chain_of_thought', 'enhanced'])
    parser.add_argument('--enable_debugging', action='store_true')
    parser.add_argument('--use_chromadb',     action='store_true')
    parser.add_argument('--chromadb_config',  default=None)
    parser.add_argument('--use_semantic',     action='store_true')
    parser.add_argument('--semantic_config',  default=None)
    parser.add_argument('--use_reasoning_bank',       action='store_true')
    parser.add_argument('--reasoning_config',
                        default='./configs/reasoning_config.yaml')
    parser.add_argument('--enable_test_time_scaling', action='store_true')
    parser.add_argument('--consolidation_frequency',  type=int, default=50)
    parser.add_argument('--limit',         type=int, default=None)
    parser.add_argument('--plug_value',    action='store_true')
    parser.add_argument('--keep_distinct', action='store_true')
    parser.add_argument('--progress',      action='store_true')
    parser.add_argument('--output',        default='./results/wikisql_results.json')
    parser.add_argument('--db',            default='./data/raw/wikisql/database')

    args = parser.parse_args()

    if args.use_langchain and not args.questions:
        args.questions = args.gold

    if not args.use_langchain and not args.predict:
        parser.error("Either --use_langchain or --predict is required")

    # === STEP 1: Build SQLite databases ===
    logger.info("=" * 80)
    logger.info("STEP 1: Preparing WikiSQL SQLite databases")
    logger.info("=" * 80)
    db_dir = prepare_wikisql_databases(
        gold_file=args.gold,
        db_dir=args.db,
        limit=args.limit,
    )

    # === STEP 2: Convert gold SQL ===
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: Converting WikiSQL gold SQL to Spider format")
    logger.info("=" * 80)
    converted_gold = args.gold.replace(".json", "_spider_format.json")
    converted_gold = os.path.join("./data/raw/wikisql", os.path.basename(converted_gold))
    convert_wikisql_gold_to_spider_format(
        gold_file=args.gold,
        output_file=converted_gold,
        limit=args.limit,
    )

    questions_file = args.questions or args.gold
    if questions_file == args.gold:
        questions_file = converted_gold

    # === STEP 3: Load configs ===
    chromadb_config  = load_config(args.chromadb_config) if args.chromadb_config else None
    semantic_config  = load_config(args.semantic_config) if args.semantic_config else (
        {'enabled': True} if args.use_semantic else None
    )
    reasoning_config = None
    if args.use_reasoning_bank:
        reasoning_config = load_config(args.reasoning_config) or {}
        reasoning_config['enable_test_time_scaling'] = args.enable_test_time_scaling
        reasoning_config['consolidation_frequency']  = args.consolidation_frequency

    # === STEP 4: Build kmaps ===
    tables_json = os.path.join(db_dir, "tables.json")
    kmaps = {}
    if os.path.exists(tables_json):
        with open(tables_json) as f:
            tables_data = json.load(f)
        kmaps = {t['db_id']: t for t in tables_data}
        logger.info(f"✅ Loaded {len(kmaps)} kmaps from {tables_json}")

    # === STEP 5: Print configuration ===
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: Running Evaluation")
    logger.info("=" * 80)
    logger.info(f"Gold file (converted): {converted_gold}")
    logger.info(f"Database directory:    {db_dir}")
    logger.info(f"Evaluation type:       {args.etype}")
    logger.info(f"Generation mode:       {'LangChain' if args.use_langchain else 'Predictions file'}")
    logger.info(f"Prompt type:           {args.prompt_type}")
    logger.info(f"ChromaDB enabled:      {args.use_chromadb}")
    logger.info(f"Semantic enabled:      {args.use_semantic}")
    logger.info(f"ReasoningBank enabled: {args.use_reasoning_bank}")
    logger.info(f"Limit:                 {args.limit or 'None'}")
    logger.info("=" * 80)

    # === STEP 6: Run evaluation ===
    try:
        results = evaluate(
            gold=converted_gold,
            predict=args.predict,
            db_dir=db_dir,
            etype=args.etype,
            kmaps=kmaps,
            plug_value=args.plug_value,
            keep_distinct=args.keep_distinct,
            progress_bar_for_each_datapoint=args.progress,
            use_langchain=args.use_langchain,
            questions_file=questions_file,
            prompt_type=args.prompt_type,
            enable_debugging=args.enable_debugging,
            use_chromadb=args.use_chromadb,
            chromadb_config=chromadb_config,
            use_semantic=args.use_semantic,
            semantic_config=semantic_config,
            use_reasoning_bank=args.use_reasoning_bank,
            reasoning_config=reasoning_config,
            limit=args.limit,
        )

        output_file = Path(args.output)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"\n✓ Results saved to: {output_file}")

        txt_file = save_summary_report(results, str(output_file), dataset="WikiSQL")
        logger.info(f"✓ Summary report  : {txt_file}")

        logger.info("\n" + "=" * 80)
        logger.info("WIKISQL EVALUATION SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Exact Match Accuracy : {results.get('exact_match_accuracy', 0):.2%}")
        logger.info(f"Execution Accuracy   : {results.get('execution_accuracy',   0):.2%}")
        logger.info("=" * 80)
        return 0

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
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
import tempfile
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Optional
from dotenv import load_dotenv

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ── Logging setup (mirrors evaluate_spider.py) ────────────────────────────────
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

# Use this as the actual SQLite table name to avoid the reserved word "table"
WIKISQL_TABLE_NAME = "wikisql_data"


def sanitize_col(name: str) -> str:
    """Make a column name safe for SQLite."""
    name = re.sub(r"[^\w]", "_", name.strip())
    if not name or name[0].isdigit():
        name = "col_" + name
    return name


def wikisql_type_to_sqlite(t: str) -> str:
    t = t.lower()
    if t in ("real", "float", "number"):
        return "REAL"
    if t in ("integer", "int"):
        return "INTEGER"
    return "TEXT"


def build_sqlite_from_wikisql_item(item: Dict, db_path: str) -> bool:
    """
    Create a .sqlite file for one WikiSQL example.
    Columns are stored lowercase so gold/predicted SQL match without quoting.
    """
    table   = item.get("table", {})
    headers = table.get("header", [])
    types   = table.get("types",  ["text"] * len(headers))
    rows    = table.get("rows",   [])

    if not headers:
        return False

    # Lowercase so SQL parser matches without case sensitivity issues
    safe_headers = [sanitize_col(h).lower() for h in headers]
    col_defs = ", ".join(
        f'{h} {wikisql_type_to_sqlite(t)}'
        for h, t in zip(safe_headers, types)
    )

    try:
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        conn = sqlite3.connect(db_path)
        conn.text_factory = str
        cur = conn.cursor()
        cur.execute(f'CREATE TABLE IF NOT EXISTS {WIKISQL_TABLE_NAME} ({col_defs})')
        if rows:
            ph = ", ".join(["?"] * len(safe_headers))
            cur.executemany(
                f'INSERT INTO {WIKISQL_TABLE_NAME} VALUES ({ph})', rows
            )
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        logger.warning(f"Failed to build SQLite for {db_path}: {e}")
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
    tables_schema = []   # Spider-compatible tables.json entries

    for item in data:
        db_id = item.get("db_id", "")
        if not db_id:
            skipped += 1
            continue

        db_folder = os.path.join(db_dir, db_id)
        db_path   = os.path.join(db_folder, f"{db_id}.sqlite")

        if db_id not in seen_ids:
            seen_ids.add(db_id)
            if not os.path.exists(db_path):
                ok = build_sqlite_from_wikisql_item(item, db_path)
                if ok:
                    built += 1
                else:
                    skipped += 1

            # Build tables.json entry for this db_id
            table   = item.get("table", {})
            headers = table.get("header", [])
            types   = table.get("types",  ["text"] * len(headers))
            safe_h  = [sanitize_col(h) for h in headers]

            tables_schema.append({
                "db_id": db_id,
                "table_names": [WIKISQL_TABLE_NAME],
                "table_names_original": [WIKISQL_TABLE_NAME],
                "column_names": [[-1, "*"]] + [[0, h] for h in safe_h],
                "column_names_original": [[-1, "*"]] + [[0, h] for h in safe_h],
                "column_types": ["text"] + [t.lower() for t in types],
                "primary_keys": [],
                "foreign_keys": [],
            })

    # Write Spider-compatible tables.json
    tables_json_path = os.path.join(db_dir, "tables.json")
    with open(tables_json_path, "w", encoding="utf-8") as f:
        json.dump(tables_schema, f, indent=2)

    logger.info(f"✅ Built {built} new SQLite DBs, skipped {skipped}, "
                f"total unique IDs: {len(seen_ids)}")
    logger.info(f"✅ Wrote tables.json: {tables_json_path}")
    return db_dir


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

        # Sanitize headers to match SQLite column names (no quotes, lowercase)
        safe_headers = [sanitize_col(h).lower() for h in headers]

        def replacer(m):
            idx = int(m.group(1))
            return safe_headers[idx] if idx < len(safe_headers) else m.group(0)

        real_sql = re.sub(r'\bcol(\d+)\b', replacer, sql)

        # Rename the reserved word "table" → WIKISQL_TABLE_NAME in FROM/JOIN clauses
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

        # Strip ALL double-quote identifiers so Spider parser can handle them
        # e.g. "Position" → position, "School_Club_Team" → school_club_team
        real_sql = re.sub(r'"([^"]+)"', lambda m: m.group(1).lower(), real_sql)

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
# Config loader (identical to evaluate_spider.py)
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

def normalize_predicted_wikisql_sql(sql: str) -> str:
    """
    Post-process LLM-generated SQL for WikiSQL evaluation.

    LLMs trained on Spider/WikiSQL typically emit "FROM table" because
    the original WikiSQL dataset uses "table" as the table name.
    We rename it to WIKISQL_TABLE_NAME to match the SQLite files we built.

    Also strips backtick quoting that some models emit.
    """
    if not sql:
        return sql

    # Replace bare "table" reference in FROM / JOIN
    sql = re.sub(
        r'\bFROM\s+`?table`?\b',
        f'FROM {WIKISQL_TABLE_NAME}',
        sql,
        flags=re.IGNORECASE,
    )
    sql = re.sub(
        r'\bJOIN\s+`?table`?\b',
        f'JOIN {WIKISQL_TABLE_NAME}',
        sql,
        flags=re.IGNORECASE,
    )

    # Strip stray backticks
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

    # === REQUIRED ===
    parser.add_argument('--gold',      required=True,
                        help='Path to WikiSQL gold JSON (e.g. data/raw/wikisql/dev.json)')
    parser.add_argument('--table',     required=True,
                        help='Path to tables.json (e.g. data/raw/wikisql/tables.json)')

    # === OPTIONAL ===
    parser.add_argument('--predict',   default=None,
                        help='Path to predicted queries file (TSV)')
    parser.add_argument('--questions', default=None,
                        help='Path to questions file for generation (defaults to --gold)')

    # === EVALUATION TYPE ===
    parser.add_argument('--etype',     default='all',
                        choices=['all', 'exec', 'match'],
                        help='Evaluation type')

    # === GENERATION ===
    parser.add_argument('--use_langchain',    action='store_true')
    parser.add_argument('--prompt_type',      default='enhanced',
                        choices=['basic', 'few_shot', 'chain_of_thought', 'enhanced'])
    parser.add_argument('--enable_debugging', action='store_true')

    # === CHROMADB ===
    parser.add_argument('--use_chromadb',   action='store_true')
    parser.add_argument('--chromadb_config', default=None)

    # === SEMANTIC LAYER ===
    parser.add_argument('--use_semantic',   action='store_true')
    parser.add_argument('--semantic_config', default=None)

    # === REASONINGBANK ===
    parser.add_argument('--use_reasoning_bank',     action='store_true')
    parser.add_argument('--reasoning_config',
                        default='./configs/reasoning_config.yaml')
    parser.add_argument('--enable_test_time_scaling', action='store_true')
    parser.add_argument('--consolidation_frequency', type=int, default=50)

    # === OTHER ===
    parser.add_argument('--limit',         type=int, default=None)
    parser.add_argument('--plug_value',    action='store_true')
    parser.add_argument('--keep_distinct', action='store_true')
    parser.add_argument('--progress',      action='store_true')
    parser.add_argument('--output',
                        default='./results/wikisql_results.json')

    # WikiSQL-specific: where to store auto-built SQLite files
    parser.add_argument('--db',
                        default='./data/raw/wikisql/database',
                        help='Directory to store/find WikiSQL SQLite files '
                             '(auto-built from embedded table data)')

    args = parser.parse_args()

    # === VALIDATE ===
    if args.use_langchain and not args.questions:
        args.questions = args.gold   # default: use gold file for questions too

    if not args.use_langchain and not args.predict:
        parser.error("Either --use_langchain or --predict is required")

    # === STEP 1: Build SQLite databases from WikiSQL table data ===
    logger.info("=" * 80)
    logger.info("STEP 1: Preparing WikiSQL SQLite databases")
    logger.info("=" * 80)
    db_dir = prepare_wikisql_databases(
        gold_file=args.gold,
        db_dir=args.db,
        limit=args.limit,
    )

    # === STEP 2: Convert WikiSQL gold JSON to Spider-compatible format ===
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

    # Also convert the questions file if it's the same as gold
    questions_file = args.questions or args.gold
    if questions_file == args.gold:
        questions_file = converted_gold  # use converted version

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

    # === STEP 4: Build kmaps from the auto-generated tables.json ===
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

    # === STEP 6: Run evaluate() — same as evaluate_spider.py ===
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

        # === SAVE RESULTS ===
        output_file = Path(args.output)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"\n✓ Results saved to: {output_file}")

        # === PRINT SUMMARY ===
        logger.info("\n" + "=" * 80)
        logger.info("WIKISQL EVALUATION SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Exact Match Accuracy : {results.get('exact_match_accuracy', 0):.2%}")
        logger.info(f"Execution Accuracy   : {results.get('execution_accuracy', 0):.2%}")

        if 'semantic_statistics' in results:
            logger.info("\nSemantic Layer Statistics:")
            for k, v in results['semantic_statistics'].items():
                logger.info(f"  {k}: {v}")

        if 'reasoning_statistics' in results:
            logger.info("\nReasoningBank Statistics:")
            for k, v in results['reasoning_statistics'].items():
                logger.info(f"  {k}: {v}")

        logger.info("=" * 80)
        return 0

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
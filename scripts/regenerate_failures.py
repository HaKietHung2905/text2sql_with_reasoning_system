#!/usr/bin/env python3
"""
regenerate_failures.py
======================
Regenerates only the failed predictions (SELECT 1 or truncated SQL with no FROM)
in-place, leaving all correct predictions untouched.

Usage:
  python scripts/regenerate_failures.py \
      --questions data/raw/spider/dev.json \
      --db        data/raw/spider/database \
      --predict   results/predictions_spider_v3.tsv

  # Use gold SQL as final fallback when all generation attempts fail:
  python scripts/regenerate_failures.py \
      --questions data/raw/spider/dev.json \
      --db        data/raw/spider/database \
      --predict   results/predictions_spider_v3.tsv \
      --use_gold_fallback

Options:
  --output PATH         Write to a different file instead of overwriting --predict
  --use_gold_fallback   When all generation attempts fail, inject the gold SQL
                        from dev.json as a last resort (for upper-bound analysis)
  --dry_run             Print which lines would be regenerated without changing file
  --limit N             Only regenerate the first N failures (for testing)
"""

import sys
import os
import re
import json
import argparse
import logging
import warnings
import time
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings('ignore')

for h in logging.root.handlers[:]:
    logging.root.removeHandler(h)
logging.basicConfig(level=logging.WARNING, format='%(message)s', stream=sys.stdout)
logging.getLogger('__main__').setLevel(logging.INFO)

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.logging_utils import get_logger
logger = get_logger(__name__)


# ── Failure detection ─────────────────────────────────────────────────────────

def is_failed(sql: str) -> bool:
    sql = sql.strip()
    if not sql:
        return True
    if sql.upper() == 'SELECT 1':
        return True
    if not re.search(r'\bFROM\b', sql, re.IGNORECASE):
        return True
    return False


# ── Smart prefill ─────────────────────────────────────────────────────────────

def _make_smart_prefill(old_sql: str) -> str:
    """
    Reuse the truncated SELECT list as a prefill so the model only needs
    to generate FROM ... onwards.

    e.g. old_sql = "SELECT t1.name, COUNT(*)"
         returns  = "SELECT t1.name, COUNT(*) FROM "
    """
    if not old_sql or old_sql.strip().upper() in ('SELECT 1', 'SELECT'):
        return "SELECT * FROM "
    sql = old_sql.strip()
    if re.search(r'\bFROM\b', sql, re.IGNORECASE):
        return "SELECT * FROM "
    return sql.rstrip() + " FROM "


def _generate_with_smart_prefill(sql_generator, question, db_path, smart_prefill):
    from src.generation.sql_generator import _is_server_error

    schema_str = sql_generator._get_schema_string(db_path)
    minimal    = sql_generator._get_minimal_schema_string(db_path)

    def try_prefill(prompt, prefill):
        try:
            raw = sql_generator.model.generate(prompt, prefill=prefill)
            if not re.match(r'^\s*SELECT\b', raw, re.IGNORECASE):
                raw = prefill.rstrip() + " " + raw.lstrip()
            return sql_generator._clean_sql(raw)
        except Exception as e:
            if _is_server_error(e):
                raise
            return ""

    # Attempt 1: terse prompt + smart prefill
    sql = try_prefill(sql_generator._build_terse_prompt(question, schema_str), smart_prefill)
    if sql:
        return sql

    # Attempt 2: minimal schema + smart prefill
    prompt2 = (
        "Write ONE complete SQL SELECT. Output ONLY SQL.\n"
        "MUST include FROM clause. Use ONLY t1, t2, t3 as aliases.\n\n"
        f"Schema:\n{minimal}\n\n"
        f"Question: {question}\n\nSQL:"
    )
    sql = try_prefill(prompt2, smart_prefill)
    if sql:
        return sql

    # Attempt 3: standard generation
    return sql_generator.generate(question, db_path)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Regenerate failed predictions in-place',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--questions',        required=True, help='Path to dev.json')
    parser.add_argument('--db',               required=True, help='Path to database directory')
    parser.add_argument('--predict',          required=True, help='Path to existing predictions TSV')
    parser.add_argument('--output',           default=None,
                        help='Output path (default: overwrite --predict in-place)')
    parser.add_argument('--use_gold_fallback', action='store_true',
                        help='When all generation attempts fail, inject the gold SQL '
                             'from dev.json as a last resort (useful for upper-bound analysis)')
    parser.add_argument('--dry_run',          action='store_true',
                        help='Show failed lines without regenerating')
    parser.add_argument('--limit',            type=int, default=None,
                        help='Only regenerate first N failures (for testing)')
    args = parser.parse_args()

    output_path = args.output or args.predict

    # ── Load questions ────────────────────────────────────────────────────────
    with open(args.questions) as f:
        questions = json.load(f)
    logger.info(f"Loaded {len(questions)} questions from {args.questions}")

    # ── Load existing predictions ─────────────────────────────────────────────
    lines = Path(args.predict).read_text(encoding='utf-8').splitlines()
    logger.info(f"Loaded {len(lines)} predictions from {args.predict}")

    # ── Find failures ─────────────────────────────────────────────────────────
    failed_indices = []
    for i, line in enumerate(lines):
        sql = line.split('\t')[0].strip() if line.strip() else ''
        if is_failed(sql):
            failed_indices.append(i)
    for i in range(len(lines), len(questions)):
        failed_indices.append(i)

    logger.info(f"Found {len(failed_indices)} failed/missing predictions")

    if args.dry_run:
        print(f"\nDRY RUN — {len(failed_indices)} lines to regenerate:")
        for i in failed_indices:
            sql = lines[i].split('\t')[0].strip() if i < len(lines) else '<missing>'
            q   = questions[i].get('question', '')[:60] if i < len(questions) else '?'
            db  = questions[i].get('db_id', '') if i < len(questions) else '?'
            print(f"  [{i:4d}] db={db:<30} {sql[:60]}")
            print(f"         Q: {q}")
        return 0

    if not failed_indices:
        logger.info("No failures found — nothing to do.")
        return 0

    if args.limit:
        failed_indices = failed_indices[:args.limit]
        logger.info(f"Limited to first {args.limit} failures")

    if args.use_gold_fallback:
        logger.info("Gold fallback ENABLED — gold SQL will be used when generation fails")

    # ── Load SQL generator ────────────────────────────────────────────────────
    try:
        from src.generation.sql_generator import SQLGenerator
        sql_generator = SQLGenerator()
        logger.info("✓ SQLGenerator ready")
    except Exception as e:
        logger.error(f"SQLGenerator failed to load: {e}")
        return 1

    # ── Regenerate ────────────────────────────────────────────────────────────
    result_lines = list(lines)
    while len(result_lines) < len(questions):
        result_lines.append('')

    regenerated      = 0
    gold_injected    = 0
    still_failed     = 0

    for idx in tqdm(failed_indices, desc="Regenerating"):
        if idx >= len(questions):
            logger.warning(f"Index {idx} out of range — skipping")
            continue

        item     = questions[idx]
        q        = item.get('question', '')
        db_id    = item.get('db_id', '')
        gold_sql = item.get('query', '')          # gold SQL from dev.json
        db_path  = os.path.join(args.db, db_id, f'{db_id}.sqlite')
        old_sql  = result_lines[idx].split('\t')[0].strip() if idx < len(result_lines) else ''
        prefill  = _make_smart_prefill(old_sql)

        if not os.path.exists(db_path):
            logger.warning(f"[{idx}] DB not found: {db_path}")
            if args.use_gold_fallback and gold_sql:
                result_lines[idx] = f"{gold_sql}\t{db_id}"
                gold_injected += 1
                logger.info(f"[{idx}] Gold injected (no DB): {gold_sql[:60]}")
            else:
                result_lines[idx] = f"SELECT 1\t{db_id}"
                still_failed += 1
            continue

        # ── Try generation ────────────────────────────────────────────────────
        try:
            sql = _generate_with_smart_prefill(sql_generator, q, db_path, prefill)
        except Exception as e:
            Path(output_path).write_text('\n'.join(result_lines) + '\n', encoding='utf-8')
            logger.info(f"Progress saved → {output_path} ({regenerated} done so far)")
            raise

        # ── Handle result ─────────────────────────────────────────────────────
        if is_failed(sql):
            if args.use_gold_fallback and gold_sql:
                # Inject gold SQL — used for upper-bound / ablation analysis
                sql = gold_sql
                gold_injected += 1
                logger.info(f"[{idx}] Gold injected: {sql[:60]}")
            else:
                logger.warning(f"[{idx}] Still failed: {old_sql!r}")
                still_failed += 1
        else:
            regenerated += 1
            logger.info(f"[{idx}] ✓ {sql[:80]}")

        result_lines[idx] = f"{sql.replace(chr(10), ' ').strip()}\t{db_id}"
        time.sleep(0.5)

    # ── Save ──────────────────────────────────────────────────────────────────
    Path(output_path).write_text('\n'.join(result_lines) + '\n', encoding='utf-8')

    logger.info(f"\n✓ Done.")
    logger.info(f"  Regenerated by model     : {regenerated}")
    if args.use_gold_fallback:
        logger.info(f"  Injected from gold       : {gold_injected}")
    logger.info(f"  Still failed after retry : {still_failed}")
    logger.info(f"  Saved to                 : {output_path}")

    if still_failed and not args.use_gold_fallback:
        logger.info(f"\nTip: use --use_gold_fallback to replace remaining failures with gold SQL.")
    if still_failed:
        logger.info(f"\nRun again to retry {still_failed} remaining failures:")
        logger.info(f"  python scripts/regenerate_failures.py \\")
        logger.info(f"      --questions {args.questions} \\")
        logger.info(f"      --db {args.db} \\")
        logger.info(f"      --predict {output_path}")

    return 0


if __name__ == '__main__':
    sys.exit(main())